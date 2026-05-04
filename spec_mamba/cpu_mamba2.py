"""
CPU-optimized Mamba2 forward pass for speculative decoding drafter.

This module provides an optimized CPU implementation of the Mamba2
single-step forward pass used during draft token generation.
It's designed for the heterogeneous CPU-drafter / GPU-verifier setup.

Key optimizations:
1. C++ kernel with AVX-512 intrinsics for the SSM state update recurrence
2. Fused conv1d + SiLU for the convolution step
3. Memory-contiguous state layout for cache-friendly access
4. Optional multi-threaded execution via OpenMP

Architecture (Mamba2-65M):
    16 layers, hidden_size=512, expand=2, d_inner=1024
    n_heads=16, head_dim=64, state_size=128, conv_kernel=4
    n_groups=1, conv_dim=1280

Single-step cached forward:
    1. in_proj: Linear(512 → 2320)  [d_mlp=0, gate=1024, conv_dim=1280, dt=16]
    2. Conv1d: sliding window (kernel=4) over conv_dim=1280
    3. SiLU activation
    4. Split: x(1024) + B(128) + C(128)
    5. SSM: h_new = h_old * exp(dt*A) + dt*B*x; y = C @ h + D*x
    6. RMSNorm with gate
    7. out_proj: Linear(1024 → 512)
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# Try to import the C++ kernel
try:
    from spec_mamba.cpu_kernels import _cpu_ssm_ops
    HAS_CPP_KERNEL = True
except ImportError:
    HAS_CPP_KERNEL = False


class CPUMamba2Cache:
    """CPU-optimized cache for Mamba2 inference.

    Stores conv_states and ssm_states in contiguous CPU memory
    with layout optimized for sequential access patterns.
    """

    def __init__(self, config, batch_size: int = 1):
        self.n_layers = config.num_hidden_layers
        self.conv_kernel = config.conv_kernel
        self.n_heads = config.num_heads
        self.head_dim = config.head_dim
        self.state_size = config.state_size
        self.intermediate_size = int(config.expand * config.hidden_size)
        self.n_groups = config.n_groups if hasattr(config, 'n_groups') else config.num_heads
        # conv_dim = d_inner + 2 * n_groups * state_size
        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.state_size

        # Conv states: [n_layers, batch, conv_dim, conv_kernel]
        self.conv_states = torch.zeros(
            self.n_layers, batch_size, self.conv_dim, self.conv_kernel,
            dtype=torch.float32,
        )
        # SSM states: [n_layers, batch, n_heads, head_dim, state_size]
        self.ssm_states = torch.zeros(
            self.n_layers, batch_size, self.n_heads, self.head_dim, self.state_size,
            dtype=torch.float32,
        )

    def snapshot(self) -> dict:
        return {
            "conv_states": self.conv_states.clone(),
            "ssm_states": self.ssm_states.clone(),
        }

    def restore(self, snap: dict):
        self.conv_states.copy_(snap["conv_states"])
        self.ssm_states.copy_(snap["ssm_states"])


def fused_conv1d_silu_cached(
    hidden_states_B_C: torch.Tensor,  # [B, 1, conv_dim]
    conv_states: torch.Tensor,         # [B, conv_dim, conv_kernel]
    conv_weight: torch.Tensor,         # [conv_dim, 1, conv_kernel]
    conv_bias: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused conv1d + SiLU for single-step cached inference.

    Updates conv_states in-place (shift left, append new input).
    Returns activated output.
    """
    # Shift conv state left and append new input
    conv_states[:, :, :-1] = conv_states[:, :, 1:].clone()
    conv_states[:, :, -1] = hidden_states_B_C[:, 0, :]

    # Depthwise convolution: sum over kernel dim
    # conv_weight shape: [conv_dim, 1, conv_kernel] → squeeze to [conv_dim, conv_kernel]
    out = (conv_states * conv_weight.squeeze(1)).sum(dim=-1)  # [B, conv_dim]
    if conv_bias is not None:
        out = out + conv_bias
    # SiLU activation
    out = out * torch.sigmoid(out)
    return out, conv_states


def ssm_step_pytorch(
    x: torch.Tensor,       # [B, n_heads, head_dim]
    B_ssm: torch.Tensor,   # [B, n_heads, state_size]
    C_ssm: torch.Tensor,   # [B, n_heads, state_size]
    dt: torch.Tensor,      # [B, n_heads, head_dim]
    A: torch.Tensor,        # [n_heads]
    D: torch.Tensor,        # [n_heads]
    ssm_state: torch.Tensor,  # [B, n_heads, head_dim, state_size]
    dt_bias: torch.Tensor,  # [n_heads]
    time_step_limit: tuple,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure PyTorch SSM single-step update.

    h_new = h_old * exp(dt * A) + dt * B * x
    y = C @ h_new + D * x
    """
    B_size = x.shape[0]

    # dt: [B, n_heads] → [B, n_heads, head_dim]
    dt_expanded = dt.expand(B_size, -1, x.shape[-1])
    dt_bias_expanded = dt_bias[:, None].expand(-1, x.shape[-1])
    dt_act = F.softplus(dt_expanded + dt_bias_expanded)
    dt_act = torch.clamp(dt_act, time_step_limit[0], time_step_limit[1])

    # dA: exp(dt * A), shape [B, n_heads, head_dim, state_size]
    A_expanded = A[:, None, None].expand(-1, x.shape[-1], ssm_state.shape[-1])
    dA = torch.exp(dt_act[..., None] * A_expanded)

    # dB: dt * B, shape [B, n_heads, head_dim, state_size]
    dB = dt_act[..., None] * B_ssm[:, :, None, :]

    # dBx: dB * x, shape [B, n_heads, head_dim, state_size]
    dBx = dB * x[..., None]

    # State update
    new_state = ssm_state * dA + dBx

    # Output: y = C @ h + D * x
    # C: [B, n_heads, state_size], h: [B, n_heads, head_dim, state_size]
    # → y: [B, n_heads, head_dim]
    y = torch.einsum('bhn,bhdn->bhd', C_ssm, new_state)

    # D skip connection
    D_expanded = D[:, None].expand(-1, x.shape[-1])
    y = y + x * D_expanded

    return y, new_state


class CPUMamba2Layer:
    """CPU-optimized single Mamba2 layer for cached inference.

    Pre-extracts all weights and stores them in CPU-friendly format.
    Avoids Python overhead from HuggingFace module dispatch.
    """

    def __init__(self, hf_block, layer_idx: int):
        self.layer_idx = layer_idx
        mixer = hf_block.mixer if hasattr(hf_block, 'mixer') else hf_block

        # Extract and store weights as contiguous CPU float32 tensors
        self.norm_weight = hf_block.norm.weight.float().contiguous().cpu()

        self.in_proj_weight = mixer.in_proj.weight.float().contiguous().cpu()
        self.in_proj_bias = (
            mixer.in_proj.bias.float().contiguous().cpu()
            if mixer.in_proj.bias is not None else None
        )

        self.conv1d_weight = mixer.conv1d.weight.float().contiguous().cpu()
        self.conv1d_bias = (
            mixer.conv1d.bias.float().contiguous().cpu()
            if mixer.use_conv_bias else None
        )

        self.A_log = mixer.A_log.float().contiguous().cpu()
        self.D = mixer.D.float().contiguous().cpu()
        self.dt_bias = mixer.dt_bias.float().contiguous().cpu()
        self.time_step_limit = mixer.time_step_limit

        self.norm_weight_ssm = mixer.norm.weight.float().contiguous().cpu()

        self.out_proj_weight = mixer.out_proj.weight.float().contiguous().cpu()
        self.out_proj_bias = (
            mixer.out_proj.bias.float().contiguous().cpu()
            if mixer.out_proj.bias is not None else None
        )

        # Architecture dims
        self.intermediate_size = mixer.intermediate_size
        self.n_heads = mixer.num_heads
        self.head_dim = mixer.head_dim
        self.n_groups = mixer.n_groups
        self.state_size = mixer.ssm_state_size
        self.conv_dim = mixer.conv_dim

        # Compute projection layout offsets
        proj_size = self.in_proj_weight.shape[0]
        self.d_mlp = (proj_size - 2 * self.intermediate_size
                      - 2 * self.n_groups * self.state_size - self.n_heads) // 2
        self.gate_start = 2 * self.d_mlp
        self.gate_end = self.gate_start + self.intermediate_size

        self.residual_in_fp32 = hf_block.residual_in_fp32 if hasattr(hf_block, 'residual_in_fp32') else True

    def forward_cached_step(
        self,
        hidden_states: torch.Tensor,   # [B, 1, hidden_size]
        cache: CPUMamba2Cache,
        guidance_delta: Optional[torch.Tensor] = None,  # [B, 1, delta_dim] or None
    ) -> torch.Tensor:
        """Single-step cached forward for one layer."""
        B = hidden_states.shape[0]

        # Residual
        residual = hidden_states
        # RMSNorm
        h_normed = _rms_norm(hidden_states, self.norm_weight)

        if self.residual_in_fp32:
            residual = residual.float()

        # in_proj
        projected = F.linear(h_normed.squeeze(1), self.in_proj_weight, self.in_proj_bias)
        # Split: [d_mlp, d_mlp, gate, hidden_states_B_C, dt]
        _, _, gate, hidden_states_B_C, dt = projected.split(
            [self.d_mlp, self.d_mlp, self.intermediate_size,
             self.conv_dim, self.n_heads],
            dim=-1,
        )

        # Inject guidance delta into x-branch if provided
        if guidance_delta is not None:
            delta = guidance_delta
            if delta.dim() == 3:
                delta = delta.squeeze(1)  # [B, delta_dim]
            # x-branch starts at gate_end within hidden_states_B_C space
            # but hidden_states_B_C is already split out, so x is at the start
            delta_x = delta[..., :self.intermediate_size]
            hidden_states_B_C = hidden_states_B_C.clone()
            hidden_states_B_C[..., :self.intermediate_size] += delta_x

        # Conv1d (cached single step)
        hidden_states_B_C_unsqueezed = hidden_states_B_C.unsqueeze(1)  # [B, 1, conv_dim]
        conv_out, cache.conv_states[self.layer_idx] = fused_conv1d_silu_cached(
            hidden_states_B_C_unsqueezed,
            cache.conv_states[self.layer_idx],
            self.conv1d_weight,
            self.conv1d_bias,
        )

        # Split: x, B_ssm, C_ssm
        x, B_ssm, C_ssm = conv_out.split(
            [self.intermediate_size,
             self.n_groups * self.state_size,
             self.n_groups * self.state_size],
            dim=-1,
        )

        # Reshape for SSM
        x = x.reshape(B, self.n_heads, self.head_dim)
        B_ssm = B_ssm.reshape(B, self.n_groups, -1)
        B_ssm = B_ssm[:, :, None, :].expand(
            B, self.n_groups, self.n_heads // self.n_groups, self.state_size
        ).contiguous().reshape(B, self.n_heads, self.state_size)
        C_ssm = C_ssm.reshape(B, self.n_groups, -1)
        C_ssm = C_ssm[:, :, None, :].expand(
            B, self.n_groups, self.n_heads // self.n_groups, self.state_size
        ).contiguous().reshape(B, self.n_heads, self.state_size)

        A = -torch.exp(self.A_log.float())

        dt_expanded = dt[:, None, :].transpose(1, 2).expand(B, self.n_heads, self.head_dim)

        # SSM step
        if HAS_CPP_KERNEL:
            y, new_state = _cpu_ssm_ops.ssm_step(
                x, B_ssm, C_ssm, dt_expanded, A, self.D,
                cache.ssm_states[self.layer_idx],
                self.dt_bias, self.time_step_limit,
            )
        else:
            y, new_state = ssm_step_pytorch(
                x, B_ssm, C_ssm, dt_expanded, A, self.D,
                cache.ssm_states[self.layer_idx],
                self.dt_bias, self.time_step_limit,
            )
        cache.ssm_states[self.layer_idx] = new_state

        # [B, n_heads, head_dim] → [B, 1, intermediate_size]
        y = y.reshape(B, -1).unsqueeze(1)

        # RMSNorm with gate
        y = _rms_norm_with_gate(y, gate.unsqueeze(1), self.norm_weight_ssm)

        # out_proj
        y = F.linear(y.squeeze(1), self.out_proj_weight, self.out_proj_bias).unsqueeze(1)

        # Residual
        return residual + y


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """RMSNorm: x * weight / sqrt(mean(x^2) + eps)"""
    dtype = x.dtype
    x = x.float()
    norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (x * norm * weight.float()).to(dtype)


def _rms_norm_with_gate(x: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """RMSNorm with SiLU gate, matching HF MambaRMSNormGated: norm(x * silu(gate)) * weight"""
    dtype = x.dtype
    x = x.float() * F.silu(gate.float())
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return (weight * x).to(dtype)


class CPUMamba2Model:
    """Complete CPU-optimized Mamba2 model for draft token generation.

    Pre-extracts all weights from the HuggingFace model and stores
    them in a CPU-friendly layout. Eliminates Python dispatch overhead
    from the HF forward path.
    """

    def __init__(self, hf_model):
        """Initialize from a HuggingFace Mamba2ForCausalLM model."""
        backbone = hf_model.backbone
        self.config = backbone.config

        # Embedding
        self.embed_weight = backbone.embeddings.weight.float().contiguous().cpu()

        # LM head
        self.lm_head_weight = hf_model.lm_head.weight.float().contiguous().cpu()
        self.lm_head_bias = (
            hf_model.lm_head.bias.float().contiguous().cpu()
            if hf_model.lm_head.bias is not None else None
        )

        # Final normalization
        self.norm_f_weight = backbone.norm_f.weight.float().contiguous().cpu()

        # Layers
        self.layers = []
        for i, block in enumerate(backbone.layers):
            self.layers.append(CPUMamba2Layer(block, layer_idx=i))

        self.n_layers = len(self.layers)
        print(f"[CPUMamba2Model] Initialized {self.n_layers} layers on CPU")

    def create_cache(self, batch_size: int = 1) -> CPUMamba2Cache:
        return CPUMamba2Cache(self.config, batch_size)

    def forward_step(
        self,
        token_ids: torch.Tensor,       # [B, 1] on CPU
        cache: CPUMamba2Cache,
        guidance_deltas: Optional[torch.Tensor] = None,  # [n_layers, B, 1, delta_dim] or None
    ) -> torch.Tensor:
        """Single-step forward pass returning logits [B, vocab_size]."""
        # Embedding
        hidden_states = F.embedding(token_ids, self.embed_weight)  # [B, 1, hidden_size]

        # Run through all layers
        for i, layer in enumerate(self.layers):
            delta = None
            if guidance_deltas is not None:
                delta = guidance_deltas[i]  # [B, 1, delta_dim]
            hidden_states = layer.forward_cached_step(hidden_states, cache, delta)

        # Final norm
        hidden_states = _rms_norm(hidden_states, self.norm_f_weight)

        # LM head
        logits = F.linear(hidden_states.squeeze(1), self.lm_head_weight, self.lm_head_bias)
        return logits

    def prefill(
        self,
        token_ids: torch.Tensor,  # [B, S] on CPU
        cache: CPUMamba2Cache,
        guidance_deltas: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Process a sequence of tokens one at a time through cached forward."""
        B, S = token_ids.shape
        for t in range(S):
            delta = None
            if guidance_deltas is not None:
                delta = guidance_deltas[:, :, t:t+1, :]
            logits = self.forward_step(
                token_ids[:, t:t+1], cache, delta,
            )
        return logits


def estimate_memory_mb(config) -> float:
    """Estimate memory footprint of the CPU model in MB."""
    n_layers = config.num_hidden_layers
    hidden = config.hidden_size
    d_inner = int(config.expand * hidden)
    n_heads = config.num_heads
    state_size = config.state_size
    n_groups = config.n_groups if hasattr(config, 'n_groups') else n_heads
    conv_dim = d_inner + 2 * n_groups * state_size

    # Per-layer weights (float32 = 4 bytes)
    per_layer = (
        hidden  # norm
        + hidden * (d_inner + conv_dim + n_heads + d_inner)  # in_proj (no d_mlp for 65M)
        + conv_dim * 4 + conv_dim  # conv1d + bias
        + n_heads  # A_log
        + n_heads  # D
        + n_heads  # dt_bias
        + d_inner  # ssm norm
        + d_inner * hidden  # out_proj
    )
    total_params = n_layers * per_layer + hidden * config.vocab_size * 2  # embed + lm_head
    return total_params * 4 / (1024 * 1024)


class FusedCPUMamba2Model:
    """Fully fused C++ Mamba2 single-step forward with BF16 weights.

    Runs the entire 16-layer forward + LM head in a single C++ call,
    eliminating Python/PyTorch dispatch overhead. Uses BF16 weights
    for memory-bandwidth-bound operations (embed, in_proj, out_proj, LM head)
    while keeping SSM state and norms in FP32.

    For B=1 only. Falls back to CPUMamba2Model for batched inference.
    """

    def __init__(self, hf_model):
        """Pack all weights into flat tensors for C++ consumption."""
        backbone = hf_model.backbone
        self.config = backbone.config
        n_layers = backbone.config.num_hidden_layers
        hidden = backbone.config.hidden_size
        mixer0 = backbone.layers[0].mixer

        # Architecture dims
        self.hidden_size = hidden
        self.d_inner = mixer0.intermediate_size
        self.conv_dim = mixer0.conv_dim
        self.n_heads = mixer0.num_heads
        self.head_dim = mixer0.head_dim
        self.state_size = mixer0.ssm_state_size
        self.n_groups = mixer0.n_groups
        self.conv_kernel = backbone.config.conv_kernel
        self.n_layers = n_layers
        self.time_step_limit = mixer0.time_step_limit

        # Projection layout
        proj_size = mixer0.in_proj.weight.shape[0]
        self.d_mlp = (proj_size - 2 * self.d_inner
                      - 2 * self.n_groups * self.state_size - self.n_heads) // 2
        self.proj_size = proj_size

        # Shared embed/LM-head weight → BF16
        self.embed_weight_bf16 = backbone.embeddings.weight.to(torch.bfloat16).contiguous().cpu()

        # Final norm weight (FP32)
        self.norm_f_weight = backbone.norm_f.weight.float().contiguous().cpu()

        # Pack per-layer weights into [n_layers, ...] tensors
        self.layer_norm_weights = torch.stack(
            [b.norm.weight.float().contiguous() for b in backbone.layers]).cpu()
        self.layer_in_proj_bf16 = torch.stack(
            [b.mixer.in_proj.weight.to(torch.bfloat16).contiguous() for b in backbone.layers]).cpu()
        self.layer_in_proj_bias = torch.stack(
            [b.mixer.in_proj.bias.float().contiguous() if b.mixer.in_proj.bias is not None
             else torch.zeros(proj_size) for b in backbone.layers]).cpu()
        self.layer_conv_weight = torch.stack(
            [b.mixer.conv1d.weight.squeeze(1).float().t().contiguous() for b in backbone.layers]).cpu()
            # ^ Transposed: HF gives [conv_dim, K], we store [K, conv_dim] for vectorized access
        self.layer_conv_bias = torch.stack(
            [b.mixer.conv1d.bias.float().contiguous() if b.mixer.use_conv_bias
             else torch.zeros(self.conv_dim) for b in backbone.layers]).cpu()
        self.layer_A_log = torch.stack(
            [b.mixer.A_log.float().contiguous() for b in backbone.layers]).cpu()
        self.layer_D = torch.stack(
            [b.mixer.D.float().contiguous() for b in backbone.layers]).cpu()
        self.layer_dt_bias = torch.stack(
            [b.mixer.dt_bias.float().contiguous() for b in backbone.layers]).cpu()
        self.layer_ssm_norm_weight = torch.stack(
            [b.mixer.norm.weight.float().contiguous() for b in backbone.layers]).cpu()
        self.layer_out_proj_bf16 = torch.stack(
            [b.mixer.out_proj.weight.to(torch.bfloat16).contiguous() for b in backbone.layers]).cpu()
        self.layer_out_proj_bias = torch.stack(
            [b.mixer.out_proj.bias.float().contiguous() if b.mixer.out_proj.bias is not None
             else torch.zeros(hidden) for b in backbone.layers]).cpu()

        # Load the fused C++ module
        from spec_mamba.cpu_kernels import get_fused_forward
        self._fused_mod = get_fused_forward()

        total_bf16_mb = (self.embed_weight_bf16.numel() + self.layer_in_proj_bf16.numel()
                         + self.layer_out_proj_bf16.numel()) * 2 / 1e6
        total_fp32_mb = sum(t.numel() * 4 for t in [
            self.norm_f_weight, self.layer_norm_weights, self.layer_in_proj_bias,
            self.layer_conv_weight, self.layer_conv_bias, self.layer_A_log,
            self.layer_D, self.layer_dt_bias, self.layer_ssm_norm_weight,
            self.layer_out_proj_bias]) / 1e6
        print(f"[FusedCPUMamba2Model] Initialized: {n_layers} layers, "
              f"BF16 weights={total_bf16_mb:.1f}MB, FP32 weights={total_fp32_mb:.1f}MB")

    def create_cache(self, batch_size: int = 1):
        """Create cache tensors for fused forward. B=1 only."""
        assert batch_size == 1, "FusedCPUMamba2Model only supports B=1"
        conv_states = torch.zeros(
            self.n_layers, self.conv_kernel, self.conv_dim, dtype=torch.float32)
            # ^ Transposed layout: [n_layers, K, conv_dim] for vectorized conv step
        ssm_states = torch.zeros(
            self.n_layers, self.n_heads, self.head_dim, self.state_size, dtype=torch.float32)
        return conv_states, ssm_states

    def forward_step(self, token_id: int, conv_states: torch.Tensor, ssm_states: torch.Tensor,
                      guidance_deltas: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Single-step forward returning logits [1, vocab_size].

        Args:
            guidance_deltas: Optional FP32 [n_layers, d_inner] tensor.
                Injected into x-branch after in_proj. Pass empty tensor or None to skip.
        """
        if guidance_deltas is None:
            guidance_deltas = torch.empty(0, dtype=torch.float32)
        return self._fused_mod.fused_mamba2_step(
            token_id,
            self.embed_weight_bf16,
            self.norm_f_weight,
            self.layer_norm_weights,
            self.layer_in_proj_bf16,
            self.layer_in_proj_bias,
            self.layer_conv_weight,
            self.layer_conv_bias,
            self.layer_A_log,
            self.layer_D,
            self.layer_dt_bias,
            self.layer_ssm_norm_weight,
            self.layer_out_proj_bf16,
            self.layer_out_proj_bias,
            conv_states,
            ssm_states,
            self.n_layers,
            self.hidden_size,
            self.d_inner,
            self.conv_dim,
            self.n_heads,
            self.head_dim,
            self.state_size,
            self.n_groups,
            self.conv_kernel,
            self.time_step_limit[0],
            self.time_step_limit[1],
            self.d_mlp,
            self.proj_size,
            guidance_deltas,
        )

    def forward_step_tensor(self, token_ids: torch.Tensor, conv_states: torch.Tensor, ssm_states: torch.Tensor,
                            guidance_deltas: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Wrapper accepting [B=1, 1] tensor input for API compatibility."""
        token_id = token_ids.item()
        return self.forward_step(token_id, conv_states, ssm_states, guidance_deltas)

    def prefill(self, token_ids: torch.Tensor, conv_states: torch.Tensor, ssm_states: torch.Tensor) -> torch.Tensor:
        """Process prompt tokens one at a time (no guidance during prefill)."""
        S = token_ids.shape[1]
        logits = None
        for t in range(S):
            logits = self.forward_step(token_ids[0, t].item(), conv_states, ssm_states)
        return logits


class Int8FusedCPUMamba2Model:
    """INT8 weight-only quantized Mamba2 forward — ~2x less memory bandwidth.

    Same fused C++ forward as FusedCPUMamba2Model but with INT8 weights
    for embed/LM-head, in_proj, and out_proj. Uses AVX-512 VNNI for
    native INT8 dot products. Conv weights and SSM state remain FP32.

    For B=1 only. No calibration needed — uses simple per-channel absmax.
    """

    @staticmethod
    def _quantize_per_row(w_fp32):
        """Symmetric per-row INT8 quantization.
        Returns: (w_int8 [M,K], scale [M], row_sum [M])
        """
        # Per-row absmax
        absmax = w_fp32.abs().max(dim=1).values.clamp(min=1e-8)
        scale = absmax / 127.0  # [M]
        # Quantize
        w_int8 = (w_fp32 / scale.unsqueeze(1)).round().clamp(-128, 127).to(torch.int8)
        # Row sum (for uint8 bias correction in VNNI)
        row_sum = w_int8.to(torch.int32).sum(dim=1).to(torch.int32)
        return w_int8, scale, row_sum

    def __init__(self, hf_model):
        """Pack all weights into INT8 tensors for C++ consumption."""
        backbone = hf_model.backbone
        self.config = backbone.config
        n_layers = backbone.config.num_hidden_layers
        hidden = backbone.config.hidden_size
        mixer0 = backbone.layers[0].mixer

        # Architecture dims
        self.hidden_size = hidden
        self.d_inner = mixer0.intermediate_size
        self.conv_dim = mixer0.conv_dim
        self.n_heads = mixer0.num_heads
        self.head_dim = mixer0.head_dim
        self.state_size = mixer0.ssm_state_size
        self.n_groups = mixer0.n_groups
        self.conv_kernel = backbone.config.conv_kernel
        self.n_layers = n_layers
        self.time_step_limit = mixer0.time_step_limit

        proj_size = mixer0.in_proj.weight.shape[0]
        self.d_mlp = (proj_size - 2 * self.d_inner
                      - 2 * self.n_groups * self.state_size - self.n_heads) // 2
        self.proj_size = proj_size

        # Quantize embed/LM-head (tied) → INT8
        embed_fp32 = backbone.embeddings.weight.float()
        self.embed_weight_int8, self.embed_scale, self.embed_row_sum = \
            self._quantize_per_row(embed_fp32)
        self.embed_weight_int8 = self.embed_weight_int8.contiguous().cpu()
        self.embed_scale = self.embed_scale.contiguous().cpu()
        self.embed_row_sum = self.embed_row_sum.contiguous().cpu()

        # Final norm (FP32)
        self.norm_f_weight = backbone.norm_f.weight.float().contiguous().cpu()

        # Per-layer norms (FP32)
        self.layer_norm_weights = torch.stack(
            [b.norm.weight.float().contiguous() for b in backbone.layers]).cpu()

        # Quantize in_proj → INT8
        in_proj_int8_list = []
        in_proj_scale_list = []
        in_proj_rsum_list = []
        for b in backbone.layers:
            w_i8, s, rs = self._quantize_per_row(b.mixer.in_proj.weight.float())
            in_proj_int8_list.append(w_i8.contiguous())
            in_proj_scale_list.append(s.contiguous())
            in_proj_rsum_list.append(rs.contiguous())
        self.layer_in_proj_int8 = torch.stack(in_proj_int8_list).cpu()
        self.layer_in_proj_scale = torch.stack(in_proj_scale_list).cpu()
        self.layer_in_proj_row_sum = torch.stack(in_proj_rsum_list).cpu()

        # in_proj bias (FP32)
        self.layer_in_proj_bias = torch.stack(
            [b.mixer.in_proj.bias.float().contiguous() if b.mixer.in_proj.bias is not None
             else torch.zeros(proj_size) for b in backbone.layers]).cpu()

        # Conv weights (FP32, transposed [K, conv_dim])
        self.layer_conv_weight = torch.stack(
            [b.mixer.conv1d.weight.squeeze(1).float().t().contiguous()
             for b in backbone.layers]).cpu()
        self.layer_conv_bias = torch.stack(
            [b.mixer.conv1d.bias.float().contiguous() if b.mixer.use_conv_bias
             else torch.zeros(self.conv_dim) for b in backbone.layers]).cpu()

        # SSM params (FP32)
        self.layer_A_log = torch.stack(
            [b.mixer.A_log.float().contiguous() for b in backbone.layers]).cpu()
        self.layer_D = torch.stack(
            [b.mixer.D.float().contiguous() for b in backbone.layers]).cpu()
        self.layer_dt_bias = torch.stack(
            [b.mixer.dt_bias.float().contiguous() for b in backbone.layers]).cpu()
        self.layer_ssm_norm_weight = torch.stack(
            [b.mixer.norm.weight.float().contiguous() for b in backbone.layers]).cpu()

        # Quantize out_proj → INT8
        out_proj_int8_list = []
        out_proj_scale_list = []
        out_proj_rsum_list = []
        for b in backbone.layers:
            w_i8, s, rs = self._quantize_per_row(b.mixer.out_proj.weight.float())
            out_proj_int8_list.append(w_i8.contiguous())
            out_proj_scale_list.append(s.contiguous())
            out_proj_rsum_list.append(rs.contiguous())
        self.layer_out_proj_int8 = torch.stack(out_proj_int8_list).cpu()
        self.layer_out_proj_scale = torch.stack(out_proj_scale_list).cpu()
        self.layer_out_proj_row_sum = torch.stack(out_proj_rsum_list).cpu()

        # out_proj bias (FP32)
        self.layer_out_proj_bias = torch.stack(
            [b.mixer.out_proj.bias.float().contiguous() if b.mixer.out_proj.bias is not None
             else torch.zeros(hidden) for b in backbone.layers]).cpu()

        # Load C++ module
        from spec_mamba.cpu_kernels import get_fused_forward
        self._fused_mod = get_fused_forward()

        # Memory summary
        int8_mb = (self.embed_weight_int8.numel() + self.layer_in_proj_int8.numel()
                   + self.layer_out_proj_int8.numel()) / 1024**2
        fp32_mb = (self.embed_scale.numel() + self.embed_row_sum.numel()
                   + self.layer_in_proj_scale.numel() + self.layer_in_proj_row_sum.numel()
                   + self.layer_out_proj_scale.numel() + self.layer_out_proj_row_sum.numel()
                   + self.layer_norm_weights.numel() + self.norm_f_weight.numel()
                   + self.layer_in_proj_bias.numel() + self.layer_out_proj_bias.numel()
                   + self.layer_conv_weight.numel() + self.layer_conv_bias.numel()
                   + self.layer_A_log.numel() + self.layer_D.numel()
                   + self.layer_dt_bias.numel() + self.layer_ssm_norm_weight.numel()
                   ) * 4 / 1024**2
        print(f"[Int8FusedCPUMamba2Model] Initialized: {n_layers} layers, "
              f"INT8 weights={int8_mb:.1f}MB, FP32 weights={fp32_mb:.1f}MB")

    def create_cache(self, batch_size: int = 1):
        """Create cache tensors for fused forward. B=1 only."""
        assert batch_size == 1
        conv_states = torch.zeros(
            self.n_layers, self.conv_kernel, self.conv_dim, dtype=torch.float32)
        ssm_states = torch.zeros(
            self.n_layers, self.n_heads, self.head_dim, self.state_size, dtype=torch.float32)
        return conv_states, ssm_states

    def forward_step(self, token_id: int, conv_states: torch.Tensor, ssm_states: torch.Tensor,
                      guidance_deltas: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Single-step forward returning logits [1, vocab_size].

        Args:
            guidance_deltas: Optional FP32 [n_layers, d_inner] tensor.
                Injected into x-branch after in_proj. Pass empty tensor or None to skip.
        """
        if guidance_deltas is None:
            guidance_deltas = torch.empty(0, dtype=torch.float32)
        return self._fused_mod.fused_mamba2_step_int8(
            token_id,
            self.embed_weight_int8,
            self.embed_scale,
            self.embed_row_sum,
            self.norm_f_weight,
            self.layer_norm_weights,
            self.layer_in_proj_int8,
            self.layer_in_proj_scale,
            self.layer_in_proj_row_sum,
            self.layer_in_proj_bias,
            self.layer_conv_weight,
            self.layer_conv_bias,
            self.layer_A_log,
            self.layer_D,
            self.layer_dt_bias,
            self.layer_ssm_norm_weight,
            self.layer_out_proj_int8,
            self.layer_out_proj_scale,
            self.layer_out_proj_row_sum,
            self.layer_out_proj_bias,
            conv_states,
            ssm_states,
            self.n_layers,
            self.hidden_size,
            self.d_inner,
            self.conv_dim,
            self.n_heads,
            self.head_dim,
            self.state_size,
            self.n_groups,
            self.conv_kernel,
            self.time_step_limit[0],
            self.time_step_limit[1],
            self.d_mlp,
            self.proj_size,
            guidance_deltas,
        )

    def forward_step_tensor(self, token_ids: torch.Tensor, conv_states: torch.Tensor, ssm_states: torch.Tensor,
                            guidance_deltas: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Wrapper accepting [B=1, 1] tensor input for API compatibility."""
        return self.forward_step(token_ids.item(), conv_states, ssm_states, guidance_deltas)

    def prefill(self, token_ids: torch.Tensor, conv_states: torch.Tensor, ssm_states: torch.Tensor) -> torch.Tensor:
        """Process prompt tokens one at a time (no guidance during prefill)."""
        S = token_ids.shape[1]
        logits = None
        for t in range(S):
            logits = self.forward_step(token_ids[0, t].item(), conv_states, ssm_states)
        return logits
