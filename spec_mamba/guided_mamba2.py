"""
Guided Mamba2 block and mixer.

Wraps the stock HuggingFace Mamba2Block/Mamba2Mixer so that an external
"guidance delta" (produced from verifier hidden states) is added to the
x-branch and optionally the z-branch after the in_proj split.

The in_proj output layout for Mamba2 (with d_mlp=0 for our 65M config):
    [d_mlp, d_mlp, gate(=d_inner), hidden_states_B_C(=conv_dim), dt(=n_heads)]

Within hidden_states_B_C:
    [x(=d_inner), B(=n_groups*state_size), C(=n_groups*state_size)]

Injection strategy:
    After in_proj but before convolution/SSM:
    - x-branch (state update path):  x += delta_x   (always)
    - z-branch (gating path):        gate += delta_z (when steer_z=True)

This approach works with both the fused CUDA kernels and the unfused
torch path by modifying projected_states in-place before they reach
the scan/conv kernels.
"""

import torch
import torch.nn as nn
from typing import Optional

from transformers import Cache
from transformers.models.mamba2.modeling_mamba2 import (
    Mamba2Block,
    Mamba2Mixer,
)


def _compute_proj_offsets(mixer: Mamba2Mixer):
    """Pre-compute the byte offsets into projected_states for each branch."""
    d_inner = mixer.intermediate_size
    conv_dim = mixer.conv_dim
    n_heads = mixer.num_heads
    n_groups = mixer.n_groups
    ssm_state_size = mixer.ssm_state_size
    proj_size = d_inner + conv_dim + n_heads  # full in_proj output dim
    d_mlp = (proj_size - 2 * d_inner - 2 * n_groups * ssm_state_size - n_heads) // 2

    # Offsets into the last dimension of projected_states
    gate_start = 2 * d_mlp
    gate_end = gate_start + d_inner
    x_start = gate_end  # x is the first d_inner of hidden_states_B_C
    x_end = x_start + d_inner
    return d_mlp, gate_start, gate_end, x_start, x_end


class GuidedMamba2Mixer(nn.Module):
    """Drop-in replacement for Mamba2Mixer that accepts guidance deltas.

    Rather than reimplementing the full forward pass, this module:
    1. Runs in_proj to get projected_states
    2. Adds deltas to the x-branch (and optionally z/gate branch)
    3. Delegates to the original mixer's cuda_kernels_forward or torch_forward
       via a patched projected_states
    """

    def __init__(self, orig_mixer: Mamba2Mixer, layer_idx: int, steer_z: bool = False):
        super().__init__()
        self.m = orig_mixer
        self.layer_idx = layer_idx
        self.steer_z = steer_z

        # Pre-compute projection layout offsets
        d_mlp, gate_start, gate_end, x_start, x_end = _compute_proj_offsets(orig_mixer)
        self.d_mlp = d_mlp
        self.gate_start = gate_start
        self.gate_end = gate_end
        self.x_start = x_start
        self.x_end = x_end
        self.d_inner = orig_mixer.intermediate_size

    # Delegate attribute lookups to the wrapped mixer transparently
    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.m, name)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        guidance_deltas: Optional[torch.Tensor] = None,
    ):
        """
        guidance_deltas: (n_guided_layers, B, S, delta_dim) or None.
            delta_dim = d_inner when steer_z is False, or 2*d_inner when steer_z is True.
        """
        if guidance_deltas is None:
            return self.m(hidden_states, cache_params=cache_params,
                          cache_position=cache_position, attention_mask=attention_mask)

        # Hook: intercept in_proj, inject deltas, then run the rest normally.
        # We temporarily replace in_proj with our wrapper that adds deltas.
        delta = guidance_deltas[self.layer_idx]  # (B, S, delta_dim), may need squeeze for single-step

        orig_in_proj = self.m.in_proj

        mixer = self.m
        guide_mixer = self

        class _InProjWithDelta(nn.Module):
            """Temporary wrapper: runs original in_proj then adds deltas."""
            def __init__(self):
                super().__init__()
                self.weight = orig_in_proj.weight
                self.bias = orig_in_proj.bias

            def forward(self, x):
                proj = orig_in_proj(x)
                # Handle both sequence (B,S,D) and single-step (B,D) shapes
                d = delta
                if proj.dim() == 2 and d.dim() == 3:
                    d = d.squeeze(1)

                # Always add delta_x to x-branch
                if guide_mixer.steer_z:
                    delta_x = d[..., :guide_mixer.d_inner]
                    delta_z = d[..., guide_mixer.d_inner:]
                else:
                    delta_x = d
                    delta_z = None

                # Clone to avoid in-place on leaf variable
                proj = proj.clone()
                proj[..., guide_mixer.x_start:guide_mixer.x_end] = (
                    proj[..., guide_mixer.x_start:guide_mixer.x_end] + delta_x
                )
                if delta_z is not None:
                    proj[..., guide_mixer.gate_start:guide_mixer.gate_end] = (
                        proj[..., guide_mixer.gate_start:guide_mixer.gate_end] + delta_z
                    )
                return proj

        # Swap in the delta-injecting in_proj, run the original forward, swap back
        wrapper = _InProjWithDelta()
        self.m.in_proj = wrapper
        try:
            out = self.m(hidden_states, cache_params=cache_params,
                         cache_position=cache_position, attention_mask=attention_mask)
        finally:
            self.m.in_proj = orig_in_proj

        return out


class GuidedMamba2Block(nn.Module):
    """Drop-in replacement for Mamba2Block that passes guidance_deltas to the mixer."""

    def __init__(self, orig_block: Mamba2Block, layer_idx: int, steer_z: bool = False):
        super().__init__()
        self.norm = orig_block.norm
        self.residual_in_fp32 = orig_block.residual_in_fp32
        self.mixer = GuidedMamba2Mixer(orig_block.mixer, layer_idx=layer_idx, steer_z=steer_z)

    def forward(
        self,
        hidden_states,
        cache_params: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        guidance_deltas: Optional[torch.Tensor] = None,
    ):
        residual = hidden_states
        hidden_states = self.norm(hidden_states.to(dtype=self.norm.weight.dtype))
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)

        hidden_states = self.mixer(
            hidden_states,
            cache_params=cache_params,
            cache_position=cache_position,
            attention_mask=attention_mask,
            guidance_deltas=guidance_deltas,
        )
        hidden_states = residual + hidden_states
        return hidden_states
