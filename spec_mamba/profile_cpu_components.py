"""Profile per-component breakdown of Mamba2 CPU single-step inference.

Measures absolute time for each component of the forward pass:
  in_proj, guidance_inject, conv1d+silu, ssm_step, norm+gate, out_proj, residual+norm

Sweeps over K=1..8 draft tokens to show scaling.
Outputs JSON for figure generation.
"""

import os
import sys
import time
import json
import torch
import torch.nn.functional as F
from pathlib import Path

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spec_mamba.cpu_mamba2 import (
    CPUMamba2Layer, CPUMamba2Cache, CPUMamba2Model,
    _rms_norm, _rms_norm_with_gate, fused_conv1d_silu_cached,
)

try:
    from spec_mamba.cpu_kernels import _cpu_ssm_ops
    HAS_CPP_KERNEL = True
except ImportError:
    HAS_CPP_KERNEL = False


def profile_single_layer(layer: CPUMamba2Layer, cache: CPUMamba2Cache,
                         hidden_states: torch.Tensor,
                         guidance_delta: torch.Tensor,
                         warmup: int = 10, iters: int = 50):
    """Profile one layer's forward_cached_step, broken into components."""
    B = hidden_states.shape[0]

    # Warmup
    for _ in range(warmup):
        # Save/restore cache to avoid state drift
        conv_snap = cache.conv_states[layer.layer_idx].clone()
        ssm_snap = cache.ssm_states[layer.layer_idx].clone()
        layer.forward_cached_step(hidden_states, cache, guidance_delta)
        cache.conv_states[layer.layer_idx] = conv_snap
        cache.ssm_states[layer.layer_idx] = ssm_snap

    timings = {k: [] for k in [
        'rms_norm_input', 'in_proj', 'guidance_inject',
        'conv1d_silu', 'ssm_step', 'norm_gate', 'out_proj', 'residual'
    ]}

    for _ in range(iters):
        conv_snap = cache.conv_states[layer.layer_idx].clone()
        ssm_snap = cache.ssm_states[layer.layer_idx].clone()

        residual = hidden_states

        # 1. RMSNorm
        t0 = time.perf_counter()
        h_normed = _rms_norm(hidden_states, layer.norm_weight)
        t1 = time.perf_counter()
        timings['rms_norm_input'].append(t1 - t0)

        # 2. in_proj
        t0 = time.perf_counter()
        projected = F.linear(h_normed.squeeze(1), layer.in_proj_weight, layer.in_proj_bias)
        _, _, gate, hidden_states_B_C, dt = projected.split(
            [layer.d_mlp, layer.d_mlp, layer.intermediate_size,
             layer.conv_dim, layer.n_heads], dim=-1)
        t1 = time.perf_counter()
        timings['in_proj'].append(t1 - t0)

        # 3. Guidance inject
        t0 = time.perf_counter()
        delta = guidance_delta.squeeze(1) if guidance_delta.dim() == 3 else guidance_delta
        delta_x = delta[..., :layer.intermediate_size]
        hidden_states_B_C = hidden_states_B_C.clone()
        hidden_states_B_C[..., :layer.intermediate_size] += delta_x
        t1 = time.perf_counter()
        timings['guidance_inject'].append(t1 - t0)

        # 4. Conv1d + SiLU
        t0 = time.perf_counter()
        conv_out, cache.conv_states[layer.layer_idx] = fused_conv1d_silu_cached(
            hidden_states_B_C.unsqueeze(1),
            cache.conv_states[layer.layer_idx],
            layer.conv1d_weight, layer.conv1d_bias)
        x, B_ssm, C_ssm = conv_out.split(
            [layer.intermediate_size,
             layer.n_groups * layer.state_size,
             layer.n_groups * layer.state_size], dim=-1)
        t1 = time.perf_counter()
        timings['conv1d_silu'].append(t1 - t0)

        # 5. SSM step (reshape + compute)
        t0 = time.perf_counter()
        x_r = x.reshape(B, layer.n_heads, layer.head_dim)
        B_r = B_ssm.reshape(B, layer.n_groups, -1)
        B_r = B_r[:, :, None, :].expand(
            B, layer.n_groups, layer.n_heads // layer.n_groups, layer.state_size
        ).contiguous().reshape(B, layer.n_heads, layer.state_size)
        C_r = C_ssm.reshape(B, layer.n_groups, -1)
        C_r = C_r[:, :, None, :].expand(
            B, layer.n_groups, layer.n_heads // layer.n_groups, layer.state_size
        ).contiguous().reshape(B, layer.n_heads, layer.state_size)
        A = -torch.exp(layer.A_log.float())
        dt_expanded = dt[:, None, :].transpose(1, 2).expand(B, layer.n_heads, layer.head_dim)
        if HAS_CPP_KERNEL:
            y, new_state = _cpu_ssm_ops.ssm_step(
                x_r, B_r, C_r, dt_expanded, A, layer.D,
                cache.ssm_states[layer.layer_idx],
                layer.dt_bias, layer.time_step_limit)
        else:
            from spec_mamba.cpu_mamba2 import ssm_step_pytorch
            y, new_state = ssm_step_pytorch(
                x_r, B_r, C_r, dt_expanded, A, layer.D,
                cache.ssm_states[layer.layer_idx],
                layer.dt_bias, layer.time_step_limit)
        cache.ssm_states[layer.layer_idx] = new_state
        y = y.reshape(B, -1).unsqueeze(1)
        t1 = time.perf_counter()
        timings['ssm_step'].append(t1 - t0)

        # 6. Norm + gate
        t0 = time.perf_counter()
        y = _rms_norm_with_gate(y, gate.unsqueeze(1), layer.norm_weight_ssm)
        t1 = time.perf_counter()
        timings['norm_gate'].append(t1 - t0)

        # 7. out_proj
        t0 = time.perf_counter()
        y = F.linear(y.squeeze(1), layer.out_proj_weight, layer.out_proj_bias).unsqueeze(1)
        t1 = time.perf_counter()
        timings['out_proj'].append(t1 - t0)

        # 8. Residual
        t0 = time.perf_counter()
        _ = residual + y
        t1 = time.perf_counter()
        timings['residual'].append(t1 - t0)

        # Restore cache
        cache.conv_states[layer.layer_idx] = conv_snap
        cache.ssm_states[layer.layer_idx] = ssm_snap

    # Convert to ms, compute stats
    results = {}
    for k, v in timings.items():
        arr = [x * 1000 for x in v]
        results[k] = {
            'mean_ms': sum(arr) / len(arr),
            'min_ms': min(arr),
            'max_ms': max(arr),
        }
    return results


def main():
    from transformers import AutoModelForCausalLM, AutoConfig
    drafter_path = "/HSC/users/qiaoye/SSM_SPEC/checkpoints/improved-mamba-alignment/checkpoint-750"

    print("Loading Mamba2 model...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        drafter_path, trust_remote_code=True, torch_dtype=torch.float32)
    hf_model = hf_model.cpu().eval()
    config = hf_model.config

    print("Converting to CPUMamba2Model...")
    model = CPUMamba2Model(hf_model)
    del hf_model

    print(f"Model: {config.num_hidden_layers} layers, hidden={config.hidden_size}, "
          f"d_inner={int(config.expand * config.hidden_size)}")
    print(f"C++ SSM kernel: {HAS_CPP_KERNEL}")
    print()

    # Profile single layer
    cache = CPUMamba2Cache(config, batch_size=1)
    layer = model.layers[0]
    hidden = torch.randn(1, 1, config.hidden_size, dtype=torch.float32)
    delta = torch.randn(1, 1, int(config.expand * config.hidden_size), dtype=torch.float32)

    print("=" * 60)
    print("Per-component breakdown (single layer, 1 token, B=1)")
    print("=" * 60)
    per_layer = profile_single_layer(layer, cache, hidden, delta, warmup=20, iters=100)
    total = sum(v['mean_ms'] for v in per_layer.values())
    for name, stats in per_layer.items():
        pct = stats['mean_ms'] / total * 100
        print(f"  {name:20s}: {stats['mean_ms']:.4f} ms  ({pct:5.1f}%)")
    print(f"  {'TOTAL':20s}: {total:.4f} ms")
    print()

    # Full 16-layer forward for K=1..8 tokens
    print("=" * 60)
    print("Full model K-token draft (16 layers + LM head)")
    print("=" * 60)
    draft_results = {}
    for K in [1, 2, 4, 8]:
        cache2 = model.create_cache(batch_size=1)
        token = torch.tensor([[1000]])
        # Warmup
        for _ in range(5):
            snap_c = [c.clone() for c in cache2.conv_states]
            snap_s = [s.clone() for s in cache2.ssm_states]
            for _ in range(K):
                model.forward_step(token, cache2)
            cache2.conv_states = snap_c
            cache2.ssm_states = snap_s

        times = []
        for _ in range(30):
            snap_c = [c.clone() for c in cache2.conv_states]
            snap_s = [s.clone() for s in cache2.ssm_states]
            t0 = time.perf_counter()
            for _ in range(K):
                model.forward_step(token, cache2)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
            cache2.conv_states = snap_c
            cache2.ssm_states = snap_s

        mean_ms = sum(times) / len(times)
        min_ms = min(times)
        print(f"  K={K}: mean={mean_ms:.2f} ms, min={min_ms:.2f} ms, per_token={mean_ms/K:.2f} ms")
        draft_results[K] = {'mean_ms': mean_ms, 'min_ms': min_ms, 'per_token_ms': mean_ms / K}

    output = {
        'per_layer_breakdown': per_layer,
        'per_layer_total_ms': total,
        'full_model_draft': draft_results,
        'config': {
            'model': drafter_path,
            'n_layers': config.num_hidden_layers,
            'hidden_size': config.hidden_size,
            'd_inner': int(config.expand * config.hidden_size),
            'cpp_kernel': HAS_CPP_KERNEL,
            'threads': os.environ.get('OMP_NUM_THREADS', 'default'),
        }
    }

    out_dir = Path("outputs/cpu_profile")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"component_profile_{ts}.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
