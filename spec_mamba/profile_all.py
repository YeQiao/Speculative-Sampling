"""
Comprehensive profiling for SpecSSM optimizations.

Benchmarks:
1. Activation replay vs full re-prefill (GPU)
2. CPU vs GPU drafter single-step latency
3. AVX-512 kernel vs PyTorch CPU for SSM step
4. Full CPU Mamba2 model single-step latency
5. End-to-end CPU draft latency for K tokens

Usage:
    python -m spec_mamba.profile_all \
        --ckpt /path/to/last.ckpt \
        --n_iters 100

    # Quick test:
    python -m spec_mamba.profile_all --ckpt /path/to/last.ckpt --n_iters 10 --quick
"""

import argparse
import json
import os
import time
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from transformers.models.mamba2.modeling_mamba2 import Mamba2Cache

torch.set_float32_matmul_precision("high")
torch.set_grad_enabled(False)


@contextmanager
def timer(name, results_dict, device="cpu"):
    """Context manager that records elapsed time in ms."""
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    yield
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000
    results_dict.setdefault(name, []).append(elapsed_ms)


def stats(times, skip_first=True):
    """Compute stats from a list of times, optionally skipping first (warmup)."""
    t = times[1:] if skip_first and len(times) > 1 else times
    if not t:
        return {"mean_ms": 0, "std_ms": 0, "min_ms": 0, "max_ms": 0, "n": 0}
    import statistics
    return {
        "mean_ms": statistics.mean(t),
        "std_ms": statistics.stdev(t) if len(t) > 1 else 0,
        "min_ms": min(t),
        "max_ms": max(t),
        "n": len(t),
    }


def load_from_ckpt(ckpt_path: str):
    from spec_mamba.trainer import SpecMambaTrainer
    st = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = {k: v for k, v in st["hyper_parameters"].items() if k != "_instantiator"}
    mod = SpecMambaTrainer(**hp)
    mod.load_state_dict(st["state_dict"], strict=False)
    return mod


# ============================================================
# Benchmark 1: AVX-512 kernel vs PyTorch CPU (SSM step only)
# ============================================================
def bench_ssm_kernel(n_iters: int = 500):
    """Benchmark the SSM single-step operation."""
    from spec_mamba.cpu_mamba2 import ssm_step_pytorch

    B, H, D, N = 1, 16, 64, 128
    torch.manual_seed(42)

    x = torch.randn(B, H, D)
    B_ssm = torch.randn(B, H, N)
    C_ssm = torch.randn(B, H, N)
    dt = torch.randn(B, H, D)
    A = -torch.rand(H).abs()
    D_skip = torch.randn(H)
    ssm_state = torch.randn(B, H, D, N)
    dt_bias = torch.randn(H)
    time_step_limit = (0.0, float("inf"))

    results = {}

    # PyTorch CPU
    for _ in range(5):  # warmup
        ssm_step_pytorch(x, B_ssm, C_ssm, dt, A, D_skip, ssm_state, dt_bias, time_step_limit)
    for i in range(n_iters):
        with timer("pytorch_cpu", results):
            ssm_step_pytorch(x, B_ssm, C_ssm, dt, A, D_skip, ssm_state.clone(), dt_bias, time_step_limit)

    # AVX-512  kernel
    try:
        from spec_mamba.cpu_kernels import get_cpu_ssm_ops
        ops = get_cpu_ssm_ops()
        for _ in range(5):
            ops.ssm_step(x, B_ssm, C_ssm, dt, A, D_skip, ssm_state, dt_bias, 0.0, float("inf"))
        for i in range(n_iters):
            with timer("avx512_kernel", results):
                ops.ssm_step(x, B_ssm, C_ssm, dt, A, D_skip, ssm_state.clone(), dt_bias, 0.0, float("inf"))
    except ImportError:
        print("  [SKIP] AVX-512 kernel not compiled")

    return {k: stats(v) for k, v in results.items()}


# ============================================================
# Benchmark 2: Full CPU Mamba2 layer single-step
# ============================================================
def bench_cpu_model_single_step(hf_model, n_iters: int = 200):
    """Benchmark a full Mamba2 single-step forward on CPU."""
    from spec_mamba.cpu_mamba2 import CPUMamba2Model

    cpu_model = CPUMamba2Model(hf_model)
    cache = cpu_model.create_cache(batch_size=1)

    # Warm up cache with a short sequence
    token = torch.randint(0, 1000, (1, 1))
    for _ in range(10):
        cpu_model.forward_step(token, cache)

    results = {}
    for i in range(n_iters):
        token = torch.randint(0, 1000, (1, 1))
        with timer("cpu_full_step", results):
            cpu_model.forward_step(token, cache)

    return {k: stats(v) for k, v in results.items()}


# ============================================================
# Benchmark 3: GPU drafter single-step
# ============================================================
def bench_gpu_drafter_step(mod, n_iters: int = 200):
    """Benchmark GPU drafter single-step cached forward."""
    device = "cuda"
    B = 1
    NG = mod.NG

    # Setup cache
    d_cache = Mamba2Cache(
        mod.d_base.backbone.config, B, device=device, dtype=torch.float32,
    )
    d_cache_pos = torch.arange(0, mod.d_base.backbone.config.conv_kernel, device=device)

    # Warm up
    dummy_deltas = mod.latent_mod_prep(
        torch.zeros(B, 1, mod.V_H_DIM, device=device)
    )
    token = torch.randint(0, 1000, (B, 1), device=device)
    for _ in range(10):
        mod._drafter_forward_cached(
            token, dummy_deltas,
            cache_params=d_cache, cache_position=torch.tensor([10], device=device),
        )

    results = {}
    for i in range(n_iters):
        token = torch.randint(0, 1000, (B, 1), device=device)
        with timer("gpu_single_step", results, device="cuda"):
            mod._drafter_forward_cached(
                token, dummy_deltas,
                cache_params=d_cache, cache_position=torch.tensor([20 + i], device=device),
            )

    return {k: stats(v) for k, v in results.items()}


# ============================================================
# Benchmark 4: Activation replay vs full re-prefill
# ============================================================
def bench_replay_vs_reprefill(mod, n_iters: int = 50):
    """Benchmark activation replay vs full re-prefill at various sequence lengths."""
    from spec_mamba.trainer import snapshot_mamba2_cache, restore_mamba2_cache

    device = "cuda"
    B = 1
    NG = mod.NG
    V_H_DIM = mod.V_H_DIM

    seq_lens = [64, 128, 256, 512]
    all_results = {}

    for seq_len in seq_lens:
        print(f"    seq_len={seq_len}...")
        dummy_ids = torch.randint(0, 1000, (B, seq_len), device=device)

        # Build initial cache
        d_cache = Mamba2Cache(
            mod.d_base.backbone.config, B, device=device, dtype=torch.float32,
        )
        d_cache_pos = torch.arange(0, mod.d_base.backbone.config.conv_kernel, device=device)
        prefill_deltas = mod.latent_mod_prep(
            torch.zeros(B, seq_len, V_H_DIM, device=device)
        )
        _, d_cache = mod._drafter_forward_cached(
            dummy_ids, prefill_deltas,
            cache_params=d_cache, cache_position=d_cache_pos,
        )

        results = {}

        # Full re-prefill
        for i in range(n_iters):
            new_cache = Mamba2Cache(
                mod.d_base.backbone.config, B, device=device, dtype=torch.float32,
            )
            new_pos = torch.arange(0, mod.d_base.backbone.config.conv_kernel, device=device)
            new_deltas = mod.latent_mod_prep(
                torch.zeros(B, seq_len, V_H_DIM, device=device)
            )
            with timer("reprefill", results, device="cuda"):
                mod._drafter_forward_cached(
                    dummy_ids, new_deltas,
                    cache_params=new_cache, cache_position=new_pos,
                )

        # Activation replay (na=4 accepted tokens)
        n_accepted = 4
        for i in range(n_iters):
            snap = snapshot_mamba2_cache(d_cache)

            # Simulate drafting (advance cache)
            for k in range(NG):
                tok = torch.randint(0, 1000, (B, 1), device=device)
                mod._drafter_forward_cached(
                    tok,
                    mod.latent_mod_prep(torch.zeros(B, 1, V_H_DIM, device=device)),
                    cache_params=d_cache,
                    cache_position=torch.tensor([seq_len + k], device=device),
                )

            replay_ids = torch.randint(0, 1000, (B, n_accepted + 1), device=device)

            with timer("replay", results, device="cuda"):
                restore_mamba2_cache(d_cache, snap)
                for ri in range(n_accepted + 1):
                    mod._drafter_forward_cached(
                        replay_ids[:, ri:ri+1],
                        mod.latent_mod_prep(torch.zeros(B, 1, V_H_DIM, device=device)),
                        cache_params=d_cache,
                        cache_position=torch.tensor([seq_len + ri], device=device),
                    )

        all_results[f"seq_{seq_len}"] = {k: stats(v) for k, v in results.items()}

    return all_results


# ============================================================
# Benchmark 5: CPU draft K tokens (end-to-end)
# ============================================================
def bench_cpu_draft_k_tokens(hf_model, K: int = 8, n_iters: int = 50):
    """Benchmark drafting K tokens on CPU (end-to-end)."""
    from spec_mamba.cpu_mamba2 import CPUMamba2Model

    cpu_model = CPUMamba2Model(hf_model)
    cache = cpu_model.create_cache(batch_size=1)

    # Warm up with a short prefill
    for t in range(32):
        token = torch.randint(0, 1000, (1, 1))
        cpu_model.forward_step(token, cache)

    results = {}
    for i in range(n_iters):
        snap = cache.snapshot()  # snapshot for activation replay later
        token = torch.randint(0, 1000, (1, 1))

        with timer(f"cpu_draft_{K}_tokens", results):
            for k in range(K):
                logits = cpu_model.forward_step(token, cache)
                token = logits.argmax(dim=-1, keepdim=True)

        # Restore cache for next iteration
        cache.restore(snap)

    return {k: stats(v) for k, v in results.items()}


def main():
    parser = argparse.ArgumentParser(description="Profile SpecSSM optimizations")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--n_iters", type=int, default=100)
    parser.add_argument("--out_file", type=str, default="spec_mamba/profile_all_results.json")
    parser.add_argument("--quick", action="store_true", help="Quick run with fewer iterations")
    args = parser.parse_args()

    n = args.n_iters
    if args.quick:
        n = max(n // 10, 5)

    all_results = {}

    # 1. SSM Kernel Benchmark
    print("\n[1/5] SSM Kernel: AVX-512 vs PyTorch CPU...")
    all_results["ssm_kernel"] = bench_ssm_kernel(n_iters=n * 5)

    # 2. Load model for GPU benchmarks
    print("\n[2/5] Loading model...")
    mod = load_from_ckpt(args.ckpt)
    mod.eval()

    # 3. CPU model single-step
    print("\n[3/5] CPU Mamba2 full single-step...")
    all_results["cpu_model_step"] = bench_cpu_model_single_step(mod.d_base, n_iters=n)

    # 4. CPU draft K tokens
    print("\n[4/5] CPU draft 8 tokens end-to-end...")
    all_results["cpu_draft_K"] = bench_cpu_draft_k_tokens(mod.d_base, K=8, n_iters=n)

    # 5. Move to GPU for activation replay benchmark
    mod.to("cuda")
    print("\n[5a/5] GPU drafter single-step...")
    all_results["gpu_drafter_step"] = bench_gpu_drafter_step(mod, n_iters=n)

    print("\n[5b/5] Activation replay vs re-prefill...")
    all_results["replay_vs_reprefill"] = bench_replay_vs_reprefill(mod, n_iters=max(n // 2, 5))

    # Save results
    os.makedirs(os.path.dirname(args.out_file) or ".", exist_ok=True)
    with open(args.out_file, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print("PROFILING SUMMARY")
    print("=" * 70)

    if "ssm_kernel" in all_results:
        print("\n--- SSM Kernel (single step, B=1, H=16, D=64, N=128) ---")
        for k, v in all_results["ssm_kernel"].items():
            print(f"  {k:20s}: {v['mean_ms']:.4f} ms (±{v['std_ms']:.4f})")
        if "avx512_kernel" in all_results["ssm_kernel"] and "pytorch_cpu" in all_results["ssm_kernel"]:
            speedup = (all_results["ssm_kernel"]["pytorch_cpu"]["mean_ms"] /
                      max(all_results["ssm_kernel"]["avx512_kernel"]["mean_ms"], 1e-9))
            print(f"  {'AVX-512 speedup':20s}: {speedup:.2f}x")

    if "cpu_model_step" in all_results:
        print("\n--- CPU Mamba2 Full Single-Step ---")
        for k, v in all_results["cpu_model_step"].items():
            print(f"  {k:20s}: {v['mean_ms']:.3f} ms (±{v['std_ms']:.3f})")

    if "cpu_draft_K" in all_results:
        print("\n--- CPU Draft 8 Tokens End-to-End ---")
        for k, v in all_results["cpu_draft_K"].items():
            print(f"  {k:20s}: {v['mean_ms']:.3f} ms (±{v['std_ms']:.3f})")

    if "gpu_drafter_step" in all_results:
        print("\n--- GPU Drafter Single-Step ---")
        for k, v in all_results["gpu_drafter_step"].items():
            print(f"  {k:20s}: {v['mean_ms']:.4f} ms (±{v['std_ms']:.4f})")

    if "replay_vs_reprefill" in all_results:
        print("\n--- Activation Replay vs Full Re-Prefill ---")
        for sl_key, data in all_results["replay_vs_reprefill"].items():
            reprefill = data.get("reprefill", {})
            replay = data.get("replay", {})
            speedup = reprefill.get("mean_ms", 0) / max(replay.get("mean_ms", 1e-9), 1e-9)
            print(f"  {sl_key}: reprefill={reprefill.get('mean_ms', 0):.2f} ms, "
                  f"replay={replay.get('mean_ms', 0):.2f} ms, speedup={speedup:.2f}x")

    # CPU vs GPU comparison
    if "cpu_model_step" in all_results and "gpu_drafter_step" in all_results:
        cpu_t = list(all_results["cpu_model_step"].values())[0]["mean_ms"]
        gpu_t = list(all_results["gpu_drafter_step"].values())[0]["mean_ms"]
        print(f"\n--- CPU vs GPU Drafter Step ---")
        print(f"  CPU: {cpu_t:.3f} ms, GPU: {gpu_t:.4f} ms, ratio: {cpu_t / max(gpu_t, 1e-9):.1f}x slower")

        # Draft K tokens
        if "cpu_draft_K" in all_results:
            cpu_k = list(all_results["cpu_draft_K"].values())[0]["mean_ms"]
            gpu_k = gpu_t * 8  # approximate GPU 8-token draft
            print(f"  CPU 8 tokens: {cpu_k:.2f} ms, GPU 8 tokens (est): {gpu_k:.2f} ms")

    print(f"\nResults saved to {args.out_file}")


if __name__ == "__main__":
    main()
