"""
Profiling script for activation replay vs full re-prefill.

Compares:
1. Full re-prefill (original): rebuild drafter cache from scratch over entire sequence
2. Activation replay: restore snapshot + replay only accepted tokens

Measures per-round cache rebuild latency at various sequence lengths.

Usage:
    python -m spec_mamba.profile_replay \
        --ckpt /path/to/last.ckpt \
        --seq_lens 64,128,256,512,1024 \
        --n_rounds 20
"""

import argparse
import json
import os
import time

import torch
from transformers.models.mamba2.modeling_mamba2 import Mamba2Cache

from spec_mamba.trainer import (
    SpecMambaTrainer,
    snapshot_mamba2_cache,
    restore_mamba2_cache,
)

torch.set_float32_matmul_precision("high")
torch.set_grad_enabled(False)


def load_from_ckpt(ckpt_path: str) -> SpecMambaTrainer:
    st = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = {k: v for k, v in st["hyper_parameters"].items() if k != "_instantiator"}
    mod = SpecMambaTrainer(**hp)
    mod.load_state_dict(st["state_dict"], strict=False)
    return mod


def profile_cache_rebuild(
    mod: SpecMambaTrainer,
    seq_len: int,
    n_rounds: int = 20,
    n_accepted: int = 4,
    device: str = "cuda",
) -> dict:
    """Profile full re-prefill vs activation replay for a given sequence length."""
    B = 1
    NG = mod.NG
    V_H_DIM = mod.V_H_DIM

    # Create dummy input sequence
    dummy_ids = torch.randint(0, 1000, (B, seq_len), device=device)

    # --- Warmup: build initial drafter cache ---
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

    # --- Profile FULL RE-PREFILL ---
    torch.cuda.synchronize()
    reprefill_times = []
    for _ in range(n_rounds):
        # Simulate: create fresh cache and re-prefill entire sequence
        new_cache = Mamba2Cache(
            mod.d_base.backbone.config, B, device=device, dtype=torch.float32,
        )
        new_cache_pos = torch.arange(0, mod.d_base.backbone.config.conv_kernel, device=device)
        new_deltas = mod.latent_mod_prep(
            torch.zeros(B, seq_len, V_H_DIM, device=device)
        )

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        _, new_cache = mod._drafter_forward_cached(
            dummy_ids, new_deltas,
            cache_params=new_cache, cache_position=new_cache_pos,
        )

        torch.cuda.synchronize()
        reprefill_times.append(time.perf_counter() - t0)

    # --- Profile ACTIVATION REPLAY ---
    # First, we need a snapshot of the cache BEFORE drafting
    snapshot = snapshot_mamba2_cache(d_cache)
    d_next_pos = seq_len

    replay_times = []
    snapshot_times = []
    restore_times = []

    for _ in range(n_rounds):
        # Measure snapshot cost
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        snap = snapshot_mamba2_cache(d_cache)
        torch.cuda.synchronize()
        snapshot_times.append(time.perf_counter() - t0)

        # Simulate: draft NG tokens (just advance cache position)
        for i in range(NG):
            dummy_tok = torch.randint(0, 1000, (B, 1), device=device)
            pos_i = torch.tensor([d_next_pos + i], device=device)
            _, d_cache = mod._drafter_forward_cached(
                dummy_tok,
                mod.latent_mod_prep(torch.zeros(B, 1, V_H_DIM, device=device)),
                cache_params=d_cache, cache_position=pos_i,
            )

        # Measure restore cost
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        restore_mamba2_cache(d_cache, snap)
        torch.cuda.synchronize()
        restore_times.append(time.perf_counter() - t0)

        # Measure replay cost (replay n_accepted + 1 tokens)
        replay_len = n_accepted + 1
        replay_ids = torch.randint(0, 1000, (B, replay_len), device=device)

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        for ri in range(replay_len):
            pos_i = torch.tensor([d_next_pos + ri], device=device)
            _, d_cache = mod._drafter_forward_cached(
                replay_ids[:, ri:ri+1],
                mod.latent_mod_prep(torch.zeros(B, 1, V_H_DIM, device=device)),
                cache_params=d_cache, cache_position=pos_i,
            )

        torch.cuda.synchronize()
        replay_times.append(time.perf_counter() - t0)

    # Skip first measurement (warmup)
    def stats(times):
        times = times[1:]
        return {
            "mean_ms": sum(times) / len(times) * 1000,
            "min_ms": min(times) * 1000,
            "max_ms": max(times) * 1000,
        }

    return {
        "seq_len": seq_len,
        "n_accepted": n_accepted,
        "full_reprefill": stats(reprefill_times),
        "snapshot": stats(snapshot_times),
        "restore": stats(restore_times),
        "replay_forward": stats(replay_times),
        "replay_total": {
            "mean_ms": (
                stats(snapshot_times)["mean_ms"]
                + stats(restore_times)["mean_ms"]
                + stats(replay_times)["mean_ms"]
            ),
        },
        "speedup": stats(reprefill_times)["mean_ms"] / (
            stats(snapshot_times)["mean_ms"]
            + stats(restore_times)["mean_ms"]
            + stats(replay_times)["mean_ms"]
            + 1e-9
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--seq_lens", type=str, default="32,64,128,256,512,1024")
    parser.add_argument("--n_rounds", type=int, default=20)
    parser.add_argument("--n_accepted", type=int, default=4,
                        help="Simulated number of accepted tokens per round")
    parser.add_argument("--out_file", type=str, default="spec_mamba/profile_replay_results.json")
    args = parser.parse_args()

    seq_lens = [int(x) for x in args.seq_lens.split(",")]

    print("Loading model...")
    mod = load_from_ckpt(args.ckpt)
    mod.eval()
    mod.to("cuda")

    results = []
    for sl in seq_lens:
        print(f"\nProfiling seq_len={sl}...")
        r = profile_cache_rebuild(
            mod, seq_len=sl, n_rounds=args.n_rounds,
            n_accepted=args.n_accepted,
        )
        results.append(r)
        print(f"  Full re-prefill:    {r['full_reprefill']['mean_ms']:.2f} ms")
        print(f"  Activation replay:  {r['replay_total']['mean_ms']:.2f} ms "
              f"(snapshot={r['snapshot']['mean_ms']:.2f}, "
              f"restore={r['restore']['mean_ms']:.2f}, "
              f"replay={r['replay_forward']['mean_ms']:.2f})")
        print(f"  Speedup:            {r['speedup']:.2f}x")

    os.makedirs(os.path.dirname(args.out_file) or ".", exist_ok=True)
    with open(args.out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.out_file}")

    # Summary table
    print("\n" + "=" * 70)
    print(f"{'SeqLen':>8} | {'Re-prefill (ms)':>15} | {'Replay (ms)':>12} | {'Speedup':>8}")
    print("-" * 70)
    for r in results:
        print(f"{r['seq_len']:>8} | {r['full_reprefill']['mean_ms']:>15.2f} | "
              f"{r['replay_total']['mean_ms']:>12.2f} | {r['speedup']:>7.2f}x")
    print("=" * 70)


if __name__ == "__main__":
    main()
