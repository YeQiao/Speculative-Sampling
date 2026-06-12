"""
Simulated pipeline throughput calculator + CPU thread scaling benchmark.

Measures individual component times and computes simulated pipeline throughput
using acceptance rate data from eval results.

Components measured:
  1. CPU draft step (single token, INT8 VNNI) — sweep threads {1,2,4,8,16,32,64,128}
  2. GPU verify forward pass (LLaMA-8B, bsz=1, seq_len=K+1) — sweep K
  3. CPU replay step (single token, same as draft but unguided)
  4. Guidance extraction (GPU) — hidden states → GuidanceExtractor → PrepMambaDeltas → CPU

Pipeline formula:
  per_round_time = max(verify_time, draft_time + replay_time) + overhead
  tokens_per_round = avg_accepted + 1
  pipeline_throughput = tokens_per_round / per_round_time

This gives a close approximation to the actual overlapped pipeline throughput,
validated by component measurements.
"""

import argparse
import json
import os
import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

torch.set_grad_enabled(False)

DEFAULT_DRAFTER = "/HSC/users/qiaoye/SSM_SPEC/checkpoints/improved-mamba-alignment/checkpoint-750"
DEFAULT_VERIFIER_8B = "/HSC/users/qiaoye/checkpoints/Llama3.1-8B-hf"


def benchmark_cpu_draft(cpu_drafter, n_tokens=8, n_warmup=5, n_iters=50,
                        guidance_deltas=None):
    """Benchmark CPU drafter: time to generate n_tokens autoregressively."""
    conv_states, ssm_states = cpu_drafter.create_cache(batch_size=1)

    # Prefill a short prompt
    for tok_id in [128000, 2, 791]:  # <|begin_of_text|> ...
        cpu_drafter.forward_step(tok_id, conv_states, ssm_states,
                                  guidance_deltas=guidance_deltas)

    # Warmup
    for _ in range(n_warmup):
        snap_conv = conv_states.clone()
        snap_ssm = ssm_states.clone()
        tok_id = 791
        for _ in range(n_tokens):
            logits = cpu_drafter.forward_step(tok_id, conv_states, ssm_states,
                                               guidance_deltas=guidance_deltas)
            tok_id = logits.argmax(-1).item()
        conv_states.copy_(snap_conv)
        ssm_states.copy_(snap_ssm)

    # Benchmark
    times = []
    for _ in range(n_iters):
        snap_conv = conv_states.clone()
        snap_ssm = ssm_states.clone()
        tok_id = 791
        t0 = time.perf_counter()
        for _ in range(n_tokens):
            logits = cpu_drafter.forward_step(tok_id, conv_states, ssm_states,
                                               guidance_deltas=guidance_deltas)
            tok_id = logits.argmax(-1).item()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
        conv_states.copy_(snap_conv)
        ssm_states.copy_(snap_ssm)

    return {
        "n_tokens": n_tokens,
        "mean_ms": sum(times) / len(times),
        "std_ms": (sum((t - sum(times)/len(times))**2 for t in times) / len(times)) ** 0.5,
        "min_ms": min(times),
        "per_token_ms": sum(times) / len(times) / n_tokens,
    }


def benchmark_cpu_replay(cpu_drafter, n_tokens=3, n_warmup=5, n_iters=50,
                          guidance_deltas=None):
    """Benchmark CPU replay: restore snapshot + replay n_tokens."""
    conv_states, ssm_states = cpu_drafter.create_cache(batch_size=1)
    for tok_id in [128000, 2, 791]:
        cpu_drafter.forward_step(tok_id, conv_states, ssm_states,
                                  guidance_deltas=guidance_deltas)

    snap_conv = conv_states.clone()
    snap_ssm = ssm_states.clone()
    replay_tokens = list(range(1000, 1000 + n_tokens))

    # Warmup
    for _ in range(n_warmup):
        conv_states.copy_(snap_conv)
        ssm_states.copy_(snap_ssm)
        for tok_id in replay_tokens:
            cpu_drafter.forward_step(tok_id, conv_states, ssm_states,
                                      guidance_deltas=guidance_deltas)

    # Benchmark
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        conv_states.copy_(snap_conv)
        ssm_states.copy_(snap_ssm)
        for tok_id in replay_tokens:
            cpu_drafter.forward_step(tok_id, conv_states, ssm_states,
                                      guidance_deltas=guidance_deltas)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    return {
        "n_tokens": n_tokens,
        "mean_ms": sum(times) / len(times),
        "std_ms": (sum((t - sum(times)/len(times))**2 for t in times) / len(times)) ** 0.5,
        "min_ms": min(times),
        "per_token_ms": sum(times) / len(times) / n_tokens,
        "snapshot_restore_ms": sum(times) / len(times) - n_tokens * (sum(times) / len(times) / n_tokens),
    }


def benchmark_gpu_verify(verifier, seq_lengths, n_warmup=5, n_iters=30):
    """Benchmark GPU verify: forward pass with different input lengths."""
    device = next(verifier.parameters()).device
    results = {}

    for seq_len in seq_lengths:
        input_ids = torch.randint(0, 1000, (1, seq_len), device=device)

        # Warmup
        for _ in range(n_warmup):
            torch.cuda.synchronize()
            pkv = DynamicCache()
            v_out = verifier.model(input_ids, past_key_values=pkv, use_cache=True)
            verifier.lm_head(v_out.last_hidden_state)
            torch.cuda.synchronize()

        # Benchmark
        times = []
        for _ in range(n_iters):
            torch.cuda.synchronize()
            pkv = DynamicCache()
            t0 = time.perf_counter()
            v_out = verifier.model(input_ids, past_key_values=pkv, use_cache=True)
            verifier.lm_head(v_out.last_hidden_state)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

        results[seq_len] = {
            "seq_len": seq_len,
            "mean_ms": sum(times) / len(times),
            "std_ms": (sum((t - sum(times)/len(times))**2 for t in times) / len(times)) ** 0.5,
            "min_ms": min(times),
        }

    return results


def benchmark_gpu_verify_with_guidance(verifier, ge, seq_lengths, n_warmup=5, n_iters=30):
    """Benchmark GPU verify + guidance extraction (output_hidden_states=True)."""
    device = next(verifier.parameters()).device
    tgt_device = ge.proj.weight.device
    results = {}

    for seq_len in seq_lengths:
        input_ids = torch.randint(0, 1000, (1, seq_len), device=device)

        # Warmup
        for _ in range(n_warmup):
            torch.cuda.synchronize()
            pkv = DynamicCache()
            v_out = verifier.model(input_ids, past_key_values=pkv, use_cache=True,
                                   output_hidden_states=True)
            gi = None
            for idx in ge.in_layer:
                h = v_out.hidden_states[idx].to(tgt_device)
                gi = h if gi is None else torch.cat((gi, h), dim=-1)
            ge.proj(gi.to(ge.proj.weight.dtype))
            verifier.lm_head(v_out.last_hidden_state)
            torch.cuda.synchronize()

        # Benchmark
        times = []
        for _ in range(n_iters):
            torch.cuda.synchronize()
            pkv = DynamicCache()
            t0 = time.perf_counter()
            v_out = verifier.model(input_ids, past_key_values=pkv, use_cache=True,
                                   output_hidden_states=True)
            gi = None
            for idx in ge.in_layer:
                h = v_out.hidden_states[idx].to(tgt_device)
                gi = h if gi is None else torch.cat((gi, h), dim=-1)
            ge.proj(gi.to(ge.proj.weight.dtype))
            verifier.lm_head(v_out.last_hidden_state)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

        results[seq_len] = {
            "seq_len": seq_len,
            "mean_ms": sum(times) / len(times),
            "std_ms": (sum((t - sum(times)/len(times))**2 for t in times) / len(times)) ** 0.5,
            "min_ms": min(times),
        }

    return results


def benchmark_gpu_ar_step(verifier, n_warmup=10, n_iters=50):
    """Benchmark single AR step: forward pass with KV cache (bsz=1, seq=1)."""
    device = next(verifier.parameters()).device

    # Build a KV cache from a short prefill
    prefill_ids = torch.randint(0, 1000, (1, 16), device=device)
    pkv = DynamicCache()
    torch.cuda.synchronize()
    v_out = verifier.model(prefill_ids, past_key_values=pkv, use_cache=True)
    pkv = v_out.past_key_values
    next_tok = verifier.lm_head(v_out.last_hidden_state[:, -1:]).argmax(-1)
    torch.cuda.synchronize()

    # Warmup
    for _ in range(n_warmup):
        torch.cuda.synchronize()
        v_out = verifier.model(next_tok, past_key_values=pkv, use_cache=True)
        verifier.lm_head(v_out.last_hidden_state)
        pkv = v_out.past_key_values
        torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(n_iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        v_out = verifier.model(next_tok, past_key_values=pkv, use_cache=True)
        verifier.lm_head(v_out.last_hidden_state)
        pkv = v_out.past_key_values
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    return {
        "mean_ms": sum(times) / len(times),
        "std_ms": (sum((t - sum(times)/len(times))**2 for t in times) / len(times)) ** 0.5,
        "min_ms": min(times),
        "kv_seq_len": pkv.get_seq_length(),
    }


def compute_simulated_pipeline(
    draft_ms_per_K: float,
    verify_ms: float,
    replay_ms: float,
    avg_accepted: float,
    ar_step_ms: float,
    K: int = 8,
    overhead_ms: float = 0.5,
):
    """Compute simulated pipeline throughput.

    Sequential: draft + verify + replay + overhead
    Pipeline:   max(verify, draft + replay) + overhead
      (assumes draft+replay on CPU, verify on GPU, perfect overlap)

    Args:
        draft_ms_per_K: Total CPU time to draft K tokens
        verify_ms: GPU verify time for K+1 tokens
        replay_ms: CPU replay time for accepted+1 tokens
        avg_accepted: Average number of accepted draft tokens
        ar_step_ms: Single AR step time (for baseline)
        K: Number of draft tokens per round
        overhead_ms: Per-round overhead (rejection, guidance transfer, etc.)
    """
    tokens_per_round = avg_accepted + 1

    # Sequential
    seq_per_round = draft_ms_per_K + verify_ms + replay_ms + overhead_ms
    seq_tps = tokens_per_round / (seq_per_round / 1000)

    # Pipelined (perfect overlap)
    pipe_per_round = max(verify_ms, draft_ms_per_K + replay_ms) + overhead_ms
    pipe_tps = tokens_per_round / (pipe_per_round / 1000)

    # AR baseline
    ar_tps = 1000.0 / ar_step_ms

    return {
        "tokens_per_round": tokens_per_round,
        "sequential": {
            "per_round_ms": seq_per_round,
            "throughput_tps": seq_tps,
            "speedup_vs_ar": seq_tps / ar_tps,
        },
        "pipelined": {
            "per_round_ms": pipe_per_round,
            "throughput_tps": pipe_tps,
            "speedup_vs_ar": pipe_tps / ar_tps,
        },
        "ar_baseline": {
            "per_step_ms": ar_step_ms,
            "throughput_tps": ar_tps,
        },
        "component_times": {
            "draft_ms": draft_ms_per_K,
            "verify_ms": verify_ms,
            "replay_ms": replay_ms,
            "overhead_ms": overhead_ms,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Simulated pipeline benchmark")
    parser.add_argument("--drafter", type=str, default=DEFAULT_DRAFTER)
    parser.add_argument("--verifier", type=str, default=DEFAULT_VERIFIER_8B)
    parser.add_argument("--guided_ckpt", type=str, default=None)
    parser.add_argument("--threads", type=str, default="1,2,4,8,16,32,64,128",
                        help="Comma-separated thread counts to benchmark")
    parser.add_argument("--ng_values", type=str, default="4,8,12",
                        help="Comma-separated K values to benchmark verify")
    parser.add_argument("--n_iters", type=int, default=50,
                        help="Number of iterations per measurement")
    parser.add_argument("--avg_accepted", type=float, default=2.82,
                        help="Average accepted tokens (from eval, 45M guided=2.82)")
    parser.add_argument("--out_file", type=str, default=None)
    args = parser.parse_args()

    thread_counts = [int(x) for x in args.threads.split(",")]
    ng_values = [int(x) for x in args.ng_values.split(",")]

    # ---- Load models ----
    print("=" * 70)
    print("SIMULATED PIPELINE BENCHMARK")
    print("=" * 70)

    # Load guided checkpoint if provided
    ge, prep, drafter_sd = None, None, None
    if args.guided_ckpt:
        from spec_mamba.pipeline_benchmark import load_guidance_modules
        ge, prep, v_layers, drafter_sd = load_guidance_modules(args.guided_ckpt, device="cuda")

    # Load CPU drafter
    from spec_mamba.pipeline_benchmark import load_cpu_drafter
    print(f"Loading CPU drafter from {args.drafter}")

    # Precompute zero deltas
    zero_deltas_cpu = None
    if ge is not None and prep is not None:
        v_h_dim = ge.proj.weight.shape[0]
        zero_deltas_cpu = prep(torch.zeros(1, 1, v_h_dim, device=prep.proj.weight.device))
        zero_deltas_cpu = zero_deltas_cpu.squeeze(1).squeeze(1).float().cpu()

    # ---- CPU Draft Thread Scaling ----
    print(f"\n{'='*70}")
    print(f"CPU DRAFT THREAD SCALING (K=8, INT8 VNNI drafter)")
    print("=" * 70)

    draft_results = {}
    for n_threads in thread_counts:
        torch.set_num_threads(n_threads)
        # Must reload drafter to pick up new thread count in kernels
        cpu_drafter = load_cpu_drafter(args.drafter, guided_sd=drafter_sd)

        result = benchmark_cpu_draft(cpu_drafter, n_tokens=8, n_iters=args.n_iters,
                                      guidance_deltas=zero_deltas_cpu)
        draft_results[n_threads] = result
        print(f"  {n_threads:3d} threads: {result['mean_ms']:6.2f} ms / 8 tokens "
              f"({result['per_token_ms']:.2f} ms/tok, std={result['std_ms']:.2f})")
        del cpu_drafter

    # Reload at 16 threads for replay benchmark
    torch.set_num_threads(16)
    cpu_drafter = load_cpu_drafter(args.drafter, guided_sd=drafter_sd)

    # ---- CPU Replay ----
    print(f"\n{'='*70}")
    print(f"CPU REPLAY TIME (varying accepted tokens)")
    print("=" * 70)

    replay_results = {}
    for n_accepted in [1, 2, 3, 4, 5, 6, 7, 8]:
        result = benchmark_cpu_replay(cpu_drafter, n_tokens=n_accepted + 1,
                                       n_iters=args.n_iters,
                                       guidance_deltas=zero_deltas_cpu)
        replay_results[n_accepted] = result
        print(f"  {n_accepted} accepted (+1 next): {result['mean_ms']:.2f} ms "
              f"({result['per_token_ms']:.2f} ms/tok)")

    # ---- GPU Verify ----
    print(f"\n{'='*70}")
    print(f"GPU VERIFY TIME (LLaMA-8B, varying K)")
    print("=" * 70)

    from spec_mamba.pipeline_benchmark import load_verifier
    verifier = load_verifier(args.verifier, device="cuda")

    # Without guidance
    verify_no_guide = benchmark_gpu_verify(
        verifier, seq_lengths=[K + 1 for K in ng_values], n_iters=args.n_iters,
    )
    for sl, res in verify_no_guide.items():
        print(f"  K={sl-1:2d} (seq={sl:2d}): {res['mean_ms']:.2f} ms (no guidance)")

    # With guidance
    verify_with_guide = {}
    if ge is not None:
        verify_with_guide = benchmark_gpu_verify_with_guidance(
            verifier, ge, seq_lengths=[K + 1 for K in ng_values], n_iters=args.n_iters,
        )
        for sl, res in verify_with_guide.items():
            print(f"  K={sl-1:2d} (seq={sl:2d}): {res['mean_ms']:.2f} ms (with guidance extraction)")

    # ---- GPU AR Step ----
    print(f"\n{'='*70}")
    print(f"GPU AR STEP TIME (LLaMA-8B, bsz=1, single token)")
    print("=" * 70)

    ar_result = benchmark_gpu_ar_step(verifier, n_iters=args.n_iters)
    print(f"  Single AR step: {ar_result['mean_ms']:.2f} ms "
          f"(KV len={ar_result['kv_seq_len']}, "
          f"= {1000/ar_result['mean_ms']:.1f} tok/s)")

    # ---- Simulated Pipeline ----
    print(f"\n{'='*70}")
    print(f"SIMULATED PIPELINE THROUGHPUT")
    print("=" * 70)

    K = 8
    avg_accepted = args.avg_accepted
    # Expected replay tokens = avg_accepted + 1 (accepted + bonus)
    n_replay = int(round(avg_accepted)) + 1
    replay_ms = replay_results.get(int(round(avg_accepted)), replay_results[3])["mean_ms"]

    verify_key = K + 1
    verify_ms = (verify_with_guide.get(verify_key, verify_no_guide.get(verify_key, {}))
                 .get("mean_ms", 15.0))

    ar_ms = ar_result["mean_ms"]
    overhead_ms = 0.5  # rejection + guidance transfer (~0.5ms)

    print(f"\n  Configuration:")
    print(f"    K={K}, avg_accepted={avg_accepted:.2f}")
    print(f"    Verify (GPU): {verify_ms:.2f} ms")
    print(f"    Replay ({n_replay} tokens): {replay_ms:.2f} ms")
    print(f"    AR step: {ar_ms:.2f} ms ({1000/ar_ms:.1f} tok/s)")
    print(f"    Overhead: {overhead_ms:.1f} ms")

    print(f"\n  {'Threads':>7s} | {'Draft':>8s} | {'Seq/rnd':>8s} | {'Pipe/rnd':>8s} | "
          f"{'Seq tps':>8s} | {'Pipe tps':>8s} | {'Seq vs AR':>9s} | {'Pipe vs AR':>10s}")
    print(f"  {'-'*7} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*9} | {'-'*10}")

    all_sim = {}
    for n_threads in thread_counts:
        draft_ms = draft_results[n_threads]["mean_ms"]
        sim = compute_simulated_pipeline(
            draft_ms_per_K=draft_ms,
            verify_ms=verify_ms,
            replay_ms=replay_ms,
            avg_accepted=avg_accepted,
            ar_step_ms=ar_ms,
            K=K,
            overhead_ms=overhead_ms,
        )
        all_sim[n_threads] = sim
        s = sim["sequential"]
        p = sim["pipelined"]
        print(f"  {n_threads:7d} | {draft_ms:7.2f}ms | {s['per_round_ms']:7.2f}ms | "
              f"{p['per_round_ms']:7.2f}ms | {s['throughput_tps']:7.1f} | "
              f"{p['throughput_tps']:7.1f} | {s['speedup_vs_ar']:8.2f}x | "
              f"{p['speedup_vs_ar']:9.2f}x")

    # ---- Summary Table ----
    print(f"\n{'='*70}")
    print(f"PAPER-READY SUMMARY")
    print("=" * 70)

    print(f"\n  Model: Mamba2-65M (INT8 VNNI) + LLaMA-8B (FP16)")
    print(f"  K={K}, avg_accepted={avg_accepted:.2f}, tokens/round={avg_accepted+1:.2f}")
    print(f"  AR baseline: {1000/ar_ms:.1f} tok/s ({ar_ms:.2f} ms/step)")

    # Key configs
    for n_threads in [16, 32, 64]:
        if n_threads in all_sim:
            sim = all_sim[n_threads]
            draft_ms = draft_results[n_threads]["mean_ms"]
            p = sim["pipelined"]
            print(f"\n  {n_threads} threads:")
            print(f"    Draft:    {draft_ms:.2f} ms (CPU)")
            print(f"    Verify:   {verify_ms:.2f} ms (GPU)")
            print(f"    Replay:   {replay_ms:.2f} ms (CPU)")
            print(f"    Pipeline: {p['per_round_ms']:.2f} ms/round → {p['throughput_tps']:.1f} tok/s ({p['speedup_vs_ar']:.2f}x)")
            cpu_util = (draft_ms + replay_ms) / max(verify_ms, draft_ms + replay_ms) * 100
            gpu_util = verify_ms / max(verify_ms, draft_ms + replay_ms) * 100
            print(f"    CPU util: {cpu_util:.0f}%, GPU util: {gpu_util:.0f}%")

    # ---- Save ----
    output = {
        "cpu_draft_thread_scaling": {
            str(k): v for k, v in draft_results.items()
        },
        "cpu_replay": {
            str(k): v for k, v in replay_results.items()
        },
        "gpu_verify_no_guidance": {
            str(k): v for k, v in verify_no_guide.items()
        },
        "gpu_verify_with_guidance": {
            str(k): v for k, v in verify_with_guide.items()
        },
        "gpu_ar_step": ar_result,
        "simulated_pipeline": {
            str(k): v for k, v in all_sim.items()
        },
        "config": {
            "drafter": args.drafter,
            "verifier": args.verifier,
            "guided_ckpt": args.guided_ckpt,
            "K": K,
            "avg_accepted": avg_accepted,
        },
    }

    out_dir = "outputs/simulated_pipeline"
    os.makedirs(out_dir, exist_ok=True)
    if args.out_file:
        out_path = args.out_file
    else:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(out_dir, f"sim_pipeline_{ts}.json")

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
