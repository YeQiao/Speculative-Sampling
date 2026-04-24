"""
CPU-offloaded Mamba2 drafter benchmark: activation replay e2e latency
and full CPU-offloaded speculative decoding wallclock.

Compares:
1. HF Mamba2 on CPU (naive re-prefill) vs CPU model with activation replay
2. Full CPU-offloaded spec dec pipeline wallclock vs GPU-only
3. Breakdown by sequence length

Usage:
    python -m spec_mamba.benchmark_cpu_offload
"""

import json
import os
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from transformers.models.mamba2.modeling_mamba2 import Mamba2Cache

from spec_mamba.cpu_mamba2 import CPUMamba2Model, CPUMamba2Cache
from spec_mamba.models.llama import LlamaForCausalLM as CustomLlamaForCausalLM

torch.set_grad_enabled(False)

DRAFTER_PATH = "/HSC/users/qiaoye/SSM_SPEC/checkpoints/improved-mamba-alignment/checkpoint-750"
VERIFIER_PATH = "/HSC/users/qiaoye/checkpoints/Llama3.1-8B-hf"
NG = 8  # draft tokens per round

PYTHON = "/HSC/users/qiaoye/envs/ssm_spec_py310/bin/python"


def load_models():
    """Load all models needed for benchmarking."""
    print("Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(VERIFIER_PATH)
    tok.pad_token = tok.eos_token

    print("Loading HF Mamba2 drafter (FP32, CPU)...")
    hf_drafter = AutoModelForCausalLM.from_pretrained(
        DRAFTER_PATH, torch_dtype=torch.float32,
    ).eval()

    print("Building CPU-optimized Mamba2 model...")
    cpu_model = CPUMamba2Model(hf_drafter)

    print("Loading verifier (LLaMA-8B, FP16, GPU)...")
    verifier = CustomLlamaForCausalLM.from_pretrained(
        VERIFIER_PATH, torch_dtype=torch.float16,
    ).cuda().eval()

    return tok, hf_drafter, cpu_model, verifier


# =========================================================================
# 1. Activation Replay E2E Benchmark
# =========================================================================

def benchmark_replay_vs_reprefill(hf_drafter, cpu_model, seq_lengths, n_accepted=5, n_warmup=1, n_trials=3):
    """
    Compare drafter cache resynchronization strategies:
    - HF Mamba2 full re-prefill (CPU)
    - CPU model full re-prefill
    - CPU model activation replay
    """
    import sys
    print("\n" + "=" * 70)
    print("BENCHMARK 1: Activation Replay vs Re-prefill (CPU drafter)")
    print("=" * 70)
    print(f"Simulating: {NG} drafted, {n_accepted} accepted, replay {n_accepted} tokens")
    print(f"Seq lengths: {seq_lengths}\n", flush=True)

    results = {}

    for seq_len in seq_lengths:
        print(f"  seq_len={seq_len}:", flush=True)

        # Create a fake prompt
        prompt = torch.randint(100, 5000, (1, seq_len))

        # ---- HF Mamba2 re-prefill (CPU) ----
        times_hf_reprefill = []
        for trial in range(n_warmup + n_trials):
            # Simulate: re-prefill entire sequence
            hf_cache = Mamba2Cache(hf_drafter.backbone.config, 1, device='cpu', dtype=torch.float32)
            hf_pos = torch.arange(0, hf_drafter.backbone.config.conv_kernel, device='cpu')

            t0 = time.perf_counter()
            _ = hf_drafter(prompt, cache_params=hf_cache, cache_position=hf_pos)
            t1 = time.perf_counter()

            if trial >= n_warmup:
                times_hf_reprefill.append((t1 - t0) * 1000)

        hf_reprefill_ms = sum(times_hf_reprefill) / len(times_hf_reprefill)

        # ---- CPU model re-prefill ----
        times_cpu_reprefill = []
        for trial in range(n_warmup + n_trials):
            cpu_cache = cpu_model.create_cache(1)

            t0 = time.perf_counter()
            cpu_model.prefill(prompt, cpu_cache)
            t1 = time.perf_counter()

            if trial >= n_warmup:
                times_cpu_reprefill.append((t1 - t0) * 1000)

        cpu_reprefill_ms = sum(times_cpu_reprefill) / len(times_cpu_reprefill)

        # ---- CPU model activation replay ----
        # First, set up a baseline cache state (simulate we already had a cache)
        cpu_cache_base = cpu_model.create_cache(1)
        cpu_model.prefill(prompt, cpu_cache_base)

        # Snapshot before drafting
        snapshot_conv = [s.clone() for s in cpu_cache_base.conv_states]
        snapshot_ssm = [s.clone() for s in cpu_cache_base.ssm_states]

        # Simulate: draft 8 tokens (pollutes cache)
        draft_tokens = torch.randint(100, 5000, (1, NG))
        for t in range(NG):
            cpu_model.forward_step(draft_tokens[:, t:t+1], cpu_cache_base)

        # Now benchmark replay: restore + replay accepted
        accepted_tokens = draft_tokens[:, :n_accepted]

        times_replay = []
        for trial in range(n_warmup + n_trials):
            t0 = time.perf_counter()

            # Restore snapshot
            for i in range(cpu_model.n_layers):
                cpu_cache_base.conv_states[i] = snapshot_conv[i].clone()
                cpu_cache_base.ssm_states[i] = snapshot_ssm[i].clone()

            # Replay accepted tokens
            for t in range(n_accepted):
                cpu_model.forward_step(accepted_tokens[:, t:t+1], cpu_cache_base)

            t1 = time.perf_counter()
            if trial >= n_warmup:
                times_replay.append((t1 - t0) * 1000)

        replay_ms = sum(times_replay) / len(times_replay)

        speedup_vs_hf = hf_reprefill_ms / replay_ms
        speedup_vs_cpu = cpu_reprefill_ms / replay_ms

        results[seq_len] = {
            "hf_reprefill_ms": round(hf_reprefill_ms, 2),
            "cpu_reprefill_ms": round(cpu_reprefill_ms, 2),
            "replay_ms": round(replay_ms, 2),
            "speedup_vs_hf_reprefill": round(speedup_vs_hf, 1),
            "speedup_vs_cpu_reprefill": round(speedup_vs_cpu, 1),
        }
        print(f"    HF re-prefill:    {hf_reprefill_ms:8.1f} ms")
        print(f"    CPU re-prefill:   {cpu_reprefill_ms:8.1f} ms")
        print(f"    Activation replay:{replay_ms:8.1f} ms  "
              f"({speedup_vs_hf:.1f}x vs HF, {speedup_vs_cpu:.1f}x vs CPU reprefill)", flush=True)

    return results


# =========================================================================
# 2. Single-Step Latency: CPU model vs HF CPU
# =========================================================================

def benchmark_single_step(hf_drafter, cpu_model, n_warmup=3, n_trials=20):
    """Compare single-step latency: HF Mamba2 on CPU vs CPU-optimized model."""
    print("\n" + "=" * 70)
    print("BENCHMARK 2: Single-Step Latency (CPU)")
    print("=" * 70)

    prompt = torch.randint(100, 5000, (1, 32))

    # Setup HF
    hf_cache = Mamba2Cache(hf_drafter.backbone.config, 1, device='cpu', dtype=torch.float32)
    hf_pos = torch.arange(0, hf_drafter.backbone.config.conv_kernel, device='cpu')
    _ = hf_drafter(prompt, cache_params=hf_cache, cache_position=hf_pos)

    # Setup CPU model
    cpu_cache = cpu_model.create_cache(1)
    cpu_model.prefill(prompt, cpu_cache)

    next_tok = torch.tensor([[16]])

    # HF single step
    times_hf = []
    for i in range(n_warmup + n_trials):
        pos = torch.tensor([hf_drafter.backbone.config.conv_kernel + i])
        t0 = time.perf_counter()
        _ = hf_drafter(next_tok, cache_params=hf_cache, cache_position=pos)
        t1 = time.perf_counter()
        if i >= n_warmup:
            times_hf.append((t1 - t0) * 1000)

    # Re-setup for CPU model (HF calls polluted its cache)
    cpu_cache2 = cpu_model.create_cache(1)
    cpu_model.prefill(prompt, cpu_cache2)

    times_cpu = []
    for i in range(n_warmup + n_trials):
        t0 = time.perf_counter()
        _ = cpu_model.forward_step(next_tok, cpu_cache2)
        t1 = time.perf_counter()
        if i >= n_warmup:
            times_cpu.append((t1 - t0) * 1000)

    hf_ms = sum(times_hf) / len(times_hf)
    cpu_ms = sum(times_cpu) / len(times_cpu)
    speedup = hf_ms / cpu_ms

    print(f"  HF Mamba2 (CPU):        {hf_ms:.2f} ms/step")
    print(f"  CPU-optimized model:    {cpu_ms:.2f} ms/step")
    print(f"  Speedup:                {speedup:.2f}x")

    return {
        "hf_cpu_ms": round(hf_ms, 2),
        "cpu_optimized_ms": round(cpu_ms, 2),
        "speedup": round(speedup, 2),
    }


# =========================================================================
# 3. CPU-Offloaded Spec Dec Wallclock
# =========================================================================

def benchmark_cpu_offloaded_spec_dec(cpu_model, verifier, tok, n_rounds=10, n_warmup=2):
    """
    Measure end-to-end wallclock for CPU-offloaded speculative decoding.

    Pipeline:
    1. CPU drafter generates K=8 draft tokens
    2. Transfer draft tokens to GPU (trivial, 8 ints)
    3. GPU verifier processes K+1 tokens
    4. Transfer logits back to CPU for comparison (or compare on GPU)

    We measure each component separately and the total.
    """
    print("\n" + "=" * 70)
    print("BENCHMARK 3: CPU-Offloaded Spec Dec Wallclock")
    print("=" * 70)
    print(f"Pipeline: CPU drafter (K={NG}) → transfer → GPU verifier → compare")

    # Create a synthetic prompt
    prompt_text = "The quick brown fox jumps over the lazy dog. In recent years, the field of machine learning has seen significant advances in"
    toks = tok(prompt_text, return_tensors="pt", truncation=True, max_length=128)
    prompt_ids = toks["input_ids"]  # [1, S]
    S = prompt_ids.shape[1]

    # Prefill CPU drafter
    cpu_cache = cpu_model.create_cache(1)
    cpu_model.prefill(prompt_ids, cpu_cache)

    # Prefill GPU verifier
    prompt_gpu = prompt_ids.cuda()
    v_pkv = DynamicCache()
    with torch.cuda.amp.autocast(dtype=torch.float16):
        v_out = verifier.get_decoder()(
            prompt_gpu, past_key_values=v_pkv, use_cache=True,
        )
    v_pkv = v_out.past_key_values
    v_last_logits = verifier.lm_head(v_out.last_hidden_state[:, -1:])
    # Initialize with verifier's next token
    next_tok_cpu = v_last_logits.argmax(dim=-1).cpu()  # [1, 1]

    # Measure individual components
    times_draft = []
    times_transfer_to_gpu = []
    times_verify = []
    times_transfer_back = []
    times_total = []

    for trial in range(n_warmup + n_rounds):
        # Save cache snapshot for replay
        snap_conv = [s.clone() for s in cpu_cache.conv_states]
        snap_ssm = [s.clone() for s in cpu_cache.ssm_states]

        # === TOTAL TIMER START ===
        total_t0 = time.perf_counter()

        # --- 1. CPU Draft ---
        t0 = time.perf_counter()
        draft_tokens = []
        draft_logits = []
        tok_in = next_tok_cpu
        for _ in range(NG):
            logits = cpu_model.forward_step(tok_in, cpu_cache)
            next_t = logits.argmax(dim=-1, keepdim=True)  # [1, 1]
            draft_tokens.append(next_t)
            draft_logits.append(logits.softmax(dim=-1))
            tok_in = next_t
        draft_seq = torch.cat(draft_tokens, dim=1)  # [1, NG]
        t1 = time.perf_counter()
        draft_time = t1 - t0

        # --- 2. Transfer to GPU ---
        t0 = time.perf_counter()
        # Transfer draft tokens + the token before them
        verify_input = torch.cat([next_tok_cpu, draft_seq], dim=1).cuda()  # [1, NG+1]
        t1 = time.perf_counter()
        transfer_to_gpu_time = t1 - t0

        # --- 3. GPU Verify ---
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.cuda.amp.autocast(dtype=torch.float16):
            v_out = verifier.get_decoder()(
                verify_input, past_key_values=v_pkv, use_cache=True,
            )
        v_logits = verifier.lm_head(v_out.last_hidden_state)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        verify_time = t1 - t0

        # --- 4. Transfer back + compare ---
        t0 = time.perf_counter()
        v_next_tokens = v_logits.argmax(dim=-1).cpu()  # [1, NG+1]
        # Simple greedy acceptance: compare draft tokens with verifier's predictions
        n_accepted = 0
        for j in range(NG):
            if draft_seq[0, j].item() == v_next_tokens[0, j].item():
                n_accepted += 1
            else:
                break
        t1 = time.perf_counter()
        transfer_back_time = t1 - t0

        total_t1 = time.perf_counter()
        total_time = total_t1 - total_t0

        # Restore cache + replay accepted (for next round)
        for i in range(cpu_model.n_layers):
            cpu_cache.conv_states[i] = snap_conv[i].clone()
            cpu_cache.ssm_states[i] = snap_ssm[i].clone()
        for t in range(n_accepted):
            cpu_model.forward_step(draft_seq[:, t:t+1], cpu_cache)

        # Use verifier's token at acceptance boundary for next round
        next_tok_cpu = v_next_tokens[:, n_accepted:n_accepted+1]

        # Roll back verifier KV cache (crop to before this round's tokens)
        # In a real system we'd keep the accepted prefix. For benchmarking, skip.
        v_pkv = v_out.past_key_values  # just keep growing

        if trial >= n_warmup:
            times_draft.append(draft_time * 1000)
            times_transfer_to_gpu.append(transfer_to_gpu_time * 1000)
            times_verify.append(verify_time * 1000)
            times_transfer_back.append(transfer_back_time * 1000)
            times_total.append(total_time * 1000)

    avg = lambda lst: sum(lst) / len(lst)
    results = {
        "draft_cpu_ms": round(avg(times_draft), 2),
        "transfer_to_gpu_ms": round(avg(times_transfer_to_gpu), 3),
        "verify_gpu_ms": round(avg(times_verify), 2),
        "transfer_back_ms": round(avg(times_transfer_back), 3),
        "total_round_ms": round(avg(times_total), 2),
        "draft_fraction": round(avg(times_draft) / avg(times_total), 3),
        "verify_fraction": round(avg(times_verify) / avg(times_total), 3),
    }

    print(f"\n  Component breakdown (avg over {n_rounds} rounds):")
    print(f"    CPU Draft ({NG} tokens):   {results['draft_cpu_ms']:7.2f} ms  ({results['draft_fraction']*100:.1f}%)")
    print(f"    Transfer → GPU:        {results['transfer_to_gpu_ms']:7.3f} ms")
    print(f"    GPU Verify ({NG}+1 tok):  {results['verify_gpu_ms']:7.2f} ms  ({results['verify_fraction']*100:.1f}%)")
    print(f"    Transfer ← CPU:        {results['transfer_back_ms']:7.3f} ms")
    print(f"    Total round:           {results['total_round_ms']:7.2f} ms")

    # Estimate throughput with typical acceptance
    for na in [1, 3, 5, 8]:
        tokens_per_round = na + 1
        replay_est = na * results['draft_cpu_ms'] / NG  # rough: linear in n_accepted
        total_est = results['total_round_ms'] + replay_est
        tps = tokens_per_round / (total_est / 1000)
        print(f"    If n_accepted={na}: ~{tps:.0f} tok/s ({tokens_per_round} tokens in {total_est:.1f}ms)")

    return results


# =========================================================================
# 4. Pipeline Overlap Potential
# =========================================================================

def benchmark_overlap_potential(cpu_model, verifier, tok):
    """
    Measure draft and verify times independently to quantify overlap potential.
    If draft_time <= verify_time, async pipeline gives free drafting.
    """
    print("\n" + "=" * 70)
    print("BENCHMARK 4: Pipeline Overlap Potential")
    print("=" * 70)

    prompt_text = "The quick brown fox jumps over the lazy dog. In recent years,"
    toks = tok(prompt_text, return_tensors="pt", truncation=True, max_length=128)
    prompt_ids = toks["input_ids"]
    S = prompt_ids.shape[1]

    # Measure CPU draft for various K
    results = {}
    for K in [4, 6, 8, 12]:
        cpu_cache = cpu_model.create_cache(1)
        cpu_model.prefill(prompt_ids, cpu_cache)

        # Draft K tokens
        times = []
        for trial in range(5 + 20):
            snap_conv = [s.clone() for s in cpu_cache.conv_states]
            snap_ssm = [s.clone() for s in cpu_cache.ssm_states]

            t0 = time.perf_counter()
            tok_in = prompt_ids[:, -1:]
            for _ in range(K):
                logits = cpu_model.forward_step(tok_in, cpu_cache)
                tok_in = logits.argmax(dim=-1, keepdim=True)
            t1 = time.perf_counter()

            for i in range(cpu_model.n_layers):
                cpu_cache.conv_states[i] = snap_conv[i]
                cpu_cache.ssm_states[i] = snap_ssm[i]

            if trial >= 5:
                times.append((t1 - t0) * 1000)

        avg_draft = sum(times) / len(times)

        # Measure GPU verify for K+1 tokens
        prompt_gpu = prompt_ids.cuda()
        v_pkv = DynamicCache()
        with torch.cuda.amp.autocast(dtype=torch.float16):
            v_out = verifier.get_decoder()(prompt_gpu, past_key_values=v_pkv, use_cache=True)
        v_pkv = v_out.past_key_values

        verify_input = torch.randint(100, 5000, (1, K + 1)).cuda()
        times_v = []
        for trial in range(5 + 20):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.cuda.amp.autocast(dtype=torch.float16):
                v_out = verifier.get_decoder()(
                    verify_input, past_key_values=v_pkv, use_cache=True,
                )
                _ = verifier.lm_head(v_out.last_hidden_state)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            if trial >= 5:
                times_v.append((t1 - t0) * 1000)
            v_pkv = v_out.past_key_values  # keep growing for realistic KV size

        avg_verify = sum(times_v) / len(times_v)

        overlap = "YES ✓" if avg_draft <= avg_verify else "NO (draft slower)"
        results[K] = {
            "cpu_draft_ms": round(avg_draft, 2),
            "gpu_verify_ms": round(avg_verify, 2),
            "draft_hidden_by_verify": avg_draft <= avg_verify,
        }
        print(f"  K={K:2d}: CPU draft={avg_draft:6.1f}ms, GPU verify={avg_verify:6.1f}ms  → overlap: {overlap}")

    return results


def main():
    tok, hf_drafter, cpu_model, verifier = load_models()

    all_results = {}

    # Benchmark 1: Activation Replay vs Re-prefill
    seq_lengths = [32, 64, 128, 256, 512]
    all_results["replay_vs_reprefill"] = benchmark_replay_vs_reprefill(
        hf_drafter, cpu_model, seq_lengths,
    )

    # Benchmark 2: Single-Step Latency
    all_results["single_step"] = benchmark_single_step(hf_drafter, cpu_model)

    # Benchmark 3: CPU-Offloaded Spec Dec
    all_results["cpu_offloaded_spec_dec"] = benchmark_cpu_offloaded_spec_dec(
        cpu_model, verifier, tok,
    )

    # Benchmark 4: Pipeline Overlap
    all_results["overlap_potential"] = benchmark_overlap_potential(
        cpu_model, verifier, tok,
    )

    # Save results
    out_file = "spec_mamba/benchmark_cpu_offload_results.json"
    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nAll results saved to {out_file}")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Replay summary
    print("\nActivation Replay Speedup (vs HF re-prefill):")
    for sl, r in all_results["replay_vs_reprefill"].items():
        print(f"  seq_len={sl:5d}: replay={r['replay_ms']:.1f}ms vs HF={r['hf_reprefill_ms']:.1f}ms "
              f"→ {r['speedup_vs_hf_reprefill']:.1f}x")

    # Single step
    ss = all_results["single_step"]
    print(f"\nSingle-step: CPU-optimized={ss['cpu_optimized_ms']:.2f}ms vs HF={ss['hf_cpu_ms']:.2f}ms "
          f"→ {ss['speedup']:.2f}x")

    # CPU offloaded
    co = all_results["cpu_offloaded_spec_dec"]
    print(f"\nCPU-offloaded round: draft={co['draft_cpu_ms']:.1f}ms + "
          f"verify={co['verify_gpu_ms']:.1f}ms = {co['total_round_ms']:.1f}ms total")

    # Overlap
    print("\nPipeline overlap feasibility:")
    for k, r in all_results["overlap_potential"].items():
        status = "✓ free draft" if r["draft_hidden_by_verify"] else "✗ draft bottleneck"
        print(f"  K={k}: CPU draft={r['cpu_draft_ms']:.1f}ms, GPU verify={r['gpu_verify_ms']:.1f}ms → {status}")


if __name__ == "__main__":
    main()
