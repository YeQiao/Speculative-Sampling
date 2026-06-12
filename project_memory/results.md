# Benchmark Results

## Overnight Experiment Results (2026-04-22)

### Critical Finding: Speedup < 1x on H100
- AR baseline: ~76 tok/s (H100 NVL, bsz=1, greedy)
- Mamba2-65M guided: 0.54x speedup (SLOWER than AR)
- Mamba2-65M unguided: 0.57x speedup (also slower, and higher accept than guided!)
- LLaMA-1B drafter: 0.83x speedup (mostly slower; humaneval reaches 1.03x)
- Root cause: H100 AR is very fast; drafter overhead dominates.

### Surprising: Unguided > Guided acceptance (no-mask)
- Unguided mean accept: 3.45 vs Guided: 3.17
- Needs investigation — guidance may hurt in the no-mask regime.

### CPU Benchmarks
- Replay speedup: 33.7x–76.8x vs HF re-prefill
- CPU-offloaded round: 80ms (64ms draft + 16ms verify)
- Pipeline overlap NOT feasible (CPU 4x slower than GPU verify)
- BF16 in_proj: 9.6x faster (AMX); FP16 SSM: <0.3% error

### Result files
- `spec_mamba/eval_results_baseline_with_ar.json`
- `spec_mamba/eval_results_llama1b_mask.json`
- `spec_mamba/benchmark_cpu_offload_results.json`
- `spec_mamba/CPU_KERNEL_OPTIMIZATION.md`
