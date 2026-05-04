yesterday(4/30/2026) — DONE
1. ✅ Prepared mixed dataset (UltraChat+HumanEval+GSM8K+Alpaca+XSum) for Gemma
2. ✅ Created config files for pretrain→guided (LLaMA & Gemma) and guided-mixed (Gemma)
3. ✅ Reduced epochs from 10→7 for new guided training (no fast CUDA path = slow)
4. ✅ Killed stale training jobs
5. ✅ Fixed checkpoint filename format ({val_loss:.4f})

today(5/1/2026) — PAPER EXPERIMENTS
Goal: Collect ALL data needed for Table 1 + ablation table

=== PHASE 1: TRAINING (parallel, ~5-7h each) ===
GPU 0: LLaMA pretrain→guided (7 epochs)
  config_guide_45m_from_pretrain.yaml
GPU 1: Gemma guided-mixed (7 epochs)
  config_guide_45m_gemma_mixed.yaml

=== PHASE 2: EVAL — ready-now models (while training runs) ===
Each eval: 96 samples, 5 datasets (humaneval,gsm8k,alpaca,ultrachat,xsum), greedy, bsz=1
Use BEST checkpoint (not last!) where applicable.

  Already trained — can eval immediately on whichever GPU has room:
  E1. LLaMA + Pretrain drafter (zeroed guidance)
  E2. LLaMA + KD drafter (zeroed guidance)
  E3. LLaMA + Guided drafter (BEST ckpt: epoch 7, step=37500)
  E4. Gemma + Pretrain drafter (zeroed guidance)
  E5. Gemma + KD drafter (zeroed guidance)
  E6. Gemma + Guided drafter (last=best, epoch 9)

=== PHASE 3: EVAL — after training finishes ===
  E7. LLaMA + Pretrain→Guided drafter (best ckpt from new training)
  E8. Gemma + Guided-Mixed drafter (best ckpt from new training)

=== PAPER TABLE 1 STRUCTURE ===
Rows (LLaMA-8B verifier):
  1. AR baseline (already have)
  2. LLaMA-1B transformer drafter (already have: eval_results_llama1b_mask.json)
  3. Mamba2-45M pretrain only         [E1]
  4. Mamba2-45M KD                    [E2]
  5. Mamba2-45M guided (best)         [E3]
  6. Mamba2-45M pretrain→guided       [E7]  ← ablation: KD matters?

Rows (Gemma-4-E4B verifier):
  7. Mamba2-45M pretrain only         [E4]
  8. Mamba2-45M KD                    [E5]
  9. Mamba2-45M guided (best)         [E6]
  10. Mamba2-45M guided-mixed         [E8]  ← new: diverse training data

Columns: humaneval, gsm8k, alpaca, ultrachat, xsum, mean

=== NOTES ===
- 2 GPUs: H100 NVL 95GB each
- Training uses ~25-30GB, eval uses ~20GB — can coexist on same GPU if careful
- For pretrain/KD baselines, use guided_mamba/eval.py --pretrained_drafter
- For guided models, use guided_mamba/eval.py --ckpt <best>
- 96 samples × 5 datasets × ~6s/sample ≈ 48 min per eval run



todo for 5/1
since we are waiting for the 70b model guide training. Lets start to build overlap pipline that show real word speed up vs auto regression using our highly optimized mamaba forward code.
1. test GPU-CPU pipline to see maximum improvement for llama3 8b and 70b model. Using benchmark data
2. test CPU only running to see how much we can improve if both drafter and verifier are runing on cpu. since cpu is very slow for auto regressive, lets use very few samples (this can start anytime)
3. do not start GPU test until the current 70b model guide training finish
4. document everything
5. when testing using the best checkpoint not last

=== CPU-ONLY RESULTS (5/2/2026) ===
Script: spec_mamba/pipeline_benchmark.py --mode cpu_verify
Config: greedy, NG=8, tgt_len=32, 4 samples, 16 threads, INT8 VNNI drafter

| Metric              | Mamba2-65M (20M backend) | Mamba2-45M (40M backend) |
|---------------------|--------------------------|--------------------------|
| Throughput          | 3.49 tok/s               | 4.32 tok/s               |
| Acceptance          | 0.89 / 8                 | 1.21 / 8                 |
| Tokens/round        | 1.89                     | 2.10                     |
| Draft time/round    | 15.44 ms                 | 16.44 ms                 |
| Verify time/round   | 534 ms                   | 478 ms                   |
| AR baseline         | 4.78 tok/s               | 5.60 tok/s               |
| Speedup vs AR       | 0.73x                    | 0.77x                    |

Key findings:
- 45M drafter: 36% better acceptance, only 6% more draft latency → "Mamba backend is free"
- CPU-only spec dec < 1x for LLaMA-8B (CPU verify cost: 9 tokens ≈ 5× single AR step)
- Draft is 29x cheaper than verify on CPU (16ms vs 478ms)
- The GPU-CPU pipeline (gpu_verify mode) is where the real speedup is
- Results in: outputs/pipeline_benchmark/benchmark_cpu_verify_*.json

TODO: Run gpu_verify mode after 70B training finishes (need GPUs free)