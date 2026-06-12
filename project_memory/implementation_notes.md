# Implementation Notes (spec_mamba)

## Key files
- Full architecture doc: `spec_mamba/ARCHITECTURE.md` (always update with new benchmark results)
- Project instructions: `.github/instructions/specssm-project.instructions.md`
- Impl instructions: `.github/instructions/spec-mamba-impl.instructions.md`
- Cross-verify script: `spec_mamba/cross_verify_mask.py`

## Off-by-one fix (verified)
- `in_layer = [v+1 for v in v_layers]` (verified diff=0.0)

## Rejection masking (verified correct)
- Rejection masking is correct (verified diff=0.02 vs gold) — lower acceptance is expected.
- "High acceptance" (~3.0) WITHOUT masking is a Mamba2-specific artifact
  (asymmetric stale-token confusion).
- Cross-verified with LLaMA-1B→8B vanilla: masked=2.734 > unmasked=2.127 (opposite direction!).
- Mamba2 masked acceptance: ~0.6–1.5 (correct), unmasked: ~3.2 (inflated).
- `--no_mask` flag wiring: `trainer.py:generate(mask_rejected=)` + `eval.py:--no_mask`.
- Latest no-mask results (96 samples, greedy, bsz=1): mean=3.171 across 5 datasets.

## Gemma-4-E4B verifier (2026-05-01)
- Works! Fixes: nested config (`text_config`), `eot_id=106` (`<turn|>`), `V_VOCAB_SIZE`,
  skip chat template override.
- humaneval=2.179 (strong); gsm8k/alpaca=0.58 (weak due to UltraChat-only training +
  single-layer guidance).
- NOT a fundamental limitation — Gemma is multimodal with complex arch (sliding/global attn
  mix, 262K vocab).
- Fix: better guide data (mix math/code/instruction) + multi-layer guidance
  (v_layers=[5,17,35] like LLaMA's [5,16,29]).
- KD stage hurt — suggests KD data distribution mismatch, needs rethinking for Gemma.

## Fused BF16 forward (2026-04-29)
- `FusedCPUMamba2Model` + `cpu_kernels/fused_forward.cpp`.
- 2.13x–2.77x speedup over old Python-dispatch FP32 model.
- 8 threads: 3.7ms/step (was 7.3ms); 8-token draft: 32ms (was 57ms).
- LM head (128256x512 GEMV) was 35% of time — BF16 halves its memory reads.
- Max logit diff ~0.04 from BF16 quantization, top-1 tokens match.
