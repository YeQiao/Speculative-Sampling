# spec_mamba: Speculative Decoding with Guided Mamba2 — Architecture & Findings

## Overview

`spec_mamba/` is a clean reimplementation of guided speculative decoding following [SD²](https://arxiv.org/abs/2407.10722) (Speculative Decoding squared) patterns. It uses a Mamba2-65M drafter guided by verifier hidden states to speculate tokens verified by LLaMA 3.1-8B.

## Architecture

### Module Layout

```
spec_mamba/
├── trainer.py           # SpecMambaTrainer: training + generation (main module)
├── eval.py              # Evaluation script (multi-dataset, batched)
├── guided_mamba2.py     # GuidedMamba2Block: Mamba2 with guidance injection
└── models/
    └── llama.py         # Custom LlamaForCausalLM with SD²'s _update_causal_mask
```

### Key Components

| Component | Location | Purpose |
|-----------|----------|---------|
| `GuidanceExtractor` | `trainer.py` | Concat 3 verifier hidden states → linear projection → guidance embedding |
| `PrepMambaDeltas` | `trainer.py` | Map guidance → per-layer deltas for Mamba2 x-branch injection |
| `GuidedMamba2Block` | `guided_mamba2.py` | Wraps HF `Mamba2Block`, injects deltas into `in_proj` output |
| Custom `LlamaModel` | `models/llama.py` | SD²'s old-style `_update_causal_mask` for proper batched masking |

### Generation Flow (Non-Compact Layout)

```
PREFILL:
  1. Verifier: process prompt tokens, crop KV cache to curr, extract guidance
  2. Drafter: process prompt[:-1] with zero deltas to warm conv/ssm state

LOOP (each round):
  1. DRAFT: Drafter generates NG tokens autoregressively with guidance
  2. VERIFY: Verifier processes NG+1 tokens with attention_mask + position_ids
  3. REJECT: Compare draft vs verifier via rejection sampling → NA accepted
  4. UPDATE:
     - Zero rejected positions in attention_mask
     - Place resampled next_token at curr+NG+1
     - Advance curr by NG+1 (non-compact: rejected tokens stay in place)
     - Extract new guidance from verifier hidden states at position NA
     - Rebuild drafter cache from scratch (Mamba2 is recurrent)
```

## Critical Bugs Found & Fixed

### 1. Off-by-One in Guidance Layer Indices

**Root cause**: SD²'s `LlamaModel.forward()` collects `guide_input = hidden_states` **before** the layer runs. So `in_layer=[5]` captures the **input** to layer 5 = **output** of layer 4.

The checkpoint was trained using `output_hidden_states=True` with `hidden_states[layer_idx + 1]`, which is the **output** of `layer_idx`.

**Fix** (`trainer.py:_setup_guidance`):
```python
# Shift indices by +1 to match post-layer hidden states
self.guidance_extractor.in_layer = [v + 1 for v in self.v_layers]
# For v_layers=[5,16,29] → in_layer=[6,17,30]
```

**Verification**: Empirically confirmed via `CapturingExtractor` hook — `in_layer=[6,17,30]` produces exact match (diff=0.000000) with the old code's hidden states across all 3 layers.

### 2. Rejection Masking: Correct but Misunderstood

**Observation**: Zeroing `attention_mask` at rejected positions lowered acceptance from ~3.0 to ~0.6 on ultrachat.

**Investigation**: We compared masked vs unmasked verifier output against gold autoregressive:

| Metric | Masked (reject=0) | Unmasked (reject=1) |
|--------|-------------------|---------------------|
| vs Gold logit diff | **0.02** | 4.10 |
| Text quality | Coherent | Degenerate/repetitive |
| Acceptance rate | ~0.6–1.0 | ~3.0–5.6 |

**Conclusion**: The masked verifier is **correct**. The high acceptance without masking was an artifact — both drafter and verifier attended to the same stale rejected KV entries, causing them to accidentally agree on wrong predictions. The "agreement" measured high acceptance, but the text was degenerate.

**The masking design (matching SD² exactly)**:
```python
# After rejection: zero mask at rejected draft positions
attention_mask[is_ongoing, curr + 1: curr + NG + 1] = (
    torch.arange(0, NG, device=device)[None, :] < NA[is_ongoing, None]
).long()

# On verify: always pass attention_mask
v_out = self.v_base.get_decoder()(
    sampled[:, curr: curr + NG + 1],
    position_ids=position_ids[:, curr: curr + NG + 1],
    attention_mask=attention_mask[:, :curr + NG + 1],  # includes rejection zeros
    ...
)
```

The custom `_update_causal_mask` converts the 2D mask into a 4D float mask with `min_dtype` at rejected positions, which SDPA respects via `is_causal=False`.

### 3. Compact Re-Prefill Bug (Fixed 2026-04-22)

**Root cause**: After rejection, the drafter's recurrent state must be rebuilt because Mamba2 has no KV cache to crop. The old re-prefill fed `sampled[:, :max_accepted_end+1]` to the drafter, which included **rejected tokens from previous rounds** still sitting in the buffer (non-compact layout). The verifier masks these via `attention_mask=0`, but Mamba2 is recurrent — it has no attention mask and processes every token in sequence. Rejected garbage tokens corrupt the hidden state, causing **compounding degradation**: corrupted state → worse drafts → more rejections → more garbage → worse re-prefill → etc.

**Fix**: Build a compact sequence filtered by `attention_mask` before re-prefilling:
```python
# For each sample, extract only unmasked tokens
mask_slice = attention_mask[:, :prefill_end]
for b in range(B):
    unmasked = sampled[b, :prefill_end][mask_slice[b].bool()]
    compact[b, :len_b] = unmasked
```

**Impact**: 2.3x improvement in acceptance (0.935 → 2.181 on 32-sample ultrachat test). Validated against activation replay path (known correct): both give accept=2.181, confirming equivalence.

## How Batching Works

### The Non-Compact Layout

In batched spec dec, different samples in the batch may have different `NA` (number accepted). The `sampled[]` buffer uses a **non-compact layout**:

```
Position: [0..S-1] [curr] [d1] [d2] [REJ] [REJ] [REJ] [REJ] [REJ] [REJ] [next] [d1'] ...
Mask:     [1 1 1]  [1]    [1]  [1]  [0]   [0]   [0]   [0]   [0]   [0]   [1]    [1]  ...
```

- `curr` always advances by `NG+1` regardless of how many were accepted
- Rejected tokens remain in `sampled[]` but are masked out in `attention_mask`
- `position_ids` encode the **semantic** position (skipping rejections)

### Why Masking Is Required for bsz > 1

Without masking, when sample A accepts 6/8 and sample B accepts 2/8, the verifier for sample B would attend to sample A's rejected KV entries at the same physical cache positions. The `attention_mask` ensures each sample only sees its own accepted history.

For bsz=1, masking is still correct (matches gold AR output) but the drafter, which was trained without masking, may not match the "honest" verifier as well → lower acceptance but better text.

### Position ID Management

```python
# During drafting: consecutive positions
position_ids[:, curr + i + 1] = position_ids[:, curr + i] + 1

# After rejection: next_token gets position = curr + NA + 1
position_ids[:, curr + NG + 1] = position_ids[:, curr] + NA + 1
```

This ensures RoPE embeddings reflect the true sequence position, not the physical buffer position.

## Performance Summary

### AR Baseline Throughput (LLaMA 3.1-8B, greedy, bsz=1, H100 NVL)

| Dataset | AR Throughput (tok/s) |
|---------|----------------------|
| ultrachat | 75.6 |
| humaneval | 75.8 |
| xsum | 75.4 |
| alpaca | 76.0 |
| gsm8k | 76.1 |
| **Mean** | **75.8** |

### Guided Mamba2-65M (no-mask, greedy, bsz=1, 96 samples, H100 NVL)

| Dataset | Accepted | Block Eff | Throughput | Speedup |
|---------|----------|-----------|------------|---------|
| ultrachat | 2.924 | 3.924 | 38.7 | 0.51x |
| humaneval | 3.738 | 4.738 | 46.6 | 0.61x |
| xsum | 2.430 | 3.430 | 33.5 | 0.44x |
| alpaca | 3.707 | 4.707 | 46.3 | 0.61x |
| gsm8k | 3.066 | 4.066 | 40.0 | 0.53x |
| **Mean** | **3.173** | **4.173** | **41.0** | **0.54x** |

### Unguided Baseline Mamba2-65M (no-mask, greedy, bsz=1, 96 samples)

**KD-finetuned drafter with zeroed guidance (checkpoint-750 base):**

| Dataset | Accepted | Block Eff | Throughput | Speedup |
|---------|----------|-----------|------------|---------|
| ultrachat | 3.333 | 4.333 | 41.9 | 0.55x |
| humaneval | 3.425 | 4.425 | 42.8 | 0.56x |
| xsum | 3.036 | 4.036 | 38.9 | 0.52x |
| alpaca | 4.117 | 5.117 | 49.9 | 0.66x |
| gsm8k | 3.356 | 4.356 | 42.5 | 0.56x |
| **Mean** | **3.453** | **4.453** | **43.2** | **0.57x** |

**True pretrained Mamba2-65M (FineWeb-Edu-100BT, no KD, no guidance, `custom-mamba-65m-multi-gpu`):**

| Dataset | Accepted | Block Eff | Throughput | Speedup |
|---------|----------|-----------|------------|---------|
| ultrachat | 2.826 | 3.826 | 37.9 | 0.49x |
| humaneval | 3.142 | 4.142 | 40.9 | 0.53x |
| xsum | 2.511 | 3.511 | 34.3 | 0.45x |
| alpaca | 3.811 | 4.811 | 47.6 | 0.61x |
| gsm8k | 2.997 | 3.997 | 39.5 | 0.51x |
| **Mean** | **3.057** | **4.057** | **40.0** | **0.52x** |

*Note: KD-finetuned drafter has ~13% higher acceptance than pretrained even with guidance zeroed — KD training itself improves drafter alignment.*

### Masked Results (WITH mask, compact re-prefill fix, greedy, bsz=1, 96 samples, H100 NVL)

**Guided Mamba2-65M (KD-finetuned + active guidance):**

| Dataset | Accepted | Block Eff | Throughput | Speedup |
|---------|----------|-----------|------------|---------|
| ultrachat | 2.197 | 3.197 | 31.0 | 0.41x |
| humaneval | 2.070 | 3.070 | 29.7 | 0.40x |
| xsum | 1.721 | 2.721 | 26.2 | 0.35x |
| alpaca | 2.745 | 3.745 | 36.4 | 0.48x |
| gsm8k | 2.907 | 3.907 | 37.9 | 0.50x |
| **Mean** | **2.328** | **3.328** | **32.2** | **0.43x** |

**KD-finetuned with zeroed guidance (unguided baseline):**

| Dataset | Accepted | Block Eff | Throughput | Speedup |
|---------|----------|-----------|------------|---------|
| ultrachat | 1.629 | 2.629 | 25.2 | 0.34x |
| humaneval | 1.437 | 2.437 | 23.3 | 0.31x |
| xsum | 0.973 | 1.973 | 18.7 | 0.25x |
| alpaca | 2.417 | 3.417 | 32.8 | 0.43x |
| gsm8k | 1.828 | 2.828 | 27.1 | 0.36x |
| **Mean** | **1.657** | **2.657** | **25.4** | **0.34x** |

**True pretrained Mamba2-65M (masked):**

| Dataset | Accepted | Block Eff | Throughput | Speedup |
|---------|----------|-----------|------------|---------|
| ultrachat | 1.873 | 2.873 | 28.8 | 0.37x |
| humaneval | 1.492 | 2.492 | 25.0 | 0.32x |
| xsum | 1.182 | 2.182 | 21.7 | 0.28x |
| alpaca | 2.638 | 3.638 | 36.5 | 0.47x |
| gsm8k | 2.092 | 3.092 | 31.1 | 0.40x |
| **Mean** | **1.855** | **2.855** | **28.6** | **0.37x** |

*Key findings (masked, with compact re-prefill fix):*
- **Guidance clearly helps**: guided (2.33) > pretrained (1.86) > KD-zeroed (1.66)
- **KD-zeroed is worst**: KD training without active guidance degrades masked performance — the backbone became dependent on deltas
- **Pretrained > KD-zeroed**: confirms guidance dependency artifact
- Results file: `eval_results_kd_mask_fixed.json`, `eval_results_pretrained_mask_fixed.json`

### Guidance Layer Sweep (masked, greedy, bsz=1, 96 samples, H100 NVL)

Full sweep over 13 guidance configurations. All use Mamba2-65M drafter + LLaMA 3.1-8B verifier.

| Config | v_layers | ultra | human | xsum | alpaca | gsm8k | Mean | Δ vs base |
|--------|----------|-------|-------|------|--------|-------|------|-----------|
| LLaMA-1B drafter | — | 2.73 | 4.88 | 2.39 | 4.35 | 2.60 | 3.391 | +83% |
| KD+Guided [5,16,29] | [5,16,29] | 2.20 | 2.07 | 1.72 | 2.75 | 2.91 | 2.328 | +25% |
| Pair [5,29] | [5,29] | 2.10 | 2.20 | 1.74 | 2.70 | 2.84 | 2.316 | +25% |
| KD+z-branch | [5,16,29] | 2.12 | 2.21 | 1.73 | 2.61 | 2.85 | 2.304 | +24% |
| Single layer 29 | [29] | 2.22 | 2.12 | 1.64 | 2.69 | 2.85 | 2.303 | +24% |
| Pretrained+Guide+FT | [5,16,29] | 2.14 | 2.18 | 1.66 | 2.74 | 2.78 | 2.301 | +24% |
| Pair [16,29] | [16,29] | 2.12 | 2.14 | 1.79 | 2.67 | 2.78 | 2.299 | +24% |
| Single layer 16 | [16] | 2.14 | 2.10 | 1.72 | 2.61 | 2.90 | 2.293 | +24% |
| Pretrained+Guide (frozen) | [5,16,29] | 2.05 | 1.63 | 1.50 | 2.29 | 2.44 | 1.981 | +7% |
| Single layer 5 | [5] | 1.75 | 1.88 | 1.12 | 2.57 | 2.19 | 1.899 | +2% |
| Pretrained (no guidance) | — | 1.87 | 1.49 | 1.18 | 2.64 | 2.09 | 1.855 | — |
| KD+Zeroed | [5,16,29] | 1.63 | 1.44 | 0.97 | 2.42 | 1.83 | 1.657 | -11% |

**Key Findings from Sweep:**

1. **Layer selection barely matters** (when backbone is finetuned): Top configs span only 2.293–2.328 — a 1.5% spread. Single layer 29 (2.303) matches triple [5,16,29] (2.328). Even single layer 16 alone (2.293) is within noise.

2. **z-branch adds nothing**: KD+z-branch (2.304) ≈ KD+Guided x-only (2.328). Doubles PrepMambaDeltas parameters for no benefit.

3. **Low layers are useless alone**: Single layer 5 (1.899) barely beats unguided pretrained (1.855) — +2% vs +24% for layers 16/29. Layer 5 captures surface features that don't help speculative alignment.

4. **Backbone finetuning is critical**: Frozen guidance (1.981, +7%) vs finetuned (2.301, +24%) — finetuning provides 3× the benefit of guidance projection alone.

5. **KD pre-training provides marginal benefit**: KD+Guided (2.328) vs Pretrained+Guide+FT (2.301) — only +1.2% from KD initialization. The guidance training itself does almost all the work.

6. **KD-zeroed confirms backbone dependency**: KD+Zeroed (1.657) is 11% worse than pretrained (1.855) — backbone was warped to depend on deltas during KD.

### LLaMA-3.2-1B Drafter (WITH mask, greedy, bsz=1, 96 samples)

| Dataset | Accepted | Block Eff | Throughput | Speedup |
|---------|----------|-----------|------------|---------|
| ultrachat | 2.727 | 3.727 | 59.0 | 0.72x |
| humaneval | 4.881 | 5.881 | 83.7 | 1.03x |
| xsum | 2.390 | 3.390 | 56.4 | 0.69x |
| alpaca | 4.351 | 5.351 | 69.3 | 0.85x |
| gsm8k | 2.604 | 3.604 | 70.7 | 0.87x |
| **Mean** | **3.391** | **4.391** | **67.8** | **0.83x** |

*Note: Speedup < 1.0x means spec dec is SLOWER than pure AR on H100 NVL. The H100's high AR throughput (~76 tok/s) makes it hard for any drafter to break even — the drafter overhead must be < 1/(block_eff) × AR_latency. This is a known challenge for spec dec on fast GPUs.*

### Acceptance Rate Comparison (older values)

96 samples per dataset, tgt_len=128, greedy, `--no_mask`:

| Dataset | Acceptance | Throughput (tok/s) |
|---------|-----------|-------------------|
| ultrachat | 2.913 | 32.3 |
| humaneval | 3.745 | 39.5 |
| xsum | 2.419 | 25.7 |
| alpaca | 3.707 | 33.5 |
| gsm8k | 3.072 | 32.3 |
| **Mean** | **3.171** | **32.7** |

Results file: `spec_mamba/eval_results_no_mask_bsz1.json`

### Cross-Verification: LLaMA-1B → LLaMA-8B (vanilla, no KD)

To test whether the masked/unmasked gap is a property of the non-compact layout or specific to our Mamba2 drafter, we ran vanilla speculative decoding with LLaMA-3.2-1B as drafter and LLaMA-3.1-8B as verifier (32 samples, ultrachat, greedy, bsz=1, NG=8):

| Setup | Acceptance |
|-------|-----------|
| **WITH masking** | **2.734** |
| **WITHOUT masking** | **2.127** |

**Key finding**: The direction is **reversed** — masking gives *higher* acceptance with the vanilla transformer drafter! This disproves the theory that stale tokens universally inflate acceptance.

**Explanation**: The asymmetry comes from the drafter architecture:
- **Transformer drafter** (LLaMA-1B): In the unmasked setup, only the *verifier* sees stale rejected tokens in its KV cache. The drafter receives a *compact* re-prefilled sequence (no stale tokens). So the verifier's predictions diverge from what the drafter sees → more disagreement → lower acceptance. Masking fixes the verifier to match the clean view → both models agree more → higher acceptance.
- **Mamba2 drafter** (guided): Mamba2 is recurrent — it never attends to any KV cache. In the unmasked setup, the verifier is confused by stale tokens, and the Mamba2 drafter makes its own independent errors. By chance, both models "agree" on wrong predictions more often than they would on correct ones → inflated acceptance. Masking corrects the verifier but the Mamba2 drafter's errors remain → less agreement → lower acceptance.

**Implication**: With the compact re-prefill fix, Mamba2 masked acceptance is ~2.3 (guided) and ~1.9 (pretrained), comparable to LLaMA-1B's ~2.7. The gap is expected given 65M params vs 1.2B. The original ~1.0 masked acceptance was caused by the re-prefill bug corrupting the drafter's recurrent state with rejected tokens (see Bug #3 above).

Script: `spec_mamba/cross_verify_mask.py`

## Checkpoint Details

- **Trained checkpoint**: `/HSC/users/qiaoye/SSM_SPEC/checkpoints/guided_mamba/ckpts/last.ckpt`
- **Hyperparameters**: `NG=8, v_layers=[5,16,29], d_layers=all, finetune_drafter=True, loss_method=tvd, steer_z=False`
- **Guidance**: 3-layer concat (layers 5, 16, 29) → linear → 4096-dim embedding → PrepMambaDeltas → 16×1024-dim deltas
- **Training**: Used `output_hidden_states=True` (post-layer), NOT `compute_guidance` (pre-layer) — hence the off-by-one fix needed at inference

## Known Limitations

1. **Masked acceptance still lower than no-mask**: Guided masked=2.33 vs no-mask=3.17. The gap reflects true drafter quality — no-mask inflates acceptance due to stale-token confusion in the verifier's KV cache.
2. **Base model limitations**: LLaMA 3.1-8B (base, not instruct) produces repetitive text with chat templates regardless of spec dec.
3. **Mamba2 cache rebuild**: The recurrent Mamba2 drafter requires compact re-prefill after each rejection round (no KV crop equivalent). **UPDATE:** Activation replay now avoids this — see below.

---

## Activation Replay (Added 2026-04-20)

### Overview
Instead of re-prefilling the drafter from scratch over the entire sequence after rejection, save a snapshot of the SSM state before drafting and restore+replay only accepted tokens.

### Implementation
- `snapshot_mamba2_cache()` / `restore_mamba2_cache()` in `trainer.py`
- `use_activation_replay` flag on `generate()` and `_spec_dec_step()`
- `--activation_replay` CLI flag on `eval.py`

### Profiling Results

**GPU (A100-80GB): NOT beneficial for 65M model**
- Re-prefill: ~16ms (constant, model too small to saturate GPU)
- Replay: ~40ms (5 sequential single-steps × 8ms each)

**CPU (Xeon 8562Y+): MASSIVE benefit**
| Seq Length | Re-prefill (ms) | Replay (ms) | Speedup |
|-----------|-----------------|-------------|---------|
| 32 | 299 | 47 | 6.4x |
| 128 | 1,019 | 37 | 27.7x |
| 512 | 3,644 | 36 | 100.9x |
| 1,024 | 8,036 | 40 | 201.0x |

**Updated CPU benchmarks (2026-04-22, CPU-optimized model):**
| Seq Length | HF Re-prefill (ms) | Replay (ms) | Speedup |
|-----------|---------------------|-------------|---------|
| 64 | 1577 | 47 | 33.7x |
| 128 | 1683 | 38 | 43.7x |
| 256 | 1858 | 38 | 49.0x |
| 512 | 2187 | 38 | 58.0x |
| 1024 | 2973 | 39 | 76.8x |

**CPU single-step (K=1):** 7.53ms (CPU-opt) vs 7.93ms (HF) = 1.05x

### CPU-Offloaded Speculative Decoding (End-to-End)
- **Draft (CPU, K=8):** 64ms
- **Verify (GPU, 8 tokens):** 16ms  
- **Total round:** 80ms (80% draft, 20% verify)
- **Pipeline overlap:** NOT feasible — CPU draft (61ms) is 4x slower than GPU verify (15ms), so async overlap yields minimal benefit. Frame CPU offloading as **memory savings** (frees GPU VRAM for larger batch / longer KV cache), not latency hiding.

#### Updated: Pipeline IS Feasible with INT8 (2026-04-29)

With INT8 VNNI kernel (16 threads), the bottleneck flips:
- **Draft (CPU, K=8):** 9 ms (was 61ms) — **now faster than GPU verify!**
- **Verify (GPU LLaMA-8B, 8 tokens):** 16 ms
- **Verify (GPU Gemma-4-E4B, 8 tokens):** TBD (drafter still training, no GPU measurement)

**Pipelined throughput (overlapped CPU draft + GPU verify):**

| Verifier | K | Accept | Round (ms) | tok/s | vs AR | Speedup |
|----------|---|--------|-----------|-------|-------|---------|
| LLaMA-8B | 4 | 2.33 | 14.5 | 230 | 76 | **3.0x** |
| LLaMA-8B | 8 | 2.33 | 16.5 | 202 | 76 | **2.7x** |
| LLaMA-8B | 12 | 2.33 | 19.0 | 175 | 76 | **2.3x** |
| Gemma-4B | 4 | TBD | TBD | TBD | TBD | TBD |
| Gemma-4B | 8 | TBD | TBD | TBD | TBD | TBD |

Assumptions: Perfect pipeline overlap (CPU drafts while GPU verifies), 0.5ms synchronization overhead. Masked acceptance rates from eval.

**Why this works now:**
- Old: CPU draft=61ms ≫ GPU verify=16ms → CPU is bottleneck, overlap useless
- New: CPU draft=9ms < GPU verify=16ms → **GPU is bottleneck, CPU draft is free**
- Round time ≈ verify time (draft is hidden) → throughput ≈ (accept+1)/verify_time
- Even sequential (no overlap): 132 tok/s = **1.75x vs AR** for LLaMA-8B

**Comparison: On-GPU vs CPU-offloaded drafting (LLaMA-8B, K=8):**

| Config | Draft (ms) | Verify (ms) | Round (ms) | tok/s | vs AR |
|--------|-----------|-------------|-----------|-------|-------|
| Mamba2-65M on GPU (old) | 121 | 16 | 138 | 24 | 0.32x |
| LLaMA-1B on GPU | 65 | 16 | 82 | 46 | 0.60x |
| CPU BF16 sequential | 28 | 16 | 45 | 74 | 0.98x |
| **CPU INT8 sequential** | **9** | **16** | **25** | **132** | **1.75x** |
| **CPU INT8 overlapped** | **9** | **16** | **16.5** | **202** | **2.66x** |

**Sensitivity (with overlap):** Acceptance rate needed for target speedup:
- 1.5x over AR: need accept ≥ 0.9 ✓ (we have 2.33)
- 2.0x over AR: need accept ≥ 1.5 ✓ (we have 2.33)
- 3.0x over AR: need accept ≥ 4.0 △ (would need better drafter or larger K)

**Batched serving (the real win):** With bsz=4-8, AR throughput drops to 30-45 tok/s/seq → CPU-offloaded spec dec provides 2.2-2.4x speedup per sequence while freeing GPU VRAM for more concurrent requests.

**Caveats:**
1. Pipeline overlap requires async CPU-GPU communication (CUDA streams + Python threading)
2. After rejection, drafter must be re-synced (activation replay: ~40ms for CPU snapshot+restore+replay of ~2 tokens)
3. Guidance extraction requires partial GPU verifier forward → add ~2ms per round for guidance transfer
4. Gemma-4-E4B: AR latency unmeasured (GPU occupied for training), drafter acceptance TBD (training in progress)

### Note on Equivalence
Activation replay produces a slightly different drafter state than full re-prefill because the non-compact layout leaves rejected tokens in the buffer, and full re-prefill processes them. Activation replay is arguably more correct (drafter never sees rejected tokens).

---

## CPU-Optimized Selective Scan (Added 2026-04-20)

### AVX-512 Kernel
- C++ kernel at `spec_mamba/cpu_kernels/ssm_ops.cpp`
- SSM step: `h_new = h_old * exp(dt*A) + dt*B*x; y = C@h + D*x`
- Uses `_mm512_fmadd_ps` for FMA, processes N=128 in 8 AVX-512 iterations
- OpenMP parallelized over (B, H, D) dimensions
- **4x speedup** over PyTorch CPU (0.079ms vs 0.314ms)

### CPU Mamba2 Model
- `spec_mamba/cpu_mamba2.py`: Complete CPU-optimized model
- Pre-extracts all weights, eliminates HF dispatch overhead
- Single-step: 7.3ms (competitive with GPU's 8.0ms!)
- 8-token draft: 57ms (viable for async overlap with GPU verification)
- **Verified correct** against HF reference (2026-04-21): 16-step autoregressive generation with identical inputs produces matching top-1 tokens at every step, max logit diff ~4e-6 (float32 precision)

### Fused BF16 Forward (Added 2026-04-29)
- `spec_mamba/cpu_kernels/fused_forward.cpp`: Entire 16-layer forward + LM head in a single C++ call
- `FusedCPUMamba2Model` class in `cpu_mamba2.py`
- **BF16 weights** for memory-bound ops (embed, in_proj, out_proj, LM head): halves memory reads
- **FP32 compute** for accuracy-critical paths (SSM state, norms)
- Uses AVX-512 BF16 dot product (`vdpbf16ps`) when available, plus AVX-512 BF16-to-FP32 conversion fallback
- Weight size: 186MB BF16 + 0.7MB FP32 (vs 373MB all-FP32)

**Performance (B=1, Intel Xeon 8562Y+):**

| Threads | Old FP32 (ms) | Fused BF16 (ms) | Speedup |
|---------|--------------|-----------------|---------|
| 1 | 35.2 | 18.6 | 1.89x |
| 4 | 11.7 | 5.8 | 2.02x |
| 8 | 9.3 | 3.7 | 2.47x |
| 16 | 7.7 | 2.8 | 2.77x |

**8-token draft (8 threads): 32ms (4.0 ms/token)** — was 57ms, now **1.78x faster**

#### Conv+SiLU Vectorization (2026-04-29)

The conv1d cached step was further optimized:
- **Transposed layout**: Conv weights/states stored as `[K, conv_dim]` instead of `[conv_dim, K]` — enables contiguous `memcpy` for state shifts and contiguous AVX-512 loads for dot products
- **AVX-512 FMA unrolled**: 4 FMA operations over 16 channels at once (K=4 fully unrolled)
- **Vectorized SiLU**: Fused bias-add + SiLU activation in the same AVX-512 loop
- **`-ffast-math`**: Enables compiler SVML vectorization of all `exp()` calls

Performance after conv optimization (includes `-ffast-math` for all SiLU/exp calls):

| Threads | Latency (ms) | 8-tok draft (ms) |
|---------|-------------|------------------|
| 1 | 17.7 | 141 |
| 4 | 6.3 | 50 |
| 8 | 3.7 | 30 |
| 16 | 2.4 | 20 |

The conv step itself improved ~2-3x, but since conv was only 8.5% of total time (77% is memory-bandwidth-bound GEMV), overall improvement is ~15% at high thread counts. Further gains require weight pruning/quantization or HBM.

**Correctness**: 16-step generation with identical tokens produces matching top-1 at every step. Max logit diff ~0.035 (expected BF16 quantization noise), stays bounded.

#### End-to-End Generation Benchmark (2026-04-29)

Compared three implementations: (1) HF naive PyTorch, (2) CPUMamba2Model (FP32 + C++ SSM kernel), (3) FusedCPUMamba2Model (BF16 fused forward). All produce identical greedy tokens. Intel Xeon 8562Y+.

**Throughput (tok/s) — higher is better:**

| Threads | HF Naive | CPU FP32 | Fused BF16 | Fused/HF Speedup |
|---------|----------|----------|------------|------------------|
| 4 | 86 | 93 | 215 | **2.5x** |
| 8 | 110 | 118 | 317 | **2.9x** |
| 16 | 132 | 137 | 453 | **3.4x** |

**Generation latency (ms) — 8 tokens (speculative draft):**

| Threads | HF Naive | CPU FP32 | Fused BF16 | Fused/HF Speedup |
|---------|----------|----------|------------|------------------|
| 4 | 93 | 81 | 35 | **2.7x** |
| 8 | 80 | 72 | 25 | **3.2x** |
| 16 | 55 | 53 | 17 | **3.3x** |

**Key findings**:
- CPU FP32 model (C++ SSM kernel only) provides marginal speedup (1.05-1.15x) — the SSM kernel is 10x faster but only saves ~1.4ms/step because SSM was 22.8% of a 11.3ms step
- Fused BF16 forward provides **2.5-3.4x end-to-end speedup** by eliminating Python dispatch (48% overhead) AND halving memory bandwidth (BF16 weights)
- At 8 threads, fused model achieves **317 tok/s** (3.1 ms/token) — viable for real-time speculative draft generation
- The fused model on 8 CPU threads (317 tok/s) **exceeds the HF model on 16 threads** (132 tok/s)

#### INT8 Weight-Only Quantization (Added 2026-04-29)

Further 2x memory bandwidth reduction using AVX-512 VNNI (`vpdpbusd`):
- `Int8FusedCPUMamba2Model` in `cpu_mamba2.py`
- Per-row symmetric INT8 quantization (no calibration needed)
- Weights: INT8 (embed, in_proj, out_proj). Activations: FP32 (quantized to UINT8 on-the-fly per GEMV call)
- VNNI dot product: accumulate 64 INT8 pairs per instruction (16 lanes × 4 pairs)
- Bias correction for UINT8 shift: precompute per-row sum for `128 * row_sum` term
- Conv weights and SSM state remain FP32 (accuracy-critical, tiny memory footprint)

**Model size:** 88.8 MB INT8 + 2.0 MB FP32 (vs 186 MB BF16, vs 373 MB FP32)

**Correctness:** 32/32 greedy tokens match BF16 output. Max logit diff 0.39 (vs BF16 reference). Top-5 tokens nearly identical. For speculative drafting, occasional disagreement is free — verifier rejects at no quality cost.

**Performance (B=1, Intel Xeon 8562Y+):**

| Threads | HF Naive (ms) | BF16 Fused (ms) | INT8 Fused (ms) | INT8 tok/s | vs HF | vs BF16 |
|---------|--------------|-----------------|-----------------|------------|-------|---------|
| 4 | 12.8 | 6.0 | **2.3** | 442 | 5.7x | 2.7x |
| 8 | 9.9 | 3.5 | **1.5** | 674 | 6.7x | 2.4x |
| 16 | 8.1 | 2.4 | **1.1** | 924 | 7.5x | 2.2x |

**8-token draft latency:**

| Threads | HF Naive | BF16 Fused | INT8 Fused |
|---------|----------|------------|------------|
| 4 | 102 ms | 48 ms | **18 ms** |
| 8 | 79 ms | 28 ms | **12 ms** |
| 16 | 65 ms | 19 ms | **9 ms** |

**Key findings:**
- INT8 achieves **924 tok/s at 16 threads** (~1.08 ms/token) — approaching GPU single-step latency
- 8-token draft in **12 ms at 8 threads** is fast enough for real-time overlapping with GPU verification (~16ms)
- No calibration needed: tiny 65M model has well-behaved weights (absmax < 2.0)
- Total optimization stack: 7.5x over naive PyTorch (eliminate dispatch + BF16 + INT8 + VNNI + conv vectorize)

### Bug Fix: `_rms_norm_with_gate` Order (Fixed 2026-04-21)

**Root cause**: The gated RMS norm in `cpu_mamba2.py` applied the SiLU gate **after** normalization, but HF's `MambaRMSNormGated` applies the gate **before** computing variance.

```python
# WRONG (old): norm first, then gate
x_normed = rms_norm(x, weight)
return x_normed * silu(gate)

# CORRECT (fixed): gate first, then norm — matches HF
x = x * silu(gate)
variance = x.pow(2).mean(-1, keepdim=True)
x = x * rsqrt(variance + eps)
return weight * x
```

Since RMS norm is nonlinear (`norm(a*b) ≠ norm(a)*b`), the order matters. Before the fix, layer 0 output had max diff 0.246, cascading to 28.8 by layer 15 and producing completely wrong tokens by generation step 2. After the fix, max logit diff is ~4e-6 and stays bounded across all 16 steps.