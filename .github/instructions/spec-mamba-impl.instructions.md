---
description: "Speculative decoding implementation details for spec_mamba module. Use when debugging acceptance rates, modifying the generation loop, changing masking behavior, or working with guidance extraction. Critical for understanding the off-by-one fix and rejection masking design."
applyTo: "spec_mamba/**"
---

# spec_mamba Implementation Rules

## Guidance Layer Index Off-by-One

SD²'s custom `LlamaModel.forward()` collects `guide_input = hidden_states` **BEFORE** the layer processes it. So `in_layer=[i]` captures the **output of layer i-1**, not layer i.

Our checkpoint was trained with `output_hidden_states=True` using `hidden_states[layer_idx + 1]` (= output of layer `layer_idx`).

**Rule**: Always set `in_layer = [v + 1 for v in v_layers]`:
```python
# v_layers=[5,16,29] → in_layer=[6,17,30]
self.guidance_extractor.in_layer = [v + 1 for v in self.v_layers]
```

If you ever change `v_layers` or re-train, re-verify which hidden states the training used.

## Rejection Masking

**Always zero rejected draft positions in `attention_mask`** and **always pass `attention_mask` to the verifier**. This matches SD² and is required for:
- Correct verifier output (verified: diff=0.02 vs gold with mask, vs 4.1 without)
- Batched generation (bsz > 1) — prevents cross-sample KV contamination

The acceptance rate with masking (~0.6–1.5) is **lower** than without masking (~3.0) because the verifier is more accurate. The high unmasked rate was an artifact of both models attending to stale garbage KVs and accidentally agreeing.

**Never remove masking to chase higher acceptance numbers.** The masked numbers are correct.

## Non-Compact Layout

- `curr` always advances by `NG+1` per round
- Rejected tokens stay in `sampled[]` at their physical positions
- `position_ids` encode semantic positions (skip rejected), not physical buffer positions
- The drafter cache is rebuilt from scratch each round (Mamba2 is recurrent, no KV crop)

## Custom LlamaModel

`spec_mamba/models/llama.py` is SD²'s custom LlamaModel with old-style `_update_causal_mask`. It:
- Always builds a 4D float causal mask (no `is_causal=True` shortcut)
- Fills `min_dtype` at positions where `attention_mask=0`
- Returns `{"out": BaseModelOutputWithPast, "guide_embd": ...}` when `compute_guidance=True`

**Do not replace this with stock HF LlamaModel** — the stock model's masking path changed across transformers versions and may not handle rejection zeros correctly with SDPA.

## Key Paths

- Trained checkpoint: `/HSC/users/qiaoye/SSM_SPEC/checkpoints/guided_mamba/ckpts/last.ckpt`
- Hyperparams: `NG=8, v_layers=[5,16,29], loss_method=tvd, steer_z=False`
- Full architecture doc: `spec_mamba/ARCHITECTURE.md`
