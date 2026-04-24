"""
Compare SD²'s custom LlamaModel (_update_causal_mask from ~4.52.4)
vs stock HF 4.55.4 LlamaModel (create_causal_mask).

We monkey-patch the stock model to swap in SD²'s mask logic,
avoiding import issues (SD²'s full file targets transformers 4.52.4).

Tests:
1. Full-sequence forward: identical logits?
2. Incremental verification without mask/pos: stock vs patched
3. Incremental verification WITH mask/pos: stock vs patched
4. Cross-comparison of all methods against gold (full recompute)
"""

import copy
import types
import torch
from transformers import AutoTokenizer, DynamicCache, LlamaForCausalLM
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.cache_utils import Cache

V_MODEL = "/HSC/users/qiaoye/checkpoints/Llama3.1-8B-hf"
DEVICE = "cuda"

torch.set_grad_enabled(False)


# =====================================================================
#  SD²'s _update_causal_mask (from transformers ~4.52.4 / SD-square)
# =====================================================================
def _update_causal_mask_sd2(
    self,
    attention_mask,
    input_tensor,
    cache_position,
    past_key_values,
    output_attentions=False,
):
    """SD²'s old-style causal mask creation (float mask, not bool)."""
    if self.config._attn_implementation == "flash_attention_2":
        if attention_mask is not None and (attention_mask == 0.0).any():
            return attention_mask
        return None

    past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
    using_compilable_cache = past_key_values.is_compileable if past_key_values is not None else False

    if (
        self.config._attn_implementation == "sdpa"
        and not using_compilable_cache
        and not output_attentions
    ):
        if AttentionMaskConverter._ignore_causal_mask_sdpa(
            attention_mask,
            inputs_embeds=input_tensor,
            past_key_values_length=past_seen_tokens,
            is_training=self.training,
        ):
            return None

    dtype = input_tensor.dtype
    sequence_length = input_tensor.shape[1]
    if using_compilable_cache:
        target_length = past_key_values.get_max_cache_shape()
    else:
        target_length = (
            attention_mask.shape[-1]
            if isinstance(attention_mask, torch.Tensor)
            else past_seen_tokens + sequence_length + 1
        )

    causal_mask = _prepare_4d_causal_attention_mask_with_cache_position_sd2(
        attention_mask,
        sequence_length=sequence_length,
        target_length=target_length,
        dtype=dtype,
        cache_position=cache_position,
        batch_size=input_tensor.shape[0],
    )

    if (
        self.config._attn_implementation == "sdpa"
        and attention_mask is not None
        and attention_mask.device.type in ["cuda", "xpu", "npu"]
        and not output_attentions
    ):
        min_dtype = torch.finfo(dtype).min
        causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

    return causal_mask


def _prepare_4d_causal_attention_mask_with_cache_position_sd2(
    attention_mask,
    sequence_length,
    target_length,
    dtype,
    cache_position,
    batch_size,
    **kwargs,
):
    """SD²'s old-style 4D float mask creation."""
    if attention_mask is not None and attention_mask.dim() == 4:
        causal_mask = attention_mask
    else:
        min_dtype = torch.finfo(dtype).min
        causal_mask = torch.full(
            (sequence_length, target_length),
            fill_value=min_dtype,
            dtype=dtype,
            device=cache_position.device,
        )
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(
            target_length, device=cache_position.device
        ) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[
                :, None, None, :
            ].to(causal_mask.device)
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[
                :, :, :, :mask_length
            ].masked_fill(padding_mask, min_dtype)
    return causal_mask


# =====================================================================
#  SD²-style forward for LlamaModel (uses _update_causal_mask instead
#  of create_causal_mask)
# =====================================================================
def _sd2_model_forward(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    use_cache=None,
    output_hidden_states=None,
    cache_position=None,
    **kwargs,
):
    """LlamaModel.forward() using SD²'s _update_causal_mask."""
    from transformers.modeling_outputs import BaseModelOutputWithPast

    output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache()

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    # KEY DIFFERENCE: use SD²'s _update_causal_mask instead of create_causal_mask
    causal_mask = _update_causal_mask_sd2(
        self, attention_mask, inputs_embeds, cache_position, past_key_values, False
    )

    hidden_states = inputs_embeds
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    all_hidden_states = () if output_hidden_states else None

    for decoder_layer in self.layers[: self.config.num_hidden_layers]:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

    hidden_states = self.norm(hidden_states)
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
        hidden_states=all_hidden_states,
    )


def load_stock_hf():
    """Load stock HF 4.55.4 LlamaForCausalLM."""
    model = LlamaForCausalLM.from_pretrained(
        V_MODEL, torch_dtype=torch.float16,
        attn_implementation="sdpa",
    ).to(DEVICE).eval()
    return model


def load_sd2_patched():
    """Load stock HF LlamaForCausalLM but monkey-patch with SD²'s masking."""
    model = LlamaForCausalLM.from_pretrained(
        V_MODEL, torch_dtype=torch.float16,
        attn_implementation="sdpa",
    ).to(DEVICE).eval()
    # Replace the model's forward with SD²'s version
    model.model.forward = types.MethodType(_sd2_model_forward, model.model)
    return model


def test_full_forward(stock_model, sd2_model, input_ids):
    """Test 1: Full-sequence forward (no cache). Should be identical."""
    print("\n=== Test 1: Full-sequence forward (no cache) ===")

    stock_out = stock_model(input_ids=input_ids, use_cache=False)
    sd2_out = sd2_model(input_ids=input_ids, use_cache=False)

    stock_logits = stock_out.logits
    sd2_logits = sd2_out.logits

    max_diff = (stock_logits.float() - sd2_logits.float()).abs().max().item()
    argmax_match = (stock_logits.argmax(-1) == sd2_logits.argmax(-1)).float().mean().item()

    print(f"  Max logit diff: {max_diff:.6f}")
    print(f"  Argmax match:   {argmax_match:.4%}")


def test_incremental_no_mask(stock_model, sd2_model, input_ids, NG=8):
    """Test 2: Incremental verify step. Both models WITHOUT explicit mask/pos_ids."""
    print(f"\n=== Test 2: Incremental verify (NO mask/pos_ids), NG={NG} ===")
    B, S = input_ids.shape

    # --- Stock HF prefill ---
    stock_pkv = DynamicCache()
    stock_prefill = stock_model.model(
        input_ids[:, :-1],
        past_key_values=stock_pkv,
        use_cache=True,
    )
    stock_pkv = stock_prefill.past_key_values

    # Simulate NG+1 drafted tokens (just use the real continuation)
    draft_ids = input_ids[:, -1:].expand(B, NG + 1)  # dummy

    stock_out = stock_model.model(
        draft_ids,
        past_key_values=stock_pkv,
        use_cache=True,
    )
    stock_logits = stock_model.lm_head(stock_out.last_hidden_state)

    # --- SD² custom prefill ---
    sd2_pkv = DynamicCache()
    sd2_prefill = sd2_model.model(
        input_ids[:, :-1],
        past_key_values=sd2_pkv,
        use_cache=True,
    )
    sd2_pkv = sd2_prefill.past_key_values

    sd2_out = sd2_model.model(
        draft_ids,
        past_key_values=sd2_pkv,
        use_cache=True,
    )
    sd2_logits = sd2_model.lm_head(sd2_out.last_hidden_state)

    max_diff = (stock_logits.float() - sd2_logits.float()).abs().max().item()
    argmax_match = (stock_logits.argmax(-1) == sd2_logits.argmax(-1)).float().mean().item()

    print(f"  Max logit diff: {max_diff:.6f}")
    print(f"  Argmax match:   {argmax_match:.4%}")
    return stock_logits, sd2_logits


def test_incremental_with_mask(stock_model, sd2_model, input_ids, NG=8):
    """Test 3: Incremental verify step. Both WITH explicit attention_mask + position_ids (SD²-style).
    
    SD² crops v_pkv after prefill so the verify step re-processes the last prefilled token.
    """
    print(f"\n=== Test 3: Incremental verify WITH mask+pos_ids (SD²-style crop), NG={NG} ===")
    B, S = input_ids.shape
    total_len = S + NG

    position_ids = torch.arange(0, total_len, device=DEVICE)[None].expand(B, -1)
    full_attn_mask = torch.ones(B, total_len, device=DEVICE, dtype=torch.long)

    curr = S - 1  # last prefilled position

    # --- Stock HF with SD²-style crop ---
    stock_pkv = DynamicCache()
    stock_model.model(
        input_ids[:, :curr + 1],
        position_ids=position_ids[:, :curr + 1],
        attention_mask=full_attn_mask[:, :curr + 1],
        past_key_values=stock_pkv,
        use_cache=True,
    )
    stock_pkv.crop(curr)  # SD²-style crop

    draft_ids = input_ids[:, -1:].expand(B, NG + 1)
    stock_out = stock_model.model(
        draft_ids,
        position_ids=position_ids[:, curr:curr + NG + 1],
        attention_mask=full_attn_mask[:, :curr + NG + 1],
        past_key_values=stock_pkv,
        use_cache=True,
    )
    stock_logits = stock_model.lm_head(stock_out.last_hidden_state)

    # --- SD² patched with crop ---
    sd2_pkv = DynamicCache()
    sd2_model.model(
        input_ids[:, :curr + 1],
        position_ids=position_ids[:, :curr + 1],
        attention_mask=full_attn_mask[:, :curr + 1],
        past_key_values=sd2_pkv,
        use_cache=True,
    )
    sd2_pkv.crop(curr)  # SD²-style crop

    sd2_out = sd2_model.model(
        draft_ids,
        position_ids=position_ids[:, curr:curr + NG + 1],
        attention_mask=full_attn_mask[:, :curr + NG + 1],
        past_key_values=sd2_pkv,
        use_cache=True,
    )
    sd2_logits = sd2_model.lm_head(sd2_out.last_hidden_state)

    max_diff = (stock_logits.float() - sd2_logits.float()).abs().max().item()
    argmax_match = (stock_logits.argmax(-1) == sd2_logits.argmax(-1)).float().mean().item()

    print(f"  Max logit diff: {max_diff:.6f}")
    print(f"  Argmax match:   {argmax_match:.4%}")
    return stock_logits, sd2_logits


def test_cross_comparison(stock_model, sd2_model, input_ids, NG=8):
    """Test 4: Compare all 4 method combinations against gold (full recompute)."""
    print(f"\n=== Test 4: Cross-comparison (gold=full recompute) NG={NG} ===")
    B, S = input_ids.shape

    curr = S - 1
    draft_ids = input_ids[:, -1:].expand(B, NG + 1)

    position_ids = torch.arange(0, S + NG, device=DEVICE)[None].expand(B, -1)
    full_mask = torch.ones(B, S + NG, device=DEVICE, dtype=torch.long)

    # ---------- Method A: Stock HF, no mask/pos (what we currently do) ----------
    pkv_a = DynamicCache()
    stock_model.model(input_ids[:, :curr + 1], past_key_values=pkv_a, use_cache=True)
    out_a = stock_model.model(draft_ids, past_key_values=pkv_a, use_cache=True)
    logits_a = stock_model.lm_head(out_a.last_hidden_state)

    # ---------- Method B: SD² mask, with mask+pos + crop (SD²'s approach) ----------
    pkv_b = DynamicCache()
    sd2_model.model(
        input_ids[:, :curr + 1],
        position_ids=position_ids[:, :curr + 1],
        attention_mask=full_mask[:, :curr + 1],
        past_key_values=pkv_b, use_cache=True,
    )
    pkv_b.crop(curr)  # SD²-style crop
    out_b = sd2_model.model(
        draft_ids,
        position_ids=position_ids[:, curr:curr + NG + 1],
        attention_mask=full_mask[:, :curr + NG + 1],
        past_key_values=pkv_b, use_cache=True,
    )
    logits_b = sd2_model.lm_head(out_b.last_hidden_state)

    # ---------- Method C: Stock HF, with mask+pos + crop ----------
    pkv_c = DynamicCache()
    stock_model.model(
        input_ids[:, :curr + 1],
        position_ids=position_ids[:, :curr + 1],
        attention_mask=full_mask[:, :curr + 1],
        past_key_values=pkv_c, use_cache=True,
    )
    pkv_c.crop(curr)  # crop like SD²
    out_c = stock_model.model(
        draft_ids,
        position_ids=position_ids[:, curr:curr + NG + 1],
        attention_mask=full_mask[:, :curr + NG + 1],
        past_key_values=pkv_c, use_cache=True,
    )
    logits_c = stock_model.lm_head(out_c.last_hidden_state)

    # ---------- Method D: SD² mask, no mask/pos ----------
    pkv_d = DynamicCache()
    sd2_model.model(input_ids[:, :curr + 1], past_key_values=pkv_d, use_cache=True)
    out_d = sd2_model.model(draft_ids, past_key_values=pkv_d, use_cache=True)
    logits_d = sd2_model.lm_head(out_d.last_hidden_state)

    # ---------- Gold: Full recompute (no cache) ----------
    # Methods A,D: prefill [0..curr], verify [curr..curr+NG] → first output = P(x|[0..curr])
    # Methods B,C: crop to curr, so first input at cache_position=curr → P(x|[0..curr-1, curr])
    # Both should produce same logits if done correctly
    full_input = torch.cat([input_ids[:, :curr + 1], draft_ids], dim=1)
    gold_out = stock_model(input_ids=full_input, use_cache=False)
    
    # For A/D (no crop): prefill puts [0..curr] into cache, verify gets [curr..curr+NG]
    # The verify output[0] = logits for token at position curr (which already saw [0..curr-1] from cache, plus curr itself)
    # This matches gold[:, curr:curr+NG+1]
    logits_gold_no_crop = gold_out.logits[:, curr:curr + NG + 1]
    
    # For B/C (with crop): cache has [0..curr-1], verify gets [curr..curr+NG] 
    # Same alignment — verify output[0] = logits for position curr, having seen [0..curr-1] + curr
    logits_gold_crop = gold_out.logits[:, curr:curr + NG + 1]

    print(f"  {'Method':<45} {'Logit max diff':>15} {'Argmax match':>15}")
    print(f"  {'='*77}")
    for name, logits, gold in [
        ("A: Stock HF, no mask/pos", logits_a, logits_gold_no_crop),
        ("B: SD² mask + crop + mask+pos", logits_b, logits_gold_crop),
        ("C: Stock HF + crop + mask+pos", logits_c, logits_gold_crop),
        ("D: SD² mask, no mask/pos", logits_d, logits_gold_no_crop),
    ]:
        diff = (logits.float() - gold.float()).abs().max().item()
        argmax = (logits.argmax(-1) == gold.argmax(-1)).float().mean().item()
        print(f"  {name:<45} {diff:>15.6f} {argmax:>14.4%}")

    # Softmax TVD comparison
    print(f"\n  {'Method':<45} {'Max prob diff':>15} {'TVD':>15}")
    print(f"  {'='*77}")
    for name, logits, gold in [
        ("A: Stock HF, no mask/pos", logits_a, logits_gold_no_crop),
        ("B: SD² mask + crop + mask+pos", logits_b, logits_gold_crop),
        ("C: Stock HF + crop + mask+pos", logits_c, logits_gold_crop),
        ("D: SD² mask, no mask/pos", logits_d, logits_gold_no_crop),
    ]:
        probs = logits.float().softmax(-1)
        probs_gold = gold.float().softmax(-1)
        prob_diff = (probs - probs_gold).abs().max().item()
        tvd = (probs - probs_gold).abs().sum(-1).mean().item() / 2
        print(f"  {name:<45} {prob_diff:>15.6f} {tvd:>14.6f}")


def main():
    print("Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(V_MODEL)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    prompt = "The capital of France is"
    input_ids = tok(prompt, return_tensors="pt")["input_ids"].to(DEVICE)
    print(f"Input: '{prompt}' -> shape {input_ids.shape}")

    print("\nLoading stock HF 4.55.4 LlamaModel...")
    stock_model = load_stock_hf()
    print("Loading SD² patched LlamaModel (same weights, old-style mask)...")
    sd2_model = load_sd2_patched()

    # Verify they share the same weights
    print("\nVerifying weight identity...")
    for (n1, p1), (n2, p2) in zip(
        stock_model.named_parameters(), sd2_model.named_parameters()
    ):
        assert n1 == n2, f"Name mismatch: {n1} vs {n2}"
        assert torch.equal(p1, p2), f"Weight mismatch at {n1}"
    print("  All weights identical ✓")

    test_full_forward(stock_model, sd2_model, input_ids)
    test_incremental_no_mask(stock_model, sd2_model, input_ids)
    test_incremental_with_mask(stock_model, sd2_model, input_ids)
    test_cross_comparison(stock_model, sd2_model, input_ids)


if __name__ == "__main__":
    main()
