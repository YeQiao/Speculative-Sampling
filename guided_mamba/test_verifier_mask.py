"""
Diagnostic: compare verifier logits with/without crop + position_ids + attention_mask.
Determines whether our changes cause the verifier to produce different outputs.
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

VERIFIER_PATH = "/HSC/users/qiaoye/checkpoints/Llama3.1-8B-hf"

def main():
    tok = AutoTokenizer.from_pretrained(VERIFIER_PATH)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        VERIFIER_PATH, torch_dtype=torch.float16, device_map="cuda"
    )
    model.eval()

    # Create a prompt with left-padding (batch of 2 different lengths)
    prompts = ["Hello world", "The quick brown fox jumps over the lazy dog today"]
    enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    B, S = input_ids.shape
    print(f"B={B}, S={S}")
    print(f"attention_mask sums: {attention_mask.sum(dim=1).tolist()}")

    curr = S - 1
    NG = 8
    PAD_FACTOR = 4
    max_new = 32

    # Build position_ids matching our generate() code
    position_ids = torch.arange(0, S + max_new * PAD_FACTOR, device="cuda")[None].expand(B, -1).clone()
    position_ids = torch.clamp_min(
        position_ids - S + attention_mask.sum(dim=1)[:, None], 0
    )

    # Extend attention_mask
    attention_mask_ext = torch.cat(
        (attention_mask.float(), torch.zeros(B, max_new * PAD_FACTOR, device="cuda")),
        dim=1,
    )

    # Extend input_ids
    sampled = torch.full((B, S + max_new * PAD_FACTOR), tok.pad_token_id, dtype=torch.long, device="cuda")
    sampled[:, :S] = input_ids

    # ============ METHOD A: No crop, no explicit mask/pos (original code) ============
    with torch.no_grad():
        v_pkv_a = DynamicCache()
        v_out_a = model.model(
            sampled[:, :curr + 1],
            past_key_values=v_pkv_a,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        v_pkv_a = v_out_a.past_key_values
        print(f"\n=== METHOD A (no fixes) ===")
        print(f"After prefill: cache_len={v_pkv_a.get_seq_length()}")

        # Simulate: pick some draft tokens (just use argmax of prefill logits)
        last_logits = model.lm_head(v_out_a.last_hidden_state[:, -1:])
        for i in range(NG):
            next_tok = last_logits[:, -1].argmax(dim=-1)
            sampled[:, curr + i + 1] = next_tok
            attention_mask_ext[:, curr + i + 1] = 1.0
            position_ids[:, curr + i + 1] = position_ids[:, curr + i] + 1

        # Verification step: feed curr..curr+NG (NO crop, NO mask, NO position_ids)
        v_out_verify_a = model.model(
            sampled[:, curr:curr + NG + 1],
            past_key_values=v_pkv_a,
            use_cache=True,
            output_hidden_states=False,
            return_dict=True,
        )
        logits_a = model.lm_head(v_out_verify_a.last_hidden_state)
        print(f"After verify: cache_len={v_out_verify_a.past_key_values.get_seq_length()}")
        print(f"Logits[0] top-5: {logits_a[0, 0].topk(5)}")

    # ============ METHOD B: With crop + position_ids + attention_mask (our fix) ============
    # Reset sampled
    sampled[:, :S] = input_ids
    with torch.no_grad():
        v_pkv_b = DynamicCache()
        v_out_b = model.model(
            sampled[:, :curr + 1],
            past_key_values=v_pkv_b,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
            attention_mask=attention_mask_ext[:, :curr + 1].to(torch.bool),
            position_ids=position_ids[:, :curr + 1],
        )
        v_pkv_b = v_out_b.past_key_values
        print(f"\n=== METHOD B (with fixes) ===")
        print(f"After prefill: cache_len={v_pkv_b.get_seq_length()}")
        v_pkv_b.crop(curr)
        print(f"After crop: cache_len={v_pkv_b.get_seq_length()}")

        # Same draft tokens
        last_logits_b = model.lm_head(v_out_b.last_hidden_state[:, -1:])
        for i in range(NG):
            next_tok = last_logits_b[:, -1].argmax(dim=-1)
            sampled[:, curr + i + 1] = next_tok
            attention_mask_ext[:, curr + i + 1] = 1.0
            position_ids[:, curr + i + 1] = position_ids[:, curr + i] + 1

        # Verification step: feed curr..curr+NG WITH crop + mask + position_ids
        v_out_verify_b = model.model(
            sampled[:, curr:curr + NG + 1],
            past_key_values=v_pkv_b,
            use_cache=True,
            output_hidden_states=False,
            return_dict=True,
            attention_mask=attention_mask_ext[:, :curr + NG + 1],
            position_ids=position_ids[:, curr:curr + NG + 1],
        )
        logits_b = model.lm_head(v_out_verify_b.last_hidden_state)
        print(f"After verify: cache_len={v_out_verify_b.past_key_values.get_seq_length()}")
        print(f"Logits[0] top-5: {logits_b[0, 0].topk(5)}")

    # ============ METHOD C: Gold standard — full sequence, no cache ============
    # Use the same draft tokens from method A
    sampled_c = sampled.clone()
    # Copy the draft tokens from method A (same tokens)
    with torch.no_grad():
        # Full forward on the entire sequence up to curr+NG+1 (no cache at all)
        full_out = model.model(
            sampled_c[:, :curr + NG + 1],
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
            # No attention_mask, no position_ids — let the model handle it
        )
        logits_c_full = model.lm_head(full_out.last_hidden_state)
        # Extract just the verification window: positions curr through curr+NG
        logits_c = logits_c_full[:, curr:curr + NG + 1]
        print(f"\n=== METHOD C (gold standard: full recompute, no cache) ===")
        print(f"Logits[0] top-5: {logits_c[0, 0].topk(5)}")

    # Also do with attention_mask for proper padding handling
    with torch.no_grad():
        full_out_mask = model.model(
            sampled_c[:, :curr + NG + 1],
            attention_mask=attention_mask_ext[:, :curr + NG + 1],
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )
        logits_d = model.lm_head(full_out_mask.last_hidden_state)[:, curr:curr + NG + 1]
        print(f"\n=== METHOD D (gold standard: full recompute + attention_mask) ===")
        print(f"Logits[0] top-5: {logits_d[0, 0].topk(5)}")

    # ============ Compare all ============
    print(f"\n=== COMPARISON ===")
    for name_x, logits_x, name_y, logits_y in [
        ("A(no fix)", logits_a, "B(crop+mask)", logits_b),
        ("A(no fix)", logits_a, "C(gold-nomask)", logits_c),
        ("A(no fix)", logits_a, "D(gold+mask)", logits_d),
        ("B(crop+mask)", logits_b, "C(gold-nomask)", logits_c),
        ("B(crop+mask)", logits_b, "D(gold+mask)", logits_d),
    ]:
        diff = (logits_x.float() - logits_y.float()).abs()
        argmax_match = (logits_x.argmax(-1) == logits_y.argmax(-1)).float().mean().item()
        print(f"  {name_x} vs {name_y}: max_diff={diff.max().item():.4f}, mean_diff={diff.mean().item():.4f}, argmax_match={argmax_match:.3f}")

    # Decoded tokens
    for name, logits_v in [("A", logits_a), ("B", logits_b), ("C", logits_c), ("D", logits_d)]:
        argmax_v = logits_v.argmax(dim=-1)
        print(f"  {name} argmax[0]: {argmax_v[0].tolist()} -> {tok.decode(argmax_v[0])}")
        print(f"  {name} argmax[1]: {argmax_v[1].tolist()} -> {tok.decode(argmax_v[1])}")


if __name__ == "__main__":
    main()
