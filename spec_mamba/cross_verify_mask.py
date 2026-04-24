"""
Cross-verify mask vs no-mask acceptance rates using vanilla LLaMA-1B → LLaMA-8B.
No guidance, no KD — pure standard speculative decoding.
This isolates whether the masked/unmasked gap is a property of the non-compact layout
or specific to our guided Mamba2 drafter.

Usage:
    python -m spec_mamba.cross_verify_mask
"""

import time
import torch
from transformers import AutoTokenizer, DynamicCache
from datasets import load_dataset

from spec_mamba.models.llama import LlamaForCausalLM as CustomLlamaForCausalLM
from spec_mamba.trainer import rejection_sampling

torch.set_float32_matmul_precision("high")
torch.set_grad_enabled(False)

VERIFIER_PATH = "/HSC/users/qiaoye/checkpoints/Llama3.1-8B-hf"
DRAFTER_PATH  = "/HSC/users/qiaoye/checkpoints/Llama-3.2-1B"
NG = 8
NUM_SAMPLES = 32
TGT_LEN = 128


def load_models():
    print("Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(VERIFIER_PATH)
    tok.pad_token = tok.eos_token

    print("Loading verifier (LLaMA 8B)...")
    verifier = CustomLlamaForCausalLM.from_pretrained(
        VERIFIER_PATH, torch_dtype=torch.float16,
    ).cuda().eval()

    print("Loading drafter (LLaMA 1B)...")
    drafter = CustomLlamaForCausalLM.from_pretrained(
        DRAFTER_PATH, torch_dtype=torch.float16,
    ).cuda().eval()

    return tok, verifier, drafter


@torch.no_grad()
def spec_dec_generate(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    verifier,
    drafter,
    max_new_tokens: int = TGT_LEN,
    mask_rejected: bool = True,
):
    """Vanilla speculative decoding, non-compact layout, bsz=1."""
    B, S = input_ids.shape
    assert B == 1, "bsz=1 only"
    device = input_ids.device
    PAD_FACTOR = 4

    # Pre-allocate buffer
    sampled = torch.full(
        (B, S + max_new_tokens * PAD_FACTOR),
        drafter.config.eos_token_id, dtype=torch.long, device=device,
    )
    sampled[:, :S] = input_ids
    curr = S - 1

    position_ids = torch.arange(
        0, S + max_new_tokens * PAD_FACTOR, device=device,
    )[None].expand(B, -1).clone()

    if attention_mask is not None:
        position_ids = torch.clamp_min_(
            position_ids - S + attention_mask.sum(dim=1)[:, None], 0
        )
        attention_mask = torch.cat(
            (attention_mask, torch.zeros(B, max_new_tokens * PAD_FACTOR, device=device, dtype=attention_mask.dtype)),
            dim=1,
        )
    else:
        attention_mask = torch.zeros(
            B, S + max_new_tokens * PAD_FACTOR, device=device, dtype=torch.long,
        )
        attention_mask[:, :S] = 1

    # ----- Verifier prefill -----
    v_pkv = DynamicCache()
    v_out = verifier.get_decoder()(
        sampled[:, :curr + 1],
        position_ids=position_ids[:, :curr + 1],
        past_key_values=v_pkv,
        attention_mask=attention_mask[:, :curr + 1],
        use_cache=True,
    )
    v_pkv = v_out.past_key_values
    v_pkv.crop(curr)

    # ----- Drafter prefill -----
    d_pkv = DynamicCache()
    d_out = drafter.get_decoder()(
        sampled[:, :curr + 1],
        position_ids=position_ids[:, :curr + 1],
        past_key_values=d_pkv,
        attention_mask=attention_mask[:, :curr + 1],
        use_cache=True,
    )
    d_pkv = d_out.past_key_values
    d_pkv.crop(curr)
    d_kv_len = curr  # track drafter KV cache length separately

    # ----- Speculative decoding loop -----
    total_na = 0
    total_steps = 0
    has_ended = torch.zeros(B, dtype=torch.bool, device=device)

    while not (
        has_ended.all()
        or (total_steps + 1) * (NG + 1) >= max_new_tokens * PAD_FACTOR
    ):
        # --- Draft NG tokens ---
        q = torch.zeros(B, NG, verifier.config.vocab_size, device=device)

        for i in range(NG):
            # Build drafter-specific attention mask matching its KV length
            d_mask_len = d_kv_len + i + 1
            d_attn_mask = torch.ones(B, d_mask_len, device=device, dtype=torch.long)
            d_out = drafter.get_decoder()(
                sampled[:, curr + i: curr + i + 1],
                position_ids=position_ids[:, curr + i: curr + i + 1],
                past_key_values=d_pkv,
                attention_mask=d_attn_mask,
                use_cache=True,
            )
            d_pkv = d_out.past_key_values
            d_logits = drafter.lm_head(d_out.last_hidden_state)
            d_prob = d_logits[:, -1].softmax(-1)

            # Pad to verifier vocab size if needed (1B has 128256 too, but just in case)
            if d_prob.shape[-1] < verifier.config.vocab_size:
                pad = torch.zeros(B, verifier.config.vocab_size - d_prob.shape[-1], device=device)
                d_prob = torch.cat([d_prob, pad], dim=-1)
            elif d_prob.shape[-1] > verifier.config.vocab_size:
                d_prob = d_prob[:, :verifier.config.vocab_size]

            q[:, i] = d_prob
            sampled[:, curr + i + 1] = d_prob.argmax(dim=-1)  # greedy
            attention_mask[:, curr + i + 1] = 1
            position_ids[:, curr + i + 1] = position_ids[:, curr + i] + 1

        # --- Verify NG+1 tokens ---
        v_out = verifier.get_decoder()(
            sampled[:, curr: curr + NG + 1],
            position_ids=position_ids[:, curr: curr + NG + 1],
            past_key_values=v_pkv,
            attention_mask=attention_mask[:, :curr + NG + 1],
            use_cache=True,
        )
        v_logits = verifier.lm_head(v_out.last_hidden_state)
        v_pkv = v_out.past_key_values

        # Greedy verifier probs
        v_probs = torch.zeros(
            B, NG + 1, verifier.config.vocab_size, device=device, dtype=v_logits.dtype,
        )
        v_probs.scatter_(-1, v_logits.argmax(dim=-1, keepdim=True), 1.0)

        # --- Rejection sampling ---
        NA, next_dist = rejection_sampling(
            q, v_probs, sampled[:, curr + 1: curr + NG + 1],
        )

        # Mask rejected positions
        if mask_rejected:
            attention_mask[:, curr + 1: curr + NG + 1] = (
                torch.arange(0, NG, device=device)[None, :] < NA[:, None]
            ).long()

        # Place next token
        position_ids[:, curr + NG + 1] = position_ids[:, curr] + NA + 1
        next_token = torch.multinomial(next_dist.clamp(min=0), 1).squeeze(1)
        sampled[:, curr + NG + 1] = next_token
        attention_mask[:, curr + NG + 1] = 1

        # EOS check
        eos_id = verifier.config.eos_token_id
        has_ended = has_ended | (
            (sampled[:, curr + 1: curr + NG + 2]
             * attention_mask[:, curr + 1: curr + NG + 2])
            == eos_id
        ).any(dim=-1)

        # Advance
        old_curr = curr
        curr = curr + NG + 1

        # Rebuild drafter KV cache
        # For the transformer drafter with non-compact layout, re-prefill
        # the accepted sequence. Build compact input from attention_mask.
        accepted_mask = attention_mask[:, :curr + 1].bool()
        compact_ids = sampled[:, :curr + 1][accepted_mask].unsqueeze(0)
        compact_len = compact_ids.shape[1]
        compact_pos = torch.arange(compact_len, device=device).unsqueeze(0)
        compact_mask = torch.ones(1, compact_len, device=device, dtype=torch.long)

        d_pkv = DynamicCache()
        d_out = drafter.get_decoder()(
            compact_ids,
            position_ids=compact_pos,
            past_key_values=d_pkv,
            attention_mask=compact_mask,
            use_cache=True,
        )
        d_pkv = d_out.past_key_values
        d_kv_len = compact_len

        # Termination
        if position_ids[:, curr] - position_ids[:, S] >= max_new_tokens:
            break

        total_na += NA.item()
        total_steps += 1

    avg_na = total_na / max(total_steps, 1)
    return sampled, avg_na, total_steps


def get_prompts(n=NUM_SAMPLES):
    data = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft")
    return data["prompt"][:n]


def main():
    tok, verifier, drafter = load_models()

    prompts = get_prompts()
    print(f"\n{'='*60}")
    print(f"Cross-verification: LLaMA-1B → LLaMA-8B spec dec (NG={NG})")
    print(f"Samples: {len(prompts)}, tgt_len={TGT_LEN}, greedy, bsz=1")
    print(f"{'='*60}")

    for mask_mode, label in [(True, "WITH masking"), (False, "WITHOUT masking")]:
        print(f"\n--- {label} ---")
        all_na = []

        for i, prompt in enumerate(prompts):
            toks = tok(
                prompt, return_tensors="pt", padding=True, truncation=True,
                max_length=512, return_attention_mask=True,
            )
            input_ids = toks["input_ids"].cuda()
            attn_mask = toks["attention_mask"].cuda()

            try:
                _, avg_na, steps = spec_dec_generate(
                    input_ids, attn_mask, verifier, drafter,
                    max_new_tokens=TGT_LEN,
                    mask_rejected=mask_mode,
                )
                all_na.append(avg_na)
                if (i + 1) % 8 == 0:
                    running_avg = sum(all_na) / len(all_na)
                    print(f"  [{i+1}/{len(prompts)}] running avg accepted: {running_avg:.3f}")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  OOM at sample {i}, skipping")
                    torch.cuda.empty_cache()
                else:
                    raise

        mean_na = sum(all_na) / len(all_na) if all_na else 0
        print(f"\n  RESULT ({label}): mean accepted = {mean_na:.3f} / {NG} ({len(all_na)} samples)")

    print(f"\n{'='*60}")
    print("If both modes show similar gap (masked << unmasked),")
    print("it confirms the inflation is a property of non-compact layout,")
    print("not specific to our guided Mamba2 drafter.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
