"""
Evaluate vanilla LLaMA-1B → LLaMA-8B speculative decoding across 5 datasets.
Reports block efficiency (n_accepted + 1) and speedup vs AR baseline.

Usage:
    python -m spec_mamba.eval_llama_drafter [--drafter_path PATH] [--mask] [--no_mask]
"""

import argparse
import gc
import json
import random
import time

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, DynamicCache

from spec_mamba.models.llama import LlamaForCausalLM as CustomLlamaForCausalLM
from spec_mamba.trainer import rejection_sampling

torch.set_float32_matmul_precision("high")
torch.set_grad_enabled(False)

VERIFIER_PATH = "/HSC/users/qiaoye/checkpoints/Llama3.1-8B-hf"
DEFAULT_DRAFTER = "/HSC/users/qiaoye/checkpoints/Llama-3.2-1B"
NG = 8
DATASETS = ["ultrachat", "humaneval", "xsum", "alpaca", "gsm8k"]
DEFAULT_TOTAL_SAMPLES = 96
DEFAULT_TGT_LEN = 128


def get_prompts(dataset: str, n: int = DEFAULT_TOTAL_SAMPLES) -> list[str]:
    if dataset == "humaneval":
        data = load_dataset("openai_humaneval", split="test")
        return data["prompt"][:n]
    elif dataset == "gsm8k":
        data = load_dataset("gsm8k", "main", split="test")
        return data["question"][:n]
    elif dataset == "ultrachat":
        data = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft")
        return data["prompt"][:n]
    elif dataset == "alpaca":
        data = load_dataset("tatsu-lab/alpaca", split="train")
        items = list(data)
        random.Random(1).shuffle(items)
        return [item["instruction"] for item in items[:n]]
    elif dataset == "xsum":
        data = load_dataset("xsum", split="validation")
        return [f"Summarize the following document: {doc}" for doc in data["document"][:n]]
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def load_models(drafter_path: str):
    print("Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(VERIFIER_PATH)
    tok.pad_token = tok.eos_token

    print(f"Loading verifier (LLaMA 8B) from {VERIFIER_PATH}...")
    verifier = CustomLlamaForCausalLM.from_pretrained(
        VERIFIER_PATH, torch_dtype=torch.float16,
    ).cuda().eval()

    print(f"Loading drafter from {drafter_path}...")
    drafter = CustomLlamaForCausalLM.from_pretrained(
        drafter_path, torch_dtype=torch.float16,
    ).cuda().eval()

    return tok, verifier, drafter


@torch.no_grad()
def spec_dec_generate(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    verifier,
    drafter,
    max_new_tokens: int = DEFAULT_TGT_LEN,
    mask_rejected: bool = True,
):
    """Vanilla speculative decoding, non-compact layout, bsz=1."""
    B, S = input_ids.shape
    assert B == 1, "bsz=1 only"
    device = input_ids.device
    PAD_FACTOR = 4

    sampled = torch.full(
        (B, S + max_new_tokens * PAD_FACTOR),
        verifier.config.eos_token_id, dtype=torch.long, device=device,
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

    # Verifier prefill
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

    # Drafter prefill
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
    d_kv_len = curr

    total_na = 0
    total_steps = 0
    has_ended = torch.zeros(B, dtype=torch.bool, device=device)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_start = time.time()

    while not (
        has_ended.all()
        or (total_steps + 1) * (NG + 1) >= max_new_tokens * PAD_FACTOR
    ):
        # Draft NG tokens
        q = torch.zeros(B, NG, verifier.config.vocab_size, device=device)

        for i in range(NG):
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

            if d_prob.shape[-1] < verifier.config.vocab_size:
                pad = torch.zeros(B, verifier.config.vocab_size - d_prob.shape[-1], device=device)
                d_prob = torch.cat([d_prob, pad], dim=-1)
            elif d_prob.shape[-1] > verifier.config.vocab_size:
                d_prob = d_prob[:, :verifier.config.vocab_size]

            q[:, i] = d_prob
            sampled[:, curr + i + 1] = d_prob.argmax(dim=-1)
            attention_mask[:, curr + i + 1] = 1
            position_ids[:, curr + i + 1] = position_ids[:, curr + i] + 1

        # Verify
        v_out = verifier.get_decoder()(
            sampled[:, curr: curr + NG + 1],
            position_ids=position_ids[:, curr: curr + NG + 1],
            past_key_values=v_pkv,
            attention_mask=attention_mask[:, :curr + NG + 1],
            use_cache=True,
        )
        v_logits = verifier.lm_head(v_out.last_hidden_state)
        v_pkv = v_out.past_key_values

        v_probs = torch.zeros(
            B, NG + 1, verifier.config.vocab_size, device=device, dtype=v_logits.dtype,
        )
        v_probs.scatter_(-1, v_logits.argmax(dim=-1, keepdim=True), 1.0)

        NA, next_dist = rejection_sampling(
            q, v_probs, sampled[:, curr + 1: curr + NG + 1],
        )

        if mask_rejected:
            attention_mask[:, curr + 1: curr + NG + 1] = (
                torch.arange(0, NG, device=device)[None, :] < NA[:, None]
            ).long()

        position_ids[:, curr + NG + 1] = position_ids[:, curr] + NA + 1
        next_token = torch.multinomial(next_dist.clamp(min=0), 1).squeeze(1)
        sampled[:, curr + NG + 1] = next_token
        attention_mask[:, curr + NG + 1] = 1

        eos_id = verifier.config.eos_token_id
        has_ended = has_ended | (
            (sampled[:, curr + 1: curr + NG + 2]
             * attention_mask[:, curr + 1: curr + NG + 2])
            == eos_id
        ).any(dim=-1)

        curr = curr + NG + 1

        # Rebuild drafter KV cache (compact re-prefill)
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

        if position_ids[:, curr] - position_ids[:, S] >= max_new_tokens:
            break

        total_na += NA.item()
        total_steps += 1

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_time = time.time() - t_start

    avg_na = total_na / max(total_steps, 1)
    # Count actual tokens generated (from attention mask)
    tokens_generated = int(attention_mask[:, S:].sum().item())
    throughput = tokens_generated / max(total_time, 1e-6)

    return avg_na, total_steps, total_time, tokens_generated, throughput


@torch.no_grad()
def measure_ar_throughput(tok, verifier, datasets, total_samples, tgt_len):
    """Measure AR-only verifier throughput."""
    ar_results = {}
    print("\nMeasuring AR baseline throughput...")
    for dataset in datasets:
        prompts = get_prompts(dataset, n=min(total_samples, 48))
        total_tokens = 0
        total_time = 0.0

        for prompt in prompts:
            toks_out = tok(prompt, return_tensors="pt", truncation=True, max_length=512)
            input_ids = toks_out["input_ids"].cuda()

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.time()

            pkv = DynamicCache()
            v_out = verifier.get_decoder()(
                input_ids, past_key_values=pkv, use_cache=True,
            )
            pkv = v_out.past_key_values
            next_tok = verifier.lm_head(v_out.last_hidden_state[:, -1:]).argmax(dim=-1)

            for _ in range(tgt_len - 1):
                v_out = verifier.get_decoder()(
                    next_tok, past_key_values=pkv, use_cache=True,
                )
                pkv = v_out.past_key_values
                next_tok = verifier.lm_head(v_out.last_hidden_state[:, -1:]).argmax(dim=-1)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_time += time.time() - t0
            total_tokens += tgt_len

        ar_tps = total_tokens / max(total_time, 1e-6)
        ar_results[dataset] = ar_tps
        print(f"  {dataset}: AR throughput = {ar_tps:.1f} tok/s")
    return ar_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drafter_path", type=str, default=DEFAULT_DRAFTER)
    parser.add_argument("--mask", action="store_true", default=True,
                        help="Use rejection masking (default)")
    parser.add_argument("--no_mask", action="store_true",
                        help="Disable rejection masking")
    parser.add_argument("--datasets", type=str, default=None)
    parser.add_argument("--total_samples", type=int, default=DEFAULT_TOTAL_SAMPLES)
    parser.add_argument("--tgt_len", type=int, default=DEFAULT_TGT_LEN)
    parser.add_argument("--out_file", type=str, default="spec_mamba/eval_results_llama1b_drafter.json")
    parser.add_argument("--measure_ar_baseline", action="store_true")
    args = parser.parse_args()

    do_mask = not args.no_mask
    datasets = args.datasets.split(",") if args.datasets else DATASETS
    drafter_name = args.drafter_path.rstrip("/").split("/")[-1]
    mask_label = "mask" if do_mask else "nomask"

    tok, verifier, drafter = load_models(args.drafter_path)

    ar_throughput = {}
    if args.measure_ar_baseline:
        ar_throughput = measure_ar_throughput(tok, verifier, datasets, args.total_samples, args.tgt_len)

    final_out = {}
    if ar_throughput:
        final_out["ar_baseline"] = ar_throughput

    print(f"\n{'='*60}")
    print(f"LLaMA drafter eval: {drafter_name} → LLaMA-8B (NG={NG})")
    print(f"Masking: {do_mask}, Samples: {args.total_samples}, tgt_len={args.tgt_len}")
    print(f"{'='*60}")

    for dataset in datasets:
        print(f"\n  Dataset: {dataset}")
        prompts = get_prompts(dataset, n=args.total_samples)
        all_na = []
        all_throughput = []
        total_tokens = 0
        total_time_all = 0.0

        for i, prompt in enumerate(prompts):
            toks_out = tok(
                prompt, return_tensors="pt", truncation=True, max_length=512,
                return_attention_mask=True,
            )
            input_ids = toks_out["input_ids"].cuda()
            attn_mask = toks_out["attention_mask"].cuda()

            try:
                avg_na, steps, elapsed, gen_toks, throughput = spec_dec_generate(
                    input_ids, attn_mask, verifier, drafter,
                    max_new_tokens=args.tgt_len, mask_rejected=do_mask,
                )
                all_na.append(avg_na)
                all_throughput.append(throughput)
                total_tokens += gen_toks
                total_time_all += elapsed

                if (i + 1) % 16 == 0:
                    running_na = sum(all_na) / len(all_na)
                    running_tps = total_tokens / max(total_time_all, 1e-6)
                    print(f"    [{i+1}/{len(prompts)}] accepted: {running_na:.3f}, throughput: {running_tps:.1f} tok/s")

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"    OOM at sample {i}, skipping")
                    torch.cuda.empty_cache()
                    gc.collect()
                else:
                    raise

        n = len(all_na)
        mean_na = sum(all_na) / n if n else 0
        mean_throughput = total_tokens / max(total_time_all, 1e-6)
        block_eff = mean_na + 1

        key = f"{drafter_name}-{mask_label}-greedy-{dataset}"
        result = {
            "avg_n_accepted": mean_na,
            "block_efficiency": block_eff,
            "avg_throughput": mean_throughput,
            "n_samples": n,
        }
        if dataset in ar_throughput:
            result["ar_throughput"] = ar_throughput[dataset]
            result["speedup"] = mean_throughput / ar_throughput[dataset] if ar_throughput[dataset] > 0 else 0

        final_out[key] = result
        speedup_str = f", speedup={result.get('speedup', 0):.2f}x" if "speedup" in result else ""
        print(f"    RESULT: accepted={mean_na:.3f}, block_eff={block_eff:.3f}, "
              f"throughput={mean_throughput:.1f} tok/s{speedup_str}")

    import os
    os.makedirs(os.path.dirname(args.out_file) or ".", exist_ok=True)
    with open(args.out_file, "w") as f:
        json.dump(final_out, f, indent=2)
    print(f"\nResults saved to {args.out_file}")

    print(f"\n{'='*60}")
    print("Summary:")
    print(f"{'='*60}")
    for key, data in final_out.items():
        if key == "ar_baseline":
            continue
        speedup_str = f", speedup={data.get('speedup', 0):.2f}x" if "speedup" in data else ""
        print(f"  {key}: block_eff={data['block_efficiency']:.3f}, "
              f"accepted={data['avg_n_accepted']:.3f}, "
              f"throughput={data['avg_throughput']:.1f}{speedup_str}")


if __name__ == "__main__":
    main()
