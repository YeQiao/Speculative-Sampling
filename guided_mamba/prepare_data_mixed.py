"""
Prepare a mixed-domain dataset for guided Mamba2 training.

Combines UltraChat, HumanEval, XSum, Alpaca, and GSM8K to improve
generalization across diverse tasks.

Usage:
    python -m guided_mamba.prepare_data_mixed \
        --output /HSC/users/qiaoye/SSM_SPEC/data/mixed_guided_gemma \
        --tokenizer google/gemma-4-E4B-it \
        --max_length 512 \
        --n_ultrachat 40000 \
        --n_other 2500
"""

import argparse
import os
import random

import datasets
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer


def load_sources(n_ultrachat: int, n_other: int, seed: int = 42):
    """Load and format all sources as list of chat messages."""
    rng = random.Random(seed)
    all_train = []
    all_val = []

    # 1. UltraChat (bulk of training data, already in chat format)
    print("Loading UltraChat...")
    raw = datasets.load_dataset("HuggingFaceH4/ultrachat_200k")
    train_msgs = [ex["messages"] for ex in raw["train_sft"]]
    rng.shuffle(train_msgs)
    all_train.extend(train_msgs[:n_ultrachat])
    val_msgs = [ex["messages"] for ex in raw["test_sft"]]
    all_val.extend(val_msgs[:1000])
    print(f"  UltraChat: {min(n_ultrachat, len(train_msgs))} train, 1000 val")

    # 2. HumanEval (code completion prompts)
    print("Loading HumanEval...")
    he = datasets.load_dataset("openai_humaneval", split="test")
    he_msgs = []
    for ex in he:
        he_msgs.append([
            {"role": "user", "content": ex["prompt"]},
            {"role": "assistant", "content": ex["canonical_solution"]},
        ])
    rng.shuffle(he_msgs)
    all_train.extend(he_msgs[:n_other])
    all_val.extend(he_msgs[n_other:n_other + 250])
    print(f"  HumanEval: {min(n_other, len(he_msgs))} train, {min(250, len(he_msgs) - n_other)} val")

    # 3. GSM8K (math reasoning)
    print("Loading GSM8K...")
    gsm = datasets.load_dataset("gsm8k", "main", split="train")
    gsm_msgs = []
    for ex in gsm:
        gsm_msgs.append([
            {"role": "user", "content": ex["question"]},
            {"role": "assistant", "content": ex["answer"]},
        ])
    rng.shuffle(gsm_msgs)
    all_train.extend(gsm_msgs[:n_other])
    all_val.extend(gsm_msgs[n_other:n_other + 250])
    print(f"  GSM8K: {min(n_other, len(gsm_msgs))} train, 250 val")

    # 4. Alpaca (instruction following)
    print("Loading Alpaca...")
    alp = datasets.load_dataset("tatsu-lab/alpaca", split="train")
    alp_msgs = []
    for ex in alp:
        inp = ex["instruction"]
        if ex["input"]:
            inp += f"\n\nInput: {ex['input']}"
        alp_msgs.append([
            {"role": "user", "content": inp},
            {"role": "assistant", "content": ex["output"]},
        ])
    rng.shuffle(alp_msgs)
    all_train.extend(alp_msgs[:n_other])
    all_val.extend(alp_msgs[n_other:n_other + 250])
    print(f"  Alpaca: {min(n_other, len(alp_msgs))} train, 250 val")

    # 5. XSum (summarization)
    print("Loading XSum...")
    xsum = datasets.load_dataset("xsum", split="train")
    xsum_msgs = []
    for ex in xsum:
        xsum_msgs.append([
            {"role": "user", "content": f"Summarize the following document:\n\n{ex['document']}"},
            {"role": "assistant", "content": ex["summary"]},
        ])
    rng.shuffle(xsum_msgs)
    all_train.extend(xsum_msgs[:n_other])
    all_val.extend(xsum_msgs[n_other:n_other + 250])
    print(f"  XSum: {min(n_other, len(xsum_msgs))} train, 250 val")

    # Shuffle combined data
    rng.shuffle(all_train)
    rng.shuffle(all_val)
    print(f"\nTotal: {len(all_train)} train, {len(all_val)} val")
    return all_train, all_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--n_ultrachat", type=int, default=40000,
                        help="Number of UltraChat samples (bulk)")
    parser.add_argument("--n_other", type=int, default=2500,
                        help="Number of samples per non-UltraChat source")
    parser.add_argument("--tokenizer", type=str,
                        default="google/gemma-4-E4B-it")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    tok.truncation_side = "left"

    train_msgs, val_msgs = load_sources(args.n_ultrachat, args.n_other, args.seed)

    def tokenize_batch(messages_list):
        """Tokenize a batch of message lists."""
        results = {"targets": [], "loss_mask": [], "prompt": []}
        for messages in messages_list:
            out = tok.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                add_generation_prompt=False,
                return_assistant_tokens_mask=True,
                padding="max_length",
                max_length=args.max_length,
                truncation=True,
            )
            results["targets"].append(out["input_ids"])
            results["loss_mask"].append(out["assistant_masks"])
            # First user message as prompt
            first_user = next(
                (m["content"] for m in messages if m["role"] == "user"), ""
            )
            results["prompt"].append(first_user[:200])  # truncate prompt string
        return results

    # Process in chunks
    def process_split(messages_list, desc="Processing"):
        all_targets, all_masks, all_prompts = [], [], []
        chunk_size = 256
        for i in range(0, len(messages_list), chunk_size):
            if i % (chunk_size * 10) == 0:
                print(f"  {desc}: {i}/{len(messages_list)}")
            chunk = messages_list[i:i + chunk_size]
            out = tokenize_batch(chunk)
            all_targets.extend(out["targets"])
            all_masks.extend(out["loss_mask"])
            all_prompts.extend(out["prompt"])
        return Dataset.from_dict({
            "targets": all_targets,
            "loss_mask": all_masks,
            "prompt": all_prompts,
        })

    print("\nTokenizing train split...")
    train_ds = process_split(train_msgs, "Train")
    print("Tokenizing val split...")
    val_ds = process_split(val_msgs, "Val")

    ds = DatasetDict({"train": train_ds, "validation": val_ds})

    os.makedirs(args.output, exist_ok=True)
    ds.save_to_disk(args.output)
    print(f"\nSaved to {args.output}")
    print(f"Train: {len(ds['train'])}, Val: {len(ds['validation'])}")

    # Quick sanity check
    sample = ds["train"][0]
    print(f"Sample targets length: {len(sample['targets'])}")
    print(f"Sample loss_mask sum: {sum(sample['loss_mask'])}")


if __name__ == "__main__":
    main()
