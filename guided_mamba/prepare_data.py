"""
Prepare UltraChat-200k dataset for guided Mamba2 training.

Tokenizes conversations using the LLaMA 3.1 tokenizer (shared with Mamba2 drafter)
and saves as a HuggingFace DatasetDict with train/validation splits.

Usage:
    python -m guided_mamba.prepare_data \
        --output /path/to/save/dataset \
        --max_length 512 \
        --n_train 50000 \
        --n_val 5000
"""

import argparse
import os

import datasets
from datasets import DatasetDict
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for the saved DatasetDict")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Max sequence length for tokenization")
    parser.add_argument("--n_train", type=int, default=None,
                        help="Limit training samples (None = use all)")
    parser.add_argument("--n_val", type=int, default=5000,
                        help="Limit validation samples")
    parser.add_argument("--tokenizer", type=str,
                        default="/HSC/users/qiaoye/checkpoints/Llama3.1-8B-hf",
                        help="Tokenizer path (shared by LLaMA verifier and Mamba2 drafter)")
    args = parser.parse_args()

    # Load tokenizer (shared between verifier and drafter)
    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    tok.truncation_side = "left"

    # Load ultrachat_200k
    print("Loading ultrachat_200k...")
    raw = datasets.load_dataset("HuggingFaceH4/ultrachat_200k")
    ds = DatasetDict({
        "train": raw["train_sft"],
        "validation": raw["test_sft"],
    })

    if args.n_train is not None:
        ds["train"] = ds["train"].select(range(min(args.n_train, len(ds["train"]))))
    if args.n_val is not None:
        ds["validation"] = ds["validation"].select(range(min(args.n_val, len(ds["validation"]))))

    print(f"Train: {len(ds['train'])} samples, Val: {len(ds['validation'])} samples")

    # Tokenize with chat template + loss mask
    def preprocess(examples):
        out = tok.apply_chat_template(
            examples["messages"],
            tokenize=True,
            return_dict=True,
            add_generation_prompt=False,
            return_assistant_tokens_mask=True,
            padding="max_length",
            max_length=args.max_length,
            truncation=True,
        )
        examples["targets"] = out["input_ids"]
        examples["loss_mask"] = out["assistant_masks"]
        # Keep first user message as prompt for eval
        prompts = []
        for convo in examples["messages"]:
            first_user = next(
                (m["content"] for m in convo if m["role"] == "user"),
                "",
            )
            prompts.append(first_user)
        examples["prompt"] = prompts
        return examples

    print("Tokenizing...")
    ds = ds.map(
        preprocess,
        batched=True,
        batch_size=256,
        num_proc=4,
        remove_columns=[c for c in ds["train"].column_names if c not in ("targets", "loss_mask", "prompt")],
        desc="Tokenizing",
    )

    # Save
    os.makedirs(args.output, exist_ok=True)
    ds.save_to_disk(args.output)
    print(f"Saved to {args.output}")

    # Print stats
    sample = ds["train"][0]
    print(f"Sample targets length: {len(sample['targets'])}")
    print(f"Sample loss_mask sum: {sum(sample['loss_mask'])} / {len(sample['loss_mask'])}")
    print(f"Sample prompt: {sample['prompt'][:100]}...")


if __name__ == "__main__":
    main()
