"""
Evaluation script for guided Mamba2 speculative decoding.

Usage:
    # Evaluate a checkpoint:
    python -m guided_mamba.eval \
        --ckpt /HSC/users/qiaoye/SSM_SPEC/checkpoints/guided_mamba/ckpts/last.ckpt

    # Evaluate with baseline (unguided) drafter for comparison:
    python -m guided_mamba.eval \
        --ckpt /HSC/users/qiaoye/SSM_SPEC/checkpoints/guided_mamba/ckpts/last.ckpt \
        --baseline

    # Filter datasets:
    python -m guided_mamba.eval --ckpt last.ckpt --datasets ultrachat,humaneval
"""

import argparse
import gc
import json
import random
import time

import torch
from datasets import load_dataset

from guided_mamba.train import GuidedMambaTrainer

torch.set_float32_matmul_precision("high")
torch.set_grad_enabled(False)

# -------------------------------------------------------------------------
#  Dataset loading (same prompts as SD²)
# -------------------------------------------------------------------------
DATASETS = ["ultrachat", "humaneval", "xsum", "alpaca", "gsm8k"]

DEFAULT_BSZ = 1
DEFAULT_TOTAL_SAMPLES = 96
DEFAULT_TGT_LEN = 128
SEEDS = [0, 1, 2]


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


# -------------------------------------------------------------------------
#  Checkpoint loading
# -------------------------------------------------------------------------
def load_from_ckpt(ckpt_path: str) -> GuidedMambaTrainer:
    """Load a GuidedMambaTrainer from a Lightning checkpoint."""
    st = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = {k: v for k, v in st["hyper_parameters"].items() if k != "_instantiator"}
    mod = GuidedMambaTrainer(**hp)

    # Load state_dict (checkpoint excludes v_base.* so strict=False)
    mod.load_state_dict(st["state_dict"], strict=False)
    return mod


def load_baseline(hp: dict) -> GuidedMambaTrainer:
    """Load an unguided baseline with the same verifier/drafter (no trained steering)."""
    mod = GuidedMambaTrainer(**hp)
    # Zero out all steering params so guidance has no effect
    with torch.no_grad():
        for p in mod.latent_mod_prep.parameters():
            p.zero_()
    return mod


# -------------------------------------------------------------------------
#  Main evaluation loop
# -------------------------------------------------------------------------
def evaluate_model(
    mod: GuidedMambaTrainer,
    name: str,
    datasets: list[str],
    greedy_modes: list[bool],
    seeds: list[int],
    out: dict,
    bsz: int = DEFAULT_BSZ,
    tgt_len: int = DEFAULT_TGT_LEN,
    total_samples: int = DEFAULT_TOTAL_SAMPLES,
    draft_temperature: float = 1.0,
):
    for g in greedy_modes:
        mod.greedy_sample = g
        mode_str = "greedy" if g else "sample"
        print(f"\n  Sampling mode: {mode_str}")

        for dataset in datasets:
            this_name = f"{name}-{mode_str}-{dataset}"
            print(f"    Dataset: {dataset}")
            prompts = get_prompts(dataset, n=total_samples)
            this_seeds = seeds[:1] if g else seeds

            all_outputs = []
            total_na = 0.0
            total_throughput = 0.0
            n_batches = 0

            for seed in this_seeds:
                torch.manual_seed(seed)
                for i in range(0, len(prompts), bsz):
                    batch_prompts = prompts[i:i + bsz]
                    try:
                        input_ids, attention_mask = mod.prep_for_gen(batch_prompts)
                        sampled, extra = mod.generate(
                            input_ids, attention_mask, max_new_tokens=tgt_len,
                            draft_temperature=1.0 if g else draft_temperature,
                        )
                        na = extra["n_accepted"].cpu().numpy()
                        throughput = (na + 1) / (extra["time_per_block"] / len(batch_prompts) + 1e-12)

                        for j in range(len(batch_prompts)):
                            out_mask = extra["attention_mask"]
                            if out_mask is not None:
                                tok_ids = sampled[j, out_mask[j].bool()].cpu()
                            else:
                                # Strip padding from the right
                                tok_ids = sampled[j][sampled[j] != mod.pad_token_id].cpu()
                            text = mod.tok.decode(tok_ids.numpy(), skip_special_tokens=True)
                            all_outputs.append({
                                "text": text,
                                "n_accepted": float(na[j]),
                                "throughput": float(throughput[j]),
                            })

                        total_na += na.sum()
                        total_throughput += throughput.sum()
                        n_batches += 1

                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            print(f"      OOM at batch {i}, skipping...")
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            gc.collect()
                        else:
                            raise

            n_samples = len(all_outputs)
            avg_na = total_na / max(n_samples, 1)
            avg_throughput = total_throughput / max(n_samples, 1)

            out[this_name] = {
                "avg_n_accepted": float(avg_na),
                "avg_throughput": float(avg_throughput),
                "n_samples": n_samples,
                "outputs": all_outputs,
            }
            print(f"      avg accepted: {avg_na:.3f}, avg throughput: {avg_throughput:.1f} tok/s")


def main():
    parser = argparse.ArgumentParser(description="Evaluate guided Mamba2 speculative decoding")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to Lightning checkpoint (.ckpt)")
    parser.add_argument("--baseline", action="store_true", help="Also evaluate unguided baseline")
    parser.add_argument("--datasets", type=str, default=None, help="Comma-separated dataset names (default: all)")
    parser.add_argument("--out_file", type=str, default="guided_mamba/eval_results.json")
    parser.add_argument("--greedy_only", action="store_true", help="Only evaluate greedy sampling")
    parser.add_argument("--sample_only", action="store_true", help="Only evaluate stochastic sampling")
    parser.add_argument("--bsz", type=int, default=DEFAULT_BSZ, help="Batch size")
    parser.add_argument("--tgt_len", type=int, default=DEFAULT_TGT_LEN, help="Max new tokens to generate")
    parser.add_argument("--total_samples", type=int, default=DEFAULT_TOTAL_SAMPLES, help="Total samples per dataset")
    parser.add_argument("--draft_temp", type=float, default=1.0, help="Temperature for drafter logits in sampling mode")
    args = parser.parse_args()

    bsz = args.bsz
    tgt_len = args.tgt_len
    total_samples = args.total_samples

    datasets = args.datasets.split(",") if args.datasets else DATASETS
    if args.greedy_only:
        greedy_modes = [True]
    elif args.sample_only:
        greedy_modes = [False]
    else:
        greedy_modes = [True, False]

    final_out = {}

    # --- Guided model ---
    print("=" * 60)
    print(f"Loading guided model from: {args.ckpt}")
    print("=" * 60)
    mod = load_from_ckpt(args.ckpt)
    mod.eval()
    mod.to("cuda")
    evaluate_model(mod, "guided", datasets, greedy_modes, SEEDS, final_out, bsz=bsz, tgt_len=tgt_len, total_samples=total_samples, draft_temperature=args.draft_temp)

    # --- Baseline (optional) ---
    if args.baseline:
        print("\n" + "=" * 60)
        print("Loading unguided baseline...")
        print("=" * 60)
        hp = dict(mod.hparams)
        del mod
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        baseline = load_baseline(hp)
        baseline.eval()
        baseline.to("cuda")
        evaluate_model(baseline, "baseline", datasets, greedy_modes, SEEDS, final_out, bsz=bsz, tgt_len=tgt_len, total_samples=total_samples, draft_temperature=args.draft_temp)
        del baseline

    # --- Save results ---
    import os
    os.makedirs(os.path.dirname(args.out_file) or ".", exist_ok=True)
    with open(args.out_file, "w") as f:
        json.dump(final_out, f, indent=2)
    print(f"\nResults saved to {args.out_file}")

    # --- Print summary ---
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    for name, data in final_out.items():
        if "outputs" not in data:
            continue
        print(f"  {name}: avg_accepted={data['avg_n_accepted']:.3f}, throughput={data['avg_throughput']:.1f}")


if __name__ == "__main__":
    main()
