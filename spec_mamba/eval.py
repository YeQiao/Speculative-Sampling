"""
Evaluation script for spec_mamba (Mamba2 speculative decoding).

Usage:
    python -m spec_mamba.eval \
        --ckpt /path/to/last.ckpt \
        --total_samples 48 --bsz 4

    python -m spec_mamba.eval \
        --ckpt /path/to/last.ckpt --baseline --bsz 12

Metrics reported:
    - avg_n_accepted: Mean tokens accepted per speculative round
    - block_efficiency: avg_n_accepted + 1 (tokens produced per round, SD² convention)
    - throughput: Tokens generated per second
    - speedup: throughput / AR_baseline_throughput (if --measure_ar_baseline)
"""

import argparse
import gc
import json
import random
import time

import torch
from datasets import load_dataset

from spec_mamba.trainer import SpecMambaTrainer

torch.set_float32_matmul_precision("high")
torch.set_grad_enabled(False)

DATASETS = ["ultrachat", "humaneval", "xsum", "alpaca", "gsm8k"]

DEFAULT_BSZ = 12
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


def load_from_ckpt(ckpt_path: str) -> SpecMambaTrainer:
    st = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = {k: v for k, v in st["hyper_parameters"].items() if k != "_instantiator"}
    mod = SpecMambaTrainer(**hp)
    mod.load_state_dict(st["state_dict"], strict=False)
    return mod


def load_baseline(hp: dict) -> SpecMambaTrainer:
    mod = SpecMambaTrainer(**hp)
    with torch.no_grad():
        for p in mod.latent_mod_prep.parameters():
            p.zero_()
    return mod


def measure_ar_throughput(
    mod: SpecMambaTrainer,
    datasets: list[str],
    bsz: int = 1,
    tgt_len: int = DEFAULT_TGT_LEN,
    total_samples: int = DEFAULT_TOTAL_SAMPLES,
) -> dict[str, float]:
    """Measure autoregressive (no spec dec) throughput of the verifier.

    Returns dict mapping dataset name to AR throughput (tok/s).
    """
    ar_results = {}
    print("\n  Measuring AR baseline throughput (verifier-only)...")
    for dataset in datasets:
        prompts = get_prompts(dataset, n=total_samples)
        total_tokens = 0
        total_time = 0.0

        for i in range(0, len(prompts), bsz):
            batch_prompts = prompts[i:i + bsz]
            try:
                input_ids, attention_mask = mod.prep_for_gen(batch_prompts)
                input_ids = input_ids.to("cuda")
                attention_mask = attention_mask.to("cuda")

                # Verifier prefill
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.time()

                from transformers import DynamicCache
                pkv = DynamicCache()
                v_out = mod.v_base.get_decoder()(
                    input_ids,
                    attention_mask=attention_mask,
                    past_key_values=pkv,
                    use_cache=True,
                )
                pkv = v_out.past_key_values
                next_tok = mod.v_base.lm_head(v_out.last_hidden_state[:, -1:]).argmax(dim=-1)

                # Generate tokens autoregressively
                cur_len = input_ids.shape[1]
                for step in range(tgt_len - 1):
                    cur_len += 1
                    new_mask = torch.ones(len(batch_prompts), cur_len, device="cuda", dtype=attention_mask.dtype)
                    v_out = mod.v_base.get_decoder()(
                        next_tok,
                        attention_mask=new_mask,
                        past_key_values=pkv,
                        use_cache=True,
                    )
                    pkv = v_out.past_key_values
                    next_tok = mod.v_base.lm_head(v_out.last_hidden_state[:, -1:]).argmax(dim=-1)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = time.time() - t0
                total_tokens += tgt_len * len(batch_prompts)
                total_time += elapsed

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"      AR OOM at batch {i}, skipping...")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                else:
                    raise

        ar_tps = total_tokens / max(total_time, 1e-6)
        ar_results[dataset] = ar_tps
        print(f"    {dataset}: AR throughput = {ar_tps:.1f} tok/s ({total_tokens} tokens in {total_time:.1f}s)")
    return ar_results


def evaluate_model(
    mod: SpecMambaTrainer,
    name: str,
    datasets: list[str],
    greedy_modes: list[bool],
    seeds: list[int],
    out: dict,
    bsz: int = DEFAULT_BSZ,
    tgt_len: int = DEFAULT_TGT_LEN,
    total_samples: int = DEFAULT_TOTAL_SAMPLES,
    draft_temperature: float = 1.0,
    mask_rejected: bool = True,
    use_activation_replay: bool = False,
    ar_throughput: dict[str, float] | None = None,
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

            for seed in this_seeds:
                torch.manual_seed(seed)
                for i in range(0, len(prompts), bsz):
                    batch_prompts = prompts[i:i + bsz]
                    try:
                        input_ids, attention_mask = mod.prep_for_gen(batch_prompts)
                        sampled, extra = mod.generate(
                            input_ids, attention_mask, max_new_tokens=tgt_len,
                            draft_temperature=1.0 if g else draft_temperature,
                            mask_rejected=mask_rejected,
                            use_activation_replay=use_activation_replay,
                        )
                        na = extra["n_accepted"].cpu().numpy()
                        throughput = (na + 1) / (extra["time_per_block"] / len(batch_prompts) + 1e-12)

                        for j in range(len(batch_prompts)):
                            out_mask = extra["attention_mask"]
                            if out_mask is not None:
                                tok_ids = sampled[j, out_mask[j].bool()].cpu()
                            else:
                                tok_ids = sampled[j][sampled[j] != mod.pad_token_id].cpu()
                            text = mod.tok.decode(tok_ids.numpy(), skip_special_tokens=True)
                            all_outputs.append({
                                "text": text,
                                "n_accepted": float(na[j]),
                                "throughput": float(throughput[j]),
                            })

                        total_na += na.sum()
                        total_throughput += throughput.sum()

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
            block_eff = avg_na + 1  # SD² convention: tokens produced per round

            result = {
                "avg_n_accepted": float(avg_na),
                "block_efficiency": float(block_eff),
                "avg_throughput": float(avg_throughput),
                "n_samples": n_samples,
                "outputs": all_outputs,
            }

            # Add speedup if AR baseline is available
            if ar_throughput and dataset in ar_throughput:
                ar_tps = ar_throughput[dataset]
                speedup = avg_throughput / ar_tps if ar_tps > 0 else 0.0
                result["ar_throughput"] = float(ar_tps)
                result["speedup"] = float(speedup)

            out[this_name] = result
            speedup_str = f", speedup={result.get('speedup', 0):.2f}x" if "speedup" in result else ""
            print(f"      avg accepted: {avg_na:.3f}, block_eff: {block_eff:.3f}, "
                  f"throughput: {avg_throughput:.1f} tok/s{speedup_str}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate spec_mamba speculative decoding")
    parser.add_argument("--ckpt", type=str, required=False, default=None)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--pretrained_drafter", type=str, default=None,
                        help="Path to a standalone HF Mamba2 checkpoint (no guidance). "
                             "Constructs SpecMambaTrainer with zeroed guidance.")
    parser.add_argument("--datasets", type=str, default=None)
    parser.add_argument("--out_file", type=str, default="spec_mamba/eval_results.json")
    parser.add_argument("--greedy_only", action="store_true")
    parser.add_argument("--sample_only", action="store_true")
    parser.add_argument("--bsz", type=int, default=DEFAULT_BSZ)
    parser.add_argument("--tgt_len", type=int, default=DEFAULT_TGT_LEN)
    parser.add_argument("--total_samples", type=int, default=DEFAULT_TOTAL_SAMPLES)
    parser.add_argument("--draft_temp", type=float, default=1.0)
    parser.add_argument("--no_mask", action="store_true",
                        help="Skip masking rejected positions (commonly reported setup)")
    parser.add_argument("--activation_replay", action="store_true",
                        help="Use activation replay instead of full re-prefill for drafter cache")
    parser.add_argument("--measure_ar_baseline", action="store_true",
                        help="Measure AR-only (verifier autoregressive) throughput for speedup calculation")
    args = parser.parse_args()

    datasets = args.datasets.split(",") if args.datasets else DATASETS
    if args.greedy_only:
        greedy_modes = [True]
    elif args.sample_only:
        greedy_modes = [False]
    else:
        greedy_modes = [True, False]

    if not args.ckpt and not args.pretrained_drafter:
        parser.error("Must specify either --ckpt or --pretrained_drafter")

    final_out = {}

    if args.pretrained_drafter:
        # Standalone pretrained Mamba2 (no guidance, no .ckpt)
        print("=" * 60)
        print(f"Loading pretrained drafter from: {args.pretrained_drafter}")
        print("=" * 60)
        mod = SpecMambaTrainer(
            verifier="/HSC/users/qiaoye/checkpoints/Llama3.1-8B-hf",
            drafter=args.pretrained_drafter,
        )
        with torch.no_grad():
            for p in mod.latent_mod_prep.parameters():
                p.zero_()
        mod.eval()
        mod.to("cuda")

        ar_throughput = None
        if args.measure_ar_baseline:
            ar_throughput = measure_ar_throughput(
                mod, datasets, bsz=1, tgt_len=args.tgt_len,
                total_samples=min(args.total_samples, 48),
            )
            final_out["ar_baseline"] = ar_throughput

        evaluate_model(
            mod, "pretrained", datasets, greedy_modes, SEEDS, final_out,
            bsz=args.bsz, tgt_len=args.tgt_len,
            total_samples=args.total_samples, draft_temperature=args.draft_temp,
            mask_rejected=not args.no_mask,
            use_activation_replay=args.activation_replay,
            ar_throughput=ar_throughput,
        )
        del mod
    else:
        print("=" * 60)
        print(f"Loading guided model from: {args.ckpt}")
        print("=" * 60)
        mod = load_from_ckpt(args.ckpt)
        mod.eval()
        mod.to("cuda")

        # Optionally measure AR baseline throughput first
        ar_throughput = None
        if args.measure_ar_baseline:
            ar_throughput = measure_ar_throughput(
                mod, datasets, bsz=1, tgt_len=args.tgt_len,
                total_samples=min(args.total_samples, 48),  # fewer samples for AR (slow)
            )
            final_out["ar_baseline"] = ar_throughput

        evaluate_model(
            mod, "guided", datasets, greedy_modes, SEEDS, final_out,
            bsz=args.bsz, tgt_len=args.tgt_len,
            total_samples=args.total_samples, draft_temperature=args.draft_temp,
            mask_rejected=not args.no_mask,
            use_activation_replay=args.activation_replay,
            ar_throughput=ar_throughput,
        )

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
            evaluate_model(
                baseline, "baseline", datasets, greedy_modes, SEEDS, final_out,
                bsz=args.bsz, tgt_len=args.tgt_len,
                total_samples=args.total_samples, draft_temperature=args.draft_temp,
                mask_rejected=not args.no_mask,
                use_activation_replay=args.activation_replay,
                ar_throughput=ar_throughput,
            )
            del baseline

    import os
    os.makedirs(os.path.dirname(args.out_file) or ".", exist_ok=True)
    with open(args.out_file, "w") as f:
        json.dump(final_out, f, indent=2)
    print(f"\nResults saved to {args.out_file}")

    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    if "ar_baseline" in final_out:
        print("  AR baseline throughput:")
        for ds, tps in final_out["ar_baseline"].items():
            print(f"    {ds}: {tps:.1f} tok/s")
        print()
    for name, data in final_out.items():
        if "outputs" not in data:
            continue
        speedup_str = f", speedup={data['speedup']:.2f}x" if "speedup" in data else ""
        print(f"  {name}: block_eff={data.get('block_efficiency', 0):.3f}, "
              f"accepted={data['avg_n_accepted']:.3f}, "
              f"throughput={data['avg_throughput']:.1f}{speedup_str}")


if __name__ == "__main__":
    main()
