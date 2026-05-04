"""
Pretrain Mamba2 drafter on FineWeb-Edu with HuggingFace Accelerate.

Usage:
  # Single GPU
  python -m pretrain.train --config 45m --batch_size 32

  # Multi-GPU (2x H100)
  accelerate launch --config_file pretrain/accelerate_config.yaml \
      -m pretrain.train --config 45m --batch_size 64
"""

import argparse
import glob
import json
import math
import os
import time

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer, Mamba2ForCausalLM, get_cosine_schedule_with_warmup

from .config import CONFIGS


# ── Streaming Dataset ───────────────────────────────────────────────

class FineWebStreamDataset(IterableDataset):
    """Tokenize-on-the-fly streaming dataset for FineWeb-Edu."""

    def __init__(self, hf_stream, tokenizer, max_length: int = 512):
        self.stream = hf_stream
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __iter__(self):
        for example in self.stream:
            text = example.get("text", "")
            if not text.strip():
                continue
            enc = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            ids = enc["input_ids"].squeeze(0)
            if ids.numel() < 16:     # skip very short docs
                continue
            yield {"input_ids": ids, "labels": ids.clone()}


def collate_fn(batch, pad_id: int = 0):
    max_len = max(b["input_ids"].size(0) for b in batch)
    ids, labels = [], []
    for b in batch:
        n = b["input_ids"].size(0)
        pad_len = max_len - n
        ids.append(torch.cat([b["input_ids"], torch.full((pad_len,), pad_id, dtype=torch.long)]))
        labels.append(torch.cat([b["labels"], torch.full((pad_len,), -100, dtype=torch.long)]))
    return {"input_ids": torch.stack(ids), "labels": torch.stack(labels)}


# ── Training ────────────────────────────────────────────────────────

def train(args):
    set_seed(args.seed)

    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=args.grad_accum,
        log_with="tensorboard",
        project_dir=os.path.join(args.output_dir, "logs"),
    )

    # ── Model ───────────────────────────────────────────────────────
    cfg = CONFIGS[args.config]
    if args.resume_from:
        accelerator.print(f"Resuming from {args.resume_from}")
        model = Mamba2ForCausalLM.from_pretrained(args.resume_from)
    else:
        model = Mamba2ForCausalLM(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    n_emb = cfg.vocab_size * cfg.hidden_size
    accelerator.print(
        f"Model: {args.config}  |  total={n_params/1e6:.1f}M  "
        f"backbone={( n_params - n_emb)/1e6:.1f}M  emb={n_emb/1e6:.1f}M"
    )

    # ── Tokenizer ───────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Dataset ─────────────────────────────────────────────────────
    data_path = args.data_path
    if os.path.isdir(data_path):
        pq = sorted(glob.glob(os.path.join(data_path, "**/*.parquet"), recursive=True))
        accelerator.print(f"Loading {len(pq)} parquet files from {data_path}")
        hf_stream = load_dataset("parquet", data_files=pq, split="train", streaming=True)
    else:
        accelerator.print(f"Streaming from HuggingFace: {data_path}")
        hf_stream = load_dataset(data_path, split="train", streaming=True)

    hf_stream = hf_stream.shuffle(seed=args.seed, buffer_size=10_000)
    ds = FineWebStreamDataset(hf_stream, tokenizer, max_length=args.max_length)
    loader = DataLoader(
        ds, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=4, pin_memory=True,
    )

    # ── Optimizer + Scheduler ───────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    total_steps = args.max_steps
    warmup_steps = min(args.warmup_steps, total_steps // 10)
    # AccelerateScheduler calls scheduler.step() num_processes times per optimizer
    # step (because it assumes dataloader batch was scaled by num_processes).
    # Compensate by scaling total_steps so the cosine cycle spans all training steps.
    sched_total = total_steps * accelerator.num_processes
    sched_warmup = warmup_steps * accelerator.num_processes
    scheduler = get_cosine_schedule_with_warmup(optimizer, sched_warmup, sched_total)

    # ── Accelerate prepare ──────────────────────────────────────────
    model, optimizer, loader, scheduler = accelerator.prepare(model, optimizer, loader, scheduler)

    # ── Training loop ───────────────────────────────────────────────
    accelerator.print(f"Training for {total_steps} steps  (warmup={warmup_steps})")
    accelerator.print(f"Batch/GPU={args.batch_size}  grad_accum={args.grad_accum}  "
                       f"effective_batch={args.batch_size * accelerator.num_processes * args.grad_accum}")

    # ── Resume from checkpoint ──────────────────────────────────────
    resume_step = 0
    if args.resume_from:
        state_file = os.path.join(args.resume_from, "training_state.json")
        if os.path.exists(state_file):
            with open(state_file) as f:
                state = json.load(f)
            resume_step = state.get("global_step", 0)
        else:
            # Infer step from checkpoint dir name (e.g. checkpoint-20000)
            basename = os.path.basename(args.resume_from.rstrip("/"))
            if basename.startswith("checkpoint-"):
                resume_step = int(basename.split("-")[1])
        if resume_step > 0:
            accelerator.print(f"Resuming from step {resume_step}, fast-forwarding scheduler...")
            for _ in range(resume_step):
                scheduler.step()

    global_step = resume_step
    running_loss = 0.0
    log_interval = args.log_every
    save_interval = args.save_every
    t0 = time.time()

    model.train()
    data_iter = iter(loader)

    while global_step < total_steps:
        # Fetch batch (restart iterator on epoch end)
        try:
            batch = next(data_iter)
        except StopIteration:
            accelerator.print(f"[step {global_step}] Dataset epoch ended, restarting...")
            data_iter = iter(loader)
            batch = next(data_iter)

        with accelerator.accumulate(model):
            outputs = model(input_ids=batch["input_ids"], labels=batch["labels"])
            loss = outputs.loss
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if accelerator.sync_gradients:
            global_step += 1
            running_loss += loss.detach().item()

            # ── Logging ─────────────────────────────────────────────
            if global_step % log_interval == 0:
                avg_loss = running_loss / log_interval
                elapsed = time.time() - t0
                steps_per_sec = global_step / elapsed
                lr_now = scheduler.get_last_lr()[0]
                tokens_per_sec = (
                    args.batch_size * accelerator.num_processes
                    * args.grad_accum * args.max_length * steps_per_sec
                )
                accelerator.print(
                    f"step {global_step:>7d}/{total_steps} | "
                    f"loss {avg_loss:.4f} | lr {lr_now:.2e} | "
                    f"{steps_per_sec:.2f} steps/s | "
                    f"{tokens_per_sec/1e6:.2f}M tok/s | "
                    f"eta {(total_steps - global_step) / steps_per_sec / 3600:.1f}h"
                )
                running_loss = 0.0

            # ── Checkpointing ───────────────────────────────────────
            if global_step % save_interval == 0:
                save_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                accelerator.print(f"Saving checkpoint to {save_dir}")
                accelerator.wait_for_everyone()
                unwrapped = accelerator.unwrap_model(model)
                unwrapped.save_pretrained(
                    save_dir,
                    is_main_process=accelerator.is_main_process,
                    save_function=accelerator.save,
                )
                if accelerator.is_main_process:
                    tokenizer.save_pretrained(save_dir)
                    with open(os.path.join(save_dir, "training_state.json"), "w") as f:
                        json.dump({"global_step": global_step, "loss": avg_loss, "lr": lr_now}, f)

    # ── Final save ──────────────────────────────────────────────────
    accelerator.wait_for_everyone()
    final_dir = os.path.join(args.output_dir, "final")
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.save_pretrained(
        final_dir,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
    )
    if accelerator.is_main_process:
        tokenizer.save_pretrained(final_dir)
        # Save config for reference
        with open(os.path.join(final_dir, "training_args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
    accelerator.print(f"Training complete. Final model saved to {final_dir}")


def main():
    parser = argparse.ArgumentParser(description="Pretrain Mamba2 drafter")
    parser.add_argument("--config", type=str, default="45m", choices=list(CONFIGS.keys()))
    parser.add_argument("--data_path", type=str,
                        default="/HSC/users/qiaoye/SSM_SPEC/fineweb-edu-100BT")
    parser.add_argument("--tokenizer", type=str,
                        default="/HSC/users/qiaoye/checkpoints/Llama3.1-8B-hf")
    parser.add_argument("--output_dir", type=str,
                        default="/HSC/users/qiaoye/SSM_SPEC/checkpoints/mamba2-45m-pretrain")
    parser.add_argument("--batch_size", type=int, default=48,
                        help="Per-GPU batch size")
    parser.add_argument("--grad_accum", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--lr", type=float, default=6e-4,
                        help="Peak learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--max_steps", type=int, default=100_000,
                        help="Total training steps")
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Resume from a checkpoint directory")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
