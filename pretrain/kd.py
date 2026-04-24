"""
Knowledge Distillation for Mamba2-45M drafter using LLaMA-3.1-8B teacher.

Clean rewrite of the old improved_alignment_training.py with fixes:
- No hidden projector bug (removed; minimal benefit at weight=0.05)
- Pure KL + CE loss (simpler, equally effective for logit-level KD)
- Cosine schedule with warmup (was linear decay)
- Proper streaming data support (larger-scale KD)

Usage:
  # Single GPU
  python -m pretrain.kd --student_path <pretrained_45m_ckpt>

  # Multi-GPU (2x H100)
  accelerate launch --config_file pretrain/accelerate_config.yaml \
      -m pretrain.kd --student_path <pretrained_45m_ckpt>
"""

import argparse
import glob
import json
import os
import time

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Mamba2ForCausalLM,
    get_cosine_schedule_with_warmup,
)


# ── Dataset ─────────────────────────────────────────────────────────

class KDStreamDataset(IterableDataset):
    """Streaming dataset for KD. Yields tokenized text on the fly."""

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
                text, truncation=True, max_length=self.max_length, return_tensors="pt",
            )
            ids = enc["input_ids"].squeeze(0)
            mask = enc["attention_mask"].squeeze(0)
            if ids.numel() < 32:
                continue
            yield {"input_ids": ids, "attention_mask": mask}


def collate_fn(batch, pad_id: int = 0):
    max_len = max(b["input_ids"].size(0) for b in batch)
    ids, masks = [], []
    for b in batch:
        n = b["input_ids"].size(0)
        pad_len = max_len - n
        ids.append(torch.cat([b["input_ids"], torch.full((pad_len,), pad_id, dtype=torch.long)]))
        masks.append(torch.cat([b["attention_mask"], torch.zeros(pad_len, dtype=torch.long)]))
    return {"input_ids": torch.stack(ids), "attention_mask": torch.stack(masks)}


# ── Loss ────────────────────────────────────────────────────────────

def kd_loss(student_logits, teacher_logits, labels, attention_mask, temperature=4.0, alpha=0.9):
    """
    Combined KD loss: alpha * KL + (1-alpha) * CE.

    KL is computed on the full vocab (no top-k filtering) with temperature scaling.
    CE provides grounding to true next-token targets.
    """
    # Shift for next-token prediction
    s_logits = student_logits[:, :-1].contiguous()
    t_logits = teacher_logits[:, :-1].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].float()

    # KL divergence (forward KL: student learns teacher's distribution)
    s_log_probs = F.log_softmax(s_logits / temperature, dim=-1)
    t_probs = F.softmax(t_logits / temperature, dim=-1)
    kl = F.kl_div(s_log_probs, t_probs, reduction="none").sum(dim=-1)
    kl = (kl * shift_mask).sum() / shift_mask.sum() * (temperature ** 2)

    # Cross-entropy on true labels
    ce = F.cross_entropy(
        s_logits.view(-1, s_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=0,  # pad
        reduction="mean",
    )

    return alpha * kl + (1 - alpha) * ce, kl.detach(), ce.detach()


# ── Training ────────────────────────────────────────────────────────

def train(args):
    set_seed(args.seed)

    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=args.grad_accum,
    )

    # ── Models ──────────────────────────────────────────────────────
    accelerator.print("Loading teacher (LLaMA-3.1-8B)...")
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher_path, torch_dtype=torch.bfloat16,
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    accelerator.print(f"Loading student from {args.student_path}...")
    student = Mamba2ForCausalLM.from_pretrained(
        args.student_path, torch_dtype=torch.float32,
    )
    student.train()

    s_params = sum(p.numel() for p in student.parameters())
    accelerator.print(f"Student: {s_params/1e6:.1f}M params")

    # ── Tokenizer ───────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Dataset ─────────────────────────────────────────────────────
    data_path = args.data_path
    if os.path.isdir(data_path):
        pq = sorted(glob.glob(os.path.join(data_path, "**/*.parquet"), recursive=True))
        accelerator.print(f"Loading {len(pq)} parquet files")
        hf_stream = load_dataset("parquet", data_files=pq, split="train", streaming=True)
    else:
        accelerator.print(f"Streaming from {data_path}")
        hf_stream = load_dataset(data_path, split="train", streaming=True)

    hf_stream = hf_stream.shuffle(seed=args.seed, buffer_size=10_000)
    ds = KDStreamDataset(hf_stream, tokenizer, max_length=args.max_length)
    loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=4, pin_memory=True)

    # ── Optimizer ───────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay,
    )
    warmup = min(args.warmup_steps, args.max_steps // 10)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, args.max_steps)

    # ── Prepare ─────────────────────────────────────────────────────
    student, teacher, optimizer, loader, scheduler = accelerator.prepare(
        student, teacher, optimizer, loader, scheduler
    )

    # ── Loop ────────────────────────────────────────────────────────
    accelerator.print(f"KD training for {args.max_steps} steps  (temp={args.temperature}, alpha={args.alpha})")
    eff_bsz = args.batch_size * accelerator.num_processes * args.grad_accum
    accelerator.print(f"Effective batch size: {eff_bsz}")

    global_step = 0
    running_loss, running_kl, running_ce = 0.0, 0.0, 0.0
    t0 = time.time()
    data_iter = iter(loader)

    while global_step < args.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            accelerator.print(f"[step {global_step}] Data epoch ended, restarting...")
            data_iter = iter(loader)
            batch = next(data_iter)

        with accelerator.accumulate(student):
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]

            # Teacher forward (no grad, bf16)
            with torch.no_grad():
                t_out = teacher(input_ids=input_ids, attention_mask=attention_mask)

            # Student forward (fp32 backbone, bf16 mixed via accelerator)
            s_out = student(input_ids=input_ids)

            loss, kl_val, ce_val = kd_loss(
                s_out.logits, t_out.logits.float(),
                input_ids, attention_mask,
                temperature=args.temperature, alpha=args.alpha,
            )

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if accelerator.sync_gradients:
            global_step += 1
            running_loss += loss.detach().item()
            running_kl += kl_val.item()
            running_ce += ce_val.item()

            if global_step % args.log_every == 0:
                n = args.log_every
                elapsed = time.time() - t0
                accelerator.print(
                    f"step {global_step:>6d}/{args.max_steps} | "
                    f"loss {running_loss/n:.4f} (kl={running_kl/n:.4f} ce={running_ce/n:.4f}) | "
                    f"lr {scheduler.get_last_lr()[0]:.2e} | "
                    f"eta {(args.max_steps - global_step) / (global_step / elapsed) / 3600:.1f}h"
                )
                running_loss, running_kl, running_ce = 0.0, 0.0, 0.0

            if global_step % args.save_every == 0:
                save_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                accelerator.print(f"Saving → {save_dir}")
                accelerator.wait_for_everyone()
                unwrapped = accelerator.unwrap_model(student)
                unwrapped.save_pretrained(
                    save_dir,
                    is_main_process=accelerator.is_main_process,
                    save_function=accelerator.save,
                )
                if accelerator.is_main_process:
                    tokenizer.save_pretrained(save_dir)

    # ── Final save ──────────────────────────────────────────────────
    accelerator.wait_for_everyone()
    final_dir = os.path.join(args.output_dir, "final")
    unwrapped = accelerator.unwrap_model(student)
    unwrapped.save_pretrained(
        final_dir,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
    )
    if accelerator.is_main_process:
        tokenizer.save_pretrained(final_dir)
        with open(os.path.join(final_dir, "training_args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
    accelerator.print(f"Done. Final model → {final_dir}")


def main():
    p = argparse.ArgumentParser(description="KD for Mamba2 drafter")
    p.add_argument("--student_path", required=True, help="Pretrained Mamba2 checkpoint")
    p.add_argument("--teacher_path", default="/HSC/users/qiaoye/checkpoints/Llama3.1-8B-hf")
    p.add_argument("--data_path", default="/HSC/users/qiaoye/SSM_SPEC/fineweb-edu-100BT",
                    help="FineWeb-Edu (local or HF name)")
    p.add_argument("--output_dir", default="/HSC/users/qiaoye/SSM_SPEC/checkpoints/mamba2-45m-kd")
    p.add_argument("--batch_size", type=int, default=8, help="Per-GPU batch size")
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--max_steps", type=int, default=3000, help="Longer than old 1500 (was still improving)")
    p.add_argument("--temperature", type=float, default=4.0)
    p.add_argument("--alpha", type=float, default=0.9, help="KL weight (1-alpha = CE weight)")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
