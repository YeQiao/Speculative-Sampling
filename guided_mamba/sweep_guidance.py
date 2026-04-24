#!/usr/bin/env python3
"""
Sweep: z-branch guidance + verifier layer combinations.

Trains guidance configs in parallel on 2 GPUs, then evaluates each.
~19 min/epoch, 5 epochs for quick screen → ~1.5h per config.
Full training: 10 epochs → ~3h per config.

Usage:
    # Quick screen (5 epochs, all configs):
    python -m guided_mamba.sweep_guidance --phase screen

    # Full training on specific configs:
    python -m guided_mamba.sweep_guidance --phase full --configs 0 3 5

    # Evaluate trained checkpoints:
    python -m guided_mamba.sweep_guidance --phase eval --configs 0 3 5

    # List all configs:
    python -m guided_mamba.sweep_guidance --phase list
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PYTHON = "/HSC/users/qiaoye/envs/ssm_spec_py310/bin/python"
BASE_DIR = Path("/HSC/users/qiaoye/SSM_SPEC/Speculative-Sampling")
CKPT_ROOT = Path("/HSC/users/qiaoye/SSM_SPEC/checkpoints/guided_sweep")
CONFIG_PATH = BASE_DIR / "guided_mamba" / "config.yaml"

# ─── Sweep Configurations ─────────────────────────────────────────────
# Each config is a dict with:
#   name: human-readable name
#   v_layers: list of verifier layer indices
#   steer_z: whether to also steer gating branch
#   priority: 1=must-run, 2=informative, 3=nice-to-have
# Drafter paths
DRAFTER_KD = "/HSC/users/qiaoye/SSM_SPEC/checkpoints/improved-mamba-alignment/checkpoint-750"
DRAFTER_PRETRAINED = "/HSC/users/qiaoye/SSM_SPEC/checkpoints/custom-mamba-65m-multi-gpu"

SWEEP_CONFIGS = [
    # === Priority 0: Pretrained backbone + guidance (key ablation) ===
    {
        "name": "pretrained_guided_5_16_29",
        "v_layers": [5, 16, 29],
        "steer_z": False,
        "priority": 0,
        "drafter": DRAFTER_PRETRAINED,
        "finetune_drafter": False,
        "note": "Pretrained backbone + frozen + guidance only. Key ablation: isolates guidance from KD",
    },
    {
        "name": "pretrained_guided_finetune_5_16_29",
        "v_layers": [5, 16, 29],
        "steer_z": False,
        "priority": 0,
        "drafter": DRAFTER_PRETRAINED,
        "finetune_drafter": True,
        "note": "Pretrained backbone + finetune + guidance. Compare with frozen to measure backbone adaptation",
    },
    # === Priority 1: z-branch test (W2) ===
    {
        "name": "z_branch_5_16_29",
        "v_layers": [5, 16, 29],
        "steer_z": True,
        "priority": 1,
        "note": "W2: z-branch ablation, same layers as baseline",
    },
    # === Priority 1: Single-layer ablation (W16) ===
    {
        "name": "single_layer_29",
        "v_layers": [29],
        "steer_z": False,
        "priority": 1,
        "note": "High layer only — captures prediction-relevant features",
    },
    {
        "name": "single_layer_16",
        "v_layers": [16],
        "steer_z": False,
        "priority": 1,
        "note": "Mid layer only — captures semantic features",
    },
    {
        "name": "single_layer_5",
        "v_layers": [5],
        "steer_z": False,
        "priority": 1,
        "note": "Low layer only — captures surface/syntax features",
    },
    # === Priority 1: Pair ablation ===
    {
        "name": "pair_5_29",
        "v_layers": [5, 29],
        "steer_z": False,
        "priority": 1,
        "note": "Low+High pair (skip mid)",
    },
    {
        "name": "pair_16_29",
        "v_layers": [16, 29],
        "steer_z": False,
        "priority": 1,
        "note": "Mid+High pair (skip low)",
    },
    # === Priority 2: Alternative spreads ===
    {
        "name": "triple_8_16_24",
        "v_layers": [8, 16, 24],
        "steer_z": False,
        "priority": 2,
        "note": "Compressed/centered spread (8-layer gaps)",
    },
    {
        "name": "quad_4_11_22_29",
        "v_layers": [4, 11, 22, 29],
        "steer_z": False,
        "priority": 2,
        "note": "4 layers evenly spread",
    },
    # === Priority 2: z-branch with best pair (run after screen) ===
    {
        "name": "z_branch_16_29",
        "v_layers": [16, 29],
        "steer_z": True,
        "priority": 2,
        "note": "z-branch with mid+high pair",
    },
    # === Priority 3: Dense layers ===
    {
        "name": "dense_8_layers",
        "v_layers": [3, 7, 11, 15, 19, 23, 27, 31],
        "steer_z": False,
        "priority": 3,
        "note": "8 layers (every 4th) — test if more layers help",
    },
    {
        "name": "pair_5_16",
        "v_layers": [5, 16],
        "steer_z": False,
        "priority": 3,
        "note": "Low+Mid pair (skip high)",
    },
]


def config_dir(cfg):
    return CKPT_ROOT / cfg["name"]


def config_ckpt(cfg):
    return config_dir(cfg) / "ckpts" / "last.ckpt"


def list_configs(priority_filter=None):
    print(f"{'#':>3}  {'Pri':>4}  {'Name':<35}  {'v_layers':<20}  {'steer_z':<8}  {'ft_draft':<8}  Note")
    print("-" * 130)
    for i, cfg in enumerate(SWEEP_CONFIGS):
        if priority_filter is not None and cfg["priority"] > priority_filter:
            continue
        layers_str = str(cfg["v_layers"])
        status = "TRAINED" if config_ckpt(cfg).exists() else "pending"
        ft = str(cfg.get("finetune_drafter", "def"))
        print(
            f"{i:>3}  P{cfg['priority']:>3}  {cfg['name']:<35}  {layers_str:<20}  "
            f"{str(cfg['steer_z']):<8}  {ft:<8}  [{status}] {cfg.get('note', '')}"
        )


def build_train_cmd(cfg, max_epochs=10, gpu_id=0):
    """Build the training command for a single config."""
    cdir = config_dir(cfg)
    ckpt_dir = cdir / "ckpts"
    v_layers_str = json.dumps(cfg["v_layers"])

    cmd = [
        PYTHON, "-m", "guided_mamba.run", "fit",
        "--config", str(CONFIG_PATH),
        f"--model.v_layers={v_layers_str}",
        f"--model.steer_z={'true' if cfg['steer_z'] else 'false'}",
        f"--trainer.max_epochs={max_epochs}",
        f"--trainer.default_root_dir={cdir}",
        f"--trainer.callbacks=[{{class_path: lightning.pytorch.callbacks.ModelCheckpoint, "
        f"init_args: {{dirpath: '{ckpt_dir}', filename: 'step-{{step}}-val_loss-{{val/loss:.4f}}', "
        f"monitor: val/loss, mode: min, save_top_k: 3, every_n_epochs: 1, save_last: true}}}}]",
    ]
    # Override drafter path if specified
    if "drafter" in cfg:
        cmd.append(f"--model.drafter={cfg['drafter']}")
    # Override finetune_drafter if specified
    if "finetune_drafter" in cfg:
        cmd.append(f"--model.finetune_drafter={'true' if cfg['finetune_drafter'] else 'false'}")
    return cmd


def build_eval_cmd(cfg, gpu_id=0):
    """Build the evaluation command for a trained config."""
    ckpt = config_ckpt(cfg)
    out_file = config_dir(cfg) / "eval_results.json"
    log_file = config_dir(cfg) / "eval_log.txt"

    cmd = (
        f"cd {BASE_DIR} && CUDA_VISIBLE_DEVICES={gpu_id} {PYTHON} -m spec_mamba.eval "
        f"--ckpt {ckpt} "
        f"--greedy_only --bsz 1 --measure_ar_baseline "
        f"--out_file {out_file} "
        f"2>&1 | tee {log_file}"
    )
    return cmd


def run_parallel_pair(cmd0, cmd1, gpu0=0, gpu1=1):
    """Run two commands in parallel on different GPUs. Returns (proc0, proc1)."""
    env0 = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu0)}
    env1 = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu1)}

    print(f"\n  GPU {gpu0}: {' '.join(cmd0[:6])}...")
    print(f"  GPU {gpu1}: {' '.join(cmd1[:6])}...")

    p0 = subprocess.Popen(cmd0, env=env0, cwd=str(BASE_DIR))
    p1 = subprocess.Popen(cmd1, env=env1, cwd=str(BASE_DIR))
    return p0, p1


def run_screen(config_indices, max_epochs=5):
    """Quick screen: train configs in parallel batches of 2."""
    configs = [SWEEP_CONFIGS[i] for i in config_indices]
    # Pair up configs for parallel GPU runs
    batches = []
    for i in range(0, len(configs), 2):
        batch = configs[i : i + 2]
        batches.append(batch)

    print(f"\n{'='*60}")
    print(f"Quick screen: {len(configs)} configs, {len(batches)} batches, {max_epochs} epochs each")
    print(f"Estimated time: ~{len(batches) * 1.5:.1f} hours")
    print(f"{'='*60}")

    for batch_idx, batch in enumerate(batches):
        print(f"\n--- Batch {batch_idx + 1}/{len(batches)} ---")
        cmds = []
        for gpu_id, cfg in enumerate(batch):
            cmd = build_train_cmd(cfg, max_epochs=max_epochs, gpu_id=gpu_id)
            cmds.append(cmd)
            print(f"  GPU {gpu_id}: {cfg['name']} (v_layers={cfg['v_layers']}, steer_z={cfg['steer_z']})")

        if len(cmds) == 2:
            p0, p1 = run_parallel_pair(cmds[0], cmds[1])
            start = time.time()
            p0.wait()
            p1.wait()
            elapsed = time.time() - start
            print(f"  Batch {batch_idx + 1} done in {elapsed / 60:.1f} min")
            if p0.returncode != 0:
                print(f"  WARNING: GPU 0 ({batch[0]['name']}) exited with code {p0.returncode}")
            if p1.returncode != 0:
                print(f"  WARNING: GPU 1 ({batch[1]['name']}) exited with code {p1.returncode}")
        else:
            # Single config left over
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0"}
            p = subprocess.Popen(cmds[0], env=env, cwd=str(BASE_DIR))
            start = time.time()
            p.wait()
            elapsed = time.time() - start
            print(f"  Batch {batch_idx + 1} done in {elapsed / 60:.1f} min")

    print(f"\n{'='*60}")
    print("Screen complete! Run with --phase eval to evaluate.")
    print(f"{'='*60}")


def run_eval(config_indices):
    """Evaluate trained configs."""
    for i in config_indices:
        cfg = SWEEP_CONFIGS[i]
        ckpt = config_ckpt(cfg)
        if not ckpt.exists():
            print(f"  SKIP {cfg['name']}: no checkpoint at {ckpt}")
            continue
        print(f"\n  Evaluating: {cfg['name']} ...")
        cmd = build_eval_cmd(cfg, gpu_id=0)
        subprocess.run(cmd, shell=True, cwd=str(BASE_DIR))

    # Collect summary
    print(f"\n{'='*60}")
    print("SWEEP RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Config':<30}  {'v_layers':<20}  {'steer_z':<8}  {'mean_accept':>12}")
    print("-" * 80)
    for i in config_indices:
        cfg = SWEEP_CONFIGS[i]
        results_file = config_dir(cfg) / "eval_results.json"
        if not results_file.exists():
            print(f"{cfg['name']:<30}  {str(cfg['v_layers']):<20}  {str(cfg['steer_z']):<8}  {'N/A':>12}")
            continue
        with open(results_file) as f:
            results = json.load(f)
        # Average acceptance across datasets
        accepts = []
        for key, val in results.items():
            if isinstance(val, dict) and "accepted" in val:
                accepts.append(val["accepted"])
        mean_acc = sum(accepts) / len(accepts) if accepts else 0
        print(
            f"{cfg['name']:<30}  {str(cfg['v_layers']):<20}  "
            f"{str(cfg['steer_z']):<8}  {mean_acc:>12.3f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Guidance sweep: z-branch + layer combos")
    parser.add_argument(
        "--phase",
        choices=["list", "screen", "full", "eval"],
        required=True,
        help="list: show configs | screen: quick 5-epoch train | full: 10-epoch train | eval: evaluate",
    )
    parser.add_argument(
        "--configs",
        type=int,
        nargs="*",
        help="Config indices to run (default: all priority-1 for screen, specified for full/eval)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override max_epochs (default: 5 for screen, 10 for full)",
    )
    parser.add_argument(
        "--priority",
        type=int,
        default=None,
        help="Only include configs up to this priority level",
    )
    args = parser.parse_args()

    if args.phase == "list":
        list_configs(args.priority)
        return

    # Determine which configs to run
    if args.configs is not None:
        indices = args.configs
    elif args.priority:
        indices = [i for i, c in enumerate(SWEEP_CONFIGS) if c["priority"] <= args.priority]
    else:
        # Default: priority 1 for screen, all for list
        indices = [i for i, c in enumerate(SWEEP_CONFIGS) if c["priority"] <= 1]

    if args.phase == "screen":
        epochs = args.epochs or 5
        run_screen(indices, max_epochs=epochs)
    elif args.phase == "full":
        epochs = args.epochs or 10
        run_screen(indices, max_epochs=epochs)
    elif args.phase == "eval":
        run_eval(indices)


if __name__ == "__main__":
    main()
