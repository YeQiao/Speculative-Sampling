#!/bin/bash
# W24 + W25: 70B guided eval (n=96) and 70B pretrain/KD eval (n=96)
# Run inside tmux. Waits for 70B guided training to finish, then launches evals.
set -e

PYTHON="/HSC/users/qiaoye/envs/ssm_spec_py310/bin/python"
WORKDIR="/HSC/users/qiaoye/SSM_SPEC/Speculative-Sampling"
CKPT_DIR="/HSC/users/qiaoye/SSM_SPEC/checkpoints/guided_mamba_70b/ckpts"
VERIFIER_70B="/HSC/users/qiaoye/checkpoints/Llama-3.1-70B"
DRAFTER_PRETRAIN="/HSC/users/qiaoye/SSM_SPEC/checkpoints/improved-mamba-alignment/checkpoint-750"
OUTDIR="$WORKDIR/guided_mamba/paper_eval_results"

cd "$WORKDIR"

echo "============================================"
echo "  70B Eval Launcher — waiting for training"
echo "============================================"
echo "Started: $(date)"

# ---- Phase 0: Wait for 70B guided training to finish ----
echo ""
echo "[Phase 0] Waiting for 70B training process to finish..."
while pgrep -f "spec_mamba.run fit.*config_70b" > /dev/null 2>&1; do
    STEP=$(tail -1 /HSC/users/qiaoye/SSM_SPEC/checkpoints/guided_mamba_70b/lightning_logs/version_0/metrics.csv 2>/dev/null | cut -d',' -f2)
    echo "  $(date +%H:%M:%S) — training still running (step ~${STEP}/31250)..."
    sleep 120
done

echo ""
echo "=========================================="
echo "  Training finished! $(date)"
echo "=========================================="
sleep 10  # let filesystem sync

# ---- Phase 1: Find best checkpoint ----
echo ""
echo "[Phase 1] Finding best checkpoint..."

# List all step-* checkpoint dirs, pick the one with lowest val_loss in filename
BEST_CKPT=""
BEST_LOSS="999"
for dir in "$CKPT_DIR"/step-step=*; do
    if [ -d "$dir" ]; then
        # Extract val_loss from directory name: step-step=XXXXX-val_loss-val
        # The actual .ckpt file is inside the directory
        loss=$(echo "$dir" | grep -oP 'loss=\K[0-9.]+')
        if [ -n "$loss" ]; then
            # Compare as strings (works for same-length decimals)
            if python3 -c "exit(0 if $loss < $BEST_LOSS else 1)"; then
                BEST_LOSS="$loss"
                # Find the .ckpt file inside the directory
                ckpt_file=$(find "$dir" -name "*.ckpt" -type f | head -1)
                if [ -n "$ckpt_file" ]; then
                    BEST_CKPT="$ckpt_file"
                fi
            fi
        fi
    fi
done

# Fallback: if no step-* dirs found or parsing failed, use last.ckpt
if [ -z "$BEST_CKPT" ]; then
    BEST_CKPT="$CKPT_DIR/last.ckpt"
    echo "  Using fallback: last.ckpt"
else
    echo "  Best checkpoint: $BEST_CKPT (val_loss=$BEST_LOSS)"
fi

# Also print all available checkpoints
echo "  All checkpoints:"
ls -lt "$CKPT_DIR"/

# Print final val losses
echo ""
echo "  Validation losses per epoch:"
awk -F',' '$5 != ""' /HSC/users/qiaoye/SSM_SPEC/checkpoints/guided_mamba_70b/lightning_logs/version_0/metrics.csv

# ---- Phase 2: W24 — 70B Guided Eval (n=96) ----
echo ""
echo "=========================================="
echo "  [W24] 70B Guided Eval (n=96, greedy, masked)"
echo "  Checkpoint: $BEST_CKPT"
echo "  Started: $(date)"
echo "=========================================="

$PYTHON -m spec_mamba.eval \
    --ckpt "$BEST_CKPT" \
    --device_map auto \
    --bsz 1 \
    --total_samples 96 \
    --greedy_only \
    --measure_ar_baseline \
    --out_file "$OUTDIR/W24_70b_guided_n96.json" \
    2>&1 | tee "$OUTDIR/W24_70b_guided_n96.log"

echo ""
echo "  [W24] Finished: $(date)"

# Free GPU memory
sleep 5

# ---- Phase 3: W25a — 70B Pretrain Eval (n=96) ----
echo ""
echo "=========================================="
echo "  [W25a] 70B Pretrain Eval (n=96, greedy, masked)"
echo "  Drafter: $DRAFTER_PRETRAIN"
echo "  Verifier: $VERIFIER_70B"
echo "  Started: $(date)"
echo "=========================================="

$PYTHON -m spec_mamba.eval \
    --pretrained_drafter "$DRAFTER_PRETRAIN" \
    --verifier "$VERIFIER_70B" \
    --device_map auto \
    --bsz 1 \
    --total_samples 96 \
    --greedy_only \
    --out_file "$OUTDIR/W25a_70b_pretrain_n96.json" \
    2>&1 | tee "$OUTDIR/W25a_70b_pretrain_n96.log"

echo ""
echo "  [W25a] Finished: $(date)"

# Free GPU memory
sleep 5

# ---- Phase 4: W25b — 70B KD Eval (n=96) ----
# KD uses the same checkpoint-750 drafter with zeroed guidance (--pretrained_drafter)
# This is equivalent to what E10 did but with n=96 instead of n=12
echo ""
echo "=========================================="
echo "  [W25b] 70B KD (zeroed guide) Eval (n=96, greedy, masked)"
echo "  Drafter: $DRAFTER_PRETRAIN (with zeroed guidance)"
echo "  Verifier: $VERIFIER_70B"
echo "  Started: $(date)"
echo "=========================================="

$PYTHON -m spec_mamba.eval \
    --pretrained_drafter "$DRAFTER_PRETRAIN" \
    --verifier "$VERIFIER_70B" \
    --device_map auto \
    --bsz 1 \
    --total_samples 96 \
    --greedy_only \
    --out_file "$OUTDIR/W25b_70b_kd_zeroed_n96.json" \
    2>&1 | tee "$OUTDIR/W25b_70b_kd_zeroed_n96.log"

echo ""
echo "  [W25b] Finished: $(date)"

echo ""
echo "=========================================="
echo "  ALL 70B EVALS COMPLETE"
echo "  $(date)"
echo "=========================================="
echo ""
echo "Results:"
echo "  W24:  $OUTDIR/W24_70b_guided_n96.json"
echo "  W25a: $OUTDIR/W25a_70b_pretrain_n96.json"
echo "  W25b: $OUTDIR/W25b_70b_kd_zeroed_n96.json"
