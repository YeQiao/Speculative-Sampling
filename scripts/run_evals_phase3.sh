#!/usr/bin/env bash
# Phase 3: Evaluate newly-trained models
# Run AFTER training finishes. Finds best checkpoint automatically.
#
# Usage: CUDA_VISIBLE_DEVICES=0 bash scripts/run_evals_phase3.sh

set -e
cd /HSC/users/qiaoye/SSM_SPEC/Speculative-Sampling
PY=/HSC/users/qiaoye/envs/ssm_spec_py310/bin/python
OUTDIR=guided_mamba/paper_eval_results
mkdir -p $OUTDIR

COMMON="--greedy_only --bsz 1 --total_samples 96 --datasets humaneval,gsm8k,alpaca,ultrachat,xsum"

echo "============================================"
echo "Phase 3: Evaluating newly-trained models"
echo "Started at: $(date)"
echo "============================================"

# ---- E7: LLaMA + Pretrain→Guided ----
echo ""
echo "[E7] LLaMA + Pretrain→Guided ($(date))"
CKPT_DIR=/HSC/users/qiaoye/SSM_SPEC/checkpoints/mamba2-45m-guided-from-pretrain/ckpts
# Find the checkpoint with lowest val_loss in filename
BEST_CKPT=$(ls "$CKPT_DIR"/step-step=*-val_loss-*.ckpt 2>/dev/null | sort -t= -k4 -g | head -1)
if [ -z "$BEST_CKPT" ]; then
    echo "  WARNING: No val_loss checkpoint found, trying last.ckpt"
    BEST_CKPT="$CKPT_DIR/last.ckpt"
fi
echo "  Using: $BEST_CKPT"
$PY -m spec_mamba.eval \
    --ckpt "$BEST_CKPT" \
    --out_file $OUTDIR/E7_llama_pretrain_guided.json \
    $COMMON 2>&1 | tail -20
echo "[E7] Done at $(date)"

# ---- E8: Gemma + Guided-Mixed ----
echo ""
echo "[E8] Gemma + Guided-Mixed ($(date))"
CKPT_DIR=/HSC/users/qiaoye/SSM_SPEC/checkpoints/mamba2-45m-gemma-guided-mixed/ckpts
BEST_CKPT=$(ls "$CKPT_DIR"/step-step=*-val_loss-*.ckpt 2>/dev/null | sort -t= -k4 -g | head -1)
if [ -z "$BEST_CKPT" ]; then
    echo "  WARNING: No val_loss checkpoint found, trying last.ckpt"
    BEST_CKPT="$CKPT_DIR/last.ckpt"
fi
echo "  Using: $BEST_CKPT"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
$PY -m spec_mamba.eval \
    --ckpt "$BEST_CKPT" \
    --out_file $OUTDIR/E8_gemma_guided_mixed.json \
    $COMMON 2>&1 | tail -20
echo "[E8] Done at $(date)"

echo ""
echo "============================================"
echo "Phase 3 complete at: $(date)"
echo "All results:"
echo "============================================"
ls -la $OUTDIR/
