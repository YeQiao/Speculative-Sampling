#!/usr/bin/env bash
# Pipeline script: run all Phase 2 evals sequentially via spec_mamba/eval.py
# Designed to run alongside training on GPU 1 (which has ~55GB free)
# Each eval uses ~20GB and releases memory after completion.
#
# Usage: CUDA_VISIBLE_DEVICES=1 bash scripts/run_evals_phase2.sh

set -e
cd /HSC/users/qiaoye/SSM_SPEC/Speculative-Sampling
PY=/HSC/users/qiaoye/envs/ssm_spec_py310/bin/python
LLAMA=/HSC/users/qiaoye/checkpoints/Llama3.1-8B-hf
GEMMA=google/gemma-4-E4B-it
OUTDIR=guided_mamba/paper_eval_results
mkdir -p $OUTDIR

COMMON="--greedy_only --bsz 1 --total_samples 96 --datasets humaneval,gsm8k,alpaca,ultrachat,xsum"

echo "============================================"
echo "Phase 2: Evaluating ready-now models"
echo "Started at: $(date)"
echo "============================================"

# ---- E1: LLaMA + Pretrain drafter ----
echo ""
echo "[E1] LLaMA + Pretrain drafter ($(date))"
$PY -m spec_mamba.eval \
    --pretrained_drafter /HSC/users/qiaoye/SSM_SPEC/checkpoints/mamba2-45m-pretrain/final \
    --verifier $LLAMA \
    --out_file $OUTDIR/E1_llama_pretrain.json \
    $COMMON 2>&1 | tail -20
echo "[E1] Done at $(date)"

# ---- E2: LLaMA + KD drafter ----
echo ""
echo "[E2] LLaMA + KD drafter ($(date))"
$PY -m spec_mamba.eval \
    --pretrained_drafter /HSC/users/qiaoye/SSM_SPEC/checkpoints/mamba2-45m-kd/final \
    --verifier $LLAMA \
    --out_file $OUTDIR/E2_llama_kd.json \
    $COMMON 2>&1 | tail -20
echo "[E2] Done at $(date)"

# ---- E3: LLaMA + Guided (BEST ckpt, epoch 7) ----
echo ""
echo "[E3] LLaMA + Guided best ckpt ($(date))"
BEST_LLAMA_GUIDED=$(ls /HSC/users/qiaoye/SSM_SPEC/checkpoints/mamba2-45m-guided/ckpts/step-step=37500-val_loss-val* 2>/dev/null | head -1)
if [ -z "$BEST_LLAMA_GUIDED" ]; then
    echo "  WARNING: Best ckpt not found, falling back to last.ckpt"
    BEST_LLAMA_GUIDED=/HSC/users/qiaoye/SSM_SPEC/checkpoints/mamba2-45m-guided/ckpts/last.ckpt
fi
echo "  Using checkpoint: $BEST_LLAMA_GUIDED"
$PY -m spec_mamba.eval \
    --ckpt "$BEST_LLAMA_GUIDED" \
    --out_file $OUTDIR/E3_llama_guided_best.json \
    $COMMON 2>&1 | tail -20
echo "[E3] Done at $(date)"

# ---- E4: Gemma + Pretrain drafter ----
echo ""
echo "[E4] Gemma + Pretrain drafter ($(date))"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
$PY -m spec_mamba.eval \
    --pretrained_drafter /HSC/users/qiaoye/SSM_SPEC/checkpoints/mamba2-45m-gemma-pretrain/final \
    --verifier $GEMMA \
    --out_file $OUTDIR/E4_gemma_pretrain.json \
    $COMMON 2>&1 | tail -20
echo "[E4] Done at $(date)"

# ---- E5: Gemma + KD drafter ----
echo ""
echo "[E5] Gemma + KD drafter ($(date))"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
$PY -m spec_mamba.eval \
    --pretrained_drafter /HSC/users/qiaoye/SSM_SPEC/checkpoints/mamba2-45m-gemma-kd/final \
    --verifier $GEMMA \
    --out_file $OUTDIR/E5_gemma_kd.json \
    $COMMON 2>&1 | tail -20
echo "[E5] Done at $(date)"

# ---- E6: Gemma + Guided (best = last, epoch 9) ----
echo ""
echo "[E6] Gemma + Guided best ckpt ($(date))"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
$PY -m spec_mamba.eval \
    --ckpt /HSC/users/qiaoye/SSM_SPEC/checkpoints/mamba2-45m-gemma-guided/ckpts/last.ckpt \
    --out_file $OUTDIR/E6_gemma_guided_best.json \
    $COMMON 2>&1 | tail -20
echo "[E6] Done at $(date)"

echo ""
echo "============================================"
echo "Phase 2 complete at: $(date)"
echo "Results in: $OUTDIR/"
echo "============================================"
ls -la $OUTDIR/
