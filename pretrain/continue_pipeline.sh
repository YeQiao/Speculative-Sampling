#!/bin/bash
# Continuation script: Gemma pretrain (from ckpt-10000) → KD → guide
# Fixed: scheduler.step() bug that caused lr to decay 8x too fast
set -eo pipefail
cd /HSC/users/qiaoye/SSM_SPEC/Speculative-Sampling

LOG_DIR=/HSC/users/qiaoye/SSM_SPEC/logs
mkdir -p "$LOG_DIR"

echo "========================================"
echo "  Gemma pipeline restart (from ckpt-10000)"
echo "  $(date)"
echo "========================================"

# ── Gemma Stage 1: Pretrain (resume from checkpoint-10000) ──
echo ""
echo "=== [1/3] Gemma Pretrain (resume from ckpt-10000, fixed lr schedule) ==="
bash pretrain/pipeline.sh gemma_pretrain 2>&1 | tee "$LOG_DIR/gemma_pretrain_fixed.log"
echo "  Gemma pretrain done at $(date)"

# ── Gemma Stage 2: KD ──
echo ""
echo "=== [2/3] Gemma KD ==="
bash pretrain/pipeline.sh gemma_kd 2>&1 | tee "$LOG_DIR/gemma_kd.log"
echo "  Gemma KD done at $(date)"

# ── Gemma Stage 3: Guided Training ──
echo ""
echo "=== [3/3] Gemma Guided Training ==="
bash pretrain/pipeline.sh gemma_guide 2>&1 | tee "$LOG_DIR/gemma_guide.log"
echo "  Gemma guide done at $(date)"

echo ""
echo "========================================"
echo "  ALL GEMMA STAGES COMPLETE"
echo "  $(date)"
echo "========================================"
