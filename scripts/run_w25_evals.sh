#!/bin/bash
# W25: 70B pretrain + KD eval (n=96) — run after W24 finishes
set -e

PYTHON="/HSC/users/qiaoye/envs/ssm_spec_py310/bin/python"
WORKDIR="/HSC/users/qiaoye/SSM_SPEC/Speculative-Sampling"
VERIFIER_70B="/HSC/users/qiaoye/checkpoints/Llama-3.1-70B"
DRAFTER_PRETRAIN="/HSC/users/qiaoye/SSM_SPEC/checkpoints/improved-mamba-alignment/checkpoint-750"
OUTDIR="$WORKDIR/guided_mamba/paper_eval_results"

cd "$WORKDIR"

echo "============================================"
echo "  W25: 70B Pretrain + KD Eval (n=96)"
echo "  Started: $(date)"
echo "============================================"

# Wait for W24 to finish (if still running)
while pgrep -f "spec_mamba.eval.*W24" > /dev/null 2>&1; do
    echo "  $(date +%H:%M:%S) — W24 still running, waiting..."
    sleep 120
done

echo ""
echo "[W25a] 70B Pretrain Eval (n=96, greedy, masked)"
echo "  Drafter: $DRAFTER_PRETRAIN"
echo "  Verifier: $VERIFIER_70B"
echo "  Started: $(date)"
echo ""

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
echo "[W25a] Finished: $(date)"
echo ""

sleep 10

# W25b: Use the 70B guided checkpoint with --baseline to get zeroed-guidance eval
# This tests the drafter backbone trained WITH 70B guidance but with guidance removed
BEST_CKPT="/HSC/users/qiaoye/SSM_SPEC/checkpoints/guided_mamba_70b/ckpts/step-step=31250-val_loss-val/loss=0.3376.ckpt"

echo "[W25b] 70B KD (zeroed guide from 70B-guided ckpt) Eval (n=96, greedy, masked)"
echo "  Checkpoint: $BEST_CKPT"
echo "  Started: $(date)"
echo ""

$PYTHON -m spec_mamba.eval \
    --ckpt "$BEST_CKPT" \
    --baseline \
    --device_map auto \
    --bsz 1 \
    --total_samples 96 \
    --greedy_only \
    --out_file "$OUTDIR/W25b_70b_kd_zeroed_n96.json" \
    2>&1 | tee "$OUTDIR/W25b_70b_kd_zeroed_n96.log"

echo ""
echo "[W25b] Finished: $(date)"
echo ""
echo "ALL W25 EVALS COMPLETE: $(date)"
echo "Results:"
echo "  W25a: $OUTDIR/W25a_70b_pretrain_n96.json"
echo "  W25b: $OUTDIR/W25b_70b_kd_zeroed_n96.json"
