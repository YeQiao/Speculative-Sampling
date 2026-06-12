#!/usr/bin/env bash
# Run guided GPU-CPU pipeline benchmarks on HumanEval + GSM8K.
# Matrix:
#   1) 27M-backbone guided drafter + LLaMA-8B verifier
#   2) 45M guided drafter + LLaMA-8B verifier
#   3) 27M-backbone guided drafter + LLaMA-70B verifier
# AR baseline is measured inside pipeline_benchmark.py for each run.

set -euo pipefail

ROOT=/HSC/users/qiaoye/SSM_SPEC/Speculative-Sampling
PY=/HSC/users/qiaoye/envs/ssm_spec_py310/bin/python
OUTDIR=$ROOT/outputs/pipeline_benchmark
LOGDIR=$OUTDIR/logs
TS=$(date +%Y%m%d_%H%M%S)

mkdir -p "$OUTDIR" "$LOGDIR"

DATASETS="humaneval,gsm8k"
TOTAL_SAMPLES=16
THREADS=16
NG=8
TGT_LEN=128

# 27M backbone (old 65M total) + 8B guided
DRAFTER_27M=/HSC/users/qiaoye/SSM_SPEC/checkpoints/improved-mamba-alignment/checkpoint-750
GUIDE_8B_27M=/HSC/users/qiaoye/SSM_SPEC/checkpoints/guided_mamba/ckpts/step-step=31250-val_loss-val/loss=0.3346.ckpt

# 45M guided + 8B
DRAFTER_45M=/HSC/users/qiaoye/SSM_SPEC/checkpoints/mamba2-45m-kd/final
GUIDE_8B_45M=/HSC/users/qiaoye/SSM_SPEC/checkpoints/mamba2-45m-guided/ckpts/step-step=37500-val_loss-val/loss=0.3974.ckpt

# 27M backbone guided + 70B
GUIDE_70B_27M=/HSC/users/qiaoye/SSM_SPEC/checkpoints/guided_mamba_70b/ckpts/step-step=31250-val_loss-val/loss=0.3376.ckpt
VERIFIER_8B=/HSC/users/qiaoye/checkpoints/Llama3.1-8B-hf
VERIFIER_70B=/HSC/users/qiaoye/checkpoints/Llama-3.1-70B

echo "============================================================"
echo "Pipeline benchmark batch started: $(date)"
echo "Datasets: $DATASETS, total_samples=$TOTAL_SAMPLES"
echo "============================================================"

run_case() {
  local name=$1
  shift
  echo ""
  echo "[$name] start: $(date)"
  OMP_NUM_THREADS=$THREADS CUDA_VISIBLE_DEVICES=0,1 \
  HF_DATASETS_OFFLINE=1 \
  "$PY" -m spec_mamba.pipeline_benchmark "$@" \
    2>&1 | tee "$LOGDIR/${name}_${TS}.log" || true
  echo "[$name] done: $(date)"
}

cd "$ROOT"

run_case "gpu_8b_guided_27m" \
  --mode gpu_verify \
  --drafter "$DRAFTER_27M" \
  --verifier "$VERIFIER_8B" \
  --guided_ckpt "$GUIDE_8B_27M" \
  --datasets "$DATASETS" \
  --ng "$NG" --tgt_len "$TGT_LEN" \
  --total_samples "$TOTAL_SAMPLES" --ar_samples "$TOTAL_SAMPLES" \
  --threads "$THREADS" \
  --out_file "$OUTDIR/benchmark_gpu_verify_guided_27m_8b_${TS}.json"

run_case "gpu_8b_guided_45m" \
  --mode gpu_verify \
  --drafter "$DRAFTER_45M" \
  --verifier "$VERIFIER_8B" \
  --guided_ckpt "$GUIDE_8B_45M" \
  --datasets "$DATASETS" \
  --ng "$NG" --tgt_len "$TGT_LEN" \
  --total_samples "$TOTAL_SAMPLES" --ar_samples "$TOTAL_SAMPLES" \
  --threads "$THREADS" \
  --out_file "$OUTDIR/benchmark_gpu_verify_guided_45m_8b_${TS}.json"

run_case "gpu_70b_guided_27m" \
  --mode gpu_verify \
  --drafter "$DRAFTER_27M" \
  --verifier "$VERIFIER_70B" \
  --device_map auto --quantize 8bit \
  --guided_ckpt "$GUIDE_70B_27M" \
  --datasets "$DATASETS" \
  --ng "$NG" --tgt_len "$TGT_LEN" \
  --total_samples "$TOTAL_SAMPLES" --ar_samples "$TOTAL_SAMPLES" \
  --threads "$THREADS" \
  --out_file "$OUTDIR/benchmark_gpu_verify_guided_27m_70b_${TS}.json"

echo ""
echo "============================================================"
echo "All pipeline jobs finished: $(date)"
echo "Output JSON files:"
ls -1 "$OUTDIR"/*"_${TS}.json"
echo "============================================================"
