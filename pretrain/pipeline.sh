#!/bin/bash
# Full Mamba2-45M training pipeline: Pretrain → KD → Guidance
# Supports both LLaMA-3.1-8B and Gemma-4-E4B verifiers.
#
# Usage:
#   bash pretrain/pipeline.sh pretrain       # LLaMA: pretrain
#   bash pretrain/pipeline.sh kd             # LLaMA: KD
#   bash pretrain/pipeline.sh guide          # LLaMA: guided training
#   bash pretrain/pipeline.sh all            # LLaMA: full pipeline
#
#   bash pretrain/pipeline.sh gemma_pretrain # Gemma: pretrain
#   bash pretrain/pipeline.sh gemma_kd       # Gemma: KD
#   bash pretrain/pipeline.sh gemma_data     # Gemma: prepare UltraChat data
#   bash pretrain/pipeline.sh gemma_guide    # Gemma: guided training
#   bash pretrain/pipeline.sh gemma_all      # Gemma: full pipeline

set -e

PYTHON=/HSC/users/qiaoye/envs/ssm_spec_py310/bin/python
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=4

STAGE=${1:-all}

# =====================================================================
#  LLaMA-3.1-8B  verifier pipeline
# =====================================================================

# ── Stage 1: Pretrain ──────────────────────────────────────────────
# ~27 hours on 2x H100 (100K steps)
run_pretrain() {
    echo "=== Stage 1: Pretraining Mamba2-45M on FineWeb-Edu ==="
    accelerate launch \
        --config_file pretrain/accelerate_config.yaml \
        -m pretrain.train \
        --config 45m \
        --data_path /HSC/users/qiaoye/SSM_SPEC/fineweb-edu-100BT \
        --output_dir /HSC/users/qiaoye/SSM_SPEC/checkpoints/mamba2-45m-pretrain \
        --batch_size 48 \
        --grad_accum 4 \
        --max_length 512 \
        --lr 6e-4 \
        --warmup_steps 2000 \
        --max_steps 100000 \
        --save_every 10000
}

# ── Stage 2: Knowledge Distillation ───────────────────────────────
# ~1.2 hours on 2x H100 (3000 steps)
run_kd() {
    echo "=== Stage 2: KD with LLaMA-3.1-8B teacher ==="
    STUDENT=/HSC/users/qiaoye/SSM_SPEC/checkpoints/mamba2-45m-pretrain/final
    if [ ! -d "$STUDENT" ]; then
        echo "ERROR: Pretrained model not found at $STUDENT. Run stage 1 first."
        exit 1
    fi
    accelerate launch \
        --config_file pretrain/accelerate_config.yaml \
        -m pretrain.kd \
        --student_path "$STUDENT" \
        --teacher_path /HSC/users/qiaoye/checkpoints/Llama3.1-8B-hf \
        --data_path /HSC/users/qiaoye/SSM_SPEC/fineweb-edu-100BT \
        --output_dir /HSC/users/qiaoye/SSM_SPEC/checkpoints/mamba2-45m-kd \
        --batch_size 8 \
        --grad_accum 4 \
        --max_length 512 \
        --lr 3e-5 \
        --temperature 4.0 \
        --alpha 0.9 \
        --warmup_steps 200 \
        --max_steps 3000 \
        --save_every 500
}

# ── Stage 3: Guided Training ──────────────────────────────────────
# ~20 min on 1x H100 (10 epochs, ~50K samples)
run_guide() {
    echo "=== Stage 3: Guided training with LLaMA verifier ==="
    KD_CKPT=/HSC/users/qiaoye/SSM_SPEC/checkpoints/mamba2-45m-kd/final
    if [ ! -d "$KD_CKPT" ]; then
        echo "ERROR: KD model not found at $KD_CKPT. Run stage 2 first."
        exit 1
    fi
    $PYTHON -m guided_mamba.run fit --config pretrain/config_guide_45m.yaml
}

# =====================================================================
#  Gemma-4-E4B  verifier pipeline
# =====================================================================
# Same Mamba2-45M backbone (47.6M), different embedding (262K vocab)
# Gemma-4-E4B: hidden=2560, 42 layers, global attn at [5,11,17,23,29,35,41]

GEMMA_VERIFIER=google/gemma-4-E4B-it

# ── Gemma Stage 1: Pretrain ───────────────────────────────────────
# ~27 hours on 2x H100 (100K steps, same as LLaMA pipeline)
run_gemma_pretrain() {
    echo "=== Gemma Stage 1: Pretraining Mamba2-45M (Gemma vocab) on FineWeb-Edu ==="
    accelerate launch \
        --config_file pretrain/accelerate_config.yaml \
        -m pretrain.train \
        --config 45m_gemma \
        --data_path /HSC/users/qiaoye/SSM_SPEC/fineweb-edu-100BT \
        --tokenizer $GEMMA_VERIFIER \
        --output_dir /HSC/users/qiaoye/SSM_SPEC/checkpoints/mamba2-45m-gemma-pretrain \
        --batch_size 48 \
        --grad_accum 4 \
        --max_length 512 \
        --lr 6e-4 \
        --warmup_steps 2000 \
        --max_steps 100000 \
        --save_every 10000
}

# ── Gemma Stage 2: Knowledge Distillation ─────────────────────────
# ~1.2 hours on 2x H100 (3000 steps)
run_gemma_kd() {
    echo "=== Gemma Stage 2: KD with Gemma-4-E4B teacher ==="
    STUDENT=/HSC/users/qiaoye/SSM_SPEC/checkpoints/mamba2-45m-gemma-pretrain/final
    if [ ! -d "$STUDENT" ]; then
        echo "ERROR: Pretrained model not found at $STUDENT. Run gemma_pretrain first."
        exit 1
    fi
    accelerate launch \
        --config_file pretrain/accelerate_config.yaml \
        -m pretrain.kd \
        --student_path "$STUDENT" \
        --teacher_path $GEMMA_VERIFIER \
        --data_path /HSC/users/qiaoye/SSM_SPEC/fineweb-edu-100BT \
        --output_dir /HSC/users/qiaoye/SSM_SPEC/checkpoints/mamba2-45m-gemma-kd \
        --batch_size 8 \
        --grad_accum 4 \
        --max_length 512 \
        --lr 3e-5 \
        --temperature 4.0 \
        --alpha 0.9 \
        --warmup_steps 200 \
        --max_steps 3000 \
        --save_every 500
}

# ── Gemma Stage 2.5: Prepare UltraChat data with Gemma tokenizer ─
run_gemma_data() {
    echo "=== Gemma Stage 2.5: Preparing UltraChat data with Gemma tokenizer ==="
    $PYTHON -m guided_mamba.prepare_data \
        --output /HSC/users/qiaoye/SSM_SPEC/data/ultrachat_guided_gemma \
        --tokenizer $GEMMA_VERIFIER \
        --max_length 512 \
        --n_train 50000 \
        --n_val 5000
}

# ── Gemma Stage 3: Guided Training ───────────────────────────────
# ~20 min on 1x H100 (10 epochs)
run_gemma_guide() {
    echo "=== Gemma Stage 3: Guided training with Gemma-4-E4B verifier ==="
    KD_CKPT=/HSC/users/qiaoye/SSM_SPEC/checkpoints/mamba2-45m-gemma-kd/final
    if [ ! -d "$KD_CKPT" ]; then
        echo "ERROR: KD model not found at $KD_CKPT. Run gemma_kd first."
        exit 1
    fi
    DATA=/HSC/users/qiaoye/SSM_SPEC/data/ultrachat_guided_gemma
    if [ ! -d "$DATA" ]; then
        echo "UltraChat (Gemma) not found. Preparing..."
        run_gemma_data
    fi
    $PYTHON -m guided_mamba.run fit --config pretrain/config_guide_45m_gemma.yaml
}

case "$STAGE" in
    # LLaMA pipeline
    pretrain) run_pretrain ;;
    kd)       run_kd ;;
    guide)    run_guide ;;
    all)      run_pretrain && run_kd && run_guide ;;
    # Gemma pipeline
    gemma_pretrain) run_gemma_pretrain ;;
    gemma_kd)       run_gemma_kd ;;
    gemma_data)     run_gemma_data ;;
    gemma_guide)    run_gemma_guide ;;
    gemma_all)      run_gemma_pretrain && run_gemma_kd && run_gemma_data && run_gemma_guide ;;
    # Both
    both)     run_pretrain && run_kd && run_guide && \
              run_gemma_pretrain && run_gemma_kd && run_gemma_data && run_gemma_guide ;;
    *)
        echo "Usage: $0 {pretrain|kd|guide|all|gemma_pretrain|gemma_kd|gemma_data|gemma_guide|gemma_all|both}"
        exit 1
        ;;
esac
