#!/bin/bash
# Pretrain Mamba2-45M drafter on FineWeb-Edu with 2x H100
# Usage: bash pretrain/launch.sh

set -e

export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=4

accelerate launch \
    --config_file pretrain/accelerate_config.yaml \
    -m pretrain.train \
    --config 45m \
    --data_path /HSC/users/qiaoye/SSM_SPEC/fineweb-edu-100BT \
    --tokenizer /HSC/users/qiaoye/checkpoints/Llama3.1-8B-hf \
    --output_dir /HSC/users/qiaoye/SSM_SPEC/checkpoints/mamba2-45m-pretrain \
    --batch_size 48 \
    --grad_accum 4 \
    --max_length 512 \
    --lr 6e-4 \
    --weight_decay 0.1 \
    --warmup_steps 2000 \
    --max_steps 100000 \
    --log_every 50 \
    --save_every 10000 \
    --seed 42 \
    "$@"
