# Checkpoint Best Practices

## Always use best/optimal checkpoints, NOT "final"

### Mamba2-65M (KD-finetuned on FineWeb-Edu)
- **Best**: `/HSC/users/qiaoye/SSM_SPEC/checkpoints/improved-mamba-alignment/checkpoint-750`
- Has: `best_model`, `final_model`, `checkpoint-{250,500,750,1000,1250,1500}`

### Mamba2-45M (pretrained on FineWeb-Edu)
- **Best**: `/HSC/users/qiaoye/SSM_SPEC/checkpoints/mamba2-45m-pretrain/checkpoint-40000` (loss=3.129)
- Runner-up: `checkpoint-60000` (loss=3.130)
- Has: `checkpoint-{10000..100000 step 10000}`, `final`
- Cosine LR with restarts; loss is NOT monotonic. "final" (100k) has loss=3.233
- Loss progression: 10k=3.444, 20k=3.290, 30k=3.172, **40k=3.129**, 50k=3.150, 60k=3.130, 70k=3.143, 80k=3.262, 90k=3.247, 100k=3.233

### Mamba2-45M (KD-finetuned)
- Path: `/HSC/users/qiaoye/SSM_SPEC/checkpoints/mamba2-45m-kd/final`

## Guided checkpoints (.ckpt format, contain guidance + drafter weights)
**IMPORTANT**: `finetune_drafter=True` for ALL guided ckpts — drafter weights INSIDE the
.ckpt differ from the original base. Must extract drafter from .ckpt OR use the matching
base for unguided comparison.

### 65M guided (LLaMA-8B verifier, v_layers=[5,16,29]) — BEST OVERALL
- Best val_loss=0.3346: `guided_mamba/ckpts/step-step=31250-val_loss-val/loss=0.3346.ckpt`
- `last.ckpt` = step 31250
- Base drafter: `improved-mamba-alignment/checkpoint-750` (65M)
- Path: `/HSC/users/qiaoye/SSM_SPEC/checkpoints/guided_mamba/ckpts/`

### 65M guided (LLaMA-70B verifier, v_layers=[75])
- Best val_loss=0.3554: step=18750
- Path: `/HSC/users/qiaoye/SSM_SPEC/checkpoints/guided_mamba_70b/ckpts/`
- Base drafter: `improved-mamba-alignment/checkpoint-750`
- Still improving at last ckpt

### 45M guided from pretrain (LLaMA-8B, v_layers=[29])
- **BROKEN val_loss logging** (all zeros in filenames)
- Base drafter: `mamba2-45m-pretrain/final` (NOT best — final=100k, loss=3.233; best=40k, loss=3.129)
- Path: `/HSC/users/qiaoye/SSM_SPEC/checkpoints/mamba2-45m-guided-from-pretrain/ckpts/last-v1.ckpt`
- E7 eval: only 0.92/8 acceptance — poor

### 45M guided from KD (LLaMA-8B, v_layers=[29])
- Best val_loss=0.3974: step=37500
- Base drafter: `mamba2-45m-kd/final`
- Path: `/HSC/users/qiaoye/SSM_SPEC/checkpoints/mamba2-45m-guided/ckpts/`

### Guided sweep results (65M, LLaMA-8B, all at step=31250, all still improving)
| Config | val_loss | Notes |
|---|---|---|
| 3-layer [5,16,29] (guided_mamba) | 0.3346 | BEST |
| single_layer_16 | 0.3460 | |
| pair_16_29 | 0.3464 | |
| pretrained_guided_finetune_5_16_29 | 0.3466 | |
| pretrained_guided_5_16_29 | 0.3485 | |
| z_branch_5_16_29 | 0.3491 | steer_z=True |
| single_layer_29 | 0.3495 | ~same as pair configs |
| pair_5_29 | 0.3496 | |
| single_layer_5 | 0.4054 | worst single layer |

## Pipeline benchmark commands (use when GPUs are free)
```bash
# Guided pipeline — 65M drafter (best guided, val_loss=0.3346)
# Drafter weights extracted from ckpt (finetune_drafter=True)
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=16 \
python -m spec_mamba.pipeline_benchmark \
  --mode gpu_verify \
  --drafter /HSC/users/qiaoye/SSM_SPEC/checkpoints/improved-mamba-alignment/checkpoint-750 \
  --verifier /HSC/users/qiaoye/checkpoints/Llama3.1-8B-hf \
  --guided_ckpt /HSC/users/qiaoye/SSM_SPEC/checkpoints/guided_mamba/ckpts/last.ckpt \
  --ng 8 --tgt_len 128 --total_samples 16

# Unguided pipeline — 45M pretrained (best ckpt)
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=16 \
python -m spec_mamba.pipeline_benchmark \
  --mode gpu_verify \
  --drafter /HSC/users/qiaoye/SSM_SPEC/checkpoints/mamba2-45m-pretrain/checkpoint-40000 \
  --verifier /HSC/users/qiaoye/checkpoints/Llama3.1-8B-hf \
  --ng 8 --tgt_len 128 --total_samples 16
```

### CAVEAT: guided ckpts have finetune_drafter=True
The `--drafter` flag loads the BASE model for the CPU kernel. If guidance was trained with
`finetune_drafter=True`, the drafter weights inside the .ckpt diverge from the base.
TODO: add `--extract_drafter_from_ckpt` mode to `pipeline_benchmark.py`.
