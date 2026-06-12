# Environment & Paths

## Environment
- **Conda env**: `/HSC/users/qiaoye/envs/ssm_spec_py310` (Python 3.10)
- **Run python via**: `/HSC/users/qiaoye/envs/ssm_spec_py310/bin/python`
  (use the interpreter directly; `conda activate` may fail)

## Verifiers
- **LLaMA 3.1-8B**: `/HSC/users/qiaoye/checkpoints/Llama3.1-8B-hf`
- **LLaMA 3.2-1B**: `/HSC/users/qiaoye/checkpoints/Llama-3.2-1B`
- **LLaMA 3.2-3B**: `/HSC/users/qiaoye/checkpoints/Llama-3.2-3B`

## Common commands
```bash
# Evaluation
python -m spec_mamba.eval --ckpt <path> [--baseline] [--no_mask] [--greedy_only] [--bsz N]

# Cross-verify masking
python -m spec_mamba.cross_verify_mask
```

## Notes
- Both Mamba2 drafters and LLaMA verifiers share the same tokenizer (vocab_size=128256).
- `torch.compile` is incompatible with Mamba2; only compile the target (LLaMA) model.
- Mamba2 uses `cache_params` (not `past_key_values`) for incremental decoding.
- Pretrained Mamba2-65M is custom (trained on FineWeb-Edu), NOT an HF download.
