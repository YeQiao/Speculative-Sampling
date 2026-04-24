---
description: "Project environment, model paths, and key facts for the SpecSSM speculative decoding project. Use when working with model loading, training configs, checkpoints, or environment setup."
applyTo: "**"
---

# SpecSSM Project Context

## Paper Plan
- **Master plan**: `paper/plan.md` — contains all work items, critical path, and paper structure
- **Always consult `paper/plan.md`** before starting new experiments or writing
- **Update `paper/plan.md`** when a work item is completed (change status)
- **Update `spec_mamba/ARCHITECTURE.md`** whenever new benchmark results are collected

## Environment
- **Conda env**: `/HSC/users/qiaoye/envs/ssm_spec_py310` (Python 3.10)
- **Run commands**: Use `/HSC/users/qiaoye/envs/ssm_spec_py310/bin/python` directly (conda activate may fail)

## Model Paths
- **Verifier (LLaMA 3.1-8B)**: `/HSC/users/qiaoye/checkpoints/Llama3.1-8B-hf`
- **Drafter (Mamba2-65M, pretrained on FineWeb-Edu, KD-finetuned)**: `/HSC/users/qiaoye/SSM_SPEC/checkpoints/improved-mamba-alignment/`
  - Best checkpoint: `checkpoint-750`
  - Other checkpoints: `checkpoint-{250,500,1000,1250,1500}`, `best_model`, `final_model`
- **Guided checkpoint (KD-trained with LLaMA-8B teacher)**: `/HSC/users/qiaoye/SSM_SPEC/checkpoints/guided_mamba/ckpts/last.ckpt`
- **Baseline transformer drafters**: `/HSC/users/qiaoye/checkpoints/Llama-3.2-1B`, `/HSC/users/qiaoye/checkpoints/Llama-3.2-3B`

## Key Architecture Facts
- Mamba2-65M: 16 layers, hidden_size=512, expand=2 (d_inner=1024), state_size=128, 16 heads, head_dim=64
- LLaMA 3.1-8B: 32 layers, hidden_size=4096
- Both models share the same tokenizer (vocab_size=128256)
- `torch.compile` is incompatible with Mamba2; only compile target (LLaMA) model
- Mamba2 uses `cache_params` (not `past_key_values`) for incremental decoding

## Workflow Rules
1. Before any experiment, check `paper/plan.md` for the relevant work item number
2. After completing an experiment, update both `plan.md` (status) and `ARCHITECTURE.md` (results)
3. Paper LaTeX lives at `paper/neurips2026-specssm/`
4. Evaluation script: `python -m spec_mamba.eval --ckpt <path> [--baseline] [--no_mask] [--greedy_only] [--bsz N]`
5. Cross-verify script: `python -m spec_mamba.cross_verify_mask`
