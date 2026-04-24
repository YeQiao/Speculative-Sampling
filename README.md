# SpecSSM: Speculative Decoding with Guided Tiny SSM Drafters

A **tiny Mamba2 drafter** (~50M backbone), guided by verifier hidden states, for speculative decoding with LLM verifiers. Supports LLaMA-3.1-8B and Gemma-4-E4B as verifiers.

## Key Idea

Instead of using a small transformer as the draft model, we use a Mamba2 SSM — orders of magnitude smaller (65M vs 1B+), with unique properties:
- **No KV cache** → constant memory regardless of sequence length
- **CPU-friendly** → recurrent single-step is fast on CPU (custom AVX-512 kernel)
- **Guided by verifier** → inject verifier hidden states into the SSM's state-update path to steer drafting

## Architecture

| Component | Details |
|-----------|---------|
| **Drafter** | Mamba2-45M (hidden=640, 18 layers, 47.6M backbone) |
| **Verifiers** | LLaMA-3.1-8B, Gemma-4-E4B |
| **Guidance** | Single verifier layer → linear proj → per-layer x-branch deltas |
| **Training** | Pretrain (FineWeb-Edu) → KD → Guided (UltraChat, TVD loss) |

## Project Structure

```
spec_mamba/          # Speculative decoding engine (eval, generation, profiling)
guided_mamba/        # Guided training module (Lightning)
pretrain/            # Pretraining, KD, and pipeline scripts
paper/               # NeurIPS 2026 LaTeX
```

## Quick Start

```bash
# Evaluate guided Mamba2 drafter with LLaMA-8B verifier
python -m spec_mamba.eval --ckpt <guided_checkpoint_path> --greedy_only --bsz 1

# Evaluate without rejection masking (inflated but comparable to prior work)
python -m spec_mamba.eval --ckpt <path> --no_mask --greedy_only

# AR baseline
python -m spec_mamba.eval --baseline --greedy_only
```

## Training Pipeline

Three-stage pipeline for each verifier family:

```bash
# Full LLaMA pipeline (~29h on 2x H100)
bash pretrain/pipeline.sh all

# Full Gemma pipeline (~29h on 2x H100)
bash pretrain/pipeline.sh gemma_all

# Or run stages individually
bash pretrain/pipeline.sh pretrain      # Stage 1: Pretrain on FineWeb-Edu
bash pretrain/pipeline.sh kd            # Stage 2: KD with LLaMA teacher
bash pretrain/pipeline.sh guide         # Stage 3: Guided training
```

## Main Results

**Guidance Layer Sweep** (Mamba2-65M, LLaMA-8B verifier, masked, greedy, bsz=1, 96 samples/dataset):

| Config | v_layers | Mean Accepted | Δ vs base |
|--------|----------|---------------|-----------|
| LLaMA-1B drafter | — | 3.39 | +83% |
| KD+Guided [5,16,29] | [5,16,29] | 2.33 | +25% |
| Single layer 29 | [29] | 2.30 | +24% |
| Single layer 16 | [16] | 2.29 | +24% |
| Pretrained+Guided (frozen) | [5,16,29] | 1.98 | +7% |
| Pretrained (no guidance) | — | 1.86 | — |

**Key findings:**
1. **Layer selection barely matters** — single layer 29 matches triple [5,16,29] (1.5% spread)
2. **Backbone finetuning is critical** — frozen (+7%) vs finetuned (+24%)
3. **z-branch adds nothing** — doubles parameters, no improvement

## Requirements

- Python 3.10+
- PyTorch 2.x with CUDA
- `transformers >= 5.6` (for Gemma-4 support)
- `mamba-ssm` + `causal-conv1d` (optional, for fused CUDA kernels)

## Citation

Paper in preparation for NeurIPS 2026.