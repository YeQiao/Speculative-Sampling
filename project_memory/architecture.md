# Architecture Facts

## Model dimensions
- **Mamba2-65M**: 16 layers, hidden_size=512, expand=2 (d_inner=1024), state_size=128,
  16 heads, head_dim=64
- **Mamba2-45M**: 18 layers, hidden=640, n_heads=20
- **LLaMA 3.1-8B**: 32 layers, hidden_size=4096
- Both Mamba and LLaMA share the same tokenizer (vocab_size=128256)

## Model naming convention (IMPORTANT)
- "65M" and "45M" refer to **pure Mamba backbone params** (LM-head agnostic).
- The Mamba2-45M backbone is actually ~2x the size of the 65M backbone (more layers/hidden).
  - 45M: 18 layers, hidden=640, n_heads=20 — total model larger than 65M
  - 65M: 16 layers, hidden=512, n_heads=16
- Both share the same LM head (vocab 128256), which dominates total param count.

## Compatibility notes
- `torch.compile` is incompatible with Mamba2; only compile the target (LLaMA) model.
- Mamba2 uses `cache_params` (not `past_key_values`) for incremental decoding.
