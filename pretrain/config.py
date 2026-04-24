"""Model configurations for Mamba2 drafters."""

from transformers import Mamba2Config

LLAMA_VOCAB = 128_256
GEMMA_VOCAB = 262_144

# Current model (27.5M backbone)
MAMBA2_27M = Mamba2Config(
    vocab_size=LLAMA_VOCAB,
    hidden_size=512,
    num_hidden_layers=16,
    num_heads=16,
    head_dim=64,
    state_size=128,
    n_groups=1,
    expand=2,
    conv_kernel=4,
    layer_norm_epsilon=1e-5,
    hidden_act="silu",
    residual_in_fp32=True,
    rms_norm=True,
    rescale_prenorm_residual=False,
    time_step_rank="auto",
    time_step_min=0.001,
    time_step_max=0.1,
    time_step_floor=0.0001,
    initializer_range=0.1,
    use_bias=False,
    use_conv_bias=True,
    use_cache=True,
    chunk_size=256,
    tie_word_embeddings=True,
    pad_token_id=0,
    bos_token_id=0,
    eos_token_id=0,
)

# New model (47.6M backbone, ~50% of official 130M's 90.4M)
MAMBA2_45M = Mamba2Config(
    vocab_size=LLAMA_VOCAB,
    hidden_size=640,
    num_hidden_layers=18,
    num_heads=20,           # d_inner=1280, 1280/64=20
    head_dim=64,
    state_size=128,
    n_groups=1,
    expand=2,
    conv_kernel=4,
    layer_norm_epsilon=1e-5,
    hidden_act="silu",
    residual_in_fp32=True,
    rms_norm=True,
    rescale_prenorm_residual=False,
    time_step_rank="auto",
    time_step_min=0.001,
    time_step_max=0.1,
    time_step_floor=0.0001,
    initializer_range=0.1,
    use_bias=False,
    use_conv_bias=True,
    use_cache=True,
    chunk_size=256,
    tie_word_embeddings=True,
    pad_token_id=0,
    bos_token_id=0,
    eos_token_id=0,
)

CONFIGS = {
    "27m": MAMBA2_27M,
    "45m": MAMBA2_45M,
}

# ── Gemma-vocab variants (same backbone, different embedding) ─────

def _with_vocab(base_cfg: Mamba2Config, vocab: int) -> Mamba2Config:
    """Clone a config with a different vocab_size."""
    d = base_cfg.to_dict()
    d["vocab_size"] = vocab
    return Mamba2Config(**{k: v for k, v in d.items() if k != "transformers_version"})

MAMBA2_45M_GEMMA = _with_vocab(MAMBA2_45M, GEMMA_VOCAB)

CONFIGS["45m_gemma"] = MAMBA2_45M_GEMMA
