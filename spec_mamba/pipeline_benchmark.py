"""
Real-world pipeline benchmark: CPU Mamba2 drafter + GPU/CPU LLaMA verifier.

Implements actual speculative decoding with the INT8 fused CPU kernel and
measures wall-clock throughput vs autoregressive baseline.

Modes:
  1. gpu_verify  — CPU Mamba2 drafts, GPU LLaMA verifies (the target deployment)
  2. cpu_verify  — Both on CPU (edge deployment, very few samples)

Usage:
  # CPU-only (can run while GPUs are busy):
    python -m spec_mamba.pipeline_benchmark --mode cpu_verify --datasets humaneval,gsm8k --total_samples 4 --verifier /path/to/Llama3.1-8B-hf

  # GPU-CPU pipeline (unguided):
    python -m spec_mamba.pipeline_benchmark --mode gpu_verify --datasets humaneval,gsm8k --total_samples 48 --verifier /path/to/Llama3.1-8B-hf

  # GPU-CPU pipeline with guidance (optimal):
    python -m spec_mamba.pipeline_benchmark --mode gpu_verify --guided_ckpt /path/to/guided.ckpt --datasets humaneval,gsm8k --total_samples 48

  # 70B verifier (multi-GPU):
  python -m spec_mamba.pipeline_benchmark --mode gpu_verify --verifier /path/to/70B --device_map auto --total_samples 48

Notes:
  - CPU drafter uses Int8FusedCPUMamba2Model (AVX-512 VNNI, B=1 only)
    - Prompts are sampled from evaluation datasets (default: humaneval,gsm8k)
  - When --guided_ckpt is provided, guidance deltas are computed on GPU
    from verifier hidden states and transferred to CPU for drafter injection
  - Uses greedy decoding for deterministic benchmarks
  - Activation replay for drafter cache resync
"""

import argparse
import json
import os
import time
import threading
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

torch.set_grad_enabled(False)

# Defaults
DEFAULT_DRAFTER = "/HSC/users/qiaoye/SSM_SPEC/checkpoints/improved-mamba-alignment/checkpoint-750"
DEFAULT_VERIFIER_8B = "/HSC/users/qiaoye/checkpoints/Llama3.1-8B-hf"
DEFAULT_NG = 8
DEFAULT_TGT_LEN = 128
DEFAULT_TOTAL_SAMPLES = 48
DEFAULT_PROMPT_DATASETS = "humaneval,gsm8k"
PYTHON = "/HSC/users/qiaoye/envs/ssm_spec_py310/bin/python"


# =========================================================================
#  Guidance loading from checkpoint
# =========================================================================

def load_guidance_modules(ckpt_path: str, device: str = "cuda"):
    """Load GuidanceExtractor + PrepMambaDeltas from a guided checkpoint.

    Returns: (guidance_extractor, prep_deltas, v_layers, drafter_state_dict)
    The guidance modules are placed on `device` in eval mode.
    The drafter_state_dict can be used to build the HF Mamba2 (if needed).
    """
    from spec_mamba.trainer import GuidanceExtractor, PrepMambaDeltas

    print(f"Loading guidance weights from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    hp = ckpt.get("hyper_parameters", {})

    # Extract config from hyperparameters
    v_layers = hp.get("v_layers", [29])
    steer_z = hp.get("steer_z", False)
    d_layers = hp.get("d_layers", "all")

    # Infer v_h_dim from GuidanceExtractor proj weight shape
    ge_weight = sd["guidance_extractor.proj.weight"]  # [v_h_dim, n_v_layers * v_h_dim]
    v_h_dim = ge_weight.shape[0]

    # Infer drafter dims from PrepMambaDeltas proj weight
    pm_weight = sd["latent_mod_prep.proj.weight"]  # [delta_dim * n_layers, v_h_dim]
    total_out = pm_weight.shape[0]

    # Get n_layers from drafter state dict
    drafter_sd = {k.removeprefix("d_base."): v for k, v in sd.items() if k.startswith("d_base.")}
    # Key pattern: backbone.layers.X.mixer.m.in_proj.weight (GuidedMamba2Block wraps mixer as .m)
    # or backbone.layers.X.mixer.in_proj.weight (standard Mamba2)
    layer_keys = [k for k in drafter_sd if "in_proj.weight" in k and "backbone.layers." in k]
    n_drafter_layers = len(layer_keys)

    if d_layers == "all":
        n_guided_layers = n_drafter_layers
    else:
        n_guided_layers = len(d_layers)

    delta_dim = total_out // n_guided_layers

    print(f"  v_layers={v_layers}, steer_z={steer_z}, d_layers={d_layers}")
    print(f"  n_drafter_layers={n_drafter_layers}, n_guided_layers={n_guided_layers}, delta_dim={delta_dim}")
    print(f"  v_h_dim={v_h_dim}")

    # Build modules
    ge = GuidanceExtractor(v_h_dim=v_h_dim, n_layers=len(v_layers))
    ge.in_layer = [v + 1 for v in v_layers]  # off-by-one: hidden_states[v+1] = output of layer v
    ge.proj.weight.data.copy_(sd["guidance_extractor.proj.weight"])
    ge.proj.bias.data.copy_(sd["guidance_extractor.proj.bias"])

    prep = PrepMambaDeltas(v_h_dim=v_h_dim, delta_dim=delta_dim, n_layers=n_guided_layers)
    prep.norm.weight.data.copy_(sd["latent_mod_prep.norm.weight"])
    prep.norm.bias.data.copy_(sd["latent_mod_prep.norm.bias"])
    prep.proj.weight.data.copy_(sd["latent_mod_prep.proj.weight"])

    ge = ge.to(device).eval()
    prep = prep.to(device).eval()

    print(f"  GuidanceExtractor: {sum(p.numel() for p in ge.parameters())/1e6:.1f}M params on {device}")
    print(f"  PrepMambaDeltas:   {sum(p.numel() for p in prep.parameters())/1e6:.1f}M params on {device}")

    del ckpt  # free memory
    return ge, prep, v_layers, drafter_sd

def _get_dataset_prompts(dataset: str, n: int) -> list[str]:
    """Load prompts from evaluation datasets used in paper experiments."""
    if dataset == "humaneval":
        data = load_dataset("openai_humaneval", split="test")
        return list(data["prompt"][:n])
    if dataset == "gsm8k":
        data = load_dataset("gsm8k", "main", split="test")
        return list(data["question"][:n])
    raise ValueError(f"Unsupported benchmark dataset: {dataset}")


def collect_benchmark_prompts(dataset_names: list[str], total_samples: int) -> tuple[list[str], dict[str, int]]:
    """Collect prompts from datasets, balancing samples across sources."""
    if not dataset_names:
        raise ValueError("No datasets provided")

    base = total_samples // len(dataset_names)
    rem = total_samples % len(dataset_names)

    prompts = []
    counts = {}
    for i, ds_name in enumerate(dataset_names):
        take = base + (1 if i < rem else 0)
        if take <= 0:
            counts[ds_name] = 0
            continue
        ds_prompts = _get_dataset_prompts(ds_name, take)
        prompts.extend(ds_prompts)
        counts[ds_name] = len(ds_prompts)
    return prompts, counts


@dataclass
class RoundTiming:
    """Timing for a single speculative decoding round."""
    draft_ms: float = 0.0
    verify_ms: float = 0.0
    replay_ms: float = 0.0
    overhead_ms: float = 0.0
    total_ms: float = 0.0
    n_accepted: int = 0
    tokens_produced: int = 0


@dataclass
class BenchmarkResult:
    """Result for one generation (one prompt)."""
    prompt: str = ""
    total_tokens: int = 0
    total_time_ms: float = 0.0
    throughput_tps: float = 0.0
    avg_accepted: float = 0.0
    rounds: list = field(default_factory=list)
    # Breakdown
    total_draft_ms: float = 0.0
    total_verify_ms: float = 0.0
    total_replay_ms: float = 0.0
    # Pipeline stats
    spec_replay_hit_rate: float = 0.0


def load_cpu_drafter(drafter_path: str, use_int8: bool = True, guided_sd: dict = None):
    """Load Mamba2 drafter as Int8 or BF16 fused CPU model.

    Args:
        guided_sd: Optional dict of fine-tuned drafter weights from a guided checkpoint.
            Keys are in GuidedMamba2 format (backbone.layers.X.mixer.m.Y); these are
            remapped to standard HF format (backbone.layers.X.mixer.Y) before loading.
            This overrides the base checkpoint weights with fine-tuned weights.
    """
    print(f"Loading HF Mamba2 from {drafter_path}...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        drafter_path, torch_dtype=torch.float32,
    ).eval()

    # Disable fast path (causal_conv1d stride issue with 45M model)
    import transformers.models.mamba2.modeling_mamba2 as _m2
    _m2.is_fast_path_available = False

    # Apply fine-tuned weights from guided checkpoint if provided.
    # The guided checkpoint stores drafter weights with GuidedMamba2Block wrapping:
    #   backbone.layers.X.mixer.m.Y  →  backbone.layers.X.mixer.Y
    if guided_sd:
        print("Applying fine-tuned drafter weights from guided checkpoint...")
        hf_sd = {}
        for k, v in guided_sd.items():
            hf_key = k.replace(".mixer.m.", ".mixer.")
            hf_sd[hf_key] = v.float()
        missing, unexpected = hf_model.load_state_dict(hf_sd, strict=False)
        if missing:
            print(f"  Missing keys ({len(missing)}): {missing[:3]}")
        if unexpected:
            print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:3]}")
        else:
            print(f"  Loaded {len(hf_sd)} fine-tuned weight tensors successfully.")

    if use_int8:
        from spec_mamba.cpu_mamba2 import Int8FusedCPUMamba2Model
        print("Building Int8FusedCPUMamba2Model...")
        cpu_model = Int8FusedCPUMamba2Model(hf_model)
    else:
        from spec_mamba.cpu_mamba2 import FusedCPUMamba2Model
        print("Building FusedCPUMamba2Model (BF16)...")
        cpu_model = FusedCPUMamba2Model(hf_model)

    del hf_model
    return cpu_model


def load_verifier(verifier_path: str, device: str = "cuda",
                  device_map: str | None = None,
                  quantize: str | None = None):
    """Load LLaMA verifier on GPU or CPU."""
    kwargs = {}
    if device == "cuda":
        kwargs["torch_dtype"] = torch.float16
        if device_map:
            kwargs["device_map"] = device_map
            if quantize == "8bit":
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            kwargs["device_map"] = None
    else:
        # CPU: use float32 (BF16 matmul is slower on most CPUs for large models)
        kwargs["torch_dtype"] = torch.float32

    print(f"Loading verifier from {verifier_path} (device={device}, device_map={device_map})...")
    verifier = AutoModelForCausalLM.from_pretrained(verifier_path, **kwargs).eval()
    if device == "cuda" and not device_map:
        verifier = verifier.cuda()
    return verifier


def greedy_rejection(draft_tokens, draft_logits_list, v_logits, NG):
    """Greedy rejection: accept while draft == verifier argmax.

    Args:
        draft_tokens: [NG] tensor of drafted token ids
        draft_logits_list: list of NG tensors, each [vocab_size]
        v_logits: [1, NG+1, vocab_size] verifier logits

    Returns:
        n_accepted: int
        next_token: int (token to continue from)
    """
    v_preds = v_logits[0].argmax(dim=-1)  # [NG+1]
    n_accepted = 0
    for j in range(NG):
        if draft_tokens[j].item() == v_preds[j].item():
            n_accepted += 1
        else:
            break
    # Next token: verifier's prediction at the acceptance boundary
    next_token = v_preds[n_accepted].item()
    return n_accepted, next_token


def stochastic_rejection(draft_tokens, draft_probs_list, v_logits, NG):
    """Standard stochastic rejection sampling (SD² algorithm).

    Args:
        draft_tokens: [NG] tensor of drafted token ids
        draft_probs_list: list of NG tensors, each [vocab_size] (softmax)
        v_logits: [1, NG+1, vocab_size] verifier logits

    Returns:
        n_accepted: int
        next_token: int
    """
    v_probs = v_logits[0].softmax(dim=-1)  # [NG+1, V]
    n_accepted = 0
    for j in range(NG):
        tok_j = draft_tokens[j].item()
        q_j = draft_probs_list[j][tok_j].item()
        p_j = v_probs[j, tok_j].item()
        r = torch.rand(1).item()
        if r < min(1, p_j / (q_j + 1e-12)):
            n_accepted += 1
        else:
            break
    # Next token from adjusted distribution
    if n_accepted < NG:
        # Sample from max(0, p - q)
        residual = (v_probs[n_accepted] - draft_probs_list[n_accepted]).clamp(min=0)
        residual = residual / (residual.sum() + 1e-12)
        next_token = torch.multinomial(residual, 1).item()
    else:
        # All accepted, sample from verifier at position NG
        next_token = torch.multinomial(v_probs[NG], 1).item()
    return n_accepted, next_token


# =========================================================================
#  Core: Speculative decoding with CPU drafter + verifier
# =========================================================================

def spec_dec_generate(
    cpu_drafter,
    verifier,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    NG: int = 8,
    device: str = "cuda",
    greedy: bool = True,
    measure_components: bool = True,
) -> BenchmarkResult:
    """Run one generation with CPU-offloaded speculative decoding.

    The CPU drafter generates K tokens, then the verifier (GPU or CPU)
    checks them in parallel. We use activation replay to resync the
    drafter cache.
    """
    result = BenchmarkResult(prompt=prompt)

    # Tokenize
    if tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt}]
        tok_out = tokenizer.apply_chat_template(
            messages, return_tensors="pt",
            add_generation_prompt=True,
            return_dict=True,
        )
        input_ids = tok_out["input_ids"]
    else:
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]

    S = input_ids.shape[1]

    # Figure out verifier device
    v_device = next(verifier.parameters()).device

    # ---- Verifier prefill ----
    input_ids_v = input_ids.to(v_device)
    v_pkv = DynamicCache()
    if v_device.type == "cuda":
        torch.cuda.synchronize()

    v_out = verifier.model(
        input_ids_v, past_key_values=v_pkv, use_cache=True,
    )
    v_pkv = v_out.past_key_values
    v_last_logits = verifier.lm_head(v_out.last_hidden_state[:, -1:])
    first_token = v_last_logits.argmax(dim=-1).item()

    if v_device.type == "cuda":
        torch.cuda.synchronize()

    # ---- CPU drafter prefill ----
    conv_states, ssm_states = cpu_drafter.create_cache(batch_size=1)
    cpu_drafter.prefill(input_ids.cpu(), conv_states, ssm_states)

    # State tracking
    generated_tokens = [first_token]
    current_token = first_token
    total_draft_ms = 0.0
    total_verify_ms = 0.0
    total_replay_ms = 0.0
    all_na = []

    gen_start = time.perf_counter()

    while len(generated_tokens) < max_new_tokens:
        round_timing = RoundTiming()
        tokens_left = max_new_tokens - len(generated_tokens)
        K = min(NG, tokens_left)

        # --- Snapshot drafter cache ---
        snap_conv = conv_states.clone()
        snap_ssm = ssm_states.clone()

        # --- CPU Draft K tokens ---
        t0 = time.perf_counter()
        draft_tokens = []
        draft_probs = []
        tok_id = current_token
        for _ in range(K):
            logits = cpu_drafter.forward_step(tok_id, conv_states, ssm_states)
            probs = logits.softmax(dim=-1).squeeze(0)  # [vocab_size]
            if greedy:
                next_t = logits.argmax(dim=-1).item()
            else:
                next_t = torch.multinomial(probs, 1).item()
            draft_tokens.append(next_t)
            draft_probs.append(probs)
            tok_id = next_t
        t1 = time.perf_counter()
        round_timing.draft_ms = (t1 - t0) * 1000

        # --- Build verify input: [current_token] + draft_tokens ---
        verify_input = torch.tensor(
            [[current_token] + draft_tokens], dtype=torch.long, device=v_device,
        )

        # --- Verify on verifier ---
        if v_device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        v_out = verifier.model(
            verify_input, past_key_values=v_pkv, use_cache=True,
        )
        v_logits = verifier.lm_head(v_out.last_hidden_state)
        if v_device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        round_timing.verify_ms = (t1 - t0) * 1000

        # --- Rejection ---
        draft_tokens_t = torch.tensor(draft_tokens)
        if greedy:
            n_accepted, next_token = greedy_rejection(
                draft_tokens_t, draft_probs, v_logits.cpu(), K,
            )
        else:
            n_accepted, next_token = stochastic_rejection(
                draft_tokens_t, draft_probs, v_logits.cpu(), K,
            )

        round_timing.n_accepted = n_accepted
        round_timing.tokens_produced = n_accepted + 1

        # Collect accepted tokens + next_token
        accepted = draft_tokens[:n_accepted] + [next_token]
        generated_tokens.extend(accepted)

        # --- Drafter cache: activation replay ---
        t0 = time.perf_counter()
        conv_states.copy_(snap_conv)
        ssm_states.copy_(snap_ssm)
        for tok_id in accepted:
            cpu_drafter.forward_step(tok_id, conv_states, ssm_states)
        t1 = time.perf_counter()
        round_timing.replay_ms = (t1 - t0) * 1000

        # --- Verifier KV cache: crop to keep only accepted ---
        # We fed K+1 tokens. We need to keep n_accepted+1 of them (including
        # the first token which was already in context). So crop to remove
        # the rejected KV entries.
        n_to_remove = K - n_accepted
        if n_to_remove > 0:
            # DynamicCache: crop removes the last n entries
            new_len = v_pkv.get_seq_length() - n_to_remove
            v_pkv.crop(new_len)

        current_token = next_token
        all_na.append(n_accepted)

        total_draft_ms += round_timing.draft_ms
        total_verify_ms += round_timing.verify_ms
        total_replay_ms += round_timing.replay_ms
        round_timing.total_ms = round_timing.draft_ms + round_timing.verify_ms + round_timing.replay_ms
        result.rounds.append(round_timing)

        # Check EOS
        if next_token == tokenizer.eos_token_id:
            break

    gen_end = time.perf_counter()

    result.total_tokens = len(generated_tokens)
    result.total_time_ms = (gen_end - gen_start) * 1000
    result.throughput_tps = result.total_tokens / (result.total_time_ms / 1000) if result.total_time_ms > 0 else 0
    result.avg_accepted = sum(all_na) / len(all_na) if all_na else 0
    result.total_draft_ms = total_draft_ms
    result.total_verify_ms = total_verify_ms
    result.total_replay_ms = total_replay_ms

    return result


def ar_generate(
    verifier,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    device: str = "cuda",
) -> tuple[int, float]:
    """Autoregressive baseline: verifier generates token by token.

    Returns (n_tokens, elapsed_ms).
    """
    if tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt}]
        tok_out = tokenizer.apply_chat_template(
            messages, return_tensors="pt",
            add_generation_prompt=True,
            return_dict=True,
        )
        input_ids = tok_out["input_ids"]
    else:
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]

    v_device = next(verifier.parameters()).device
    input_ids = input_ids.to(v_device)

    # Prefill
    v_pkv = DynamicCache()
    if v_device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    v_out = verifier.model(
        input_ids, past_key_values=v_pkv, use_cache=True,
    )
    v_pkv = v_out.past_key_values
    next_tok = verifier.lm_head(v_out.last_hidden_state[:, -1:]).argmax(dim=-1)
    n_tokens = 1

    for _ in range(max_new_tokens - 1):
        v_out = verifier.model(
            next_tok, past_key_values=v_pkv, use_cache=True,
        )
        v_pkv = v_out.past_key_values
        next_tok = verifier.lm_head(v_out.last_hidden_state[:, -1:]).argmax(dim=-1)
        n_tokens += 1
        if next_tok.item() == tokenizer.eos_token_id:
            break

    if v_device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) * 1000

    return n_tokens, elapsed


# =========================================================================
#  Pipeline overlap benchmark (async CPU draft || GPU verify)
# =========================================================================

def spec_dec_generate_pipelined(
    cpu_drafter,
    verifier,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    NG: int = 8,
    greedy: bool = True,
) -> BenchmarkResult:
    """Speculative decoding with async overlap: CPU drafts while GPU verifies.

    Pipeline design:
    - Round 0: CPU drafts → GPU verifies (sequential, no overlap possible)
    - Round N (N>0):
      - After GPU verify of round N-1 completes, we know accepted tokens
      - CPU replays accepted tokens (cache resync)
      - Then: CPU starts drafting round N
      - Simultaneously: GPU is idle (unless we speculate ahead)

    The TRUE pipeline benefit: while GPU verifies round N, we CANNOT draft
    round N+1 because we don't know what was accepted. The overlap comes from
    running the drafter cache replay CONCURRENTLY with any GPU post-processing.

    For this benchmark, we focus on the sequential case and measure the
    component times to compute both sequential and pipeline-overlap throughput.
    """
    result = BenchmarkResult(prompt=prompt)

    # Tokenize
    if tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt}]
        tok_out = tokenizer.apply_chat_template(
            messages, return_tensors="pt",
            add_generation_prompt=True,
            return_dict=True,
        )
        input_ids = tok_out["input_ids"]
    else:
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]

    v_device = next(verifier.parameters()).device
    assert v_device.type == "cuda", "Pipelined mode requires GPU verifier"

    # Verifier prefill
    input_ids_v = input_ids.to(v_device)
    v_pkv = DynamicCache()
    torch.cuda.synchronize()
    v_out = verifier.model(input_ids_v, past_key_values=v_pkv, use_cache=True)
    v_pkv = v_out.past_key_values
    first_token = verifier.lm_head(v_out.last_hidden_state[:, -1:]).argmax(dim=-1).item()
    torch.cuda.synchronize()

    # CPU drafter prefill
    conv_states, ssm_states = cpu_drafter.create_cache(batch_size=1)
    cpu_drafter.prefill(input_ids.cpu(), conv_states, ssm_states)

    generated_tokens = [first_token]
    current_token = first_token
    all_na = []
    total_draft_ms = 0.0
    total_verify_ms = 0.0
    total_replay_ms = 0.0

    gen_start = time.perf_counter()

    while len(generated_tokens) < max_new_tokens:
        tokens_left = max_new_tokens - len(generated_tokens)
        K = min(NG, tokens_left)
        round_timing = RoundTiming()

        # Snapshot
        snap_conv = conv_states.clone()
        snap_ssm = ssm_states.clone()

        # CPU Draft
        t0 = time.perf_counter()
        draft_tokens = []
        draft_probs = []
        tok_id = current_token
        for _ in range(K):
            logits = cpu_drafter.forward_step(tok_id, conv_states, ssm_states)
            probs = logits.softmax(dim=-1).squeeze(0)
            next_t = logits.argmax(dim=-1).item() if greedy else torch.multinomial(probs, 1).item()
            draft_tokens.append(next_t)
            draft_probs.append(probs)
            tok_id = next_t
        t1 = time.perf_counter()
        round_timing.draft_ms = (t1 - t0) * 1000

        # GPU Verify (async via CUDA stream overlap is automatic with sync)
        verify_input = torch.tensor(
            [[current_token] + draft_tokens], dtype=torch.long, device=v_device,
        )
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        v_out = verifier.model(verify_input, past_key_values=v_pkv, use_cache=True)
        v_logits = verifier.lm_head(v_out.last_hidden_state)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        round_timing.verify_ms = (t1 - t0) * 1000

        # Rejection
        draft_tokens_t = torch.tensor(draft_tokens)
        if greedy:
            n_accepted, next_token = greedy_rejection(
                draft_tokens_t, draft_probs, v_logits.cpu(), K,
            )
        else:
            n_accepted, next_token = stochastic_rejection(
                draft_tokens_t, draft_probs, v_logits.cpu(), K,
            )

        round_timing.n_accepted = n_accepted
        round_timing.tokens_produced = n_accepted + 1
        accepted = draft_tokens[:n_accepted] + [next_token]
        generated_tokens.extend(accepted)

        # Drafter replay
        t0 = time.perf_counter()
        conv_states.copy_(snap_conv)
        ssm_states.copy_(snap_ssm)
        for tok_id in accepted:
            cpu_drafter.forward_step(tok_id, conv_states, ssm_states)
        t1 = time.perf_counter()
        round_timing.replay_ms = (t1 - t0) * 1000

        # Crop verifier KV
        n_to_remove = K - n_accepted
        if n_to_remove > 0:
            new_len = v_pkv.get_seq_length() - n_to_remove
            v_pkv.crop(new_len)

        current_token = next_token
        all_na.append(n_accepted)
        total_draft_ms += round_timing.draft_ms
        total_verify_ms += round_timing.verify_ms
        total_replay_ms += round_timing.replay_ms
        round_timing.total_ms = round_timing.draft_ms + round_timing.verify_ms + round_timing.replay_ms
        result.rounds.append(round_timing)

        if next_token == tokenizer.eos_token_id:
            break

    gen_end = time.perf_counter()

    result.total_tokens = len(generated_tokens)
    result.total_time_ms = (gen_end - gen_start) * 1000
    result.throughput_tps = result.total_tokens / (result.total_time_ms / 1000) if result.total_time_ms > 0 else 0
    result.avg_accepted = sum(all_na) / len(all_na) if all_na else 0
    result.total_draft_ms = total_draft_ms
    result.total_verify_ms = total_verify_ms
    result.total_replay_ms = total_replay_ms

    return result


# =========================================================================
#  Guided pipeline: GPU verify + guidance extraction → CPU guided draft
# =========================================================================

def _verify_with_guidance(verifier, verify_input, v_pkv, ge, prep, v_device, n_accepted=None):
    """Run verifier forward with hidden state extraction + guidance computation.

    Args:
        verifier: LLaMA model on GPU
        verify_input: [1, S] token ids on GPU
        v_pkv: DynamicCache
        ge: GuidanceExtractor on GPU
        prep: PrepMambaDeltas on GPU
        v_device: verifier device
        n_accepted: If provided, extract guidance at position n_accepted
                    (the acceptance boundary). If None, use last position.

    Returns:
        v_logits: [1, S, V] on GPU
        guide_embd: [1, S, v_h_dim] on GPU (full sequence, for later position selection)
        v_pkv: updated KV cache
    """
    # Run verifier decoder with hidden states
    v_out = verifier.model(
        verify_input, past_key_values=v_pkv, use_cache=True,
        output_hidden_states=True,
    )
    v_pkv = v_out.past_key_values

    # Extract guidance hidden states from specified layers
    tgt_device = ge.proj.weight.device
    guide_input = None
    for idx in ge.in_layer:
        h = v_out.hidden_states[idx].to(tgt_device)
        guide_input = h if guide_input is None else torch.cat((guide_input, h), dim=-1)

    # Free hidden states immediately
    v_out.hidden_states = None

    # GuidanceExtractor: [1, S, concat_dim] → [1, S, v_h_dim]
    guide_embd = ge.proj(guide_input.to(ge.proj.weight.dtype))

    # Compute logits
    v_logits = verifier.lm_head(v_out.last_hidden_state)

    return v_logits, guide_embd, v_pkv


def _guidance_from_embd(guide_embd, prep, pos):
    """Extract guidance deltas from guide_embd at a given position.

    Args:
        guide_embd: [1, S, v_h_dim] tensor
        prep: PrepMambaDeltas module
        pos: position index to extract from

    Returns:
        guide_deltas_cpu: [n_layers, delta_dim] FP32 on CPU
    """
    guide_pos = guide_embd[:, pos, None]  # [1, 1, v_h_dim]
    deltas = prep(guide_pos)  # [n_layers, 1, 1, delta_dim]
    return deltas.squeeze(1).squeeze(1).float().cpu()


def spec_dec_generate_guided(
    cpu_drafter,
    verifier,
    ge,
    prep,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    NG: int = 8,
    greedy: bool = True,
) -> BenchmarkResult:
    """Speculative decoding with GPU guidance → CPU guided draft.

    Pipeline per round:
      1. GPU: verify draft tokens + extract hidden states + compute guidance deltas
      2. Transfer: deltas [n_layers, d_inner] → CPU (tiny: ~92KB)
      3. CPU: replay accepted tokens (unguided), then draft NG tokens WITH guidance
    """
    result = BenchmarkResult(prompt=prompt)

    # Tokenize
    if tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt}]
        tok_out = tokenizer.apply_chat_template(
            messages, return_tensors="pt",
            add_generation_prompt=True,
            return_dict=True,
        )
        input_ids = tok_out["input_ids"]
    else:
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]

    v_device = next(verifier.parameters()).device

    # ---- Verifier prefill with guidance extraction ----
    input_ids_v = input_ids.to(v_device)
    v_pkv = DynamicCache()
    torch.cuda.synchronize()

    # Prefill: get hidden states for initial guidance
    v_out = verifier.model(
        input_ids_v, past_key_values=v_pkv, use_cache=True,
        output_hidden_states=True,
    )
    v_pkv = v_out.past_key_values
    v_last_logits = verifier.lm_head(v_out.last_hidden_state[:, -1:])
    first_token = v_last_logits.argmax(dim=-1).item()

    # Extract initial guidance from last position
    tgt_device = ge.proj.weight.device
    guide_input = None
    for idx in ge.in_layer:
        h = v_out.hidden_states[idx].to(tgt_device)
        guide_input = h if guide_input is None else torch.cat((guide_input, h), dim=-1)
    v_out.hidden_states = None

    guide_embd = ge.proj(guide_input.to(ge.proj.weight.dtype))
    current_deltas_cpu = _guidance_from_embd(guide_embd, prep, -1)

    # Keep full KV cache (no crop). The pipeline generates first_token from
    # prefill and then drafts from first_token onward. The verifier KV retains
    # the full prompt context so verify calls get correct position encoding.

    torch.cuda.synchronize()

    # ---- Precompute prep(zeros) bias for CPU drafter ----
    # The trainer passes prep(torch.zeros(...)) during prefill/replay, which is
    # NOT zero (LayerNorm bias + trained proj). CPU drafter must use the same bias.
    with torch.no_grad():
        v_h_dim = ge.proj.weight.shape[0]
        zero_deltas_cpu = prep(torch.zeros(1, 1, v_h_dim, device=prep.proj.weight.device))
        zero_deltas_cpu = zero_deltas_cpu.squeeze(1).squeeze(1).float().cpu()  # [n_layers, delta_dim]

    # ---- CPU drafter prefill (with prep(zeros) bias, matching trainer) ----
    conv_states, ssm_states = cpu_drafter.create_cache(batch_size=1)
    input_ids_cpu = input_ids.cpu()
    for t in range(input_ids_cpu.shape[1]):
        cpu_drafter.forward_step(
            input_ids_cpu[0, t].item(), conv_states, ssm_states,
            guidance_deltas=zero_deltas_cpu,
        )

    # State tracking
    generated_tokens = [first_token]
    current_token = first_token
    total_draft_ms = 0.0
    total_verify_ms = 0.0
    total_replay_ms = 0.0
    total_guidance_ms = 0.0
    all_na = []

    gen_start = time.perf_counter()

    while len(generated_tokens) < max_new_tokens:
        round_timing = RoundTiming()
        tokens_left = max_new_tokens - len(generated_tokens)
        K = min(NG, tokens_left)

        # --- Snapshot drafter cache ---
        snap_conv = conv_states.clone()
        snap_ssm = ssm_states.clone()

        # --- CPU Draft K tokens WITH guidance ---
        t0 = time.perf_counter()
        draft_tokens = []
        draft_probs = []
        tok_id = current_token
        for _ in range(K):
            logits = cpu_drafter.forward_step(
                tok_id, conv_states, ssm_states,
                guidance_deltas=current_deltas_cpu,
            )
            probs = logits.softmax(dim=-1).squeeze(0)
            if greedy:
                next_t = logits.argmax(dim=-1).item()
            else:
                next_t = torch.multinomial(probs, 1).item()
            draft_tokens.append(next_t)
            draft_probs.append(probs)
            tok_id = next_t
        t1 = time.perf_counter()
        round_timing.draft_ms = (t1 - t0) * 1000

        # --- GPU Verify + extract guidance ---
        verify_input = torch.tensor(
            [[current_token] + draft_tokens], dtype=torch.long, device=v_device,
        )
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        v_logits, guide_embd, v_pkv = _verify_with_guidance(
            verifier, verify_input, v_pkv, ge, prep, v_device,
        )
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        round_timing.verify_ms = (t1 - t0) * 1000

        # --- Rejection ---
        draft_tokens_t = torch.tensor(draft_tokens)
        if greedy:
            n_accepted, next_token = greedy_rejection(
                draft_tokens_t, draft_probs, v_logits.cpu(), K,
            )
        else:
            n_accepted, next_token = stochastic_rejection(
                draft_tokens_t, draft_probs, v_logits.cpu(), K,
            )

        round_timing.n_accepted = n_accepted
        round_timing.tokens_produced = n_accepted + 1

        # Collect accepted tokens + next_token
        accepted = draft_tokens[:n_accepted] + [next_token]
        generated_tokens.extend(accepted)

        # --- Extract guidance at acceptance boundary (position n_accepted) ---
        # Matches trainer.py: guide = prep(guide_embd[arange(B), NA, None])
        # Position n_accepted in the verify output corresponds to the token
        # at the acceptance boundary — this is where the next round should start.
        current_deltas_cpu = _guidance_from_embd(guide_embd, prep, n_accepted)
        del guide_embd

        # --- Drafter cache: activation replay with prep(zeros) bias ---
        # The trainer passes prep(torch.zeros(...)) during replay, which is NOT
        # actual zeros (norm bias + trained proj produce non-zero deltas).
        # CPU drafter must use the same bias to maintain state alignment.
        t0 = time.perf_counter()
        conv_states.copy_(snap_conv)
        ssm_states.copy_(snap_ssm)
        for tok_id in accepted:
            cpu_drafter.forward_step(tok_id, conv_states, ssm_states,
                                     guidance_deltas=zero_deltas_cpu)
        t1 = time.perf_counter()
        round_timing.replay_ms = (t1 - t0) * 1000

        # --- Verifier KV cache: crop to keep only accepted ---
        n_to_remove = K - n_accepted
        if n_to_remove > 0:
            new_len = v_pkv.get_seq_length() - n_to_remove
            v_pkv.crop(new_len)

        current_token = next_token
        all_na.append(n_accepted)

        total_draft_ms += round_timing.draft_ms
        total_verify_ms += round_timing.verify_ms
        total_replay_ms += round_timing.replay_ms
        round_timing.total_ms = round_timing.draft_ms + round_timing.verify_ms + round_timing.replay_ms
        result.rounds.append(round_timing)

        if next_token == tokenizer.eos_token_id:
            break

    gen_end = time.perf_counter()

    result.total_tokens = len(generated_tokens)
    result.total_time_ms = (gen_end - gen_start) * 1000
    result.throughput_tps = result.total_tokens / (result.total_time_ms / 1000) if result.total_time_ms > 0 else 0
    result.avg_accepted = sum(all_na) / len(all_na) if all_na else 0
    result.total_draft_ms = total_draft_ms
    result.total_verify_ms = total_verify_ms
    result.total_replay_ms = total_replay_ms

    return result

def spec_dec_generate_guided_pipelined(
    cpu_drafter,
    verifier,
    ge,
    prep,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    NG: int = 8,
    greedy: bool = True,
) -> BenchmarkResult:
    """Pipelined speculative decoding attempt: overlap CPU replay with GPU verify.

    FINDING: model.forward() in eager PyTorch is SYNCHRONOUS — by the time it
    returns, GPU is already done (torch.cuda.synchronize() = 0.04ms after).
    There is NO async window for CPU overlap.

    This function still implements speculative replay to measure the maximum
    achievable benefit IF async overlap existed (theoretical). In practice,
    the speculative replay adds sequential overhead, making this SLOWER than
    the sequential version.

    Kept for comparison purposes. Use --skip-pipeline to skip.
    """
    result = BenchmarkResult(prompt=prompt)

    # Tokenize
    if tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt}]
        tok_out = tokenizer.apply_chat_template(
            messages, return_tensors="pt",
            add_generation_prompt=True,
            return_dict=True,
        )
        input_ids = tok_out["input_ids"]
    else:
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]

    v_device = next(verifier.parameters()).device
    assert v_device.type == "cuda", "Pipelined guided mode requires GPU verifier"

    # ---- Verifier prefill with guidance extraction ----
    input_ids_v = input_ids.to(v_device)
    v_pkv = DynamicCache()
    torch.cuda.synchronize()

    v_out = verifier.model(
        input_ids_v, past_key_values=v_pkv, use_cache=True,
        output_hidden_states=True,
    )
    v_pkv = v_out.past_key_values
    v_last_logits = verifier.lm_head(v_out.last_hidden_state[:, -1:])
    first_token = v_last_logits.argmax(dim=-1).item()

    # Extract initial guidance from last position
    tgt_device = ge.proj.weight.device
    guide_input = None
    for idx in ge.in_layer:
        h = v_out.hidden_states[idx].to(tgt_device)
        guide_input = h if guide_input is None else torch.cat((guide_input, h), dim=-1)
    v_out.hidden_states = None
    guide_embd = ge.proj(guide_input.to(ge.proj.weight.dtype))
    current_deltas_cpu = _guidance_from_embd(guide_embd, prep, -1)

    torch.cuda.synchronize()

    # ---- Precompute prep(zeros) bias for CPU drafter ----
    with torch.no_grad():
        v_h_dim = ge.proj.weight.shape[0]
        zero_deltas_cpu = prep(torch.zeros(1, 1, v_h_dim, device=prep.proj.weight.device))
        zero_deltas_cpu = zero_deltas_cpu.squeeze(1).squeeze(1).float().cpu()

    # ---- CPU drafter prefill ----
    conv_states, ssm_states = cpu_drafter.create_cache(batch_size=1)
    input_ids_cpu = input_ids.cpu()
    for t in range(input_ids_cpu.shape[1]):
        cpu_drafter.forward_step(
            input_ids_cpu[0, t].item(), conv_states, ssm_states,
            guidance_deltas=zero_deltas_cpu,
        )

    # ---- Pre-allocate speculative replay buffers (avoids malloc per round) ----
    spec_conv = conv_states.clone()
    spec_ssm = ssm_states.clone()

    # State tracking
    generated_tokens = [first_token]
    current_token = first_token
    total_draft_ms = 0.0
    total_verify_ms = 0.0
    total_replay_ms = 0.0
    all_na = []
    spec_replay_hits = 0
    spec_replay_total = 0

    gen_start = time.perf_counter()

    while len(generated_tokens) < max_new_tokens:
        round_timing = RoundTiming()
        tokens_left = max_new_tokens - len(generated_tokens)
        K = min(NG, tokens_left)

        # --- Snapshot drafter cache (pre-draft state) ---
        snap_conv = conv_states.clone()
        snap_ssm = ssm_states.clone()

        # --- CPU Draft K tokens WITH guidance ---
        t0 = time.perf_counter()
        draft_tokens = []
        draft_probs = []
        tok_id = current_token
        for _ in range(K):
            logits = cpu_drafter.forward_step(
                tok_id, conv_states, ssm_states,
                guidance_deltas=current_deltas_cpu,
            )
            probs = logits.softmax(dim=-1).squeeze(0)
            if greedy:
                next_t = logits.argmax(dim=-1).item()
            else:
                next_t = torch.multinomial(probs, 1).item()
            draft_tokens.append(next_t)
            draft_probs.append(probs)
            tok_id = next_t
        t1 = time.perf_counter()
        round_timing.draft_ms = (t1 - t0) * 1000

        # --- GPU Verify (SYNCHRONOUS: model.forward blocks for full duration) ---
        verify_input = torch.tensor(
            [[current_token] + draft_tokens], dtype=torch.long, device=v_device,
        )
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        # model.forward() is synchronous in eager PyTorch (~17ms wall clock)
        # By the time it returns, GPU is already done. No async window exists.
        v_out = verifier.model(
            verify_input, past_key_values=v_pkv, use_cache=True,
            output_hidden_states=True,
        )
        v_pkv = v_out.past_key_values
        guide_input = None
        for idx in ge.in_layer:
            h = v_out.hidden_states[idx].to(tgt_device)
            guide_input = h if guide_input is None else torch.cat((guide_input, h), dim=-1)
        v_out.hidden_states = None
        guide_embd = ge.proj(guide_input.to(ge.proj.weight.dtype))
        v_logits = verifier.lm_head(v_out.last_hidden_state)

        # Speculative replay: runs AFTER model.forward returns (no overlap).
        # This is pure sequential overhead.
        spec_conv.copy_(snap_conv)
        spec_ssm.copy_(snap_ssm)
        for tok_id_spec in draft_tokens:
            cpu_drafter.forward_step(tok_id_spec, spec_conv, spec_ssm,
                                     guidance_deltas=zero_deltas_cpu)

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        round_timing.verify_ms = (t1 - t0) * 1000

        # --- Rejection (CPU, fast ~0.5ms) ---
        draft_tokens_t = torch.tensor(draft_tokens)
        if greedy:
            n_accepted, next_token = greedy_rejection(
                draft_tokens_t, draft_probs, v_logits.cpu(), K,
            )
        else:
            n_accepted, next_token = stochastic_rejection(
                draft_tokens_t, draft_probs, v_logits.cpu(), K,
            )

        round_timing.n_accepted = n_accepted
        round_timing.tokens_produced = n_accepted + 1
        accepted = draft_tokens[:n_accepted] + [next_token]
        generated_tokens.extend(accepted)

        # --- Extract guidance at acceptance boundary ---
        current_deltas_cpu = _guidance_from_embd(guide_embd, prep, n_accepted)
        del guide_embd

        # --- Resolve speculative replay ---
        t0 = time.perf_counter()
        spec_replay_total += 1
        if n_accepted == K:
            # HIT: all K accepted. Speculative state is correct.
            conv_states.copy_(spec_conv)
            ssm_states.copy_(spec_ssm)
            cpu_drafter.forward_step(next_token, conv_states, ssm_states,
                                     guidance_deltas=zero_deltas_cpu)
            spec_replay_hits += 1
        else:
            # MISS: restore from snapshot, replay correct tokens.
            conv_states.copy_(snap_conv)
            ssm_states.copy_(snap_ssm)
            for tok_id_r in accepted:
                cpu_drafter.forward_step(tok_id_r, conv_states, ssm_states,
                                         guidance_deltas=zero_deltas_cpu)
        t1 = time.perf_counter()
        round_timing.replay_ms = (t1 - t0) * 1000

        # --- Verifier KV cache: crop to keep only accepted ---
        n_to_remove = K - n_accepted
        if n_to_remove > 0:
            new_len = v_pkv.get_seq_length() - n_to_remove
            v_pkv.crop(new_len)

        current_token = next_token
        all_na.append(n_accepted)

        total_draft_ms += round_timing.draft_ms
        total_verify_ms += round_timing.verify_ms
        total_replay_ms += round_timing.replay_ms
        round_timing.total_ms = round_timing.draft_ms + round_timing.verify_ms + round_timing.replay_ms
        result.rounds.append(round_timing)

        if next_token == tokenizer.eos_token_id:
            break

    gen_end = time.perf_counter()

    result.total_tokens = len(generated_tokens)
    result.total_time_ms = (gen_end - gen_start) * 1000
    result.throughput_tps = result.total_tokens / (result.total_time_ms / 1000) if result.total_time_ms > 0 else 0
    result.avg_accepted = sum(all_na) / len(all_na) if all_na else 0
    result.total_draft_ms = total_draft_ms
    result.total_verify_ms = total_verify_ms
    result.total_replay_ms = total_replay_ms
    result.spec_replay_hit_rate = spec_replay_hits / max(spec_replay_total, 1)

    return result


def run_benchmark(args):
    dataset_names = [d.strip() for d in args.datasets.split(",") if d.strip()]

    print("=" * 70)
    print(f"Pipeline Benchmark: mode={args.mode}")
    print(f"  Drafter:  {args.drafter}")
    print(f"  Verifier: {args.verifier}")
    if args.guided_ckpt:
        print(f"  Guided:   {args.guided_ckpt}")
    print(f"  Datasets: {','.join(dataset_names)}")
    print(f"  NG={args.ng}, max_new_tokens={args.tgt_len}")
    print(f"  Total samples: {args.total_samples}")
    print(f"  Greedy: {not args.stochastic}")
    print(f"  Threads: {args.threads}")
    print("=" * 70)

    # Set CPU threads
    torch.set_num_threads(args.threads)
    os.environ["OMP_NUM_THREADS"] = str(args.threads)

    # Load guidance modules (if guided)
    ge, prep = None, None
    if args.guided_ckpt:
        v_device_for_guide = "cpu" if args.mode == "cpu_verify" else "cuda"
        ge, prep, v_layers, drafter_sd = load_guidance_modules(args.guided_ckpt, device=v_device_for_guide)

    # Load drafter — use fine-tuned weights from guided checkpoint when available
    cpu_drafter = load_cpu_drafter(
        args.drafter, use_int8=not args.bf16,
        guided_sd=drafter_sd if args.guided_ckpt else None,
    )
    if args.guided_ckpt:
        del drafter_sd  # free ~260MB of raw checkpoint tensors

    # Load verifier
    v_device = "cpu" if args.mode == "cpu_verify" else "cuda"
    verifier = load_verifier(
        args.verifier, device=v_device,
        device_map=args.device_map,
        quantize=args.quantize,
    )

    # Tokenizer (must set same chat template as trainer for correct prompt formatting)
    tokenizer = AutoTokenizer.from_pretrained(args.verifier)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not tokenizer.chat_template:
        # Llama 3.1 base model has no chat template; use the one from training
        tokenizer.chat_template = (
            "{% for message in messages %}\n"
            "  {% if (message['role'] != 'assistant') %}\n"
            " {{'<|start_header_id|>' + message['role'] + '<|end_header_id|>\n'"
            " + message['content'] + '<|eot_id|>' + '\n'}}\n"
            " {% elif (message['role'] == 'assistant')%}\n"
            " {{'<|start_header_id|>' + message['role'] + '<|end_header_id|>\n'}}\n"
            " {% generation %}\n"
            " {{message['content'] + '<|eot_id|>'}}\n"
            " {% endgeneration %}\n"
            " {{'\n'}}\n"
            " {% endif %}\n"
            " {% endfor %}\n"
            "{%- if add_generation_prompt %}\n"
            "    {{- '<|start_header_id|>assistant<|end_header_id|>\\n' }}\n"
            "{%- endif %}"
        )

    # Collect prompts from requested evaluation datasets
    prompts, prompt_counts = collect_benchmark_prompts(dataset_names, args.total_samples)
    print(f"\nUsing {len(prompts)} prompts for benchmark")
    print(f"Prompt split: {prompt_counts}\n")

    # ---- Warmup ----
    print("Warmup (2 short generations)...")
    for i in range(min(2, len(prompts))):
        if ge is not None and v_device != "cpu":
            _ = spec_dec_generate_guided(
                cpu_drafter, verifier, ge, prep, tokenizer, prompts[i],
                max_new_tokens=16, NG=args.ng, greedy=not args.stochastic,
            )
        else:
            _ = spec_dec_generate(
                cpu_drafter, verifier, tokenizer, prompts[i],
                max_new_tokens=16, NG=args.ng, device=v_device,
                greedy=not args.stochastic,
            )
    print("Warmup done.\n")

    # ---- Determine which generate function to use ----
    use_guided = ge is not None and v_device != "cpu"
    mode_label = "GUIDED" if use_guided else "UNGUIDED"

    # ---- Speculative Decoding Benchmark ----
    print("=" * 70)
    print(f"SPECULATIVE DECODING ({mode_label}, CPU drafter + {v_device} verifier)")
    print("=" * 70)
    spec_results = []
    for i, prompt in enumerate(prompts):
        if use_guided:
            result = spec_dec_generate_guided(
                cpu_drafter, verifier, ge, prep, tokenizer, prompt,
                max_new_tokens=args.tgt_len, NG=args.ng,
                greedy=not args.stochastic,
            )
        else:
            result = spec_dec_generate(
                cpu_drafter, verifier, tokenizer, prompt,
                max_new_tokens=args.tgt_len, NG=args.ng, device=v_device,
                greedy=not args.stochastic,
            )
        spec_results.append(result)
        print(f"  [{i+1}/{len(prompts)}] tokens={result.total_tokens:3d}, "
              f"time={result.total_time_ms:.0f}ms, "
              f"tps={result.throughput_tps:.1f}, "
              f"accepted={result.avg_accepted:.2f}, "
              f"draft={result.total_draft_ms:.0f}ms, "
              f"verify={result.total_verify_ms:.0f}ms, "
              f"replay={result.total_replay_ms:.0f}ms")

    # Aggregate
    total_tok = sum(r.total_tokens for r in spec_results)
    total_time = sum(r.total_time_ms for r in spec_results)
    avg_tps = total_tok / (total_time / 1000) if total_time > 0 else 0
    avg_accept = sum(r.avg_accepted for r in spec_results) / len(spec_results)
    avg_draft_frac = sum(r.total_draft_ms for r in spec_results) / total_time if total_time > 0 else 0
    avg_verify_frac = sum(r.total_verify_ms for r in spec_results) / total_time if total_time > 0 else 0
    avg_replay_frac = sum(r.total_replay_ms for r in spec_results) / total_time if total_time > 0 else 0

    print(f"\n  Spec Dec Summary ({mode_label}):")
    print(f"    Total tokens: {total_tok}")
    print(f"    Throughput:   {avg_tps:.2f} tok/s")
    print(f"    Avg accepted: {avg_accept:.2f} / {args.ng}")
    print(f"    Time breakdown: draft={avg_draft_frac*100:.1f}%, "
          f"verify={avg_verify_frac*100:.1f}%, "
          f"replay={avg_replay_frac*100:.1f}%, "
          f"overhead={100-100*(avg_draft_frac+avg_verify_frac+avg_replay_frac):.1f}%")

    # Compute pipelined throughput estimate (overlap draft+replay with verify)
    # NOTE: This estimate is UNREALIZABLE in practice. model.forward() in eager
    # PyTorch is synchronous (blocks Python for ~17ms). By the time it returns,
    # GPU is already done. No async window exists for CPU overlap.
    # The formula max(verify, draft+replay) would require CUDA graphs, torch.compile,
    # or a custom kernel to be achievable.
    total_draft_ms = sum(r.total_draft_ms for r in spec_results)
    total_verify_ms = sum(r.total_verify_ms for r in spec_results)
    total_replay_ms = sum(r.total_replay_ms for r in spec_results)
    pipelined_est_ms = 0
    for r in spec_results:
        for rd in r.rounds:
            pipelined_est_ms += max(rd.verify_ms, rd.draft_ms + rd.replay_ms)
    pipelined_tps = total_tok / (pipelined_est_ms / 1000) if pipelined_est_ms > 0 else 0

    print(f"\n  Pipeline overlap estimate (THEORETICAL — unrealizable in eager PyTorch):")
    print(f"    Sequential (actual):  {total_time:.0f} ms -> {avg_tps:.2f} tok/s")
    print(f"    Pipelined (theory):   {pipelined_est_ms:.0f} ms -> {pipelined_tps:.2f} tok/s")
    print(f"    Theoretical boost:    {pipelined_tps/avg_tps:.2f}x (requires CUDA graphs/compile)")

    # ---- Pipelined (speculative replay) benchmark ----
    pipe_results = []
    if use_guided and not getattr(args, 'skip_pipeline', False):
        print(f"\n{'='*70}")
        print(f"PIPELINED SPEC DEC (speculative replay — expected SLOWER, for validation)")
        print(f"  NOTE: model.forward() is synchronous; spec replay adds overhead.")
        print("=" * 70)
        for i, prompt in enumerate(prompts):
            result = spec_dec_generate_guided_pipelined(
                cpu_drafter, verifier, ge, prep, tokenizer, prompt,
                max_new_tokens=args.tgt_len, NG=args.ng,
                greedy=not args.stochastic,
            )
            pipe_results.append(result)
            print(f"  [{i+1}/{len(prompts)}] tokens={result.total_tokens:3d}, "
                  f"time={result.total_time_ms:.0f}ms, "
                  f"tps={result.throughput_tps:.1f}, "
                  f"accepted={result.avg_accepted:.2f}, "
                  f"replay_hit={result.spec_replay_hit_rate:.0%}")

        # Aggregate pipelined
        pipe_tok = sum(r.total_tokens for r in pipe_results)
        pipe_time = sum(r.total_time_ms for r in pipe_results)
        pipe_tps = pipe_tok / (pipe_time / 1000) if pipe_time > 0 else 0
        pipe_accept = sum(r.avg_accepted for r in pipe_results) / len(pipe_results)
        pipe_hit_rate = sum(r.spec_replay_hit_rate for r in pipe_results) / len(pipe_results)

        print(f"\n  Pipelined Summary:")
        print(f"    Total tokens:    {pipe_tok}")
        print(f"    Throughput:      {pipe_tps:.2f} tok/s")
        print(f"    Avg accepted:    {pipe_accept:.2f} / {args.ng}")
        print(f"    Spec replay hit: {pipe_hit_rate:.1%} (all-K-accepted rounds)")
        print(f"    vs Sequential:   {pipe_tps/avg_tps:.2f}x")

    # ---- AR Baseline ----
    print(f"\n{'='*70}")
    print("AUTOREGRESSIVE BASELINE (verifier only)")
    print("=" * 70)
    ar_prompts = prompts[:min(args.total_samples, args.ar_samples)]
    ar_total_tok = 0
    ar_total_ms = 0.0
    for i, prompt in enumerate(ar_prompts):
        n_tok, elapsed = ar_generate(
            verifier, tokenizer, prompt,
            max_new_tokens=args.tgt_len, device=v_device,
        )
        ar_total_tok += n_tok
        ar_total_ms += elapsed
        print(f"  [{i+1}/{len(ar_prompts)}] tokens={n_tok}, "
              f"time={elapsed:.0f}ms, "
              f"tps={n_tok/(elapsed/1000):.1f}")

    ar_tps = ar_total_tok / (ar_total_ms / 1000) if ar_total_ms > 0 else 0
    print(f"\n  AR Baseline: {ar_tps:.2f} tok/s")

    # ---- Speedup ----
    speedup_seq = avg_tps / ar_tps if ar_tps > 0 else 0
    speedup_pipe_est = pipelined_tps / ar_tps if ar_tps > 0 else 0
    speedup_pipe_real = (pipe_tps / ar_tps) if (pipe_results and ar_tps > 0) else 0

    print(f"\n{'='*70}")
    print("SPEEDUP SUMMARY")
    print("=" * 70)
    print(f"  AR baseline:            {ar_tps:.2f} tok/s")
    print(f"  Spec dec (sequential):  {avg_tps:.2f} tok/s  ({speedup_seq:.2f}x)")
    if pipe_results:
        print(f"  Spec dec (pipelined):   {pipe_tps:.2f} tok/s  ({speedup_pipe_real:.2f}x)  [SLOWER — confirms no async overlap]")
    print(f"  Theoretical max:        {pipelined_tps:.2f} tok/s  ({speedup_pipe_est:.2f}x)  [UNREALIZABLE — eager PyTorch sync]")
    print(f"  Avg acceptance:         {avg_accept:.2f} / {args.ng}")
    print(f"  Guidance:               {mode_label}")

    # Per-round breakdown
    all_rounds = [rd for r in spec_results for rd in r.rounds]
    if all_rounds:
        avg_d = sum(rd.draft_ms for rd in all_rounds) / len(all_rounds)
        avg_v = sum(rd.verify_ms for rd in all_rounds) / len(all_rounds)
        avg_r = sum(rd.replay_ms for rd in all_rounds) / len(all_rounds)
        avg_tok = sum(rd.tokens_produced for rd in all_rounds) / len(all_rounds)
        print(f"\n  Per-round averages ({len(all_rounds)} rounds):")
        print(f"    Draft:   {avg_d:.2f} ms")
        print(f"    Verify:  {avg_v:.2f} ms")
        print(f"    Replay:  {avg_r:.2f} ms")
        print(f"    Tokens:  {avg_tok:.2f} per round")

    # ---- Save results ----
    output = {
        "config": {
            "mode": args.mode,
            "drafter": args.drafter,
            "verifier": args.verifier,
            "guided_ckpt": args.guided_ckpt,
            "guided": use_guided,
            "datasets": dataset_names,
            "prompt_counts": prompt_counts,
            "ng": args.ng,
            "tgt_len": args.tgt_len,
            "threads": args.threads,
            "greedy": not args.stochastic,
            "int8": not args.bf16,
        },
        "spec_dec": {
            "throughput_tps": round(avg_tps, 2),
            "avg_accepted": round(avg_accept, 2),
            "total_tokens": total_tok,
            "total_time_ms": round(total_time, 1),
            "draft_frac": round(avg_draft_frac, 3),
            "verify_frac": round(avg_verify_frac, 3),
            "replay_frac": round(avg_replay_frac, 3),
        },
        "pipeline_estimate": {
            "throughput_tps": round(pipelined_tps, 2),
            "total_time_ms": round(pipelined_est_ms, 1),
        },
        "pipelined_measured": {
            "throughput_tps": round(pipe_tps, 2) if pipe_results else None,
            "total_time_ms": round(pipe_time, 1) if pipe_results else None,
            "avg_accepted": round(pipe_accept, 2) if pipe_results else None,
            "spec_replay_hit_rate": round(pipe_hit_rate, 3) if pipe_results else None,
        },
        "ar_baseline": {
            "throughput_tps": round(ar_tps, 2),
            "total_tokens": ar_total_tok,
            "total_time_ms": round(ar_total_ms, 1),
        },
        "speedup": {
            "sequential": round(speedup_seq, 3),
            "pipelined_estimate": round(speedup_pipe_est, 3),
            "pipelined_measured": round(speedup_pipe_real, 3) if pipe_results else None,
        },
        "per_round": {
            "avg_draft_ms": round(avg_d, 2) if all_rounds else 0,
            "avg_verify_ms": round(avg_v, 2) if all_rounds else 0,
            "avg_replay_ms": round(avg_r, 2) if all_rounds else 0,
            "avg_tokens_per_round": round(avg_tok, 2) if all_rounds else 0,
            "n_rounds": len(all_rounds),
        },
        "per_sample": [
            {
                "prompt": r.prompt[:80],
                "total_tokens": r.total_tokens,
                "throughput_tps": round(r.throughput_tps, 2),
                "avg_accepted": round(r.avg_accepted, 2),
                "total_time_ms": round(r.total_time_ms, 1),
            }
            for r in spec_results
        ],
    }

    out_dir = "outputs/pipeline_benchmark"
    os.makedirs(out_dir, exist_ok=True)
    if args.out_file:
        out_file = args.out_file
        os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
    else:
        tag = "guided" if use_guided else "unguided"
        out_file = os.path.join(out_dir, f"benchmark_{args.mode}_{tag}_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_file}")

    return output


def main():
    parser = argparse.ArgumentParser(description="Pipeline benchmark: CPU Mamba2 drafter + LLaMA verifier")
    parser.add_argument("--mode", choices=["gpu_verify", "cpu_verify"], required=True,
                        help="gpu_verify: CPU draft + GPU verify. cpu_verify: both on CPU.")
    parser.add_argument("--drafter", type=str, default=DEFAULT_DRAFTER,
                        help="Path to HF Mamba2 checkpoint for CPU drafter")
    parser.add_argument("--verifier", type=str, default=DEFAULT_VERIFIER_8B,
                        help="Path to LLaMA verifier")
    parser.add_argument("--guided_ckpt", type=str, default=None,
                        help="Path to guided .ckpt with GuidanceExtractor + PrepMambaDeltas weights. "
                             "When provided, guidance deltas are computed on GPU and injected into CPU drafter.")
    parser.add_argument("--datasets", type=str, default=DEFAULT_PROMPT_DATASETS,
                        help="Comma-separated prompt datasets for benchmark (supported: humaneval,gsm8k)")
    parser.add_argument("--ng", type=int, default=DEFAULT_NG,
                        help="Number of draft tokens per round")
    parser.add_argument("--tgt_len", type=int, default=DEFAULT_TGT_LEN,
                        help="Max new tokens to generate")
    parser.add_argument("--total_samples", type=int, default=DEFAULT_TOTAL_SAMPLES,
                        help="Number of prompts to benchmark")
    parser.add_argument("--ar_samples", type=int, default=None,
                        help="Number of AR baseline samples (default: same as total_samples, or fewer for cpu_verify)")
    parser.add_argument("--threads", type=int, default=16,
                        help="CPU threads for drafter (and verifier in cpu_verify mode)")
    parser.add_argument("--device_map", type=str, default=None,
                        help="device_map for verifier (e.g. 'auto' for 70B multi-GPU)")
    parser.add_argument("--quantize", type=str, default=None, choices=["8bit"],
                        help="Quantize verifier (e.g. 8bit for 70B)")
    parser.add_argument("--stochastic", action="store_true",
                        help="Use stochastic rejection sampling instead of greedy")
    parser.add_argument("--bf16", action="store_true",
                        help="Use BF16 fused model instead of INT8")
    parser.add_argument("--skip_pipeline", action="store_true",
                        help="Skip the pipelined benchmark (it's slower, just for validation)")
    parser.add_argument("--out_file", type=str, default=None,
                        help="Output file path (default: auto-generated in outputs/pipeline_benchmark/)")
    args = parser.parse_args()

    if args.ar_samples is None:
        args.ar_samples = min(4, args.total_samples) if args.mode == "cpu_verify" else args.total_samples

    run_benchmark(args)


if __name__ == "__main__":
    main()
