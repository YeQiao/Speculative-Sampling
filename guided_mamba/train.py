"""
Training module for guided Mamba2 speculative decoding.

Adapts the SD² training approach for a Mamba2 drafter steered by
LLaMA verifier hidden states.
"""

import copy
import time
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import lightning as L
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch import loggers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModel,
)

from guided_mamba.guided_mamba2 import GuidedMamba2Block


# ---------------------------------------------------------------------------
#  Guidance extraction: compress verifier hidden states at [low,mid,high]
#  into a single guidance embedding.
# ---------------------------------------------------------------------------
class GuidanceExtractor(nn.Module):
    """
    Attached to the verifier model.  During its forward pass the hook
    collects hidden states at ``v_layers`` and this module compresses
    them (concat → linear) into a guidance embedding of size ``v_h_dim``.

    Initialised so that output = mean(h_low, h_mid, h_high).
    """

    def __init__(self, v_h_dim: int, n_layers: int = 3):
        super().__init__()
        self.proj = nn.Linear(n_layers * v_h_dim, v_h_dim)
        # Identity-average init (like SD²)
        with torch.no_grad():
            I = torch.eye(v_h_dim)
            self.proj.weight.copy_(torch.cat([I] * n_layers, dim=1))
            self.proj.bias.zero_()

    def forward(self, *layer_hiddens: torch.Tensor) -> torch.Tensor:
        return self.proj(torch.cat(layer_hiddens, dim=-1))


# ---------------------------------------------------------------------------
#  PrepLatentDeltas: map guidance (B,S,v_h_dim) → per-layer deltas
#  for the Mamba2 x-branch (and optionally z-branch).
# ---------------------------------------------------------------------------
class PrepMambaDeltas(nn.Module):
    """
    Maps ``(B, S, v_h_dim)`` → ``(n_layers, B, S, delta_dim)``
    where ``delta_dim = d_inner`` (x only) or ``2 * d_inner`` (x + z).
    Zero-initialised so training starts with no guidance effect.
    """

    def __init__(self, v_h_dim: int, delta_dim: int, n_layers: int):
        super().__init__()
        self.norm = nn.LayerNorm(v_h_dim)
        self.proj = nn.Linear(v_h_dim, delta_dim * n_layers, bias=False)
        nn.init.zeros_(self.proj.weight)
        self.n_layers = n_layers
        self.delta_dim = delta_dim

    def forward(self, guide_embd: torch.Tensor) -> torch.Tensor:
        # guide_embd: (B, S, v_h_dim) or (B, 1, v_h_dim)
        x = self.proj(self.norm(guide_embd))
        shape = guide_embd.shape[:-1]  # (B, S)
        x = x.view(*shape, self.n_layers, self.delta_dim)
        # Permute to (n_layers, B, S, delta_dim)
        if x.dim() == 4:  # (B, S, nL, D)
            x = x.permute(2, 0, 1, 3)
        else:  # (B, nL, D) — single-token
            x = x.permute(1, 0, 2)
        return x


# ---------------------------------------------------------------------------
#  Full training module (Lightning)
# ---------------------------------------------------------------------------
class GuidedMambaTrainer(L.LightningModule):
    def __init__(
        self,
        # Model paths
        verifier: str = "meta-llama/Llama-3.1-8B-Instruct",
        drafter: str = "/HSC/users/qiaoye/SSM_SPEC/checkpoints/improved-mamba-alignment/checkpoint-750",
        # Guidance extraction
        v_layers: list[int] = [5, 16, 29],
        # Guidance application
        steer_z: bool = False,
        d_layers: str | list[int] = "all",
        # Speculation
        ngram: int = 8,
        # Training offset
        pos_method: Literal["regular", "blocked"] = "regular",
        # Training
        loss_method: Literal["tvd", "kl", "rkl"] = "tvd",
        finetune_drafter: bool = False,
        lr_start: float = 3e-4,
        lr_end: float = 3e-5,
        warmup_steps: int = 30,
        # Eval
        n_spec_dec_batches: int = 4,
        tgt_len: int = 64,
        greedy_sample: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.strict_loading = False

        self.NG = ngram
        self.steer_z = steer_z
        self.loss_method = loss_method
        self.finetune_drafter = finetune_drafter
        self.n_spec_dec_batches = n_spec_dec_batches
        self.tgt_len = tgt_len
        self.greedy_sample = greedy_sample
        self.pos_method = pos_method

        # --- Load models ---
        self._load_models()
        # --- Setup guidance pipeline ---
        self._setup_guidance(v_layers, d_layers)

    # ------------------------------------------------------------------
    #  Model loading
    # ------------------------------------------------------------------
    def _load_models(self):
        # Verifier (frozen) — supports LLaMA, Gemma, and other CausalLM families
        self.v_base = AutoModelForCausalLM.from_pretrained(
            self.hparams["verifier"],
            torch_dtype=torch.float16,
        )
        for p in self.v_base.parameters():
            p.requires_grad = False

        # Resolve text config (may be nested for multimodal models like Gemma 4)
        v_cfg = self.v_base.config
        if hasattr(v_cfg, "text_config"):
            v_cfg = v_cfg.text_config
        self.V_H_DIM = v_cfg.hidden_size
        self.V_N_LAYERS = v_cfg.num_hidden_layers

        # Resolve text backbone: Gemma4 nests it under .model.language_model
        if hasattr(self.v_base, "model") and hasattr(self.v_base.model, "language_model"):
            self._v_text_model = self.v_base.model.language_model
        elif hasattr(self.v_base, "model"):
            self._v_text_model = self.v_base.model
        else:
            self._v_text_model = self.v_base

        # Tokenizer
        self.tok = AutoTokenizer.from_pretrained(
            self.hparams["verifier"], use_fast=True
        )
        self.tok.truncation_side = "left"
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.pad_token_id = self.tok.pad_token_id

        # Drafter (Mamba2)
        self.d_base = AutoModelForCausalLM.from_pretrained(
            self.hparams["drafter"],
            torch_dtype=torch.float32,
        )
        self.D_INNER = int(self.d_base.config.expand * self.d_base.config.hidden_size)
        self.D_N_LAYERS = self.d_base.config.num_hidden_layers
        if self.finetune_drafter:
            self.d_base.train()
        else:
            for p in self.d_base.parameters():
                p.requires_grad = False

    # ------------------------------------------------------------------
    #  Guidance setup
    # ------------------------------------------------------------------
    def _setup_guidance(self, v_layers, d_layers):
        # Validate layer indices
        self.v_layers = list(v_layers)
        for l in self.v_layers:
            assert 0 <= l < self.V_N_LAYERS, f"v_layer {l} out of range [0, {self.V_N_LAYERS})"

        # 1. Guidance extractor (attached to verifier)
        self.guidance_extractor = GuidanceExtractor(
            self.V_H_DIM, n_layers=len(self.v_layers)
        )

        # 2. Guidance projector: V_H_DIM → per-Mamba-layer deltas
        if isinstance(d_layers, str) and d_layers == "all":
            self.d_layer_indices = list(range(self.D_N_LAYERS))
        elif isinstance(d_layers, list):
            self.d_layer_indices = d_layers
        else:
            self.d_layer_indices = list(range(self.D_N_LAYERS))

        delta_dim = self.D_INNER * (2 if self.steer_z else 1)
        self.latent_mod_prep = PrepMambaDeltas(
            self.V_H_DIM, delta_dim, len(self.d_layer_indices)
        )

        # 3. Replace drafter layers with guided versions
        backbone = self.d_base.backbone
        for new_idx, orig_layer_idx in enumerate(self.d_layer_indices):
            orig_block = backbone.layers[orig_layer_idx]
            guided_block = GuidedMamba2Block(
                orig_block, layer_idx=new_idx, steer_z=self.steer_z
            )
            backbone.layers[orig_layer_idx] = guided_block

    # ------------------------------------------------------------------
    #  Verifier forward with guidance extraction
    # ------------------------------------------------------------------
    def _verifier_forward(self, input_ids: torch.Tensor):
        """
        Run verifier, collect hidden states at self.v_layers,
        and return (v_logits, guide_embd).
        """
        with torch.no_grad():
            v_outputs = self._v_text_model(
                input_ids,
                output_hidden_states=True,
                return_dict=True,
            )
        # v_outputs.hidden_states is a tuple of (n_layers+1) tensors
        # Index 0 = embedding output, index i = output of layer i-1
        collected = []
        for layer_idx in self.v_layers:
            # hidden_states[layer_idx + 1] = output of layer layer_idx
            collected.append(v_outputs.hidden_states[layer_idx + 1])

        guide_embd = self.guidance_extractor(*collected)

        with torch.no_grad():
            v_logits = self.v_base.lm_head(v_outputs.last_hidden_state)

        return v_logits, guide_embd

    # ------------------------------------------------------------------
    #  Drafter forward with guidance
    # ------------------------------------------------------------------
    def _drafter_forward(self, input_ids: torch.Tensor, guidance_deltas: torch.Tensor):
        """
        Run drafter with per-layer guidance deltas injected.
        guidance_deltas: (n_guided_layers, B, S, delta_dim)
        """
        backbone = self.d_base.backbone
        hidden_states = backbone.embeddings(input_ids)

        # Force train mode so Mamba2 uses the fused kernel path.
        # The unfused (eval) path hits a causal_conv1d stride alignment bug
        # when proj_dim % 8 != 0 (e.g. 2836 for 45M config).
        # Mamba2 has no dropout/batchnorm so train vs eval is otherwise identical.
        was_training = backbone.training
        backbone.train()
        try:
            for block in backbone.layers:
                if isinstance(block, GuidedMamba2Block):
                    hidden_states = block(
                        hidden_states,
                        guidance_deltas=guidance_deltas,
                    )
                else:
                    hidden_states = block(hidden_states)
        finally:
            backbone.train(was_training)

        hidden_states = backbone.norm_f(hidden_states)
        d_logits = self.d_base.lm_head(hidden_states.to(self.d_base.lm_head.weight.dtype))
        return d_logits

    # ------------------------------------------------------------------
    #  Training step
    # ------------------------------------------------------------------
    def process_batch(self, batch, compute_tvd=False):
        targets = batch["targets"]
        B, S = targets.shape

        if "loss_mask" in batch:
            loss_mask = batch["loss_mask"].float()
            # Fall back to non-padding mask if chat template didn't produce
            # assistant masks (e.g. Gemma tokenizer lacks {% generation %})
            if loss_mask.sum() == 0:
                loss_mask = (targets != self.pad_token_id).float()
        else:
            loss_mask = (targets != self.pad_token_id).float()
            loss_mask[:, 0] = 1.0

        # 1. Verifier forward → logits + guidance
        v_logits, guide_embd = self._verifier_forward(targets)

        # 2. Simulate speculative offset (same as SD²)
        if self.pos_method == "regular":
            offset = torch.randint(1, self.NG + 1, (1,), device=self.device)[0]
            guide_pos = torch.arange(0, S, device=self.device) - offset
            guide_pos = guide_pos.clamp_min_(0)
        else:
            offset = torch.randint(0, self.NG, (1,), device=self.device)[0]
            guide_pos = (
                (torch.arange(0, S, device=self.device) - offset - 1)
                // self.NG
                * self.NG
            ) + offset
            guide_pos = guide_pos.clamp_min_(0)

        guide_embd_shifted = guide_embd[:, guide_pos]
        # Mask guidance at padding positions (rolled by 1 since we predict next token)
        guide_embd_shifted = guide_embd_shifted * loss_mask[:, :, None].roll(-1, dims=1)

        # 3. Project guidance → per-layer deltas
        guidance_deltas = self.latent_mod_prep(guide_embd_shifted)

        # 4. Drafter forward with guidance
        d_logits = self._drafter_forward(targets, guidance_deltas)

        # 5. Loss
        metrics = {}
        v_probs = v_logits.flatten(0, 1).softmax(dim=-1)
        d_probs = d_logits.flatten(0, 1).softmax(dim=-1)

        if self.loss_method == "tvd" or compute_tvd:
            tvd_loss = (v_probs - d_probs).abs().sum(dim=-1) * 0.5
            tvd_loss = (tvd_loss * loss_mask.flatten()).sum() / (loss_mask.sum() + 1e-4)
            metrics["loss_tvd"] = tvd_loss
            if self.loss_method == "tvd":
                loss = tvd_loss

        if self.loss_method in ("kl", "rkl"):
            v_log = v_logits.flatten(0, 1).log_softmax(dim=-1)
            d_log = d_logits.flatten(0, 1).log_softmax(dim=-1)
            if self.loss_method == "rkl":
                kl = F.kl_div(v_log, d_log, log_target=True, reduction="none")
            else:
                kl = F.kl_div(d_log, v_log, log_target=True, reduction="none")
            kl = kl.sum(dim=-1)
            kl = (kl * loss_mask.flatten()).sum() / (loss_mask.sum() + 1e-4)
            metrics["loss_kl"] = kl
            loss = kl

        return loss, metrics  # type: ignore

    def training_step(self, batch, batch_idx):
        loss, metrics = self.process_batch(
            batch, compute_tvd=batch_idx % 50 == 0,
        )
        self.log("train/loss", loss)
        for k, v in metrics.items():
            self.log(f"train/{k}", v)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, metrics = self.process_batch(batch, compute_tvd=True)
        self.log("val/loss", loss)
        for k, v in metrics.items():
            self.log(f"val/{k}", v)
        return loss

    # ------------------------------------------------------------------
    #  Optimizer
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        params = []
        params.append(self.guidance_extractor.parameters())
        params.append(self.latent_mod_prep.parameters())
        if self.finetune_drafter:
            params.append(self.d_base.parameters())

        from itertools import chain
        optimizer = torch.optim.AdamW(
            chain(*params),
            lr=self.hparams["lr_start"],
        )
        scheduler = torch.optim.lr_scheduler.ChainedScheduler(
            [
                torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=0.1, end_factor=1,
                    total_iters=self.hparams["warmup_steps"],
                ),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=int(self.trainer.estimated_stepping_batches) - self.hparams["warmup_steps"],
                    eta_min=self.hparams["lr_end"],
                ),
            ],
            optimizer,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    # ------------------------------------------------------------------
    #  Exclude verifier from checkpoints
    # ------------------------------------------------------------------
    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        keys_to_remove = [k for k in state.keys() if k.startswith("v_base.")]
        for k in keys_to_remove:
            state.pop(k)
        return state

    # ------------------------------------------------------------------
    #  Steering-only save/load (adapter-style, works with frozen drafter)
    # ------------------------------------------------------------------
    def save_steering(self, path: str):
        """Save only the steering modules (guidance_extractor + latent_mod_prep).

        Use when finetune_drafter=False to get a small checkpoint that can be
        loaded on top of the original drafter weights.
        """
        import os
        os.makedirs(path, exist_ok=True)
        torch.save({
            "guidance_extractor": self.guidance_extractor.state_dict(),
            "latent_mod_prep": self.latent_mod_prep.state_dict(),
            "hparams": dict(self.hparams),
        }, os.path.join(path, "steering.pt"))

    def load_steering(self, path: str):
        """Load steering modules from a save_steering() checkpoint."""
        import os
        ckpt = torch.load(os.path.join(path, "steering.pt"), map_location="cpu")
        self.guidance_extractor.load_state_dict(ckpt["guidance_extractor"])
        self.latent_mod_prep.load_state_dict(ckpt["latent_mod_prep"])

    # ------------------------------------------------------------------
    #  Drafter forward with cache support (for generation)
    # ------------------------------------------------------------------
    def _drafter_forward_cached(
        self,
        input_ids: torch.Tensor,
        guidance_deltas: torch.Tensor,
        cache_params=None,
        cache_position=None,
    ):
        """Run guided drafter with Mamba2Cache support for incremental decoding."""
        from transformers.models.mamba2.modeling_mamba2 import Mamba2Cache

        backbone = self.d_base.backbone

        # If no cache, create one and set cache_position for prefill
        if cache_params is None:
            cache_params = Mamba2Cache(
                backbone.config,
                input_ids.size(0),
                device=input_ids.device,
                dtype=torch.float32,
            )
            cache_position = torch.arange(
                0, backbone.config.conv_kernel, device=input_ids.device
            )

        hidden_states = backbone.embeddings(input_ids)

        for block in backbone.layers:
            if isinstance(block, GuidedMamba2Block):
                hidden_states = block(
                    hidden_states,
                    cache_params=cache_params,
                    cache_position=cache_position,
                    guidance_deltas=guidance_deltas,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    cache_params=cache_params,
                    cache_position=cache_position,
                )

        hidden_states = backbone.norm_f(hidden_states)
        d_logits = self.d_base.lm_head(hidden_states.to(self.d_base.lm_head.weight.dtype))
        return d_logits, cache_params

    # ------------------------------------------------------------------
    #  Verifier forward with KV cache support (for generation)
    # ------------------------------------------------------------------
    def _verifier_forward_cached(
        self,
        input_ids: torch.Tensor,
        past_key_values=None,
        position_ids=None,
        attention_mask=None,
    ):
        """Run verifier with KV cache, collect hidden states at v_layers.

        Returns (v_logits, guide_embd, past_key_values).
        """
        v_outputs = self.v_base.model(
            input_ids,
            past_key_values=past_key_values,
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        collected = []
        for layer_idx in self.v_layers:
            collected.append(v_outputs.hidden_states[layer_idx + 1].float())
        guide_embd = self.guidance_extractor(*collected)
        v_logits = self.v_base.lm_head(v_outputs.last_hidden_state)
        return v_logits, guide_embd, v_outputs.past_key_values

    # ------------------------------------------------------------------
    #  Prompt preparation (chat template)
    # ------------------------------------------------------------------
    def prep_for_gen(self, prompts: list[str]):
        """Tokenize prompts with chat template. Returns (input_ids, attention_mask)."""
        # LLaMA 3.1 chat template
        if not hasattr(self.tok, 'chat_template') or self.tok.chat_template is None:
            self.tok.chat_template = (
                "{% for message in messages %}\n"
                "  {% if (message['role'] != 'assistant') %}\n"
                " {{'<|start_header_id|>' + message['role'] + '<|end_header_id|>\n' + message['content'] + '<|eot_id|>' + '\n'}}\n"
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
        toks = self.tok.apply_chat_template(
            [[{"role": "user", "content": msg}] for msg in prompts],
            return_tensors="pt",
            add_generation_prompt=True,
            padding=True,
            tokenizer_kwargs={"return_attention_mask": True},
            return_dict=True,
        )
        return toks["input_ids"].to(self.device), toks["attention_mask"].to(self.device)

    # ------------------------------------------------------------------
    #  Speculative decoding generation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 128,
        draft_temperature: float = 1.0,
    ):
        """
        Speculative decoding with guided Mamba2 drafter and LLaMA verifier.

        Returns:
            sampled: (B, S + generated) token ids
            extra: dict with n_accepted, time_per_block, attention_mask
        """
        from transformers import DynamicCache
        from transformers.models.mamba2.modeling_mamba2 import Mamba2Cache

        B, S = input_ids.shape
        device = input_ids.device
        NG = self.NG
        PAD_FACTOR = 4

        # Pre-allocate output buffer
        sampled = torch.full(
            (B, S + max_new_tokens * PAD_FACTOR),
            self.pad_token_id,
            dtype=torch.long,
            device=device,
        )
        sampled[:, :S] = input_ids
        curr = S - 1  # index of last "committed" token

        # Position tracking
        position_ids = torch.arange(
            0, S + max_new_tokens * PAD_FACTOR, device=device
        )[None].expand(B, -1)
        if attention_mask is not None:
            position_ids = torch.clamp_min(
                position_ids - S + attention_mask.sum(dim=1)[:, None], 0
            )
            attention_mask = torch.cat(
                (attention_mask, torch.zeros(B, max_new_tokens * PAD_FACTOR, device=device, dtype=attention_mask.dtype)),
                dim=1,
            )
        else:
            # Create all-ones mask (needed for SD²-style rejected-token masking)
            attention_mask = torch.zeros(B, S + max_new_tokens * PAD_FACTOR, device=device, dtype=torch.long)
            attention_mask[:, :S] = 1

        # ---- Verifier prefill (pass attention_mask for left-padding) ----
        v_pkv = DynamicCache()
        v_logits, guide_embd, v_pkv = self._verifier_forward_cached(
            sampled[:, :curr + 1], past_key_values=v_pkv,
            position_ids=position_ids[:, :curr + 1],
            attention_mask=attention_mask[:, :curr + 1],
        )
        v_pkv.crop(curr)  # crop last entry so verify step re-processes it
        # guide for first drafting step: guidance at the last committed position
        guide = self.latent_mod_prep(guide_embd[:, -1:, :])

        # ---- Drafter prefill ----
        # Drafter sees tokens [0 .. curr-1] (everything before the token we'll draft from)
        d_cache = Mamba2Cache(
            self.d_base.backbone.config,
            B,
            device=device,
            dtype=torch.float32,
        )
        d_cache_pos = torch.arange(
            0, self.d_base.backbone.config.conv_kernel, device=device
        )
        if curr > 0:
            # Prefill drafter on prefix (without the last committed token — that's the first draft input)
            prefill_deltas = self.latent_mod_prep(
                torch.zeros(B, curr, self.V_H_DIM, device=device)
            )  # zero guidance for prefill (no stale guidance available)
            _, d_cache = self._drafter_forward_cached(
                sampled[:, :curr], prefill_deltas,
                cache_params=d_cache, cache_position=d_cache_pos,
            )
        d_next_pos = curr  # next position index for drafter cache

        # ---- Speculative decoding loop ----
        NA_COUNT = torch.zeros(B, dtype=torch.long, device=device)
        STEPS = torch.zeros(B, dtype=torch.long, device=device)
        has_ended = torch.zeros(B, dtype=torch.bool, device=device)
        total_time = 0.0
        total_steps = 0

        eot_id = self.tok.eos_token_id

        while not (
            has_ended.all()
            or (total_steps + 1) * (NG + 1) >= max_new_tokens * PAD_FACTOR
        ):
            torch.cuda.synchronize()
            start = time.time()

            # --- Draft NG tokens ---
            q = torch.zeros(B, NG, self.v_base.config.vocab_size, device=device)
            for i in range(NG):
                d_cache_pos_i = torch.tensor(
                    [d_next_pos + i], device=device, dtype=torch.long
                )
                d_logits_i, d_cache = self._drafter_forward_cached(
                    sampled[:, curr + i : curr + i + 1],
                    guide,  # same guidance for all NG draft steps (stale by design)
                    cache_params=d_cache,
                    cache_position=d_cache_pos_i,
                )
                d_prob = (d_logits_i[:, -1] / draft_temperature).softmax(-1)
                q[:, i] = d_prob
                if self.greedy_sample:
                    sampled[:, curr + i + 1] = d_prob.argmax(dim=-1)
                else:
                    sampled[:, curr + i + 1] = torch.multinomial(d_prob, 1).squeeze(1)
                attention_mask[:, curr + i + 1] = 1
                position_ids[:, curr + i + 1] = position_ids[:, curr + i] + 1

            # --- Verify NG+1 tokens (crop + position_ids for correct RoPE) ---
            v_logits, guide_embd_new, v_pkv = self._verifier_forward_cached(
                sampled[:, curr:curr + NG + 1],
                past_key_values=v_pkv,
                position_ids=position_ids[:, curr:curr + NG + 1],
            )
            if self.greedy_sample:
                v_probs = torch.zeros(
                    B, NG + 1, self.v_base.config.vocab_size,
                    device=device, dtype=v_logits.dtype,
                )
                v_probs.scatter_(-1, v_logits.argmax(dim=-1, keepdim=True), 1.0)
            else:
                v_probs = v_logits.float().softmax(-1)

            # --- Rejection sampling ---
            NA, next_dist = self._rejection_sampling(
                q, v_probs,
                sampled[:, curr + 1:curr + NG + 1],
            )

            # Update attention mask for accepted/rejected tokens
            is_ongoing = ~has_ended
            attention_mask[is_ongoing, curr + 1:curr + NG + 1] = (
                torch.arange(0, NG, device=device)[None, :]
                < NA[is_ongoing, None]
            ).long()

            # Sample next token from adjusted distribution
            position_ids[:, curr + NG + 1] = position_ids[:, curr] + NA + 1
            next_token = torch.multinomial(next_dist.clamp(min=0), 1).squeeze(1)
            sampled[:, curr + NG + 1] = next_token
            attention_mask[:, curr + NG + 1] = 1

            # Check for end of sequence
            has_ended = has_ended | (
                (
                    sampled[:, curr + 1:curr + NG + 2]
                    * attention_mask[:, curr + 1:curr + NG + 2]
                )
                == eot_id
            ).any(dim=-1)
            attention_mask[has_ended, curr + 1:curr + NG + 2] = 0

            # Update guidance for next round: use verifier output at the accepted position
            guide = self.latent_mod_prep(
                guide_embd_new[torch.arange(B), NA, None]
            )

            # Advance curr (no between-round v_pkv crop — stale entries are harmless)
            old_curr = curr
            curr = curr + NG + 1

            # Rebuild drafter cache from scratch for the accepted prefix
            max_accepted_end = (old_curr + NA.max() + 1).item()
            d_cache = Mamba2Cache(
                self.d_base.backbone.config, B,
                device=device, dtype=torch.float32,
            )
            d_cache_pos = torch.arange(
                0, self.d_base.backbone.config.conv_kernel, device=device
            )
            prefill_len = max_accepted_end + 1  # up to and including the next_token
            prefill_deltas = self.latent_mod_prep(
                torch.zeros(B, prefill_len, self.V_H_DIM, device=device)
            )
            _, d_cache = self._drafter_forward_cached(
                sampled[:, :prefill_len], prefill_deltas,
                cache_params=d_cache, cache_position=d_cache_pos,
            )
            d_next_pos = prefill_len

            torch.cuda.synchronize()
            block_time = time.time() - start
            total_time += block_time

            # Check position-based termination
            has_ended = has_ended | (
                position_ids[:, curr] - position_ids[:, S] >= max_new_tokens
            )

            is_ongoing = ~has_ended
            NA_COUNT[is_ongoing] += NA[is_ongoing]
            STEPS[is_ongoing] += 1
            total_steps += 1

        return sampled, {
            "n_accepted": NA_COUNT.float() / (STEPS.float() + 1e-6),
            "time_per_block": total_time / max(total_steps, 1),
            "attention_mask": attention_mask,
            "total_steps": total_steps,
            "total_time": total_time,
        }

    @staticmethod
    def _rejection_sampling(
        drafted: torch.Tensor,
        model: torch.Tensor,
        sampled_idx: torch.Tensor,
    ):
        """Batched rejection sampling (same as SD²).

        Args:
            drafted: (B, NG, V) draft probabilities
            model: (B, NG+1, V) verifier probabilities
            sampled_idx: (B, NG) drafted token ids
        Returns:
            NA: (B,) number of accepted tokens per sample
            next_dist: (B, V) adjusted distribution for the next token
        """
        B, NG, V = drafted.shape
        q_sample = drafted.gather(-1, sampled_idx[:, :, None])[:, :, 0]
        p_sample = model[:, :-1].gather(-1, sampled_idx[:, :, None])[:, :, 0]
        rejected = (p_sample / (q_sample + 1e-12)) < torch.rand(B, NG, device=drafted.device)
        has_rejection, NA = torch.max(rejected, dim=-1)
        NA[~has_rejection] = NG
        next_dist = model[torch.arange(B), NA, :]
        next_dist[has_rejection] -= drafted[has_rejection, NA[has_rejection], :]
        next_dist = next_dist.clamp_min_(0)
        # Normalize
        next_dist = next_dist / (next_dist.sum(dim=-1, keepdim=True) + 1e-12)
        return NA, next_dist
