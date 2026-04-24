"""
Training & generation module for Mamba2 speculative decoding with LLaMA verifier.

Follows SD²'s architecture and patterns closely:
- Custom LlamaModel with old-style _update_causal_mask (proper batched masking)
- v_pkv.crop(curr) after prefill for correct RoPE positions
- attention_mask passed to ALL verifier calls (prefill + verify)
- Non-compact layout: curr += NG + 1 after each round
"""

import copy
import time
from itertools import chain
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import lightning as L
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch import loggers
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

from spec_mamba.models.llama import LlamaForCausalLM as CustomLlamaForCausalLM
from spec_mamba.guided_mamba2 import GuidedMamba2Block

PRETTY_PRINT = False


# ---------------------------------------------------------------------------
#  Mamba2 cache snapshot/restore utilities for activation replay
# ---------------------------------------------------------------------------
def snapshot_mamba2_cache(cache) -> dict:
    """Save a deep copy of Mamba2Cache conv_states and ssm_states."""
    return {
        "conv_states": cache.conv_states.clone(),
        "ssm_states": cache.ssm_states.clone(),
    }


def restore_mamba2_cache(cache, snapshot: dict):
    """Restore Mamba2Cache from a snapshot (in-place)."""
    cache.conv_states.copy_(snapshot["conv_states"])
    cache.ssm_states.copy_(snapshot["ssm_states"])


# ---------------------------------------------------------------------------
#  Guidance extraction: compress verifier hidden states at [low,mid,high]
#  into a single guidance embedding (same as SD²'s WithKwargs(nn.Linear(...)))
# ---------------------------------------------------------------------------
class GuidanceExtractor(nn.Module):
    """3-layer concat → linear, identity-average init."""

    def __init__(self, v_h_dim: int, n_layers: int = 3):
        super().__init__()
        self.in_layer: list[int] = []  # set by setup code
        self.proj = nn.Linear(n_layers * v_h_dim, v_h_dim)
        with torch.no_grad():
            I = torch.eye(v_h_dim)
            self.proj.weight.copy_(torch.cat([I] * n_layers, dim=1))
            self.proj.bias.zero_()

    def forward(self, x, **kwargs):
        return self.proj(x.to(self.proj.weight.dtype))


# ---------------------------------------------------------------------------
#  PrepMambaDeltas: map guidance (B,S,v_h_dim) → per-layer deltas for Mamba2
# ---------------------------------------------------------------------------
class PrepMambaDeltas(nn.Module):
    """Maps (B, S, v_h_dim) → (n_layers, B, S, delta_dim). Zero-init."""

    def __init__(self, v_h_dim: int, delta_dim: int, n_layers: int):
        super().__init__()
        self.norm = nn.LayerNorm(v_h_dim)
        self.proj = nn.Linear(v_h_dim, delta_dim * n_layers, bias=False)
        nn.init.zeros_(self.proj.weight)
        self.n_layers = n_layers
        self.delta_dim = delta_dim

    def forward(self, guide_embd: torch.Tensor) -> torch.Tensor:
        x = self.proj(self.norm(guide_embd))
        shape = guide_embd.shape[:-1]
        x = x.view(*shape, self.n_layers, self.delta_dim)
        if x.dim() == 4:  # (B, S, nL, D) → (nL, B, S, D)
            x = x.permute(2, 0, 1, 3)
        else:  # (B, nL, D) → (nL, B, D)
            x = x.permute(1, 0, 2)
        return x


# ---------------------------------------------------------------------------
#  Rejection sampling (identical to SD²)
# ---------------------------------------------------------------------------
def rejection_sampling(
    drafted: torch.Tensor,
    model: torch.Tensor,
    sampled_idx: torch.Tensor,
):
    """
    Args:
        drafted: [B, ND, V] draft probabilities
        model: [B, ND+1, V] verifier probabilities
        sampled_idx: [B, ND] drafted token ids
    Returns:
        NA: [B] number accepted (ND if all accepted)
        next_dist: [B, V] adjusted distribution for next token
    """
    B, ND, V = drafted.shape
    q_sample = drafted.gather(-1, sampled_idx[:, :, None])[:, :, 0]
    p_sample = model[:, :-1].gather(-1, sampled_idx[:, :, None])[:, :, 0]
    rejected = (p_sample / (q_sample + 1e-12)) < torch.rand(B, ND, device=drafted.device)
    has_rejection, NA = torch.max(rejected, dim=-1)
    NA[~has_rejection] = ND
    next_dist = model[torch.arange(B), NA, :]
    next_dist[has_rejection] -= drafted[has_rejection, NA[has_rejection], :]
    next_dist = next_dist.clamp_min_(0)
    return NA, next_dist


# ---------------------------------------------------------------------------
#  Main training module
# ---------------------------------------------------------------------------
class SpecMambaTrainer(L.LightningModule):
    def __init__(
        self,
        # Model paths
        verifier: str = "meta-llama/Llama-3.1-8B-Instruct",
        drafter: str = "/HSC/users/qiaoye/SSM_SPEC/checkpoints/improved-mamba-alignment/checkpoint-750",
        # Guidance extraction
        v_layers: tuple[int, int, int] = (5, 16, 29),
        # Guidance application
        steer_z: bool = False,
        d_layers: str | list[int] = "all",
        # Speculation
        ngram: int = 8,
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

        self._load_models()
        self._setup_guidance(v_layers, d_layers)

    # ------------------------------------------------------------------
    #  Model loading
    # ------------------------------------------------------------------
    def _load_models(self):
        # Verifier: use our custom LlamaForCausalLM with old-style _update_causal_mask
        self.v_base = CustomLlamaForCausalLM.from_pretrained(
            self.hparams["verifier"],
            torch_dtype=torch.float16,
        )
        for p in self.v_base.parameters():
            p.requires_grad = False

        self.V_H_DIM = self.v_base.config.hidden_size
        self.V_N_LAYERS = self.v_base.config.num_hidden_layers

        # Tokenizer
        self.tok = AutoTokenizer.from_pretrained(
            self.hparams["verifier"], use_fast=True
        )
        self.tok.truncation_side = "left"
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.pad_token_id = self.tok.pad_token_id
        self.eot_id = 128009  # LLaMA 3.x <|eot_id|>

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
        self.v_layers = list(v_layers)

        # 1. Guidance extractor (same interface as SD²'s guidance_embd_layer)
        self.guidance_extractor = GuidanceExtractor(self.V_H_DIM, n_layers=len(self.v_layers))
        # SD²'s LlamaModel collects hidden_states BEFORE the layer runs,
        # so in_layer=[i] collects the INPUT to layer i = OUTPUT of layer i-1.
        # Our checkpoint was trained with output_hidden_states[layer_idx + 1],
        # which is the OUTPUT of layer_idx.  To match, shift each index by +1.
        self.guidance_extractor.in_layer = [v + 1 for v in self.v_layers]
        # Attach to verifier model so it's called during forward
        self.v_base.get_decoder().guidance_embd_layer = self.guidance_extractor

        # 2. Prep deltas for Mamba2 layers
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
    #  Verifier forward (training, no cache)
    # ------------------------------------------------------------------
    def _verifier_forward(self, input_ids: torch.Tensor):
        """Run verifier, extract guidance, return (v_logits, guide_embd)."""
        with torch.no_grad():
            v_out = self.v_base.get_decoder()(
                input_ids,
                compute_guidance=True,
                return_dict=True,
            )
        guide_embd = v_out["guide_embd"]
        v_logits = self.v_base.lm_head(v_out["out"].last_hidden_state)
        return v_logits, guide_embd

    # ------------------------------------------------------------------
    #  Drafter forward (training, no cache)
    # ------------------------------------------------------------------
    def _drafter_forward(self, input_ids: torch.Tensor, guidance_deltas: torch.Tensor):
        """Run drafter with per-layer guidance deltas injected."""
        backbone = self.d_base.backbone
        hidden_states = backbone.embeddings(input_ids)

        for block in backbone.layers:
            if isinstance(block, GuidedMamba2Block):
                hidden_states = block(hidden_states, guidance_deltas=guidance_deltas)
            else:
                hidden_states = block(hidden_states)

        hidden_states = backbone.norm_f(hidden_states)
        d_logits = self.d_base.lm_head(hidden_states.to(self.d_base.lm_head.weight.dtype))
        return d_logits

    # ------------------------------------------------------------------
    #  Drafter forward with cache (generation)
    # ------------------------------------------------------------------
    def _drafter_forward_cached(
        self,
        input_ids: torch.Tensor,
        guidance_deltas: torch.Tensor,
        cache_params=None,
        cache_position=None,
    ):
        from transformers.models.mamba2.modeling_mamba2 import Mamba2Cache

        backbone = self.d_base.backbone
        if cache_params is None:
            cache_params = Mamba2Cache(
                backbone.config, input_ids.size(0),
                device=input_ids.device, dtype=torch.float32,
            )
            cache_position = torch.arange(
                0, backbone.config.conv_kernel, device=input_ids.device
            )

        hidden_states = backbone.embeddings(input_ids)
        for block in backbone.layers:
            if isinstance(block, GuidedMamba2Block):
                hidden_states = block(
                    hidden_states, cache_params=cache_params,
                    cache_position=cache_position, guidance_deltas=guidance_deltas,
                )
            else:
                hidden_states = block(
                    hidden_states, cache_params=cache_params,
                    cache_position=cache_position,
                )

        hidden_states = backbone.norm_f(hidden_states)
        d_logits = self.d_base.lm_head(hidden_states.to(self.d_base.lm_head.weight.dtype))
        return d_logits, cache_params

    # ------------------------------------------------------------------
    #  Training
    # ------------------------------------------------------------------
    def process_batch(self, batch, compute_tvd=False):
        targets = batch["targets"]
        B, S = targets.shape

        if "loss_mask" in batch:
            loss_mask = batch["loss_mask"].float()
        else:
            loss_mask = (targets != self.pad_token_id).float()
            loss_mask[:, 0] = 1.0

        v_logits, guide_embd = self._verifier_forward(targets)

        if self.pos_method == "regular":
            offset = torch.randint(1, self.NG + 1, (1,), device=self.device)[0]
            guide_pos = torch.arange(0, S, device=self.device) - offset
            guide_pos = guide_pos.clamp_min_(0)
        else:
            offset = torch.randint(0, self.NG, (1,), device=self.device)[0]
            guide_pos = (
                (torch.arange(0, S, device=self.device) - offset - 1)
                // self.NG * self.NG
            ) + offset
            guide_pos = guide_pos.clamp_min_(0)

        guide_embd_shifted = guide_embd[:, guide_pos]
        guide_embd_shifted = guide_embd_shifted * loss_mask[:, :, None].roll(-1, dims=1)
        guidance_deltas = self.latent_mod_prep(guide_embd_shifted)

        d_logits = self._drafter_forward(targets, guidance_deltas)

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

        return loss, metrics

    def training_step(self, batch, batch_idx):
        loss, metrics = self.process_batch(batch, compute_tvd=batch_idx % 50 == 0)
        self.log("train/loss", loss)
        for k, v in metrics.items():
            self.log(f"train/{k}", v)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, metrics = self.process_batch(batch, compute_tvd=True)
        self.log("val/loss", loss)
        for k, v in metrics.items():
            self.log(f"val/{k}", v)

        if batch_idx < self.n_spec_dec_batches:
            input_ids, attention_mask = self.prep_for_gen(batch["prompt"])
            _, sd_metrics = self.generate(
                input_ids, attention_mask, max_new_tokens=self.tgt_len,
            )
            metrics["spec_dec_na"] = sd_metrics["n_accepted"].mean().item()
            if batch_idx > 0:
                metrics["time_per_block"] = sd_metrics["time_per_block"]

        self.log("val/loss", loss)
        for k, v in metrics.items():
            self.log(f"val/{k}", v)
        return loss

    def configure_optimizers(self):
        params = []
        params.append(self.guidance_extractor.parameters())
        params.append(self.latent_mod_prep.parameters())
        if self.finetune_drafter:
            params.append(self.d_base.parameters())

        optimizer = torch.optim.AdamW(chain(*params), lr=self.hparams["lr_start"])
        scheduler = torch.optim.lr_scheduler.ChainedScheduler([
            torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, end_factor=1,
                total_iters=self.hparams["warmup_steps"],
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=int(self.trainer.estimated_stepping_batches) - self.hparams["warmup_steps"],
                eta_min=self.hparams["lr_end"],
            ),
        ], optimizer)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        keys_to_remove = [k for k in state.keys() if k.startswith("v_base.")]
        for k in keys_to_remove:
            state.pop(k)
        return state

    # ------------------------------------------------------------------
    #  Chat template & tokenization
    # ------------------------------------------------------------------
    def prep_for_gen(self, prompts: list[str]):
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
    #  Speculative decoding generation (follows SD² exactly)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 128,
        draft_temperature: float = 1.0,
        mask_rejected: bool = True,
        use_activation_replay: bool = False,
    ):
        from transformers.models.mamba2.modeling_mamba2 import Mamba2Cache

        B, S = input_ids.shape
        device = input_ids.device
        NG = self.NG
        PAD_FACTOR = 4

        # Pre-allocate output buffer
        sampled = torch.full(
            (B, S + max_new_tokens * PAD_FACTOR),
            self.pad_token_id, dtype=torch.long, device=device,
        )
        sampled[:, :S] = input_ids
        curr = S - 1

        # Position tracking (same as SD²)
        position_ids = torch.arange(
            0, S + max_new_tokens * PAD_FACTOR, device=device
        )[None].expand(B, -1).clone()

        if attention_mask is not None:
            position_ids = torch.clamp_min_(
                position_ids - S + attention_mask.sum(dim=1)[:, None], 0
            )
            attention_mask = torch.cat(
                (attention_mask, torch.zeros(B, max_new_tokens * PAD_FACTOR, device=device, dtype=attention_mask.dtype)),
                dim=1,
            )
        else:
            attention_mask = torch.zeros(
                B, S + max_new_tokens * PAD_FACTOR, device=device, dtype=torch.long)
            attention_mask[:, :S] = 1

        # ---- Verifier prefill ----
        v_pkv = DynamicCache()
        v_out = self.v_base.get_decoder()(
            sampled[:, :curr + 1],
            position_ids=position_ids[:, :curr + 1],
            compute_guidance=True,
            past_key_values=v_pkv,
            attention_mask=attention_mask[:, :curr + 1],
            use_cache=True,
        )
        v_pkv = v_out["out"].past_key_values
        v_pkv.crop(curr)  # SD² fix: crop so verify re-processes at correct position
        guide = self.latent_mod_prep(v_out["guide_embd"][:, -1, None])

        # ---- Drafter prefill ----
        d_cache = Mamba2Cache(
            self.d_base.backbone.config, B, device=device, dtype=torch.float32,
        )
        d_cache_pos = torch.arange(0, self.d_base.backbone.config.conv_kernel, device=device)
        if curr > 0:
            prefill_deltas = self.latent_mod_prep(
                torch.zeros(B, curr, self.V_H_DIM, device=device)
            )
            _, d_cache = self._drafter_forward_cached(
                sampled[:, :curr], prefill_deltas,
                cache_params=d_cache, cache_position=d_cache_pos,
            )
        d_next_pos = curr

        # ---- Speculative decoding loop ----
        NA_COUNT = torch.zeros(B, dtype=torch.long, device=device)
        STEPS = torch.zeros(B, dtype=torch.long, device=device)
        has_ended = torch.zeros(B, dtype=torch.bool, device=device)
        total_time = 0.0
        total_steps = 0

        while not (
            has_ended.all()
            or (total_steps + 1) * (NG + 1) >= max_new_tokens * PAD_FACTOR
        ):
            torch.cuda.synchronize()
            start = time.time()

            (sampled, position_ids, attention_mask, NA, curr,
             guide, d_cache, d_next_pos, v_pkv, has_ended) = self._spec_dec_step(
                sampled, position_ids, attention_mask, curr,
                guide, d_cache, d_next_pos, v_pkv, has_ended,
                draft_temperature=draft_temperature,
                mask_rejected=mask_rejected,
                use_activation_replay=use_activation_replay,
            )

            torch.cuda.synchronize()
            block_time = time.time() - start
            total_time += block_time

            # Position-based termination
            has_ended = has_ended | (
                position_ids[:, curr] - position_ids[:, S] >= max_new_tokens
            )

            if total_steps == 0:
                NA_COUNT[has_ended] += NG
                STEPS[has_ended] += 1

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

    @torch.no_grad()
    def _spec_dec_step(
        self,
        sampled: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        curr: int,
        guide: torch.Tensor,
        d_cache,
        d_next_pos: int,
        v_pkv: DynamicCache,
        has_ended: torch.Tensor,
        draft_temperature: float = 1.0,
        mask_rejected: bool = True,
        use_activation_replay: bool = False,
    ):
        """One round of draft + verify + reject. Returns updated state."""
        from transformers.models.mamba2.modeling_mamba2 import Mamba2Cache

        B = sampled.shape[0]
        NG = self.NG
        device = sampled.device

        q = torch.zeros(B, NG, self.v_base.config.vocab_size, device=device)

        # --- Snapshot drafter cache before drafting (for activation replay) ---
        if use_activation_replay:
            d_cache_snapshot = snapshot_mamba2_cache(d_cache)
            d_next_pos_snapshot = d_next_pos

        # --- Draft NG tokens ---
        for i in range(NG):
            d_cache_pos_i = torch.tensor([d_next_pos + i], device=device, dtype=torch.long)
            d_logits_i, d_cache = self._drafter_forward_cached(
                sampled[:, curr + i: curr + i + 1],
                guide,
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

        # --- Verify NG+1 tokens (SD² pattern: pass attention_mask!) ---
        v_out = self.v_base.get_decoder()(
            sampled[:, curr: curr + NG + 1],
            position_ids=position_ids[:, curr: curr + NG + 1],
            compute_guidance=True,
            return_dict=True,
            use_cache=True,
            past_key_values=v_pkv,
            attention_mask=attention_mask[:, :curr + NG + 1],
        )
        v_logits = self.v_base.lm_head(v_out["out"].last_hidden_state)
        v_pkv = v_out["out"].past_key_values

        if self.greedy_sample:
            v_probs = torch.zeros(
                B, NG + 1, self.v_base.config.vocab_size,
                device=device, dtype=v_logits.dtype,
            )
            v_probs.scatter_(-1, v_logits.argmax(dim=-1, keepdim=True), 1.0)
        else:
            v_probs = v_logits.softmax(-1)

        # --- Rejection sampling ---
        NA, next_dist = rejection_sampling(
            q, v_probs, sampled[:, curr + 1: curr + NG + 1],
        )

        # Update attention mask: zero rejected positions so future rounds
        # don't attend to stale KV entries (matches SD² exactly).
        if mask_rejected:
            is_ongoing = ~has_ended
            attention_mask[is_ongoing, curr + 1: curr + NG + 1] = (
                torch.arange(0, NG, device=device)[None, :] < NA[is_ongoing, None]
            ).long()

        # Sample next token from adjusted distribution
        position_ids[:, curr + NG + 1] = position_ids[:, curr] + NA + 1
        next_token = torch.multinomial(next_dist.clamp(min=0), 1).squeeze(1)
        sampled[:, curr + NG + 1] = next_token
        attention_mask[:, curr + NG + 1] = 1

        # Check for end of sequence
        has_ended = has_ended | (
            (sampled[:, curr + 1: curr + NG + 2]
             * attention_mask[:, curr + 1: curr + NG + 2])
            == self.eot_id
        ).any(dim=-1)
        attention_mask[has_ended, curr + 1: curr + NG + 2] = 0

        # Advance curr (non-compact layout, same as SD²)
        old_curr = curr
        curr = curr + NG + 1

        # Update guidance
        guide = self.latent_mod_prep(
            v_out["guide_embd"][torch.arange(B), NA, None]
        )

        # Rebuild drafter cache (Mamba2 is recurrent — must re-prefill)
        if use_activation_replay:
            # --- Activation Replay: restore snapshot and replay accepted tokens ---
            # Instead of re-prefilling from scratch over the entire sequence,
            # we restore the cache to its state before drafting and replay only
            # the accepted tokens + resampled token through the drafter.
            restore_mamba2_cache(d_cache, d_cache_snapshot)

            # Build the replay sequence: accepted draft tokens + resampled next_token
            # In non-compact layout, accepted tokens are at old_curr+1..old_curr+NA[b]
            # and the resampled token is at old_curr+NG+1 (= curr).
            # We need to replay these through the drafter to get the correct state.
            max_replay_len = NA.max().item() + 1  # +1 for the resampled token
            replay_ids = torch.full(
                (B, max_replay_len), self.pad_token_id, dtype=torch.long, device=device,
            )
            for b in range(B):
                na_b = NA[b].item()
                # Copy accepted tokens
                replay_ids[b, :na_b] = sampled[b, old_curr + 1: old_curr + 1 + na_b]
                # Append the resampled next_token
                replay_ids[b, na_b] = next_token[b]

            replay_deltas = self.latent_mod_prep(
                torch.zeros(B, max_replay_len, self.V_H_DIM, device=device)
            )
            d_cache_pos_replay = torch.tensor(
                [d_next_pos_snapshot], device=device, dtype=torch.long
            )
            # Process replay tokens one at a time (single-step cached forward)
            for ri in range(max_replay_len):
                d_cache_pos_i = torch.tensor(
                    [d_next_pos_snapshot + ri], device=device, dtype=torch.long,
                )
                _, d_cache = self._drafter_forward_cached(
                    replay_ids[:, ri: ri + 1],
                    self.latent_mod_prep(
                        torch.zeros(B, 1, self.V_H_DIM, device=device)
                    ),
                    cache_params=d_cache,
                    cache_position=d_cache_pos_i,
                )
            d_next_pos = d_next_pos_snapshot + max_replay_len

        else:
            # --- Full re-prefill with COMPACT sequence ---
            # The non-compact layout leaves rejected tokens in sampled[].
            # The verifier masks them out via attention_mask, but the Mamba2
            # drafter is recurrent and has no attention mask — it processes
            # all tokens sequentially. Re-prefilling with the raw buffer
            # would feed rejected tokens into the drafter, corrupting its
            # recurrent state and causing compounding accuracy loss.
            #
            # Fix: build a compact sequence of only the unmasked tokens
            # (accepted + resampled), matching what the verifier attends to.
            d_cache = Mamba2Cache(
                self.d_base.backbone.config, B, device=device, dtype=torch.float32,
            )
            d_cache_pos = torch.arange(
                0, self.d_base.backbone.config.conv_kernel, device=device
            )
            # Include everything up through the resampled next_token
            prefill_end = curr + 1  # curr points to the resampled next_token
            mask_slice = attention_mask[:, :prefill_end]  # [B, prefill_end]

            # Build per-sample compact sequences (pad shorter ones)
            compact_lens = mask_slice.sum(dim=1).long()  # [B]
            max_compact_len = compact_lens.max().item()
            compact_ids = torch.full(
                (B, max_compact_len), self.pad_token_id,
                dtype=torch.long, device=device,
            )
            for b in range(B):
                unmasked = sampled[b, :prefill_end][mask_slice[b].bool()]
                compact_ids[b, :unmasked.shape[0]] = unmasked

            prefill_deltas = self.latent_mod_prep(
                torch.zeros(B, max_compact_len, self.V_H_DIM, device=device)
            )
            _, d_cache = self._drafter_forward_cached(
                compact_ids, prefill_deltas,
                cache_params=d_cache, cache_position=d_cache_pos,
            )
            d_next_pos = max_compact_len

        return (
            sampled, position_ids, attention_mask, NA, curr,
            guide, d_cache, d_next_pos, v_pkv, has_ended,
        )
