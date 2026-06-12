"""Quick A/B test: base drafter weights vs fine-tuned guided drafter weights.

Measures greedy acceptance rate on the original hardcoded prompts to validate
that the drafter_sd bug fix actually improves alignment.
"""
import os, torch, time
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import DynamicCache

PROMPTS = [
    "Give me a brief history of the Roman Empire.",
    "Explain the concept of supply and demand in economics.",
    "Write a Python function that finds the longest common subsequence of two strings.",
    "Implement a binary search tree in Python with insert and search methods.",
    "Solve the equation: 3x^2 - 12x + 9 = 0. Show your work step by step.",
    "A rectangular garden has a perimeter of 56 meters. If the length is 4 meters more than the width, find the dimensions.",
]

DRAFTER_PATH = "/HSC/users/qiaoye/SSM_SPEC/checkpoints/improved-mamba-alignment/checkpoint-750"
GUIDED_CKPT  = "/HSC/users/qiaoye/SSM_SPEC/checkpoints/guided_mamba/ckpts/step-step=31250-val_loss-val/loss=0.3346.ckpt"
VERIFIER_PATH = "/HSC/users/qiaoye/checkpoints/Llama3.1-8B-hf"
NG = 8
MAX_NEW = 64

os.environ.setdefault("OMP_NUM_THREADS", "16")
torch.set_num_threads(16)

# ---- Load verifier ----
print("Loading verifier (8B)...")
import transformers.models.mamba2.modeling_mamba2 as _m2
_m2.is_fast_path_available = False
verifier = AutoModelForCausalLM.from_pretrained(VERIFIER_PATH, torch_dtype=torch.float16).cuda().eval()
tokenizer = AutoTokenizer.from_pretrained(VERIFIER_PATH)

# ---- Load guidance modules ----
print("Loading guidance modules...")
from spec_mamba.trainer import GuidanceExtractor, PrepMambaDeltas
ckpt = torch.load(GUIDED_CKPT, map_location="cpu", weights_only=False)
sd = ckpt["state_dict"]
hp = ckpt["hyper_parameters"]
v_layers = hp["v_layers"]

ge_weight = sd["guidance_extractor.proj.weight"]
v_h_dim = ge_weight.shape[0]
pm_weight = sd["latent_mod_prep.proj.weight"]
n_guide_layers = 16
delta_dim = pm_weight.shape[0] // n_guide_layers

ge = GuidanceExtractor(v_h_dim=v_h_dim, n_layers=len(v_layers))
ge.in_layer = [v + 1 for v in v_layers]
ge.proj.weight.data.copy_(sd["guidance_extractor.proj.weight"])
ge.proj.bias.data.copy_(sd["guidance_extractor.proj.bias"])
ge = ge.cuda().eval()

prep = PrepMambaDeltas(v_h_dim=v_h_dim, delta_dim=delta_dim, n_layers=n_guide_layers)
prep.norm.weight.data.copy_(sd["latent_mod_prep.norm.weight"])
prep.norm.bias.data.copy_(sd["latent_mod_prep.norm.bias"])
prep.proj.weight.data.copy_(sd["latent_mod_prep.proj.weight"])
prep = prep.cuda().eval()

drafter_sd_raw = {k.removeprefix("d_base."): v for k, v in sd.items() if k.startswith("d_base.")}
drafter_sd_hf = {k.replace(".mixer.m.", ".mixer."): v.float() for k, v in drafter_sd_raw.items()}
del ckpt, sd


def build_cpu_drafter(use_finetuned: bool):
    hf = AutoModelForCausalLM.from_pretrained(DRAFTER_PATH, torch_dtype=torch.float32).eval()
    if use_finetuned:
        missing, unexpected = hf.load_state_dict(drafter_sd_hf, strict=False)
        assert not missing and not unexpected, f"missing={missing}, unexpected={unexpected}"
    from spec_mamba.cpu_mamba2 import Int8FusedCPUMamba2Model
    cpu = Int8FusedCPUMamba2Model(hf)
    del hf
    return cpu


def extract_guidance(verifier, ge, prep, input_ids_v, v_pkv):
    v_out = verifier.model(input_ids_v, past_key_values=v_pkv, use_cache=True, output_hidden_states=True)
    v_pkv = v_out.past_key_values
    guide_input = None
    for idx in ge.in_layer:
        h = v_out.hidden_states[idx]
        guide_input = h if guide_input is None else torch.cat((guide_input, h), dim=-1)
    v_out.hidden_states = None
    guide_embd = ge.proj(guide_input.to(ge.proj.weight.dtype))
    guide_last = guide_embd[:, -1, None]
    deltas = prep(guide_last)
    deltas_cpu = deltas.squeeze(1).squeeze(1).float().cpu()
    v_logits = verifier.lm_head(v_out.last_hidden_state)
    return v_logits, deltas_cpu, v_pkv


def run_trial(cpu_drafter, label):
    all_greedy_acc = []
    all_stoch_acc = []
    for prompt in PROMPTS:
        # Use chat template to match real benchmark conditions
        if tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt}]
            tok_out = tokenizer.apply_chat_template(
                messages, return_tensors="pt",
                add_generation_prompt=True, return_dict=True,
            )
            input_ids = tok_out["input_ids"]
        else:
            input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
        input_ids_v = input_ids.cuda()

        # Verifier prefill + get initial guidance
        v_pkv = DynamicCache()
        torch.cuda.synchronize()
        v_out = verifier.model(input_ids_v, past_key_values=v_pkv, use_cache=True, output_hidden_states=True)
        v_pkv = v_out.past_key_values
        first_tok = verifier.lm_head(v_out.last_hidden_state[:, -1:]).argmax(dim=-1).item()

        guide_input = None
        for idx in ge.in_layer:
            h = v_out.hidden_states[idx]
            guide_input = h if guide_input is None else torch.cat((guide_input, h), dim=-1)
        v_out.hidden_states = None
        guide_embd = ge.proj(guide_input.to(ge.proj.weight.dtype))
        guide_last = guide_embd[:, -1, None]
        deltas = prep(guide_last)
        cur_deltas = deltas.squeeze(1).squeeze(1).float().cpu()
        torch.cuda.synchronize()

        # CPU drafter prefill
        conv, ssm = cpu_drafter.create_cache(batch_size=1)
        cpu_drafter.prefill(input_ids.cpu(), conv, ssm)

        current_tok = first_tok
        generated = [first_tok]

        while len(generated) < MAX_NEW:
            snap_conv = conv.clone()
            snap_ssm = ssm.clone()
            K = min(NG, MAX_NEW - len(generated))

            # Draft K tokens with guidance
            draft_toks = []
            draft_probs_list = []
            tok = current_tok
            for _ in range(K):
                logits = cpu_drafter.forward_step(tok, conv, ssm, guidance_deltas=cur_deltas)
                probs = logits.softmax(dim=-1).squeeze(0)
                tok = logits.argmax(dim=-1).item()
                draft_toks.append(tok)
                draft_probs_list.append(probs)

            # Verify
            verify_input = torch.tensor([[current_tok] + draft_toks], dtype=torch.long, device="cuda")
            torch.cuda.synchronize()
            v_logits, new_deltas, v_pkv = extract_guidance(verifier, ge, prep, verify_input, v_pkv)
            torch.cuda.synchronize()
            cur_deltas = new_deltas

            # Count accepted (greedy)
            v_preds = v_logits[0].argmax(dim=-1).cpu()
            n_acc = 0
            for j in range(K):
                if draft_toks[j] == v_preds[j].item():
                    n_acc += 1
                else:
                    break
            next_tok = v_preds[n_acc].item()
            all_accepted.append(n_acc)

            accepted = draft_toks[:n_acc] + [next_tok]
            generated.extend(accepted)
            current_tok = next_tok

            # Resync drafter cache
            conv.copy_(snap_conv); ssm.copy_(snap_ssm)
            for t in accepted:
                cpu_drafter.forward_step(t, conv, ssm, guidance_deltas=cur_deltas)

            if next_tok == tokenizer.eos_token_id:
                break

        n_rounds = len(all_accepted) if all_accepted else 1
        avg = sum(all_accepted[-n_rounds:]) / n_rounds

    overall_avg = sum(all_accepted) / len(all_accepted)
    n_rounds = len(all_accepted)
    print(f"  [{label}] avg_accepted = {overall_avg:.3f} / {NG}  ({n_rounds} rounds across {len(PROMPTS)} prompts)")
    return overall_avg


print("\n" + "="*60)
print("A/B: base drafter vs fine-tuned drafter (guided 27M+8B)")
print("="*60)

print("\nBuilding BASE drafter (original pretrained weights)...")
drafter_base = build_cpu_drafter(use_finetuned=False)
avg_base = run_trial(drafter_base, "BASE   (no drafter_sd)")
del drafter_base

print("\nBuilding FINE-TUNED drafter (guided checkpoint weights)...")
drafter_ft = build_cpu_drafter(use_finetuned=True)
avg_ft = run_trial(drafter_ft, "FINE-TUNED (drafter_sd applied)")
del drafter_ft

print(f"\n{'='*60}")
print(f"Summary:")
print(f"  Base:       {avg_base:.3f} / {NG}")
print(f"  Fine-tuned: {avg_ft:.3f} / {NG}")
print(f"  Delta:      {avg_ft - avg_base:+.3f}")
print(f"{'='*60}")
