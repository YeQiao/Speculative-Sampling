"""
Focused diagnostic: compare drafter q and verifier p distributions in sampling mode.
Check if the probability distributions are reasonable for acceptance.
"""
import torch
from guided_mamba.train import GuidedMambaTrainer

CKPT = "/HSC/users/qiaoye/SSM_SPEC/checkpoints/guided_mamba/ckpts/last.ckpt"

def main():
    # Load model
    st = torch.load(CKPT, map_location="cpu", weights_only=False)
    hp = {k: v for k, v in st["hyper_parameters"].items() if k != "_instantiator"}
    mod = GuidedMambaTrainer(**hp)
    mod.load_state_dict(st["state_dict"], strict=False)
    mod.eval()
    mod.to("cuda")

    # Use a simple prompt
    prompts = ["What is the capital of France?"]
    input_ids, attention_mask = mod.prep_for_gen(prompts)
    B, S = input_ids.shape
    print(f"Prompt tokens: {S}")

    # Set to sampling mode
    mod.greedy_sample = False

    # Manually run one round to inspect distributions
    from transformers import DynamicCache
    from transformers.models.mamba2.modeling_mamba2 import Mamba2Cache

    device = input_ids.device
    NG = mod.NG

    sampled = torch.full((B, S + 128 * 4), mod.pad_token_id, dtype=torch.long, device=device)
    sampled[:, :S] = input_ids
    curr = S - 1

    # Verifier prefill
    v_pkv = DynamicCache()
    v_logits, guide_embd, v_pkv = mod._verifier_forward_cached(
        sampled[:, :curr + 1], past_key_values=v_pkv,
    )
    guide = mod.latent_mod_prep(guide_embd[:, -1:, :])

    # Drafter prefill
    d_cache = Mamba2Cache(mod.d_base.backbone.config, B, device=device, dtype=torch.float32)
    d_cache_pos = torch.arange(0, mod.d_base.backbone.config.conv_kernel, device=device)
    if curr > 0:
        prefill_deltas = mod.latent_mod_prep(torch.zeros(B, curr, mod.V_H_DIM, device=device))
        _, d_cache = mod._drafter_forward_cached(
            sampled[:, :curr], prefill_deltas, cache_params=d_cache, cache_position=d_cache_pos,
        )
    d_next_pos = curr

    # Draft NG tokens
    q = torch.zeros(B, NG, mod.v_base.config.vocab_size, device=device)
    for i in range(NG):
        d_cache_pos_i = torch.tensor([d_next_pos + i], device=device, dtype=torch.long)
        d_logits_i, d_cache = mod._drafter_forward_cached(
            sampled[:, curr + i:curr + i + 1], guide,
            cache_params=d_cache, cache_position=d_cache_pos_i,
        )
        d_prob = d_logits_i[:, -1].softmax(-1)
        q[:, i] = d_prob

        # Draft with multinomial (sampling mode)
        sampled[:, curr + i + 1] = torch.multinomial(d_prob, 1).squeeze(1)

    # Verify
    v_logits, guide_embd_new, v_pkv = mod._verifier_forward_cached(
        sampled[:, curr:curr + NG + 1], past_key_values=v_pkv,
    )
    v_probs_fp16 = v_logits.softmax(-1)
    v_probs_fp32 = v_logits.float().softmax(-1)

    # Analyze for each drafted position
    for i in range(NG):
        tok = sampled[0, curr + i + 1].item()
        tok_str = mod.tok.decode([tok])
        q_val = q[0, i, tok].item()
        p_val_fp16 = v_probs_fp16[0, i, tok].item()
        p_val_fp32 = v_probs_fp32[0, i, tok].item()

        # What's the drafter's top prediction?
        d_top = q[0, i].argmax().item()
        d_top_str = mod.tok.decode([d_top])
        d_top_prob = q[0, i, d_top].item()

        # What's the verifier's top prediction?
        v_top = v_probs_fp32[0, i].argmax().item()
        v_top_str = mod.tok.decode([v_top])
        v_top_prob = v_probs_fp32[0, i, v_top].item()

        ratio = p_val_fp32 / (q_val + 1e-12)
        print(f"  Pos {i}: sampled='{tok_str}' (id={tok})")
        print(f"    q(tok)={q_val:.6f}, p_fp16(tok)={p_val_fp16:.6f}, p_fp32(tok)={p_val_fp32:.6f}")
        print(f"    ratio p/q={ratio:.4f} {'ACCEPT' if ratio >= 1 else 'likely reject'}")
        print(f"    drafter top: '{d_top_str}' q={d_top_prob:.4f}")
        print(f"    verifier top: '{v_top_str}' p={v_top_prob:.4f}")
        print(f"    argmax match: {d_top == v_top}")
        print()

    # Entropy analysis across all positions
    print("=== ENTROPY ANALYSIS ===")
    for i in range(NG):
        d_entropy = -(q[0, i] * (q[0, i] + 1e-12).log()).sum().item()
        v_entropy = -(v_probs_fp32[0, i] * (v_probs_fp32[0, i] + 1e-12).log()).sum().item()
        print(f"  Pos {i}: drafter H={d_entropy:.3f} nats, verifier H={v_entropy:.3f} nats, "
              f"{'drafter MORE peaked' if d_entropy < v_entropy else 'verifier MORE peaked'}")

    # Run multiple seeds to get average statistics
    print("\n=== MULTI-SEED ANALYSIS (10 drafting rounds) ===")
    n_rounds = 10
    total_accept = 0
    total_tok = 0
    total_argmax_match = 0
    total_pq_when_match = 0
    n_match = 0

    # Reset for multi-round
    torch.manual_seed(42)
    for r in range(n_rounds):
        for i in range(NG):
            tok = torch.multinomial(q[0, i:i+1], 1).squeeze().item()
            q_val = q[0, i, tok].item()
            p_val = v_probs_fp32[0, i, tok].item()
            ratio = p_val / (q_val + 1e-12)
            accepted = torch.rand(1).item() < min(1.0, ratio)
            total_tok += 1
            if accepted:
                total_accept += 1
            # Check argmax match
            if q[0, i].argmax().item() == v_probs_fp32[0, i].argmax().item():
                total_argmax_match += 1
                total_pq_when_match += min(1.0, p_val / (q_val + 1e-12))
                n_match += 1

    print(f"  Per-token acceptance rate: {total_accept/total_tok:.3f}")
    print(f"  Argmax match rate: {total_argmax_match/total_tok:.3f}")
    if n_match > 0:
        print(f"  Avg p/q when argmax matches: {total_pq_when_match/n_match:.3f}")


if __name__ == "__main__":
    with torch.no_grad():
        main()
