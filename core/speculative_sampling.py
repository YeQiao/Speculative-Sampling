import torch, time
from core.utils import sample_from_draft_model, sample, sample_from_draft_model_original ,get_distribution

def speculative_sampling(
    target_model,
    draft_model,
    initial_prompt_seq,
    target_len,
    tokenizer,
    lookahead=2,
    temperature=1.0,
    debug=False,
    collect_stats=False,
    profile=False,
    use_compile=False,
    use_target_cache=True,
):
    """Speculative decoding using block verification (optimized).

    Steps per cycle:
      1. Draft model proposes `lookahead` tokens (incremental draft caching inside helper).
      2. Target model runs ONCE on (prefix + proposed block) to obtain logits for each proposed step (+ one extra position).
      3. Acceptance loop uses target vs draft probabilities per proposed token.
      4. If all accepted and budget remains, sample one extra token from target (last logits position).
    
    Optimizations:
      - use_compile: Enable torch.compile for forward passes
      - use_target_cache: Enable KV cache for target model verification
      - Minimal stats tracking unless collect_stats=True
    """
    assert initial_prompt_seq.shape[0] == 1, "Batch size should be 1"

    fin_prompt_seq = initial_prompt_seq.detach().clone()
    accepted_tokens = 0
    total_draft_tokens = 0
    n = fin_prompt_seq.shape[-1]

    # Only create stats dict if needed
    if collect_stats or profile:
        stats = {
            'draft_time_s': 0.0,
            'target_time_s': 0.0,
            'accept_loop_time_s': 0.0,
            'cycles': 0,
            'lookahead': lookahead,
            'accepted_tokens': 0,
            'proposed_tokens': 0,
        }
        use_timing = True
    else:
        stats = None
        use_timing = False

    # Compile models if requested (only once per model)
    # Note: Only compile target model - draft model (Mamba) has torch.compile compatibility issues
    if use_compile and not hasattr(target_model, '_compiled'):
        try:
            target_model.forward = torch.compile(target_model.forward, mode='reduce-overhead')
            target_model._compiled = True
        except Exception as e:
            print(f"Warning: Failed to compile target model: {e}")
    
    # Skip compiling draft model for now due to Mamba compatibility issues with torch.compile
    # if use_compile and not hasattr(draft_model, '_compiled'):
    #     draft_model.forward = torch.compile(draft_model.forward, mode='reduce-overhead')
    #     draft_model._compiled = True

    # Target model KV cache
    past_key_values = None
    
    # Start total timing
    if profile:
        start_total = time.perf_counter()

    while n < target_len:
        prefix_len = fin_prompt_seq.shape[-1]

        # 1. Draft proposals (incremental inside helper)
        if use_timing:
            t0 = time.perf_counter()
        
        draft_extended, draft_logits = sample_from_draft_model(
            draft_model,
            fin_prompt_seq,
            new_tokens=lookahead,
            temperature=temperature,
            use_cache=False,
        )
        
        if use_timing:
            if torch.cuda.is_available(): torch.cuda.synchronize()
            stats['draft_time_s'] += time.perf_counter() - t0

        proposed_block = draft_extended[:, prefix_len:prefix_len+lookahead]
        if debug:
            print("Proposed:", tokenizer.decode(proposed_block[0], skip_special_tokens=True))

        # 2. Single target forward over extended sequence with optional KV cache
        if use_timing:
            t1 = time.perf_counter()
        
        if use_target_cache and past_key_values is not None:
            # Use KV cache - feed last accepted token + drafts for correct logit alignment
            # (cache covers 0..prefix_len-2, so token at prefix_len-1 is the first new input)
            new_input_ids = draft_extended[:, prefix_len - 1:]
            target_outputs = target_model(new_input_ids, past_key_values=past_key_values, use_cache=True)
            target_logits = target_outputs.logits
            past_key_values = target_outputs.past_key_values
        else:
            # No cache - process full sequence
            target_outputs = target_model(draft_extended, use_cache=use_target_cache)
            target_logits = target_outputs.logits[:, -lookahead-1:, :]
            if use_target_cache:
                past_key_values = target_outputs.past_key_values
        
        if use_timing:
            if torch.cuda.is_available(): torch.cuda.synchronize()
            stats['target_time_s'] += time.perf_counter() - t1

        # Extract target logits for the proposed block (both paths produce [1, K+1, V])
        target_block_logits = target_logits[:, :lookahead, :]
        
        accepted_all = True

        if use_timing:
            t2 = time.perf_counter()
        
        # Optimized acceptance loop - minimize tensor operations
        temp_inv = 1.0 / (temperature + 1e-10) if temperature > 0 else 1.0
        
        for t in range(lookahead):
            # Always track draft tokens for accurate acceptance rate (minimal overhead)
            total_draft_tokens += 1
            
            tgt_logits_t = target_block_logits[:, t, :]
            drf_logits_t = draft_logits[:, t, :]

            tgt_probs_t = torch.softmax(tgt_logits_t * temp_inv, dim=-1)
            drf_probs_t = torch.softmax(drf_logits_t * temp_inv, dim=-1)

            token_id = proposed_block[0, t]
            ratio = tgt_probs_t[0, token_id] / (drf_probs_t[0, token_id] + 1e-12)
            
            if torch.rand((), device=ratio.device) < torch.clamp(ratio, max=1.0):
                fin_prompt_seq = torch.cat([fin_prompt_seq, proposed_block[:, t:t+1]], dim=-1)
                accepted_tokens += 1
                n += 1
            else:
                residual = torch.clamp(tgt_probs_t - drf_probs_t, min=0.0)
                residual = residual / residual.sum(dim=-1, keepdim=True)
                fallback_token = torch.multinomial(residual, num_samples=1)
                fin_prompt_seq = torch.cat([fin_prompt_seq, fallback_token], dim=-1)
                n += 1
                accepted_all = False
                past_key_values = None  # Reset cache on rejection
                break

            if n >= target_len:
                accepted_all = False
                break

        if use_timing:
            if torch.cuda.is_available(): torch.cuda.synchronize()
            stats['accept_loop_time_s'] += time.perf_counter() - t2

        # 3. Extra target token if all accepted and still space
        if accepted_all and n < target_len:
            extra_logits = target_logits[:, -1, :]
            extra_token = sample(extra_logits, temperature=temperature)
            fin_prompt_seq = torch.cat([fin_prompt_seq, extra_token[None, ...]], dim=-1)
            n += 1

        if collect_stats:
            stats['cycles'] += 1
            stats['accepted_tokens'] = accepted_tokens
            stats['proposed_tokens'] = total_draft_tokens

        if debug:
            print("Accepted so far:", tokenizer.decode(fin_prompt_seq[0][initial_prompt_seq.shape[-1]:], skip_special_tokens=True))

    # End total timing (before stats computation to get accurate wall-clock time)
    if profile:
        if torch.cuda.is_available(): 
            torch.cuda.synchronize()
        total_time = time.perf_counter() - start_total
        stats['total_time_s'] = total_time
        stats['tokens_per_second'] = (fin_prompt_seq.shape[-1] - initial_prompt_seq.shape[-1]) / total_time if total_time > 0 else 0.0
    
    # Calculate acceptance rate (now always accurate since we track total_draft_tokens)
    acceptance_rate = accepted_tokens / total_draft_tokens if total_draft_tokens > 0 else 0.0
    
    if collect_stats or profile:
        stats['acceptance_rate'] = acceptance_rate
        stats['generated_tokens'] = fin_prompt_seq.shape[-1] - initial_prompt_seq.shape[-1]
        return fin_prompt_seq, acceptance_rate, stats
    
    return fin_prompt_seq, acceptance_rate


def speculative_sampling_original(target_model, draft_model, initial_prompt_seq, target_len, tokenizer, lookahead=2, temperature=1.0, debug=False):
    '''
    Implementation of Algorithm 2 of the paper - Accelerating Large Language Model Decoding 
    with Speculative Sampling (https://arxiv.org/abs/2302.01318)
    '''
    assert initial_prompt_seq.shape[0] == 1, 'Batch size should be 1'

    n = initial_prompt_seq.shape[-1]
    fin_prompt_seq = initial_prompt_seq.detach().clone()

    while n < target_len:
        n_orig = n
        N = fin_prompt_seq.shape[-1]
        draft_outputs, draft_logits = sample_from_draft_model_original(draft_model, fin_prompt_seq, new_tokens=lookahead, temperature=temperature)
        
        if debug:
            print(f"Possible continuations: {tokenizer.decode(draft_outputs[0,n_orig:], skip_special_tokens=True)}")

        target_logits = target_model(draft_outputs).logits[:, -lookahead-1:, :]

        target_model_distribution = get_distribution(target_logits, temperature)
        draft_model_distribution = get_distribution(draft_logits, temperature)

        accepted_flag = 1
        
        for t in range(lookahead):
            numerator = target_model_distribution[:, t, draft_outputs[0, N+t]]
            denominator = draft_model_distribution[:, t, draft_outputs[0, N+t]]
            ratio = (numerator / denominator)
            uniform_distribution = torch.rand_like(numerator)
            ones_tensor = torch.ones_like(numerator)

            # Rejection Sampling
            ## Acceptance
            if (uniform_distribution < torch.min(ones_tensor, ratio)).any():
                fin_prompt_seq = torch.concat([fin_prompt_seq, draft_outputs[:, N+t].unsqueeze(dim=-1)], dim=-1)
                n += 1

            ## Rejection
            else:
                new_dist = (target_model_distribution[:, t, :] - draft_model_distribution[:, t, :])
                new_dist = torch.max(torch.zeros_like(new_dist), new_dist)
                new_dist = new_dist / new_dist.sum(dim=-1, keepdim=True)
                token_id = torch.multinomial(new_dist, num_samples=1)[0]
                fin_prompt_seq = torch.concat([fin_prompt_seq, token_id[None,...]], dim=-1)
                accepted_flag = 0
                break

        if accepted_flag == 1:
            sample_token = sample(target_logits[:, -1, :], temperature=temperature)
            fin_prompt_seq = torch.concat([fin_prompt_seq, sample_token[None,...]], dim=-1)
        
        if debug:
            print(f"Accepted continuations: {tokenizer.decode(fin_prompt_seq[0,n_orig:], skip_special_tokens=True)}")

        n += 1

    return fin_prompt_seq