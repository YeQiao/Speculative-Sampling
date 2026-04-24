import torch
import inspect
import time as _time

def get_distribution(logits, temperature):
    probs = torch.softmax(logits / (temperature + 1e-10), dim=-1)
    return probs

def sample(logits, temperature):
    probs = get_distribution(logits, temperature)
    return torch.multinomial(probs, num_samples=1)[0]

def _extract_attr(obj, *names):
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
        if isinstance(obj, dict) and n in obj:
            return obj[n]
    return None

def sample_from_draft_model_original(model, initial_prompt_seq, new_tokens, temperature=1.0):
    fin_prompt_seq = initial_prompt_seq.detach().clone()
    out_logits = []

    for _ in range(new_tokens):
        sample_token_logits = model(fin_prompt_seq).logits[:, -1, :]
        sample_token = sample(sample_token_logits, temperature=temperature)
        fin_prompt_seq = torch.concat([fin_prompt_seq, sample_token[None,...]], dim=-1)
        out_logits.append(sample_token_logits)

    out_logits = torch.stack(out_logits, dim=1)
    return fin_prompt_seq, out_logits

def sample_from_draft_model(
    model,
    initial_prompt_seq,
    new_tokens,
    temperature=1.0,
    use_cache=True,
    return_stats=False,
    profile=False,
):
    """
    Generate draft tokens using (optional) model caching/state.

    Attempts, in order:
      1. Transformer-style past_key_values incremental decoding (HuggingFace API)
      2. Mamba-style state passing (looks for forward(state=...) or forward(cache=...))
      3. Fallback full-prefix recompute each token

    Returns:
      fin_prompt_seq: (1, original_len + new_tokens)
      out_logits: (1, new_tokens, vocab)
    """
    fin_prompt_seq = initial_prompt_seq.detach().clone()
    logits_list = []

    # Quick exit
    if new_tokens <= 0:
        return fin_prompt_seq, torch.empty((1,0, model.config.vocab_size), device=fin_prompt_seq.device)

    stats = {
        'used_incremental': False,
        'incremental_tokens': 0,
        'full_recompute_tokens': 0,
        'initial_forward_time_ms': 0.0,
        'incremental_time_ms': 0.0,
        'full_recompute_time_ms': 0.0,
        'state_param_name': None,
        'cache_type': None,
    }

    def _sync():
        if profile and torch.cuda.is_available():
            torch.cuda.synchronize()

    # Use no_grad (not inference_mode) to avoid RuntimeError when downstream needs autograd on target logits
    # Inference tensors forbid saving for backward; speculative pipeline may still build graphs for target model elsewhere.
    with torch.no_grad():
        past_key_values = None
        recurrent_state = None
        cache_params = None  # For Mamba2 style cache
        cache_position = None  # Tensor positions (if model requires explicit positions)
        next_position_index = None  # Integer position counter for Mamba2
        use_incremental = False
        state_param_name = None

        if use_cache:
            # Cache signature parsing on model to avoid repeated introspection cost
            if not hasattr(model, '_draft_sig_cached'):
                try:
                    sig = inspect.signature(model.forward)
                    model._draft_sig_params = set(sig.parameters.keys())
                except Exception:
                    model._draft_sig_params = set()
                model._draft_sig_cached = True
            params = getattr(model, '_draft_sig_params', set())
            for candidate in ["state", "cache", "mamba_state", "cache_params"]:
                if candidate in params:
                    state_param_name = candidate
                    break

            # Prime caches only once per call (cannot persist across cycles yet because
            # speculative acceptance may insert tokens unseen by draft model). This is
            # still cheaper than k full-prefix passes when lookahead>1.
            try:
                _sync(); _t0 = _time.perf_counter()
                initial_out = model(fin_prompt_seq, use_cache=True)
            except TypeError:
                _sync(); _t0 = _time.perf_counter()
                initial_out = model(input_ids=fin_prompt_seq, use_cache=True)
            except Exception:
                _sync(); _t0 = _time.perf_counter()
                initial_out = model(fin_prompt_seq)
            _sync(); stats['initial_forward_time_ms'] = (_time.perf_counter() - _t0) * 1000.0

            past_key_values = _extract_attr(initial_out, 'past_key_values')
            recurrent_state = _extract_attr(initial_out, 'state', 'cache', 'mamba_state')
            cache_params = _extract_attr(initial_out, 'cache_params')

            if cache_params is not None:
                seq_len = fin_prompt_seq.shape[-1]
                try:
                    cache_position = torch.arange(seq_len, device=fin_prompt_seq.device, dtype=torch.long)
                except Exception:
                    cache_position = None
                next_position_index = seq_len
                use_incremental = True
                stats['used_incremental'] = True
                stats['cache_type'] = 'cache_params'
                stats['state_param_name'] = 'cache_params'
            elif past_key_values is not None or recurrent_state is not None:
                use_incremental = True
                stats['used_incremental'] = True
                stats['state_param_name'] = state_param_name
                stats['cache_type'] = 'kv' if past_key_values is not None else 'recurrent'

        # Use priming logits for first draft token to avoid double-processing last input token
        remaining_tokens = new_tokens
        if use_incremental and new_tokens > 0:
            first_logits = initial_out.logits[:, -1, :]
            first_token = sample(first_logits, temperature=temperature)
            fin_prompt_seq = torch.cat([fin_prompt_seq, first_token[None, ...]], dim=-1)
            logits_list.append(first_logits)
            remaining_tokens = new_tokens - 1

        for _ in range(remaining_tokens):
            if use_incremental:
                last_token = fin_prompt_seq[:, -1:]
                try:
                    _sync(); _t0i = _time.perf_counter()
                    if cache_params is not None:
                        # Provide current next position for the new token
                        try:
                            if next_position_index is None:
                                next_position_index = fin_prompt_seq.shape[-1]
                            cache_position = torch.tensor([next_position_index], device=fin_prompt_seq.device, dtype=torch.long)
                        except Exception:
                            cache_position = None
                        # Attempt incremental call
                        try:
                            out = model(input_ids=last_token, cache_params=cache_params, cache_position=cache_position, use_cache=True)
                        except TypeError:
                            out = model(input_ids=last_token, cache_params=cache_params, use_cache=True)
                        # Update cache_params if returned
                        cache_params = _extract_attr(out, 'cache_params') or cache_params
                        if next_position_index is not None:
                            next_position_index += 1
                    elif past_key_values is not None:
                        out = model(input_ids=last_token, past_key_values=past_key_values, use_cache=True)
                        past_key_values = _extract_attr(out, 'past_key_values') or past_key_values
                    elif recurrent_state is not None:
                        # Build kwargs dynamically
                        kwargs = {state_param_name: recurrent_state} if state_param_name else {}
                        out = model(input_ids=last_token, **kwargs)
                        recurrent_state = _extract_attr(out, 'state', 'cache', 'mamba_state') or recurrent_state
                    else:
                        # Should not reach here; fallback full forward
                        out = model(fin_prompt_seq)
                    _sync(); stats['incremental_time_ms'] += (_time.perf_counter() - _t0i) * 1000.0
                    stats['incremental_tokens'] += 1
                except Exception:
                    # Fallback to full sequence recompute if incremental fails mid-way
                    use_incremental = False
                    cache_params = None
                    out = model(fin_prompt_seq)
                    _sync()
                    stats['full_recompute_tokens'] += 1
            else:
                _sync(); _t0f = _time.perf_counter()
                out = model(fin_prompt_seq)
                _sync(); stats['full_recompute_time_ms'] += (_time.perf_counter() - _t0f) * 1000.0
                stats['full_recompute_tokens'] += 1

            token_logits = out.logits[:, -1, :]
            next_token = sample(token_logits, temperature=temperature)
            fin_prompt_seq = torch.cat([fin_prompt_seq, next_token[None, ...]], dim=-1)
            logits_list.append(token_logits)

    out_logits = torch.stack(logits_list, dim=1)
    if return_stats:
        return fin_prompt_seq, out_logits, stats
    return fin_prompt_seq, out_logits