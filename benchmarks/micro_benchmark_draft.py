import argparse
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import sample_from_draft_model

def measure(model, prompt_len=128, new_tokens=64, use_cache=True, runs=3, temperature=1.0, return_stats=False):
    device = next(model.parameters()).device
    # Simple prompt of a single token repeated; assumes token id 1 is valid (avoid special BOS=0 for variety)
    prompt_token_id = 1
    prompt = torch.full((1, prompt_len), prompt_token_id, dtype=torch.long, device=device)
    # Warmup
    _ = sample_from_draft_model(model, prompt, min(2, new_tokens), temperature=temperature, use_cache=use_cache)

    times = []
    agg_stats = None
    for _ in range(runs):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.perf_counter()
        if return_stats:
            _, _, stats = sample_from_draft_model(model, prompt, new_tokens, temperature=temperature, use_cache=use_cache, return_stats=True)
        else:
            _, _ = sample_from_draft_model(model, prompt, new_tokens, temperature=temperature, use_cache=use_cache)
            stats = None
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t1 = time.perf_counter()
        times.append(t1 - t0)
        if stats:
            if agg_stats is None:
                agg_stats = {k: 0 for k in stats}
            for k,v in stats.items():
                if isinstance(v, (int, float)):
                    agg_stats[k] += v
    avg_time = sum(times)/len(times)
    tok_per_s = new_tokens / avg_time
    if return_stats and agg_stats:
        for k in agg_stats:
            agg_stats[k] /= runs
    return tok_per_s, avg_time, agg_stats


def main():
    parser = argparse.ArgumentParser(description="Micro-benchmark draft model incremental decoding")
    parser.add_argument('--model', type=str, required=True, help='Path or HF id of draft model')
    parser.add_argument('--prompt-lens', type=int, nargs='*', default=[64,128,256], help='Prompt lengths to test')
    parser.add_argument('--new-tokens', type=int, default=64)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--dtype', type=str, default='bfloat16', choices=['float16','bfloat16','float32'])
    parser.add_argument('--no-cache', action='store_true', help='Disable cache/state usage')
    parser.add_argument('--compare', action='store_true', help='Also benchmark with cache disabled for comparison')
    args = parser.parse_args()

    dtype_map = {
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
        'float32': torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading model {args.model} ...")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch_dtype, device_map=None)
    model.to(device)
    model.eval()

    # Display forward signature for inspection
    import inspect
    try:
        sig = inspect.signature(model.forward)
        print("Forward signature:", sig)
    except Exception as e:
        print("Could not inspect forward signature:", e)

    print("Attributes present on model output likely: logits, past_key_values, state/cache/mamba_state (if supported)")

    header = f"{'prompt_len':>10} | {'cache?':>6} | {'tok/s':>10} | {'avg_s':>8} | incr_tokens | full_tokens | incr_ms | full_ms | used_incr"
    print(header)
    print('-'*len(header))

    for L in args.prompt_lens:
        tokps, avg, stats = measure(model, prompt_len=L, new_tokens=args.new_tokens, use_cache=not args.no_cache, runs=args.runs, return_stats=True)
        row = [L, 'yes' if not args.no_cache else 'no', f"{tokps:10.2f}", f"{avg:8.3f}"]
        if stats:
            row.extend([
                f"{int(stats['incremental_tokens']):11d}",
                f"{int(stats['full_recompute_tokens']):11d}",
                f"{stats['incremental_time_ms']:7.1f}",
                f"{stats['full_recompute_time_ms']:7.1f}",
                f"{str(stats['used_incremental']):9s}",
            ])
        print(' | '.join(map(str,row)))
        if args.compare and not args.no_cache:
            tokps_nc, avg_nc, stats_nc = measure(model, prompt_len=L, new_tokens=args.new_tokens, use_cache=False, runs=args.runs, return_stats=True)
            row = [L, 'no', f"{tokps_nc:10.2f}", f"{avg_nc:8.3f}"]
            if stats_nc:
                row.extend([
                    f"{int(stats_nc['incremental_tokens']):11d}",
                    f"{int(stats_nc['full_recompute_tokens']):11d}",
                    f"{stats_nc['incremental_time_ms']:7.1f}",
                    f"{stats_nc['full_recompute_time_ms']:7.1f}",
                    f"{str(stats_nc['used_incremental']):9s}",
                ])
            print(' | '.join(map(str,row)))

    print("\nInterpretation hints:")
    print("  - incremental_tokens should equal new_tokens when caching works.")
    print("  - full_recompute_tokens > 0 indicates fallback or no cache path.")
    print("  - If used_incremental is False, we didn't detect state/past_key_values.")
    print("  - Compare tok/s between cache yes vs no to estimate benefit.")

if __name__ == '__main__':
    main()
