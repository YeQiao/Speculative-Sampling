#!/usr/bin/env python3
"""
Profile the sample_from_draft_model() function to compare draft model speeds.
Tests Mamba vs Llama-1B for drafting performance.
"""

import os
import torch
import time
import sys
from pathlib import Path
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from transformers import AutoTokenizer, AutoModelForCausalLM
from core.utils import sample_from_draft_model

device = "cuda" if torch.cuda.is_available() else "cpu"


def profile_draft_model(model_path, num_tokens=5, prompt_length=32, use_cache=True):
    """Profile sample_from_draft_model() throughput"""
    print(f"{'='*80}")
    print(f"Profiling Draft Model: {model_path}")
    if torch.cuda.is_available():
        print(f"Device: CUDA - {torch.cuda.get_device_name(0)}")
    else:
        print(f"Device: CPU")
    print(f"Prompt length: {prompt_length}, Draft tokens: {num_tokens}")
    print(f"Cache: {'Enabled' if use_cache else 'Disabled'}")
    print(f"{'='*80}\n")
    
    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Use same loading strategy as test_mamba_model.py for optimal CPU performance
    if device == "cpu":
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map={"": "cpu"}  # Force CPU explicitly
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    
    model.eval()
    
    # Prepare input
    prompt = "The quick brown fox jumps over the lazy dog. " * (prompt_length // 10)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    
    # Warmup
    print("Warming up...")
    with torch.no_grad():
        for _ in range(3):
            _, _ = sample_from_draft_model(
                model, input_ids, new_tokens=num_tokens, 
                temperature=0.0, use_cache=use_cache
            )
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    print("\n" + "="*80)
    print("THROUGHPUT MEASUREMENT")
    print("="*80)
    
    num_runs = 20
    times = []
    
    for i in range(num_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        with torch.no_grad():
            output, logits = sample_from_draft_model(
                model, input_ids, new_tokens=num_tokens,
                temperature=0.0, use_cache=use_cache
            )
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time
        times.append(elapsed)
        
        tokens_generated = output.shape[1] - input_ids.shape[1]
        throughput = tokens_generated / elapsed
        
        print(f"Run {i+1:2d}: {elapsed*1000:.2f}ms, {throughput:.2f} tok/s")
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    avg_throughput = num_tokens / avg_time
    
    print(f"\nAverage: {avg_time*1000:.2f}ms ± {std_time*1000:.2f}ms")
    print(f"Average throughput: {avg_throughput:.2f} tok/s")
    print(f"Time per token: {avg_time/num_tokens*1000:.2f}ms")
    
    # Test with stats and memory tracking
    print("\n" + "="*80)
    print("CACHE BEHAVIOR & TIMING BREAKDOWN")
    print("="*80)
    
    # Measure memory before
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated() / (1024**2)  # MB
    
    with torch.no_grad():
        _, _, stats = sample_from_draft_model(
            model, input_ids, new_tokens=num_tokens,
            temperature=0.0, use_cache=use_cache, return_stats=True, profile=True
        )
    
    # Measure memory after
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        mem_after = torch.cuda.memory_allocated() / (1024**2)  # MB
        mem_peak = torch.cuda.max_memory_allocated() / (1024**2)  # MB
        mem_used = mem_after - mem_before
        mem_peak_increase = mem_peak - mem_before
        
        print(f"Used incremental: {stats['used_incremental']}")
        print(f"Cache type: {stats['cache_type']}")
        print(f"State param: {stats['state_param_name']}")
        print(f"\n--- Memory Usage ---")
        print(f"Memory increase: {mem_used:.2f} MB")
        print(f"Peak memory increase: {mem_peak_increase:.2f} MB")
        print(f"Total allocated: {mem_after:.2f} MB")
    else:
        print(f"Used incremental: {stats['used_incremental']}")
        print(f"Cache type: {stats['cache_type']}")
        print(f"State param: {stats['state_param_name']}")
        print(f"\n--- Memory Usage ---")
        print(f"Memory tracking not available on CPU")
    print(f"\n--- Timing Breakdown ---")
    print(f"Prefill time: {stats['initial_forward_time_ms']:.2f}ms")
    print(f"Incremental tokens: {stats['incremental_tokens']}")
    print(f"Full recompute tokens: {stats['full_recompute_tokens']}")
    if stats['incremental_tokens'] > 0:
        print(f"Total incremental time: {stats['incremental_time_ms']:.2f}ms")
        print(f"Avg incremental time: {stats['incremental_time_ms']/stats['incremental_tokens']:.2f}ms per token")
    if stats['full_recompute_tokens'] > 0:
        print(f"Total full recompute time: {stats['full_recompute_time_ms']:.2f}ms")
        print(f"Avg full recompute time: {stats['full_recompute_time_ms']/stats['full_recompute_tokens']:.2f}ms per token")
    
    total_time = stats['initial_forward_time_ms'] + stats['incremental_time_ms'] + stats['full_recompute_time_ms']
    print(f"\nTotal time: {total_time:.2f}ms")
    if stats['incremental_tokens'] > 0:
        print(f"Prefill throughput: {prompt_length / (stats['initial_forward_time_ms']/1000):.2f} tok/s")
        print(f"Generation throughput: {stats['incremental_tokens'] / (stats['incremental_time_ms']/1000):.2f} tok/s")
    
    del model
    torch.cuda.empty_cache()
    
    return {
        'model_path': model_path,
        'avg_time_ms': avg_time * 1000,
        'std_time_ms': std_time * 1000,
        'avg_throughput': avg_throughput,
        'time_per_token_ms': avg_time / num_tokens * 1000,
        'cache_enabled': use_cache,
        'used_incremental': stats['used_incremental'],
        'cache_type': stats['cache_type'],
    }


def profile_prefill_vs_generation(model_path, prompt_length=512, num_tokens=5):
    """Separately measure prefill and per-token generation speed"""
    print(f"\n{'='*80}")
    print(f"PREFILL vs GENERATION SPEED")
    print(f"Model: {model_path}")
    if torch.cuda.is_available():
        print(f"Device: CUDA - {torch.cuda.get_device_name(0)}")
    else:
        print(f"Device: CPU")
    print(f"{'='*80}\n")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Use same loading strategy as test_mamba_model.py for optimal CPU performance
    if device == "cpu":
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map={"": "cpu"}  # Force CPU explicitly
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    
    model.eval()
    
    prompt = "The quick brown fox jumps over the lazy dog. " * (prompt_length // 10)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(input_ids, use_cache=True)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Measure prefill only
    print("Measuring prefill (full prompt processing)...")
    prefill_times = []
    for i in range(10):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        
        with torch.no_grad():
            out = model(input_ids, use_cache=True)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        elapsed = time.perf_counter() - start
        prefill_times.append(elapsed * 1000)  # ms
        print(f"  Run {i+1}: {elapsed*1000:.2f}ms")
    
    avg_prefill = np.mean(prefill_times)
    std_prefill = np.std(prefill_times)
    prefill_throughput = prompt_length / (avg_prefill / 1000)
    
    print(f"\nPrefill: {avg_prefill:.2f}ms ± {std_prefill:.2f}ms")
    print(f"Prefill throughput: {prefill_throughput:.2f} tok/s")
    
    # Measure single-token generation with cache
    print("\nMeasuring per-token generation (with cache)...")
    
    # Initialize cache
    with torch.no_grad():
        initial_out = model(input_ids, use_cache=True)
        cache_params = _extract_attr(initial_out, 'cache_params')
        past_key_values = _extract_attr(initial_out, 'past_key_values')
    
    gen_times = []
    for i in range(20):
        # Create a new single token input
        next_token = torch.randint(0, model.config.vocab_size, (1, 1), device=device)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        
        with torch.no_grad():
            if cache_params is not None:
                cache_position = torch.tensor([input_ids.shape[1] + i], device=device, dtype=torch.long)
                try:
                    out = model(input_ids=next_token, cache_params=cache_params, 
                              cache_position=cache_position, use_cache=True)
                except:
                    out = model(input_ids=next_token, cache_params=cache_params, use_cache=True)
                cache_params = _extract_attr(out, 'cache_params') or cache_params
            elif past_key_values is not None:
                out = model(input_ids=next_token, past_key_values=past_key_values, use_cache=True)
                past_key_values = _extract_attr(out, 'past_key_values') or past_key_values
            else:
                # No cache available
                out = model(input_ids=next_token)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        gen_times.append(elapsed * 1000)  # ms
        print(f"  Token {i+1}: {elapsed*1000:.2f}ms")
    
    avg_gen = np.mean(gen_times)
    std_gen = np.std(gen_times)
    gen_throughput = 1000 / avg_gen  # tok/s
    
    print(f"\nPer-token generation: {avg_gen:.2f}ms ± {std_gen:.2f}ms")
    print(f"Generation throughput: {gen_throughput:.2f} tok/s")
    
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Prefill: {avg_prefill:.2f}ms ({prefill_throughput:.2f} tok/s) for {prompt_length} tokens")
    print(f"  Per-token cost: {avg_prefill/prompt_length:.4f}ms")
    print(f"Generation: {avg_gen:.2f}ms per token ({gen_throughput:.2f} tok/s)")
    print(f"Generation is {avg_gen/(avg_prefill/prompt_length):.1f}x slower per token than prefill")
    
    del model
    torch.cuda.empty_cache()


def _extract_attr(obj, *names):
    """Extract attribute from object or dict"""
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
        if isinstance(obj, dict) and n in obj:
            return obj[n]
    return None


def compare_cache_impact(model_path, num_tokens=5, prompt_length=512):
    """Compare performance and memory with and without cache"""
    print(f"\n{'='*80}")
    print(f"CACHE IMPACT ANALYSIS")
    print(f"Model: {model_path}")
    if torch.cuda.is_available():
        print(f"Device: CUDA - {torch.cuda.get_device_name(0)}")
    else:
        print(f"Device: CPU")
    print(f"{'='*80}\n")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Use same loading strategy as test_mamba_model.py for optimal CPU performance  
    if device == "cpu":
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map={"": "cpu"}  # Force CPU explicitly
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    
    model.eval()
    
    prompt = "The quick brown fox jumps over the lazy dog. " * (prompt_length // 10)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    
    results = {}
    
    for cache_enabled in [True, False]:
        cache_str = "WITH" if cache_enabled else "WITHOUT"
        print(f"\n{'-'*80}")
        print(f"Testing {cache_str} Cache")
        print(f"{'-'*80}")
        
        # Warmup
        with torch.no_grad():
            for _ in range(2):
                _, _ = sample_from_draft_model(
                    model, input_ids, new_tokens=num_tokens,
                    temperature=0.0, use_cache=cache_enabled
                )
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        
        # Benchmark
        times = []
        for _ in range(10):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            
            with torch.no_grad():
                output, logits = sample_from_draft_model(
                    model, input_ids, new_tokens=num_tokens,
                    temperature=0.0, use_cache=cache_enabled
                )
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        
        avg_time = np.mean(times)
        throughput = num_tokens / avg_time
        
        # Memory measurement
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            mem_allocated = torch.cuda.memory_allocated() / (1024**2)  # MB
            mem_peak = torch.cuda.max_memory_allocated() / (1024**2)  # MB
        else:
            mem_allocated = 0
            mem_peak = 0
        
        results[cache_str] = {
            'time_ms': avg_time * 1000,
            'throughput': throughput,
            'mem_allocated_mb': mem_allocated,
            'mem_peak_mb': mem_peak,
        }
        
        print(f"Time: {avg_time*1000:.2f}ms")
        print(f"Throughput: {throughput:.2f} tok/s")
        if torch.cuda.is_available():
            print(f"Memory allocated: {mem_allocated:.2f} MB")
            print(f"Peak memory: {mem_peak:.2f} MB")
        else:
            print(f"Memory tracking: Not available on CPU")
    
    # Comparison
    print(f"\n{'='*80}")
    print(f"COMPARISON")
    print(f"{'='*80}")
    speedup = results['WITHOUT']['time_ms'] / results['WITH']['time_ms']
    
    print(f"Cache speedup: {speedup:.2f}x")
    print(f"\nWITH cache:    {results['WITH']['throughput']:.2f} tok/s")
    print(f"WITHOUT cache: {results['WITHOUT']['throughput']:.2f} tok/s")
    
    if torch.cuda.is_available():
        mem_overhead = results['WITH']['mem_peak_mb'] - results['WITHOUT']['mem_peak_mb']
        print(f"\nCache memory overhead: {mem_overhead:.2f} MB")
        print(f"WITH cache:    {results['WITH']['mem_peak_mb']:.2f} MB peak")
        print(f"WITHOUT cache: {results['WITHOUT']['mem_peak_mb']:.2f} MB peak")
    
    del model
    torch.cuda.empty_cache()


def compare_models(models, num_tokens=5, prompt_length=32, use_cache=True):
    """Compare multiple draft models"""
    results = []
    
    for model_name, model_path in models.items():
        print(f"\n\n{'#'*80}")
        print(f"# Testing: {model_name}")
        print(f"{'#'*80}")
        result = profile_draft_model(model_path, num_tokens, prompt_length, use_cache)
        result['model_name'] = model_name
        results.append(result)
    
    # Summary comparison
    print("\n\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Model':<30} {'Throughput':>12} {'Time/Token':>12} {'Cache Type':<15}")
    print("-"*80)
    
    for r in results:
        print(f"{r['model_name']:<30} {r['avg_throughput']:>10.2f} t/s {r['time_per_token_ms']:>10.2f}ms {r['cache_type'] or 'none':<15}")
    
    # Speedup analysis
    if len(results) >= 2:
        baseline = results[0]
        print("\n" + "="*80)
        print("RELATIVE PERFORMANCE (vs first model)")
        print("="*80)
        for r in results:
            speedup = r['avg_throughput'] / baseline['avg_throughput']
            print(f"{r['model_name']:<30} {speedup:>6.2f}x")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Profile sample_from_draft_model() performance")
    parser.add_argument(
        "--models",
        type=str,
        default="mamba,llama1b",
        help="Models to test (comma-separated): mamba, llama1b, llama3b, all"
    )
    parser.add_argument(
        "--tokens",
        type=int,
        default=5,
        help="Number of draft tokens to generate (lookahead K)"
    )
    parser.add_argument(
        "--prompt-length",
        type=int,
        default=512,
        help="Length of input prompt"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching"
    )
    parser.add_argument(
        "--separate-timing",
        action="store_true",
        help="Measure prefill and generation speed separately"
    )
    parser.add_argument(
        "--cache-comparison",
        action="store_true",
        help="Compare performance and memory with and without cache"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU execution (disable CUDA)"
    )
    
    args = parser.parse_args()
    
    # Force CPU if requested
    global device
    if args.cpu:
        device = "cpu"
        print("Forcing CPU execution (CUDA disabled)")
        
        # Optimize CPU threading
        cpu_count = os.cpu_count() or 4
        # Set environment variables for optimal CPU performance
        if 'OMP_NUM_THREADS' not in os.environ:
            os.environ['OMP_NUM_THREADS'] = str(min(cpu_count, 64))
        if 'MKL_NUM_THREADS' not in os.environ:
            os.environ['MKL_NUM_THREADS'] = str(min(cpu_count, 64))
        
        # Set PyTorch threads
        torch.set_num_threads(min(cpu_count, 64))
        torch.set_num_interop_threads(1)
        
        print(f"CPU threads configured: {torch.get_num_threads()}")
        print(f"OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'not set')}")
        print()
    
    # Available models
    all_models = {
        "mamba": "/HSC/users/qiaoye/SSM_SPEC/checkpoints/custom-mamba-65m-multi-gpu",
        "llama1b": "/HSC/users/qiaoye/checkpoints/Llama-3.2-1B",
        "llama3b": "/HSC/users/qiaoye/checkpoints/Llama-3.2-3B",
    }
    
    # Select models to test
    if args.models == "all":
        models_to_test = all_models
    else:
        model_keys = [k.strip() for k in args.models.split(',')]
        models_to_test = {k: all_models[k] for k in model_keys if k in all_models}
    
    if not models_to_test:
        print(f"No valid models specified. Available: {list(all_models.keys())}")
        return
    
    use_cache = not args.no_cache
    
    # If --cache-comparison, run cache impact analysis
    if args.cache_comparison:
        for model_name, model_path in models_to_test.items():
            print(f"\n\n{'#'*80}")
            print(f"# {model_name}")
            print(f"{'#'*80}")
            compare_cache_impact(model_path, num_tokens=args.tokens, prompt_length=args.prompt_length)
    # If --separate-timing, run detailed prefill/generation profiling
    elif args.separate_timing:
        for model_name, model_path in models_to_test.items():
            print(f"\n\n{'#'*80}")
            print(f"# {model_name}")
            print(f"{'#'*80}")
            profile_prefill_vs_generation(model_path, prompt_length=args.prompt_length)
    else:
        compare_models(
            models_to_test,
            num_tokens=args.tokens,
            prompt_length=args.prompt_length,
            use_cache=use_cache
        )


if __name__ == "__main__":
    main()
