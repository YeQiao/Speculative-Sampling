#!/usr/bin/env python3
"""
Profile Mamba model inference to identify bottlenecks.
Focus on selective scan operations and overall throughput.
"""

import torch
import time
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.profiler import profile, record_function, ProfilerActivity
import sys

device = "cuda" if torch.cuda.is_available() else "cpu"

def profile_mamba_inference(model_path, num_tokens=100, batch_size=1, prompt_length=32, use_cache=True):
    """
    Profile Mamba model inference with torch profiler.
    """
    print(f"{'='*80}")
    print(f"Profiling Mamba Model: {model_path}")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Prompt length: {prompt_length}, Generate tokens: {num_tokens}, Batch size: {batch_size}")
    print(f"Cache: {'Enabled' if use_cache else 'Disabled'}")
    print(f"{'='*80}\n")
    
    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    model.eval()
    
    # Prepare input
    prompt = "The quick brown fox jumps over the lazy dog. " * (prompt_length // 10)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    
    # Warmup
    print("Warming up...")
    with torch.no_grad():
        for _ in range(5):
            _ = model.generate(input_ids, max_new_tokens=10, do_sample=False)
    torch.cuda.synchronize()
    
    print("\n" + "="*80)
    print("PROFILING WITH TORCH PROFILER")
    print("="*80 + "\n")
    
    # Profile with torch profiler - minimal settings only
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    ) as prof:
        with torch.no_grad():
            with record_function("mamba_generation"):
                output = model.generate(
                    input_ids,
                    max_new_tokens=num_tokens,
                    do_sample=False,
                    use_cache=use_cache,
                )
    
    # Print profiler results
    print("\n" + "="*80)
    print("TOP 20 OPERATIONS BY CUDA TIME")
    print("="*80)
    print(prof.key_averages().table(
        sort_by="cuda_time_total", 
        row_limit=20,
        max_name_column_width=80
    ))
    
    print("\n" + "="*80)
    print("TOP 20 OPERATIONS BY CPU TIME")
    print("="*80)
    print(prof.key_averages().table(
        sort_by="cpu_time_total", 
        row_limit=20,
        max_name_column_width=80
    ))
    
    print("\n" + "="*80)
    print("OPERATIONS CONTAINING 'SCAN' OR 'SELECTIVE'")
    print("="*80)
    events = prof.key_averages()
    for evt in events:
        evt_name = evt.key.lower()
        if any(keyword in evt_name for keyword in ['scan', 'selective', 'ssm']):
            print(f"\n{evt.key}")
            print(f"  CPU time: {evt.cpu_time_total/1000:.2f} ms")
            # cuda_time_total is available in CUDA profiling records
            if hasattr(evt, 'self_cuda_time_total'):
                print(f"  CUDA time: {evt.self_cuda_time_total/1000:.2f} ms")
            print(f"  Calls: {evt.count}")
    
    # Measure raw throughput
    print("="*80)
    print("THROUGHPUT MEASUREMENT")
    print("="*80)
    
    num_runs = 10
    times = []
    
    for i in range(num_runs):
        torch.cuda.synchronize()
        start_time = time.time()
        
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=num_tokens,
                do_sample=False,
                use_cache=use_cache,
            )
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        elapsed = end_time - start_time
        times.append(elapsed)
        
        tokens_generated = output.shape[1] - input_ids.shape[1]
        throughput = tokens_generated / elapsed
        
        print(f"Run {i+1}: {elapsed:.3f}s, {throughput:.2f} tok/s")
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    avg_throughput = num_tokens / avg_time
    
    print(f"\nAverage: {avg_time:.3f}s ± {std_time:.3f}s")
    print(f"Average throughput: {avg_throughput:.2f} tok/s")
    
    # Memory usage
    if torch.cuda.is_available():
        print("\n" + "="*80)
        print("MEMORY USAGE")
        print("="*80)
        print(f"Allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        print(f"Reserved: {torch.cuda.memory_reserved()/1024**2:.2f} MB")
        print(f"Max allocated: {torch.cuda.max_memory_allocated()/1024**2:.2f} MB")


def compare_batch_sizes(model_path, prompt_length=32):
    """
    Compare throughput at different batch sizes to identify bottlenecks.
    """
    print("\n" + "="*80)
    print("BATCH SIZE SCALING ANALYSIS")
    print("="*80 + "\n")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    model.eval()
    
    batch_sizes = [1, 2, 4, 8]
    prompt = "The quick brown fox jumps over the lazy dog. " * (prompt_length // 10)
    num_tokens = 50
    
    for bs in batch_sizes:
        inputs = tokenizer([prompt] * bs, return_tensors="pt", padding=True).to(device)
        input_ids = inputs.input_ids
        
        # Warmup
        with torch.no_grad():
            _ = model.generate(input_ids, max_new_tokens=10, do_sample=False)
        
        # Measure
        torch.cuda.synchronize()
        start = time.time()
        
        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens=num_tokens, do_sample=False)
        
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        total_tokens = (output.shape[1] - input_ids.shape[1]) * bs
        throughput = total_tokens / elapsed
        throughput_per_sample = throughput / bs
        
        print(f"Batch size {bs:2d}: {throughput:6.2f} tok/s total, "
              f"{throughput_per_sample:6.2f} tok/s per sample, "
              f"Time: {elapsed:.3f}s")
    
    del model
    torch.cuda.empty_cache()


def profile_forward_pass(model_path, prompt_length=32):
    """
    Profile a single forward pass to see layer-by-layer breakdown.
    """
    print("\n" + "="*80)
    print("SINGLE FORWARD PASS PROFILING")
    print("="*80 + "\n")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    model.eval()
    
    prompt = "The quick brown fox jumps over the lazy dog. " * (prompt_length // 10)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    
    # Warmup
    with torch.no_grad():
        _ = model(input_ids)
    
    # Profile single forward pass
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
    ) as prof:
        with torch.no_grad():
            with record_function("forward_pass"):
                _ = model(input_ids)
    
    print("Top operations in forward pass:")
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=30,
        max_name_column_width=80
    ))
    
    del model
    torch.cuda.empty_cache()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Profile Mamba model inference")
    parser.add_argument(
        "--model",
        type=str,
        default="/HSC/users/qiaoye/SSM_SPEC/checkpoints/custom-mamba-65m-multi-gpu",
        help="Path to Mamba model"
    )
    parser.add_argument(
        "--tokens",
        type=int,
        default=100,
        help="Number of tokens to generate for profiling"
    )
    parser.add_argument(
        "--prompt-length",
        type=int,
        default=32,
        help="Length of input prompt"
    )
    parser.add_argument(
        "--skip-batch",
        action="store_true",
        help="Skip batch size analysis"
    )
    parser.add_argument(
        "--skip-forward",
        action="store_true",
        help="Skip forward pass profiling"
    )
    parser.add_argument(
        "--multi-length",
        action="store_true",
        help="Profile multiple prompt lengths (short, medium, long) and save results"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable Mamba hidden state caching (default: cache enabled)"
    )
    
    args = parser.parse_args()
    use_cache = not args.no_cache
    
    if args.multi_length:
        # Profile with representative prompt lengths
        import os
        os.makedirs("outputs/mamba_profile", exist_ok=True)
        
        prompt_lengths = [
            (32, "short"),
            (512, "medium"),
            (2048, "long")
        ]
        
        for prompt_len, label in prompt_lengths:
            print("\n" + "="*80)
            print(f"PROFILING WITH {label.upper()} PROMPT (length={prompt_len})")
            print("="*80)
            
            output_file = f"outputs/mamba_profile/profile_{label}_prompt{prompt_len}.txt"
            
            # Redirect output to file
            original_stdout = sys.stdout
            with open(output_file, 'w') as f:
                sys.stdout = f
                
                print("="*80)
                print(f"Mamba Profiling Results - {label.upper()} Prompt")
                print(f"Prompt Length: {prompt_len} tokens")
                print(f"Model: {args.model}")
                print("="*80 + "\n")
                
                profile_mamba_inference(
                    args.model,
                    num_tokens=args.tokens,
                    prompt_length=prompt_len,
                    use_cache=use_cache
                )
                
                if not args.skip_batch:
                    compare_batch_sizes(args.model, prompt_length=prompt_len)
                
                if not args.skip_forward:
                    profile_forward_pass(args.model, prompt_length=prompt_len)
            
            sys.stdout = original_stdout
            print(f"Results saved to: {output_file}")
    else:
        # Single profiling run
        profile_mamba_inference(
            args.model,
            num_tokens=args.tokens,
            prompt_length=args.prompt_length,
            use_cache=use_cache
        )
        
        # Batch size analysis
        if not args.skip_batch:
            compare_batch_sizes(args.model)
        
        # Forward pass profiling
        if not args.skip_forward:
            profile_forward_pass(args.model)
    
    print("\n" + "="*80)
    print("PROFILING COMPLETE")
    print("="*80)
    print("\nKey files generated:")
    print("  - outputs/mamba_profile_trace.json (Chrome trace)")
    print("\nTo visualize:")
    print("  1. Open chrome://tracing in Chrome browser")
    print("  2. Load the trace file")
    print("  3. Look for long-running operations")


if __name__ == "__main__":
    main()
