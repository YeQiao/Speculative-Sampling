#!/usr/bin/env python3
"""
Compare CPU performance with different dtypes (float32 vs float16)
to identify why original test was slow
"""

import os
import sys

# Set threading BEFORE importing torch
cpu_count = os.cpu_count() or 4
os.environ['OMP_NUM_THREADS'] = str(min(cpu_count, 64))
os.environ['MKL_NUM_THREADS'] = str(min(cpu_count, 64))

import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

# Force CPU
device = "cpu"

torch.set_num_threads(min(cpu_count, 64))
torch.set_num_interop_threads(1)

print("="*80)
print("CPU PERFORMANCE: FLOAT32 vs FLOAT16 COMPARISON")
print("="*80)
print(f"Device: {device}")
print(f"Threads: {torch.get_num_threads()}")
print()

model_path = "/HSC/users/qiaoye/SSM_SPEC/checkpoints/custom-mamba-65m-multi-gpu"
tokenizer = AutoTokenizer.from_pretrained(model_path)

prompt = "The quick brown fox jumps over the lazy dog. " * 5
inputs = tokenizer(prompt, return_tensors="pt").to(device)
input_ids = inputs.input_ids

print(f"Prompt length: {input_ids.shape[1]} tokens")
print(f"Generating: 20 tokens")
print()

# Test configurations
configs = [
    {"dtype": torch.float32, "name": "FLOAT32 (original slow test)"},
    {"dtype": torch.float16, "name": "FLOAT16 (from test_mamba_model.py)"},
]

results = {}

for config in configs:
    dtype = config["dtype"]
    name = config["name"]
    
    print("="*80)
    print(f"Testing: {name}")
    print("="*80)
    
    # Load model with specific dtype
    print(f"Loading model with dtype={dtype}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=dtype
    ).to(device)
    model.eval()
    
    # Warmup
    print("Warming up...")
    with torch.no_grad():
        _ = model.generate(input_ids, max_new_tokens=5, do_sample=False)
    
    # Benchmark
    print("\nBenchmarking generation (5 runs)...")
    times = []
    
    for i in range(5):
        start = time.perf_counter()
        
        with torch.no_grad():
            output = model.generate(
                input_ids, 
                max_new_tokens=20, 
                do_sample=False,
                use_cache=True
            )
        
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        throughput = 20 / elapsed
        print(f"  Run {i+1}: {elapsed*1000:.2f}ms, {throughput:.2f} tok/s")
    
    import numpy as np
    avg_time = np.mean(times)
    std_time = np.std(times)
    avg_throughput = 20 / avg_time
    
    print(f"\nResults for {name}:")
    print(f"  Average time: {avg_time*1000:.2f}ms ± {std_time*1000:.2f}ms")
    print(f"  Average throughput: {avg_throughput:.2f} tok/s")
    
    results[name] = {
        "avg_time_ms": avg_time * 1000,
        "avg_throughput": avg_throughput,
        "dtype": str(dtype)
    }
    
    # Cleanup
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print()

# Comparison
print("="*80)
print("COMPARISON SUMMARY")
print("="*80)

for name, result in results.items():
    print(f"\n{name}:")
    print(f"  Throughput: {result['avg_throughput']:.2f} tok/s")
    print(f"  Time: {result['avg_time_ms']:.2f}ms")
    print(f"  Dtype: {result['dtype']}")

if len(results) == 2:
    float32_tps = results["FLOAT32 (original slow test)"]["avg_throughput"]
    float16_tps = results["FLOAT16 (from test_mamba_model.py)"]["avg_throughput"]
    speedup = float16_tps / float32_tps
    
    print(f"\n{'='*80}")
    print(f"SPEEDUP: Float16 is {speedup:.2f}x faster than Float32")
    print(f"{'='*80}")
    
    if speedup > 2:
        print("\n✅ CONCLUSION: The slow performance was due to using float32!")
        print("   Solution: Use torch_dtype=torch.float16 for CPU inference")
    else:
        print("\n⚠️  Dtype difference is not the main issue")
        print("   Both are similarly slow - might be Mamba2 architecture limitation")
