#!/usr/bin/env python3
"""
Test Mamba CPU performance with optimal threading configuration.
This sets environment variables BEFORE importing torch.
"""

import os
import sys

# Set threading environment variables BEFORE importing torch
cpu_count = os.cpu_count() or 4
optimal_threads = min(cpu_count, 64)

os.environ['OMP_NUM_THREADS'] = str(optimal_threads)
os.environ['MKL_NUM_THREADS'] = str(optimal_threads)
os.environ['OPENBLAS_NUM_THREADS'] = str(optimal_threads)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(optimal_threads)
os.environ['NUMEXPR_NUM_THREADS'] = str(optimal_threads)

# Disable TensorFloat32 for consistency
os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = '0'

print(f"Environment configured with {optimal_threads} threads")
print(f"CPU count: {cpu_count}")
print()

# NOW import torch
import torch
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from transformers import AutoTokenizer, AutoModelForCausalLM

# Force CPU
device = "cpu"

# Set PyTorch threads after import
torch.set_num_threads(optimal_threads)
torch.set_num_interop_threads(1)

print(f"PyTorch version: {torch.__version__}")
print(f"PyTorch threads: {torch.get_num_threads()}")
print(f"PyTorch interop threads: {torch.get_num_interop_threads()}")
print(f"MKL available: {torch.backends.mkldnn.is_available()}")
print()

# Load model
model_path = "/HSC/users/qiaoye/SSM_SPEC/checkpoints/custom-mamba-65m-multi-gpu"
print(f"Loading model from: {model_path}")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32).to(device)
model.eval()

# Test with different prompt lengths
for prompt_len in [10, 50, 100, 200]:
    prompt = "The quick brown fox jumps over the lazy dog. " * (prompt_len // 10 + 1)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs.input_ids[:, :prompt_len]  # Trim to exact length
    
    print(f"\n{'='*80}")
    print(f"Testing with prompt length: {input_ids.shape[1]} tokens")
    print(f"{'='*80}")
    
    # Warmup
    with torch.no_grad():
        _ = model.generate(input_ids, max_new_tokens=5, do_sample=False)
    
    # Benchmark generation
    num_new_tokens = 20
    times = []
    
    for i in range(5):
        start = time.perf_counter()
        
        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens=num_new_tokens, do_sample=False)
        
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        throughput = num_new_tokens / elapsed
        print(f"Run {i+1}: {elapsed*1000:.2f}ms, {throughput:.2f} tok/s")
    
    import numpy as np
    avg_time = np.mean(times)
    avg_throughput = num_new_tokens / avg_time
    print(f"\nAverage throughput: {avg_throughput:.2f} tok/s")
    
    # Also test single forward pass
    print(f"\nSingle forward pass (prefill {input_ids.shape[1]} tokens):")
    start = time.perf_counter()
    with torch.no_grad():
        _ = model(input_ids)
    elapsed = time.perf_counter() - start
    print(f"Time: {elapsed*1000:.2f}ms")
    print(f"Throughput: {input_ids.shape[1]/elapsed:.2f} tok/s")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("If throughput is still ~5 tok/s, the issue is likely:")
print("1. Mamba2 architecture is CPU-slow (needs optimized kernels)")
print("2. Model checkpoint has CUDA-compiled operations")
print("3. Need to use original Mamba (not Mamba2) for CPU performance")
