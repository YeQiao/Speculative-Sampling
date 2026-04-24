#!/usr/bin/env python3
"""
Debug CPU performance for Mamba model.
Compare our sample_from_draft_model() with HuggingFace's native generate().
"""

import torch
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from transformers import AutoTokenizer, AutoModelForCausalLM
from core.utils import sample_from_draft_model

# Force CPU
device = "cpu"

print("="*80)
print("CPU PERFORMANCE DEBUG")
print("="*80)
print(f"\nPyTorch version: {torch.__version__}")
print(f"Number of CPU threads: {torch.get_num_threads()}")
print(f"Number of interop threads: {torch.get_num_interop_threads()}")
print(f"MKL available: {torch.backends.mkldnn.is_available()}")
print(f"OpenMP available: {'OMP_NUM_THREADS' in __import__('os').environ}")

# Try to optimize threading
import os
cpu_count = os.cpu_count() or 4
print(f"\nCPU count: {cpu_count}")

# Set threading for optimal performance
print("\nSetting optimal thread configuration...")
torch.set_num_threads(cpu_count)
torch.set_num_interop_threads(1)
print(f"PyTorch threads set to: {torch.get_num_threads()}")

print("\n" + "="*80)

# Load model
model_path = "/HSC/users/qiaoye/SSM_SPEC/checkpoints/custom-mamba-65m-multi-gpu"
print(f"\nLoading model from: {model_path}")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
model.eval()

prompt = "The quick brown fox jumps over the lazy dog. " * 5
inputs = tokenizer(prompt, return_tensors="pt").to(device)
input_ids = inputs.input_ids

print(f"Prompt length: {input_ids.shape[1]} tokens")
print(f"Generating 10 new tokens")

print("\n" + "="*80)
print("TEST 1: HuggingFace native generate()")
print("="*80)

# Warmup
with torch.no_grad():
    _ = model.generate(input_ids, max_new_tokens=5, do_sample=False)

# Benchmark
times = []
for i in range(10):
    start = time.perf_counter()
    
    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=10, do_sample=False)
    
    elapsed = time.perf_counter() - start
    times.append(elapsed)
    throughput = 10 / elapsed
    print(f"Run {i+1:2d}: {elapsed*1000:.2f}ms, {throughput:.2f} tok/s")

import numpy as np
avg_time = np.mean(times)
avg_throughput = 10 / avg_time
print(f"\nAverage: {avg_time*1000:.2f}ms")
print(f"Average throughput: {avg_throughput:.2f} tok/s")

print("\n" + "="*80)
print("TEST 2: Our sample_from_draft_model() with cache")
print("="*80)

# Warmup
with torch.no_grad():
    _, _ = sample_from_draft_model(model, input_ids, new_tokens=5, temperature=0.0, use_cache=True)

# Benchmark
times = []
for i in range(10):
    start = time.perf_counter()
    
    with torch.no_grad():
        output, logits = sample_from_draft_model(
            model, input_ids, new_tokens=10, temperature=0.0, use_cache=True
        )
    
    elapsed = time.perf_counter() - start
    times.append(elapsed)
    throughput = 10 / elapsed
    print(f"Run {i+1:2d}: {elapsed*1000:.2f}ms, {throughput:.2f} tok/s")

avg_time = np.mean(times)
avg_throughput = 10 / avg_time
print(f"\nAverage: {avg_time*1000:.2f}ms")
print(f"Average throughput: {avg_throughput:.2f} tok/s")

print("\n" + "="*80)
print("TEST 3: Our sample_from_draft_model() WITHOUT cache")
print("="*80)

# Warmup
with torch.no_grad():
    _, _ = sample_from_draft_model(model, input_ids, new_tokens=5, temperature=0.0, use_cache=False)

# Benchmark
times = []
for i in range(10):
    start = time.perf_counter()
    
    with torch.no_grad():
        output, logits = sample_from_draft_model(
            model, input_ids, new_tokens=10, temperature=0.0, use_cache=False
        )
    
    elapsed = time.perf_counter() - start
    times.append(elapsed)
    throughput = 10 / elapsed
    print(f"Run {i+1:2d}: {elapsed*1000:.2f}ms, {throughput:.2f} tok/s")

avg_time = np.mean(times)
avg_throughput = 10 / avg_time
print(f"\nAverage: {avg_time*1000:.2f}ms")
print(f"Average throughput: {avg_throughput:.2f} tok/s")

print("\n" + "="*80)
print("TEST 4: Detailed timing breakdown with stats")
print("="*80)

with torch.no_grad():
    _, _, stats = sample_from_draft_model(
        model, input_ids, new_tokens=10, temperature=0.0, 
        use_cache=True, return_stats=True, profile=True
    )

print(f"\nUsed incremental: {stats['used_incremental']}")
print(f"Cache type: {stats['cache_type']}")
print(f"Incremental tokens: {stats['incremental_tokens']}")
print(f"Full recompute tokens: {stats['full_recompute_tokens']}")
print(f"\nPrefill time: {stats['initial_forward_time_ms']:.2f}ms")
if stats['incremental_tokens'] > 0:
    print(f"Incremental time: {stats['incremental_time_ms']:.2f}ms (avg {stats['incremental_time_ms']/stats['incremental_tokens']:.2f}ms/tok)")
if stats['full_recompute_tokens'] > 0:
    print(f"Full recompute time: {stats['full_recompute_time_ms']:.2f}ms (avg {stats['full_recompute_time_ms']/stats['full_recompute_tokens']:.2f}ms/tok)")

total = stats['initial_forward_time_ms'] + stats['incremental_time_ms'] + stats['full_recompute_time_ms']
print(f"\nTotal time: {total:.2f}ms")
print(f"Overall throughput: {10 / (total/1000):.2f} tok/s")

print("\n" + "="*80)
print("ANALYSIS")
print("="*80)
print("\nIf HuggingFace generate() is much faster, the issue is in our implementation.")
print("If both are slow, the issue is PyTorch CPU threading configuration.")
print("Expected: ~30 tok/s for Mamba-65M on CPU")
