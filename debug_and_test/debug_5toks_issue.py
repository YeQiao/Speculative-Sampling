#!/usr/bin/env python3
"""
Demonstrate why the original measurement showed 5 tok/s instead of 30+ tok/s.
The issue: measuring TOTAL time / TOTAL tokens vs GENERATION time / GENERATION tokens
"""

import os
import sys

cpu_count = os.cpu_count() or 4
os.environ['OMP_NUM_THREADS'] = str(min(cpu_count, 64))

import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cpu"
torch.set_num_threads(min(cpu_count, 64))

model_path = "/HSC/users/qiaoye/SSM_SPEC/checkpoints/custom-mamba-65m-multi-gpu"

print("="*80)
print("WHY DID WE GET 5 TOK/S BEFORE?")
print("="*80)
print()

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map={"": "cpu"}
)
model.eval()

# Recreate the original test scenario
prompt_length = 512  # Long prompt (like in original profile_draft.py test)
new_tokens = 5       # Short generation (like in original test)

prompt = "The quick brown fox jumps over the lazy dog. " * (prompt_length // 10)
inputs = tokenizer(prompt, return_tensors="pt").to(device)
input_ids = inputs.input_ids[:, :prompt_length]

print(f"Test scenario (original problematic case):")
print(f"  Prompt length: {input_ids.shape[1]} tokens")
print(f"  New tokens to generate: {new_tokens}")
print()

# Warmup
with torch.no_grad():
    _ = model.generate(input_ids, max_new_tokens=3, do_sample=False)

# Measure the way we did before (WRONG)
print("="*80)
print("METHOD 1: Original measurement (WRONG)")
print("="*80)
print()

times = []
for i in range(5):
    start = time.perf_counter()
    
    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=new_tokens, do_sample=False)
    
    elapsed = time.perf_counter() - start
    times.append(elapsed)
    
    # WRONG: Dividing by new_tokens only
    throughput_wrong = new_tokens / elapsed
    print(f"Run {i+1}: {elapsed*1000:.2f}ms, {throughput_wrong:.2f} tok/s (WRONG calculation)")

avg_time = sum(times) / len(times)
avg_throughput_wrong = new_tokens / avg_time

print(f"\n❌ Average (WRONG): {avg_throughput_wrong:.2f} tok/s")
print(f"   Total time: {avg_time*1000:.2f}ms")
print(f"   New tokens: {new_tokens}")
print(f"   Calculation: {new_tokens} / {avg_time:.2f}s = {avg_throughput_wrong:.2f} tok/s")
print()
print("⚠️  This is MISLEADING because it includes prefill time!")
print(f"   Prefill of {prompt_length} tokens takes ~{prompt_length/30:.1f}s (~30 tok/s)")
print(f"   Generation of {new_tokens} tokens takes ~{new_tokens*0.01:.2f}s (~100 tok/s)")
print(f"   Total: ~{prompt_length/30 + new_tokens*0.01:.2f}s for just {new_tokens} tokens")

# Now measure correctly
print()
print("="*80)
print("METHOD 2: Correct measurement (SEPARATE prefill and generation)")
print("="*80)
print()

# Measure prefill separately
prefill_times = []
for i in range(5):
    start = time.perf_counter()
    with torch.no_grad():
        out = model(input_ids, use_cache=True)
    elapsed = time.perf_counter() - start
    prefill_times.append(elapsed)

avg_prefill = sum(prefill_times) / len(prefill_times)
prefill_throughput = prompt_length / avg_prefill

print(f"Prefill phase:")
print(f"  Time: {avg_prefill*1000:.2f}ms")
print(f"  Tokens processed: {prompt_length}")
print(f"  Throughput: {prefill_throughput:.2f} tok/s")

# Measure generation separately
with torch.no_grad():
    initial_out = model(input_ids, use_cache=True)
    cache_params = initial_out.cache_params if hasattr(initial_out, 'cache_params') else None

gen_times = []
for i in range(new_tokens):
    next_token = torch.randint(0, model.config.vocab_size, (1, 1), device=device)
    
    start = time.perf_counter()
    with torch.no_grad():
        if cache_params is not None:
            cache_position = torch.tensor([input_ids.shape[1] + i], device=device, dtype=torch.long)
            try:
                out = model(input_ids=next_token, cache_params=cache_params, 
                          cache_position=cache_position, use_cache=True)
            except:
                out = model(input_ids=next_token, cache_params=cache_params, use_cache=True)
            if hasattr(out, 'cache_params'):
                cache_params = out.cache_params
        else:
            out = model(input_ids=next_token)
    
    elapsed = time.perf_counter() - start
    gen_times.append(elapsed)

avg_gen = sum(gen_times) / len(gen_times)
total_gen_time = sum(gen_times)
gen_throughput = 1 / avg_gen

print(f"\nGeneration phase:")
print(f"  Time per token: {avg_gen*1000:.2f}ms")
print(f"  Total generation time: {total_gen_time*1000:.2f}ms")
print(f"  Tokens generated: {new_tokens}")
print(f"  Throughput: {gen_throughput:.2f} tok/s")

print()
print("="*80)
print("COMPARISON")
print("="*80)
print()

total_time = avg_prefill + total_gen_time
total_tokens = new_tokens  # Only counting NEW tokens generated

# This is what we measured before
apparent_throughput = total_tokens / total_time

print(f"Total pipeline:")
print(f"  Prefill time: {avg_prefill*1000:.2f}ms ({prompt_length} tokens)")
print(f"  Generation time: {total_gen_time*1000:.2f}ms ({new_tokens} tokens)")
print(f"  Total time: {total_time*1000:.2f}ms")
print()
print(f"If we measure: {new_tokens} tokens / {total_time:.3f}s = {apparent_throughput:.2f} tok/s")
print(f"  ❌ This is WRONG! It's comparing generated tokens to total time")
print()
print(f"Correct measurements:")
print(f"  ✅ Prefill throughput: {prefill_throughput:.2f} tok/s")
print(f"  ✅ Generation throughput: {gen_throughput:.2f} tok/s")
print()
print(f"For user-facing metrics:")
print(f"  Time to first token (TTFT): {avg_prefill*1000:.2f}ms")
print(f"  Time per output token: {avg_gen*1000:.2f}ms")

print()
print("="*80)
print("WHY THIS MATTERS")
print("="*80)
print(f"""
The original test was:
- Prompt: {prompt_length} tokens ({avg_prefill*1000:.0f}ms to process)
- Generate: {new_tokens} tokens ({total_gen_time*1000:.0f}ms to generate)
- Total: {total_time*1000:.0f}ms

Wrong calculation:
  {new_tokens} new tokens / {total_time:.2f}s = {apparent_throughput:.2f} tok/s ❌
  
This makes it LOOK slow because we're dividing a tiny number of output tokens
by a large total time (mostly spent on prefill).

Correct understanding:
  - Prefill processes at {prefill_throughput:.1f} tok/s (parallel)
  - Generation runs at {gen_throughput:.1f} tok/s (sequential with cache)
  
The "apparent" {apparent_throughput:.1f} tok/s is just an artifact of:
  - Long prompt ({prompt_length} tokens)
  - Short generation ({new_tokens} tokens)
  - Measuring wrong metric (new tokens / total time)

Solution: Generate MORE tokens to amortize prefill cost!
With 100 tokens: {100 / (avg_prefill + 100*avg_gen):.1f} tok/s ✅
""")
