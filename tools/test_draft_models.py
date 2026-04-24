import time
import warnings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from autoregressive_sampling import autoregressive_sampling
from speculative_sampling import speculative_sampling

# Suppress NVML warnings
warnings.filterwarnings("ignore", message=".*NVML.*")
warnings.filterwarnings("ignore", message=".*Can't initialize NVML.*")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load target model
print("Loading target model...")
target_model = AutoModelForCausalLM.from_pretrained("/HSC/users/qiaoye/checkpoints/Llama3.1-8B-hf").to(device)
tokenizer = AutoTokenizer.from_pretrained("/HSC/users/qiaoye/checkpoints/Llama3.1-8B-hf")

# Test different draft models
draft_models_to_test = [
    ("/HSC/users/qiaoye/SSM_SPEC/checkpoints/custom-mamba-65m-multi-gpu", "Current Mamba-65M"),
    ("gpt2", "GPT-2 (124M)"),
    ("distilgpt2", "DistilGPT-2 (82M)"),
]

prompt = "Emily found a mysterious letter on her doorstep one sunny morning."
inputs = tokenizer(prompt, return_tensors="pt").to(device)
MAX_NEW_TOKENS = 16  # Shorter for faster testing
TEMPERATURE = 0.0

print(f"\nBenchmarking different draft models")
print(f"Prompt: {prompt}")
print(f"Max new tokens: {MAX_NEW_TOKENS}")
print("=" * 80)

# Baseline: Autoregressive
print(f"\n🔄 BASELINE: Autoregressive Sampling")
start_time = time.time_ns()
auto_tokens = autoregressive_sampling(target_model, initial_prompt_seq=inputs.input_ids, 
                                    target_len=MAX_NEW_TOKENS+len(inputs.input_ids), temperature=TEMPERATURE)
end_time = time.time_ns()

auto_time = (end_time - start_time) / 1_000_000_000
auto_new_tokens = len(auto_tokens[0]) - len(inputs.input_ids)
auto_throughput = auto_new_tokens / auto_time

print(f"Time: {auto_time:.4f}s")
print(f"Throughput: {auto_throughput:.2f} tok/s")
baseline_text = tokenizer.decode(auto_tokens[0], skip_special_tokens=True)

print("-" * 80)

for draft_model_path, draft_model_name in draft_models_to_test:
    print(f"\n🚀 TESTING: {draft_model_name}")
    print(f"Model: {draft_model_path}")
    
    try:
        # Load draft model
        print("Loading draft model...")
        draft_model = AutoModelForCausalLM.from_pretrained(draft_model_path).to(device)
        
        # Single token generation test for draft model speed
        print("Testing single token generation speed...")
        test_input = inputs.input_ids
        
        # Measure draft model speed
        start_time = time.time_ns()
        with torch.no_grad():
            logits = draft_model(test_input).logits
            sample_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        end_time = time.time_ns()
        
        draft_single_token_time = (end_time - start_time) / 1_000_000_000
        print(f"Draft model single token time: {draft_single_token_time*1000:.2f}ms")
        
        # Measure target model speed for comparison
        start_time = time.time_ns()
        with torch.no_grad():
            logits = target_model(test_input).logits
            sample_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        end_time = time.time_ns()
        
        target_single_token_time = (end_time - start_time) / 1_000_000_000
        print(f"Target model single token time: {target_single_token_time*1000:.2f}ms")
        
        speed_ratio = target_single_token_time / draft_single_token_time
        print(f"Speed ratio (target/draft): {speed_ratio:.2f}x")
        
        if speed_ratio < 2:
            print(f"⚠️  WARNING: Draft model is not significantly faster than target model!")
            print(f"   For speculative sampling to work, draft should be 3-10x faster")
        
        # Test speculative sampling
        print("Running speculative sampling...")
        start_time = time.time_ns()
        spec_tokens, acceptance_rate = speculative_sampling(
            target_model, draft_model, 
            initial_prompt_seq=inputs.input_ids, 
            target_len=MAX_NEW_TOKENS+len(inputs.input_ids), 
            tokenizer=tokenizer, 
            temperature=TEMPERATURE, 
            debug=False
        )
        end_time = time.time_ns()
        
        spec_time = (end_time - start_time) / 1_000_000_000
        spec_new_tokens = len(spec_tokens[0]) - len(inputs.input_ids)
        spec_throughput = spec_new_tokens / spec_time
        
        speedup = auto_time / spec_time
        
        print(f"Time: {spec_time:.4f}s")
        print(f"Throughput: {spec_throughput:.2f} tok/s") 
        print(f"Acceptance rate: {acceptance_rate:.1%}")
        
        if speedup > 1:
            print(f"✅ Result: {speedup:.2f}x FASTER than autoregressive")
        else:
            print(f"❌ Result: {1/speedup:.2f}x SLOWER than autoregressive")
        
        # Verify output quality
        spec_text = tokenizer.decode(spec_tokens[0], skip_special_tokens=True)
        if TEMPERATURE == 0.0:  # Deterministic, should match
            if spec_text == baseline_text:
                print("✅ Output matches autoregressive (deterministic)")
            else:
                print("⚠️  Output differs from autoregressive")
        
        # Clean up
        del draft_model
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"❌ Failed to test {draft_model_name}: {str(e)}")
    
    print("-" * 80)

print(f"\n💡 KEY INSIGHTS:")
print(f"1. Draft model should be 3-10x faster than target model")
print(f"2. If draft model is slow, speculative sampling will be slower")
print(f"3. Mamba models need fast kernels (mamba-ssm, causal-conv1d) to be efficient")
print(f"4. Consider using smaller transformer models as draft models")

print(f"\n🔧 TO FIX MAMBA PERFORMANCE:")
print(f"pip install causal-conv1d>=1.0.0")
print(f"pip install mamba-ssm")