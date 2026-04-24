import time
import random
import warnings
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from autoregressive_sampling import autoregressive_sampling
from speculative_sampling import speculative_sampling

# Suppress NVML warnings
warnings.filterwarnings("ignore", message=".*NVML.*")
warnings.filterwarnings("ignore", message=".*Can't initialize NVML.*")

def profile_speculative_sampling(target_model, draft_model, initial_prompt_seq, target_len, tokenizer, lookahead=4, temperature=1.0, debug=False):
    '''
    Profiled version of speculative sampling to identify bottlenecks
    '''
    assert initial_prompt_seq.shape[0] == 1, 'Batch size should be 1'

    n = initial_prompt_seq.shape[-1]
    fin_prompt_seq = initial_prompt_seq.detach().clone()
    
    # Track acceptance statistics
    total_draft_tokens = 0
    accepted_tokens = 0
    
    # Profiling variables
    draft_time = 0
    target_time = 0
    sampling_time = 0
    overhead_time = 0
    num_iterations = 0

    while n < target_len:
        iter_start = time.time_ns()
        num_iterations += 1
        
        n_orig = n
        N = fin_prompt_seq.shape[-1]
        
        # Time draft model sampling
        draft_start = time.time_ns()
        draft_outputs, draft_logits = sample_from_draft_model(draft_model, fin_prompt_seq, new_tokens=lookahead, temperature=temperature)
        draft_end = time.time_ns()
        draft_time += (draft_end - draft_start)
        
        # Time target model forward pass
        target_start = time.time_ns()
        target_logits = target_model(draft_outputs).logits[:, -lookahead-1:, :]
        target_end = time.time_ns()
        target_time += (target_end - target_start)
        
        # Time distribution calculation and sampling
        sampling_start = time.time_ns()
        target_model_distribution = get_distribution(target_logits, temperature)
        draft_model_distribution = get_distribution(draft_logits, temperature)

        accepted_flag = 1
        
        for t in range(lookahead):
            total_draft_tokens += 1
            numerator = target_model_distribution[:, t, draft_outputs[0, N+t]]
            denominator = draft_model_distribution[:, t, draft_outputs[0, N+t]]
            ratio = (numerator / denominator)
            uniform_distribution = torch.rand_like(numerator)
            ones_tensor = torch.ones_like(numerator)

            # Rejection Sampling
            ## Acceptance
            if (uniform_distribution < torch.min(ones_tensor, ratio)).any():
                fin_prompt_seq = torch.concat([fin_prompt_seq, draft_outputs[:, N+t].unsqueeze(dim=-1)], dim=-1)
                accepted_tokens += 1
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
        
        sampling_end = time.time_ns()
        sampling_time += (sampling_end - sampling_start)
        
        iter_end = time.time_ns()
        overhead_time += (iter_end - iter_start) - (draft_end - draft_start) - (target_end - target_start) - (sampling_end - sampling_start)
        
        n += 1

    # Calculate acceptance rate
    acceptance_rate = accepted_tokens / total_draft_tokens if total_draft_tokens > 0 else 0.0
    
    # Convert times to seconds
    draft_time_s = draft_time / 1_000_000_000
    target_time_s = target_time / 1_000_000_000
    sampling_time_s = sampling_time / 1_000_000_000
    overhead_time_s = overhead_time / 1_000_000_000
    
    profile_info = {
        'draft_time': draft_time_s,
        'target_time': target_time_s,
        'sampling_time': sampling_time_s,
        'overhead_time': overhead_time_s,
        'num_iterations': num_iterations,
        'acceptance_rate': acceptance_rate
    }
    
    return fin_prompt_seq, acceptance_rate, profile_info

# Import required functions
from utils import sample_from_draft_model, get_distribution, sample

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

print("Loading models...")
target_model = AutoModelForCausalLM.from_pretrained("/HSC/users/qiaoye/checkpoints/Llama3.1-8B-hf").to(device)
draft_model = AutoModelForCausalLM.from_pretrained("/HSC/users/qiaoye/SSM_SPEC/checkpoints/custom-mamba-65m-multi-gpu").to(device)
tokenizer = AutoTokenizer.from_pretrained("/HSC/users/qiaoye/checkpoints/Llama3.1-8B-hf")

# Test prompts
test_prompts = [
    "Emily found a mysterious letter on her doorstep one sunny morning.",
    "The old lighthouse had been abandoned for years, but its beam suddenly flickered to life.",
    "What did Rutherford discover?"
]

MAX_NEW_TOKENS = 32  # Reduced for faster testing
TEMPERATURE = 0.0

print(f"\nProfiling Analysis (MAX_NEW_TOKENS={MAX_NEW_TOKENS}, TEMPERATURE={TEMPERATURE})")
print("=" * 80)

for i, prompt in enumerate(test_prompts):
    print(f"\nTest {i+1}: {prompt[:50]}...")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Autoregressive baseline
    print("\n--- Autoregressive Sampling ---")
    start_time = time.time_ns()
    auto_tokens = autoregressive_sampling(target_model, initial_prompt_seq=inputs.input_ids, 
                                        target_len=MAX_NEW_TOKENS+len(inputs.input_ids), temperature=TEMPERATURE)
    end_time = time.time_ns()
    
    auto_time = (end_time - start_time) / 1_000_000_000
    auto_new_tokens = len(auto_tokens[0]) - len(inputs.input_ids)
    auto_throughput = auto_new_tokens / auto_time
    
    print(f"Time: {auto_time:.4f}s")
    print(f"Tokens generated: {auto_new_tokens}")
    print(f"Throughput: {auto_throughput:.2f} tok/s")
    
    # Speculative sampling with profiling
    print("\n--- Speculative Sampling (Profiled) ---")
    start_time = time.time_ns()
    spec_tokens, acceptance_rate, profile_info = profile_speculative_sampling(
        target_model, draft_model, initial_prompt_seq=inputs.input_ids, 
        target_len=MAX_NEW_TOKENS+len(inputs.input_ids), tokenizer=tokenizer, 
        temperature=TEMPERATURE, debug=False
    )
    end_time = time.time_ns()
    
    spec_time = (end_time - start_time) / 1_000_000_000
    spec_new_tokens = len(spec_tokens[0]) - len(inputs.input_ids)
    spec_throughput = spec_new_tokens / spec_time
    
    print(f"Total time: {spec_time:.4f}s")
    print(f"Tokens generated: {spec_new_tokens}")
    print(f"Throughput: {spec_throughput:.2f} tok/s")
    print(f"Acceptance rate: {acceptance_rate:.2%}")
    print(f"Iterations: {profile_info['num_iterations']}")
    
    print(f"\nTime breakdown:")
    print(f"  Draft model time:    {profile_info['draft_time']:.4f}s ({profile_info['draft_time']/spec_time*100:.1f}%)")
    print(f"  Target model time:   {profile_info['target_time']:.4f}s ({profile_info['target_time']/spec_time*100:.1f}%)")
    print(f"  Sampling/logic time: {profile_info['sampling_time']:.4f}s ({profile_info['sampling_time']/spec_time*100:.1f}%)")
    print(f"  Overhead time:       {profile_info['overhead_time']:.4f}s ({profile_info['overhead_time']/spec_time*100:.1f}%)")
    
    print(f"\nPerformance comparison:")
    speedup = auto_time / spec_time
    if speedup > 1:
        print(f"  Speculative is {speedup:.2f}x FASTER than autoregressive")
    else:
        print(f"  Speculative is {1/speedup:.2f}x SLOWER than autoregressive")
    
    print(f"  Throughput ratio: {spec_throughput/auto_throughput:.2f}x")
    
    # Analysis
    print(f"\nAnalysis:")
    total_model_time = profile_info['draft_time'] + profile_info['target_time']
    print(f"  Total model computation: {total_model_time:.4f}s ({total_model_time/spec_time*100:.1f}%)")
    print(f"  Draft model calls per iteration: {profile_info['draft_time']/(profile_info['num_iterations']*spec_time)*spec_time:.4f}s avg")
    print(f"  Target model calls per iteration: {profile_info['target_time']/(profile_info['num_iterations']*spec_time)*spec_time:.4f}s avg")
    
    print("-" * 80)

print(f"\nRecommendations for improvement:")
print("1. If target model time dominates: Increase lookahead (more draft tokens per target call)")
print("2. If draft model time is high: Use a smaller/faster draft model")
print("3. If acceptance rate is low: Use a better draft model or adjust temperature")
print("4. If overhead is high: Optimize the sampling logic and tensor operations")