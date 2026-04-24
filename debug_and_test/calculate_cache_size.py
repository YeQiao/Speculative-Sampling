#!/usr/bin/env python3
"""
Calculate theoretical cache sizes for Mamba and Transformer models.
"""

import torch
from transformers import AutoConfig, AutoModelForCausalLM
import sys


def calculate_mamba_cache_size(model_path, seq_length):
    """
    Calculate Mamba hidden state cache size.
    
    Mamba cache_params stores:
    - conv_states: (batch, d_inner, d_conv-1) per layer
    - ssm_states: (batch, d_inner, d_state) per layer
    """
    config = AutoConfig.from_pretrained(model_path)
    
    # Mamba-specific parameters
    d_model = config.hidden_size  # e.g., 768 for Mamba-65M
    n_layers = config.num_hidden_layers
    expand_factor = getattr(config, 'expand', 2)
    d_inner = int(expand_factor * d_model)
    d_conv = getattr(config, 'd_conv', 4)
    d_state = getattr(config, 'd_state', 16)
    
    batch_size = 1  # assume batch=1
    
    # Cache per layer
    conv_states_per_layer = batch_size * d_inner * (d_conv - 1)
    ssm_states_per_layer = batch_size * d_inner * d_state
    cache_per_layer = conv_states_per_layer + ssm_states_per_layer
    
    # Total cache
    total_cache_elements = cache_per_layer * n_layers
    
    # Convert to bytes (float32 = 4 bytes)
    bytes_per_element = 4
    total_bytes = total_cache_elements * bytes_per_element
    total_mb = total_bytes / (1024**2)
    
    print(f"{'='*80}")
    print(f"Mamba Cache Size Calculation")
    print(f"Model: {model_path}")
    print(f"{'='*80}")
    print(f"Architecture:")
    print(f"  d_model (hidden_size): {d_model}")
    print(f"  n_layers: {n_layers}")
    print(f"  expand_factor: {expand_factor}")
    print(f"  d_inner: {d_inner}")
    print(f"  d_conv: {d_conv}")
    print(f"  d_state: {d_state}")
    print(f"\nCache per layer:")
    print(f"  conv_states: {batch_size} × {d_inner} × {d_conv-1} = {conv_states_per_layer:,} elements")
    print(f"  ssm_states:  {batch_size} × {d_inner} × {d_state} = {ssm_states_per_layer:,} elements")
    print(f"  Total per layer: {cache_per_layer:,} elements")
    print(f"\nTotal cache:")
    print(f"  Elements: {total_cache_elements:,}")
    print(f"  Size (float32): {total_mb:.2f} MB")
    print(f"\nNote: Mamba cache is sequence-length independent!")
    print(f"  Cache size is the same for seq_len=1 or seq_len={seq_length}")
    print(f"{'='*80}\n")
    
    return total_mb


def calculate_transformer_kv_cache_size(model_path, seq_length):
    """
    Calculate Transformer KV cache size.
    
    KV cache stores:
    - key: (batch, num_heads, seq_len, head_dim) per layer
    - value: (batch, num_heads, seq_len, head_dim) per layer
    """
    config = AutoConfig.from_pretrained(model_path)
    
    # Transformer parameters
    d_model = config.hidden_size
    n_layers = config.num_hidden_layers
    n_heads = config.num_attention_heads
    head_dim = d_model // n_heads
    
    batch_size = 1  # assume batch=1
    
    # Cache per layer
    key_per_layer = batch_size * n_heads * seq_length * head_dim
    value_per_layer = batch_size * n_heads * seq_length * head_dim
    cache_per_layer = key_per_layer + value_per_layer
    
    # Total cache
    total_cache_elements = cache_per_layer * n_layers
    
    # Convert to bytes (float32 = 4 bytes)
    bytes_per_element = 4
    total_bytes = total_cache_elements * bytes_per_element
    total_mb = total_bytes / (1024**2)
    
    print(f"{'='*80}")
    print(f"Transformer KV Cache Size Calculation")
    print(f"Model: {model_path}")
    print(f"{'='*80}")
    print(f"Architecture:")
    print(f"  d_model (hidden_size): {d_model}")
    print(f"  n_layers: {n_layers}")
    print(f"  n_heads: {n_heads}")
    print(f"  head_dim: {head_dim}")
    print(f"\nCache per layer:")
    print(f"  key:   {batch_size} × {n_heads} × {seq_length} × {head_dim} = {key_per_layer:,} elements")
    print(f"  value: {batch_size} × {n_heads} × {seq_length} × {head_dim} = {value_per_layer:,} elements")
    print(f"  Total per layer: {cache_per_layer:,} elements")
    print(f"\nTotal cache:")
    print(f"  Elements: {total_cache_elements:,}")
    print(f"  Size (float32): {total_mb:.2f} MB")
    print(f"\nNote: KV cache grows linearly with sequence length!")
    print(f"  For seq_len={seq_length}: {total_mb:.2f} MB")
    print(f"  For seq_len={seq_length*2}: {total_mb*2:.2f} MB (2x)")
    print(f"  For seq_len={seq_length*10}: {total_mb*10:.2f} MB (10x)")
    print(f"{'='*80}\n")
    
    return total_mb


def compare_cache_growth(mamba_path, llama_path, seq_lengths):
    """Compare cache growth across different sequence lengths"""
    print(f"\n{'='*80}")
    print(f"CACHE SIZE COMPARISON: Mamba vs Transformer")
    print(f"{'='*80}\n")
    
    # Calculate once for Mamba (sequence-independent)
    mamba_size = calculate_mamba_cache_size(mamba_path, seq_lengths[0])
    
    print(f"\n{'='*80}")
    print(f"Cache Size vs Sequence Length")
    print(f"{'='*80}")
    print(f"{'Seq Length':>12} | {'Mamba (MB)':>12} | {'Llama-1B (MB)':>15} | {'Ratio (Llama/Mamba)':>20}")
    print(f"{'-'*80}")
    
    for seq_len in seq_lengths:
        llama_size = calculate_transformer_kv_cache_size(llama_path, seq_len)
        ratio = llama_size / mamba_size
        print(f"{seq_len:>12,} | {mamba_size:>12.2f} | {llama_size:>15.2f} | {ratio:>20.2f}x")


def verify_with_actual_model(model_path, model_type="mamba"):
    """Load model and verify cache size by inspecting actual cache"""
    print(f"\n{'='*80}")
    print(f"VERIFICATION: Actual Cache Size from Model")
    print(f"Model: {model_path}")
    print(f"{'='*80}\n")
    
    model = AutoModelForCausalLM.from_pretrained(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    # Create dummy input
    input_ids = torch.randint(0, 1000, (1, 10), device=device)
    
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
    
    if model_type == "mamba":
        cache = getattr(outputs, 'cache_params', None)
        if cache is None:
            print("No cache_params found in outputs!")
            return
        
        total_size = 0
        print("Cache structure:")
        for i, (conv_state, ssm_state) in enumerate(zip(cache.conv_states, cache.ssm_states)):
            conv_size = conv_state.numel() * conv_state.element_size() / (1024**2)
            ssm_size = ssm_state.numel() * ssm_state.element_size() / (1024**2)
            layer_size = conv_size + ssm_size
            total_size += layer_size
            if i < 3 or i >= len(cache.conv_states) - 2:  # Show first 3 and last 2 layers
                print(f"  Layer {i}: conv_state {tuple(conv_state.shape)} ({conv_size:.2f} MB) + "
                      f"ssm_state {tuple(ssm_state.shape)} ({ssm_size:.2f} MB) = {layer_size:.2f} MB")
            elif i == 3:
                print(f"  ...")
        
        print(f"\nTotal cache size: {total_size:.2f} MB")
        print(f"Number of layers: {len(cache.conv_states)}")
        
    else:  # transformer
        cache = getattr(outputs, 'past_key_values', None)
        if cache is None:
            print("No past_key_values found in outputs!")
            return
        
        total_size = 0
        seq_len = cache[0][0].shape[2]  # (batch, heads, seq_len, head_dim)
        print(f"Sequence length in cache: {seq_len}")
        print("\nCache structure:")
        for i, (key, value) in enumerate(cache):
            key_size = key.numel() * key.element_size() / (1024**2)
            value_size = value.numel() * value.element_size() / (1024**2)
            layer_size = key_size + value_size
            total_size += layer_size
            if i < 3 or i >= len(cache) - 2:  # Show first 3 and last 2 layers
                print(f"  Layer {i}: key {tuple(key.shape)} ({key_size:.2f} MB) + "
                      f"value {tuple(value.shape)} ({value_size:.2f} MB) = {layer_size:.2f} MB")
            elif i == 3:
                print(f"  ...")
        
        print(f"\nTotal cache size: {total_size:.2f} MB (for seq_len={seq_len})")
        print(f"Number of layers: {len(cache)}")
        print(f"Cache per token: {total_size/seq_len:.2f} MB/token")
    
    del model
    torch.cuda.empty_cache()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Calculate cache sizes")
    parser.add_argument(
        "--mamba",
        default="/HSC/users/qiaoye/SSM_SPEC/checkpoints/custom-mamba-65m-multi-gpu",
        help="Path to Mamba model"
    )
    parser.add_argument(
        "--llama",
        default="/HSC/users/qiaoye/checkpoints/Llama-3.2-1B",
        help="Path to Llama model"
    )
    parser.add_argument(
        "--seq-lengths",
        default="512,1024,2048,5120",
        help="Comma-separated sequence lengths to test"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Load models and verify actual cache sizes"
    )
    
    args = parser.parse_args()
    
    seq_lengths = [int(x.strip()) for x in args.seq_lengths.split(',')]
    
    # Theoretical calculation
    compare_cache_growth(args.mamba, args.llama, seq_lengths)
    
    # Actual verification
    if args.verify:
        print("\n\n")
        verify_with_actual_model(args.mamba, model_type="mamba")
        print("\n")
        verify_with_actual_model(args.llama, model_type="transformer")


if __name__ == "__main__":
    main()
