# Activation Replay & CPU-Optimized Selective Scan — Implementation Details

*Last updated: 2026-04-20*

This document covers the two core systems optimizations in SpecSSM:
1. **Activation Replay** — efficient SSM cache resynchronization after speculative rejection
2. **CPU-Optimized Selective Scan** — AVX-512 kernel and model-level optimizations for CPU-side drafting

---

## Table of Contents

- [1. The Problem: SSM Cache Resynchronization](#1-the-problem-ssm-cache-resynchronization)
- [2. Activation Replay](#2-activation-replay)
  - [2.1 Design Rationale](#21-design-rationale)
  - [2.2 Implementation](#22-implementation)
  - [2.3 Correctness Considerations](#23-correctness-considerations)
  - [2.4 Profiling Results](#24-profiling-results)
- [3. CPU-Optimized Selective Scan](#3-cpu-optimized-selective-scan)
  - [3.1 Why CPU?](#31-why-cpu)
  - [3.2 Mamba2 Single-Step Breakdown](#32-mamba2-single-step-breakdown)
  - [3.3 AVX-512 SSM Kernel](#33-avx-512-ssm-kernel)
  - [3.4 Model-Level Optimizations](#34-model-level-optimizations)
  - [3.5 Profiling Results](#35-profiling-results)
- [4. Files and Usage](#4-files-and-usage)

---

## 1. The Problem: SSM Cache Resynchronization

In speculative decoding, the drafter proposes K=8 tokens. The verifier accepts some prefix (say n_a tokens) and rejects the rest. The drafter must then prepare its internal state to continue drafting from the last accepted position.

**Transformer drafters** solve this trivially: truncate the KV cache to position n_a. This is an O(1) pointer update.

**SSM drafters** have no such luxury. Mamba2's recurrent hidden state `h_t` encodes the *entire* history through the recurrence:

```
h_t = A_bar * h_{t-1} + B_bar * x_t
```

After drafting K tokens, the SSM state `h_{t+K}` has been contaminated by rejected tokens `x_{t+n_a+1}, ..., x_{t+K}`. There is no way to "undo" these updates — the state is a lossy compression of the full history.

The convolution buffer (kernel size 4) has a similar problem: it stores the last 4 projected inputs, and rejected tokens have been shifted into it.

### The Original Approach: Full Re-Prefill

The naive fix (and our original implementation) was to discard the drafter cache entirely after each rejection round and re-run the drafter over the **entire** accepted sequence from scratch:

```python
# Original: rebuild from scratch
d_cache = Mamba2Cache(config, B, device=device, dtype=torch.float32)
_, d_cache = drafter_forward_cached(
    sampled[:, :prefill_len],   # <-- re-processes ALL accepted tokens
    zero_deltas, cache_params=d_cache, ...
)
```

The cost grows linearly with sequence length. At 1024 tokens, this re-prefill on CPU takes ~8 seconds per round — completely unacceptable.

---

## 2. Activation Replay

### 2.1 Design Rationale

The key observation: **we only need to process the newly accepted tokens**, not the entire history. If we save the drafter's state *before* drafting begins, we can restore it after rejection and replay just the accepted tokens plus the resampled correction token.

**Why this works**: The SSM recurrence is deterministic. Starting from the same state `h_t` and processing the same tokens `x_{t+1}, ..., x_{t+n_a}` will always produce the same state `h_{t+n_a}`. So restoring a pre-draft snapshot and replaying =n_a + 1 tokens gives us an identical state to having never drafted the rejected tokens.

**Why snapshot+restore instead of backward unrolling**: The SSM recurrence `h_new = h_old * exp(dt*A) + dt*B*x` mixes the state multiplicatively and additively at every step. Reversing this would require storing intermediate `dA` values for every drafted token and doing the same number of operations in reverse — no savings. Snapshot+restore is one tensor copy.

**Complexity**: Full re-prefill is O(T) where T is total sequence length. Activation replay is O(n_a + 1) where n_a ≤ K is the number of accepted tokens. Since n_a is at most 8 (our K), **the replay cost is constant regardless of sequence length**.

### 2.2 Implementation

The implementation lives in `spec_mamba/trainer.py` and consists of three parts:

#### Part 1: Snapshot/Restore Utilities

```python
def snapshot_mamba2_cache(cache) -> dict:
    """Save a deep copy of Mamba2Cache conv_states and ssm_states."""
    return {
        "conv_states": cache.conv_states.clone(),
        "ssm_states": cache.ssm_states.clone(),
    }

def restore_mamba2_cache(cache, snapshot: dict):
    """Restore Mamba2Cache from a snapshot (in-place)."""
    cache.conv_states.copy_(snapshot["conv_states"])
    cache.ssm_states.copy_(snapshot["ssm_states"])
```

**What gets saved**: The `Mamba2Cache` object contains two tensors:
- `conv_states`: `[16 layers, B, 3072, 4]` — the sliding window buffer for the depthwise conv1d (kernel size 4)
- `ssm_states`: `[16 layers, B, 16 heads, 64 head_dim, 128 state_size]` — the recurrent hidden state

**Total snapshot size**: ~8.96 MB for Mamba2-65M (B=1). This is small enough that the clone cost is negligible (~0.02ms on GPU, ~0.1ms on CPU).

We use `clone()` for the snapshot (not a reference) because the cache is modified in-place during drafting. We use `copy_()` for restore (not re-assignment) because the cache object is shared by reference with the Mamba2 layers.

#### Part 2: Snapshot Before Drafting

In `_spec_dec_step()`, right before the draft loop:

```python
if use_activation_replay:
    d_cache_snapshot = snapshot_mamba2_cache(d_cache)
    d_next_pos_snapshot = d_next_pos
```

We also save `d_next_pos` — the cache position counter that tracks how many tokens the drafter has processed. This is needed so replay tokens are written to the correct position in the conv buffer.

#### Part 3: Replay After Rejection

After the verifier determines `NA` (number accepted per batch element) and performs rejection sampling:

```python
if use_activation_replay:
    # 1. Restore pre-draft state
    restore_mamba2_cache(d_cache, d_cache_snapshot)

    # 2. Build replay sequence: accepted tokens + resampled next_token
    max_replay_len = NA.max().item() + 1
    replay_ids = torch.full((B, max_replay_len), pad_token_id, ...)
    for b in range(B):
        na_b = NA[b].item()
        replay_ids[b, :na_b] = sampled[b, old_curr+1 : old_curr+1+na_b]  # accepted
        replay_ids[b, na_b] = next_token[b]                                # resampled

    # 3. Replay tokens one-at-a-time through cached forward
    for ri in range(max_replay_len):
        _, d_cache = self._drafter_forward_cached(
            replay_ids[:, ri:ri+1],
            zero_guidance_deltas,
            cache_params=d_cache,
            cache_position=torch.tensor([d_next_pos_snapshot + ri]),
        )
    d_next_pos = d_next_pos_snapshot + max_replay_len
```

**Why one-at-a-time**: The cached Mamba2 forward path processes one token per call, updating `conv_states` and `ssm_states` incrementally. Batching multiple replay tokens into a single call would require using the non-cached (prefill) forward path, which allocates temporary tensors and uses a different code path. Single-step is simpler and matches exactly how the drafter is used during actual drafting.

**Why zero guidance deltas during replay**: Guidance deltas are derived from verifier hidden states at the *previous* round's verification position. During replay, we're reconstructing the drafter's state using accepted tokens — these tokens were drafted *without* updated guidance. Using zero deltas is consistent with how the drafter was called during the original drafting loop (where the guidance comes from the previous round, not the current round). A future improvement could inject per-position guidance, but this requires storing the verifier's hidden states for each accepted position.

### 2.3 Correctness Considerations

**Is the replay state bit-identical to never having rejected?**

In principle, yes: restoring the snapshot produces the exact pre-draft state, and replaying the accepted tokens produces a state that only reflects those tokens. However, in practice, the replay state differs from the full re-prefill state because of the **non-compact layout**:

- **Full re-prefill** processes `sampled[:, :prefill_len]`, which includes *all* tokens in the buffer — that means all previously rejected tokens from prior rounds are also in the sequence (at positions where `attention_mask=0`). The Mamba2 drafter has no attention mechanism, so it actually *processes* these tokens (attention_mask is only used by the transformer verifier). This means the drafter's state after full re-prefill is contaminated by all prior rejected tokens.

- **Activation replay** never sees rejected tokens at all — neither from the current round nor from prior rounds.

**Activation replay is arguably more correct**: the drafter should not be influenced by tokens that the verifier rejected. The full re-prefill state includes noise from rejected tokens, which is a subtle bug in the original approach. However, since the drafter was trained in the full-re-prefill regime, there may be a slight distribution shift when switching to replay.

### 2.4 Profiling Results

**GPU (NVIDIA A100-80GB)**: Activation replay is NOT beneficial for Mamba2-65M on GPU.

| Seq Length | Re-prefill | Replay | Speedup |
|-----------|------------|--------|---------|
| 64 | 16.2 ms | 40.9 ms | 0.40x (slower) |
| 128 | 16.2 ms | 40.4 ms | 0.40x |
| 256 | 16.8 ms | 40.6 ms | 0.41x |
| 512 | 16.9 ms | 40.6 ms | 0.42x |

Why: The 65M model is too small to saturate the GPU. Re-prefill processes 64-512 tokens in a single parallel call (~16ms regardless of length). Replay makes 5 sequential single-step calls (each 8ms due to kernel launch overhead). The GPU's parallelism makes bulk operations cheap.

**CPU (Intel Xeon 8562Y+)**: Activation replay is massively beneficial.

| Seq Length | Re-prefill | Replay | Speedup |
|-----------|------------|--------|---------|
| 32 | 299 ms | 47 ms | **6.4x** |
| 64 | 602 ms | 46 ms | **13.1x** |
| 128 | 1,019 ms | 37 ms | **27.7x** |
| 256 | 1,841 ms | 37 ms | **50.4x** |
| 512 | 3,644 ms | 36 ms | **100.9x** |
| 1,024 | 8,036 ms | 40 ms | **201.0x** |

Why: On CPU, each token is processed sequentially through 16 layers with no parallelism benefit from batching. Re-prefill cost scales linearly with sequence length (7ms/token × T tokens). Replay cost is constant at ~37ms (7ms/token × 5 tokens).

**Conclusion**: Activation replay is essential for CPU-side drafting and irrelevant for GPU-side drafting (at this model size).

---

## 3. CPU-Optimized Selective Scan

### 3.1 Why CPU?

The Mamba2-65M drafter is extremely small:
- **Parameters**: 65M (~130 MB in FP32)
- **Cache**: ~9 MB per batch element
- **Total memory**: ~140 MB

On a GPU serving a 16GB LLaMA-3.1-8B, the drafter is less than 1% of memory — but while it's generating draft tokens, the GPU's verifier is idle. Meanwhile, the CPU has nothing to do during drafting.

**Heterogeneous deployment**: By offloading the drafter to CPU, we achieve two benefits:
1. **GPU memory savings**: The full GPU memory is available for the verifier's KV cache, enabling longer sequences or larger batch sizes
2. **Pipeline parallelism** (future work): Draft on CPU and verify on GPU concurrently

**Feasibility**: Our profiling shows that a **CPU single-step takes 7.3ms** vs GPU's 8.0ms (the model is too small to benefit from GPU parallelism). For 8-token drafting, the CPU takes ~57ms — comparable to a single GPU verifier forward pass for LLaMA-8B, making async overlap viable.

**Correctness**: The CPU model was verified (2026-04-21) to produce identical top-1 tokens as HF's Mamba2 across 16 autoregressive steps with max logit diff ~4e-6. A bug in `_rms_norm_with_gate` (gate applied after norm instead of before) caused large divergence and was fixed.

### 3.2 Mamba2 Single-Step Breakdown

Each token through one Mamba2 layer involves these operations:

```
1. RMSNorm          [B, 512]      → [B, 512]          matmul-free
2. in_proj           Linear(512 → 2320)                1.2M FLOPs
3. Guidance inject   x += delta                        addition
4. Conv1d (cached)   slide window(4) over 1280 dims    5K FLOPs
5. SiLU activation                                     element-wise
6. Split: x(1024) + B(128) + C(128)                   view ops  [n_groups=1]
7. SSM step          h = h*dA + dB*x; y = C@h + D*x   main bottleneck
8. RMSNorm + gate    norm(y * silu(gate)) * weight     element-wise
9. out_proj          Linear(1024 → 512)                0.5M FLOPs
10. Residual add                                       element-wise
```

The **SSM step** (step 7) is the computational bottleneck because it operates on the full state tensor:
- State shape: `[B, 16 heads, 64 head_dim, 128 state_size]` = 131,072 elements per batch
- Each element requires: 1 exp, 1 multiply, 1 FMA for state update, plus 1 FMA for output reduction
- Total per layer: ~16 × 64 × 128 × 4 = 524,288 FLOPs

Across 16 layers, the SSM recurrence accounts for ~8.4M FLOPs per token — by far the largest operation.

### 3.3 AVX-512 SSM Kernel

**Location**: `spec_mamba/cpu_kernels/ssm_ops.cpp`

The SSM single-step recurrence processes independently over all (batch, head, head_dim) positions. For each position:

```
dt_val = clamp(softplus(dt + dt_bias), lo, hi)    // activation
dA     = exp(dt_val * A)                           // scalar
// For each n in 0..N-1:
h_new[n] = h_old[n] * dA + dt_val * x * B[n]      // state update (FMA)
y       += C[n] * h_new[n]                          // output reduction (FMA)
y       += D * x                                    // skip connection
```

#### Why AVX-512 is a perfect fit

The inner loop is over `N=128` (state_size), and each iteration does two FMA operations on independent data. AVX-512 processes 16 float32s per instruction, so 128 elements = exactly **8 AVX-512 iterations**. No remainder handling needed (though we include a scalar fallback for portability).

The core loop in AVX-512 intrinsics:

```c
__m512 v_dA = _mm512_set1_ps(dA_val);
__m512 v_dBx_scale = _mm512_set1_ps(dt_val * x_val);
__m512 v_y_accum = _mm512_setzero_ps();

for (int n = 0; n + 16 <= N; n += 16) {
    __m512 v_h_old = _mm512_loadu_ps(h_old + n);
    __m512 v_B     = _mm512_loadu_ps(B_bh + n);
    __m512 v_C     = _mm512_loadu_ps(C_bh + n);

    // h_new = h_old * dA + dBx_scale * B    (FMA)
    __m512 v_h_new = _mm512_fmadd_ps(v_h_old, v_dA,
                         _mm512_mul_ps(v_dBx_scale, v_B));
    _mm512_storeu_ps(h_new + n, v_h_new);

    // y += C * h_new    (FMA)
    v_y_accum = _mm512_fmadd_ps(v_C, v_h_new, v_y_accum);
}
y_accum = _mm512_reduce_add_ps(v_y_accum);  // horizontal sum
```

**Key design choices**:

1. **FMA instructions** (`_mm512_fmadd_ps`): The state update `h_old * dA + scale * B` maps directly to a fused multiply-add. Using FMA instead of separate mul+add halves the instruction count and improves numerical precision.

2. **Broadcast scalars**: `dA` and `dBx_scale` are constant across the N dimension, so we broadcast them once into AVX-512 registers (`_mm512_set1_ps`).

3. **Horizontal reduction only once**: The output `y` accumulates 128 products into a single scalar. We use a vector accumulator throughout the loop and reduce only at the end with `_mm512_reduce_add_ps`.

4. **OpenMP parallelization** (`#pragma omp parallel for collapse(3)`): The outer loops over (B, H, D) = (1, 16, 64) = 1024 independent work items are parallelized across CPU cores. With 128 cores available on the Xeon 8562Y+, this is well-utilized.

5. **Contiguous memory access**: The state tensor layout `[B, H, D, N]` means the inner loop over N accesses contiguous memory, maximizing L1/L2 cache line utilization. Each AVX-512 load fetches exactly one 64-byte cache line (16 × 4 bytes).

6. **No dynamic memory allocation**: The kernel pre-allocates output tensors and writes directly to them. No intermediate buffers are created inside the hot loop.

#### Why not vectorize the `exp()` call?

The `exp(dt_val * A)` computation is a scalar (one per (b,h,d) position). Vectorizing it across the N dimension wouldn't help because dA is constant there. Across the D dimension it could help, but D=64 is small and `exp()` is not available as an AVX-512 intrinsic — it would require a polynomial approximation (like Intel SVML's `_mm512_exp_ps`), and the scalar `std::exp` from libm is accurate and fast enough given that it's called only once per (b,h,d) position.

#### Fused Conv1d + SiLU

The kernel also provides `fused_conv_silu_step()` which combines:
- Conv state shift (shift left, append new input)
- Depthwise conv1d dot product (4 multiplies + 3 adds per channel)
- SiLU activation

This avoids materializing intermediate tensors for the conv output.

### 3.4 Model-Level Optimizations

**Location**: `spec_mamba/cpu_mamba2.py`

Beyond the SSM kernel, the full CPU model (`CPUMamba2Model`) includes several optimizations:

#### Weight Pre-Extraction

The HuggingFace `Mamba2ForCausalLM` uses nested `nn.Module` dispatch with Python `__call__` overhead, hook resolution, and dtype casting at every layer. Our CPU model pre-extracts all weights at initialization:

```python
class CPUMamba2Layer:
    def __init__(self, hf_block, layer_idx):
        # Store weights as contiguous CPU float32 tensors
        self.in_proj_weight = mixer.in_proj.weight.float().contiguous().cpu()
        self.conv1d_weight = mixer.conv1d.weight.float().contiguous().cpu()
        self.A_log = mixer.A_log.float().contiguous().cpu()
        # ... etc
```

**Why this helps**: Each `nn.Module.__call__` in PyTorch invokes ~20 Python function calls (hooks, context managers, autograd dispatch). For 16 layers × 10 operations/layer, that's ~3200 Python calls per token. By storing raw tensors and using `F.linear()` directly, we eliminate this overhead.

**Why `.contiguous().cpu()`**: Ensures tensors have a contiguous memory layout on CPU. Some HF model weights may be non-contiguous after slicing/transposition, which would cause performance degradation in BLAS calls.

#### Cache Layout

`CPUMamba2Cache` mirrors the HF `Mamba2Cache` but as a plain Python class (not `nn.Module`), with `snapshot()`/`restore()` methods built in:

- `conv_states`: `[16, B, 3072, 4]` — conv kernel=4, so 4 history entries per channel per layer
- `ssm_states`: `[16, B, 16, 64, 128]` — 16 heads × 64 head_dim × 128 state_size per layer

The N=128 (state_size) dimension is innermost, matching the AVX-512 kernel's access pattern.

#### Explicit Layer Pipeline

Instead of relying on HF's `model.forward()` dispatcher:

```python
def forward_step(self, token_ids, cache, guidance_deltas=None):
    hidden = F.embedding(token_ids, self.embed_weight)
    for i, layer in enumerate(self.layers):
        hidden = layer.forward_cached_step(hidden, cache, guidance_deltas[i])
    hidden = rms_norm(hidden, self.norm_f_weight)
    logits = F.linear(hidden.squeeze(1), self.lm_head_weight)
    return logits
```

No autograd, no hooks, no dtype checks, no device transfers. Pure compute.

### 3.5 Profiling Results

**Hardware**: Intel Xeon Platinum 8562Y+, 128 cores, AVX-512 support.

#### SSM Kernel Micro-Benchmark (B=1, H=16, D=64, N=128)

| Backend | Latency | Speedup |
|---------|---------|---------|
| PyTorch CPU (torch ops) | 0.314 ms | 1.0x |
| AVX-512 C++ kernel | 0.078 ms | **4.0x** |

The 4x speedup comes from: (a) eliminating PyTorch operator dispatch and tensor allocation, (b) using FMA intrinsics instead of separate mul/add, (c) efficient horizontal reduction, (d) no intermediate tensor materialization.

#### Full Model Single-Step

| Backend | Latency | Notes |
|---------|---------|-------|
| GPU (A100, HF forward) | 8.03 ms | Kernel launch overhead dominates |
| CPU (optimized, 16 layers) | 7.31 ms | 0.9x GPU — **competitive!** |

The GPU is not faster because: (a) the 65M model has no large matrix multiply that benefits from GPU SIMD width, (b) each single-step involves 16 sequential layer calls with GPU kernel launch overhead (~0.5ms each), (c) CPU-GPU data transfer would add additional latency in a heterogeneous setup.

#### Draft K=8 Tokens End-to-End

| Backend | Latency |
|---------|---------|
| CPU | 57.1 ms |
| GPU (estimated) | 64.2 ms |

The CPU is actually **faster** for generating 8 draft tokens, because it avoids kernel launch overhead.

#### Context: LLaMA-3.1-8B Verifier Forward Pass

For reference, a single LLaMA-8B verifier forward pass (processing 9 tokens: K=8 draft + 1 continuation) takes approximately 50-60ms on an A100. This means the CPU drafter's 57ms fits perfectly within the verifier's processing window for future async pipeline overlap.

---

## 4. Files and Usage

### Source Files

| File | Purpose |
|------|---------|
| `spec_mamba/trainer.py` | `snapshot_mamba2_cache()`, `restore_mamba2_cache()`, `use_activation_replay` parameter on `generate()` and `_spec_dec_step()` |
| `spec_mamba/cpu_mamba2.py` | `CPUMamba2Model`, `CPUMamba2Cache`, `CPUMamba2Layer`, `ssm_step_pytorch()`, `fused_conv1d_silu_cached()` |
| `spec_mamba/cpu_kernels/ssm_ops.cpp` | AVX-512 C++ kernel: `ssm_step_avx512()`, `fused_conv_silu_step()` |
| `spec_mamba/cpu_kernels/__init__.py` | JIT compilation loader (`get_cpu_ssm_ops()`) |
| `spec_mamba/profile_all.py` | Comprehensive profiling script |
| `spec_mamba/profile_replay.py` | Activation replay-specific profiling |
| `spec_mamba/eval.py` | Evaluation with `--activation_replay` flag |

### Running Evaluation with Activation Replay

```bash
# GPU with activation replay (not recommended for this model size)
python -m spec_mamba.eval \
    --ckpt /path/to/last.ckpt \
    --activation_replay --no_mask --greedy_only --bsz 1

# Run profiling
python -m spec_mamba.profile_all \
    --ckpt /path/to/last.ckpt --quick
```

### Building the AVX-512 Kernel

The kernel is JIT-compiled on first import:

```python
from spec_mamba.cpu_kernels import get_cpu_ssm_ops
ops = get_cpu_ssm_ops()  # compiles on first call, cached afterwards
```

Requirements: `pybind11`, a C++17 compiler with AVX-512 support, OpenMP.

Compile flags: `-O3 -march=native -mavx512f -mavx512bw -mavx512dq -mavx512vl -mfma -fopenmp -std=c++17`

### Using the CPU Model Directly

```python
from spec_mamba.cpu_mamba2 import CPUMamba2Model

# Initialize from HuggingFace model
cpu_model = CPUMamba2Model(hf_mamba2_model)
cache = cpu_model.create_cache(batch_size=1)

# Prefill
cpu_model.prefill(prompt_ids, cache)

# Generate
for _ in range(K):
    logits = cpu_model.forward_step(token_ids, cache)
    token_ids = logits.argmax(dim=-1, keepdim=True)
```
