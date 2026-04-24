# CPU SSM Kernel Optimization Roadmap

*Created: 2026-04-21*
*Hardware: Intel Xeon Platinum 8562Y+ (Sapphire Rapids)*

## Current State

Our AVX-512 FP32 kernel achieves **4x speedup** over PyTorch CPU:
- SSM step: 0.079ms (AVX-512) vs 0.314ms (PyTorch)
- Full single-step through 16 layers: ~7.3ms
- 8-token draft: ~57ms total

## Available Hardware Instructions

The Xeon 8562Y+ (Sapphire Rapids) supports:
- `avx512f`, `avx512bw`, `avx512dq`, `avx512vl` — used in current kernel
- **`avx512_fp16`** — native FP16 arithmetic (32 FP16 per register!)
- **`avx512_bf16`** — native BF16 multiply-accumulate
- **`avx512_vnni`** — INT8/INT16 dot products (4x INT8 per lane)
- **`amx_bf16`** — tile-based BF16 matrix multiply (for matmul)
- **`amx_int8`** — tile-based INT8 matrix multiply
- `f16c` — FP16↔FP32 conversion

## Optimization Options

### Option 1: FP16 SSM Kernel (`avx512_fp16`)

**What changes**: The SSM recurrence processes N=128 states per (head, dim) position. With FP16:
- **32 FP16 values per AVX-512 register** (vs 16 FP32)
- N=128 = **4 AVX-512 iterations** (vs 8 for FP32)
- ~2x bandwidth reduction (256 bytes → 128 bytes per state vector)
- `_mm512_fmadd_ph` for FMA in FP16 natively

**Concerns**:
- SSM recurrence involves `exp(dt * A)` — this can produce extreme values. FP16 range is ±65504, which may overflow for large dt×A products.
- State accumulation: `h_new = h_old * dA + dBx` over many steps could lose precision if h_old grows large.
- The `exp()` and `softplus()` in dt processing are inherently FP32 operations.

**Recommended approach**: 
- Keep dt processing (softplus, clamp, exp) in FP32
- Store and accumulate states in FP16 using `_mm512_fmadd_ph`
- Use FP16 for B, C, x vectors
- Profile accuracy: compare FP16 SSM output vs FP32 reference over 1000 steps

**Expected speedup**: ~1.5–2x over current FP32 kernel (2x from register width, minus conversion overhead).

**Accuracy test** (200 sequential steps with random inputs):
| Steps | Max |y_diff| | Max |state_diff| | Relative state error |
|---|---|---|---|
| 1 | 0.026 | 0.009 | — |
| 50 | 0.103 | 0.020 | 0.10% |
| 100 | 0.132 | 0.020 | 0.08% |
| 200 | 0.220 | 0.041 | 0.21% |

Conclusion: **FP16 is viable** — relative error stays under 0.3% even after 200 steps.

**Implementation complexity**: Medium. Need `#include <immintrin.h>` with `AVX512FP16` intrinsics (_ph suffix). Compiler flag: `-mavx512fp16`.

### Option 2: BF16 SSM Kernel (`avx512_bf16`)

**What changes**: BF16 has the same exponent range as FP32 (8 bits) but only 7-bit mantissa. Solves the overflow concern of FP16.

**Tradeoff**: 
- `avx512_bf16` only supports multiply-accumulate (`_mm512_dpbf16_ps`): BF16 × BF16 → accumulated into FP32
- Cannot do pure BF16 FMA for the state update `h_new = h_old * dA + dBx`
- Need to convert h_old to BF16, multiply, accumulate in FP32, then convert back
- Still 32 BF16 per register for storage, but arithmetic is BF16→FP32 accumulate

**Expected speedup**: ~1.3x (mainly from reduced memory bandwidth, not compute).

**Implementation complexity**: Medium-low. The `_mm512_dpbf16_ps` paradigm is well-suited for the `y += C * h_new` reduction but less ideal for the state update.

### Option 3: INT8 State Quantization (`avx512_vnni`)

**What changes**: Quantize the SSM state `h[B,H,D,N]` to INT8, keeping computation in FP32 but reducing memory bandwidth.

**Why this could work**:
- The SSM state is the largest tensor: [1, 16, 64, 128] = 128K floats = 512KB
- Reading/writing 128KB per layer per step is the bandwidth bottleneck
- INT8 state = 128KB → much more cache-friendly
- `avx512_vnni` provides `_mm512_dpbusd_epi32`: 4×INT8 dot product → INT32

**Challenge**: SSM state values are not naturally bounded. Need dynamic quantization:
```
h_quant = round(h_float * scale)      // FP32 → INT8
h_float = h_quant / scale             // INT8 → FP32 (when needed)
```
- The `scale` must be updated every step based on max(|h_new|)
- Quantization error accumulates: ~1% per step → significant after 100+ steps

**Expected speedup**: ~1.5x from memory bandwidth reduction, but accuracy degradation is a real risk.

**Implementation complexity**: High. Dynamic quantization per-step, accuracy validation needed.

### Option 4: AMX Tiles for Linear Layers (`amx_bf16` / `amx_int8`)

**What changes**: The bottleneck is NOT the SSM step (0.079ms) — it's the linear projections:
- `in_proj`: Linear(512 → 2320) → most expensive at ~60% of per-layer time
- `out_proj`: Linear(1024 → 512) → ~25%
- SSM step → ~10%

AMX (Advanced Matrix Extensions) provides hardware matrix multiply:
- AMX-BF16: tile-based BF16 GEMM, up to 16×16 tiles
- AMX-INT8: tile-based INT8 GEMM

**Why this is the biggest win**: `F.linear(x, weight)` calls into MKL/BLAS which may already use AMX internally for BF16/INT8, but our weights are currently FP32. Converting weights to BF16 and using `amx_bf16` could give:
- **2x speedup on linear layers** (which are 85% of total time)
- Overall single-step speedup: 7.3ms → ~4ms

**Recommended approach**:
1. Store `in_proj_weight`, `out_proj_weight`, `conv1d_weight` in BF16
2. Use `torch.mm` with BF16 tensors (PyTorch should use AMX automatically on Sapphire Rapids)
3. Keep activations in FP32 (compute in mixed precision)

**Expected speedup**: 1.5–2x on total single-step time.

**Empirical result** (quick test on Xeon 8562Y+):
| Layer | FP32 | BF16 | Speedup |
|---|---|---|---|
| in_proj (512 → 2320) | 0.379 ms | 0.039 ms | **9.6x** |
| out_proj (1024 → 512) | 0.016 ms | 0.042 ms | 0.39x (overhead) |

Key insight: BF16/AMX gives massive speedup for larger matmuls (in_proj) but adds overhead for tiny ones (out_proj). **Strategy: use BF16 only for in_proj, keep FP32 for out_proj.**

**Implementation complexity**: Low! Just convert weights:
```python
self.in_proj_weight = backbone.layers[i].mixer.in_proj.weight.bfloat16().contiguous().cpu()
# Then in forward: F.linear(x.bfloat16(), self.in_proj_weight).float()
```

### Option 5: Torch.compile with CPU Backend

**What changes**: Use `torch.compile` on the full `forward_step` to get fused kernels.

**Concerns**: `torch.compile` may not support our custom C++ kernel calls. Could work if we:
- Use the PyTorch fallback SSM step (no AVX-512)
- Let `torch.compile` fuse the entire layer including SSM

**Expected speedup**: Unpredictable. Could be 1.5–3x if fusion works, or slower if compilation fails.

## Priority Ranking

| Priority | Option | Expected Speedup | Effort | Risk |
|---|---|---|---|---|
| **P0** | #4: BF16 weights + AMX matmul | 1.5–2x total | Low | Low |
| **P1** | #1: FP16 SSM kernel | 1.5x SSM step | Medium | Medium (accuracy) |
| **P2** | #5: torch.compile | Unknown | Low | High |
| **P3** | #2: BF16 SSM kernel | 1.3x SSM step | Medium | Low |
| **P4** | #3: INT8 quantization | 1.5x bandwidth | High | High (accuracy) |

## Recommended Next Steps

1. **Quick win**: Convert weights to BF16, measure single-step time. This requires ~10 lines of code.
2. **FP16 SSM kernel**: Write an `avx512_fp16` variant of `ssm_step_avx512`. Validate accuracy over 100+ steps.
3. **Profile first**: Run `perf stat` on the current kernel to measure actual CPI, cache misses, and bandwidth utilization before optimizing.

## Profile Command
```bash
perf stat -e task-clock,cycles,instructions,cache-references,cache-misses \
  python -c "from spec_mamba.cpu_mamba2 import ...; # run 1000 steps"
```
