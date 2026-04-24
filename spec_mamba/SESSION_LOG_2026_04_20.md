# SpecSSM Implementation Progress — April 20, 2026

## Overview

This document summarizes all work completed in this session: finishing the paper draft, implementing activation replay for SSM cache resynchronization, building a CPU-optimized selective scan kernel with AVX-512, and comprehensive profiling.

---

## 1. Paper Writing (Completed)

All sections of the NeurIPS 2026 paper have been written with tentative text. Each section contains `% TODO` markers for missing data/experiments.

### Files Modified
| File | Status | Notes |
|------|--------|-------|
| `sections/abstract.tex` | ✅ Written | ~150 words, mean 3.17 accepted tokens |
| `sections/introduction.tex` | ✅ Written | 4 paragraphs + 3 contribution bullets |
| `sections/background.tex` | ✅ Written | AR decoding, spec dec, SSM/Mamba2 notation |
| `sections/related_work.tex` | ✅ Written | Spec dec, draft models, SSMs, positioning |
| `sections/method.tex` | ✅ Written | Guided SSM (§3.1), Activation Replay (§3.2), CPU Offload (§3.3) |
| `sections/experiments_setup.tex` | ✅ Written | Models, baselines, benchmarks, metrics, hardware |
| `sections/results_main.tex` | ✅ Written | References Table 1, discusses acceptance rates |
| `sections/results_analysis.tex` | ✅ Written | Masking analysis with cross-verification table |
| `sections/conclusion.tex` | ✅ Written | Summary + 3 future directions |
| `sections/limitations.tex` | ✅ Written | Model scope, training, masking, broader impact |
| `references.bib` | ✅ Written | 18 BibTeX entries (all cited papers) |
| `tables/main/results.tex` | ✅ Created | Main results table (guided Mamba2 filled in, TODO: baselines) |
| `tables/main/mask_cross.tex` | ✅ Created | Masking analysis cross-verification table |

### Key TODOs in Paper
- Fill in unguided baseline numbers (need `eval.py --baseline`)
- Fill in AR baseline throughput
- Add LLaMA-1B/3B drafter comparison rows
- Add wall-clock speedup measurements
- Add activation replay vs re-prefill latency comparison
- Add CPU offloading paragraph once kernel results are finalized
- Add training hyperparameters details

---

## 2. Activation Replay (Implemented)

### What It Is
After speculative rejection, instead of re-prefilling the entire drafter from scratch (which processes the full sequence), we **snapshot** the drafter's recurrent state before drafting, then **restore + replay** only the accepted tokens.

### Implementation Details

**New utility functions** in `spec_mamba/trainer.py`:
```python
def snapshot_mamba2_cache(cache) -> dict:
    """Save conv_states [16, B, 3072, 4] and ssm_states [16, B, 16, 64, 128]."""

def restore_mamba2_cache(cache, snapshot: dict):
    """Restore Mamba2Cache in-place from snapshot."""
```

**Modified methods:**
- `generate()`: New `use_activation_replay: bool = False` parameter
- `_spec_dec_step()`: New `use_activation_replay: bool = False` parameter
  - Before drafting: `d_cache_snapshot = snapshot_mamba2_cache(d_cache)`
  - After rejection: restore snapshot, replay accepted tokens sequentially

**CLI support:**
- `eval.py`: New `--activation_replay` flag, plumbed through `evaluate_model()` → `mod.generate()`

### Design Choice: Replay vs Re-prefill Equivalence
The activation replay produces a **different** (arguably more correct) drafter state than full re-prefill:
- **Full re-prefill**: processes `sampled[:, :prefill_len]` which includes rejected tokens from previous rounds (the non-compact layout leaves them in place)
- **Activation replay**: restores pre-draft state and only processes accepted tokens

Both produce valid speculative decoding generation. The replay state is cleaner because the drafter never "sees" rejected tokens.

### Cache Memory Footprint
Per batch element (Mamba2-65M, FP32):
- `conv_states`: 16 layers × 3072 × 4 = 768 KB
- `ssm_states`: 16 layers × 16 heads × 64 × 128 = 8,192 KB
- **Total: ~9 MB** per snapshot (negligible)

---

## 3. CPU-Optimized Selective Scan (Implemented)

### Architecture

```
spec_mamba/
├── cpu_mamba2.py           # CPU-optimized Mamba2 forward pass
├── cpu_kernels/
│   ├── __init__.py         # JIT compilation utilities
│   └── ssm_ops.cpp         # AVX-512 C++ kernel
├── profile_replay.py       # Replay-specific profiler
└── profile_all.py          # Comprehensive profiling script
```

### C++ AVX-512 Kernel (`cpu_kernels/ssm_ops.cpp`)

**Two functions exposed via pybind11:**

1. **`ssm_step`**: SSM single-step state update with AVX-512
   ```
   h_new[b,h,d,n] = h_old[b,h,d,n] * exp(dt*A) + dt * B[n] * x[d]
   y[b,h,d] = sum_n(C[n] * h_new[b,h,d,n]) + D * x[d]
   ```
   - Inner loop over `N=128` state dim uses AVX-512 (16 floats/register → 8 iterations)
   - `_mm512_fmadd_ps` for fused multiply-add
   - `_mm512_reduce_add_ps` for horizontal reduction
   - OpenMP parallelization over `(B, H, D)` dimensions

2. **`fused_conv_silu_step`**: Fused depthwise conv1d + SiLU activation
   - Shift sliding window, dot-product with conv weights, SiLU in one pass
   - OpenMP parallelized over `(B, conv_dim)` dimensions

**Build:**
```bash
# JIT compilation (automatic on first import)
python -c "from spec_mamba.cpu_kernels import get_cpu_ssm_ops; get_cpu_ssm_ops()"

# Or via setup.py
python setup_cpu_kernels.py build_ext --inplace
```

**Correctness:** Verified against PyTorch reference (max diff < 1.5e-5, `torch.allclose(atol=1e-5)` passes).

### CPU Mamba2 Model (`cpu_mamba2.py`)

Complete CPU-optimized Mamba2 inference module:

- **`CPUMamba2Cache`**: Memory-contiguous cache with `snapshot()`/`restore()` methods
- **`CPUMamba2Layer`**: Pre-extracts all weights from HF model, eliminates Python dispatch overhead
  - `forward_cached_step()`: Single-step forward with guidance delta injection support
- **`CPUMamba2Model`**: Full model with `forward_step()` and `prefill()` methods
- **`ssm_step_pytorch()`**: Pure PyTorch fallback for correctness verification

---

## 4. Profiling Results

### Hardware
- **CPU**: Intel Xeon Platinum 8562Y+ (128 cores, AVX-512)
- **GPU**: NVIDIA A100-80GB

### Benchmark Results

#### 4.1 AVX-512 Kernel vs PyTorch CPU (SSM step only)
| Implementation | Latency (ms) | Speedup |
|---------------|-------------|---------|
| PyTorch CPU | 0.314 | 1.0x |
| AVX-512 kernel | 0.079 | **4.0x** |

*(B=1, H=16, D=64, N=128 — single SSM step)*

#### 4.2 CPU vs GPU Drafter Single-Step
| Device | Latency (ms) |
|--------|-------------|
| CPU (full model) | 7.3 |
| GPU (HF model) | 8.0 |

**Key finding: CPU is AS FAST as GPU for Mamba2-65M** because the model is too small to saturate GPU compute; kernel launch overhead dominates on GPU.

#### 4.3 CPU Draft 8 Tokens
| Metric | Value |
|--------|-------|
| 8-token draft latency | 57 ms |
| Per-token latency | 7.1 ms |

This is fast enough for async overlap with GPU verification (~100–200ms for LLaMA-8B).

#### 4.4 Activation Replay on GPU
| Seq Length | Re-prefill (ms) | Replay (ms) | Speedup |
|-----------|-----------------|-------------|---------|
| 64 | 16.2 | 40.9 | 0.40x |
| 128 | 16.2 | 40.4 | 0.40x |
| 256 | 16.8 | 40.6 | 0.41x |
| 512 | 16.9 | 40.6 | 0.42x |

**Replay is SLOWER on GPU** because re-prefill uses efficient batched SSD (constant cost for tiny model) while replay needs sequential single-step calls.

#### 4.5 Activation Replay on CPU ⭐
| Seq Length | Re-prefill (ms) | Replay (ms) | Speedup |
|-----------|-----------------|-------------|---------|
| 32 | 299 | 47 | **6.4x** |
| 64 | 602 | 46 | **13.1x** |
| 128 | 1,019 | 37 | **27.7x** |
| 256 | 1,841 | 37 | **50.4x** |
| 512 | 3,644 | 36 | **100.9x** |
| 1,024 | 8,036 | 40 | **201.0x** |

**Activation replay is massively beneficial on CPU!** Re-prefill cost grows linearly with sequence length while replay cost is constant (~37–40ms, proportional only to `n_accepted + 1` tokens).

### Key Insights

1. **CPU offloading is viable**: The 65M Mamba2 drafter runs at near-GPU speed on CPU, enabling true heterogeneous deployment where GPU memory is fully reserved for the verifier.

2. **Activation replay is critical for CPU path**: Without replay, the CPU re-prefill after each round would be 300ms–8000ms (growing with sequence length). With replay, it's a constant ~37ms.

3. **Activation replay is NOT helpful on GPU for tiny models**: The GPU can re-prefill any sequence length in ~16ms because the model is too small to saturate GPU compute. The overhead of sequential single-step replay (5 × 8ms = 40ms) exceeds this.

4. **Pipeline parallelism opportunity**: CPU drafts 8 tokens in 57ms. LLaMA-8B verification takes ~100–200ms on GPU. These can overlap, giving near-full utilization of both CPU and GPU.

---

## 5. Files Created/Modified

### New Files
| File | Purpose |
|------|---------|
| `spec_mamba/cpu_mamba2.py` | CPU-optimized Mamba2 forward pass module |
| `spec_mamba/cpu_kernels/__init__.py` | Kernel build utilities |
| `spec_mamba/cpu_kernels/ssm_ops.cpp` | AVX-512 SSM kernel |
| `spec_mamba/profile_replay.py` | Replay-specific profiling script |
| `spec_mamba/profile_all.py` | Comprehensive profiling script |
| `setup_cpu_kernels.py` | Build script for C++ extension |
| `paper/neurips2026-specssm/tables/main/results.tex` | Main results table |
| `paper/neurips2026-specssm/tables/main/mask_cross.tex` | Masking analysis table |

### Modified Files
| File | Changes |
|------|---------|
| `spec_mamba/trainer.py` | Added `snapshot_mamba2_cache()`, `restore_mamba2_cache()`, `use_activation_replay` parameter to `generate()` and `_spec_dec_step()` |
| `spec_mamba/eval.py` | Added `--activation_replay` flag, plumbed through to `generate()` |
| `paper/neurips2026-specssm/sections/*.tex` | All 9 sections written with tentative text |
| `paper/neurips2026-specssm/references.bib` | 18 BibTeX entries added |

---

## 6. Remaining Work

### P0 (Must-have for paper)
- [ ] **W1**: Run unguided baseline: `python -m spec_mamba.eval --ckpt ... --baseline --greedy_only --bsz 1 --no_mask`
- [ ] **W5**: Measure AR baseline throughput (autoregressive LLaMA-8B alone)
- [ ] **W9**: Text quality metrics (BLEU/ROUGE of spec dec output vs gold AR)
- [ ] Integrate activation replay profiling numbers into paper Table
- [ ] Verify activation replay doesn't degrade acceptance rates significantly (run full eval with `--activation_replay`)

### P1 (Highly desirable)
- [ ] Implement async CPU-GPU pipeline (CPU drafts while GPU verifies)
- [ ] Use AVX-512 kernel in `CPUMamba2Layer.forward_cached_step()` for fused conv+SiLU
- [ ] Batched CPU drafting for bsz > 1
- [ ] NG sweep (K=4,6,8,12)
- [ ] Guidance layer ablation
- [ ] z-branch ablation (retrain with steer_z=True)

### P2 (Nice to have)
- [ ] Drafter scaling study (130M, 370M)
- [ ] Tree-structured drafting
- [ ] INT8 quantized drafter

---

## 7. How to Run

### Profiling
```bash
# Full profiling suite
python -m spec_mamba.profile_all \
    --ckpt /HSC/users/qiaoye/SSM_SPEC/checkpoints/guided_mamba/ckpts/last.ckpt \
    --n_iters 100

# Quick test
python -m spec_mamba.profile_all --ckpt ... --quick

# Replay-specific profiling
python -m spec_mamba.profile_replay \
    --ckpt /HSC/users/qiaoye/SSM_SPEC/checkpoints/guided_mamba/ckpts/last.ckpt \
    --seq_lens 64,128,256,512,1024
```

### Evaluation with Activation Replay
```bash
python -m spec_mamba.eval \
    --ckpt /HSC/users/qiaoye/SSM_SPEC/checkpoints/guided_mamba/ckpts/last.ckpt \
    --greedy_only --bsz 1 --no_mask --activation_replay \
    --out_file spec_mamba/eval_results_replay.json
```

### Building the C++ Kernel
```bash
# JIT (automatic on first import)
python -c "from spec_mamba.cpu_kernels import get_cpu_ssm_ops; get_cpu_ssm_ops()"

# Or manual build
python setup_cpu_kernels.py build_ext --inplace
```
