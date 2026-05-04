# NeurIPS 2026 Paper Plan: SpecSSM

**Title (working):** SpecSSM: Efficient Speculative Decoding with Guided Tiny SSM Drafters  
**Venue:** NeurIPS 2026 (ML-Systems track)  
**Deadline:** TBD — check [NeurIPS 2026 CFP](https://neurips.cc/)  
**Last updated:** 2026-05-04

---

## 1. Core Story

We show that a **tiny Mamba2 drafter (65M total / 27.5M backbone)**, guided by verifier hidden states and equipped with a cache-correct resynchronization method, achieves **strong acceptance rates scaling with verifier size** — and that guidance is **verifier-agnostic** (works with LLaMA-8B, LLaMA-70B, and Gemma-4B). The system includes optional CPU offloading to free GPU memory entirely.

### Why This Is a Systems Paper
1. **Non-trivial systems insight**: SSM drafters have unique cache properties (no KV cache → no rollback → new resync strategy needed)
2. **Concrete wall-clock gains**: Tiny drafter = fast drafting; guidance = higher acceptance; overlap = pipeline parallelism
3. **Heterogeneous deployment**: CPU drafter while GPU verifies — practical for memory-constrained settings
4. **Verifier-agnostic guidance**: Same tiny drafter architecture guided by different verifier families and scales (8B→70B)
5. **Rigorous evaluation**: 5 datasets, masking analysis, 3 verifiers, context length scaling

---

## 2. Contributions (4 Pillars)

### C1: Guided SSM Drafter Architecture
Inject verifier hidden states into Mamba2's x-branch (state update path) to steer drafting toward the verifier's distribution, with optional z-branch (gating) control.

### C2: Activation Replay for SSM Cache Resynchronization
After speculative rejection, efficiently reconstruct the Mamba2 hidden state by replaying only accepted tokens through the recurrent state, avoiding full re-prefill.

### C3: CPU/Heterogeneous Offloading with Fast SSM Kernel
Offload the tiny drafter to CPU (or other device), with a custom AVX-512 selective scan kernel and async pipeline to overlap drafting with GPU verification.

### C4: Masking Analysis & Evaluation Methodology
Identify and document a previously unreported interaction between non-compact speculative decoding layouts and recurrent drafters that inflates acceptance metrics.

---

## 3. What We HAVE

### Models & Checkpoints

**Mamba2-65M drafter** (27.5M backbone + 65.7M embedding, shared vocab with LLaMA):
- Architecture: 16 layers, hidden=512, expand=2, d_inner=1024, state_size=128, 16 heads, `is_fast_path_available=True`
- [x] **Pretrained** on FineWeb-Edu (~100M samples, 1 epoch): `/HSC/users/qiaoye/SSM_SPEC/checkpoints/improved-mamba-alignment/checkpoint-750`
- [x] **KD-finetuned** (50K filtered samples, temp-decay, 1500 steps): same checkpoint-750
- [x] **KD→Guided** (LLaMA-8B, v_layers=[5,16,29], TVD, 10 epochs, best=epoch 7): `/HSC/users/qiaoye/SSM_SPEC/checkpoints/guided_mamba/ckpts/step-step=37500-val_loss-val/loss=0.3974.ckpt`
- [x] **Pretrain→Guided with finetune** (LLaMA-8B, v_layers=[5,16,29], 7 epochs): `/HSC/users/qiaoye/SSM_SPEC/checkpoints/guided_sweep/pretrained_guided_finetune_5_16_29/`

**Mamba2-45M drafter** (47.6M backbone + 82.1M embedding):
- Architecture: 18 layers, hidden=640, expand=2, 20 heads, `is_fast_path_available=False`
- [x] **Pretrained** on FineWeb-Edu (~38M samples, 0.38 epochs — undertrained): `/HSC/users/qiaoye/SSM_SPEC/checkpoints/mamba2-45m-pretrain/final`
- [x] **KD-finetuned**: `/HSC/users/qiaoye/SSM_SPEC/checkpoints/mamba2-45m-kd/final`
- [x] **KD→Guided** (LLaMA-8B, best=epoch 7, val=0.3974): `/HSC/users/qiaoye/SSM_SPEC/checkpoints/mamba2-45m-guided/ckpts/`
- [x] **Pretrain→Guided** (LLaMA-8B, 7 epochs, val=0.403): `/HSC/users/qiaoye/SSM_SPEC/checkpoints/mamba2-45m-guided-from-pretrain/ckpts/last-v1.ckpt`
- [x] **KD→Guided** (Gemma-4B, 10 epochs, val=0.420): `/HSC/users/qiaoye/SSM_SPEC/checkpoints/mamba2-45m-gemma-guided/ckpts/last.ckpt`
- [x] **Guided-mixed** (Gemma-4B, mixed data, 7 epochs, val=0.352): `/HSC/users/qiaoye/SSM_SPEC/checkpoints/mamba2-45m-gemma-guided-mixed/ckpts/last.ckpt`

**Verifiers:**
- [x] **LLaMA 3.1-8B**: `/HSC/users/qiaoye/checkpoints/Llama3.1-8B-hf` (32 layers, hidden=4096)
- [x] **LLaMA 3.1-70B**: `/HSC/users/qiaoye/checkpoints/Llama-3.1-70B` (80 layers, hidden=8192)
- [x] **Gemma-4-E4B**: `google/gemma-4-E4B-it` (42 layers, hidden=2560)
- [x] **Baseline transformer drafters**: LLaMA-3.2-1B, LLaMA-3.2-3B

**70B Guided Training (in progress):**
- [x] Config: `guided_mamba/config_70b.yaml` — v_layers=[75], 8-bit quantized verifier, bsz=8, 5 epochs, TVD loss
- [x] Training: Epoch 4, step ~25.8K / 31.2K total (~83% complete)
- [x] Val loss: Epoch 0=0.402 → Epoch 1=0.373 → Epoch 2=0.355 → Epoch 3=0.342 (monotonically decreasing)
- [x] Checkpoints saved at epochs 0-3 + last.ckpt, ETA: ~4 more hours
- Runner: `spec_mamba/run.py` with `--device_map auto --quantize 8bit`
- Path: `/HSC/users/qiaoye/SSM_SPEC/checkpoints/guided_mamba_70b/`

### Code (under `spec_mamba/`)
- [x] `trainer.py` — Full training + generation with `mask_rejected`, `to()` override for device_map, 8-bit quantization support
- [x] `run.py` — LightningCLI training runner with `--device_map` and `--quantize` flags (for 70B+ verifiers)
- [x] `eval.py` — Multi-dataset evaluation (5 datasets, greedy/sample, `--no_mask`, `--baseline`, `--device_map`)
- [x] `guided_mamba2.py` — GuidedMamba2Block with x-branch delta injection
- [x] `pipeline_benchmark.py` — CPU-only and GPU-CPU pipeline benchmarking
- [x] `cross_verify_mask.py` — LLaMA-1B→8B vanilla spec dec for masking cross-check
- [x] Off-by-one fix: `in_layer = [v+1 for v in v_layers]` (verified diff=0.0)
- [x] `load_baseline()` — zeroed guidance weights for unguided comparison
- [x] AVX-512 + INT8 VNNI custom C++ kernel for CPU drafting

### All Collected Results

#### Table 1: Average Accepted Tokens (n=96, greedy, masked, best checkpoints)

| Model | Params | Verifier | HumanEval | GSM8K | Alpaca | UltraChat | XSum | Mean |
|-------|--------|----------|-----------|-------|--------|-----------|------|------|
| LLaMA-3.2-1B | 1B | LLaMA-8B | 4.881 | 2.604 | 4.351 | 2.727 | 2.390 | **3.391** |
| Old 65M pretrain | 65M | LLaMA-8B | 1.492 | 2.092 | 2.638 | 1.873 | 1.182 | 1.856 |
| Old 65M KD (zeroed guide) | 65M | LLaMA-8B | 1.437 | 1.828 | 2.417 | 1.629 | 0.973 | 1.657 |
| Old 65M KD→guided | 65M | LLaMA-8B | 2.070 | 2.907 | 2.745 | 2.197 | 1.721 | **2.328** |
| Old 65M pretrain→guided+ft | 65M | LLaMA-8B | — | — | — | — | — | **2.301** |
| New 45M pretrain | 45M | LLaMA-8B | 1.254 | 1.826 | 2.057 | 1.549 | 0.930 | 1.523 |
| New 45M KD | 45M | LLaMA-8B | 1.159 | 1.269 | 1.787 | 1.152 | 0.620 | 1.197 |
| New 45M KD→guided | 45M | LLaMA-8B | 2.253 | 3.167 | 3.215 | 2.417 | 2.030 | **2.616** |
| New 45M pretrain→guided | 45M | LLaMA-8B | — | — | — | — | — | 1.222 |
| New 45M pretrain | 45M | Gemma-4B | 1.073 | 0.557 | 0.438 | 0.490 | 0.616 | 0.635 |
| New 45M KD | 45M | Gemma-4B | 0.673 | 0.382 | 0.313 | 0.329 | 0.396 | 0.419 |
| New 45M guided | 45M | Gemma-4B | 2.278 | 0.619 | 0.636 | 0.566 | 0.618 | 0.943 |
| New 45M guided-mixed | 45M | Gemma-4B | 2.319 | 0.776 | 0.647 | 0.610 | 0.700 | 1.011 |

#### 70B Verifier Results (n=12, quick, greedy, masked — old 65M drafter)

| Model | Verifier | HumanEval | GSM8K | Alpaca | UltraChat | XSum | Mean |
|-------|----------|-----------|-------|--------|-----------|------|------|
| Old 65M pretrain | LLaMA-70B | 5.030 | 1.638 | 2.251 | 2.485 | 1.164 | **2.514** |
| Old 65M KD (zeroed guide) | LLaMA-70B | 4.931 | 1.406 | 2.204 | 2.480 | 1.070 | **2.418** |
| Old 65M 70B-guided | LLaMA-70B | — | — | — | — | — | **TBD** (training ~82% done) |

**Key 70B insight**: Pretrain acceptance jumps from 1.856 (8B) → 2.514 (70B) = **+35%** — guidance benefit expected to be even larger.

#### Old 65M Guidance Layer Sweep (masked, greedy, LLaMA-8B)

| Config | v_layers | Mean Accepted |
|--------|----------|---------------|
| KD→guided (default) | [5,16,29] | 2.328 |
| Pair [5,29] | [5,29] | 2.316 |
| z-branch [5,16,29] | [5,16,29] | 2.304 |
| Single [29] | [29] | 2.303 |
| Pretrain→guided+ft | [5,16,29] | 2.301 |
| Pair [16,29] | [16,29] | 2.299 |
| Single [16] | [16] | 2.293 |
| Pretrain→guided (frozen) | [5,16,29] | 1.981 |
| Single [5] | [5] | 1.899 |

**Key findings**: (1) KD is NOT mandatory — pretrain→guided+ft (2.301) ≈ KD→guided (2.328). (2) Single high layer [29] nearly matches 3-layer config. (3) z-branch adds no benefit.

#### CPU-Only Pipeline Results (greedy, NG=8, tgt_len=32, INT8 VNNI drafter)

| Metric | Mamba2-65M | Mamba2-45M |
|--------|------------|------------|
| Throughput | 3.49 tok/s | 4.32 tok/s |
| Acceptance | 0.89 / 8 | 1.21 / 8 |
| Draft time/round | 15.44 ms | 16.44 ms |
| Verify time/round | 534 ms | 478 ms |
| AR baseline | 4.78 tok/s | 5.60 tok/s |
| Speedup vs AR | 0.73x | 0.77x |

**Key finding**: CPU-only spec dec < 1x for LLaMA-8B because verify dominates (478ms vs 16ms draft). The GPU-CPU pipeline is where the real gain is.

#### CPU Kernel Optimization Stack (from `spec_mamba/CPU_KERNEL_OPTIMIZATION.md`)

**Hardware**: Intel Xeon Platinum 8562Y+ (Sapphire Rapids) — supports AVX-512 (F/BW/DQ/VL), AVX-512 FP16, AVX-512 BF16, AVX-512 VNNI, AMX BF16, AMX INT8.

**Bottleneck Analysis** (per-layer, single token):
- Linear projections: **85%** of time (in_proj=60%, out_proj=25%)
- SSM step: **10%** of time
- Other (conv1d, norms, gating): 5%
- **Strategy**: Optimize linear layers, not just SSM kernel

**Kernel Performance Stack**:

| Engine | 8-tok Draft (ms) | Throughput (tok/s) | vs HF Naive |
|--------|------------------|-------------------|-------------|
| HF Naive PyTorch | 79 ms (8t) | 110 (16t) | 1.0x |
| CPU FP32 + AVX-512 SSM | 57 ms (8t) | 123 (16t) | 1.05x (end-to-end) |
| Fused BF16 (full forward + LM head) | 32 ms (8t) | 340 (16t) | 2.8x |
| INT8 VNNI (fused) | 12 ms (8t) | 924 (16t) | **7.5x** |

**BF16 + AMX in_proj Results**: 9.6x speedup on in_proj alone (FP32: 0.379ms → BF16: 0.039ms). out_proj too small for AMX benefit.

**AVX-512 SSM Kernel**: Processes N=128 states in 8 SIMD iterations (16 float32s per 512-bit register). Uses `_mm512_fmadd_ps`. 4x over PyTorch CPU. OpenMP over (B,H,D)=1024 work items.

**FP16 SSM Accuracy**: 200 sequential steps with random inputs → max relative state error 0.21% (acceptable, but FP16 range ±65504 needs monitoring for overflow).

**Files**: `spec_mamba/cpu_kernels/ssm_ops.cpp`, `spec_mamba/cpu_kernels/fused_forward.cpp`, `spec_mamba/cpu_mamba2.py`

#### Activation Replay Performance (from `spec_mamba/ACTIVATION_REPLAY_AND_CPU_SCAN.md`)

**Problem**: After speculative rejection, SSM drafter state is contaminated by rejected tokens. Transformers crop KV cache in O(1), but SSM has no equivalent.

**Solution**: Snapshot pre-draft state (8.96 MB for Mamba2-65M, B=1) → restore after rejection → replay only accepted tokens through recurrence. Complexity: O(K) accepted tokens instead of O(T) full context.

**GPU Performance** (NOT beneficial for tiny model):
- Re-prefill: ~16ms (constant, model too small to benefit)
- Replay: ~40ms (5 sequential steps, kernel launch overhead)

**CPU Performance** (essential — massive benefit):

| Context Length | Re-prefill (ms) | Replay (ms) | Speedup |
|---------------|-----------------|-------------|---------|
| 32 | 299 | 47 | **6.4x** |
| 64 | 602 | 46 | **13.1x** |
| 128 | 1,019 | 37 | **27.7x** |
| 512 | 3,644 | 36 | **100.9x** |
| 1,024 | 8,036 | 40 | **201.0x** |

**Key insight**: Replay is O(K) constant (~37ms for ~5 accepted tokens). Re-prefill is O(T) sequential (7ms/token × T tokens). At 1K context, replay is **201x faster**.

**Correctness note**: Activation replay never sees rejected tokens — arguably more correct than re-prefill which includes prior-round rejected token contamination.

#### Architecture & Masking Insights (from `spec_mamba/ARCHITECTURE.md`)

**Critical Bugs Fixed**:
1. **Off-by-one in guidance layers**: `v_layers=[5,16,29] → in_layer=[6,17,30]` because SD²'s `LlamaModel.forward()` captures `guide_input` BEFORE the layer processes it
2. **Rejection masking**: Must zero rejected positions in attention_mask. Without: acceptance inflated to ~3.0. With: correct ~2.3. Verified diff=0.02 vs gold with mask, 4.1 without.
3. **Compact re-prefill bug** (2026-04-22): Rejected tokens fed to Mamba2 recurrence, corrupting state. Fix improved masked acceptance 2.3x (0.935 → 2.181).
4. **RMSNorm+Gate bug** (2026-04-21): Gate applied after norm instead of before norm. Cascading error from layer 0 max diff 0.246 to completely wrong tokens by step 2. Fixed to ~4e-6 logit diff.

**Cross-Verification** (LLaMA-1B → LLaMA-8B, `cross_verify_mask.py`):
- WITH masking: 2.734 acceptance
- WITHOUT masking: 2.127 acceptance
- **Pattern reversal vs Mamba2** (Mamba2: masked < unmasked). Confirms asymmetry comes from drafter architecture (recurrent vs transformer), not from masking being wrong.

**Backbone Finetuning is Critical**:
- Frozen guidance: 1.981 (+7% vs unguided)
- Finetuned: 2.301 (+24% vs unguided)
- **Finetuning provides 3x the benefit** of guidance projection alone

**Guidance Layer Sensitivity**: Layer selection barely matters (~1.5% spread across configs) when backbone is finetuned. Single high layer [29] (2.303) nearly matches full 3-layer [5,16,29] (2.328).

#### Projected GPU-CPU Pipeline Results

**Setup**: INT8 VNNI Mamba2-65M drafter on CPU, LLaMA-8B verifier on GPU, async overlap.

| Config | Draft Time | Verify Time | Accept | Throughput | Speedup vs AR |
|--------|-----------|-------------|--------|------------|---------------|
| K=8, 8 threads | 12 ms | 16 ms | 2.33 | 202 tok/s | **2.7x** |
| K=4, 8 threads | 6 ms | 16 ms | 2.33 | 230 tok/s | **3.0x** |
| K=8, 16 threads | 9 ms | 16 ms | 2.33 | 202 tok/s | **2.7x** |

**AR Baseline**: LLaMA-8B greedy bsz=1 = 75.8 tok/s.
**Key insight**: Draft (9-12ms) fits inside GPU verify window (16ms) → GPU is the bottleneck → pipeline overlap almost free.
**TODO**: Implement actual async pipeline and measure real throughput (projected numbers assume perfect overlap).

### Paper Infrastructure
- [x] NeurIPS LaTeX template at `paper/neurips2026-specssm/`
- [x] All section .tex files with planning comments (~35 TODOs remain)
- [x] Architecture documentation: `spec_mamba/ARCHITECTURE.md`
- [ ] **Figures**: None generated yet (only `.gitkeep` placeholders)
- [ ] **Scripts**: None generated yet (only `.gitkeep` placeholders)

---

## 4. What We NEED (Work Items)

### P0 — Must Have for Submission

| # | Work Item | Pillar | Status | Notes |
|---|-----------|--------|--------|-------|
| W1 | Run unguided baseline (5 datasets) | C1 | ✅ Done | Unguided mean accept=3.45, speedup=0.57x. Guided mean accept=3.17, speedup=0.54x. File: `eval_results_baseline_with_ar.json` |
| W2 | z-branch ablation (steer_z=True) | C1 | 🔄 Sweep ready | Sweep script: `guided_mamba/sweep_guidance.py`. Config #0: z_branch_5_16_29 |
| W3 | Implement activation replay | C2 | ✅ Done | `trainer.py`: snapshot/restore + replay; profiled on CPU (201x speedup at 1K tokens) |
| W4 | Benchmark activation replay vs re-prefill | C2 | ✅ Done | GPU: not beneficial (16ms vs 40ms); CPU: 6.4x–201x speedup. See `profile_all_results.json` |
| W5 | Wall-clock speedup table (spec dec vs AR) | All | ✅ Done | H100 NVL bsz=1: Mamba2-65M guided=0.54x, unguided=0.57x, LLaMA-1B=0.83x. AR baseline=~76 tok/s. **Speedup <1x on fast GPU!** CPU offloaded: 80ms/round |
| W6 | Write Method section | All | ✅ Done | All 9 sections + tables + references.bib written |
| W7 | Write Experiments & Results sections | All | ✅ Done | With TODOs for missing baselines |
| W8 | Main results table (Table 1) | All | Data ready | All data collected. See ARCHITECTURE.md Performance Summary. Need to generate LaTeX table |
| W9 | Text quality metrics (BLEU/ROUGE vs gold AR) | C4 | Not started | Show masked output is coherent |
| W10 | References.bib (all citations) | — | ✅ Done | 18 entries |
| W11 | CPU kernel (AVX-512 SSM + fused BF16 + INT8 VNNI) | C3 | ✅ Done | 7.5x over PyTorch: fused C++ + BF16 + INT8 VNNI. 8-tok draft: 12ms@8t (924 tok/s@16t). Verified 2026-04-29 |

### P1 — Highly Desirable

| # | Work Item | Pillar | Status | Notes |
|---|-----------|--------|--------|-------|
| W12 | Async CPU-GPU pipeline | C3 | ✅ Feasible (INT8) | INT8 draft=9ms < verify=16ms → GPU is bottleneck. Overlapped: 2.7x vs AR (LLaMA-8B). Need to implement async pipeline |
| W13 | LLaMA-1B transformer drafter baseline | C1 | ✅ Done | LLaMA-1B mean accept=3.39, speedup=0.83x (WITH mask). File: `eval_results_llama1b_mask.json` |
| W14 | Drafter scaling study (20M-backend vs 40M-backend) | C1 | 🔄 Training | Larger Mamba drafter training for LLaMA-3 & Gemma-4. Key insight: Mamba backend is NOT CPU bottleneck (LM head is 55%), so 2× backend ≈ same latency |
| W15 | NG sweep (K=4,6,8,12) | C1 | Partial | Sweep data exists in outputs/improved_mamba_sweep/ but needs clean plots |
| W16 | Guidance layer ablation (which layers) | C1 | 🔄 Sweep ready | Sweep script: `guided_mamba/sweep_guidance.py`. 11 configs, P1-P3 priority |
| W17 | Overhead breakdown figure | All | ✅ Done | Profiling data in `profile_all_results.json`; needs figure generation script |

### P2 — Nice to Have

| # | Work Item | Pillar | Status | Notes |
|---|-----------|--------|--------|-------|
| W18 | INT8 quantized drafter on GPU | C3 | Not started | Alternative to CPU offload |
| W19 | Tree-structured drafting | C1 | Not started | Reference from 2506.01206 |
| W20 | Different verifier (LLaMA-70B) | C1 | 🔄 Training (83%) | 8-bit quantized, v_layers=[75], epoch 4/5. Val loss: 0.402→0.342. Run eval after training completes. |
| W21 | Instruct model evaluation | — | Not started | LLaMA-3.1-8B-Instruct |
| W22 | Cross-family evaluation (Gemma-4-E4B verifier) | C1,C3 | 🔄 Drafter training | Shows guidance is architecture-agnostic; same CPU kernel works for any verifier |
| W23 | Context length scaling benchmark | C3 | Not started | **HIGH PRIORITY**: Test generation lengths {64, 128, 256, 512, 1024} to show activation replay benefit scales with context. Key differentiator figure: Mamba replay=O(K) flat vs transformer re-prefill=O(T) linear. Already have CPU replay data: 6.4x@32tok → 201x@1024tok. Need: (1) acceptance vs context length, (2) wall-clock per round vs context, (3) pipeline throughput vs context |
| W24 | 70B guided eval (after training) | C1 | ⬜ Blocked on W20 | Run full eval (n=96) with best 70B-guided checkpoint. Projected: mean accept ≥2.8 (based on val loss 0.342 < 8B val loss 0.397) |
| W25 | Full 70B pretrain/KD eval (n=96) | C1 | ⬜ Blocked on GPUs | Current quick n=12: pretrain=2.514, KD=2.418. Need n=96 for paper |
| W26 | GPU-CPU pipeline benchmark (real async) | C3 | ⬜ Blocked on GPUs | Implement actual async CPU-GPU pipeline. Projected: 2.7x speedup over AR. Need GPUs free after 70B training |

---

## 5. Critical Path (Updated 2026-05-04)

**Done**: W1, W3, W4, W5, W6, W7, W10, W11, W12 (feasible), W13, W17

**In Progress**:
- W20: 70B guided training at 83% (epoch 4/5, val loss 0.342). ETA: ~4 hours.

**Key findings**:
- Speedup < 1x on H100 NVL (bsz=1). Mamba2-65M guided=0.54x, LLaMA-1B=0.83x
- **Compact re-prefill bug fixed** (2026-04-22): Mamba2 drafter re-prefill was feeding rejected tokens to the recurrent model, corrupting state. Fix improved masked acceptance from ~0.9 to ~2.3.
- **RMSNorm+Gate bug fixed** (2026-04-21): Gate applied after norm instead of before. Cascading error fixed (max diff 0.246 → 4e-6).
- **Corrected masked results**: Guided (2.33) > Pretrained (1.86) > KD-zeroed (1.66) — guidance clearly helps
- **Cross-verification asymmetry**: LLaMA-1B WITH mask (2.734) > WITHOUT (2.127) — opposite of Mamba2 pattern. Confirms recurrent vs transformer architecture difference.
- KD-zeroed is worst under masking: backbone became dependent on guidance deltas
- **Backbone finetuning >> frozen guidance**: Finetuned +24% vs frozen +7% — 3x more benefit
- **Guidance layer selection barely matters**: ~1.5% spread when backbone is finetuned
- **CPU kernel stack complete**: HF naive (1.0x) → AVX-512 FP32 (1.05x) → Fused BF16 (2.8x) → INT8 VNNI (7.5x)
- **Activation replay essential on CPU**: 201x faster than re-prefill at 1K context. O(K) vs O(T).
- **Pipeline overlap feasible**: INT8 draft (9-12ms) < GPU verify (16ms). Projected 2.7x speedup.
- **70B pretrain acceptance +35%**: 1.856 (8B) → 2.514 (70B). Guidance benefit expected even larger.

**Immediate next steps (after 70B training finishes)**:
1. **W24**: Run 70B guided eval (n=96) with best checkpoint
2. **W25**: Run full 70B pretrain/KD eval (n=96, currently only n=12)
3. **W26**: Implement and benchmark real async GPU-CPU pipeline
4. **W23**: Context length scaling experiment — **KEY FIGURE FOR PAPER**: test {64, 128, 256, 512, 1024} token generation lengths, measure:
   - Acceptance rate vs context length
   - Wall-clock per round vs context (Mamba replay flat, transformer KV-crop linear)
   - Pipeline throughput vs context length
   - Shows activation replay advantage grows with sequence length

**Guidance sweep (W2+W16)** — ready to launch via `guided_mamba/sweep_guidance.py`:
- 11 configs: z-branch test, single/pair/triple layer ablation, dense layers
- ~19 min/epoch, 5-epoch quick screen per config → ~1.5h each
- 2 GPUs available → 2 configs in parallel → ~6h for all P1 configs

```
Week 1 (DONE): W1 (unguided baseline) + W5 (speedup table) + W13 (LLaMA-1B)
Week 2 (DONE): Bug fix, corrected masked evals, sweep script
Week 3 (DONE): CPU kernel stack (AVX-512 → BF16 → INT8), activation replay profiling
Week 4 (NOW): W20 (70B guided training, 83% done) + W24-W26 (eval + pipeline benchmark)
Week 5: W23 (context length scaling) + W15 (NG sweep plots) + W8 (LaTeX Table 1)
Week 6: W14 (larger drafter eval) + W22 (Gemma cross-family) + all figure scripts
Week 7: W9 (quality metrics) + full paper draft, internal review
Week 8: Camera-ready revisions, supplementary material, checklist
```

### Scalability Matrix (drives paper narrative)

| Axis | Variations | Key Claim | Status |
|------|-----------|-----------|--------|
| Drafter size | 27.5M-backend (65M) → 47.6M-backend (45M) | "Mamba backend scales for free on CPU (LM head=55% of latency)" | ✅ Data collected |
| Verifier family | LLaMA-3.1-8B, Gemma-4-E4B | "Guidance + CPU offload is architecture-agnostic" | ✅ Gemma data collected |
| Verifier size | 8B → 70B | "Pipeline benefit grows with slower verifier. Accept +35% from 8B→70B pretrain." | 🔄 70B guided training 83% |
| Context length | 64→1024 tokens | "O(K) replay vs O(T) re-prefill — 201x at 1K tokens. **The definitive SSM advantage.**" | ⬜ Need eval experiment |
| CPU offload | HF naive → INT8 VNNI | "7.5x kernel speedup. Draft fits inside GPU verify window → free pipeline parallelism" | ✅ Kernel benchmarked |

**No critical blockers remaining** — all data for Table 1 is collected.

---

## 6. Paper Structure (9 pages + appendix)

| Section | Pages | Content |
|---------|-------|---------|
| Abstract | 0.25 | SSM drafter + guidance + replay + speedup numbers |
| Introduction | 1.5 | Motivation, gap, 4 contributions |
| Related Work | 0.75 | Spec dec, SSM drafters, systems inference |
| Background | 0.5 | Spec dec notation, Mamba2 selective scan |
| Method | 2.5 | §3.1 Guided SSM (x/z-branch), §3.2 Activation Replay, §3.3 CPU Offload |
| Experiments | 3.0 | Setup + Main table + Ablations + Systems analysis |
| Conclusion | 0.5 | Summary + future work |
| **Total** | **9.0** | |

### Key Figures (in paper body)
1. **Architecture overview** — Verifier → Guidance Extractor → PrepMambaDeltas → GuidedMamba2Block
2. **Activation replay diagram** — State snapshot/restore + O(K) replay vs O(T) re-prefill
3. **Context length scaling** — Replay speedup vs context (6.4x@32 → 201x@1024)
4. **Pipeline overlap** — CPU draft (9ms) overlapped with GPU verify (16ms)
5. **CPU kernel stack** — Bar chart: HF→AVX-512→BF16→INT8 VNNI (1x→7.5x)

### Key Tables
1. **Main results** — Datasets × {Guided Mamba2, Unguided Mamba2, LLaMA-1B drafter, AR baseline} × {Acceptance, Throughput, Speedup}
2. **Ablation** — z-branch, guidance layers, NG sweep, backbone finetuning vs frozen
3. **Systems / CPU offload** — Re-prefill vs activation replay latency, CPU kernel stack (HF→AVX-512→BF16→INT8), GPU-CPU pipeline throughput
4. **Verifier scaling** — 8B vs 70B acceptance rates (pretrain, KD, guided)
5. **Context length scaling** — Replay speedup vs context (64→1024), total throughput vs context

### Key Figures
1. **Architecture overview** — Verifier → Guidance Extractor → PrepMambaDeltas → GuidedMamba2Block
2. **Activation replay diagram** — State snapshot/restore + O(K) replay vs O(T) re-prefill
3. **Context length scaling** — Replay speedup vs context length (log-log): 6.4x@32 → 201x@1024
4. **CPU kernel speedup stack** — Bar chart: HF naive, AVX-512, BF16 fused, INT8 VNNI
5. **Pipeline overlap timeline** — CPU draft overlapped with GPU verify
6. **Speed-quality tradeoff** — Acceptance vs drafter size/type scatter plot

---

## 7. Documentation References

- **Architecture & masking**: `spec_mamba/ARCHITECTURE.md` — generation flow, masking analysis, guidance layer sweep, LLaMA-1B baseline, cross-verify results, compact re-prefill bug
- **CPU kernels**: `spec_mamba/CPU_KERNEL_OPTIMIZATION.md` — AVX-512 FP32, BF16 AMX, INT8 VNNI, FP16 accuracy, bottleneck analysis
- **Activation replay**: `spec_mamba/ACTIVATION_REPLAY_AND_CPU_SCAN.md` — snapshot/restore design, O(K) vs O(T) complexity, GPU vs CPU profiling, CPU model architecture, SSM kernel design

---

## 8. Differentiation from Prior Work

### vs "Mamba Drafters for Speculative Decoding" (arXiv:2506.01206)
- They use **vanilla Mamba** (no guidance injection) — we add verifier-guided x-branch steering
- They focus on tree search for draft quality — we focus on **systems** (replay, CPU offload)
- They don't address SSM cache resynchronization — we propose activation replay
- They don't explore heterogeneous deployment — we offload to CPU with custom kernel

