# NeurIPS 2026 Paper Plan: SpecSSM

**Title (working):** SpecSSM: Efficient Speculative Decoding with Guided Tiny SSM Drafters  
**Venue:** NeurIPS 2026 (ML-Systems track)  
**Deadline:** TBD — check [NeurIPS 2026 CFP](https://neurips.cc/)

---

## 1. Core Story

We show that a **tiny 65M Mamba2 drafter**, guided by verifier hidden states and equipped with a cache-correct resynchronization method, can significantly accelerate LLM inference via speculative decoding — with optional CPU offloading to free GPU memory entirely.

### Why This Is a Systems Paper
1. **Non-trivial systems insight**: SSM drafters have unique cache properties (no KV cache → no rollback → new resync strategy needed)
2. **Concrete wall-clock gains**: Tiny drafter = fast drafting; guidance = higher acceptance; overlap = pipeline parallelism
3. **Heterogeneous deployment**: CPU drafter while GPU verifies — practical for memory-constrained settings
4. **Rigorous evaluation**: 5 datasets, masking analysis, architecture-dependent effects, scaling

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
- [x] **Pretrained Mamba2-65M** on FineWeb-Edu: `/HSC/users/qiaoye/SSM_SPEC/checkpoints/improved-mamba-alignment/checkpoint-750`
  - Architecture: 16 layers, hidden=512, expand=2, d_inner=1024, state_size=128, 16 heads
  - Multiple checkpoints: checkpoint-{250,500,750,1000,1250,1500}, best_model, final_model
- [x] **KD-finetuned guided checkpoint**: `/HSC/users/qiaoye/SSM_SPEC/checkpoints/guided_mamba/ckpts/last.ckpt`
  - NG=8, v_layers=[5,16,29], d_layers=all, loss_method=TVD, finetune_drafter=True, steer_z=False
- [x] **Verifier**: LLaMA 3.1-8B at `/HSC/users/qiaoye/checkpoints/Llama3.1-8B-hf`
- [x] **Baseline transformer drafters** available: LLaMA-3.2-1B, LLaMA-3.2-3B

### Code (under `spec_mamba/`)
- [x] `trainer.py` — Full training + generation with `mask_rejected` flag
- [x] `eval.py` — Multi-dataset evaluation (5 datasets, greedy/sample, `--no_mask`, `--baseline`)
- [x] `guided_mamba2.py` — GuidedMamba2Block with x-branch delta injection
- [x] `models/llama.py` — Custom LlamaForCausalLM (SD²-style `_update_causal_mask`)
- [x] `cross_verify_mask.py` — LLaMA-1B→8B vanilla spec dec for masking cross-check
- [x] Off-by-one fix: `in_layer = [v+1 for v in v_layers]` (verified diff=0.0)
- [x] `load_baseline()` — zeroed guidance weights for unguided comparison

### Results Already Collected
- [x] **Guided, no-mask, greedy, bsz=1** (96 samples per dataset):
  - ultrachat=2.913, humaneval=3.745, xsum=2.419, alpaca=3.707, gsm8k=3.072 (mean=3.171)
  - File: `spec_mamba/eval_results_no_mask_bsz1.json`
- [x] **Guided, no-mask, greedy, bsz=1** (48 samples, alternate run):
  - alpaca=3.88, humaneval=3.87, gsm8k=3.10, ultrachat=2.98, xsum=2.23
  - File: `spec_mamba/eval_results_bsz1_greedy.json`
- [x] **Masked eval (small)**: ultrachat only, 16 samples, accept=0.80 — statistically weak, needs re-run with 96 samples
- [x] **Masked vs gold AR comparison**: diff=0.02 (masked), diff=4.1 (unmasked)
- [x] **Cross-verification LLaMA-1B→8B**: masked=2.734, unmasked=2.127 (opposite direction!)
- [x] **Profiling data**: `outputs/mamba_profile/` (draft latency breakdown)
- [x] **Draft latency benchmarks**: LLaMA-1B=8.15ms/step, Mamba-65M=15.19ms/step on GPU
- [x] **Sweep data (improved_mamba)**: best=mamba-750-K3: 47.7 tok/s, accept=0.79 (AR=24.1 → 1.98x speedup)
- [x] **Sweep data (transformer baselines)**: best=LLaMA-1B-K3: 73.4 tok/s, accept=0.82 (AR=23.8 → 3.08x speedup)
- [x] **CPU profiling**: AVX-512 SSM kernel 4x speedup, CPU single-step 7.3ms, activation replay 201x at 1K tokens
- [x] **CPU model correctness**: Verified 2026-04-21, 16-step generation matches HF reference (max logit diff ~4e-6)

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
| W11 | CPU kernel (AVX-512 selective scan) | C3 | ✅ Done | 4x speedup over PyTorch CPU; JIT compiled; model verified correct 2026-04-21 |

### P1 — Highly Desirable

| # | Work Item | Pillar | Status | Notes |
|---|-----------|--------|--------|-------|
| W12 | Async CPU-GPU pipeline | C3 | ❌ Not feasible | CPU draft (64ms) is 4x slower than GPU verify (16ms). Overlap yields minimal benefit. Frame as memory savings instead |
| W13 | LLaMA-1B transformer drafter baseline | C1 | ✅ Done | LLaMA-1B mean accept=3.39, speedup=0.83x (WITH mask). File: `eval_results_llama1b_mask.json` |
| W14 | Drafter scaling study (65M vs 130M) | C1 | Need 130M model | May use HF checkpoint if available |
| W15 | NG sweep (K=4,6,8,12) | C1 | Partial | Sweep data exists in outputs/improved_mamba_sweep/ but needs clean plots |
| W16 | Guidance layer ablation (which layers) | C1 | 🔄 Sweep ready | Sweep script: `guided_mamba/sweep_guidance.py`. 11 configs, P1-P3 priority |
| W17 | Overhead breakdown figure | All | ✅ Done | Profiling data in `profile_all_results.json`; needs figure generation script |

### P2 — Nice to Have

| # | Work Item | Pillar | Status | Notes |
|---|-----------|--------|--------|-------|
| W18 | INT8 quantized drafter on GPU | C3 | Not started | Alternative to CPU offload |
| W19 | Tree-structured drafting | C1 | Not started | Reference from 2506.01206 |
| W20 | Different verifier (LLaMA-70B) | C1 | Have checkpoint | Much slower but higher acceptance |
| W21 | Instruct model evaluation | — | Not started | LLaMA-3.1-8B-Instruct |

---

## 5. Critical Path (Updated 2026-04-22)

**Done**: W1, W3, W4, W5, W6, W7, W10, W11, W12 (infeasible), W13, W17

**Key findings**:
- Speedup < 1x on H100 NVL (bsz=1). Mamba2-65M guided=0.54x, LLaMA-1B=0.83x
- **Compact re-prefill bug fixed** (2026-04-22): Mamba2 drafter re-prefill was feeding rejected tokens to the recurrent model, corrupting state. Fix improved masked acceptance from ~0.9 to ~2.3.
- **Corrected masked results**: Guided (2.33) > Pretrained (1.86) > KD-zeroed (1.66) — guidance clearly helps
- KD-zeroed is worst under masking: backbone became dependent on guidance deltas

**Guidance sweep (W2+W16)** — ready to launch via `guided_mamba/sweep_guidance.py`:
- 11 configs: z-branch test, single/pair/triple layer ablation, dense layers
- ~19 min/epoch, 5-epoch quick screen per config → ~1.5h each
- 2 GPUs available → 2 configs in parallel → ~6h for all P1 configs

```
Week 1 (DONE): W1 (unguided baseline) + W5 (speedup table) + W13 (LLaMA-1B)
Week 2 (DONE): Bug fix, corrected masked evals, sweep script
Week 3 (NOW): W2+W16 (z-branch + layer guidance sweep) + W8 (LaTeX Table 1) + W15 (NG sweep plots)
Week 4: W9 (quality metrics) + all figure scripts + polish
Week 5: Full paper draft, internal review, iterate
Week 6: Camera-ready revisions, supplementary material, checklist
```

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

### Key Figures
1. **Architecture overview** — Verifier → Guidance Extractor → PrepMambaDeltas → GuidedMamba2Block
2. **Activation replay diagram** — Show state rollback vs full re-prefill
3. **Speed-quality tradeoff** — Acceptance vs drafter size/type scatter plot
4. **Latency breakdown** — Stacked bar: draft time, verify time, overhead

### Key Tables
1. **Main results** — Datasets × {Guided Mamba2, Unguided Mamba2, LLaMA-1B drafter, AR baseline} × {Acceptance, Throughput, Speedup}
2. **Ablation** — z-branch, guidance layers, NG sweep
3. **Systems** — Re-prefill vs activation replay latency, CPU vs GPU draft latency

---

## 7. Differentiation from Prior Work

### vs "Mamba Drafters for Speculative Decoding" (arXiv:2506.01206)
- They use **vanilla Mamba** (no guidance injection) — we add verifier-guided x-branch steering
- They focus on tree search for draft quality — we focus on **systems** (replay, CPU offload)
- They don't address SSM cache resynchronization — we propose activation replay
- They don't explore heterogeneous deployment — we offload to CPU with custom kernel

### vs SD² (arXiv:2407.10722)
- SD² uses transformer drafter — we use SSM (O(1) memory, no KV cache)
- We adapt their guidance mechanism to SSM's x-branch/z-branch structure
- We identify architecture-dependent masking effects they didn't study

---

## 8. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Mamba2-65M acceptance stays ~1.0 masked | High | Emphasize wall-clock (tiny=fast, acceptance is just one factor); scaling study |
| CPU kernel too slow | Medium | Fall back to GPU INT8 quantized drafter; focus on memory savings |
| Reviewer: "incremental over 2506.01206" | High | Clearly differentiate: guidance, replay, CPU offload, masking analysis |
| Activation replay has correctness bugs | Medium | Verify output matches full re-prefill bit-for-bit |
| Not enough experiments by deadline | Medium | Prioritize P0 items; P1/P2 go to appendix or future work |
