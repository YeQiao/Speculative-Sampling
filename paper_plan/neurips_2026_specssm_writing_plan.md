# SpecSSM NeurIPS 2026 Writing Plan

## 1) Submission Constraints (lock these first)
- Venue: NeurIPS 2026 main track format (`neurips_2026.sty`)
- Main paper limit: 9 pages including figures/tables
- Extra pages allowed: references, checklist, acknowledgments, appendix/supplement
- Mandatory checklist: include `checklist.tex` at submission (missing checklist can cause desk rejection)
- Blind review: remove identifying info, use third-person self-citation style

Template source in repo:
- `Formatting_Instructions_For_NeurIPS_2026/neurips_2026.tex`
- `Formatting_Instructions_For_NeurIPS_2026/checklist.tex`

## 2) Paper Workspace Structure (recommended)
Create a dedicated paper workspace to avoid mixing with code:

```text
paper/
  neurips2026-specssm/
    main.tex
    references.bib
    Makefile
    sections/
      abstract.tex
      introduction.tex
      related_work.tex
      background.tex
      method.tex
      experiments_setup.tex
      results_main.tex
      results_analysis.tex
      limitations.tex
      conclusion.tex
    figures/
      main/
      appendix/
    tables/
      main/
      appendix/
    scripts/
      figures/
      tables/
      stats/
    appendix/
      appendix.tex
      reproducibility.tex
      additional_results.tex
      implementation_details.tex
      prompts_and_datasets.tex
    checklist/
      neurips_checklist.tex
```

## 3) What Goes in Each Subdirectory

### sections/
Goal: single source of truth for narrative.

- `abstract.tex`
  - 4 sentence structure: problem, gap, method, quantitative result.
- `introduction.tex`
  - Motivation, problem framing, contribution bullets, paper roadmap.
- `related_work.tex`
  - Group by: speculative decoding, draft model design, SSM/Mamba acceleration, edge inference.
- `background.tex`
  - Minimal background needed to read your method quickly.
- `method.tex`
  - Core method and algorithm details for SpecSSM.
- `experiments_setup.tex`
  - Models, checkpoints, hardware, datasets, metrics, reproducibility details.
- `results_main.tex`
  - Primary results that support claims (speedup/acceptance/quality).
- `results_analysis.tex`
  - Ablations and diagnostic analyses (e.g., masking behavior, overhead decomposition).
- `limitations.tex`
  - Explicit limits, failure modes, and scope.
- `conclusion.tex`
  - Summary + concrete future work.

### figures/main/
Goal: 4-6 high-value figures only (page-budget aware).

Planned figure set:
1. **Method overview** (pipeline diagram)
2. **Throughput vs baseline** (bar chart across platforms/configs)
3. **Acceptance rate vs checkpoint/lookahead** (line plot)
4. **Quality vs speed tradeoff** (scatter/Pareto)
5. **Overhead breakdown** (stacked bars: drafting, verification, transfer, sampling)
6. **Optional:** long-context scaling curve (if space permits)

### figures/appendix/
- Additional dataset plots
- Per-prompt distributions
- Sensitivity curves (`K`, temperature, batch size)
- Qualitative examples (if needed)

### tables/main/
Goal: keep main claims auditable in compact form.

Planned table set:
1. **Main benchmark table** (speedup, throughput, acceptance, quality metric)
2. **Ablation table** (mask/no-mask, checkpoint variants, lookahead choices)
3. **Compute/resource table** (GPU/CPU type, memory, runtime per setting)

### tables/appendix/
- Full benchmark matrix by dataset + model + prompt category
- Hyperparameter sweeps and confidence interval details

### scripts/figures/
- One script per figure, deterministic, reads JSON results only
- Save vector PDFs for LaTeX include
- Naming convention: `fig_<topic>.py` -> `figures/main/<topic>.pdf`

### scripts/tables/
- Scripts that convert JSON experiment outputs to LaTeX tables
- Naming convention: `tab_<topic>.py` -> `tables/main/<topic>.tex`

### scripts/stats/
- CI/error bars/bootstrap scripts
- Statistical significance checks for key claims

### appendix/
- `appendix.tex`: includes all appendix sections
- `reproducibility.tex`: exact commands, seeds, environment, checkpoint IDs
- `implementation_details.tex`: engineering specifics (cache/masking choices)
- `additional_results.tex`: full tables/plots omitted from main text
- `prompts_and_datasets.tex`: dataset split and prompt templates

### checklist/
- `neurips_checklist.tex`: copy/adapt official checklist content and fill with section references

## 4) Section-by-Section Content Outline (target for 9 pages)

### 1. Introduction (~1.0 page)
- Why speculative decoding still has deployment pain points.
- Why tiny SSM drafter is compelling for compute-constrained settings.
- Contributions (3-4 bullets): method, implementation fix/insight, empirical gains, reproducibility artifacts.

### 2. Related Work (~0.8 page)
- Prior speculative decoding methods and their bottlenecks.
- Draft model alternatives (small LMs, distilled models, SSM drafters).
- Distinguish your practical contribution and empirical regime.

### 3. Method (~1.7 pages)
- Formal setup and notation.
- SpecSSM decoding procedure.
- Acceptance/masking and verifier interaction.
- Complexity and expected speedup intuition.

### 4. Experimental Setup (~1.0 page)
- Verifier model(s), drafter checkpoints, tokenization assumptions.
- Hardware and software stack.
- Datasets/prompts and evaluation protocol.
- Metrics: throughput, avg accepted tokens, quality proxy, compute cost.

### 5. Main Results (~1.6 pages)
- Core speedup table and key figure(s).
- Main quantitative claims with confidence intervals.

### 6. Analysis & Ablation (~1.3 pages)
- Checkpoint sweep and acceptance dynamics.
- Masking/off-by-one correctness impact.
- Overhead decomposition (draft/verify/transfer).

### 7. Limitations and Broader Impact (~0.6 page)
- Failure cases, generalization limits, evaluation scope constraints.
- Responsible release and misuse considerations.

### 8. Conclusion (~0.3 page)
- Takeaways and 2-3 concrete future directions.

## 5) Figure and Table Mapping to Existing Repo Artifacts
Use these as data sources first:
- `outputs/improved_mamba_sweep/`
- `outputs/improved_mamba_analysis/`
- `outputs/mamba_aligned_3b_analysis/`
- `outputs/temperature_0/`
- `outputs/temperature_05/`
- `spec_mamba/eval_results.json`
- `spec_mamba/eval_results_bsz1_greedy.json`
- `guided_mamba/eval_results.json`
- `guided_mamba/eval_results_bsz1.json`

Action: normalize all JSON outputs into one canonical schema before plotting.

## 6) Writing and Experiment Execution Timeline (from now to submission)

### Phase A (Week 1-2): Freeze claims and data schema
- Decide top 3 claims to defend in main paper.
- Standardize metric definitions and confidence intervals.
- Build result aggregation scripts.

### Phase B (Week 3-4): Produce main figures/tables
- Generate all main paper figures and tables from scripts.
- Validate consistency against raw logs.
- Select final 4-6 figures and 2-3 tables for page budget.

### Phase C (Week 5-6): Draft complete manuscript
- Draft sections in order: intro -> method -> setup -> results -> analysis -> related -> abstract.
- Keep each section aligned to one claim.

### Phase D (Week 7): Internal review and rebuttal-hardening
- Check claim-evidence traceability.
- Strengthen limitation discussion.
- Ensure ablation supports every design choice.

### Phase E (Week 8): NeurIPS compliance and polish
- Blindness pass, checklist completion, reference cleanup.
- Final page-budget trim and caption polish.
- Produce supplementary appendix package.

## 7) Claim-Evidence Traceability Checklist (must pass before submission)
For each main claim:
1. Claim appears in abstract and intro.
2. Claim has one primary table/figure in main paper.
3. Metric definition is explicit in setup.
4. Statistical variability is shown or justified.
5. Limitation scope is explicitly stated.

## 8. Immediate Next 10 Tasks (Updated 2026-04-21)
1. ~~Create `paper/neurips2026-specssm/` structure.~~ ✅
2. ~~Build canonical results from existing outputs.~~ ✅ (multiple JSON files exist)
3. ~~Draft `method.tex` with algorithm block and notation.~~ ✅
4. ~~Draft `experiments_setup.tex` with exact environment/checkpoint paths.~~ ✅
5. **[BLOCKING] Run unguided baseline: `eval.py --baseline --greedy_only --bsz 1 --no_mask`**
6. **[BLOCKING] Run masked eval with 96 samples: `eval.py --ckpt <guided> --greedy_only --bsz 1`**
7. **Generate Figure 1** (method overview / architecture diagram)
8. **Generate Table 1** (main benchmark results — needs W1 + AR baseline)
9. **Run LLaMA-1B full drafter eval** for 5 datasets (direct comparison row in Table 1)
10. **Generate Figure 2** (throughput vs AR baseline bar chart from sweep data)

## 9) Risk Register (Updated 2026-04-21)
- **Mamba-65M gives only ~2x speedup vs LLaMA-1B's ~3x**: Must frame story around memory efficiency + CPU offloading, not raw speedup
- **GPU draft latency**: Mamba-65M is 15.2ms/step vs LLaMA-1B at 8.2ms on GPU — SSM advantage only materializes on CPU (7.3ms)
- Main claims exceed what your strongest results support.
- Missing statistical significance/variance reporting (only 1 random seed).
- Incomplete reproducibility details for critical numbers.
- Too many figures in main paper causing weak narrative density.
- Checklist answers not tied to concrete sections.

## 10) Definition of Done for Submission
- 9-page main paper that is self-contained and claim-complete.
- Reproducible figure/table generation scripts in `scripts/`.
- Checklist fully answered with section references.
- Supplementary appendix with full experimental details.
- Blind and camera-ready modes both compile cleanly.
