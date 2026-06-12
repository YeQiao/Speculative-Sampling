# SpecSSM Project Memory

This directory consolidates the working knowledge accumulated during the SpecSSM
project so it can travel with the repository (e.g., when moving to another machine).

**Project / paper full title** (use EXACTLY):
"SpecSSM: Efficient Speculative Decoding with Guided Tiny SSM Drafters and CPU Offloading"

- Open-source repo: https://github.com/YeQiao/SpecSSM_Realease
  (note: the URL slug has a "Realease" typo, but the project NAME/title is the full title above)
- Copyright: "Copyright (c) 2026 Hewlett Packard Enterprise Development LP. Author: Ye Qiao."
  (work done at HPE research group)
- Git identity for release repo: name="Ye Qiao", email="yeq6@uci.edu"

## Contents

| File | What it covers |
|---|---|
| [environment.md](environment.md) | Conda env, Python interpreter, run commands, model/verifier paths |
| [checkpoints.md](checkpoints.md) | All drafter/guided checkpoints, which ones are "best", and sweep results |
| [architecture.md](architecture.md) | Model architecture facts, model naming conventions (65M vs 45M) |
| [implementation_notes.md](implementation_notes.md) | spec_mamba fixes: off-by-one, rejection masking, CPU kernels, verifiers |
| [results.md](results.md) | Benchmark results (speedup, acceptance, CPU offload) |
| [workflow_rules.md](workflow_rules.md) | Critical rules, workflow, and paper-writing conventions |

## Key references in the repo

- **Paper plan**: `paper/plan.md` — always check before experiments, update after
- **Architecture doc**: `spec_mamba/ARCHITECTURE.md` — update with new benchmark results
- **Project instructions**: `.github/instructions/specssm-project.instructions.md`
- **Impl instructions**: `.github/instructions/spec-mamba-impl.instructions.md`
- **Paper LaTeX**: `paper/neurips2026-specssm/`
