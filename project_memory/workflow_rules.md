# Workflow & Rules

## Critical Rules

### NEVER modify files in parent codebase directories
- The `guided_mamba/`, `spec_mamba/`, `core/`, etc. dirs in the workspace root are for the
  open-source release.
- ALL experimental code changes go in the `exploration/` directory ONLY.
- If you need a file from the parent codebase, COPY it to the exploration dir first.
  This includes `train.py`, config files, data scripts — everything.
- NO EXCEPTIONS.

### NEVER delete checkpoints, data, or any potentially useful files
- Even if analysis suggests they are "useless" — the analysis can be WRONG.
- Always rename/move to a trash directory instead of `rm -rf`.
- Applies to: model checkpoints, training logs, datasets, experiment outputs.
- The cost of keeping files (disk space) is always less than the cost of losing them.

### READ THE ACTUAL LOG before diagnosing issues
- Look at the empirical data (loss trajectory, lr values in logs) FIRST.
- Do NOT run speculative simulations to predict what "should" happen.
- If loss is improving in the log, training is working — period.
- Misdiagnosis + destructive action = catastrophic.

## Accelerate + scheduler behavior (verified)
- AccelerateScheduler wraps `scheduler.step()` — handles grad_accum automatically.
- AccelerateScheduler calls the underlying scheduler `num_processes` times per optimizer step.
- To get a correct cosine schedule: pass `total_steps * num_processes` to the scheduler.
- `scheduler.step()` INSIDE `accelerator.accumulate()` is the CORRECT placement.

## General workflow
- **Update documents after major changes**: whenever making significant code changes
  (bug fixes, new features, architecture changes), update relevant documentation files
  (`ARCHITECTURE.md`, repo memory, etc.) before finishing.
- Before any experiment, check `paper/plan.md` for the relevant work item number.
- After completing an experiment, update both `paper/plan.md` (status) and
  `spec_mamba/ARCHITECTURE.md` (results).

## Paper Writing Rules

### Style
- Avoid overusing em-dashes. Use commas, parentheses, or separate sentences instead.
- Write smooth, logical, readable prose (human academic style).
- Avoid using `\paragraph`.

### Role boundaries
- NEVER remove content, columns, figures, or data from the paper without asking first.
- If something could be removed or simplified, ASK. Do not assume.
- The user makes all editorial decisions about what belongs in the paper.

### Citations & references
- NEVER hallucinate references. Verify a citation exists before adding it (exact title,
  authors, venue, year).
- If unsure whether a paper exists, flag it rather than guessing.
- Use consistent BibTeX keys.
- Always double-check arxiv ID → actual paper title match.

### Overleaf workflow
- Edit directly in `/HSC/users/qiaoye/SSM_SPEC/overleaf-worktree/`
  (on `overleaf-local` branch tracking `overleaf/master`).
- NEVER push from the main repo to Overleaf (diverged histories, wrong tree structure).
- NEVER `cp` files into `overleaf-worktree` (destroys collaborator edits).
- Steps:
  1. PULL first: `cd .../overleaf-worktree && git pull --rebase overleaf master`
  2. Edit files directly in the worktree
  3. Commit and push: `git add -A && git commit -m "<msg>" && git push overleaf HEAD:master`
     - If push rejected: `git pull --rebase overleaf master` then push again.
- No local LaTeX compiler; Overleaf is the only way to compile/preview.
