---
name: academic-paper
description: "AI-assisted academic paper writing with LaTeX. Use this skill whenever the user wants to write, draft, or edit an academic paper, generate a paper outline or structure, create publication-quality figures with Python, compile LaTeX documents, manage BibTeX references, or mentions anything related to academic/scientific writing, LaTeX, paper sections (abstract, introduction, methodology, results), or research papers. Even if the user just says 'start a new paper' or 'help me with my paper', this skill applies."
---

# Academic Paper Writing

Help write academic papers in LaTeX through a staged workflow. Each stage has its own reference file with detailed instructions — read the relevant one based on what the user needs.

## Stage Routing

Identify the user's intent and read the corresponding reference file:

| User intent | Reference file | When to use |
|-------------|---------------|-------------|
| Start a new paper, create outline | [references/outline.md](./references/outline.md) | User wants to begin a paper from scratch or generate structure |
| Write or improve a section | [references/drafting.md](./references/drafting.md) | User wants to draft, expand, rewrite, or polish specific sections |
| Generate plots or charts | [references/figures.md](./references/figures.md) | User needs data visualization with Python (matplotlib, seaborn, etc.) |
| Compile to PDF or fix errors | [references/compile.md](./references/compile.md) | User wants to build the PDF or is stuck on compilation errors |
| Add or manage citations | [references/bibliography.md](./references/bibliography.md) | User needs to add BibTeX entries, insert \cite commands, or clean up .bib |

If the intent is ambiguous, ask the user which stage they need help with.

## Project Structure Convention

All paper projects follow this layout. When creating a new project (outline stage), scaffold this structure. When working on an existing project, expect files in these locations:

```
my-paper/
├── main.tex              # Primary document (includes all sections)
├── sections/             # One .tex file per section
│   ├── abstract.tex
│   ├── introduction.tex
│   ├── related-work.tex
│   ├── methodology.tex
│   ├── experiments.tex
│   ├── results.tex
│   └── conclusion.tex
├── figures/              # Python-generated plots (PDF/PNG)
├── scripts/              # Python scripts that generate figures
├── references.bib        # Bibliography database
└── Makefile              # Compilation commands (make pdf / make clean)
```

## LaTeX Conventions

These apply across all stages:

**Cross-references** — always use `~` (non-breaking space) before `\ref`:
- `Figure~\ref{fig:xxx}`, `Table~\ref{tab:xxx}`, `Section~\ref{sec:xxx}`

**Float placement** — use `[htbp]`, avoid `[H]` unless absolutely necessary.

**Standard packages** (included in the template):

| Package | Purpose |
|---------|---------|
| amsmath, amssymb, amsthm | Mathematical typesetting |
| graphicx | Image inclusion |
| hyperref | Clickable links (blue links, green citations) |
| booktabs | Professional tables (\toprule, \midrule, \bottomrule) |
| natbib | Bibliography management (\citep, \citet) |
| caption, subcaption | Multi-part figures |
| microtype | Typography refinement (fixes overfull hbox) |

## Important

- Always read the relevant reference file before starting work on a stage.
- When the user's request spans multiple stages (e.g., "write the results section and add a comparison figure"), handle them sequentially — read each reference file as you enter that stage.
- Do not generate placeholder or lorem-ipsum content. Every sentence should be meaningful, even in a first draft.
