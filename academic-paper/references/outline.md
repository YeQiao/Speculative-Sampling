# Outline Generation

Generate a paper outline and scaffold the project directory.

## Workflow

1. **Gather context** — ask the user (one question at a time):
   - What is the paper about? (topic, abstract, or key idea)
   - Target venue? (journal name, conference, or general — this determines format and length expectations)
   - What is the core contribution? (what makes this work novel)

2. **Generate outline** — present a Markdown outline for the user to review:
   - Proposed title
   - Each section with 2-3 sentences describing what it will cover
   - Suggested figure/table plan (what visuals the paper might need)

3. **Wait for user approval** — let them modify the outline before scaffolding.

4. **Scaffold the project** — create all files:

### main.tex

Use this template (Research Paper, extracted from claude-prism):

```latex
\documentclass[12pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb,amsthm}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{natbib}
\usepackage{float}
\usepackage{caption,subcaption}
\usepackage{microtype}
\usepackage{hyperref}

\hypersetup{
  colorlinks=true,
  linkcolor=blue!70!black,
  citecolor=green!50!black,
  urlcolor=blue!80!black
}

\newtheorem{theorem}{Theorem}[section]
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{definition}[theorem]{Definition}

\begin{document}

\title{PAPER TITLE}
\author{Author Name \\ Institution \\ \texttt{email@example.com}}
\date{}
\maketitle

\begin{abstract}
\input{sections/abstract}
\end{abstract}

\section{Introduction}\label{sec:intro}
\input{sections/introduction}

\section{Related Work}\label{sec:related}
\input{sections/related-work}

\section{Methodology}\label{sec:method}
\input{sections/methodology}

\section{Experiments}\label{sec:experiments}
\input{sections/experiments}

\section{Results and Analysis}\label{sec:results}
\input{sections/results}

\section{Conclusion}\label{sec:conclusion}
\input{sections/conclusion}

\bibliographystyle{plainnat}
\bibliography{references}

\end{document}
```

Adapt the section list based on the outline — not every paper needs all sections (e.g., a survey paper might skip Experiments).

### Section skeleton files

Create each section as `sections/<name>.tex` with writing prompts based on the outline:

```latex
% TODO: [2-3 sentences from the outline describing what this section should cover]
%
% Key points to address:
% - [point 1]
% - [point 2]
% - [point 3]
```

### Makefile

```makefile
TEX = pdflatex
BIB = bibtex
MAIN = main

.PHONY: pdf clean

pdf:
	$(TEX) $(MAIN)
	$(BIB) $(MAIN)
	$(TEX) $(MAIN)
	$(TEX) $(MAIN)

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot *.fls *.fdb_latexmk *.synctex.gz
```

### Other files

- `references.bib` — empty file (will be populated later)
- `figures/` — empty directory (create with a `.gitkeep`)
- `scripts/` — empty directory (create with a `.gitkeep`)

## Key Principles

- The outline is a thinking tool, not a commitment. Encourage the user to revise it.
- Section count and names should match the paper's needs, not a fixed template. A methods paper might have 3 methodology subsections; a systems paper might have an "Architecture" section instead of "Methodology".
- Each TODO comment should be specific to THIS paper, not generic boilerplate.
