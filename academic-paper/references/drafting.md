# Section Drafting

Write or improve specific sections of the paper.

## Before Writing

1. **Read main.tex** to understand the full paper structure and preamble.
2. **Read the target section** to see what already exists (might be a TODO skeleton or a previous draft).
3. **Read adjacent sections** for coherence — the introduction should flow into related work, methodology should set up experiments, etc.

## Per-Section Writing Guide

### Abstract

Structure: 4 sentences, each with a distinct role.

1. **Problem**: What problem does this paper address? (broad context)
2. **Gap**: Why is it unsolved or insufficiently addressed?
3. **Method**: What do we propose? (one sentence, concrete)
4. **Result**: What did we achieve? (quantitative if possible)

Keep it under 250 words. No citations in the abstract. No undefined acronyms.

### Introduction

Structure: funnel shape, narrowing from broad to specific.

1. **Context** (1-2 paragraphs): Establish the broad area and its importance.
2. **Problem** (1 paragraph): Narrow to the specific problem this paper tackles.
3. **Limitations of existing work** (1 paragraph): Why current solutions fall short — without turning this into a related work section.
4. **Our contribution** (1 paragraph): What this paper does differently. Use a bulleted list for multiple contributions:
   ```latex
   Our contributions are as follows:
   \begin{itemize}
     \item We propose ...
     \item We demonstrate ...
     \item We release ...
   \end{itemize}
   ```
5. **Paper organization** (1-2 sentences): "The rest of this paper is organized as follows. Section~\ref{sec:related} reviews ..."

### Related Work

Structure: group by theme, not chronologically.

- Identify 3-5 research directions relevant to this paper.
- For each group: summarize the approach, cite key papers, then explain how THIS paper differs.
- End with a paragraph positioning this work relative to the closest competitors.

Use `\citep` for parenthetical citations and `\citet` when the author is the sentence subject:
```latex
\citet{vaswani2017attention} introduced the Transformer architecture.
Recent work has explored efficient variants \citep{kitaev2020reformer, wang2020linformer}.
```

### Methodology

Structure: intuition first, then formalize.

1. **Overview** (1 paragraph): High-level idea in plain language. A reader should understand the approach after this paragraph alone.
2. **Problem formulation**: Define notation, input/output, objective function.
3. **Method details**: Step through the approach. Use theorem/definition environments for formal statements:
   ```latex
   \begin{definition}[Name]
     Let $x \in \mathbb{R}^n$ be ...
   \end{definition}
   ```
4. **Algorithm** (if applicable): Use `algorithmic` environment for pseudocode.

Use `\label{eq:xxx}` for every equation, `Equation~\eqref{eq:xxx}` to reference.

### Experiments

Structure: answer "what", "how", "against what".

1. **Datasets**: Name, size, source, preprocessing steps. Use a table if multiple datasets.
2. **Baselines**: What methods are we comparing against? Why these?
3. **Implementation details**: Framework, hyperparameters, hardware, training time. Enough for reproducibility.
4. **Evaluation metrics**: What metrics, and why these metrics are appropriate.

### Results and Analysis

Structure: present results, then explain them.

- **Main results table**: Use `booktabs` for professional formatting:
  ```latex
  \begin{table}[htbp]
    \centering
    \caption{Main results on [dataset]. Best results in \textbf{bold}.}
    \label{tab:main-results}
    \begin{tabular}{lccc}
      \toprule
      Method & Metric 1 & Metric 2 & Metric 3 \\
      \midrule
      Baseline 1 & 0.80 & 0.75 & 0.82 \\
      Baseline 2 & 0.83 & 0.78 & 0.85 \\
      \textbf{Ours} & \textbf{0.89} & \textbf{0.84} & \textbf{0.91} \\
      \bottomrule
    \end{tabular}
  \end{table}
  ```
- **Analysis paragraphs**: One paragraph per key finding. Reference specific numbers from the table.
- **Ablation study** (if applicable): Show the contribution of each component by removing it.
- **Figures**: Reference with `Figure~\ref{fig:xxx}`. If a figure is needed, tell the user and suggest using the figures stage.

### Conclusion

Structure: 3 parts.

1. **Summary** (1 paragraph): Restate what was done and the main result, without copying the abstract.
2. **Limitations** (1 paragraph): Be honest about what the approach cannot do.
3. **Future work** (1-2 sentences): Concrete next steps, not vague wishes.

## Writing Style

- Use active voice: "We propose" not "It is proposed".
- Be precise: "reduces latency by 23%" not "significantly improves performance".
- One idea per paragraph. The first sentence of each paragraph should be readable as a standalone summary.
- Avoid weasel words: "very", "clearly", "obviously", "it is well known that".
