# Python Figure Generation

Generate publication-quality figures with Python and insert them into the LaTeX document.

## Workflow

1. **Clarify** what the user needs:
   - Data source: user-provided file (CSV, JSON, etc.) or generate simulated/example data?
   - Chart type: line, bar, scatter, heatmap, box plot, violin, etc.?
   - Single figure or multi-panel (subfigures)?

2. **Write Python script** to `scripts/generate_<descriptive-name>.py`

3. **Execute the script** with `python scripts/generate_<name>.py`

4. **Insert into LaTeX** — add `\includegraphics` block in the appropriate .tex file

## Publication-Quality Template

Every figure script should follow this pattern:

```python
import matplotlib.pyplot as plt
import numpy as np

# -- Publication-quality settings --
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'lines.linewidth': 1.5,
    'figure.figsize': (6, 4),       # single-column paper
    # 'figure.figsize': (3.5, 2.8), # two-column paper (IEEE, ACM)
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# -- Your plotting code here --
fig, ax = plt.subplots()
# ax.plot(...), ax.bar(...), etc.

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.legend(loc='best')

# -- Save --
plt.tight_layout()
plt.savefig('figures/descriptive-name.pdf', dpi=300, bbox_inches='tight')
plt.close()
print('Saved to figures/descriptive-name.pdf')
```

## LaTeX Insertion

### Single figure

```latex
\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.8\columnwidth]{figures/descriptive-name}
  \caption{A clear, complete caption that describes what the figure shows
    and highlights the key takeaway.}
  \label{fig:descriptive-name}
\end{figure}
```

### Multi-panel figure (subfigures)

```latex
\begin{figure}[htbp]
  \centering
  \begin{subfigure}[b]{0.48\columnwidth}
    \includegraphics[width=\textwidth]{figures/panel-a}
    \caption{Description of panel (a).}
    \label{fig:panel-a}
  \end{subfigure}
  \hfill
  \begin{subfigure}[b]{0.48\columnwidth}
    \includegraphics[width=\textwidth]{figures/panel-b}
    \caption{Description of panel (b).}
    \label{fig:panel-b}
  \end{subfigure}
  \caption{Overall caption for the combined figure.}
  \label{fig:combined}
\end{figure}
```

## Rules

- **Format**: Save as PDF (vector graphics) by default. Use PNG at 300 dpi only for photographs or screenshots.
- **File extension**: Omit the extension in `\includegraphics` — write `{figures/name}` not `{figures/name.pdf}`. LaTeX picks the right file automatically.
- **Float specifier**: Use `[htbp]`. Avoid `[H]` (from the `float` package) unless the figure absolutely must appear at that exact location — it often causes ugly whitespace.
- **Reproducibility**: Save all scripts to `scripts/` so figures can be regenerated. Include any data loading or preprocessing in the script.
- **Color**: Use colorblind-friendly palettes. matplotlib's `tab10` default is acceptable. For explicit control, consider `seaborn.color_palette("colorblind")`.
- **Naming**: Use descriptive names: `accuracy-comparison.pdf` not `fig1.pdf`. Label names should match: `\label{fig:accuracy-comparison}`.

## Common Chart Patterns

### Comparison bar chart (baseline vs. ours)

```python
methods = ['Baseline 1', 'Baseline 2', 'Ours']
scores = [0.80, 0.83, 0.89]
colors = ['#7f7f7f', '#7f7f7f', '#1f77b4']  # gray for baselines, blue for ours

fig, ax = plt.subplots()
bars = ax.bar(methods, scores, color=colors, edgecolor='black', linewidth=0.5)
ax.set_ylabel('Accuracy')
ax.set_ylim(0.7, 0.95)
for bar, score in zip(bars, scores):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{score:.2f}', ha='center', va='bottom', fontsize=9)
```

### Training curve (loss over epochs)

```python
epochs = np.arange(1, 101)
train_loss = ...  # your data
val_loss = ...

fig, ax = plt.subplots()
ax.plot(epochs, train_loss, label='Train', linewidth=1.5)
ax.plot(epochs, val_loss, label='Validation', linewidth=1.5, linestyle='--')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend()
```

### Heatmap (confusion matrix, attention weights)

```python
import seaborn as sns

fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(matrix, annot=True, fmt='.2f', cmap='Blues', ax=ax,
            xticklabels=labels, yticklabels=labels)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
```
