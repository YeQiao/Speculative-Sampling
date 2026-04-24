# LaTeX Compilation

Compile the paper to PDF and diagnose errors.

## Compilation Pipeline

The full compilation requires multiple passes to resolve cross-references and citations:

```bash
pdflatex main.tex     # First pass: generate .aux files
bibtex main           # Process citations from .bib
pdflatex main.tex     # Second pass: resolve citations
pdflatex main.tex     # Third pass: finalize cross-references
```

If the project has a Makefile (created by the outline stage), just run:

```bash
make pdf
```

To clean build artifacts:

```bash
make clean
```

## Before Compiling

Check these prerequisites:

1. **LaTeX installed?** Run `which pdflatex` — if not found, the user needs to install TeX Live / MacTeX / MiKTeX.
2. **main.tex exists?** Confirm the file is in the current directory.
3. **All \input files exist?** Each `\input{sections/xxx}` must have a corresponding file, even if empty.
4. **references.bib exists?** Even if empty, bibtex expects this file.

## Error Diagnosis

When compilation fails, read the `.log` file to find the error. Common patterns:

### Missing character (Unicode issue)

**Symptom**: `Missing character: There is no X in font ...` in the log.

**Cause**: pdflatex cannot handle certain Unicode characters.

**Fix**: Switch to xelatex. Update the Makefile:
```makefile
TEX = xelatex
```
Or if no Makefile, run: `xelatex main.tex` instead of `pdflatex main.tex`.

### Mismatched \begin and \end

**Symptom**: `Missing \endgroup inserted` or `Extra \endgroup`.

**Fix**: Search for unmatched environment pairs. Common culprits:
- Nested `\begin{figure}` inside `\begin{table}`
- Missing `\end{itemize}` or `\end{enumerate}`
- Copy-paste errors leaving orphaned `\begin{}` or `\end{}`

Grep for environment boundaries:
```bash
grep -n '\\begin\|\\end' sections/*.tex main.tex
```

### Citations showing as [?]

**Symptom**: PDF shows `[?]` instead of citation numbers/names.

**Checklist**:
1. Did bibtex run? Check for `main.bbl` file.
2. Does the cite key exist in `references.bib`? Run: `grep 'key_name' references.bib`
3. Is `\bibliography{references}` in main.tex (without `.bib` extension)?
4. Is `\bibliographystyle{plainnat}` present?

If all checks pass, run the full 3-pass compilation again.

### File not found

**Symptom**: `File 'figures/xxx' not found` or `Cannot find image file`.

**Fix**: Verify the file exists at the referenced path:
```bash
ls figures/
```
Common issues:
- Wrong filename or typo
- Extension mismatch (wrote `.pdf` but file is `.png`)
- File not yet generated (run the Python script first)

### Overfull hbox

**Symptom**: `Overfull \hbox (Xpt too wide)` warning. Text extends past the margin.

**Fix** (in order of preference):
1. Add `\usepackage{microtype}` to the preamble (often fixes it automatically)
2. Rephrase the sentence to break differently
3. For URLs: use `\usepackage{url}` and `\url{...}` which allows line breaks
4. For tables: use `tabularx` or reduce font size with `{\small ...}`

### Table overflows margin

**Fix options**:
- Reduce font: `{\small \begin{tabular}...}`
- Use `tabularx` for auto-width columns: `\begin{tabularx}{\columnwidth}{lXXX}`
- Use `\resizebox{\columnwidth}{!}{\begin{tabular}...}` as last resort (changes font size non-uniformly)

## Quick Reference

| Want to... | Command |
|------------|---------|
| Full compile | `make pdf` or run the 3-pass pipeline |
| Clean artifacts | `make clean` |
| See errors | `grep -A2 '^\!' main.log` |
| See warnings | `grep 'Warning' main.log` |
| Check page count | `pdfinfo main.pdf \| grep Pages` |
| Use xelatex instead | Change `TEX = xelatex` in Makefile |
