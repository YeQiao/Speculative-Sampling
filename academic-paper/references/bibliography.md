# Bibliography Management

Manage BibTeX references and citations in the paper.

## Operations

### Add a new entry

When the user provides a paper title, DOI, or URL:

1. Generate a properly formatted BibTeX entry
2. Append it to `references.bib`
3. If the user indicates where it should be cited, insert `\citep{key}` or `\citet{key}` at that location

**Key naming convention**: `first_author_lastname` + `year` + `keyword`

Examples:
- `vaswani2017attention`
- `he2016deep`
- `brown2020language`

### Format and clean up .bib

When asked to format or clean the bibliography:

1. Read `references.bib`
2. Standardize formatting (consistent indentation, field order)
3. Fill in missing fields where possible (common: missing `pages`, `volume`, `publisher`)
4. Remove duplicate entries
5. Sort entries alphabetically by key

### Insert citations in text

When the user says "this needs a citation" or "cite X here":

1. Find or create the BibTeX entry
2. Choose the right citation command based on context
3. Insert at the appropriate location in the .tex file

### Check for issues

When asked to verify references:

```bash
# Find all cite keys used in .tex files
grep -roh '\\cite[pt]*{[^}]*}' sections/*.tex main.tex | sort -u

# Find all keys defined in .bib
grep -o '@[a-z]*{[^,]*' references.bib | cut -d'{' -f2 | sort -u

# Compare to find mismatches
```

Report:
- **Uncited entries**: keys in .bib that never appear in any \cite command
- **Unresolved citations**: \cite{key} in .tex where the key is not in .bib

## BibTeX Entry Templates

### Journal article

```bibtex
@article{author2024keyword,
  title     = {Full Paper Title},
  author    = {Last, First and Last2, First2 and Last3, First3},
  journal   = {Journal Name},
  volume    = {10},
  number    = {3},
  pages     = {1--20},
  year      = {2024},
  publisher = {Publisher Name}
}
```

### Conference paper

```bibtex
@inproceedings{author2024keyword,
  title     = {Conference Paper Title},
  author    = {Last, First and Last2, First2},
  booktitle = {Proceedings of the Conference on X (CONF'24)},
  pages     = {100--110},
  year      = {2024}
}
```

### Preprint (arXiv)

```bibtex
@article{author2024keyword,
  title   = {Preprint Title},
  author  = {Last, First and Last2, First2},
  journal = {arXiv preprint arXiv:2401.12345},
  year    = {2024}
}
```

### Book

```bibtex
@book{author2024keyword,
  title     = {Book Title},
  author    = {Last, First},
  publisher = {Publisher Name},
  year      = {2024},
  edition   = {2nd}
}
```

## Citation Commands (natbib)

| Command | Output | When to use |
|---------|--------|-------------|
| `\citep{key}` | (Author, 2024) | Default choice. Parenthetical citation at end of sentence. |
| `\citet{key}` | Author (2024) | When the author is the grammatical subject: "\citet{vaswani2017attention} proposed..." |
| `\citep[p.~5]{key}` | (Author, 2024, p. 5) | Citing a specific page or section. |
| `\citep{k1,k2}` | (A, 2024; B, 2023) | Multiple works supporting the same claim. Order chronologically. |
| `\citeauthor{key}` | Author | Author name only, no year. Rare — use when year was already mentioned. |
| `\citeyear{key}` | 2024 | Year only, no author. Rare. |

## Style Rules

- **Order multiple citations chronologically**: `\citep{older2020, newer2023}` not the reverse.
- **Author name format in .bib**: Use `Last, First` format. For multi-word last names, use braces: `{Van der Berg}, Jan`.
- **Consistent venue names**: Use full names or consistent abbreviations. Do not mix "ICML" and "International Conference on Machine Learning" in the same .bib file.
- **Page ranges**: Use en-dash `--` not hyphen `-`: `pages = {1--20}` not `pages = {1-20}`.
- **Special characters in titles**: Wrap in braces to preserve capitalization: `title = {Learning {B}ayesian {N}etworks}`.
