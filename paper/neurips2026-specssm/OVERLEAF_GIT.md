# Overleaf Git Setup

This project is prepared to compile directly on Overleaf.

## 1) Get your Overleaf Git URL
1. Create a blank project in Overleaf.
2. In Overleaf, open `Menu` -> `Git`.
3. Copy the project Git URL.

## 2) Initialize and push this paper folder
Run these commands from this folder (`paper/neurips2026-specssm`):

```bash
git init
git add .
git commit -m "Initialize NeurIPS 2026 paper"
git branch -M main
git remote add overleaf <OVERLEAF_GIT_URL>
git push -u overleaf main
```

## 3) Daily sync workflow

Push local updates:

```bash
git add .
git commit -m "Update paper"
git push
```

Pull changes from Overleaf:

```bash
git pull --rebase overleaf main
```

## Notes
- The NeurIPS style file `neurips_2026.sty` is vendored locally.
- The official checklist is vendored at `checklist/neurips_checklist.tex`.
- Keep the project root as this folder when using Overleaf Git.
