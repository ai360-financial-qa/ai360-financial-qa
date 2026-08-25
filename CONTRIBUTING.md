# Contributing

## Setup

```bash
uv sync --extra dev
```

## Branches and commits

- Branch off `main`: `feat/<topic>`, `fix/<topic>`, `exp/<topic>`.
- Rebase instead of merging when updating a branch: `git pull --rebase`.
  Merge commits titled "Merge branch 'main' of github.com:..." are noise; the
  history was cleaned once already to remove them.
- Write commit subjects as `type(scope): summary` in the imperative mood:
  `feat(chunkers): add table-split chunker`, `fix(rag): normalize embeddings`.
  Types: `feat`, `fix`, `refactor`, `perf`, `docs`, `test`, `chore`, `exp`.

## What not to commit

- `indexes/` — embedding indexes are regenerable; rebuild them with
  `python -m financial_qa.rag.precalc`.
- `logs/` and any `*.zip` of run logs.
- Notebook outputs larger than a screenful. Clear them before committing:
  `jupyter nbconvert --clear-output --inplace notebooks/**/*.ipynb`.

Source PDFs under `data/raw/` *are* tracked, since they are not regenerable.

## Adding a chunker

Subclass `financial_qa.base.BaseChunker`, implement `chunk()`, and export it
from `financial_qa/chunkers/__init__.py`. Add an experiment notebook under
`notebooks/chunking/` if you are comparing retrieval quality.

## Before opening a PR

```bash
pytest
ruff check src tests
```
