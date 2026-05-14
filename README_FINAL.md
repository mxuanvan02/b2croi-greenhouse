# FINAL package — B2CRoI-H(Q) COMPAG submission

This folder is the cleaned handoff package. Older drafts, logs, review artifacts, and exploratory rebuild files were moved to `_archive/final_cleanup_20260502_2350/` and were not deleted.

## Main paper

- `paper_compag_submission/` — final Elsevier/COMPAG `elsarticle` submission source folder.
- `paper_compag_submission/main_compag_elsarticle.tex` — main journal-template source.
- `paper_compag_submission/main_compag_elsarticle.pdf` — compiled COMPAG manuscript PDF.
- `B2CRoI-HQ_COMPAG_submission_source_20260502.zip` — zipped submission source package.

## Standalone paper backup

- `paper_standalone/` — non-template manuscript source/PDF backup.

## Final scripts

- `scripts_final/` — final scripts used for figures, benchmark tables, sensitivity, alarm activation, and external validation.

## Rebuild outputs

- `rebuild_outputs/tables/` — final CSV/LaTeX result tables.
- `rebuild_outputs/external_datasets/` — external validation dataset cache.

## Build command

From `paper_compag_submission/`:

```bash
xelatex -interaction=nonstopmode main_compag_elsarticle.tex
bibtex main_compag_elsarticle
xelatex -interaction=nonstopmode main_compag_elsarticle.tex
xelatex -interaction=nonstopmode main_compag_elsarticle.tex
```
