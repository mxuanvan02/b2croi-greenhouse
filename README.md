# B2CRoI-HQ: Academic Project Repository

This repository contains the manuscript, reproducible scripts, generated data tables, journal template material, and presentation slides for the B2CRoI-HQ greenhouse reliability/safety-factor study.

## Repository structure

```text
FINAL/
├── README.md
├── README_FINAL.md
├── data/
│   ├── raw/
│   ├── external/
│   └── processed/
├── scripts/
├── manuscript/
│   ├── compag/
│   └── standalone/
├── slides/
├── journal_templates/
└── outputs/
    └── submission/
```

## Recommended workflow

1. Treat `data/raw/` and `data/external/` as source inputs.
2. Run scripts from the project root when reproducing tables/figures.
3. Keep generated CSV/LaTeX tables under `data/processed/`.
4. Edit manuscript sources under `manuscript/compag/` for journal submission.
5. Keep presentation materials under `slides/`.

## Main artifacts

- Journal submission source: `manuscript/compag/main_compag_template_harv.tex`
- Compiled journal PDF: `manuscript/compag/main_compag_template_harv.pdf`
- Standalone manuscript: `manuscript/standalone/main_b2croi_h.tex`
- Public result tables: `data/processed/public_table_*`
- Seminar/advisor slides: `slides/`

## Git hygiene

LaTeX intermediate files, Python cache files, and packaged archives are ignored via `.gitignore`. Source `.tex`, `.bib`, scripts, figures, CSV tables, and final PDFs remain trackable.
