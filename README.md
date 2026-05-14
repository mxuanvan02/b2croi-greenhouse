# B2CRoI Greenhouse

Public code and result artifacts for B2CRoI-H(Q), a burst-belief Risk-of-Information scheduler for safety--fairness trade-offs in smart greenhouse sensing.

## Repository Layout

```text
scripts/                 Experiment, validation, table, and figure-generation scripts
reproducibility/         Quick verification, full-run driver, tests, and baseline notes
data/processed/          Public result tables used by the study
assets/figures/          Release figure assets checked by the verifier
assets/tables/           Release table assets
```

Raw third-party datasets are not redistributed here. Download them from their original public records and place them under the paths below.

## Data Setup

Primary dataset:

```text
data/raw/Full Data Set.csv
```

External validation dataset:

```text
data/external/mendeley_3dw54yhhcc/Greenhouse climate dataset from 26 to 30-01-2019.xlsx
```

Custom locations are supported through environment variables:

```bash
export B2CROI_DATA_ROOT=/path/to/primary/data/folder
export B2CROI_EXTERNAL_DATA_ROOT=/path/to/external/data/folder
export B2CROI_EXTERNAL_DATA_FILE="Greenhouse climate dataset from 26 to 30-01-2019.xlsx"
```

## Environment

Python 3.10+ is recommended.

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r reproducibility/requirements.txt
```

## Quick Verification

```bash
cd reproducibility
./run_all.sh
```

The quick check verifies public result tables, checks required figure assets, copies checked tables into `reproducibility/results/tables/`, and writes `reproducibility/results/verification_summary.json`.

Optional plot regeneration:

```bash
cd reproducibility
python3 run_verification.py --quick --with-plots
```

## Full Reproduction

After placing the raw datasets, run from the repository root:

```bash
bash reproducibility/run_full_end_to_end.sh
```

This runs the primary benchmark, ablations, stress tests, sensitivity sweeps, external validation, public table generation, verification, and baseline decision tests. Existing processed outputs are snapshotted before overwrite.

## Main Scripts

- `scripts/run_primary_benchmark.py`: primary paired benchmark.
- `scripts/run_network_stress.py`: network-size and heterogeneity stress tests.
- `scripts/run_mode_sensitivity.py`: mode-switch parameter sensitivity.
- `scripts/run_alarm_diagnostics.py`: fairness-alarm activation diagnostics.
- `scripts/run_method_ablation.py`: method-component ablation.
- `scripts/run_metadata_ablation.py`: metadata-free scoring ablation.
- `scripts/second_dataset_validate_mendeley.py`: external greenhouse dataset validation.
- `scripts/make_public_tables.py`: public table generation.

## Release Notes

- `data/raw/` and `data/external/` are ignored by default because they contain downloaded third-party data.
- Generated caches, logs, local environments, and archive files are ignored by `.gitignore`.
- Baseline implementation notes are in `reproducibility/baselines/`.
