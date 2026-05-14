#!/usr/bin/env bash
# End-to-end full-window reproduction for B2CRoI-H(Q)  package.
# Run from FINAL/ or FINAL/reproducibility/.
# This script never deletes old outputs permanently: it snapshots/archives before overwriting.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FINAL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$FINAL_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="reproducibility/results/logs"
SNAP_DIR="data/processed/_snapshot_before_full_${STAMP}"
ARCH_DIR="data/processed/_archive_unused_after_full_${STAMP}"
mkdir -p "$LOG_DIR" "$SNAP_DIR" "$ARCH_DIR"
LOG="$LOG_DIR/full_end_to_end_${STAMP}.log"

exec > >(tee -a "$LOG") 2>&1

echo "[INFO] FINAL_DIR=$FINAL_DIR"
echo "[INFO] Log: $LOG"
echo "[INFO] Snapshot: $SNAP_DIR"

if [ ! -f "data/raw/Full Data Set.csv" ]; then
  echo "[ERROR] Missing primary dataset: data/raw/Full Data Set.csv" >&2
  exit 1
fi
if [ ! -f "data/external/mendeley_3dw54yhhcc/Greenhouse climate dataset from 26 to 30-01-2019.xlsx" ]; then
  echo "[ERROR] Missing external dataset Excel under data/external/mendeley_3dw54yhhcc/" >&2
  exit 1
fi

# Snapshot current top-level processed files before overwriting.
find data/processed -maxdepth 1 -type f -name '*.csv' -o -maxdepth 1 -type f -name '*.tex' | while read -r f; do
  cp "$f" "$SNAP_DIR/"
done

python3 - <<'PY'
import importlib.util, sys
missing=[]
for mod in ['numpy','pandas','matplotlib','openpyxl']:
    if importlib.util.find_spec(mod) is None:
        missing.append(mod)
if missing:
    print('[ERROR] Missing Python packages:', ', '.join(missing))
    print('Install with: python3 -m pip install -r reproducibility/requirements.txt')
    sys.exit(1)
print('[OK] Python dependencies available')
PY

echo "[STEP] Primary benchmark: full available weekly windows"
python3 scripts/b2croi_v8q_benchmark.py --n-windows -1

echo "[STEP] Ablation: full available weekly windows"
python3 scripts/b2croi_v8_ablation.py --n-windows -1

echo "[STEP] N/heterogeneity stress test: full available weekly windows"
python3 scripts/b2croi_v8q_stress_n.py --n-windows -1

echo "[STEP] Sensitivity sweep: full available weekly windows"
python3 scripts/b2croi_v8q_sensitivity.py --n-windows -1

echo "[STEP] Alarm activation diagnostics"
python3 scripts/b2croi_v8q_alarm_activation.py

echo "[STEP] External validation"
python3 scripts/second_dataset_validate_mendeley.py

echo "[STEP] Public table generation"
python3 scripts/make_public_tables.py

echo "[STEP] Reproducibility verification + formal plots"
cd reproducibility
python3 run_verification.py --quick --with-plots
cd "$FINAL_DIR"

echo "[STEP] Baseline decision tests"
python3 reproducibility/tests/test_baseline_decisions.py

echo "[STEP] Optional decisive plots (non-fatal if matplotlib is unavailable)"
python3 reproducibility/src/b2croi/extra_plots.py || echo "[WARN] Optional decisive plots were not rendered; install numpy matplotlib and rerun extra_plots.py"

echo "[STEP] Archive unused legacy/intermediate processed results"
cat > "$ARCH_DIR/MANIFEST.md" <<EOF
# Archive after full end-to-end run

Created: $STAMP
Reason: keep final B2CRoI-H(Q) reproducibility outputs in data/processed while moving legacy/intermediate files out of the active result set.
Restore by moving selected files back to data/processed/.
EOF
python3 - <<PY
from pathlib import Path
import shutil
base=Path('data/processed')
arch=Path('$ARCH_DIR')
keep={
 'b2croi_v8q_raw.csv','b2croi_v8q_summary.csv','b2croi_v8q_paired.csv',
 'b2croi_v8q_stress_n_raw.csv','b2croi_v8q_stress_n_summary.csv','b2croi_v8q_stress_n_paired.csv',
 'b2croi_v8q_sensitivity_raw.csv','b2croi_v8q_sensitivity_summary.csv',
 'b2croi_v8q_alarm_activation_raw.csv','b2croi_v8q_alarm_activation_summary.csv',
 'b2croi_v8_ablation_raw.csv','b2croi_v8_ablation_summary.csv','b2croi_v8_ablation_paired.csv',
 'b2croi_v8_ablation_stress_raw.csv','b2croi_v8_ablation_stress_summary.csv','b2croi_v8_ablation_stress_paired.csv',
 'final_v6_v7_v8_summary.csv','final_v6_v7_v8_stress_by_N.csv','final_v6_v7_v8_stress_case_counts.csv',
 'residual_quantile_table.csv','safety_probability_calibration_raw.csv','safety_probability_calibration_summary.csv','safety_probability_calibration_bins.csv',
 'second_dataset_mendeley_raw.csv','second_dataset_mendeley_summary.csv','second_dataset_mendeley_paired.csv','second_dataset_mendeley_case_counts.csv',
 'public_table_operating_regimes.csv','public_table_ablation.csv','public_table_safety_calibration.csv','public_table_external_validation.csv',
 'public_table_operating_regimes_latex.tex','public_table_ablation_latex.tex','public_table_safety_calibration_latex.tex','public_table_external_validation_latex.tex',
}
for name in ['public_table_standard_baselines_latex.tex','public_table_sensitivity_latex.tex']:
    if (base/name).exists(): keep.add(name)
moved=[]
for p in list(base.iterdir()):
    if p.is_file() and p.name not in keep:
        shutil.move(str(p), str(arch/p.name)); moved.append(p.name)
(arch/'archived_files.txt').write_text('\n'.join(sorted(moved))+'\n')
print(f'[OK] Archived {len(moved)} unused files to {arch}')
PY

echo "[SUMMARY] Active processed files: $(find data/processed -maxdepth 1 -type f | wc -l | tr -d ' ')"
echo "[SUMMARY] Verification summary: reproducibility/results/verification_summary.json"
echo "[SUMMARY] Formal plots: reproducibility/results/figures/"
echo "[SUMMARY] Log: $LOG"
echo "[OK] Full end-to-end reproduction completed"
