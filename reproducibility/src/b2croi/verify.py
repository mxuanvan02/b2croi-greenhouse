from __future__ import annotations

import csv
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def copy_public_tables(data_dir: Path, out_dir: Path, names: list[str]) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name in names:
        src = data_dir / name
        if not src.exists():
            raise FileNotFoundError(src)
        rows = read_csv(src)
        if not rows:
            raise ValueError(f'Empty table: {src}')
        (out_dir / name).write_text(src.read_text(encoding='utf-8'), encoding='utf-8')
        summary[name] = {'rows': len(rows), 'columns': list(rows[0].keys())}
    return summary


def numeric_sanity(data_dir: Path) -> dict:
    op = read_csv(data_dir / 'public_table_operating_regimes.csv')
    cal = read_csv(data_dir / 'public_table_safety_calibration.csv')
    ext = read_csv(data_dir / 'public_table_external_validation.csv')

    fairness = [float(r['Mean fairness $\\Delta$']) for r in op]
    if not any(v > 0 for v in fairness):
        raise ValueError('No positive fairness gain in operating-regime table')

    brier = [float(r['Brier score']) for r in cal]
    if min(brier) >= max(brier):
        raise ValueError('Calibration Brier scores are not separable')

    cases_total = sum(int(float(r['Cases'])) for r in ext)
    if cases_total <= 0:
        raise ValueError('External validation cases are not positive')

    return {
        'positive_fairness_gain': True,
        'brier_min': min(brier),
        'brier_max': max(brier),
        'external_cases_total': cases_total,
    }


def verify_files(paths: list[Path], min_bytes: int = 1024) -> dict[str, int]:
    out = {}
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        size = path.stat().st_size
        if size <= min_bytes:
            raise ValueError(f'Suspiciously small file: {path} ({size} bytes)')
        out[path.name] = size
    return out
