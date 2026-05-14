#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
DATA = PROJECT / 'data' / 'processed'
RESULTS = ROOT / 'results'
TABLE_OUT = RESULTS / 'tables'
FIG_OUT = RESULTS / 'figures'
PAPER_FIG = PROJECT / 'manuscript' / 'compag' / 'assets' / 'figures'

sys.path.insert(0, str(ROOT / 'src'))
from b2croi.verify import copy_public_tables, numeric_sanity, verify_files  # noqa: E402

REQUIRED_TABLES = [
    'public_table_operating_regimes.csv',
    'public_table_safety_calibration.csv',
    'public_table_external_validation.csv',
    'public_table_ablation.csv',
]

PAPER_FIGURES = [
    # Track the assets currently referenced by main_compag_elsarticle.tex.
    'fig_architecture.pdf',
    'fig_mode_switch.pdf',
    'fig_loss_fairness_pareto.tex',
    'fig_robustness_grid.pdf',
    'fig_external_validation.pdf',
]


def ok(msg: str) -> None:
    print(f'[OK] {msg}')


def fail(msg: str) -> None:
    print(f'[FAIL] {msg}')
    raise SystemExit(1)


def regenerate_figures_if_available() -> list[str]:
    """Generate formal plots when pandas/numpy/matplotlib are installed.

    The default quick verification remains runnable with the Python standard library only.
    """
    try:
        from b2croi.plotting import make_all
    except Exception as exc:  # dependency-light fallback
        print(f'[SKIP] Formal plot regeneration unavailable: {exc}')
        print('[SKIP] Install requirements.txt to enable matplotlib-based figure regeneration.')
        return []
    outputs = make_all(DATA, FIG_OUT)
    generated = []
    for out in outputs:
        if not out.exists() or out.stat().st_size <= 1024:
            fail(f'Figure generation failed or too small: {out}')
        rel = out.relative_to(ROOT)
        ok(f'Generated {rel}')
        generated.append(str(rel))
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description='Verify B2CRoI-H(Q) public tables, manuscript figures, and optional formal plots.')
    parser.add_argument('--quick', action='store_true', help='Run fast verification checks.')
    parser.add_argument('--with-plots', action='store_true', help='Also regenerate formal matplotlib plots when dependencies are installed.')
    args = parser.parse_args()

    try:
        tables = copy_public_tables(DATA, TABLE_OUT, REQUIRED_TABLES)
        for name, meta in tables.items():
            ok(f'Loaded {name}: {meta["rows"]} rows')
        claims = numeric_sanity(DATA)
        ok('Operating-regime table contains positive fairness gains')
        ok(f'Calibration Brier range: {claims["brier_min"]:.4f} to {claims["brier_max"]:.4f}')
        ok(f'External validation cases: {claims["external_cases_total"]}')
        fig_sizes = verify_files([PAPER_FIG / name for name in PAPER_FIGURES])
        for name, size in fig_sizes.items():
            ok(f'Found manuscript figure {name} ({size} bytes)')
        generated = regenerate_figures_if_available() if args.with_plots else []
    except Exception as exc:
        fail(str(exc))

    summary = {
        'tables': tables,
        'numeric_claims': claims,
        'paper_figures': fig_sizes,
        'generated_figures': generated,
        'plot_regeneration': 'enabled' if generated else 'not_run_or_unavailable',
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / 'verification_summary.json'
    out.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    ok(f'Wrote {out.relative_to(ROOT)}')
    ok('Verification completed')


if __name__ == '__main__':
    main()
