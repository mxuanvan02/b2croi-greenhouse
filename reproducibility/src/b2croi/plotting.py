from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

MUTED_BLUE = '#6B8FBF'
MUTED_TEAL = '#7BB6A4'
MUTED_ORANGE = '#D8A15D'
SOFT_GRAY = '#7A8491'
PALE_GREEN = '#A9CFA4'
PALE_RED = '#D99A8C'
DARK = '#2F3640'
GRID = '#D8DEE8'


def set_style() -> None:
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['DejaVu Serif', 'Computer Modern Roman'],
        'mathtext.fontset': 'cm',
        'font.size': 8.5,
        'axes.titlesize': 9.5,
        'axes.labelsize': 8.5,
        'xtick.labelsize': 7.8,
        'ytick.labelsize': 7.8,
        'legend.fontsize': 7.8,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'axes.edgecolor': SOFT_GRAY,
        'axes.labelcolor': DARK,
        'xtick.color': DARK,
        'ytick.color': DARK,
    })


def _finish(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f'{name}.pdf', bbox_inches='tight')
    fig.savefig(out / f'{name}.png', dpi=350, bbox_inches='tight')
    plt.close(fig)


def _clean(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(True, axis='x', color=GRID, lw=0.55, alpha=0.65)
    ax.set_axisbelow(True)


def plot_operating_tradeoff(data_dir: Path, out_dir: Path) -> list[Path]:
    set_style()
    df = pd.read_csv(data_dir / 'public_table_operating_regimes.csv')
    labels = [x.replace('Fairness-constrained', 'Fairness constrained').replace('Loss-prioritized', 'Loss prioritized').replace('Hybrid constrained', 'Hybrid constrained') for x in df['Operating regime']]
    y = np.arange(len(df))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.55), gridspec_kw={'width_ratios': [1.15, 1.0]})

    ax = axes[0]
    loss_reduction = -df['Mean loss $\\Delta$']
    miss_reduction = -df['Mean missed $\\Delta$ (pp)']
    fairness_gain = df['Mean fairness $\\Delta$']
    ax.scatter(loss_reduction, y + 0.18, s=44, color=MUTED_BLUE, label='Safety-loss reduction')
    ax.scatter(miss_reduction, y, s=44, color=MUTED_TEAL, label='Missed-violation reduction')
    ax.scatter(fairness_gain * 10, y - 0.18, s=44, color=MUTED_ORANGE, label='Fairness gain ×10')
    for yi in y:
        ax.axhline(yi, color=GRID, lw=0.45, alpha=0.55)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Improvement magnitude')
    ax.set_title('Benefit dimensions')
    _clean(ax)
    ax.legend(frameon=False, loc='lower center', bbox_to_anchor=(0.5, -0.34), ncol=1)

    ax = axes[1]
    ax.axvline(0.05, color=PALE_RED, lw=1.0, ls='--', label='RMSE +0.05')
    ax.scatter(df['Mean RMSE $\\Delta$'], y, s=68, color=SOFT_GRAY)
    for yi, val in zip(y, df['Mean RMSE $\\Delta$']):
        ax.plot([0, val], [yi, yi], color=GRID, lw=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_xlabel('Mean RMSE delta')
    ax.set_title('Accuracy cost')
    _clean(ax)
    ax.legend(frameon=False, loc='lower center', bbox_to_anchor=(0.5, -0.34), ncol=1)

    fig.suptitle('Operating regimes reveal a controlled safety--fairness--accuracy trade-off', fontsize=10.3, fontweight='bold')
    fig.tight_layout(rect=[0, 0.12, 1, 0.94])
    _finish(fig, out_dir, 'result_operating_regimes_formal')
    return [out_dir / 'result_operating_regimes_formal.pdf', out_dir / 'result_operating_regimes_formal.png']


def plot_calibration_slope(data_dir: Path, out_dir: Path) -> list[Path]:
    set_style()
    df = pd.read_csv(data_dir / 'public_table_safety_calibration.csv')
    fig, ax = plt.subplots(figsize=(5.6, 3.45))
    x = np.arange(len(df))
    for metric, color, ypos in [('Brier score', MUTED_BLUE, 0), ('ECE', MUTED_TEAL, 1)]:
        vals = df[metric].to_numpy()
        ax.plot(x, vals, marker='o', color=color, lw=1.4, label=metric)
        for xi, v in zip(x, vals):
            ax.text(xi, v, f'{v:.3f}', ha='center', va='bottom', fontsize=7.2, color=DARK)
    ax.set_xticks(x)
    ax.set_xticklabels(df['Model'])
    ax.set_ylabel('Calibration error (lower is better)')
    ax.set_title('Empirical residual tails improve probability calibration', fontweight='bold')
    ax.grid(True, axis='y', color=GRID, lw=0.55, alpha=0.70)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=2)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    _finish(fig, out_dir, 'result_calibration_formal')
    return [out_dir / 'result_calibration_formal.pdf', out_dir / 'result_calibration_formal.png']


def _count_to_percent(series: pd.Series, cases: pd.Series) -> pd.Series:
    return series.astype(str).str.split('/').str[0].astype(float) / cases.astype(float) * 100.0


def plot_external_dotplot(data_dir: Path, out_dir: Path) -> list[Path]:
    set_style()
    df = pd.read_csv(data_dir / 'public_table_external_validation.csv')
    metrics = [
        ('Loss better', 'Loss better', MUTED_BLUE),
        ('Missed better', 'Missed better', MUTED_TEAL),
        ('Fairness better', 'Fairness better', MUTED_ORANGE),
        ('RMSE $\\Delta\\leq0.05$', 'RMSE within 0.05', SOFT_GRAY),
    ]
    fig, ax = plt.subplots(figsize=(6.8, 3.65))
    base_y = np.arange(len(df))[::-1]
    offsets = np.linspace(0.24, -0.24, len(metrics))
    for off, (col, label, color) in zip(offsets, metrics):
        vals = _count_to_percent(df[col], df['Cases'])
        ax.scatter(vals, base_y + off, s=46, color=color, label=label)
    ax.set_yticks(base_y)
    ax.set_yticklabels(df['Variable set'])
    ax.set_xlim(0, 105)
    ax.set_xlabel('Paired cases satisfying criterion (%)')
    ax.set_title('External validation across independent variable sets', fontweight='bold')
    _clean(ax)
    ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2)
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    _finish(fig, out_dir, 'result_external_validation_formal')
    return [out_dir / 'result_external_validation_formal.pdf', out_dir / 'result_external_validation_formal.png']


def make_all(data_dir: Path, out_dir: Path) -> list[Path]:
    outputs = []
    outputs += plot_operating_tradeoff(data_dir, out_dir)
    outputs += plot_calibration_slope(data_dir, out_dir)
    outputs += plot_external_dotplot(data_dir, out_dir)
    return outputs
