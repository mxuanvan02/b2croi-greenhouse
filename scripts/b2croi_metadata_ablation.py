#!/usr/bin/env python3
"""Ablate B2CRoI-H(Q)'s local decision-time metadata.

The primary B2CRoI-H(Q) run is edge-assisted: each loop exposes a light
risk/innovation signal before the bandwidth-limited full refresh. This script
compares that setting with a metadata-free gateway-only forecast variant that
uses only AR(1) forecasted values in the B2CRoI score.
"""
from pathlib import Path
import numpy as np
import pandas as pd

import b2croi_v8q_benchmark as bench

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)
TABLES = [
    ROOT / "study" / "" / "assets" / "tables" / "public_table_metadata_ablation_latex.tex",
    ROOT / "study" / "standalone" / "assets" / "tables" / "public_table_metadata_ablation_latex.tex",
]


def metadata_free_score(xh, xt, a, ar, pi_bad, dual_f, dual_s, kind, counts, total_choices,
                        h=4, q_s=None, q_f=None, q_e=0.0, variant="b2croi_v8q",
                        tau_J=0.985, kappa_J=35):
    """B2CRoI score with local current metadata removed from priority scoring."""
    mu, var = bench.forecast_stats(xh, a, ar, h=h)
    pvio = bench.empirical_safety_prob(mu, var)
    psucc = bench.predict_success(pi_bad, kind)
    n = len(a)
    if q_f is None:
        q_f = np.zeros(n)

    # Gateway-only proxy: evaluate candidate refreshes using predicted state mu,
    # not current local metadata xt. Packet outcomes and true refresh updates are
    # still handled by the simulator after scheduling.
    cur, _, _ = bench.loss_terms(mu, xh)
    loss_gain = []
    residual_after = []
    max_residual_after = []
    cur_res = ((mu - xh) / bench.RANGE) ** 2
    for i in range(n):
        cand = xh.copy()
        cand[i] = mu[i]
        nxt, _, _ = bench.loss_terms(mu, cand)
        loss_gain.append(float(np.sum(cur - nxt)))
        res = ((mu - cand) / bench.RANGE) ** 2
        residual_after.append(float(np.mean(res)))
        max_residual_after.append(float(np.max(res)))
    loss_gain = np.asarray(loss_gain)
    residual_after = np.asarray(residual_after)
    max_residual_after = np.asarray(max_residual_after)

    loss_norm = bench._norm01(loss_gain)
    innovation = bench._norm01(cur_res)
    pred_risk = bench._norm01(np.clip(pvio * bench.risk(mu) * bench.mismatch(xh, mu), 0, 0.35))
    safety_now = pred_risk

    if total_choices <= 0:
        share = np.ones(n) / n
    else:
        share = counts / max(total_choices, 1)
    target = 1.0 / n
    service_gap = np.maximum(0.0, target - share) / max(target, 1e-12)
    age_pressure = np.clip(np.maximum(0, a - n) / max(n, 1), 0, 2)
    debt_raw = service_gap + 0.25 * age_pressure
    debt_norm = bench._norm01(debt_raw)

    best = float(np.max(loss_gain))
    gap = np.maximum(0.0, best - loss_gain)
    near_best = np.exp(-gap / (abs(best) + 1e-9 + 0.05))
    severe_debt = 1 / (1 + np.exp(-5 * (debt_norm - 0.72)))
    fair_now = bench.jain(counts + 1e-9)
    global_fair_alarm = 1 / (1 + np.exp(kappa_J * (fair_now - tau_J)))
    local_floor_alarm = float(np.max(severe_debt))
    switch = float((fair_now < tau_J) or (float(np.max(debt_norm)) > 0.72))
    mode = switch * np.clip(max(global_fair_alarm, local_floor_alarm), 0, 1)

    reliability = np.clip(psucc, 0.25, 1.0)
    rmse_cost = 0.70 * bench._norm01(residual_after) + 0.30 * bench._norm01(max_residual_after)
    rmse_relief = bench._norm01(cur_res)
    score = reliability * (0.66 * loss_norm + 0.14 * innovation + 0.10 * safety_now + 0.07 * pred_risk)
    score += 0.05 * rmse_relief - 0.05 * rmse_cost
    fairness_gate = np.maximum(near_best, severe_debt)
    constraint_bonus = fairness_gate * (0.18 * debt_norm + 0.08 * np.log1p(np.maximum(q_f, 0.0)) * debt_norm)
    score += mode * constraint_bonus
    score -= mode * 0.04 * rmse_cost
    return score


def choose_metadata_free(policy, krel, xt, xh, a, bw, ar, kind, pi_bad, dual_f, dual_s,
                         counts, total_choices, q_s=0.0, q_f=None, q_e=0.0,
                         tau_J=0.985, kappa_J=35):
    if policy != "b2croi_v8q":
        return bench.choose(policy, krel, xt, xh, a, bw, ar, kind, pi_bad, dual_f, dual_s,
                            counts, total_choices, q_s=q_s, q_f=q_f, q_e=q_e,
                            tau_J=tau_J, kappa_J=kappa_J)
    score = metadata_free_score(xh, xt, a, ar, pi_bad, dual_f, dual_s, kind, counts,
                                total_choices, q_s=q_s, q_f=q_f, q_e=q_e,
                                tau_J=tau_J, kappa_J=kappa_J)
    return list(np.argsort(score)[::-1][:bw])


def ci95(x):
    x = np.asarray(x, float)
    return float(1.96 * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0


def summarize(raw):
    rows = []
    for (net, policy), g in raw.groupby(["network", "policy"]):
        rec = {"network": net, "policy": policy, "n_windows": len(g)}
        for m in ["rmse_mean", "loss_mean", "missed_violation_pct", "choice_fairness", "avg_aoi"]:
            rec[m] = float(g[m].mean())
            rec[m + "_ci95"] = ci95(g[m].to_numpy())
        rows.append(rec)
    return pd.DataFrame(rows)


def latex_table(summary):
    labels = {
        "b2croi_edge_metadata": "Edge metadata",
        "b2croi_gateway_forecast": "Gateway forecast",
        "error_trigger": "Error-trigger",
        "cvoi_sf": "CVoI-SF",
        "oracle": "Upper-ref.",
    }
    order = ["oracle", "b2croi_edge_metadata", "b2croi_gateway_forecast", "cvoi_sf", "error_trigger"]
    netname = {"burst": "Burst", "severe_burst": "Severe burst"}
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Metadata ablation for B2CRoI-H(Q). Edge metadata uses local decision-time risk/innovation signals before the bandwidth-limited full refresh; gateway forecast removes that metadata and scores candidates from AR(1) forecasts only. Lower RMSE, loss, and missed violations are better; higher Jain fairness is better.}",
        r"\label{tab:metadata_ablation}",
        r"\resizebox{0.94\textwidth}{!}{%",
        r"\begin{tabular}{llrrrr}",
        r"\hline",
        r"Network & Variant & RMSE & Loss & Missed (\%) & Jain \\",
        r"\hline",
    ]
    for net in ["burst", "severe_burst"]:
        df = summary[summary.network == net].set_index("policy")
        for pol in order:
            r = df.loc[pol]
            lines.append(f"{netname[net]} & {labels[pol]} & {r.rmse_mean:.4f} & {r.loss_mean:.4f} & {r.missed_violation_pct:.3f} & {r.choice_fairness:.4f} " + r"\\")
        lines.append(r"\hline")
    lines += [r"\end{tabular}%", r"}", r"\end{table}", ""]
    return "\n".join(lines)


def main():
    args = bench.parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    full = data_root / "Full Data Set.csv"
    data = bench.load_numeric_panel(full, label="metadata ablation")
    window = 7 * 24 * 12
    ar = bench.fit_ar1(data[:window])
    alpha, beta, sigma = ar
    train = data[:window]
    bench.set_emp_residuals(train[1:] - (train[:-1] * alpha + beta))
    starts = bench.select_starts(len(data), window, args.n_windows)
    rows = []
    old_choose = bench.choose
    try:
        for net in bench.NETWORKS:
            for wi, st in enumerate(starts):
                # Main edge-assisted variant and selected context baselines.
                bench.choose = old_choose
                for pol in ["oracle", "b2croi_v8q", "cvoi_sf", "error_trigger"]:
                    rec = bench.run(data, pol, net, 2026 + wi, st, st + window, ar, bw=1)
                    rec["policy"] = "b2croi_edge_metadata" if pol == "b2croi_v8q" else pol
                    rows.append(rec)
                # Metadata-free ablation.
                bench.choose = choose_metadata_free
                rec = bench.run(data, "b2croi_v8q", net, 2026 + wi, st, st + window, ar, bw=1)
                rec["policy"] = "b2croi_gateway_forecast"
                rows.append(rec)
    finally:
        bench.choose = old_choose

    raw = pd.DataFrame(rows)
    summary = summarize(raw)
    raw.to_csv(OUT / "b2croi_metadata_ablation_raw.csv", index=False)
    summary.to_csv(OUT / "public_table_metadata_ablation.csv", index=False)
    tex = latex_table(summary)
    for path in TABLES:
        path.write_text(tex)
    print(summary.sort_values(["network", "loss_mean"]).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("wrote", OUT / "public_table_metadata_ablation.csv")


if __name__ == "__main__":
    main()
