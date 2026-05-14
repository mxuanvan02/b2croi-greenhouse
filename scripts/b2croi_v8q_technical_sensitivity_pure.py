#!/usr/bin/env python3
"""Pure-Python technical sensitivity checks for B2CRoI-H(Q).

This script avoids third-party dependencies so it can run in minimal review/build
environments. It is intentionally smaller than the main benchmark: it compares
B2CRoI-H(Q) with the error-triggered baseline under variations of safety-loss
weights and burst-channel severity on paired weekly windows.
"""
from __future__ import annotations

import csv
import math
import random
import statistics as stats
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "Full Data Set.csv"
OUT = ROOT / "data" / "processed"
TABLES = [ROOT / "manuscript" / "compag" / "assets" / "tables", ROOT / "manuscript" / "standalone" / "assets" / "tables"]
SAFE_MIN, SAFE_MAX = 22.0, 30.0
RANGE = SAFE_MAX - SAFE_MIN
COLS = ["Temperature Front", "Temperature Middle", "Temperature Back"]
BASE_CHANNELS = {
    "burst": dict(good_loss=0.06, p_gb=0.035, bad_loss=0.65, p_bg=0.22),
    "severe_burst": dict(good_loss=0.08, p_gb=0.055, bad_loss=0.82, p_bg=0.15),
}
SETTINGS = [
    ("Nominal", 6.0, 1.5, 1.00),
    ("Missed weight 4", 4.0, 1.5, 1.00),
    ("Missed weight 8", 8.0, 1.5, 1.00),
    ("False-alarm weight 1", 6.0, 1.0, 1.00),
    ("False-alarm weight 2", 6.0, 2.0, 1.00),
    ("Channel loss x0.8", 6.0, 1.5, 0.80),
    ("Channel loss x1.2", 6.0, 1.5, 1.20),
]


def load_panel():
    with RAW.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = []
        for r in reader:
            try:
                rows.append([float(r[c]) for c in COLS])
            except Exception:
                continue
    return rows


def fit_ar1(train):
    alphas, betas, residuals = [], [], []
    for j in range(len(train[0])):
        xs = [row[j] for row in train[:-1]]
        ys = [row[j] for row in train[1:]]
        mx, my = stats.fmean(xs), stats.fmean(ys)
        var = sum((x - mx) ** 2 for x in xs) or 1e-12
        cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        a = cov / var
        b = my - a * mx
        alphas.append(a)
        betas.append(b)
        residuals.append([y - (a * x + b) for x, y in zip(xs, ys)])
    return alphas, betas, residuals


def forecast(xh, age, ar, h=4):
    alphas, betas, _ = ar
    mu = list(xh)
    for j in range(len(mu)):
        for _ in range(max(1, age[j] + h)):
            mu[j] = alphas[j] * mu[j] + betas[j]
    return mu


def empirical_prob(mu, ar):
    _, _, residuals = ar
    probs = []
    for j, m in enumerate(mu):
        rs = residuals[j]
        bad = sum(1 for r in rs if m + r < SAFE_MIN or m + r > SAFE_MAX)
        probs.append(bad / max(len(rs), 1))
    return probs


def norm(vals):
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return [0.0 for _ in vals]
    return [(v - lo) / (hi - lo) for v in vals]


def jain(counts):
    s = sum(counts)
    q = sum(c * c for c in counts)
    return (s * s) / (len(counts) * q + 1e-12) if s > 0 else 1.0


def violation(x):
    return x < SAFE_MIN or x > SAFE_MAX


def risk_scalar(x, delta=2.0):
    margin = min(x - SAFE_MIN, SAFE_MAX - x)
    exceed = max(SAFE_MIN - x, x - SAFE_MAX, 0.0)
    return max(0.0, min(1.0, 1.0 - margin / delta)) + exceed / (delta + exceed) if exceed > 0 else max(0.0, min(1.0, 1.0 - margin / delta))


def channel_params(kind, scale):
    c = dict(BASE_CHANNELS[kind])
    c["good_loss"] = min(0.95, c["good_loss"] * scale)
    c["bad_loss"] = min(0.98, c["bad_loss"] * scale)
    return c


def channel_step(c, rng, bad):
    if bad:
        ok = rng.random() > c["bad_loss"]
        if rng.random() < c["p_bg"]:
            bad = False
    else:
        ok = rng.random() > c["good_loss"]
        if rng.random() < c["p_gb"]:
            bad = True
    return ok, bad


def update_belief(pi_bad, observed_ok, c):
    pi_pred = pi_bad * (1 - c["p_bg"]) + (1 - pi_bad) * c["p_gb"]
    like_bad = (1 - c["bad_loss"]) if observed_ok else c["bad_loss"]
    like_good = (1 - c["good_loss"]) if observed_ok else c["good_loss"]
    den = pi_pred * like_bad + (1 - pi_pred) * like_good
    return (pi_pred * like_bad) / max(den, 1e-12)


def predict_success(pi_bad, c):
    pi_pred = pi_bad * (1 - c["p_bg"]) + (1 - pi_bad) * c["p_gb"]
    return (1 - pi_pred) * (1 - c["good_loss"]) + pi_pred * (1 - c["bad_loss"])


def loss_vec(xt, xh, missed_w, false_w):
    vals, missed = [], []
    for x, h in zip(xt, xh):
        mi = violation(x) and not violation(h)
        fa = (not violation(x)) and violation(h)
        vals.append(((h - x) / RANGE) ** 2 + missed_w * mi + false_w * fa)
        missed.append(mi)
    return vals, missed


def choose_error(xt, xh):
    vals = [abs(x - h) / RANGE for x, h in zip(xt, xh)]
    return max(range(len(vals)), key=lambda i: vals[i])


def choose_b2(xt, xh, age, counts, total, pi_bad, c, ar, missed_w, false_w):
    n = len(xt)
    mu = forecast(xh, age, ar)
    pvio = empirical_prob(mu, ar)
    cur_loss, _ = loss_vec(xt, xh, missed_w, false_w)
    cur_res = [((h - x) / RANGE) ** 2 for x, h in zip(xt, xh)]
    gains, residual_after, max_after = [], [], []
    for i in range(n):
        cand = list(xh)
        # Matched to the benchmark oracle-style selector used by the existing
        # simulation scripts: evaluate the loss reduction that would occur if
        # the candidate loop were successfully refreshed at this step.
        cand[i] = xt[i]
        nxt_loss, _ = loss_vec(xt, cand, missed_w, false_w)
        gains.append(sum(a - b for a, b in zip(cur_loss, nxt_loss)))
        res = [((h - x) / RANGE) ** 2 for x, h in zip(xt, cand)]
        residual_after.append(stats.fmean(res))
        max_after.append(max(res))
    loss_norm = norm(gains)
    innovation = norm(cur_res)
    pred_risk = norm([min(0.35, p * risk_scalar(m) * abs(h - m) / RANGE) for p, m, h in zip(pvio, mu, xh)])
    safety_now = norm([risk_scalar(x) * abs(h - x) / RANGE for x, h in zip(xt, xh)])
    if total <= 0:
        share = [1 / n] * n
    else:
        share = [cc / max(total, 1) for cc in counts]
    target = 1 / n
    debt_raw = [max(0.0, target - sh) / target + 0.25 * max(0, a - n) / n for sh, a in zip(share, age)]
    debt_norm = norm(debt_raw)
    best = max(gains)
    near_best = [math.exp(-max(0.0, best - g) / (abs(best) + 1e-9 + 0.05)) for g in gains]
    severe_debt = [1 / (1 + math.exp(-5 * (d - 0.72))) for d in debt_norm]
    fair_now = jain([cc + 1e-9 for cc in counts])
    global_alarm = 1 / (1 + math.exp(35 * (fair_now - 0.985)))
    local_alarm = max(severe_debt)
    switch = 1.0 if (fair_now < 0.985 or max(debt_norm) > 0.72) else 0.0
    mode = switch * max(global_alarm, local_alarm)
    reliability = max(0.25, min(1.0, predict_success(pi_bad, c)))
    rmse_cost = [0.70 * a + 0.30 * b for a, b in zip(norm(residual_after), norm(max_after))]
    rmse_relief = norm(cur_res)
    scores = []
    for i in range(n):
        score = reliability * (0.66 * loss_norm[i] + 0.14 * innovation[i] + 0.10 * safety_now[i] + 0.07 * pred_risk[i])
        score += 0.05 * rmse_relief[i] - 0.05 * rmse_cost[i]
        fairness_gate = max(near_best[i], severe_debt[i])
        score += mode * fairness_gate * (0.18 * debt_norm[i])
        score -= mode * 0.04 * rmse_cost[i]
        scores.append(score)
    return max(range(n), key=lambda i: scores[i])


def run(data, start, end, policy, network, seed, missed_w, false_w, channel_scale, ar):
    rng = random.Random(seed)
    c = channel_params(network, channel_scale)
    pi_bad = c["p_gb"] / (c["p_gb"] + c["p_bg"])
    bad = False
    xh = list(data[start])
    age = [0] * len(xh)
    counts = [0] * len(xh)
    total = 0
    err_sq, losses, missed_all, ages = [], [], [], []
    for k in range(start + 1, end):
        xt = data[k]
        if policy == "error_trigger":
            idx = choose_error(xt, xh)
        else:
            idx = choose_b2(xt, xh, age, counts, total, pi_bad, c, ar, missed_w, false_w)
        age = [a + 1 for a in age]
        counts[idx] += 1
        total += 1
        ok, bad = channel_step(c, rng, bad)
        pi_bad = update_belief(pi_bad, ok, c)
        if ok:
            xh[idx] = xt[idx]
            age[idx] = 0
        lv, mi = loss_vec(xt, xh, missed_w, false_w)
        losses.extend(lv)
        missed_all.extend(mi)
        err_sq.extend([(h - x) ** 2 for x, h in zip(xt, xh)])
        ages.extend(age)
    return {
        "rmse": math.sqrt(stats.fmean(err_sq)),
        "loss": stats.fmean(losses),
        "missed_pct": 100 * stats.fmean([1.0 if x else 0.0 for x in missed_all]),
        "fairness": jain(counts),
        "avg_aoi": stats.fmean(ages),
    }


def ci95(vals):
    return 1.96 * stats.stdev(vals) / math.sqrt(len(vals)) if len(vals) > 1 else 0.0


def main():
    data = load_panel()
    window = 7 * 24 * 12
    ar = fit_ar1(data[:window])
    starts = list(range(window, len(data) - window, window))[:6]
    rows = []
    raw_rows = []
    for setting, mw, fw, scale in SETTINGS:
        for net in ("burst", "severe_burst"):
            deltas = {"rmse": [], "loss": [], "missed_pct": [], "fairness": [], "avg_aoi": []}
            for wi, st in enumerate(starts):
                base = run(data, st, st + window, "error_trigger", net, 8000 + wi, mw, fw, scale, ar)
                prop = run(data, st, st + window, "b2croi_v8q", net, 8000 + wi, mw, fw, scale, ar)
                rec = {"setting": setting, "network": net, "window": wi}
                for m in deltas:
                    d = prop[m] - base[m]
                    deltas[m].append(d)
                    rec[m + "_delta"] = d
                raw_rows.append(rec)
            rows.append({
                "Setting": setting,
                "Network": net.replace("_", " "),
                "RMSE delta": stats.fmean(deltas["rmse"]),
                "RMSE CI95": ci95(deltas["rmse"]),
                "Loss delta": stats.fmean(deltas["loss"]),
                "Loss CI95": ci95(deltas["loss"]),
                "Missed delta pp": stats.fmean(deltas["missed_pct"]),
                "Missed CI95 pp": ci95(deltas["missed_pct"]),
                "Fairness delta": stats.fmean(deltas["fairness"]),
                "Fairness CI95": ci95(deltas["fairness"]),
            })
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "b2croi_v8q_technical_sensitivity_raw.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(raw_rows[0].keys()))
        w.writeheader(); w.writerows(raw_rows)
    with (OUT / "public_table_technical_sensitivity.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    latex = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Technical sensitivity of B2CRoI-H(Q) versus error-triggered scheduling under loss-weight and channel-severity variations. Values are paired mean deltas over six weekly windows; negative RMSE/loss/missed deltas and positive fairness deltas favor B2CRoI-H(Q).}",
        r"\label{tab:technical_sensitivity}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrrr}",
        r"\hline",
        r"Setting & Network & RMSE $\Delta$ & Loss $\Delta$ & Missed $\Delta$ (pp) & Fairness $\Delta$ \\",
        r"\hline",
    ]
    for r in rows:
        row = (
            f"{r['Setting']} & {r['Network']} & "
            f"{r['RMSE delta']:.4f}$\\pm${r['RMSE CI95']:.4f} & "
            f"{r['Loss delta']:.4f}$\\pm${r['Loss CI95']:.4f} & "
            f"{r['Missed delta pp']:.3f}$\\pm${r['Missed CI95 pp']:.3f} & "
            f"{r['Fairness delta']:.4f}$\\pm${r['Fairness CI95']:.4f} "
            + r"\\"
        )
        latex.append(row)
    latex += [r"\hline", r"\end{tabular}%", r"}", r"\end{table*}", ""]
    for d in TABLES:
        d.mkdir(parents=True, exist_ok=True)
        (d / "public_table_technical_sensitivity_latex.tex").write_text("\n".join(latex))
    print("wrote", OUT / "public_table_technical_sensitivity.csv")


if __name__ == "__main__":
    main()
