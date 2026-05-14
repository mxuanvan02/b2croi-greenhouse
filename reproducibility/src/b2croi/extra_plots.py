#!/usr/bin/env python3
"""Generate optional diagnostic decisive plots from processed B2CRoI-H(Q) CSVs.

Outputs PDF+PNG to:
- reproducibility/results/figures/
- assets/figures/

No values are hand-edited; all plotted values come from CSV files in data/processed.
"""
from __future__ import annotations
import csv
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "data" / "processed"
OUTS = [
    ROOT / "reproducibility" / "results" / "figures",
    ROOT / "study" / "" / "assets" / "figures",
    ROOT / "study" / "standalone" / "assets" / "figures",
]


def rows(name):
    with open(SRC / name, newline="") as f:
        return list(csv.DictReader(f))


def save(fig, name):
    for out in OUTS:
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
        fig.savefig(out / f"{name}.png", dpi=240, bbox_inches="tight")


def f(x): return float(x)

def label_policy(p):
    return {
        "round_robin":"Round-robin", "max_aoi":"Max-AoI", "channel_aware_rr":"Burst-aware cyclic",
        "error_trigger":"Error-trigger", "generic_voi":"Generic VoI", "cvoi_sf":"CVoI-SF",
        "b2croi_hq":"B2CRoI-H(Q)", "oracle":"Upper-ref."
    }.get(p,p)


def plot_pareto(plt):
    data=rows("b2croi_hq_summary.csv")
    fig,ax=plt.subplots(figsize=(7.2,4.6))
    policies=["round_robin","max_aoi","channel_aware_rr","error_trigger","generic_voi","cvoi_sf","b2croi_hq","oracle"]
    markers={"burst":"o","severe_burst":"s"}
    colors={p:c for p,c in zip(policies,["#6b8e23","#d9802e","#7f7f7f","#b55d60","#9467bd","#17becf","#1f77b4","#2ca02c"])}
    offsets={
        "round_robin":(5,6), "max_aoi":(5,-10), "channel_aware_rr":(5,5), "error_trigger":(5,-12),
        "generic_voi":(5,6), "cvoi_sf":(5,-10), "b2croi_hq":(5,7), "oracle":(5,-12),
    }
    for r in data:
        pol=r["policy"]
        if pol not in policies: continue
        x=f(r["choice_fairness_mean"]); y=f(r["loss_mean_mean"])
        ax.scatter(x,y,s=64,marker=markers.get(r["network"],"o"),color=colors[pol],alpha=.9,
                   edgecolor="white",linewidth=.6)
        dx,dy=offsets[pol]
        net="B" if r["network"]=="burst" else "S"
        ax.annotate(f"{label_policy(pol)} ({net})", (x,y), xytext=(dx,dy), textcoords="offset points", fontsize=6.3)
    ax.scatter([],[],marker="o",color="black",label="Burst")
    ax.scatter([],[],marker="s",color="black",label="Severe burst")
    ax.legend(frameon=False,fontsize=7,loc="upper right")
    ax.set_xlabel("Jain fairness (higher is better)")
    ax.set_ylabel("Safety loss (lower is better)")
    ax.set_title("Loss--fairness Pareto context")
    ax.grid(True,alpha=.25)
    save(fig,"b2croi_hq_loss_fairness_pareto")


def plot_alarm(plt):
    data=rows("b2croi_hq_alarm_activation_summary.csv")
    agg=defaultdict(lambda:[0,0,0])
    for r in data:
        a=agg[r["network"]]; a[0]+=f(r["global_dominates_pct"]); a[1]+=f(r["local_dominates_pct"]); a[2]+=1
    nets=list(agg)
    x=range(len(nets)); g=[agg[n][0]/agg[n][2] for n in nets]; l=[agg[n][1]/agg[n][2] for n in nets]
    fig,ax=plt.subplots(figsize=(5.8,3.8))
    ax.bar(x,g,label="Global dominates",color="#7aa6c2")
    ax.bar(x,l,bottom=g,label="Local dominates",color="#d98c5f")
    ax.set_xticks(list(x), [n.replace("_"," ").title() for n in nets])
    ax.set_ylabel("Scheduling steps (%)")
    ax.set_ylim(0,100)
    ax.set_title("Alarm branch dominance")
    ax.legend(frameon=False,fontsize=8)
    save(fig,"b2croi_hq_alarm_activation_bars")


def plot_robustness(plt):
    """Robustness heatmap: B2CRoI-H(Q) loss minus error-trigger loss.

    Prefer paired deltas when the paired file has proposed/baseline columns.
    Fall back to computing deltas from summary means when schema differs.
    """
    data=[]
    try:
        paired=rows("b2croi_hq_stress_n_paired.csv")
        data=[r for r in paired if r.get("proposed")=="b2croi_hq" and r.get("baseline")=="error_trigger"]
    except FileNotFoundError:
        data=[]
    hs=["low","medium","high"]
    agg=defaultdict(list)
    if data:
        Ns=sorted({int(f(r["N"])) for r in data})
        for r in data:
            agg[(int(f(r["N"])),r["heterogeneity"])].append(f(r["loss_mean_delta_mean"]))
    else:
        summary=rows("b2croi_hq_stress_n_summary.csv")
        grouped=defaultdict(dict)
        for r in summary:
            key=(int(f(r["N"])), r["heterogeneity"], r.get("bw",""), r["network"])
            grouped[key][r["policy"]]=f(r["loss_mean_mean"])
        Ns=sorted({k[0] for k in grouped})
        for (N,h,bw,net), vals in grouped.items():
            if "b2croi_hq" in vals and "error_trigger" in vals:
                agg[(N,h)].append(vals["b2croi_hq"]-vals["error_trigger"])
    mat=[]
    for N in Ns:
        row=[]
        for h in hs:
            vals=agg[(N,h)]
            row.append(sum(vals)/len(vals) if vals else float("nan"))
        mat.append(row)
    if not mat or all(all(v != v for v in row) for row in mat):
        print("[WARN] Robustness grid skipped: no compatible stress data")
        return
    fig,ax=plt.subplots(figsize=(5.8,4.0))
    im=ax.imshow(mat,cmap="RdBu_r",aspect="auto")
    ax.set_xticks(range(len(hs)), [h.title() for h in hs]); ax.set_yticks(range(len(Ns)), Ns)
    ax.set_xlabel("Heterogeneity"); ax.set_ylabel("Number of loops N")
    ax.set_title("Robustness: mean loss delta vs error-trigger")
    for i,N in enumerate(Ns):
        for j,h in enumerate(hs):
            v=mat[i][j]
            ax.text(j,i,"NA" if v!=v else f"{v:.3f}",ha="center",va="center",fontsize=8)
    fig.colorbar(im,ax=ax,label="ΔLoss (lower is better)")
    save(fig,"b2croi_hq_robustness_grid")


def plot_runtime(plt):
    data=rows("b2croi_hq_stress_n_summary.csv")
    # runtime not in stress summary; fallback to primary summary if absent.
    if "runtime_us_mean_mean" not in data[0]: data=rows("b2croi_hq_summary.csv")
    pols=["round_robin","max_aoi","error_trigger","generic_voi","cvoi_sf","b2croi_hq","oracle"]
    vals=[]
    for p in pols:
        xs=[f(r["runtime_us_mean_mean"]) for r in data if r.get("policy")==p and "runtime_us_mean_mean" in r]
        vals.append(sum(xs)/len(xs) if xs else 0)
    fig,ax=plt.subplots(figsize=(6.4,3.8))
    ax.bar(range(len(pols)), vals, color="#6b8fb3")
    ax.set_yscale("log")
    ax.set_xticks(range(len(pols)), [label_policy(p) for p in pols], rotation=35, ha="right")
    ax.set_ylabel("Runtime per decision (µs, log scale)")
    ax.set_title("Online scheduling overhead")
    ax.grid(True,axis="y",alpha=.25)
    save(fig,"b2croi_hq_runtime_scalability")


def main():
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as e:
        raise SystemExit("[ERROR] Missing matplotlib. Install with: python3 -m pip install numpy matplotlib") from e
    for fn in [plot_pareto, plot_alarm, plot_robustness, plot_runtime]:
        fn(plt)
    print("[OK] Extra decisive plots generated")

if __name__ == "__main__":
    main()
