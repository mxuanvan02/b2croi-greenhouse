#!/usr/bin/env python3
"""N/heterogeneity stress test for B2CRoI.

Creates synthetic multi-loop panels from real greenhouse temperature dynamics by
using shifted/scaled/noisy copies of the measured zones. This is not a substitute
for a second dataset; it tests whether the scheduling effect is structurally tied
to N=3 or persists under larger heterogeneous loop sets.
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import argparse
import os
import numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('b2', ROOT/'scripts/b2croi_v8q_benchmark.py')
b2=importlib.util.module_from_spec(spec); spec.loader.exec_module(b2)
OUT=ROOT/'data/processed'; OUT.mkdir(parents=True,exist_ok=True)
POLICIES=['error_trigger','generic_voi','ar1_growth_voi','cvoi_sf','b2croi_v8q','oracle']

def make_panel(base, N, hetero, seed):
    rng=np.random.default_rng(seed)
    cols=[]
    base=base.copy()
    for i in range(N):
        src=base[:, i % base.shape[1]]
        shift=(i*37 + seed) % 600
        x=np.roll(src, shift)
        if hetero=='low':
            scale=rng.normal(1.0,0.015); offset=rng.normal(0,0.15); noise=0.03
        elif hetero=='medium':
            scale=rng.normal(1.0,0.04); offset=rng.normal(0,0.35); noise=0.07
        elif hetero=='high':
            scale=rng.normal(1.0,0.08); offset=rng.normal(0,0.65); noise=0.12
        else: raise ValueError(hetero)
        # smooth perturbation to mimic different thermal inertia/exposure
        eps=rng.normal(0,noise,len(x))
        y=scale*x+offset+eps
        # mild low-pass for some loops, preserving realistic range
        if i % 3 == 1:
            for k in range(1,len(y)): y[k]=0.92*y[k-1]+0.08*y[k]
        elif i % 3 == 2:
            for k in range(1,len(y)): y[k]=0.75*y[k-1]+0.25*y[k]
        cols.append(y)
    return np.vstack(cols).T

def fit_ar1(data): return b2.fit_ar1(data)

def run_policy(data,policy,network,seed,start,end,ar,bw):
    # Reuse b2 internals, which are dimension-generic except fixed safety funcs.
    return b2.run(data,policy,network,seed,start,end,ar,bw=bw)

def ci95(x):
    x=np.asarray(x,float); return float(1.96*x.std(ddof=1)/np.sqrt(len(x))) if len(x)>1 else 0.0

def parse_args():
    ap=argparse.ArgumentParser(description='N/heterogeneity stress test for B2CRoI-H(Q).')
    ap.add_argument('--n-windows', type=int, default=int(os.getenv('B2CROI_STRESS_N_WINDOWS','6')), help='Number of weekly windows per stress panel. Use -1 for all available windows. Default: 6.')
    return ap.parse_args()

def main():
    args=parse_args()
    base=b2.load_numeric_panel(label='N/heterogeneity stress test')
    window=7*24*12
    rows=[]
    for N in [3,6,9,12]:
      for hetero in ['low','medium','high']:
        panel=make_panel(base,N,hetero,seed=100+N)
        ar=fit_ar1(panel[:window])
        bw=max(1, N//3)  # scarce but scales with panel size
        starts=b2.select_starts(len(panel), window, args.n_windows)
        print(f'[N={N}, heterogeneity={hetero}] Evaluation windows: {len(starts)} (n_windows={args.n_windows}).')
        for net in b2.NETWORKS:
          for wi,st in enumerate(starts):
            for pol in POLICIES:
              r=run_policy(panel,pol,net,3000+wi,st,st+window,ar,bw=bw)
              r.update(N=N,heterogeneity=hetero,bw=bw)
              rows.append(r)
    raw=pd.DataFrame(rows); raw.to_csv(OUT/'b2croi_v8q_stress_n_raw.csv',index=False)
    metrics=['rmse_mean','loss_mean','missed_violation_pct','choice_fairness','avg_aoi']
    summ=[]
    for keys,sub in raw.groupby(['N','heterogeneity','bw','network','policy']):
        rec=dict(zip(['N','heterogeneity','bw','network','policy'],keys)); rec['n_windows']=len(sub)
        for m in metrics:
            rec[m+'_mean']=float(sub[m].mean()); rec[m+'_ci95']=ci95(sub[m])
        summ.append(rec)
    summ=pd.DataFrame(summ); summ.to_csv(OUT/'b2croi_v8q_stress_n_summary.csv',index=False)
    # paired b2croi vs key baselines
    pairs=[]
    for (N,het,bw,net),grp in raw.groupby(['N','heterogeneity','bw','network']):
        p=grp[grp.policy=='b2croi_v8q'].sort_values('window_start')
        for basepol in ['error_trigger','generic_voi','cvoi_sf','oracle']:
            b=grp[grp.policy==basepol].sort_values('window_start')
            rec={'N':N,'heterogeneity':het,'bw':bw,'network':net,'baseline':basepol,'n_windows':len(p)}
            for m in metrics:
                d=p[m].to_numpy()-b[m].to_numpy()
                rec[m+'_delta_mean']=float(d.mean()); rec[m+'_delta_ci95']=ci95(d)
            pairs.append(rec)
    paired=pd.DataFrame(pairs); paired.to_csv(OUT/'b2croi_v8q_stress_n_paired.csv',index=False)
    print('Summary rows', len(summ), 'paired rows', len(paired))
    show=paired[paired.baseline=='error_trigger'][['N','heterogeneity','network','loss_mean_delta_mean','missed_violation_pct_delta_mean','rmse_mean_delta_mean','choice_fairness_delta_mean']]
    print(show.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
if __name__=='__main__': main()
