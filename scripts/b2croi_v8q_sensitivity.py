#!/usr/bin/env python3
"""Sensitivity sweep for B2CRoI-H(Q) mode-switch parameters tau_J and kappa_J."""
from __future__ import annotations
from pathlib import Path
import argparse
import os
import numpy as np
import pandas as pd
import b2croi_v8q_benchmark as b8

OUT = b8.OUT

def run_variant(data, net, tau_J, kappa_J, starts, ar, bw=1):
    rows=[]
    for wi,st in enumerate(starts):
        # Inline copy of b8.run with parameterized tau_J/kappa_J.
        rng=np.random.default_rng(3026+wi)
        bad=False; pi_bad=b8.CHANNELS[net]['p_gb']/(b8.CHANNELS[net]['p_gb']+b8.CHANNELS[net]['p_bg'])
        xh=data[st].copy(); a=np.zeros(data.shape[1],dtype=int)
        counts=np.zeros(data.shape[1],dtype=float); total_choices=0
        dual_f=0.35; dual_s=0.20; q_s=0.0; q_f=np.zeros(data.shape[1]); q_e=0.0
        errs=[]; losses=[]; missed=[]; false=[]; ages=[]; choices=[]; success=[]; fairness_trace=[]
        for k in range(st+1,st+7*24*12):
            xt=data[k]; M=len(xt); bw=min(bw,M)
            sc=b8.b2croi_score(xh,xt,a,ar,pi_bad,dual_f,dual_s,net,counts,total_choices,q_s=q_s,q_f=q_f,q_e=q_e,tau_J=tau_J,kappa_J=kappa_J)
            idx=list(np.argsort(sc)[::-1][:bw])
            a+=1
            for i in idx:
                choices.append(i); counts[i]+=1; total_choices+=1
                ok,bad=b8.channel_step(net,rng,bad); success.append(ok)
                pi_bad=b8.update_bad_belief(pi_bad,ok,net)
                if ok:
                    xh[i]=xt[i]; a[i]=0
            lt,mi,fa=b8.loss_terms(xt,xh)
            fair_now=b8.jain(counts)
            target_share=1.0/data.shape[1]
            served=np.zeros(data.shape[1]); served[idx]=1.0/max(len(idx),1)
            q_s=max(0.0, q_s + float(mi.mean()) - 0.006)
            q_f=np.maximum(0.0, q_f + target_share - served)
            q_e=max(0.0, q_e + float(np.mean(((xh-xt)/b8.RANGE)**2)) - 0.006)
            dual_f=np.clip(dual_f + 0.02*(0.985-fair_now), 0.05, 1.2)
            dual_s=np.clip(dual_s + 0.015*((1.0 if bool(mi.any()) else 0.0)-0.02), 0.05, 1.0)
            errs.append((xh-xt).copy()); losses.append(lt); missed.append(mi); false.append(fa); ages.append(a.copy()); fairness_trace.append(fair_now)
        errs=np.array(errs); losses=np.array(losses); missed=np.array(missed); false=np.array(false); ages=np.array(ages); choices=np.array(choices)
        final_counts=np.array([(choices==i).sum() for i in range(data.shape[1])]) if len(choices) else np.zeros(data.shape[1])
        rows.append(dict(network=net,tau_J=tau_J,kappa_J=kappa_J,seed=3026+wi,window_start=st,
            rmse_mean=float(np.mean(np.sqrt(np.mean(errs**2,axis=0)))), loss_mean=float(losses.mean()),
            missed_violation_pct=float(100*missed.mean()), false_alarm_pct=float(100*false.mean()),
            avg_aoi=float(ages.mean()), max_aoi=float(ages.max()), choice_fairness=b8.jain(final_counts)))
    return rows

def parse_args():
    ap=argparse.ArgumentParser(description='Sensitivity sweep for B2CRoI-H(Q).')
    ap.add_argument('--n-windows', type=int, default=int(os.getenv('B2CROI_SENSITIVITY_N_WINDOWS','6')), help='Number of weekly windows for sensitivity sanity sweep. Use -1 for all available windows. Default: 6.')
    return ap.parse_args()

def main():
    args=parse_args()
    data=b8.load_numeric_panel(label='sensitivity sweep')
    window=7*24*12
    ar=b8.fit_ar1(data[:window])
    alpha,beta,sigma=ar; train=data[:window]
    b8.set_emp_residuals(train[1:]-(train[:-1]*alpha+beta))
    starts=b8.select_starts(len(data), window, args.n_windows)
    print(f'[sensitivity sweep] Evaluation windows: {len(starts)} (n_windows={args.n_windows}).')
    configs=[]
    for tau in [0.970,0.980,0.985,0.990,0.995]: configs.append((tau,35))
    for kap in [10,20,35,50,70]: configs.append((0.985,kap))
    configs=sorted(set(configs))
    rows=[]
    for net in b8.NETWORKS:
        for tau,kap in configs:
            rows.extend(run_variant(data,net,tau,kap,starts,ar,bw=1))
    raw=pd.DataFrame(rows)
    raw.to_csv(OUT/'b2croi_v8q_sensitivity_raw.csv',index=False)
    metrics=['rmse_mean','loss_mean','missed_violation_pct','choice_fairness']
    out=[]
    for (net,tau,kap),g in raw.groupby(['network','tau_J','kappa_J']):
        rec={'network':net,'tau_J':tau,'kappa_J':kap,'n_windows':len(g)}
        for m in metrics:
            rec[m+'_mean']=float(g[m].mean()); rec[m+'_ci95']=b8.ci95(g[m])
        out.append(rec)
    summ=pd.DataFrame(out).sort_values(['network','tau_J','kappa_J'])
    summ.to_csv(OUT/'b2croi_v8q_sensitivity_summary.csv',index=False)
    print(summ.to_string(index=False,float_format=lambda x:f'{x:.4f}'))

if __name__=='__main__': main()
