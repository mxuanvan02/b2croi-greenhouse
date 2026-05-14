#!/usr/bin/env python3
"""Report global/local fairness-alarm activation rates for B2CRoI-H(Q)."""
from __future__ import annotations
import numpy as np
import pandas as pd
import b2croi_hq_benchmark as b8

OUT=b8.OUT

def run_alarm(data, network='burst', seed=2026, start=None, ar=None, bw=1):
    window=7*24*12
    if start is None: start=window
    rng=np.random.default_rng(seed)
    bad=False; pi_bad=b8.CHANNELS[network]['p_gb']/(b8.CHANNELS[network]['p_gb']+b8.CHANNELS[network]['p_bg'])
    xh=data[start].copy(); a=np.zeros(data.shape[1],dtype=int)
    counts=np.zeros(data.shape[1],dtype=float); total_choices=0
    dual_f=0.35; dual_s=0.20; q_s=0.0; q_f=np.zeros(data.shape[1]); q_e=0.0
    rows=[]
    for k in range(start+1,start+window):
        xt=data[k]
        # Reproduce the score terms needed for alarm logging.
        n=len(a)
        if total_choices <= 0:
            share=np.ones(n)/n
        else:
            share=counts/max(total_choices,1)
        target=1.0/n
        service_gap=np.maximum(0.0,target-share)/max(target,1e-12)
        age_pressure=np.clip(np.maximum(0,a-n)/max(n,1),0,2)
        debt_raw=service_gap + 0.25*age_pressure
        debt_norm=b8._norm01(debt_raw)
        severe_debt=1/(1+np.exp(-5*(debt_norm-0.72)))
        fair_now=b8.jain(counts+1e-9)
        global_alarm=1/(1+np.exp(35*(fair_now-0.985)))
        local_alarm=float(np.max(severe_debt))
        lam=max(global_alarm,local_alarm)
        sc=b8.b2croi_score(xh,xt,a,ar,pi_bad,dual_f,dual_s,network,counts,total_choices,q_s=q_s,q_f=q_f,q_e=q_e)
        idx=list(np.argsort(sc)[::-1][:bw])
        rows.append(dict(network=network,seed=seed,window_start=start,step=k-start,
            jain=fair_now, global_alarm=global_alarm, local_alarm=local_alarm, lambda_F=lam,
            global_gt_05=global_alarm>0.5, local_gt_05=local_alarm>0.5,
            global_dominates=(global_alarm>local_alarm), local_dominates=(local_alarm>=global_alarm)))
        a+=1
        for i in idx:
            counts[i]+=1; total_choices+=1
            ok,bad=b8.channel_step(network,rng,bad)
            pi_bad=b8.update_bad_belief(pi_bad,ok,network)
            if ok:
                xh[i]=xt[i]; a[i]=0
        lt,mi,fa=b8.loss_terms(xt,xh)
        fair_now=b8.jain(counts)
        q_s=max(0.0, q_s + float(mi.mean()) - 0.006)
        target_share=1.0/data.shape[1]
        served=np.zeros(data.shape[1]); served[idx]=1.0/max(len(idx),1)
        q_f=np.maximum(0.0, q_f + target_share - served)
        q_e=max(0.0, q_e + float(np.mean(((xh-xt)/b8.RANGE)**2)) - 0.006)
        dual_f=np.clip(dual_f + 0.02*(0.985-fair_now), 0.05, 1.2)
        dual_s=np.clip(dual_s + 0.015*((1.0 if bool(mi.any()) else 0.0)-0.02), 0.05, 1.0)
    return pd.DataFrame(rows)

def main():
    df=pd.read_csv(b8.FULL)
    numeric=df[b8.COLS].apply(pd.to_numeric,errors='coerce')
    before=len(numeric); clean=numeric.dropna(); dropped=before-len(clean)
    print(f'[alarm activation] Dropped {dropped} rows during numeric coercion/dropna out of {before} rows ({(100*dropped/before if before else 0):.3f}%).')
    data=clean.to_numpy(float)
    window=7*24*12
    ar=b8.fit_ar1(data[:window])
    alpha,beta,sigma=ar; train=data[:window]
    b8.set_emp_residuals(train[1:]-(train[:-1]*alpha+beta))
    starts=list(range(window,len(data)-window,window))[:3]
    raws=[]
    for net in b8.NETWORKS:
        for wi,st in enumerate(starts):
            raws.append(run_alarm(data,net,2026+wi,st,ar,bw=1))
    raw=pd.concat(raws,ignore_index=True)
    raw.to_csv(OUT/'b2croi_hq_alarm_activation_raw.csv',index=False)
    rows=[]
    for (net,st),g in raw.groupby(['network','window_start']):
        rows.append(dict(network=net,window_start=st,n_steps=len(g),
            global_gt_05_pct=100*g.global_gt_05.mean(), local_gt_05_pct=100*g.local_gt_05.mean(),
            global_dominates_pct=100*g.global_dominates.mean(), local_dominates_pct=100*g.local_dominates.mean(),
            mean_global_alarm=g.global_alarm.mean(), mean_local_alarm=g.local_alarm.mean(), mean_lambda_F=g.lambda_F.mean()))
    summ=pd.DataFrame(rows)
    summ.to_csv(OUT/'b2croi_hq_alarm_activation_summary.csv',index=False)
    print(summ.to_string(index=False,float_format=lambda x:f'{x:.4f}'))

if __name__=='__main__': main()
