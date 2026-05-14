#!/usr/bin/env python3
from pathlib import Path
import importlib.util, numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('b2', ROOT/'scripts/b2croi_v8_ablation.py')
b2=importlib.util.module_from_spec(spec); spec.loader.exec_module(b2)
OUT=ROOT/'data/processed'; OUT.mkdir(exist_ok=True,parents=True)
POLICIES=['error_trigger','generic_voi','cvoi_sf','b2croi_v8','v8_no_mode','v8_no_constraint_bonus','v8_no_rmse_cost','v8_no_reliability','oracle']

def make_panel(base,N,hetero,seed):
    rng=np.random.default_rng(seed); cols=[]
    for i in range(N):
        src=base[:,i%base.shape[1]]; x=np.roll(src,(i*37+seed)%600)
        if hetero=='low': scale=rng.normal(1,.015); offset=rng.normal(0,.15); noise=.03
        elif hetero=='medium': scale=rng.normal(1,.04); offset=rng.normal(0,.35); noise=.07
        else: scale=rng.normal(1,.08); offset=rng.normal(0,.65); noise=.12
        y=scale*x+offset+rng.normal(0,noise,len(x))
        if i%3==1:
            for k in range(1,len(y)): y[k]=.92*y[k-1]+.08*y[k]
        elif i%3==2:
            for k in range(1,len(y)): y[k]=.75*y[k-1]+.25*y[k]
        cols.append(y)
    return np.vstack(cols).T

def ci95(x):
    x=np.asarray(x,float); return float(1.96*x.std(ddof=1)/np.sqrt(len(x))) if len(x)>1 else 0.0

def main():
    df=pd.read_csv(b2.FULL); numeric=df[b2.COLS].apply(pd.to_numeric,errors='coerce')
    before=len(numeric); clean=numeric.dropna(); dropped=before-len(clean)
    print(f'[ablation stress] Dropped {dropped} rows during numeric coercion/dropna out of {before} rows ({(100*dropped/before if before else 0):.3f}%).')
    base=clean.to_numpy(float)
    window=7*24*12; rows=[]
    # Reduced but representative grid to keep ablation tractable.
    for N in [3,6,12]:
      for hetero in ['low','high']:
        panel=make_panel(base,N,hetero,seed=200+N); ar=b2.fit_ar1(panel[:window]); bw=max(1,N//3)
        starts=list(range(window,len(panel)-window,window))[:4]
        for net in b2.NETWORKS:
          for wi,st in enumerate(starts):
            for pol in POLICIES:
              r=b2.run(panel,pol,net,5000+wi,st,st+window,ar,bw=bw); r.update(N=N,heterogeneity=hetero,bw=bw); rows.append(r)
    raw=pd.DataFrame(rows); raw.to_csv(OUT/'b2croi_v8_ablation_stress_raw.csv',index=False)
    metrics=['rmse_mean','loss_mean','missed_violation_pct','choice_fairness']
    summ=[]
    for keys,sub in raw.groupby(['N','heterogeneity','bw','network','policy']):
        rec=dict(zip(['N','heterogeneity','bw','network','policy'],keys)); rec['n_windows']=len(sub)
        for m in metrics: rec[m+'_mean']=float(sub[m].mean()); rec[m+'_ci95']=ci95(sub[m])
        summ.append(rec)
    pd.DataFrame(summ).to_csv(OUT/'b2croi_v8_ablation_stress_summary.csv',index=False)
    pairs=[]
    for (N,het,bw,net),grp in raw.groupby(['N','heterogeneity','bw','network']):
        p=grp[grp.policy=='b2croi_v8'].sort_values('window_start')
        for basepol in [x for x in POLICIES if x!='b2croi_v8']:
            b=grp[grp.policy==basepol].sort_values('window_start')
            rec={'N':N,'heterogeneity':het,'bw':bw,'network':net,'baseline':basepol,'n_windows':len(p)}
            for m in metrics:
                d=p[m].to_numpy()-b[m].to_numpy(); rec[m+'_delta_mean']=float(d.mean()); rec[m+'_delta_ci95']=ci95(d)
            pairs.append(rec)
    paired=pd.DataFrame(pairs); paired.to_csv(OUT/'b2croi_v8_ablation_stress_paired.csv',index=False)
    print('rows',len(raw),'summary',len(summ),'paired',len(paired))
    # Component damage: compare each ablation to full v6 (ablation - full)
    for basepol in ['v8_no_mode','v8_no_constraint_bonus','v8_no_rmse_cost','v8_no_reliability']:
        sub=paired[paired.baseline==basepol]
        print('\nfull v8 minus',basepol)
        print(sub.groupby('N')[['loss_mean_delta_mean','missed_violation_pct_delta_mean','rmse_mean_delta_mean','choice_fairness_delta_mean']].mean().to_string(float_format=lambda x:f'{x:.4f}'))
if __name__=='__main__': main()
