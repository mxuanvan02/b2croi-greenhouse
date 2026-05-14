#!/usr/bin/env python3
from pathlib import Path
import argparse, os
import importlib.util, shutil
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
EXTERNAL_DATA_ROOT = Path(os.getenv('B2CROI_EXTERNAL_DATA_ROOT', ROOT/'data/external/mendeley_3dw54yhhcc'))
EXTERNAL_DATA_FILE = os.getenv('B2CROI_EXTERNAL_DATA_FILE', 'Greenhouse climate dataset from 26 to 30-01-2019.xlsx')
DATA = EXTERNAL_DATA_ROOT / EXTERNAL_DATA_FILE
OUT=ROOT/'data/processed'; FIG=ROOT/'manuscript/compag/assets/figures'; OUT.mkdir(exist_ok=True,parents=True); FIG.mkdir(exist_ok=True,parents=True)
spec=importlib.util.spec_from_file_location('v8q',ROOT/'scripts/b2croi_v8q_benchmark.py')
v8q=importlib.util.module_from_spec(spec); spec.loader.exec_module(v8q)

POLICIES=['oracle','error_trigger','generic_voi','cvoi_sf','b2croi_v8q']
NETWORKS=['burst','severe_burst']
COLSETS={
 'temperature_inside':['Tin1','Tin2','Tout'],
 'humidity_inside':['Hin1','Hin2','Hout'],
}
SAFES={
 'temperature_inside':(12.0,32.0),
 'humidity_inside':(35.0,85.0),
}

def patch_safety(lo,hi,cols):
    v8q.SAFE_MIN=lo; v8q.SAFE_MAX=hi; v8q.RANGE=hi-lo; v8q.COLS=cols

def parse_args():
    ap=argparse.ArgumentParser(description='Run external validation for B2CRoI-H(Q).')
    ap.add_argument('--external-data-root', default=os.getenv('B2CROI_EXTERNAL_DATA_ROOT', str(EXTERNAL_DATA_ROOT)), help='Directory containing the external validation dataset file.')
    ap.add_argument('--external-data-file', default=os.getenv('B2CROI_EXTERNAL_DATA_FILE', EXTERNAL_DATA_FILE), help='External validation Excel filename.')
    return ap.parse_args()

def prep_data(cols):
    df=pd.read_excel(DATA)
    numeric=df[cols].apply(pd.to_numeric,errors='coerce')
    interpolated=numeric.interpolate(limit_direction='both')
    before=len(interpolated); clean=interpolated.dropna(); dropped=before-len(clean)
    print(f'[external validation] Dropped {dropped} rows after numeric coercion/interpolation/dropna out of {before} rows ({(100*dropped/before if before else 0):.3f}%).')
    d=clean.to_numpy(float)
    return d

def run_colset(name,cols,lo,hi):
    patch_safety(lo,hi,cols)
    data=prep_data(cols)
    train_end=min(2*24*60, len(data)//2)
    ar=v8q.fit_ar1(data[:train_end])
    alpha,beta,sigma=ar; train=data[:train_end]
    v8q.set_emp_residuals(train[1:]-(train[:-1]*alpha+beta))
    # evaluate validation days 3-5, in daily windows
    win=24*60; starts=list(range(train_end, len(data)-win, win))
    rows=[]
    for network in NETWORKS:
      for seed in range(20):
        for si,st in enumerate(starts):
          for pol in POLICIES:
            m=v8q.run(data,pol,network,seed,st,st+win,ar,bw=1)
            m.update({'dataset':'mendeley_3dw54yhhcc','variable_set':name,'safe_min':lo,'safe_max':hi,'network':network,'seed':seed,'window':si,'policy':pol})
            rows.append(m)
    raw=pd.DataFrame(rows)
    return raw

def summarize(raw):
    metrics=['rmse_mean','loss_mean','missed_violation_pct','choice_fairness','runtime_us_mean']
    g=raw.groupby(['variable_set','network','policy'])[metrics]
    mean=g.mean().reset_index()
    ci=g.sem().reset_index()
    out=mean.copy()
    for m in metrics: out[m+'_ci95']=ci[m]*1.96
    return out

def paired(raw):
    base=raw[raw.policy=='error_trigger'].set_index(['variable_set','network','seed','window'])
    rows=[]
    for pol in ['b2croi_v8q','cvoi_sf','generic_voi']:
      cur=raw[raw.policy==pol].set_index(['variable_set','network','seed','window'])
      idx=cur.index.intersection(base.index)
      for key in idx:
        r={'variable_set':key[0],'network':key[1],'seed':key[2],'window':key[3],'policy':pol,'baseline':'error_trigger'}
        for m in ['loss_mean','missed_violation_pct','rmse_mean','choice_fairness']:
          r[m+'_delta']=float(cur.loc[key,m]-base.loc[key,m])
        rows.append(r)
    return pd.DataFrame(rows)

def main():
    parts=[]
    for name,cols in COLSETS.items():
        lo,hi=SAFES[name]
        parts.append(run_colset(name,cols,lo,hi))
    raw=pd.concat(parts,ignore_index=True)
    raw.to_csv(OUT/'second_dataset_mendeley_raw.csv',index=False)
    summ=summarize(raw); summ.to_csv(OUT/'second_dataset_mendeley_summary.csv',index=False)
    pr=paired(raw); pr.to_csv(OUT/'second_dataset_mendeley_paired.csv',index=False)
    counts=pr.groupby(['variable_set','policy']).agg(
      n_cases=('loss_mean_delta','size'),
      loss_better=('loss_mean_delta',lambda x:int((x<0).sum())),
      missed_better=('missed_violation_pct_delta',lambda x:int((x<0).sum())),
      fair_better=('choice_fairness_delta',lambda x:int((x>0).sum())),
      rmse_le_0p05=('rmse_mean_delta',lambda x:int((x<=0.05).sum())),
      loss_delta_avg=('loss_mean_delta','mean'),
      missed_delta_avg=('missed_violation_pct_delta','mean'),
      rmse_delta_avg=('rmse_mean_delta','mean'),
      fairness_delta_avg=('choice_fairness_delta','mean')
    ).reset_index()
    counts.to_csv(OUT/'second_dataset_mendeley_case_counts.csv',index=False)
    # figure
    fig,axs=plt.subplots(1,2,figsize=(9,4),sharey=False)
    for ax,(name,sub) in zip(axs,summ.groupby('variable_set')):
      piv=sub[sub.network=='burst'].copy()
      ax.scatter(piv.loss_mean,piv.choice_fairness,s=45)
      for _,r in piv.iterrows(): ax.annotate(r.policy,(r.loss_mean,r.choice_fairness),fontsize=7,xytext=(3,3),textcoords='offset points')
      ax.set_title(name+' / burst'); ax.set_xlabel('loss'); ax.set_ylabel('fairness'); ax.grid(True,alpha=.25)
    fig.tight_layout(); fig.savefig(FIG/'second_dataset_mendeley_loss_fairness.pdf'); fig.savefig(FIG/'second_dataset_mendeley_loss_fairness.png',dpi=220); plt.close(fig)
    print(counts.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
if __name__=='__main__':
    args=parse_args()
    DATA = Path(args.external_data_root).expanduser().resolve() / args.external_data_file
    if not DATA.exists():
        raise FileNotFoundError(f'External validation dataset not found: {DATA}. Pass --external-data-root or set B2CROI_EXTERNAL_DATA_ROOT.')
    print(f'[external validation] Reading dataset: {DATA}')
    main()
