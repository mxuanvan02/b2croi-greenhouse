#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
T=ROOT/'data/processed'; T.mkdir(parents=True,exist_ok=True)

def fmt(x, nd=4):
    try: return f'{float(x):.{nd}f}'
    except Exception: return str(x)

# Table 1: operating regime comparison from final counts
counts=pd.read_csv(T/'final_v6_v7_v8_stress_case_counts.csv')
rows=[]
name_map={'b2croi_v6':'Fairness-constrained','b2croi_v7':'Loss-prioritized','b2croi_v8':'Hybrid constrained'}
for _,r in counts.iterrows():
    pol=r.get('policy',r.get('version',''))
    pol_key={'v6':'b2croi_v6','v7':'b2croi_v7','v8':'b2croi_v8'}.get(pol,pol)
    rows.append({
        'Operating regime': name_map.get(pol_key,pol),
        'Internal policy': pol_key,
        'Loss better': f"{int(r['loss_better_cases'])}/{int(r['n_cases'])}",
        'Missed-violation better': f"{int(r['missed_better_cases'])}/{int(r['n_cases'])}",
        'Fairness better': f"{int(r['fairness_better_cases'])}/{int(r['n_cases'])}",
        'RMSE $\\Delta\\leq0.05$': f"{int(r['rmse_le_0p05_cases'])}/{int(r['n_cases'])}",
        'Mean loss $\\Delta$': fmt(r['loss_mean_delta_mean_avg']),
        'Mean missed $\\Delta$ (pp)': fmt(r['missed_violation_pct_delta_mean_avg']),
        'Mean RMSE $\\Delta$': fmt(r['rmse_mean_delta_mean_avg']),
        'Mean fairness $\\Delta$': fmt(r['choice_fairness_delta_mean_avg']),
    })
pd.DataFrame(rows).to_csv(T/'public_table_operating_regimes.csv',index=False)

# Table 2: v8 ablation reduced stress favorable cases
paired=pd.read_csv(T/'b2croi_v8_ablation_stress_paired.csv')
base=[]
for pol,g in paired.groupby('baseline'):
    if pol=='oracle': continue
    n=len(g)
    base.append({
        'Variant': pol,
        'Loss better': f"{int((g.loss_mean_delta_mean<0).sum())}/{n}",
        'Missed better': f"{int((g.missed_violation_pct_delta_mean<0).sum())}/{n}",
        'Fairness better': f"{int((g.choice_fairness_delta_mean>0).sum())}/{n}",
        'RMSE $\\Delta\\leq0.05$': f"{int((g.rmse_mean_delta_mean<=0.05).sum())}/{n}",
        'Mean loss $\\Delta$': fmt(g.loss_mean_delta_mean.mean()),
        'Mean missed $\\Delta$ (pp)': fmt(g.missed_violation_pct_delta_mean.mean()),
        'Mean RMSE $\\Delta$': fmt(g.rmse_mean_delta_mean.mean()),
        'Mean fairness $\\Delta$': fmt(g.choice_fairness_delta_mean.mean()),
    })
pd.DataFrame(base).sort_values('Variant').to_csv(T/'public_table_ablation.csv',index=False)

# Table 3: safety calibration
cal=pd.read_csv(T/'safety_probability_calibration_summary.csv')
cal['Model']=cal['model'].map({'p_gaussian':'Gaussian AR(1)','p_empirical':'Empirical residual'}).fillna(cal['model'])
cal[['Model','brier','ece','mean_pred','obs_rate']].rename(columns={'brier':'Brier score','ece':'ECE','mean_pred':'Mean predicted','obs_rate':'Observed rate'}).to_csv(T/'public_table_safety_calibration.csv',index=False)

# Table 4: external validation
ext=pd.read_csv(T/'second_dataset_mendeley_case_counts.csv')
ext=ext[ext.policy=='b2croi_v8q'].copy()
ext['Variable set']=ext['variable_set'].map({'temperature_inside':'Temperature','humidity_inside':'Humidity'}).fillna(ext['variable_set'])
ext_out=pd.DataFrame({
    'Variable set': ext['Variable set'],
    'Cases': ext['n_cases'].astype(int),
    'Loss better': ext.apply(lambda r:f"{int(r.loss_better)}/{int(r.n_cases)}",axis=1),
    'Missed better': ext.apply(lambda r:f"{int(r.missed_better)}/{int(r.n_cases)}",axis=1),
    'Fairness better': ext.apply(lambda r:f"{int(r.fair_better)}/{int(r.n_cases)}",axis=1),
    'RMSE $\\Delta\\leq0.05$': ext.apply(lambda r:f"{int(r.rmse_le_0p05)}/{int(r.n_cases)}",axis=1),
    'Mean loss $\\Delta$': ext.loss_delta_avg.map(fmt),
    'Mean missed $\\Delta$ (pp)': ext.missed_delta_avg.map(fmt),
    'Mean RMSE $\\Delta$': ext.rmse_delta_avg.map(fmt),
    'Mean fairness $\\Delta$': ext.fairness_delta_avg.map(fmt),
})
ext_out.to_csv(T/'public_table_external_validation.csv',index=False)
print('Wrote public tables:', '\n'.join(str(p) for p in sorted(T.glob('public_table_*.csv'))))
