#!/usr/bin/env python3
"""B2CRoI-H(Q) benchmark: H with empirical residual-quantile safety calibration.

Public reproducibility script for the final B2CRoI-H(Q) method.
"""
from __future__ import annotations
from pathlib import Path
from time import perf_counter
import argparse
import os
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.getenv('B2CROI_DATA_ROOT', ROOT / 'data' / 'raw'))
FULL = DATA_ROOT / 'Full Data Set.csv'
OUT = ROOT / 'data' / 'processed'
OUT.mkdir(parents=True, exist_ok=True)
SAFE_MIN, SAFE_MAX = 22.0, 30.0
RANGE = SAFE_MAX - SAFE_MIN
COLS = ['Temperature Front', 'Temperature Middle', 'Temperature Back']
POLICIES = ['round_robin','max_aoi','channel_aware_rr','error_trigger','generic_voi','cvoi_sf','b2croi_hq','oracle']
NETWORKS = ['burst','severe_burst']

CHANNELS = {
    'burst': dict(good_loss=0.06, p_gb=0.035, bad_loss=0.65, p_bg=0.22),
    'severe_burst': dict(good_loss=0.08, p_gb=0.055, bad_loss=0.82, p_bg=0.15),
}

def violation(x): return (x < SAFE_MIN) | (x > SAFE_MAX)
def jain(x):
    x=np.asarray(x,float)
    return float((x.sum()**2)/(len(x)*(x*x).sum()+1e-12)) if x.sum()>0 else 1.0

def risk(x, delta=2.0, lam=1.0):
    b=np.minimum(x-SAFE_MIN, SAFE_MAX-x)
    v=np.maximum.reduce([SAFE_MIN-x, x-SAFE_MAX, np.zeros_like(x)])
    return np.clip(1.0-b/delta,0,1) + lam*v/(delta+v)

def mismatch(xh,xt): return np.abs(xt-xh)/RANGE
def gen_voi(xh,xt,a): return ((xt-xh)/RANGE)**2*(1+0.08*np.log1p(a)+0.004*(a**1.25))
def delay_voi(xh,xt,a): return np.maximum(((xt-xh)/RANGE)**2*(1+0.10*np.log1p(a)+0.008*(a**1.4))-0.012,0)
def loss_terms(xt,xh):
    rmse2=((xh-xt)/RANGE)**2
    missed=violation(xt)&(~violation(xh))
    false=(~violation(xt))&violation(xh)
    return rmse2 + 6*missed.astype(float) + 1.5*false.astype(float), missed, false

def fit_ar1(train):
    x=train[:-1]; y=train[1:]
    alphas=[]; betas=[]; sigmas=[]
    for j in range(train.shape[1]):
        X=np.c_[x[:,j], np.ones(len(x))]
        coef,*_=np.linalg.lstsq(X,y[:,j],rcond=None)
        pred=X@coef
        alphas.append(float(coef[0])); betas.append(float(coef[1])); sigmas.append(float(np.var(y[:,j]-pred)))
    return np.array(alphas), np.array(betas), np.array(sigmas)

def forecast_stats(xh, a, ar, h=6):
    alpha,beta,sigma=ar
    # h-step mean from current estimate; uncertainty includes age and horizon growth.
    mu=xh.copy().astype(float)
    var=np.zeros_like(mu)
    steps=np.maximum(a+h,1)
    # closed-form-ish iterative for numerical stability
    for _ in range(int(max(steps))):
        active=steps>_
        mu[active]=alpha[active]*mu[active]+beta[active]
        var[active]=(alpha[active]**2)*var[active]+sigma[active]
    return mu, np.maximum(var,1e-9)

def normal_cdf(z):
    # erf approximation via numpy vectorize math.erf unavailable as np.erf in minimal envs
    import math
    return np.vectorize(lambda t: 0.5*(1+math.erf(float(t)/math.sqrt(2))))(z)

def safety_prob(mu,var):
    sd=np.sqrt(var)
    p_low=normal_cdf((SAFE_MIN-mu)/sd)
    p_high=1-normal_cdf((SAFE_MAX-mu)/sd)
    return np.clip(p_low+p_high,0,1)


EMP_RES = None

def set_emp_residuals(res):
    global EMP_RES
    EMP_RES = res

def empirical_safety_prob(mu,var):
    global EMP_RES
    if EMP_RES is None:
        return safety_prob(mu,var)
    probs=[]
    for j in range(len(mu)):
        rr=EMP_RES[:, j % EMP_RES.shape[1]]
        probs.append(float(np.mean((mu[j]+rr<SAFE_MIN)|(mu[j]+rr>SAFE_MAX))))
    return np.asarray(probs)

def ar1_growth_score(xh,xt,a,ar):
    alpha,beta,sigma=ar
    growth=sigma*(1-np.power(alpha,2*np.maximum(a,1)))/np.maximum(1e-9,1-alpha**2)
    return ((xt-xh)/RANGE)**2 + growth/(RANGE**2)

def channel_step(kind,rng,bad):
    c=CHANNELS[kind]
    if bad:
        ok=rng.random()>c['bad_loss']
        if rng.random()<c['p_bg']: bad=False
    else:
        ok=rng.random()>c['good_loss']
        if rng.random()<c['p_gb']: bad=True
    return ok,bad

def update_bad_belief(pi_bad, observed_ok, kind):
    c=CHANNELS[kind]
    # predict transition
    pi_pred=pi_bad*(1-c['p_bg']) + (1-pi_bad)*c['p_gb']
    like_bad=(1-c['bad_loss']) if observed_ok else c['bad_loss']
    like_good=(1-c['good_loss']) if observed_ok else c['good_loss']
    denom=pi_pred*like_bad + (1-pi_pred)*like_good
    return float((pi_pred*like_bad)/max(denom,1e-12))

def predict_success(pi_bad, kind):
    c=CHANNELS[kind]
    pi_pred=pi_bad*(1-c['p_bg']) + (1-pi_bad)*c['p_gb']
    return (1-pi_pred)*(1-c['good_loss']) + pi_pred*(1-c['bad_loss'])

def _norm01(x):
    x=np.asarray(x,float)
    lo=float(np.min(x)); hi=float(np.max(x))
    if hi-lo < 1e-12:
        return np.zeros_like(x)
    return (x-lo)/(hi-lo)

def b2croi_score(xh,xt,a,ar,pi_bad,dual_f,dual_s,kind,counts,total_choices,h=4, q_s=None, q_f=None, q_e=0.0, variant="b2croi_hq", tau_J=0.985, kappa_J=35):
    # H: hybrid mode-switch scheduler.
    # Default mode is loss-prioritized predictive loss-gain scheduling.
    # Constraint mode activates only when service-floor/fairness debt becomes unsafe.
    mu,var=forecast_stats(xh,a,ar,h=h)
    pvio=empirical_safety_prob(mu,var)
    psucc=predict_success(pi_bad,kind)
    n=len(a)
    if q_s is None: q_s=0.0
    if q_f is None: q_f=np.zeros(n)

    cur,_,_=loss_terms(xt,xh)
    loss_gain=[]; residual_after=[]; max_residual_after=[]
    cur_res=((xt-xh)/RANGE)**2
    for i in range(n):
        cand=xh.copy(); cand[i]=xt[i]
        nxt,_,_=loss_terms(xt,cand)
        loss_gain.append(float(np.sum(cur-nxt)))
        res=((xt-cand)/RANGE)**2
        residual_after.append(float(np.mean(res)))
        max_residual_after.append(float(np.max(res)))
    loss_gain=np.asarray(loss_gain)
    residual_after=np.asarray(residual_after)
    max_residual_after=np.asarray(max_residual_after)
    loss_norm=_norm01(loss_gain)

    innovation=_norm01(cur_res)
    pred_risk=_norm01(np.clip(pvio*risk(mu)*mismatch(xh,xt),0,0.35))
    safety_now=_norm01(risk(xt)*mismatch(xh,xt))

    # Service-floor debt: compare realized share to target. This is used for mode
    # switching, not as a permanent dominant score component.
    if total_choices <= 0:
        share=np.ones(n)/n
    else:
        share=counts/max(total_choices,1)
    target=1.0/n
    service_gap=np.maximum(0.0,target-share)/max(target,1e-12)
    age_pressure=np.clip(np.maximum(0,a-n)/max(n,1),0,2)
    debt_raw=service_gap + 0.25*age_pressure
    debt_norm=_norm01(debt_raw)

    best=float(np.max(loss_gain))
    gap=np.maximum(0.0,best-loss_gain)
    near_best=np.exp(-gap/(abs(best)+1e-9+0.05))
    severe_debt=1/(1+np.exp(-5*(debt_norm-0.72)))

    # Hard switch + smooth multiplier: the constrained branch is entered only
    # after a service-floor condition is violated; the sigmoid values then scale
    # the strength of the fairness pressure inside that branch.
    fair_now=jain(counts+1e-9)
    global_fair_alarm=1/(1+np.exp(kappa_J*(fair_now-tau_J)))
    local_floor_alarm=float(np.max(severe_debt))
    switch=float((fair_now < tau_J) or (float(np.max(debt_norm)) > 0.72))
    mode=switch*np.clip(max(global_fair_alarm,local_floor_alarm),0,1)

    reliability=np.clip(psucc,0.25,1.0)
    rmse_cost=0.70*_norm01(residual_after)+0.30*_norm01(max_residual_after)
    rmse_relief=_norm01(cur_res)

    # loss-prioritized base: near-oracle loss/RMSE behavior
    score = reliability*(0.66*loss_norm + 0.14*innovation + 0.10*safety_now + 0.07*pred_risk)
    score += 0.05*rmse_relief - 0.05*rmse_cost

    # Constraint add-on only when needed: fairness can override only if loss gap is
    # small or service-floor debt is severe.
    fairness_gate=np.maximum(near_best,severe_debt)
    constraint_bonus= fairness_gate*(0.18*debt_norm + 0.08*np.log1p(np.maximum(q_f,0.0))*debt_norm)
    score += mode*constraint_bonus

    # When in constraint mode, avoid creating large residuals elsewhere.
    score -= mode*0.04*rmse_cost
    return score

def choose(policy,krel,xt,xh,a,bw,ar,kind,pi_bad,dual_f,dual_s,counts,total_choices,q_s=0.0,q_f=None,q_e=0.0, tau_J=0.985, kappa_J=35):
    M=len(xt)
    if bw>=M: return list(range(M))
    if policy=='round_robin': return [int((krel+j)%M) for j in range(bw)]
    if policy=='channel_aware_rr':
        # Burst-aware cyclic baseline: keep cyclic service but prioritize loops with
        # higher predicted delivery probability under the current GE belief.
        base=np.array([((krel+j)%M) for j in range(M)])
        freshness=-np.array([((j-krel)%M) for j in range(M)], dtype=float)/max(M,1)
        sc=0.70*predict_success(pi_bad,kind)+0.30*_norm01(a)+0.01*freshness
        return list(np.argsort(sc)[::-1][:bw])
    if policy=='oracle':
        cur,_,_=loss_terms(xt,xh); vals=[]
        for i in range(M):
            cand=xh.copy(); cand[i]=xt[i]
            nxt,_,_=loss_terms(xt,cand)
            vals.append(float(np.sum(cur-nxt)))
        return list(np.argsort(vals)[::-1][:bw])
    if policy=='max_aoi': sc=a.astype(float)
    elif policy=='error_trigger': sc=mismatch(xh,xt)
    elif policy=='generic_voi': sc=gen_voi(xh,xt,a)
    elif policy=='delay_voi': sc=delay_voi(xh,xt,a)
    elif policy=='ar1_growth_voi': sc=ar1_growth_score(xh,xt,a,ar)
    elif policy=='cvoi_sf': sc=gen_voi(xh,xt,a)+0.35*risk(xt)*mismatch(xh,xt)+0.015*a
    elif policy=='b2croi_hq':
        sc=b2croi_score(xh,xt,a,ar,pi_bad,dual_f,dual_s,kind,counts,total_choices,q_s=q_s,q_f=q_f,q_e=q_e,variant='b2croi_hq',tau_J=tau_J,kappa_J=kappa_J)
    else: raise ValueError(policy)
    return list(np.argsort(sc)[::-1][:bw])

def run(data,policy,network,seed,start,end,ar,bw=1):
    rng=np.random.default_rng(seed)
    bad=False; pi_bad=CHANNELS[network]['p_gb']/(CHANNELS[network]['p_gb']+CHANNELS[network]['p_bg'])
    xh=data[start].copy(); a=np.zeros(data.shape[1],dtype=int)
    counts=np.zeros(data.shape[1],dtype=float); total_choices=0
    dual_f=0.35; dual_s=0.20
    q_s=0.0; q_f=np.zeros(data.shape[1],dtype=float); q_e=0.0
    errs=[]; losses=[]; missed=[]; false=[]; ages=[]; choices=[]; success=[]; runt=[]; fairness_trace=[]; pbad_trace=[]
    for k in range(start+1,end):
        xt=data[k]
        t0=perf_counter(); idx=choose(policy,k-start-1,xt,xh,a,bw,ar,network,pi_bad,dual_f,dual_s,counts,total_choices,q_s=q_s,q_f=q_f,q_e=q_e); runt.append(perf_counter()-t0)
        a+=1
        step_missed_any=False
        for i in idx:
            choices.append(i); counts[i]+=1; total_choices+=1
            ok,bad=channel_step(network,rng,bad); success.append(ok)
            pi_bad=update_bad_belief(pi_bad,ok,network)
            if ok:
                xh[i]=xt[i]; a[i]=0
        lt,mi,fa=loss_terms(xt,xh)
        step_missed_any=bool(mi.any())
        fair_now=jain(counts)
        # Lyapunov virtual queues. Thresholds are budgets, not tuned scoring weights.
        q_s=max(0.0, q_s + float(mi.mean()) - 0.006)
        target_share=1.0/data.shape[1]
        served=np.zeros(data.shape[1]); served[idx]=1.0/max(len(idx),1)
        q_f=np.maximum(0.0, q_f + target_share - served)
        q_e=max(0.0, q_e + float(np.mean(((xh-xt)/RANGE)**2)) - 0.006)
        # legacy duals retained for compatibility with old policies; v5 mainly uses q_s/q_f/q_e.
        dual_f=np.clip(dual_f + 0.02*(0.985-fair_now), 0.05, 1.2)
        dual_s=np.clip(dual_s + 0.015*((1.0 if step_missed_any else 0.0)-0.02), 0.05, 1.0)
        errs.append((xh-xt).copy()); losses.append(lt); missed.append(mi); false.append(fa); ages.append(a.copy()); fairness_trace.append(fair_now); pbad_trace.append(pi_bad)
    errs=np.array(errs); losses=np.array(losses); missed=np.array(missed); false=np.array(false); ages=np.array(ages); choices=np.array(choices); success=np.array(success,dtype=bool)
    final_counts=np.array([(choices==i).sum() for i in range(data.shape[1])]) if len(choices) else np.zeros(data.shape[1])
    return dict(policy=policy, network=network, seed=seed, window_start=start, n_steps=end-start-1,
        rmse_mean=float(np.mean(np.sqrt(np.mean(errs**2,axis=0)))), loss_mean=float(losses.mean()),
        missed_violation_pct=float(100*missed.mean()), false_alarm_pct=float(100*false.mean()), avg_aoi=float(ages.mean()), max_aoi=float(ages.max()),
        choice_fairness=jain(final_counts), packet_success_pct=float(100*success.mean()) if len(success) else 100.0,
        runtime_us_mean=float(1e6*np.mean(runt)), final_dual_f=float(dual_f), final_dual_s=float(dual_s), mean_pbad=float(np.mean(pbad_trace)), final_q_s=float(q_s), final_q_e=float(q_e), mean_q_f=float(np.mean(q_f)))

def ci95(x):
    x=np.asarray(x,float)
    return float(1.96*x.std(ddof=1)/np.sqrt(len(x))) if len(x)>1 else 0.0

def summarize(raw):
    metrics=['rmse_mean','loss_mean','missed_violation_pct','false_alarm_pct','avg_aoi','max_aoi','choice_fairness','packet_success_pct','runtime_us_mean']
    rows=[]
    for (net,pol),df in raw.groupby(['network','policy']):
        rec={'network':net,'policy':pol,'n_windows':len(df)}
        for m in metrics:
            rec[m+'_mean']=float(df[m].mean()); rec[m+'_ci95']=ci95(df[m])
        rows.append(rec)
    return pd.DataFrame(rows)

def paired(raw, proposed='b2croi_hq'):
    metrics=['rmse_mean','loss_mean','missed_violation_pct','choice_fairness','avg_aoi']
    rows=[]
    for net in NETWORKS:
        p=raw[(raw.network==net)&(raw.policy==proposed)].sort_values('window_start')
        for base in POLICIES:
            if base==proposed: continue
            b=raw[(raw.network==net)&(raw.policy==base)].sort_values('window_start')
            rec={'network':net,'proposed':proposed,'baseline':base,'n_windows':len(p)}
            for m in metrics:
                d=p[m].to_numpy()-b[m].to_numpy()
                rec[m+'_delta_mean']=float(d.mean()); rec[m+'_delta_ci95']=ci95(d); rec[m+'_win_pct']=float(100*np.mean(d<0 if m!='choice_fairness' else d>0))
            rows.append(rec)
    return pd.DataFrame(rows)

def parse_args():
    ap=argparse.ArgumentParser(description='Run the official B2CRoI-H(Q) primary benchmark.')
    ap.add_argument('--n-windows', type=int, default=int(os.getenv('B2CROI_N_WINDOWS','10')), help='Number of weekly evaluation windows after the calibration week. Use -1 for all available windows. Default: 10, matching the reported benchmark.')
    ap.add_argument('--data-root', default=os.getenv('B2CROI_DATA_ROOT', str(DATA_ROOT)), help='Directory containing the primary raw dataset file: Full Data Set.csv. Default: FINAL/data/raw or B2CROI_DATA_ROOT.')
    return ap.parse_args()

def select_starts(data_len, window, n_windows):
    starts=list(range(window,data_len-window,window))
    return starts if n_windows is None or n_windows < 0 else starts[:n_windows]

def load_numeric_panel(path=FULL, cols=COLS, label='primary benchmark'):
    df=pd.read_csv(path)
    numeric=df[cols].apply(pd.to_numeric,errors='coerce')
    before=len(numeric)
    clean=numeric.dropna()
    dropped=before-len(clean)
    print(f'[{label}] Dropped {dropped} rows during numeric coercion/dropna out of {before} rows ({(100*dropped/before if before else 0):.3f}%).')
    return clean.to_numpy(float)

def main():
    args=parse_args()
    data_root=Path(args.data_root).expanduser().resolve()
    full=data_root / 'Full Data Set.csv'
    if not full.exists():
        raise FileNotFoundError(f'Primary dataset not found: {full}. Pass --data-root /path/to/dataset_dir or set B2CROI_DATA_ROOT.')
    print(f'[primary benchmark] Reading dataset: {full}')
    data=load_numeric_panel(full)
    window=7*24*12
    ar=fit_ar1(data[:window])
    # empirical residuals for safety calibration
    alpha,beta,sigma=ar
    train=data[:window]
    set_emp_residuals(train[1:]-(train[:-1]*alpha+beta))
    starts=select_starts(len(data), window, args.n_windows)
    print(f'[primary benchmark] Evaluation windows: {len(starts)} (n_windows={args.n_windows}).')
    rows=[]
    for net in NETWORKS:
        for wi,st in enumerate(starts):
            for pol in POLICIES:
                rows.append(run(data,pol,net,2026+wi,st,st+window,ar,bw=1))
    raw=pd.DataFrame(rows); summ=summarize(raw); comp=paired(raw)
    raw.to_csv(OUT/'b2croi_hq_raw.csv',index=False)
    summ.to_csv(OUT/'b2croi_hq_summary.csv',index=False)
    comp.to_csv(OUT/'b2croi_hq_paired.csv',index=False)
    print(summ[['network','policy','rmse_mean_mean','rmse_mean_ci95','loss_mean_mean','loss_mean_ci95','missed_violation_pct_mean','missed_violation_pct_ci95','choice_fairness_mean','choice_fairness_ci95','runtime_us_mean_mean']].sort_values(['network','loss_mean_mean']).to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    print('\nPaired B2CRoI deltas written to', OUT/'b2croi_hq_paired.csv')

if __name__=='__main__': main()
