#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'tables'; DST=ROOT/'manuscript/tables'
configs={
 'public_table_operating_regimes':['Regime','Policy','Loss','Missed','Fair','RMSE<=.05','dLoss','dMiss','dRMSE','dFair'],
 'public_table_ablation':['Comparator','Cases','Loss','Missed','Fair','RMSE<=0','dLoss','dMiss','dRMSE','dFair'],
 'public_table_safety_calibration':['Model','Brier','ECE','Mean pred.','Observed'],
 'public_table_external_validation':['Var. set','Cases','Loss','Missed','Fair','RMSE<=.05','dLoss','dMiss','dRMSE','dFair'],
}
for stem,cols in configs.items():
 df=pd.read_csv(SRC/f'{stem}.csv')
 df.columns=cols
 # shorten internal labels
 for c in df.columns:
  if df[c].dtype==object:
   df[c]=df[c].astype(str).str.replace('b2croi_','',regex=False).str.replace('_','-',regex=False)
 tex=df.to_latex(index=False, escape=True, column_format='l'+'c'*(len(df.columns)-1))
 tex='\\resizebox{\\linewidth}{!}{%\n'+tex+'}\n'
 (DST/f'{stem}_latex.tex').write_text(tex)
 print(DST/f'{stem}_latex.tex')
