#!/usr/bin/env python3
from pathlib import Path
import textwrap
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / 'assets' / 'figures'
DATA = ROOT / 'data'
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Computer Modern Roman'],
    'mathtext.fontset': 'cm',
    'font.size': 8.5,
    'axes.titlesize': 10,
    'axes.labelsize': 8.5,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})
BLUE='#2563EB'; GREEN='#16A34A'; ORANGE='#EA580C'; RED='#DC2626'; GRAY='#475569'; LIGHT='#F8FAFC'

def wrap(txt, width=16):
    return '\n'.join(textwrap.wrap(txt, width=width))

def box(ax, xy, wh, text, fc=LIGHT, ec='#334155', lw=1.0, fs=8.5):
    x,y=xy; w,h=wh
    p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.035,rounding_size=0.06',fc=fc,ec=ec,lw=lw,zorder=2)
    ax.add_patch(p)
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fs,zorder=3,linespacing=1.15)
    return p

def arrow(ax, a, b, rad=0.0, color=GRAY):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=11,lw=1.05,color=color,connectionstyle=f'arc3,rad={rad}',zorder=1))

def save(fig, name):
    fig.savefig(FIG/f'{name}.pdf', bbox_inches='tight')
    fig.savefig(FIG/f'{name}.png', dpi=350, bbox_inches='tight')
    plt.close(fig)

def pipeline():
    fig,ax=plt.subplots(figsize=(7.6,3.25)); ax.set_xlim(0,12); ax.set_ylim(0,5); ax.axis('off')
    ax.text(6,4.78,'B2CRoI-H(Q): from sensed climate risk to scheduled updates',ha='center',va='top',fontsize=11,fontweight='bold')
    xs=[.45,2.35,4.25,6.15,8.05,9.95]
    labels=['Sensor history','Short-horizon prediction','Empirical residual calibration','Burst-channel belief','Hybrid risk-of-information score','Top-B scheduled updates']
    colors=['#DCFCE7','#EFF6FF','#F0FDFA','#FFF7ED','#EEF2FF','#E0F2FE']
    for x,l,c in zip(xs,labels,colors): box(ax,(x,2.35),(1.35,1.0),wrap(l,14),fc=c,fs=8.2)
    for i in range(len(xs)-1): arrow(ax,(xs[i]+1.35,2.85),(xs[i+1],2.85))
    box(ax,(4.45,.72),(1.18,.62),'Safety interval\n$[L,U]$',fc='#FEF3C7',fs=8); arrow(ax,(5.04,1.34),(4.93,2.35))
    box(ax,(8.05,.72),(1.38,.62),'Service floor\nand fairness',fc='#FEF3C7',fs=8); arrow(ax,(8.74,1.34),(8.72,2.35))
    box(ax,(6.12,4.0),(1.42,.58),'Packet outcomes',fc='#DCFCE7',fs=8); arrow(ax,(6.83,4.0),(6.82,3.35))
    ax.text(6,.22,'Visual message: B2CRoI-H(Q) schedules only the updates with high safety value under bursty communication and fairness constraints.',ha='center',fontsize=8,color=GRAY)
    save(fig,'b2croi_h_pipeline')

def mode_switch():
    fig,ax=plt.subplots(figsize=(7.4,4.1)); ax.set_xlim(0,12); ax.set_ylim(0,6); ax.axis('off')
    ax.text(6,5.72,'Hybrid mode switch: fairness pressure is activated only when needed',ha='center',va='top',fontsize=11,fontweight='bold')
    box(ax,(.55,2.55),(1.65,.82),'Service-state\ncheck',fc='#F8FAFC')
    box(ax,(3.15,4.05),(2.65,.82),'Loss-prioritized mode\n$R_i+C_i-\\lambda_E E_i$',fc='#DBEAFE')
    box(ax,(3.15,1.05),(2.65,.82),'Fairness-constrained mode\n$R_i+C_i+\\lambda_F F_i-\\lambda_E E_i$',fc='#FEE2E2')
    box(ax,(7.1,2.55),(1.75,.82),'Rank loops\nand select top-B',fc='#EEF2FF')
    box(ax,(9.9,2.55),(1.32,.82),'Update\nstates',fc='#DCFCE7')
    arrow(ax,(2.2,3.05),(3.15,4.46)); ax.text(2.36,4.05,'safe',fontsize=8,color=GREEN)
    arrow(ax,(2.2,2.72),(3.15,1.46)); ax.text(2.27,1.72,'unsafe',fontsize=8,color=RED)
    arrow(ax,(5.8,4.46),(7.1,3.15),rad=-.12); arrow(ax,(5.8,1.46),(7.1,2.75),rad=.12)
    arrow(ax,(8.85,2.96),(9.9,2.96)); arrow(ax,(10.55,3.37),(1.35,3.37),rad=.18)
    ax.text(6,.25,'Visual message: the constrained branch explains the RMSE--fairness trade-off instead of hiding it.',ha='center',fontsize=8,color=GRAY)
    save(fig,'b2croi_h_mode_switch')

def architecture():
    fig,ax=plt.subplots(figsize=(7.6,4.0)); ax.set_xlim(0,12); ax.set_ylim(0,6); ax.axis('off')
    ax.text(6,5.72,'Smart greenhouse networked sensing under burst packet loss',ha='center',va='top',fontsize=11,fontweight='bold')
    for y,t in [(4.25,'Zone 1 sensor'),(3.05,'Zone 2 sensor'),(1.85,'Zone N sensor')]: box(ax,(.55,y),(1.55,.72),t.replace(' ','\n'),fc='#DCFCE7',fs=8.2)
    box(ax,(3.25,2.45),(1.85,1.05),'Bursty wireless link\nGilbert--Elliott',fc='#FFEDD5',fs=8.2)
    box(ax,(6.25,2.45),(1.95,1.05),'B2CRoI-H(Q)\nedge scheduler',fc='#DBEAFE',fs=8.2)
    box(ax,(9.35,3.65),(1.75,.78),'Controller /\ndigital twin',fc='#F8FAFC',fs=8.2)
    box(ax,(9.35,1.62),(1.75,.78),'Monitoring\ndashboard',fc='#F8FAFC',fs=8.2)
    for y in [4.61,3.41,2.21]: arrow(ax,(2.1,y),(3.25,2.98))
    arrow(ax,(5.1,2.98),(6.25,2.98)); arrow(ax,(8.2,2.98),(9.35,4.04)); arrow(ax,(8.2,2.98),(9.35,2.01))
    arrow(ax,(10.22,3.65),(7.22,3.5),rad=.16)
    ax.text(5.3,3.72,'packet loss\nand bursts',fontsize=8,color=ORANGE,ha='center')
    ax.text(8.3,4.25,'selected\ntop-B updates',fontsize=8,color=BLUE,ha='center')
    ax.text(6,.25,'Visual message: scarce wireless capacity turns greenhouse sensing into a safety-aware scheduling problem.',ha='center',fontsize=8,color=GRAY)
    save(fig,'b2croi_h_architecture')

def evaluation_matrix():
    fig,ax=plt.subplots(figsize=(7.5,3.7)); ax.set_xlim(0,12); ax.set_ylim(0,6); ax.axis('off')
    ax.text(6,5.7,'Evaluation matrix: every claim is tied to a metric family',ha='center',va='top',fontsize=11,fontweight='bold')
    rows=['Primary greenhouse','Stress grid','External validation']
    cols=['Safety loss','Missed violation','RMSE','Jain fairness','Calibration']
    x0,y0,cw,ch=3.0,4.25,1.55,.76
    for j,c in enumerate(cols): box(ax,(x0+j*cw,y0),(cw-.12,.58),wrap(c,12),fc='#F8FAFC',fs=8)
    for i,r in enumerate(rows): box(ax,(.65,y0-(i+1)*ch),(1.75,.58),wrap(r,14),fc='#F8FAFC',fs=8)
    marks=[['Yes','Yes','Yes','Yes','Yes'],['Yes','Yes','Yes','Yes',''],['Yes','Yes','Yes','Yes','']]
    for i in range(3):
        for j in range(5):
            fc='#DBEAFE' if marks[i][j] else '#F1F5F9'
            if i==2 and marks[i][j]: fc='#DCFCE7'
            box(ax,(x0+j*cw,y0-(i+1)*ch),(cw-.12,.58),marks[i][j],fc=fc,fs=8.8)
    ax.text(6,.35,'Visual message: the study separates primary evidence, stress evidence, and external-validation evidence.',ha='center',fontsize=8,color=GRAY)
    save(fig,'b2croi_h_evaluation_matrix')

def result_operating_regimes():
    df=pd.read_csv(DATA/'public_table_operating_regimes.csv')
    regimes=df['Operating regime'].str.replace('Fairness-constrained','Fairness\nconstrained').str.replace('Loss-prioritized','Loss\nprioritized').str.replace('Hybrid constrained','Hybrid\nconstrained')
    fig,axs=plt.subplots(1,2,figsize=(7.6,4.4),gridspec_kw={'width_ratios':[1.12,1]})
    x=np.arange(len(df)); w=.25
    axs[0].bar(x-w, -df['Mean loss $\\Delta$'], w, label='Loss reduction', color=BLUE)
    axs[0].bar(x, -df['Mean missed $\\Delta$ (pp)'], w, label='Missed-violation reduction', color=GREEN)
    axs[0].bar(x+w, df['Mean fairness $\\Delta$']*10, w, label='Fairness gain ×10', color=ORANGE)
    axs[0].set_xticks(x); axs[0].set_xticklabels(regimes)
    axs[0].set_title('What improves?')
    axs[0].set_ylabel('Improvement magnitude')
    axs[0].legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5,-0.30), ncol=1, borderaxespad=0.0)
    axs[0].grid(axis='y',alpha=.18,lw=.6)
    axs[1].bar(x, df['Mean RMSE $\\Delta$'], color=[ORANGE,BLUE,ORANGE], width=.48)
    axs[1].axhline(.05,color=RED,ls='--',lw=1,label='RMSE +0.05')
    axs[1].set_xticks(x); axs[1].set_xticklabels(regimes)
    axs[1].set_title('What is the cost?')
    axs[1].set_ylabel('Mean RMSE delta')
    axs[1].legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5,-0.30), ncol=1, borderaxespad=0.0)
    axs[1].grid(axis='y',alpha=.18,lw=.6)
    for ax in axs:
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Operating regimes expose the safety--fairness--accuracy trade-off',fontweight='bold',fontsize=10.6)
    fig.tight_layout(pad=.8, rect=[0,0.15,1,.94])
    save(fig,'result_operating_regimes_story')

def result_calibration():
    df=pd.read_csv(DATA/'public_table_safety_calibration.csv')
    fig,ax=plt.subplots(figsize=(5.9,4.05))
    x=np.arange(len(df)); w=.32
    ax.bar(x-w/2,df['Brier score'],w,label='Brier score',color=BLUE)
    ax.bar(x+w/2,df['ECE'],w,label='ECE',color=GREEN)
    ax.set_xticks(x); ax.set_xticklabels(df['Model'])
    ax.set_ylabel('Calibration error (lower is better)')
    ax.set_title('Empirical residual tails improve safety-probability calibration',fontweight='bold',fontsize=10.2)
    ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5,-0.17), ncol=2)
    ymax=max(df['Brier score'].max(),df['ECE'].max())
    ax.set_ylim(0, ymax*1.36)
    for i,row in df.iterrows():
        ax.text(i, ymax*1.21, f"Observed rate\n{row['Observed rate']:.2f}", ha='center', va='top', fontsize=7.2, color=GRAY,
                bbox=dict(boxstyle='round,pad=0.18', fc='white', ec='none', alpha=0.90))
    ax.grid(axis='y',alpha=.18,lw=.6)
    ax.spines[['top','right']].set_visible(False)
    fig.tight_layout(pad=.8, rect=[0,0.09,1,1])
    save(fig,'result_calibration_story')

def result_external_validation():
    df=pd.read_csv(DATA/'public_table_external_validation.csv')
    fig,ax=plt.subplots(figsize=(7.25,4.20))
    x=np.arange(len(df)); w=.20
    metrics=[('Loss better','Loss better',BLUE),('Missed better','Missed better',GREEN),('Fairness better','Fairness better',ORANGE),('RMSE $\\Delta\\leq0.05$','RMSE within 0.05',GRAY)]
    for k,(col,label,color) in enumerate(metrics):
        vals=df[col].astype(str).str.split('/').str[0].astype(float)/df['Cases']*100
        ax.bar(x+(k-1.5)*w,vals,w,label=label,color=color)
    ax.set_ylim(0,112)
    ax.set_ylabel('Paired cases satisfying criterion (%)')
    ax.set_xticks(x); ax.set_xticklabels(df['Variable set'])
    ax.set_title('External validation checks whether trends persist across variables',fontweight='bold',fontsize=10.2)
    ax.legend(frameon=False,ncol=2,loc='upper center',bbox_to_anchor=(.5,-.20),columnspacing=1.2)
    ax.grid(axis='y',alpha=.18,lw=.6)
    ax.spines[['top','right']].set_visible(False)
    fig.tight_layout(pad=.8, rect=[0,0.11,1,1])
    save(fig,'result_external_validation_story')

if __name__=='__main__':
    pipeline(); mode_switch(); architecture(); evaluation_matrix()
    result_operating_regimes(); result_calibration(); result_external_validation()
    print('wrote figures to', FIG)
