#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle, Polygon, Arc

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / 'assets' / 'figures'
DATA = ROOT / 'data'
FIG.mkdir(parents=True, exist_ok=True)

BLUE='#2563EB'; GREEN='#16A34A'; ORANGE='#EA580C'; RED='#DC2626'; GRAY='#475569'; DARK='#0F172A'
LIGHT='#F8FAFC'; SKY='#DBEAFE'; MINT='#DCFCE7'; AMBER='#FEF3C7'; PEACH='#FFEDD5'; LAV='#EEF2FF'

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Computer Modern Roman'],
    'mathtext.fontset': 'cm',
    'font.size': 8.5,
    'axes.titlesize': 9.5,
    'axes.labelsize': 8.5,
    'xtick.labelsize': 7.7,
    'ytick.labelsize': 7.7,
    'legend.fontsize': 7.7,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

def save(fig, name):
    fig.savefig(FIG / f'{name}.pdf', bbox_inches='tight')
    fig.savefig(FIG / f'{name}.png', dpi=350, bbox_inches='tight')
    plt.close(fig)

def box(ax, x, y, w, h, txt, fc=LIGHT, ec=GRAY, fs=8.5, weight='normal', z=3):
    p = FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.03,rounding_size=0.08',fc=fc,ec=ec,lw=1.1,zorder=z)
    ax.add_patch(p)
    ax.text(x+w/2,y+h/2,txt,ha='center',va='center',fontsize=fs,fontweight=weight,color=DARK,linespacing=1.1,zorder=z+1)
    return p

def arrow(ax, a, b, color=GRAY, rad=0, lw=1.35, ms=13):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=ms,lw=lw,color=color,connectionstyle=f'arc3,rad={rad}',zorder=2))

def greenhouse(ax, x=0.55, y=1.35, s=1.0):
    ax.add_patch(Rectangle((x,y),2.0*s,1.85*s,fc='#ECFDF5',ec=GREEN,lw=1.1,zorder=1))
    ax.add_patch(Polygon([[x-.1*s,y+1.85*s],[x+s,y+2.6*s],[x+2.1*s,y+1.85*s]],fc='#D1FAE5',ec=GREEN,lw=1.1,zorder=1))
    for px,py in [(x+.45*s,y+.7*s),(x+1.0*s,y+1.18*s),(x+1.55*s,y+.75*s)]:
        ax.add_patch(Circle((px,py),.09*s,fc=GREEN,ec='white',lw=.8,zorder=4))
        ax.plot([px,px],[py-.09*s,py-.42*s],color=GREEN,lw=1.1,zorder=4)

def architecture_story():
    fig, ax = plt.subplots(figsize=(7.45,3.75)); ax.set_xlim(0,12); ax.set_ylim(0,5.7); ax.axis('off')
    ax.text(6,5.45,'Architecture: safety-aware greenhouse sensing over bursty wireless links',ha='center',va='top',fontsize=10.8,fontweight='bold')
    greenhouse(ax,.45,1.55,.9); box(ax,.62,.55,1.75,.45,'Sensors',fc=MINT,ec=GREEN,fs=8.7,weight='bold')
    for y in [3.65,3.25,2.85]: arrow(ax,(2.55,y),(3.35,3.05),GREEN,lw=1.1,ms=10)
    for r,a in [(0.28,1),(0.52,.75),(0.78,.55)]: ax.add_patch(Arc((4.0,3.05),2*r,2*r,theta1=-45,theta2=45,color=ORANGE,lw=1.7,alpha=a))
    for px in [3.15,3.45,3.75]: ax.add_patch(Rectangle((px,2.18),.16,.10,fc=ORANGE,ec='none',alpha=.35))
    box(ax,3.25,.55,1.55,.45,'Burst channel',fc=PEACH,ec=ORANGE,fs=8.5,weight='bold')
    arrow(ax,(4.85,3.05),(5.65,3.05),ORANGE,lw=1.5)
    box(ax,5.65,2.35,1.95,1.08,'B2CRoI-H(Q)',fc=SKY,ec=BLUE,fs=9.6,weight='bold')
    ax.text(6.63,2.10,'belief  +  risk  +  fairness',ha='center',fontsize=7.6,color=GRAY)
    box(ax,7.95,3.62,1.35,.48,'Top-B updates',fc='#E0F2FE',ec=BLUE,fs=8.4,weight='bold')
    arrow(ax,(7.60,3.15),(7.95,3.88),BLUE,rad=.05,lw=1.35)
    arrow(ax,(9.30,3.86),(9.95,4.12),BLUE,lw=1.35)
    box(ax,9.95,3.74,1.42,.62,'Controller',fc=LIGHT,ec=GRAY,fs=8.4,weight='bold')
    arrow(ax,(9.30,3.65),(9.95,2.30),BLUE,rad=-.15,lw=1.25)
    box(ax,9.95,1.95,1.42,.62,'Dashboard',fc=LIGHT,ec=GRAY,fs=8.4,weight='bold')
    ax.text(6,.18,'Visual message: scarce and bursty links turn greenhouse sensing into a selective update-scheduling problem.',ha='center',fontsize=7.8,color=GRAY)
    save(fig,'b2croi_architecture_story_formal')

def mode_story():
    # Clean academic vector figure. No decorative icons; all labels are short.
    fig, ax = plt.subplots(figsize=(7.2,4.8)); ax.set_xlim(0,10); ax.set_ylim(0,6); ax.axis('off')
    ax.text(5,5.72,'Hybrid mode-switch logic',ha='center',va='top',fontsize=11.0,fontweight='bold',color=DARK)

    # main flow
    box(ax,.45,4.15,1.55,.78,'Service\nstate',fc=LIGHT,ec=GRAY,fs=8.8,weight='bold')
    box(ax,2.55,4.15,1.80,.78,'Loss-priority\nscore',fc=SKY,ec=BLUE,fs=8.8,weight='bold')
    box(ax,5.00,4.15,1.80,.78,'Top-$B$\nselection',fc=MINT,ec=GREEN,fs=8.8,weight='bold')
    box(ax,7.45,4.15,1.75,.78,'State\nupdate',fc=LIGHT,ec=GRAY,fs=8.8,weight='bold')
    arrow(ax,(2.00,4.54),(2.55,4.54),BLUE,lw=1.25,ms=11)
    arrow(ax,(4.35,4.54),(5.00,4.54),BLUE,lw=1.25,ms=11)
    arrow(ax,(6.80,4.54),(7.45,4.54),GREEN,lw=1.25,ms=11)

    # alarm branch
    box(ax,2.15,2.15,2.15,.82,'Fairness /\nservice alarm',fc=AMBER,ec=ORANGE,fs=8.8,weight='bold')
    box(ax,5.05,2.15,2.15,.82,'Constrained\nscore',fc='#FEE2E2',ec=RED,fs=8.8,weight='bold')
    arrow(ax,(1.22,4.15),(2.90,2.97),ORANGE,rad=-.14,lw=1.15,ms=10)
    arrow(ax,(4.30,2.56),(5.05,2.56),ORANGE,lw=1.25,ms=11)
    arrow(ax,(6.12,2.97),(5.90,4.15),RED,rad=.12,lw=1.15,ms=10)

    # equation notes as compact text blocks
    ax.text(3.45,3.72,r'$R_{i,t}+C_{i,t}-\lambda_EE_{i,t}$',ha='center',fontsize=8.1,color=BLUE)
    ax.text(6.12,1.78,r'$R_{i,t}+C_{i,t}+\lambda_F(t)F_{i,t}-\lambda_EE_{i,t}$',ha='center',fontsize=8.1,color=RED)
    ax.text(5,0.55,'The fairness branch is activated only when service conditions become unsafe.',ha='center',fontsize=8.1,color=GRAY)
    save(fig,'b2croi_mode_switch_story_formal')

def evaluation_story():
    # Clean academic evidence map. No icons or bottom slogan.
    fig, ax = plt.subplots(figsize=(7.2,4.8)); ax.set_xlim(0,10); ax.set_ylim(0,6); ax.axis('off')
    ax.text(5,5.72,'Evaluation map',ha='center',va='top',fontsize=11.0,fontweight='bold',color=DARK)

    box(ax,3.85,2.60,2.30,.86,'B2CRoI-H(Q)\nscheduler',fc=SKY,ec=BLUE,fs=9.0,weight='bold')

    box(ax,.55,4.10,2.25,.86,'Operating\nregimes',fc=MINT,ec=GREEN,fs=8.8,weight='bold')
    ax.text(1.68,3.82,'loss, RMSE, fairness',ha='center',fontsize=7.6,color=GRAY)
    box(ax,.55,1.20,2.25,.86,'Calibration',fc=AMBER,ec=ORANGE,fs=8.8,weight='bold')
    ax.text(1.68,.92,'Brier score, ECE',ha='center',fontsize=7.6,color=GRAY)
    box(ax,7.20,4.10,2.25,.86,'External\nvalidation',fc='#E0F2FE',ec=BLUE,fs=8.8,weight='bold')
    ax.text(8.32,3.82,'independent dataset',ha='center',fontsize=7.6,color=GRAY)
    box(ax,7.20,1.20,2.25,.86,'Baselines /\nablation',fc=LAV,ec=GRAY,fs=8.8,weight='bold')
    ax.text(8.32,.92,'AoI, RR, components',ha='center',fontsize=7.6,color=GRAY)

    arrow(ax,(3.85,3.20),(2.80,4.45),GREEN,rad=.05,lw=1.15,ms=10)
    arrow(ax,(3.85,2.82),(2.80,1.62),ORANGE,rad=-.05,lw=1.15,ms=10)
    arrow(ax,(6.15,3.20),(7.20,4.45),BLUE,rad=-.05,lw=1.15,ms=10)
    arrow(ax,(6.15,2.82),(7.20,1.62),GRAY,rad=.05,lw=1.15,ms=10)

    ax.text(5,0.30,'Each evidence block supports a separate part of the safety--fairness claim.',ha='center',fontsize=8.0,color=GRAY)
    save(fig,'b2croi_evaluation_story_formal')

def operating_formal():
    df=pd.read_csv(DATA/'public_table_operating_regimes.csv')
    regimes=df['Operating regime'].str.replace('Fairness-constrained','Fairness\nconstrained').str.replace('Loss-prioritized','Loss\nprioritized').str.replace('Hybrid constrained','Hybrid\nconstrained')
    fig,axs=plt.subplots(1,2,figsize=(7.45,4.15),gridspec_kw={'width_ratios':[1.15,1]})
    x=np.arange(len(df)); w=.25
    axs[0].bar(x-w, -df['Mean loss $\\Delta$'], w, label='Loss reduction', color=BLUE)
    axs[0].bar(x, -df['Mean missed $\\Delta$ (pp)'], w, label='Missed-violation reduction', color=GREEN)
    axs[0].bar(x+w, df['Mean fairness $\\Delta$']*10, w, label='Fairness gain ×10', color=ORANGE)
    axs[0].set_xticks(x); axs[0].set_xticklabels(regimes)
    axs[0].set_ylabel('Improvement magnitude')
    axs[0].set_title('What improves?')
    axs[0].legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5,-0.28), ncol=1, borderaxespad=0.0)
    axs[0].grid(axis='y',alpha=.18,lw=.6)
    bars=axs[1].bar(x, df['Mean RMSE $\\Delta$'], color=[ORANGE,BLUE,ORANGE], width=.48)
    axs[1].axhline(.05,color=RED,ls='--',lw=1,label='RMSE +0.05')
    axs[1].set_xticks(x); axs[1].set_xticklabels(regimes)
    axs[1].set_ylabel('Mean RMSE delta')
    axs[1].set_title('What is the cost?')
    axs[1].legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5,-0.28), ncol=1, borderaxespad=0.0)
    axs[1].grid(axis='y',alpha=.18,lw=.6)
    for ax in axs:
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Operating regimes expose the safety--fairness--accuracy trade-off',fontweight='bold',fontsize=10.6)
    fig.tight_layout(pad=.8, rect=[0,0.13,1,.94])
    save(fig,'result_operating_regimes_story')

def calibration_formal():
    df=pd.read_csv(DATA/'public_table_safety_calibration.csv')
    fig,ax=plt.subplots(figsize=(5.85,3.95))
    x=np.arange(len(df)); w=.32
    ax.bar(x-w/2,df['Brier score'],w,label='Brier score',color=BLUE)
    ax.bar(x+w/2,df['ECE'],w,label='ECE',color=GREEN)
    ax.set_xticks(x); ax.set_xticklabels(df['Model'])
    ax.set_ylabel('Calibration error (lower is better)')
    ax.set_title('Empirical residual tails improve safety-probability calibration',fontweight='bold',fontsize=10.2)
    ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5,-0.16), ncol=2)
    ax.grid(axis='y',alpha=.18,lw=.6); ax.spines[['top','right']].set_visible(False)
    ymax=max(df['Brier score'].max(),df['ECE'].max())
    ax.set_ylim(0, ymax*1.34)
    for i,row in df.iterrows():
        ax.text(i, ymax*1.20, f"Observed rate\n{row['Observed rate']:.2f}", ha='center', va='top', fontsize=7.2, color=GRAY,
                bbox=dict(boxstyle='round,pad=0.18', fc='white', ec='none', alpha=0.85))
    fig.tight_layout(pad=.8, rect=[0,0.08,1,1])
    save(fig,'result_calibration_story')

def external_formal():
    df=pd.read_csv(DATA/'public_table_external_validation.csv')
    fig,ax=plt.subplots(figsize=(7.20,4.15))
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
    ax.grid(axis='y',alpha=.18,lw=.6); ax.spines[['top','right']].set_visible(False)
    fig.tight_layout(pad=.8, rect=[0,0.10,1,1])
    save(fig,'result_external_validation_story')

if __name__ == '__main__':
    architecture_story(); mode_story(); evaluation_story()
    operating_formal(); calibration_formal(); external_formal()
    print('wrote formal story figures to', FIG)
