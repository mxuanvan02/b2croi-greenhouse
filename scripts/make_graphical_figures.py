#!/usr/bin/env python3
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon, Circle, Rectangle, Arc
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
FIG=ROOT/'assets'/'figures'
FIG.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
BLUE='#2563EB'; GREEN='#16A34A'; ORANGE='#F97316'; SLATE='#334155'; LIGHT='#F8FAFC'; RED='#EF4444'

def label_box(ax,x,y,w,h,text,fc,ec=SLATE,fs=10,weight='bold'):
    box=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.03,rounding_size=0.10',fc=fc,ec=ec,lw=1.3,zorder=3)
    ax.add_patch(box)
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fs,fontweight=weight,color='#0F172A',zorder=4)

def arrow(ax,a,b,color=SLATE,rad=0,lw=1.8):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=16,lw=lw,color=color,connectionstyle=f'arc3,rad={rad}',zorder=2))

def greenhouse(ax):
    ax.add_patch(Rectangle((0.6,1.3),2.4,2.5,fc='#ECFDF5',ec=GREEN,lw=1.4,zorder=1))
    roof=Polygon([[0.45,3.8],[1.8,4.75],[3.15,3.8]],closed=True,fc='#D1FAE5',ec=GREEN,lw=1.4,zorder=1)
    ax.add_patch(roof)
    for x in [1.0,1.55,2.1,2.65]: ax.plot([x,x],[1.35,3.75],color='#A7F3D0',lw=.8,zorder=1)
    for x,y in [(1.05,2.1),(1.75,2.85),(2.45,2.05)]:
        ax.add_patch(Circle((x,y),.13,fc=GREEN,ec='white',lw=1.0,zorder=4))
        ax.plot([x,x],[y-.13,y-.55],color=GREEN,lw=1.4,zorder=4)
    ax.text(1.8,1.0,'Greenhouse sensors',ha='center',va='center',fontsize=11,fontweight='bold',color=GREEN)

def wireless(ax):
    cx,cy=4.45,3.0
    for r,alpha in [(0.38,.95),(0.72,.75),(1.06,.55)]:
        ax.add_patch(Arc((cx,cy),2*r,2*r,theta1=-45,theta2=45,color=ORANGE,lw=2,alpha=alpha,zorder=2))
    ax.add_patch(Circle((cx,cy),.08,fc=ORANGE,ec=ORANGE,zorder=3))
    label_box(ax,3.55,1.55,1.8,.66,'Bursty wireless link','#FFF7ED',ec=ORANGE,fs=9.5)
    ax.text(4.45,2.1,'packet loss bursts',ha='center',fontsize=8.5,color=ORANGE)

def scheduler(ax):
    label_box(ax,6.0,2.35,2.25,1.18,'B2CRoI-H(Q)\nscheduler','#DBEAFE',ec=BLUE,fs=11)
    for i,c in enumerate([BLUE,GREEN,ORANGE]):
        ax.add_patch(Circle((6.35+i*.38,3.78),.10,fc=c,ec='white',lw=.7,zorder=4))
    ax.text(7.13,2.15,'risk + burst belief + fairness',ha='center',fontsize=8.5,color=SLATE)

def outputs(ax):
    label_box(ax,9.35,3.38,2.1,.74,'Controller / digital twin','#F8FAFC',ec=SLATE,fs=9.5)
    label_box(ax,9.35,1.78,2.1,.74,'Dashboard','#F8FAFC',ec=SLATE,fs=10)
    ax.add_patch(Rectangle((9.72,3.55),.26,.18,fc=BLUE,ec='none',zorder=5)); ax.add_patch(Rectangle((10.08,3.50),.26,.23,fc=GREEN,ec='none',zorder=5)); ax.add_patch(Rectangle((10.44,3.45),.26,.28,fc=ORANGE,ec='none',zorder=5))
    ax.plot([9.75,10.0,10.25,10.55,10.85],[2.07,2.2,2.03,2.28,2.16],color=BLUE,lw=1.6,zorder=5)

def main():
    fig,ax=plt.subplots(figsize=(11,5.6))
    ax.set_xlim(0,12); ax.set_ylim(.4,5.4); ax.axis('off')
    ax.text(6,5.22,'Safety-aware greenhouse sensing under bursty wireless connectivity',ha='center',va='top',fontsize=16,fontweight='bold',color='#0F172A')
    greenhouse(ax); wireless(ax); scheduler(ax); outputs(ax)
    arrow(ax,(3.05,3.0),(3.55,3.0),GREEN,lw=2.0)
    arrow(ax,(5.35,3.0),(6.0,3.0),ORANGE,lw=2.0)
    arrow(ax,(8.25,3.0),(9.35,3.78),BLUE,rad=.08,lw=2.0)
    arrow(ax,(8.25,2.85),(9.35,2.15),BLUE,rad=-.08,lw=2.0)
    label_box(ax,7.9,4.15,1.35,.50,'Top-B updates','#E0F2FE',ec=BLUE,fs=9.5)
    arrow(ax,(8.55,4.15),(9.35,4.02),BLUE,rad=-.05,lw=1.6)
    ax.text(6,.62,'Only the most safety-informative updates are transmitted when bandwidth and reliability are limited.',ha='center',fontsize=10,color=SLATE)
    fig.tight_layout(pad=.25)
    fig.savefig(FIG/'b2croi_architecture_graphical.pdf',bbox_inches='tight')
    fig.savefig(FIG/'b2croi_architecture_graphical.png',dpi=350,bbox_inches='tight')
    print(FIG/'b2croi_architecture_graphical.png')
if __name__=='__main__': main()
