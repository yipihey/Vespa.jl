#!/usr/bin/env python
"""Simulation P(k) (EnzoNG GPU, Metal hydro+gravity, H+D chemistry) vs CICASS
linear theory AT THE SAME output redshifts.  Markers = measured; solid lines =
CICASS analytic linear P(k) generated at each z (2-fluid transfer, so the baryon
catch-up after recombination is included).  Left = baryons, right = DM."""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import matplotlib.cm as cm

def load(fn):
    B={};cur=None;ks=[];Ps=[]
    def flush():
        nonlocal cur,ks,Ps
        if cur and ks: B[cur]=(np.array(ks),np.array(Ps))
        ks,Ps=[],[]
    for line in open(fn):
        line=line.strip()
        if not line or line.startswith("#"): continue
        if line.startswith("@"): flush();p=line.split();cur=(float(p[1].split("=")[1]),p[2])
        else: a,b=line.split();ks.append(float(a));Ps.append(float(b))
    flush();return B

R="/Users/tabel/Projects/enzo-dev/EnzoNG.jl/reports/multicode/"
sim=load(R+"cicass_highz_pk.dat"); lin=load(R+"cicass_linear_pk.dat")
simz=sorted({k[0] for k in sim},reverse=True)
linz=sorted({k[0] for k in lin},reverse=True)
colors=cm.viridis(np.linspace(0,0.9,len(simz)))

fig,(axb,axd)=plt.subplots(1,2,figsize=(13,5.6),sharex=True,sharey=True)
for ax,comp,title in ((axb,"baryon","Baryons (gas density)"),(axd,"dm","Dark matter (particles)")):
    for sz,c in zip(simz,colors):
        lz=min(linz,key=lambda z:abs(z-sz))
        if (sz,comp) in sim:
            k,P=sim[(sz,comp)]; ax.loglog(k,P,ls="none",marker="o",ms=4,mfc="none",mec=c,label=f"z={sz:.0f}")
        if (lz,comp) in lin:
            k,P=lin[(lz,comp)]; ax.loglog(k,P,color=c,lw=1.8,alpha=0.9)
    ax.set_title(title,fontsize=11); ax.set_xlabel(r"$k\ [h\,{\rm Mpc}^{-1}]$")
    ax.grid(which="both",alpha=0.15)
axb.set_ylabel(r"$P(k)\ [(h^{-1}{\rm Mpc})^3]$")
axb.legend(title="markers = EnzoNG GPU sim\nlines = CICASS linear (same z)",fontsize=8,ncol=2,loc="lower left",title_fontsize=8)
fig.suptitle("CICASS 128 kpc/h box: EnzoNG GPU sim vs CICASS linear theory at matched redshifts (z=1000→75)",fontsize=12)
fig.tight_layout()
out=R+"cicass_vs_linear.png"; fig.savefig(out,dpi=140); print("wrote",out)

# baryon catch-up + DM tracking, large-scale band
def band(B,key,nlo=1,nhi=8):
    if key not in B: return np.nan
    k,P=B[key]; return np.mean(P[nlo:nhi])
print(f"\n{'z':>6} | {'DM sim/lin':>10} | {'bary sim/lin':>12} | {'bary/dm sim':>11} {'bary/dm lin':>11}")
for sz in simz:
    lz=min(linz,key=lambda z:abs(z-sz))
    dms,dml=band(sim,(sz,"dm")),band(lin,(lz,"dm"))
    bas,bal=band(sim,(sz,"baryon")),band(lin,(lz,"baryon"))
    print(f"{sz:6.0f} | {dms/dml:10.3f} | {bas/bal:12.3f} | {bas/dms:11.4f} {bal/dml:11.4f}")
