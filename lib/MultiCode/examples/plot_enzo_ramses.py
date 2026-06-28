#!/usr/bin/env python
"""Enzo vs RAMSES on one CICASS realization: density & potential consistency.
Left: density P(k) (gas + DM) for both codes + CICASS linear DM.  Right:
mode-by-mode cross-correlation r(k) of the DM density and the gravitational
potential between the two codes (same realization -> r->1 on shared scales)."""
import os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
N=128; L=0.128; R=(os.environ.get("VESPA_RUN_DIR") or os.getcwd())+"/"  # run dir (set VESPA_RUN_DIR)

def load_pk(fn):
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
def load_bin(fn):
    d=np.fromfile(fn,dtype=np.float64); return d[:N**3].reshape(N,N,N), d[N**3:2*N**3].reshape(N,N,N)
kf=np.fft.fftfreq(N)*N
KX,KY,KZ=np.meshgrid(kf,kf,kf,indexing='ij'); km=np.sqrt(KX**2+KY**2+KZ**2)
def xcorr_k(a,b,nb=12):
    a=a-a.mean(); b=b-b.mean(); fa=np.fft.fftn(a); fb=np.fft.fftn(b)
    kb=np.linspace(1,N//2,nb+1); out=[]
    for i in range(nb):
        m=(km>=kb[i])&(km<kb[i+1])
        cr=np.real(fa[m]*np.conj(fb[m])).sum(); na=(np.abs(fa[m])**2).sum(); nv=(np.abs(fb[m])**2).sum()
        out.append((0.5*(kb[i]+kb[i+1])*2*np.pi/L, cr/np.sqrt(na*nv) if na*nv>0 else 0))
    return np.array(out)

enz=load_pk(R+"cicass_enzo_phi.dat"); ram=load_pk(R+"cicass_ramses_pk.dat"); lin=load_pk(R+"cicass_linear_pk.dat")
fig,(axp,axc)=plt.subplots(1,2,figsize=(13,5.6))
for B,tag,c,lbl in [(enz,(250.0,"dm"),"C0","Enzo DM"),(ram,(249.12,"dm"),"C1","RAMSES DM"),
                    (enz,(250.0,"baryon"),"C0","Enzo gas"),(ram,(249.12,"baryon"),"C1","RAMSES gas"),
                    (lin,(274.0,"dm"),"k","CICASS lin DM (z274)")]:
    if tag in B:
        k,P=B[tag]; ls="--" if "gas" in lbl else (":" if "lin" in lbl else "-")
        axp.loglog(k,P,ls,color=c,lw=1.8,label=lbl,alpha=0.85)
axp.set_xlabel(r"$k\ [h\,{\rm Mpc}^{-1}]$"); axp.set_ylabel(r"$P(k)\ [(h^{-1}{\rm Mpc})^3]$")
axp.set_title("Density power spectra at z$\\approx$250"); axp.legend(fontsize=8); axp.grid(which="both",alpha=0.15)
for ez,rz,c in [("z985","z957","C3"),("z392","z384","C2"),("z250","z249","C0")]:
    pe,de=load_bin(R+f"enzo_fields_{ez}.bin"); pr,dr=load_bin(R+f"ramses_fields_{rz}.bin")
    rd=xcorr_k(de,dr); rp=xcorr_k(pe,pr); zz=ez[1:]
    axc.semilogx(rd[:,0],rd[:,1],"-o",color=c,ms=3,label=f"DM z={zz}")
    axc.semilogx(rp[:,0],rp[:,1],"--s",color=c,ms=3,mfc="none",label=f"phi z={zz}")
axc.axhline(1,color="0.7",lw=0.8); axc.set_ylim(-0.05,1.05)
axc.set_xlabel(r"$k\ [h\,{\rm Mpc}^{-1}]$"); axc.set_ylabel("cross-correlation $r(k)$ Enzo x RAMSES")
axc.set_title("Mode-by-mode consistency (same realization)"); axc.legend(fontsize=7,ncol=2,loc="lower left"); axc.grid(which="both",alpha=0.15)
fig.suptitle("Enzo vs RAMSES on one CICASS realization: densities & potential",fontsize=12)
fig.tight_layout(); out=R+"enzo_ramses_consistency.png"; fig.savefig(out,dpi=140); print("wrote",out)
