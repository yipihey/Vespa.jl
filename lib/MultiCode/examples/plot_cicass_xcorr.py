#!/usr/bin/env python3
"""Enzo↔RAMSES cross-correlation r(k) = P_ER/√(P_E·P_R) at each output redshift, gas + DM.
The ICs are identical (r=1 at z=1000, all k); r(k) shows WHERE (scale) and WHEN (z) the two
solvers decorrelate as structure evolves. r=1 → mode-for-mode identical; r<1 → solver divergence.

Run (point at a run dir on scratch/archive):
    VESPA_RUN_DIR=/zpool/.../<run-id> CIC_BOX=0.128 python3 plot_cicass_xcorr.py
    # or:  python3 plot_cicass_xcorr.py <run-dir>
"""
import os, sys, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from vespa_io import open_run

rd = open_run()
L = float(os.environ.get("CIC_BOX", "0.128"))

def rk(a, b, n):
    da = np.fft.rfftn(a/a.mean()-1.0); db = np.fft.rfftn(b/b.mean()-1.0)
    kf = 2*np.pi/L
    kx=np.fft.fftfreq(n,1.0/n)[:,None,None];ky=np.fft.fftfreq(n,1.0/n)[None,:,None];kz=np.fft.rfftfreq(n,1.0/n)[None,None,:]
    km=(np.sqrt(kx**2+ky**2+kz**2)*kf).ravel()
    Pa=(da*np.conj(da)).real.ravel();Pb=(db*np.conj(db)).real.ravel();Px=(da*np.conj(db)).real.ravel()
    kb=np.arange(1,n//2)*kf; kc=0.5*(kb[:-1]+kb[1:]); ik=np.digitize(km,kb)-1
    kk=[];rr=[]
    for i in range(len(kc)):
        s=ik==i
        if s.sum()>0:
            pa,pb=Pa[s].mean(),Pb[s].mean()
            if pa>0 and pb>0: kk.append(kc[i]); rr.append(Px[s].mean()/np.sqrt(pa*pb))
    return np.array(kk), np.array(rr)

# redshifts present for BOTH codes' xspec dumps
ez = set(rd.redshifts("enzo_xspec")); rz = set(rd.redshifts("ramses_xspec"))
zs = sorted(ez & rz)
fig,ax=plt.subplots(1,2,figsize=(12,5))
cmap=plt.cm.viridis
print(f"Enzo↔RAMSES r(k) in {rd.path}.  z list: {zs[::-1]}")
print(f"{'z':>5} | {'gas r@kmin':>10} {'gas r@kmid':>10} {'gas r@kmax':>10} | {'dm r@kmin':>9} {'dm r@kmid':>9} {'dm r@kmax':>9}")
for j,z in enumerate(sorted(zs,reverse=True)):
    E=rd.grid("enzo_xspec",z); Rg=rd.grid("ramses_xspec",z); n=E["_N"]
    kg,rg=rk(E["rho_b"],Rg["rho_b"],n); kd,rdm=rk(E["rho_dm"],Rg["rho_dm"],n)
    c=cmap(j/max(1,len(zs)-1))
    ax[0].semilogx(kg,rg,'-',color=c,label=f"z={z}"); ax[1].semilogx(kd,rdm,'-',color=c,label=f"z={z}")
    q=lambda r,f:r[int(f*(len(r)-1))]
    print(f"{z:5d} | {q(rg,0):10.5f} {q(rg,0.5):10.5f} {q(rg,1):10.5f} | {q(rdm,0):9.5f} {q(rdm,0.5):9.5f} {q(rdm,1):9.5f}")
for a,t in ((ax[0],"gas (baryon)"),(ax[1],"dark matter")):
    a.set_title(f"Enzo↔RAMSES r(k) — {t}"); a.set_xlabel("k [h/Mpc]"); a.set_ylabel("cross-corr r(k)")
    a.axhline(1.0,color='k',lw=0.6,ls=':'); a.set_ylim(0.0,1.02); a.grid(alpha=0.3,which="both"); a.legend(fontsize=7,ncol=2)
fig.suptitle("Identical ICs (r=1 @z=1000); r(k) shows where/when the solvers decorrelate")
fig.tight_layout(rect=[0,0,1,0.96]); out=os.path.join(rd.path,"cicass_xcorr.png"); fig.savefig(out,dpi=140); print("wrote",out)
