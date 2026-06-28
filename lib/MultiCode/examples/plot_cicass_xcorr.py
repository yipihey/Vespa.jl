#!/usr/bin/env python3
"""Enzo↔RAMSES cross-correlation r(k) = P_ER/√(P_E·P_R) at each output redshift, gas + DM.
The ICs are identical (r=1 at z=1000, all k); r(k) shows WHERE (scale) and WHEN (z) the two
solvers decorrelate as structure evolves. r=1 → mode-for-mode identical; r<1 → solver divergence.

Run:  CIC_TAG=_c256 CIC_BOX=0.128 <anaconda python3> plot_cicass_xcorr.py
"""
import numpy as np, os, glob, re
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

R = os.path.join(os.path.dirname(__file__), "..", "..", "..", "reports", "multicode") + "/"
L = float(os.environ.get("CIC_BOX", "0.128")); TAG = os.environ.get("CIC_TAG", "_c256")

def load(fn):
    raw = np.fromfile(fn, dtype=np.float64); n = int(np.frombuffer(raw[:1].tobytes(), dtype=np.int64)[0]); m = n**3
    return n, raw[1:1+m].reshape((n,n,n),order="F"), raw[1+m:1+2*m].reshape((n,n,n),order="F")

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

zs=[]
for f in glob.glob(R+f"enzo_xspec{TAG}_z*.bin"):
    mm=re.search(r"_z(\d+)\.bin$",f);
    if mm and os.path.exists(R+f"ramses_xspec{TAG}_z{mm.group(1)}.bin"): zs.append(int(mm.group(1)))
zs=sorted(set(zs))
fig,ax=plt.subplots(1,2,figsize=(12,5))
cmap=plt.cm.viridis
print(f"Enzo↔RAMSES r(k) {TAG[1:]}.  z list: {zs[::-1]}")
print(f"{'z':>5} | {'gas r@kmin':>10} {'gas r@kmid':>10} {'gas r@kmax':>10} | {'dm r@kmin':>9} {'dm r@kmid':>9} {'dm r@kmax':>9}")
for j,z in enumerate(sorted(zs,reverse=True)):
    n,Eb,Ed=load(R+f"enzo_xspec{TAG}_z{z}.bin"); _,Rb,Rd=load(R+f"ramses_xspec{TAG}_z{z}.bin")
    kg,rg=rk(Eb,Rb,n); kd,rd=rk(Ed,Rd,n)
    c=cmap(j/max(1,len(zs)-1))
    ax[0].semilogx(kg,rg,'-',color=c,label=f"z={z}"); ax[1].semilogx(kd,rd,'-',color=c,label=f"z={z}")
    q=lambda r,f:r[int(f*(len(r)-1))]
    print(f"{z:5d} | {q(rg,0):10.5f} {q(rg,0.5):10.5f} {q(rg,1):10.5f} | {q(rd,0):9.5f} {q(rd,0.5):9.5f} {q(rd,1):9.5f}")
for a,t in ((ax[0],"gas (baryon)"),(ax[1],"dark matter")):
    a.set_title(f"Enzo↔RAMSES r(k) — {t}"); a.set_xlabel("k [h/Mpc]"); a.set_ylabel("cross-corr r(k)")
    a.axhline(1.0,color='k',lw=0.6,ls=':'); a.set_ylim(0.0,1.02); a.grid(alpha=0.3,which="both"); a.legend(fontsize=7,ncol=2)
fig.suptitle(f"Identical ICs (r=1 @z=1000); r(k) shows where/when the solvers decorrelate ({TAG[1:]})")
fig.tight_layout(rect=[0,0,1,0.96]); out=R+f"cicass_xcorr{TAG}.png"; fig.savefig(out,dpi=140); print("wrote",out)
