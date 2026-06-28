#!/usr/bin/env python3
"""Overlay the directional gas P(k,μ) of Enzo-GPU and RAMSES-GPU on the SAME axes, at 2× linear
resolution (256³), vs CICASS 2-fluid linear.  Shows the v_bc streaming signature directly: the
stream-PARALLEL modes (μ=k̂·v̂_bc → 1) are pressure-suppressed, the PERPENDICULAR modes (μ→0) are
not.  One panel per redshift; Enzo solid, RAMSES dashed, CICASS-linear dotted; blue=parallel,
red=perpendicular.  Optionally overlays the 128³ run (CIC_TAG2) faded, for resolution comparison.

Run:  CIC_TAG=_c256 CIC_NGRID=256 [CIC_TAG2=_c128 CIC_N2=128] <anaconda python3> plot_cicass_aniso_overlay.py
"""
import numpy as np, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

R = os.path.join(os.path.dirname(__file__), "..", "..", "..", "reports", "multicode") + "/"
L = float(os.environ.get("CIC_BOX", "0.128"))
TAG = os.environ.get("CIC_TAG", "_c256"); N = int(os.environ.get("CIC_NGRID", "256"))
TAG2 = os.environ.get("CIC_TAG2", ""); N2 = int(os.environ.get("CIC_N2", "128"))
MU_EDGES = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

def load_xspec(fn, n):
    raw = np.fromfile(fn, dtype=np.float64)
    nn = int(np.frombuffer(raw[:1].tobytes(), dtype=np.int64)[0])
    rb = raw[1:1+nn**3].reshape((nn, nn, nn), order="F")
    return nn, rb

def pk_mu(field, n):
    d = field/field.mean() - 1.0
    dk = np.fft.rfftn(d); pk3 = (dk*np.conj(dk)).real / field.size**2 * L**3
    kf = 2*np.pi/L
    kx = np.fft.fftfreq(n, 1.0/n)[:, None, None]; ky = np.fft.fftfreq(n, 1.0/n)[None, :, None]
    kz = np.fft.rfftfreq(n, 1.0/n)[None, None, :]
    kmag = np.sqrt(kx**2+ky**2+kz**2); mu = np.where(kmag > 0, np.abs(kx)/np.maximum(kmag, 1e-12), 0.0)*np.ones_like(kmag)
    kbins = np.arange(1, n//2)*kf; kcen = 0.5*(kbins[:-1]+kbins[1:])
    ik = np.digitize((kmag*kf).ravel(), kbins)-1; imu = np.digitize(mu.ravel(), MU_EDGES)-1; pr = pk3.ravel()
    P = np.full((len(kcen), 4), np.nan)
    for a in range(len(kcen)):
        for b in range(4):
            s = (ik == a) & (imu == b)
            if s.sum() > 0: P[a, b] = pr[s].mean()
    return kcen, P

def load_lin(fn):
    B={};cur=None;ks=[];Ps=[]
    for line in open(fn):
        s=line.strip()
        if not s or s.startswith("#"):continue
        if s.startswith("@"):
            if cur and ks:B[cur]=(np.array(ks),np.array(Ps))
            p=s.split();cur=(float(p[1].split("=")[1]),p[2]);ks=[];Ps=[]
        else:a,b=s.split()[:2];ks.append(float(a));Ps.append(float(b))
    if cur and ks:B[cur]=(np.array(ks),np.array(Ps))
    return B
lin = load_lin(R+"cicass_linear_pk.dat")
def lslice(z, mu):
    cand=[(zz,t) for (zz,t) in lin if t==f"baryon_mu{mu:.3f}"]
    if not cand:return None
    zn=min(cand,key=lambda x:abs(x[0]-z));return lin[zn]

ZS = [int(x) for x in os.environ.get("CIC_ZS","460,100,20").split(",")]
fig, axes = plt.subplots(1, len(ZS), figsize=(5.2*len(ZS), 4.6), squeeze=False)
mu_par, mu_perp = 0.875, 0.125; b_par, b_perp = 3, 0
for col, z in enumerate(ZS):
    ax = axes[0][col]
    for tag, n, alpha, lw in ([(TAG, N, 1.0, 1.8)] + ([(TAG2, N2, 0.35, 1.2)] if TAG2 else [])):
        for code, ls in (("enzo", "-"), ("ramses", "--")):
            fn = R+f"{code}_xspec{tag}_z{z}.bin"
            if not os.path.exists(fn): continue
            nn, rb = load_xspec(fn, n); kc, P = pk_mu(rb, nn)
            lbl = f"{'Enzo' if code=='enzo' else 'RAMSES'} {nn}³" if alpha==1.0 else None
            ax.loglog(kc, P[:, b_par],  ls, color="C0", lw=lw, alpha=alpha, label=(lbl+" ∥" if lbl else None))
            ax.loglog(kc, P[:, b_perp], ls, color="C3", lw=lw, alpha=alpha, label=(lbl+" ⊥" if lbl else None))
    for mu, c in ((mu_par,"C0"),(mu_perp,"C3")):
        s = lslice(z, mu)
        if s is not None: ax.loglog(s[0], s[1], ":", color=c, lw=2.2, alpha=0.9)
    ax.set_title(f"gas P(k,μ)  z={z}"); ax.set_xlabel("k [h/Mpc]")
    if col==0: ax.set_ylabel("P(k,μ) [(Mpc/h)³]")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=7, ncol=2)
fig.suptitle("Streaming anisotropy: stream-∥ (blue, suppressed) vs ⊥ (red) gas P(k) — "
             "Enzo-GPU (solid) & RAMSES-GPU (dashed) at 2× linear res; ⋯ = CICASS 2-fluid linear",
             fontsize=11)
fig.tight_layout(rect=[0,0,1,0.95]); out=R+"cicass_aniso_overlay.png"; fig.savefig(out, dpi=140)
print("wrote", out)
