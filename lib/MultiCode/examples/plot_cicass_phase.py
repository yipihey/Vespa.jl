#!/usr/bin/env python3
"""Phase diagrams (T vs overdensity) across codes to chase thermal-history differences.

Reads the cellcmp dumps (Int64 N, then ρ, x_HII, f_H2, f_HD, T — all on the same N³ grid,
T computed with the SAME grackle species-μ in every code, so directly comparable). For each
code and redshift it builds the median T(δ) relation in log-overdensity bins (+16/84 percentile
band) and the cell-by-cell T(δ) 2D histogram, so we can see WHERE the thermal histories diverge
(low-δ adiabatic floor vs high-δ shock/compression heating) despite identical chemistry+cooling.

Run:  CIC_TAG=_c128 <anaconda python3> plot_cicass_phase.py
"""
import numpy as np, os, glob, re
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

R = (os.environ.get("VESPA_RUN_DIR") or os.getcwd()) + "/"  # run dir on scratch/archive (set VESPA_RUN_DIR)
TAG = os.environ.get("CIC_TAG", "_c128")

def load(fn):
    raw = np.fromfile(fn, dtype=np.float64)
    n = int(np.frombuffer(raw[:1].tobytes(), dtype=np.int64)[0]); m = n**3
    rho = raw[1:1+m]; T = raw[1+4*m:1+5*m]
    return rho, T

def zof(pref):
    out = {}
    for f in glob.glob(R + pref + "_z*.bin"):
        mm = re.search(r"_z(\d+)\.bin$", f)
        if mm and "run" not in f: out[int(mm.group(1))] = f
    return out

CODES = [("Enzo-GPU", "enzo_cellcmp"+TAG, "C0"),
         ("RAMSES",   "ramses_cellcmp"+TAG, "C2"),
         ("Arepo",    "arepo_cellcmp"+TAG, "C3")]
CD = [(nm, zof(pref), c) for nm, pref, c in CODES]
CD = [(nm, zd, c) for nm, zd, c in CD if zd]

# overdensity bins (log), shared
DBINS = np.logspace(-1.0, 1.5, 26); DCEN = np.sqrt(DBINS[:-1]*DBINS[1:])
def relation(rho, T):
    d = rho/np.mean(rho); m = np.isfinite(d)&np.isfinite(T)&(T>0)&(d>0)
    d, T = d[m], T[m]
    idx = np.digitize(d, DBINS)-1
    med = np.full(len(DCEN), np.nan); lo = med.copy(); hi = med.copy()
    for b in range(len(DCEN)):
        s = idx == b
        if s.sum() > 20:
            med[b] = np.median(T[s]); lo[b] = np.percentile(T[s],16); hi[b] = np.percentile(T[s],84)
    return med, lo, hi

# choose redshifts present in all codes
zall = sorted(set.intersection(*[set(zd) for _, zd, _ in CD]), reverse=True) if CD else []
ZSEL = [z for z in (460, 100, 45, 20) if z in zall] or zall[:4]

# ── Figure 1: median T(δ) relations, one panel per z, all codes overlaid ──
fig, axs = plt.subplots(1, len(ZSEL), figsize=(4.4*len(ZSEL), 4.2), squeeze=False)
print(f"Median T(δ) by code/redshift  (δ = ρ/ρ̄):")
for j, z in enumerate(ZSEL):
    ax = axs[0][j]
    print(f"\n  z={z}:  δ-bin   " + "  ".join(f"{nm:>9}" for nm,_,_ in CD))
    rels = {}
    for nm, zd, c in CD:
        zc = min(zd, key=lambda x: abs(x-z)); rho, T = load(zd[zc]); med, lo, hi = relation(rho, T)
        rels[nm] = med
        ax.fill_between(DCEN, lo, hi, color=c, alpha=0.12)
        ax.loglog(DCEN, med, "-", color=c, lw=1.8, label=nm)
    for bi in (3, 8, 13, 18, 23):
        if bi < len(DCEN):
            print(f"    δ={DCEN[bi]:6.2f} " + "  ".join(f"{rels[nm][bi]:9.2f}" if np.isfinite(rels[nm][bi]) else f"{'--':>9}" for nm,_,_ in CD))
    ax.set_title(f"z = {z}"); ax.set_xlabel("ρ/ρ̄"); ax.grid(alpha=0.3, which="both")
    if j == 0: ax.set_ylabel("T [K] (median, ±16/84%)")
    ax.legend(fontsize=8)
fig.suptitle(f"Thermal history: median T(ρ) per code ({TAG[1:]}³ fixed-res, identical chem+cooling)", fontsize=12)
fig.tight_layout(rect=[0,0,1,0.95]); fn1 = R+f"cicass_phase_Tofrho{TAG}.png"; fig.savefig(fn1, dpi=140); print("\nwrote", fn1)

# ── Figure 2: 2D phase histograms (δ–T) per code at z=20 (or lowest ZSEL) ──
z0 = ZSEL[-1]
fig, axs = plt.subplots(1, len(CD), figsize=(4.6*len(CD), 4.2), squeeze=False)
Tb = np.logspace(0, 3.2, 80)
for j, (nm, zd, c) in enumerate(CD):
    zc = min(zd, key=lambda x: abs(x-z0)); rho, T = load(zd[zc]); d = rho/np.mean(rho)
    m = np.isfinite(d)&np.isfinite(T)&(T>0)&(d>0)
    ax = axs[0][j]
    ax.hist2d(np.log10(d[m]), np.log10(T[m]), bins=[np.log10(DBINS), np.log10(Tb)], cmap="viridis",
              norm=matplotlib.colors.LogNorm())
    ax.set_title(f"{nm}  z={zc}"); ax.set_xlabel("log₁₀ ρ/ρ̄")
    if j == 0: ax.set_ylabel("log₁₀ T [K]")
fig.suptitle(f"δ–T phase diagram at z={z0} (cell-by-cell)", fontsize=12)
fig.tight_layout(rect=[0,0,1,0.95]); fn2 = R+f"cicass_phase_hist{TAG}.png"; fig.savefig(fn2, dpi=140); print("wrote", fn2)
