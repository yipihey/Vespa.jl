#!/usr/bin/env python3
"""Cell-by-cell comparison of x_HII and T_gas between Enzo and RAMSES on one CICASS
realization (chem runs).  cellcmp dump = Int64 N, then ρ, x_HII, f_H2, f_HD, T (f64 N³).
Both on the same N³ grid (DM r(k)=1 confirms alignment).  Reports per-field median
ratio, cell correlation, and scatter at matched redshifts; writes scatter plots.
"""
import numpy as np, os, glob, re
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

R = os.path.join(os.path.dirname(__file__), "..", "..", "..", "reports", "multicode") + "/"

XH = 0.76
def grackle_mu(xHII, fH2):
    # reduced-network mean molecular weight (grackle calculate_temperature): neutral He,
    # n_e=n_HII; μ=1/[(X_H+Y/4)+X_H(x_HII−f_H2/2)].  → 1.22 neutral, ~1.17 at x_HII=0.047.
    return 1.0 / ((XH + (1.0 - XH) / 4.0) + XH * (xHII - 0.5 * fH2))

def load_cc(fn):
    raw = np.fromfile(fn, dtype=np.float64)
    n = int(np.frombuffer(raw[:1].tobytes(), dtype=np.int64)[0]); m = n**3
    # drivers now dump T already at the grackle species-μ (consistent across codes).
    return dict(N=n, rho=raw[1:1+m], xHII=raw[1+m:1+2*m], fH2=raw[1+2*m:1+3*m],
                fHD=raw[1+3*m:1+4*m], T=raw[1+4*m:1+5*m])

def zof(pref):
    out = {}
    for f in glob.glob(R + pref + "_z*.bin"):
        m = re.search(r"_z(\d+)\.bin$", f)
        if m and "run" not in f: out[int(m.group(1))] = f
    return out

enz = zof("enzo_cellcmp"); ram = zof("ramses_cellcmp")
print(f"Enzo cellcmp z: {sorted(enz)}\nRAMSES cellcmp z: {sorted(ram)}\n")

def stats(a, b):
    m = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    if m.sum() < 10: return None
    r = b[m] / a[m]
    cc = np.corrcoef(np.log(a[m]), np.log(b[m]))[0, 1]
    return np.median(r), np.exp(np.std(np.log(r))), cc, m.sum()

print(f"{'z(E/R)':>9} | {'xHII RAM/ENZ':>12} {'scatter':>8} {'corr':>6} | {'T RAM/ENZ':>10} {'scatter':>8} {'corr':>6}")
pairs = []
for zE in sorted(enz, reverse=True):
    zR = min(ram, key=lambda x: abs(x - zE)) if ram else None
    if zR is None: continue
    E = load_cc(enz[zE]); M = load_cc(ram[zR])
    sx = stats(E["xHII"], M["xHII"]); sT = stats(E["T"], M["T"])
    if sx and sT:
        print(f"{zE:4d}/{zR:<4d} | {sx[0]:12.3f} {sx[1]:8.3f} {sx[2]:6.3f} | "
              f"{sT[0]:10.3f} {sT[1]:8.3f} {sT[2]:6.3f}")
        pairs.append((zE, zR, E, M))

# scatter plots at a few z
if pairs:
    sel = [pairs[0], pairs[len(pairs)//2], pairs[-1]]
    fig, ax = plt.subplots(2, len(sel), figsize=(4*len(sel), 8), squeeze=False)
    for c, (zE, zR, E, M) in enumerate(sel):
        for row, key, lab in ((0, "xHII", "x_HII"), (1, "T", "T [K]")):
            a = E[key]; b = M[key]; m = np.isfinite(a)&np.isfinite(b)&(a>0)&(b>0)
            ss = np.random.default_rng(0).choice(np.where(m)[0], min(4000, m.sum()), replace=False)
            ax[row][c].loglog(a[ss], b[ss], ".", ms=1, alpha=0.3)
            lo = min(a[ss].min(), b[ss].min()); hi = max(a[ss].max(), b[ss].max())
            ax[row][c].plot([lo, hi], [lo, hi], "r-", lw=0.8)
            ax[row][c].set_title(f"{lab}  z={zE}"); ax[row][c].set_xlabel(f"Enzo {lab}")
            if c == 0: ax[row][c].set_ylabel(f"RAMSES {lab}")
    fig.tight_layout(); out = R + "cicass_cellcmp.png"; fig.savefig(out, dpi=140)
    print("\nwrote", out)
