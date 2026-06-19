#!/usr/bin/env python3
# Enzo-vs-RAMSES cell-by-cell scatter for rho, xHII, fH2, T from the cellcmp binaries.
# Binary format: int64 N, then N^3 float64 each of rho, xHII, fH2, fHD, T.
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterSciNotation
import os

REP = os.path.dirname(os.path.abspath(__file__))
TAG = os.environ.get("CIC_TAG", "_t64_me3x")
ZS  = [int(z) for z in os.environ.get("ZCOLS", "680,460,315,145,65,20").split(",")]
NSUB = 25000   # points drawn per panel (subsample for speed/clarity)

def read(path):
    with open(path, "rb") as f:
        N = np.fromfile(f, np.int64, 1)[0]
        n = N**3
        d = {k: np.fromfile(f, np.float64, n) for k in ("rho","xHII","fH2","fHD","T")}
    return d

FIELDS = [("rho","ρ  [code]"), ("xHII","x_HII"), ("fH2","f_H2"), ("T","T  [K]")]

fig, axes = plt.subplots(len(FIELDS), len(ZS), figsize=(3.0*len(ZS), 3.0*len(FIELDS)))
rng = np.random.default_rng(0)

for j, z in enumerate(ZS):
    rp = os.path.join(REP, f"ramses_cellcmp{TAG}_z{z}.bin")
    ep = os.path.join(REP, f"enzo_cellcmp{TAG}_z{z}.bin")
    if not (os.path.isfile(rp) and os.path.isfile(ep)):
        for i in range(len(FIELDS)):
            axes[i, j].text(0.5, 0.5, f"missing z={z}", ha="center")
        continue
    R, E = read(rp), read(ep)
    n = len(R["rho"])
    idx = rng.choice(n, size=min(NSUB, n), replace=False)
    for i, (key, lab) in enumerate(FIELDS):
        ax = axes[i, j]
        e = E[key][idx]; r = R[key][idx]
        good = np.isfinite(e) & np.isfinite(r) & (e > 0) & (r > 0)
        e, r = e[good], r[good]
        ax.scatter(e, r, s=2, alpha=0.06, lw=0, color="C0", rasterized=True)
        lo = min(e.min(), r.min()); hi = max(e.max(), r.max())
        pad = (hi/lo)**0.05 if lo > 0 else 1.0   # 5% log padding
        lo, hi = lo/pad, hi*pad
        ax.plot([lo, hi], [lo, hi], "r-", lw=0.8)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        # at most 3 decade ticks; compact sci-notation labels
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(LogLocator(numticks=4))
            axis.set_minor_locator(LogLocator(subs=(), numticks=4))
            axis.set_major_formatter(LogFormatterSciNotation(minor_thresholds=(2,0.5)))
        # median relative error annotation
        rel = np.median(np.abs(e - r) / (0.5*(np.abs(e)+np.abs(r)) + 1e-30))
        ax.text(0.05, 0.95, f"med {100*rel:.2f}%", transform=ax.transAxes,
                fontsize=8, va="top",
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.8))
        if i == 0:
            ax.set_title(f"z = {z}", fontsize=11)
        if j == 0:
            ax.set_ylabel(f"{lab}\nRAMSES", fontsize=9)
        if i == len(FIELDS)-1:
            ax.set_xlabel("Enzo", fontsize=9)
        ax.tick_params(labelsize=6, rotation=0)

fig.suptitle(f"Enzo vs RAMSES cell-by-cell  (tag {TAG}, da/a=0.03)  — red = y=x", fontsize=11)
fig.tight_layout(rect=[0,0,1,0.98])
out = os.path.join(REP, f"scatter_cellcmp{TAG}.png")
fig.savefig(out, dpi=130)
print("wrote", out)
