#!/usr/bin/env python
"""Baryon & dark-matter density power spectra from the CICASS high-z Vespa GPU run,
overlaid on the CICASS linear-theory prediction (IC power grown by D(a)^2), at
logarithmic intervals in scale factor z=1000 -> 20.  Markers = measured (GPU FFT of
the live Enzo gas density / CIC of the live DM particles); solid lines = linear theory."""
import os, sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import matplotlib.cm as cm
sys.path.insert(0, os.path.dirname(__file__))
from vespa_io import open_run

# Run dir on scratch/archive (set VESPA_RUN_DIR, or pass the dir as argv[1]).
rd = open_run()
blocks = rd.pk(pattern="cicass_highz_pk*.dat")   # {(z, tag) -> (k[], P[])}

zs = sorted({z for (z, _t) in blocks}, reverse=True)   # high z first
colors = cm.viridis(np.linspace(0, 0.9, len(zs)))

fig, (axb, axd) = plt.subplots(1, 2, figsize=(12.5, 5.6), sharex=True, sharey=True)
for ax, comp, th, thc, title in (
        (axb, "baryon", "theory_b", "theory_b_cic", "Baryons (gas density)"),
        (axd, "dm", "theory_dm", "theory_dm_cic", "Dark matter (particles, CIC)")):
    for z, c in zip(zs, colors):
        if (z, comp) in blocks:
            k, P = blocks[(z, comp)]
            ax.loglog(k, P, ls="none", marker="o", ms=4, mfc="none", mec=c,
                      label=f"z={z:.0f}")
        if (z, th) in blocks:
            k, P = blocks[(z, th)]
            ax.loglog(k, P, color=c, lw=1.8, alpha=0.9)
        if (z, thc) in blocks:
            k, P = blocks[(z, thc)]
            ax.loglog(k, P, color=c, lw=1.2, ls="--", alpha=0.7)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(r"$k\ [h\,{\rm Mpc}^{-1}]$")
    ax.grid(which="both", alpha=0.15)
axb.set_ylabel(r"$P(k)\ [(h^{-1}{\rm Mpc})^3]$")
axb.legend(title="○ Enzo GPU (measured)\n— IC×D(a)²   -- CICASS analytic×D(a)²",
           fontsize=8, ncol=2, loc="lower left", title_fontsize=8)
fig.suptitle("CICASS streaming box, Vespa GPU (PPM+Poisson Metal) + H+D chemistry: "
             "measured vs linear-theory $P(k)$", fontsize=12)
fig.tight_layout()
out = os.path.join(rd.path, "cicass_highz_pk.png")
fig.savefig(out, dpi=140); print("wrote", out)

# agreement report over a robust large-scale band (mean of the lowest few k-bins,
# avoiding the single-mode k_min) — measured vs realization-grown linear theory.
def band_ratio(meas, theo, nlo=2, nhi=6):
    km, Pm = meas; kt, Pt = theo
    a = np.mean(Pm[nlo:nhi]); b = np.mean(Pt[nlo:nhi])
    return a/b if b > 0 else float("nan")
print("\n  z      DM P/Plin   baryon P/Plin   (large-scale band, vs IC-grown linear)")
for z in zs:
    rd = band_ratio(blocks[(z,"dm")], blocks[(z,"theory_dm")]) if (z,"dm") in blocks and (z,"theory_dm") in blocks else float("nan")
    rb = band_ratio(blocks[(z,"baryon")], blocks[(z,"theory_b")]) if (z,"baryon") in blocks and (z,"theory_b") in blocks else float("nan")
    # Meszaros CDM-growth expectation a^0.90 vs a^1 (f_cdm~0.84), normalized at z_start
    print(f"  {z:6.1f}  {rd:9.3f}   {rb:11.3f}")
