#!/usr/bin/env python3
"""Whole-box H2 vs density scatter from a CICASS per-cell dump (patch or enzo).

Reads either layout (Int64(N) header + 5 Float64 columns of length N = NGRID^3):
  * patch_cellcmp / enzo_cellcmp : (ρb, x_HII, f_H2, f_HD, T[K])   -- default
  * enzo_phase (CIC_XSPEC=1)     : (ρ/ρ̄, n_H, T, f_H2, x_HII)
and plots whole-box f_H2 vs overdensity δ=ρ/ρ̄ as a 2D histogram (a 100M-cell "scatter"
is a density map), with the median f_H2(δ) relation + 16/84 band overlaid, per redshift.
Built for the fast analytic H+H2 chemistry runs (CIC_CHEM=analytic_h2, f16/u16 fast path).

Run:  VESPA_RUN_DIR=<run dir>  CIC_TAG=_c256pa_h2  [CIC_DUMP=patch_cellcmp] \
      <anaconda python3> plot_cicass_h2rho.py
"""
import numpy as np, os, glob, re
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

R    = (os.environ.get("VESPA_RUN_DIR") or os.getcwd()) + "/"
TAG  = os.environ.get("CIC_TAG", "_c256pa_h2")
PREF = os.environ.get("CIC_DUMP", "patch_cellcmp")       # patch_cellcmp | enzo_cellcmp | enzo_phase

def load_cells(fn):
    """5 Float64 columns of length N³ after an Int64(N) header (N = grid dim); (δ, f_H2)."""
    raw = np.fromfile(fn, dtype=np.float64)
    N = int(np.frombuffer(raw[:1].tobytes(), dtype=np.int64)[0]); m = N**3
    c = raw[1:1+5*m].reshape(5, m)
    if PREF == "enzo_phase":                             # rrel, nH, T, fH2, xHII
        delta, fH2 = c[0], c[3]
    else:                                                # cellcmp: rho, xHII, fH2, fHD, T
        delta, fH2 = c[0]/np.mean(c[0]), c[2]
    return delta, fH2

def zof(pref):
    out = {}                                             # tag-agnostic (patch adds an extra _ before TAG)
    for f in glob.glob(R + pref + "*_z*.bin"):
        mm = re.search(r"_z(\d+)\.bin$", f)
        if mm and "run" not in f:
            out[int(mm.group(1))] = f
    return out

zd = zof(PREF)
if not zd:
    raise SystemExit(f"no {PREF}*_z*.bin in {R}")
zs_all = sorted(zd, reverse=True)
ZSEL = [z for z in (145, 45, 30, 20) if z in zd] or zs_all[-4:]
print(f"{PREF}{TAG}: redshifts {zs_all}; plotting {ZSEL}")

DBINS = np.logspace(-1.5, 4.5, 120)                       # ρ/ρ̄ up to first-halo overdensities
FBINS = np.logspace(-12, -2, 120)                         # f_H2
DCEN  = np.sqrt(DBINS[:-1]*DBINS[1:])

fig, axs = plt.subplots(1, len(ZSEL), figsize=(4.6*len(ZSEL), 4.4), squeeze=False)
print("\nMedian f_H2(δ):")
for j, z in enumerate(ZSEL):
    delta, fH2 = load_cells(zd[z])
    m = np.isfinite(delta) & np.isfinite(fH2) & (delta > 0) & (fH2 > 0)
    dm, fm = delta[m], fH2[m]
    ax = axs[0][j]
    h = ax.hist2d(np.log10(dm), np.log10(fm), bins=[np.log10(DBINS), np.log10(FBINS)],
                  cmap="magma", norm=matplotlib.colors.LogNorm())
    # median relation + 16/84 band
    idx = np.digitize(dm, DBINS) - 1
    med = np.full(len(DCEN), np.nan); lo = med.copy(); hi = med.copy()
    for b in range(len(DCEN)):
        s = idx == b
        if s.sum() > 20:
            med[b] = np.median(fm[s]); lo[b] = np.percentile(fm[s], 16); hi[b] = np.percentile(fm[s], 84)
    ax.plot(np.log10(DCEN), np.log10(med), "c-", lw=1.8)
    ax.plot(np.log10(DCEN), np.log10(lo), "c--", lw=0.8, alpha=0.7)
    ax.plot(np.log10(DCEN), np.log10(hi), "c--", lw=0.8, alpha=0.7)
    ax.set_title(f"z = {z}   ({len(dm):,} cells)")
    ax.set_xlabel("log₁₀ ρ/ρ̄")
    if j == 0: ax.set_ylabel("log₁₀ f_H₂  (2 n_H₂ / n_H)")
    fig.colorbar(h[3], ax=ax, label="cells", fraction=0.046, pad=0.04)
    # console summary at a few overdensities
    print(f"  z={z}: " + "  ".join(
        f"δ={DCEN[b]:8.2f}→{med[b]:.2e}" for b in (20, 50, 80, 100) if b < len(DCEN) and np.isfinite(med[b])))

box = os.environ.get("CIC_BOX", "?"); ng = os.environ.get("CIC_NGRID", "?")
fig.suptitle(f"H₂ vs density (whole box, analytic chem)  {ng}³, {box} Mpc/h", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fn = R + f"cicass_h2rho{TAG}.png"; fig.savefig(fn, dpi=140)
print("\nwrote", fn)
