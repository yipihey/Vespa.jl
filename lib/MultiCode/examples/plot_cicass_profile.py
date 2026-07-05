#!/usr/bin/env python3
"""Spherical profiles around the gas-density peak vs ENCLOSED GAS MASS.

Reads a CICASS per-cell dump (Int64(N) header + 5 Float64 columns: rho, x_HII,
f_H2, f_HD, T[K]), finds the densest cell, sorts all cells by periodic radius
from it, and plots T, f_H2, n_H (+ overdensity), x_HII against the enclosed
gas mass M(<r) in Msun — all axes logarithmic.  Mass-weighted shell averages
in equal-log-mass bins.

NOTE: the dump is the LEVEL-0 restriction of the AMR hierarchy (cell = box/N),
so the innermost bins average over the finest-level structure — central
densities/temperatures are lower limits at the level-0 cell scale.

Run:  VESPA_RUN_DIR=<run dir> CIC_TAG=_bamr256L6e [CIC_DUMP=bamr_cellcmp]
      [CIC_BOX=0.256] [CIC_HUBBLE=0.71] [CIC_OMEGAB=0.046] [CIC_XH=0.76]
      <anaconda python3> plot_cicass_profile.py
"""
import numpy as np, os, glob, re
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

R    = (os.environ.get("VESPA_RUN_DIR") or os.getcwd()) + "/"
TAG  = os.environ.get("CIC_TAG", "_bamr256L6e")
PREF = os.environ.get("CIC_DUMP", "bamr_cellcmp")
BOX  = float(os.environ.get("CIC_BOX", "0.256"))          # cMpc/h
h    = float(os.environ.get("CIC_HUBBLE", "0.71"))
OmB  = float(os.environ.get("CIC_OMEGAB", "0.046"))
XH   = float(os.environ.get("CIC_XH", "0.76"))
MH   = 1.6726e-24                                          # g
RHOC0 = 1.8788e-29                                         # h^2 g/cm^3
MPC   = 3.0857e24                                          # cm
MSUN  = 1.989e33                                           # g

files = sorted(glob.glob(R + PREF + TAG + "_z*.bin"))
assert files, f"no {PREF}{TAG}_z*.bin in {R}"
fn = files[-1]
z  = int(re.search(r"_z(\d+)\.bin$", fn).group(1))
raw = np.fromfile(fn, dtype=np.float64)
N   = int(np.frombuffer(raw[:1].tobytes(), dtype=np.int64)[0]); m = N**3
rho, xhii, fh2, _, T = raw[1:1+5*m].reshape(5, m)
delta = rho / rho.mean()

# ── peak + periodic radii (index space; any unravel convention is consistent) ─
p = int(np.argmax(rho))
pi, pj, pk = np.unravel_index(p, (N, N, N))
ax = np.arange(N)
d1 = (ax - pi + N//2) % N - N//2
d2 = (ax - pj + N//2) % N - N//2
d3 = (ax - pk + N//2) % N - N//2
r2 = (d1[:, None, None]**2 + d2[None, :, None]**2 + d3[None, None, :]**2).ravel()
order = np.argsort(r2, kind="stable")
cell = BOX / N                                             # cMpc/h
r_s  = np.sqrt(r2[order].astype(np.float64)) * cell        # cMpc/h

# ── enclosed gas mass [Msun] ──────────────────────────────────────────────────
rhob_com = OmB * RHOC0 * h**2                              # g/cm^3 comoving
mcell_mean = rhob_com * (cell / h * MPC)**3 / MSUN         # Msun per mean-density cell
mass_s = delta[order] * mcell_mean
Menc   = np.cumsum(mass_s)

# ── equal-log-mass shells, mass-weighted averages ─────────────────────────────
lo = np.log10(Menc[0] * 1.5); hi = np.log10(Menc[-1] * 0.999)
edges = np.searchsorted(Menc, 10**np.linspace(lo, hi, 61))
edges = np.unique(np.clip(edges, 1, m))
Mx, Tb, H2b, XEb, NHb, DLb = [], [], [], [], [], []
for a, b in zip(np.r_[0, edges[:-1]], edges):
    if b <= a: continue
    sl = order[a:b]; w = mass_s[a:b]; W = w.sum()
    Mx.append(Menc[b-1])
    Tb.append((T[sl]*w).sum()/W); H2b.append((fh2[sl]*w).sum()/W)
    XEb.append((xhii[sl]*w).sum()/W)
    vol = (b - a) * (cell / h * MPC)**3                    # shell volume, cm^3 (comoving)
    dl = W * MSUN / vol / rhob_com                         # shell mean overdensity
    DLb.append(dl)
    NHb.append(dl * rhob_com * (1 + z)**3 * XH / MH)       # physical n_H, cm^-3
Mx, Tb, H2b, XEb, NHb, DLb = map(np.asarray, (Mx, Tb, H2b, XEb, NHb, DLb))

# ── 4 stacked log-log panels sharing the enclosed-mass axis ───────────────────
fig, axs = plt.subplots(4, 1, figsize=(6.5, 11), sharex=True)
panels = [
    (NHb, r"$n_{\rm H}$  [cm$^{-3}$] (physical)", "tab:blue"),
    (Tb,  r"$T$  [K]", "tab:red"),
    (H2b, r"$f_{\rm H_2}$", "tab:green"),
    (XEb, r"$x_{\rm HII}$", "tab:purple"),
]
for a, (y, lab, col) in zip(axs, panels):
    a.loglog(Mx, y, color=col, lw=1.8)
    a.set_ylabel(lab)
    a.grid(alpha=0.3, which="both")
ax2 = axs[0].twinx()
ax2.loglog(Mx, DLb, color="tab:blue", lw=0)                # align scales
ax2.set_ylabel(r"$\rho/\bar\rho$ (comoving)")
ax2.set_ylim(np.array(axs[0].get_ylim()) / (rhob_com * (1+z)**3 * XH / MH))
axs[-1].set_xlabel(r"enclosed gas mass  $M(<r)$  [M$_\odot$]")
axs[0].set_title(f"{PREF}{TAG}  z={z}   profiles around the density peak\n"
                 f"peak cell ({pi},{pj},{pk}), level-0 grid ({cell*1e3:.2f} kpc/h cells)")
fig.tight_layout()
out = R + f"cicass_profile{TAG}_z{z}.png"
fig.savefig(out, dpi=140)
print("peak delta =", delta[p], " r_max =", r_s[-1], "cMpc/h")
print("M_enc range:", Menc[0], "-", Menc[-1], "Msun")
print("wrote", out)
