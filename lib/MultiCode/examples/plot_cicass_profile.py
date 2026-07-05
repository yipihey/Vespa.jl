#!/usr/bin/env python3
"""Spherical profiles around the gas-density peak vs ENCLOSED GAS MASS.

Sources (newest z of each in VESPA_RUN_DIR):
  * bamr_leafprof{TAG}_z*.bin — LEAF cells of ALL AMR levels within R_prof of
    the peak (Int64 n + 8×Float64[n]: x,y,z,dx box units, rho code, x_HII,
    f_H2, T[K]); written by the driver with the cut on the containing level-0
    cell center, so it is the exact complement of…
  * {CIC_DUMP}{TAG}_z*.bin — the restricted level-0 grid (Int64 N + 5×F64[N³]:
    rho, x_HII, f_H2, f_HD, T), used for r > R_prof.
Without a leafprof file it falls back to level-0 only (previous behaviour).

Plots T, f_H2, n_H (physical, + comoving-overdensity axis), x_HII against the
enclosed gas mass M(<r) [Msun], mass-weighted shell averages in equal-log-mass
bins — all axes logarithmic.

Run:  VESPA_RUN_DIR=<run dir> CIC_TAG=_bamr256L6e [CIC_DUMP=bamr_cellcmp]
      [CIC_BOX=0.256] [CIC_PROFR=0.1875] [CIC_HUBBLE=0.71] [CIC_OMEGAB=0.046]
      <anaconda python3> plot_cicass_profile.py
"""
import numpy as np, os, glob, re
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

R    = (os.environ.get("VESPA_RUN_DIR") or os.getcwd()) + "/"
TAG  = os.environ.get("CIC_TAG", "_bamr256L6e")
PREF = os.environ.get("CIC_DUMP", "bamr_cellcmp")
BOX  = float(os.environ.get("CIC_BOX", "0.256"))          # cMpc/h
RP   = float(os.environ.get("CIC_PROFR", "0.1875"))       # box units
h    = float(os.environ.get("CIC_HUBBLE", "0.71"))
OmB  = float(os.environ.get("CIC_OMEGAB", "0.046"))
XH   = float(os.environ.get("CIC_XH", "0.76"))
MH   = 1.6726e-24; RHOC0 = 1.8788e-29; MPC = 3.0857e24; MSUN = 1.989e33

fn = sorted(glob.glob(R + PREF + TAG + "_z*.bin"))[-1]
z  = int(re.search(r"_z(\d+)\.bin$", fn).group(1))
raw = np.fromfile(fn, dtype=np.float64)
N   = int(np.frombuffer(raw[:1].tobytes(), dtype=np.int64)[0]); m = N**3
rho0, xh0, f20, _, T0 = raw[1:1+5*m].reshape(5, m)
rhomean = rho0.mean()

# peak = densest level-0 cell (same argmax the driver used for the leaf cut)
p = int(np.argmax(rho0))
# level-0 dump layout: gidx = gi + (gj-1)N + (gk-1)N² (i fastest) → order='F'-ish
pi = p % N; pj = (p // N) % N; pk = p // N**2
pc = np.array([(pi + 0.5) / N, (pj + 0.5) / N, (pk + 0.5) / N])

def perdist(x, y, zc):
    return np.sqrt(((x - pc[0] + 0.5) % 1 - 0.5)**2 +
                   ((y - pc[1] + 0.5) % 1 - 0.5)**2 +
                   ((zc - pc[2] + 0.5) % 1 - 0.5)**2)

# level-0 cell coordinates (dump order: i fastest)
ii = np.arange(m) % N; jj = (np.arange(m) // N) % N; kk = np.arange(m) // N**2
x0 = (ii + 0.5) / N; y0 = (jj + 0.5) / N; z0c = (kk + 0.5) / N
r0 = perdist(x0, y0, z0c)

lf = sorted(glob.glob(R + "bamr_leafprof" + TAG + f"_z{z}.bin"))
if lf:
    lraw = np.fromfile(lf[-1], dtype=np.float64)
    n = int(np.frombuffer(lraw[:1].tobytes(), dtype=np.int64)[0])
    lx, ly, lz, ldx, lrho, lxh, lf2, lT = lraw[1:1+8*n].reshape(8, n)
    ext = r0 > RP                                          # exterior: restricted level 0
    xs  = np.r_[lx, x0[ext]];  ys = np.r_[ly, y0[ext]];  zs = np.r_[lz, z0c[ext]]
    dxs = np.r_[ldx, np.full(ext.sum(), 1.0 / N)]
    rhs = np.r_[lrho, rho0[ext]]; xhs = np.r_[lxh, xh0[ext]]
    f2s = np.r_[lf2, f20[ext]];   Ts  = np.r_[lT, T0[ext]]
    src = f"leaf cells (all levels, r<{RP:.3f}) + level-0 exterior; finest dx={ldx.min()*BOX*1e6:.0f} pc/h"
else:
    xs, ys, zs, dxs = x0, y0, z0c, np.full(m, 1.0 / N)
    rhs, xhs, f2s, Ts = rho0, xh0, f20, T0
    src = "level-0 restriction only"

r = perdist(xs, ys, zs)
order = np.argsort(r, kind="stable")

# ── enclosed gas mass [Msun]: M_cell = (rho/rhomean)·dx³·ρ̄b·V_box ────────────
rhob_com = OmB * RHOC0 * h**2                              # g/cm³ comoving
Mbox = rhob_com * (BOX / h * MPC)**3 / MSUN                # Msun
mass_s = (rhs[order] / rhomean) * dxs[order]**3 * Mbox
Menc = np.cumsum(mass_s)
r_s  = r[order]

# ── equal-log-mass shells, mass-weighted averages ─────────────────────────────
lo = np.log10(Menc[max(4, np.searchsorted(Menc, Menc[0]*3))])
edges = np.searchsorted(Menc, 10**np.linspace(lo, np.log10(Menc[-1]*0.999), 75))
edges = np.unique(np.clip(edges, 1, len(Menc)))
Mx, Tb, H2b, XEb, NHb, DLb = [], [], [], [], [], []
for a, b in zip(np.r_[0, edges[:-1]], edges):
    if b <= a: continue
    sl = order[a:b]; w = mass_s[a:b]; W = w.sum()
    Mx.append(Menc[b-1])
    Tb.append((Ts[sl]*w).sum()/W); H2b.append((f2s[sl]*w).sum()/W)
    XEb.append((xhs[sl]*w).sum()/W)
    vol = (dxs[sl]**3).sum() * (BOX / h * MPC)**3          # true leaf shell volume
    dl = W * MSUN / vol / rhob_com
    DLb.append(dl); NHb.append(dl * rhob_com * (1 + z)**3 * XH / MH)
Mx, Tb, H2b, XEb, NHb, DLb = map(np.asarray, (Mx, Tb, H2b, XEb, NHb, DLb))

fig, axs = plt.subplots(4, 1, figsize=(6.5, 11), sharex=True)
for a, (y, lab, col) in zip(axs, [
        (NHb, r"$n_{\rm H}$  [cm$^{-3}$] (physical)", "tab:blue"),
        (Tb,  r"$T$  [K]", "tab:red"),
        (H2b, r"$f_{\rm H_2}$", "tab:green"),
        (XEb, r"$x_{\rm HII}$", "tab:purple")]):
    a.loglog(Mx, y, color=col, lw=1.8)
    a.set_ylabel(lab); a.grid(alpha=0.3, which="both")
ax2 = axs[0].twinx()
ax2.set_yscale("log")
ax2.set_ylim(np.array(axs[0].get_ylim()) / (rhob_com * (1+z)**3 * XH / MH))
ax2.set_ylabel(r"$\rho/\bar\rho$ (comoving)")
axs[-1].set_xlabel(r"enclosed gas mass  $M(<r)$  [M$_\odot$]")
axs[0].set_title(f"{TAG.lstrip('_')}  z={z}   profiles around the density peak\n{src}",
                 fontsize=10)
fig.tight_layout()
out = R + f"cicass_profile{TAG}_z{z}.png"
fig.savefig(out, dpi=140)
print("peak delta(level0) =", rho0[p]/rhomean, " leaf cells:", (len(lx) if lf else 0))
print("central bin: nH=%.3g cm^-3  T=%.1f K  fH2=%.3g" % (NHb[0], Tb[0], H2b[0]))
print("M_enc range:", Menc[0], "-", Menc[-1], "Msun")
print("wrote", out)
