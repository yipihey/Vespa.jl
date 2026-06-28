#!/usr/bin/env python3
"""Enzo vs RAMSES vs CICASS-linear, on ONE CICASS realization, at matched output z.

Reads three @-block P(k) tables written by the drivers:
  cicass_highz_pk.dat   (Enzo:  blocks 'dm','baryon', + theory overlays)
  cicass_ramses_pk.dat  (RAMSES: blocks 'dm','baryon')
  cicass_linear_pk.dat  (CICASS analytic 2-fluid LINEAR P(k) at the sim output z's)

For each Enzo output z it nearest-matches a RAMSES z and a CICASS-linear z, interpolates
all onto the Enzo k-grid (log-log), and reports band-averaged P(k) ratios:
  large scale = lowest LSBINS k-bins ; small scale = highest SSBINS k-bins.
Prints a headless table (DM + baryon) and writes a multi-panel PNG.
"""
import numpy as np, os, sys
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

R = (os.environ.get("VESPA_RUN_DIR") or os.getcwd()) + "/"  # run dir on scratch/archive (set VESPA_RUN_DIR)
LSBINS = int(os.environ.get("LSBINS", "4"))
SSBINS = int(os.environ.get("SSBINS", "8"))

def load_pk(fn):
    B = {}; cur = None; ks = []; Ps = []
    def flush():
        nonlocal cur, ks, Ps
        if cur and ks: B[cur] = (np.array(ks), np.array(Ps))
        ks, Ps = [], []
    if not os.path.exists(fn):
        print("MISSING", fn); return B
    for line in open(fn):
        s = line.strip()
        if not s or s.startswith("#"): continue
        if s.startswith("@"):
            flush(); p = s.split(); cur = (float(p[1].split("=")[1]), p[2])
        else:
            a, b = s.split()[:2]; ks.append(float(a)); Ps.append(float(b))
    flush(); return B

def zs_with(B, tag):
    return sorted({z for (z, t) in B if t == tag})

def get(B, z, tag):
    zz = zs_with(B, tag)
    if not zz: return None, None
    znear = min(zz, key=lambda x: abs(x - z))
    return znear, B[(znear, tag)]

def interp(k_to, k_from, P_from):
    lo = (k_from > 0) & (P_from > 0)
    return np.exp(np.interp(np.log(k_to), np.log(k_from[lo]), np.log(P_from[lo])))

def band(ratio, lo=True, n=LSBINS):
    sub = ratio[:n] if lo else ratio[-n:]
    sub = sub[np.isfinite(sub) & (sub > 0)]
    return float(np.exp(np.mean(np.log(sub)))) if len(sub) else float("nan")

enz = load_pk(R + "cicass_highz_pk.dat")
ram = load_pk(R + "cicass_ramses_pk.dat")
lin = load_pk(R + "cicass_linear_pk.dat")

ez = zs_with(enz, "dm")
print(f"Enzo dm output z: {ez}")
print(f"RAMSES dm z:      {zs_with(ram,'dm')}")
print(f"CICASS-lin dm z:  {zs_with(lin,'dm')}\n")

for comp in ("dm", "baryon"):
    print(f"================  {comp.upper()}  P(k) band ratios (geometric mean)  ================")
    print(f"{'z_enzo':>8} {'RAM/ENZ ls':>11} {'RAM/ENZ ss':>11} {'ENZ/LIN ls':>11} "
          f"{'RAM/LIN ls':>11} {'ENZ/LIN ss':>11} {'RAM/LIN ss':>11}")
    for z in ez:
        ke, Pe = enz[(z, comp)]
        zr, (kr, Pr) = get(ram, z, comp)
        zl, lret = get(lin, z, comp)
        Pr_i = interp(ke, kr, Pr)
        ram_enz = Pr_i / Pe
        if lret is not None:
            kl, Pl = lret
            Pl_i = interp(ke, kl, Pl)
            enz_lin = Pe / Pl_i; ram_lin = Pr_i / Pl_i
        else:
            enz_lin = ram_lin = np.full_like(Pe, np.nan)
        print(f"{z:8.1f} {band(ram_enz):11.3f} {band(ram_enz,False,SSBINS):11.3f} "
              f"{band(enz_lin):11.3f} {band(ram_lin):11.3f} "
              f"{band(enz_lin,False,SSBINS):11.3f} {band(ram_lin,False,SSBINS):11.3f}")
    print()

# ---- plot: P(k) at a few z, DM (top) + baryon (bottom), 3 curves each ----
zsel = [ez[i] for i in (0, len(ez)//2, len(ez)-1)] if len(ez) >= 3 else ez
fig, axes = plt.subplots(2, len(zsel), figsize=(4*len(zsel), 8), squeeze=False)
for col, z in enumerate(zsel):
    for row, comp in enumerate(("dm", "baryon")):
        ax = axes[row][col]
        ke, Pe = enz[(z, comp)]; ax.loglog(ke, Pe, "o-", ms=3, label="Enzo")
        zr, (kr, Pr) = get(ram, z, comp); ax.loglog(kr, Pr, "s--", ms=3, label=f"RAMSES z{zr:.0f}")
        klo, khi = ke[0], ke[-1]; yv = []
        for kk, PP in ((ke, Pe), (kr, Pr)):
            m = (kk >= klo) & (kk <= khi) & (PP > 0); yv += list(PP[m])
        zl, lret = get(lin, z, comp)
        if lret is not None:
            kl, Pl = lret; ax.loglog(kl, Pl, "-", color="k", lw=1, label=f"CICASS-lin z{zl:.0f}")
            m = (kl >= klo) & (kl <= khi) & (Pl > 0); yv += list(Pl[m])
        if yv: ax.set_ylim(min(yv)/1.6, max(yv)*1.6)       # snug to data in the visible k-window
        ax.set_xlim(klo, khi)
        ax.set_title(f"{comp} z={z:.0f}"); ax.set_xlabel("k [h/Mpc]")
        if col == 0: ax.set_ylabel("P(k) [(Mpc/h)^3]")
        ax.legend(fontsize=7)
fig.tight_layout(); out = R + "cicass_3way_pk.png"; fig.savefig(out, dpi=140)
print("wrote", out)
