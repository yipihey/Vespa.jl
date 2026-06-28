#!/usr/bin/env python3
"""Early-redshift 3-code comparison while the Arepo CPU run is still finishing.

Same CICASS realization, matched output z.  Restricts the comparison to the
redshifts the (in-progress) Arepo run has ALREADY written, so all three codes
plus the CICASS-linear 2-fluid prediction are shown on the same panels with no
nearest-z snapping.  Reads:
  cicass_highz_pk{TAG}.dat   Enzo  (GPU)
  cicass_ramses_pk{TAG}.dat  RAMSES (GPU)
  cicass_arepo_pk{TAG}.dat   Arepo (CPU, partial)   [override with AREPO_DAT]
  cicass_linear_pk.dat       CICASS analytic 2-fluid linear
"""
import numpy as np, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

R = (os.environ.get("VESPA_RUN_DIR") or os.getcwd()) + "/"  # run dir on scratch/archive (set VESPA_RUN_DIR)
TAG = os.environ.get("CIC_TAG", "_c64")
LSBINS = int(os.environ.get("LSBINS", "4")); SSBINS = int(os.environ.get("SSBINS", "8"))
AREPO_DAT = os.environ.get("AREPO_DAT", R + f"cicass_arepo_pk{TAG}.dat")

CODES = [("Enzo",  R + f"cicass_highz_pk{TAG}.dat",  "o-",  "C0"),
         ("RAMSES", R + f"cicass_ramses_pk{TAG}.dat", "s--", "C1"),
         ("Arepo",  AREPO_DAT,                         "^:",  "C2")]

def load_pk(fn):
    B, cur, ks, Ps = {}, None, [], []
    def flush():
        nonlocal cur, ks, Ps
        if cur and ks: B[cur] = (np.array(ks), np.array(Ps))
        ks, Ps = [], []
    if not os.path.exists(fn): return None
    for line in open(fn):
        s = line.strip()
        if not s or s.startswith("#"): continue
        if s.startswith("@"):
            flush(); p = s.split(); cur = (float(p[1].split("=")[1]), p[2])
        else:
            a, b = s.split()[:2]; ks.append(float(a)); Ps.append(float(b))
    flush(); return B

def zs_with(B, tag): return sorted({z for (z, t) in B if t == tag}, reverse=True)
def getz(B, z, tag, rtol=1e-3):
    if B is None: return None
    zz = zs_with(B, tag)
    if not zz: return None
    zn = min(zz, key=lambda x: abs(x - z))
    return B[(zn, tag)] if abs(zn - z) <= rtol * (1 + z) else None
def interp(k_to, k_from, P_from):
    lo = (k_from > 0) & (P_from > 0)
    if lo.sum() < 2: return np.full_like(k_to, np.nan)
    return np.exp(np.interp(np.log(k_to), np.log(k_from[lo]), np.log(P_from[lo])))
def band(r, lo=True, n=LSBINS):
    sub = r[:n] if lo else r[-n:]
    sub = sub[np.isfinite(sub) & (sub > 0)]
    return float(np.exp(np.mean(np.log(sub)))) if len(sub) else float("nan")

data = {lab: load_pk(fn) for (lab, fn, _, _) in CODES}
lin  = load_pk(R + "cicass_linear_pk.dat")

# canonical early z = the ones Arepo has finished (intersect with Enzo+RAMSES)
az = zs_with(data["Arepo"], "dm") if data["Arepo"] else []
zlist = [z for z in az
         if getz(data["Enzo"], z, "dm") is not None
         and getz(data["RAMSES"], z, "dm") is not None]
print(f"TAG={TAG}  Arepo-available z (3-code): {zlist}")
for lab, fn, _, _ in CODES:
    print(f"  {lab:7s} {'ok' if data[lab] else 'MISSING':6s} {os.path.basename(fn)}")
print()

for comp in ("dm", "baryon"):
    print(f"========  {comp.upper()}  band-ratio vs CICASS-linear (geo-mean) + Arp/Enz  ========")
    print(f"{'z':>7} {'Enz/lin ls':>11} {'Ram/lin ls':>11} {'Arp/lin ls':>11} "
          f"{'Enz/lin ss':>11} {'Ram/lin ss':>11} {'Arp/lin ss':>11} {'Arp/Enz ls':>11} {'Ram/Enz ls':>11}")
    for z in zlist:
        e = getz(data["Enzo"], z, comp); kref = e[0]
        ll = getz(lin, z, comp)
        Pli = interp(kref, ll[0], ll[1]) if ll is not None else np.full_like(kref, np.nan)
        Pe = interp(kref, e[0], e[1])
        cells = []
        rats = {}
        for lab in ("Enzo", "RAMSES", "Arepo"):
            d = getz(data[lab], z, comp)
            rats[lab] = interp(kref, d[0], d[1]) if d is not None else np.full_like(kref, np.nan)
        for lab in ("Enzo", "RAMSES", "Arepo"): cells.append(band(rats[lab]/Pli))
        for lab in ("Enzo", "RAMSES", "Arepo"): cells.append(band(rats[lab]/Pli, False, SSBINS))
        cells.append(band(rats["Arepo"]/Pe)); cells.append(band(rats["RAMSES"]/Pe))
        print(f"{z:7.1f} " + " ".join(f"{c:11.3f}" for c in cells))
    print()

# panels: dm (top) + baryon (bottom), one column per early z
if zlist:
    fig, axes = plt.subplots(2, len(zlist), figsize=(4*len(zlist), 8), squeeze=False)
    for col, z in enumerate(zlist):
        for row, comp in enumerate(("dm", "baryon")):
            ax = axes[row][col]; yv = []
            for (lab, fn, sty, c) in CODES:
                d = getz(data[lab], z, comp)
                if d is None: continue
                ax.loglog(d[0], d[1], sty, ms=3, color=c, lw=1, label=lab)
                yv += list(d[1][d[1] > 0])
            ll = getz(lin, z, comp)
            if ll is not None:
                ax.loglog(ll[0], ll[1], "-", color="k", lw=1.5, label="CICASS-lin")
                yv += list(ll[1][ll[1] > 0])
            if yv: ax.set_ylim(min(yv)/1.6, max(yv)*1.6)
            ax.set_title(f"{comp} z={z:.0f}"); ax.set_xlabel("k [h/Mpc]")
            if col == 0: ax.set_ylabel("P(k) [(Mpc/h)^3]")
            ax.legend(fontsize=7)
    fig.suptitle(f"CICASS streaming ICs ({TAG}) early-z — Enzo(GPU) / RAMSES(GPU) / Arepo(CPU) / CICASS-linear")
    fig.tight_layout(); out = R + f"cicass_early_pk{TAG}.png"; fig.savefig(out, dpi=140)
    print("wrote", out)
