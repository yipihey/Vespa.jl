#!/usr/bin/env python3
"""Four-way comparison: Enzo vs RAMSES vs Arepo vs CICASS-linear on ONE CICASS
streaming-velocity realization, at the SAME exact output redshifts.

Because all codes now land EXACTLY on the common output list (Enzo max_dt cap,
RAMSES set_time_cap, Arepo OUTPUTLIST_EXACT, CICASS-linear by construction), z's
match to ~1e-8 and we compare without any nearest-z fudge.

Reads @-block P(k) tables (TAG defaults to _c64; override with $CIC_TAG or argv[1]):
  cicass_highz_pk{TAG}.dat      Enzo   (blocks 'dm','baryon')
  cicass_ramses_pk{TAG}.dat     RAMSES-CPU
  cicass_ramses_pk_c64m.dat     RAMSES-Metal (optional GPU-parity check)
  cicass_arepo_pk{TAG}.dat      Arepo
  cicass_linear_pk.dat          CICASS analytic 2-fluid LINEAR P(k) (untagged)

Emits a headless ratio table (DM + baryon, each code vs linear + cross-code), and
three PNGs: P(k) panels, large-scale growth vs z, baryon/DM suppression vs z.
"""
import numpy as np, os, sys
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

R = (os.environ.get("VESPA_RUN_DIR") or os.getcwd()) + "/"  # run dir on scratch/archive (set VESPA_RUN_DIR)
TAG = (sys.argv[1] if len(sys.argv) > 1 else os.environ.get("CIC_TAG", "_c64"))
LSBINS = int(os.environ.get("LSBINS", "4"))
SSBINS = int(os.environ.get("SSBINS", "8"))

CODES = [  # (label, short, filename, plot-style, color)
    ("Enzo",      "Enz",  f"cicass_highz_pk{TAG}.dat",  "o-",  "C0"),
    ("RAMSES",    "Ram",  f"cicass_ramses_pk{TAG}.dat", "s--", "C1"),
    ("Arepo",     "Arp",  f"cicass_arepo_pk{TAG}.dat",  "^:",  "C2"),
    ("RAMSES-Mtl","RamM", "cicass_ramses_pk_c64m.dat",  "x-.", "C3"),
]
SHORT = {lab: sh for (lab, sh, *_ ) in CODES}

def load_pk(fn):
    B, cur, ks, Ps = {}, None, [], []
    def flush():
        nonlocal cur, ks, Ps
        if cur and ks: B[cur] = (np.array(ks), np.array(Ps))
        ks, Ps = [], []
    if not os.path.exists(fn):
        return None
    for line in open(fn):
        s = line.strip()
        if not s or s.startswith("#"): continue
        if s.startswith("@"):
            flush(); p = s.split(); cur = (float(p[1].split("=")[1]), p[2])
        else:
            a, b = s.split()[:2]; ks.append(float(a)); Ps.append(float(b))
    flush(); return B

def zs_with(B, tag):
    return sorted({z for (z, t) in B if t == tag}, reverse=True)

def getz(B, z, tag, rtol=1e-3):
    if B is None: return None
    zz = zs_with(B, tag)
    if not zz: return None
    znear = min(zz, key=lambda x: abs(x - z))
    return B[(znear, tag)] if abs(znear - z) <= rtol * (1 + z) else None

def interp(k_to, k_from, P_from):
    lo = (k_from > 0) & (P_from > 0)
    if lo.sum() < 2: return np.full_like(k_to, np.nan)
    return np.exp(np.interp(np.log(k_to), np.log(k_from[lo]), np.log(P_from[lo])))

def band(ratio, lo=True, n=LSBINS):
    sub = ratio[:n] if lo else ratio[-n:]
    sub = sub[np.isfinite(sub) & (sub > 0)]
    return float(np.exp(np.mean(np.log(sub)))) if len(sub) else float("nan")

data = {lab: load_pk(R + fn) for (lab, sh, fn, _, _) in CODES}
lin  = load_pk(R + "cicass_linear_pk.dat")
present = [lab for lab in data if data[lab] is not None]
print(f"TAG={TAG}  present: {present}  linear={'yes' if lin else 'NO'}\n")
for (lab, sh, fn, _, _) in CODES:
    st = "ok" if data[lab] is not None else "MISSING"
    print(f"  {lab:11s} {fn:32s} {st}")
print()

# canonical z list = linear's (the exact requested list)
zlist = zs_with(lin, "dm") if lin else (zs_with(data[present[0]], "dm") if present else [])

for comp in ("dm", "baryon"):
    print(f"================  {comp.upper()}  band-ratio table (geometric mean over k-bins)  ================")
    hdr = f"{'z':>7}"
    for lab in present: hdr += f" {SHORT[lab]+'/lin ls':>13} {SHORT[lab]+'/lin ss':>13}"
    print(hdr)
    for z in zlist:
        kref = None; line = f"{z:7.1f}"
        # reference k-grid = Enzo if present else first present
        refl = "Enzo" if "Enzo" in present else (present[0] if present else None)
        ref = getz(data.get(refl), z, comp) if refl else None
        if ref is not None: kref = ref[0]
        ll = getz(lin, z, comp)
        for lab in present:
            d = getz(data[lab], z, comp)
            if d is None or ll is None or kref is None:
                line += f" {'--':>13} {'--':>13}"; continue
            kk, PP = d; kl, Pl = ll
            Pi = interp(kref, kk, PP); Pli = interp(kref, kl, Pl)
            r = Pi / Pli
            line += f" {band(r):13.3f} {band(r,False,SSBINS):13.3f}"
        print(line)
    print()

# cross-code vs Enzo (large scale) + RAMSES Metal/CPU parity
if "Enzo" in present:
    print("================  cross-code large-scale ratio vs Enzo (DM | baryon)  ================")
    print(f"{'z':>7} " + " ".join(f"{SHORT[lab]+'/Enz':>13}" for lab in present if lab != "Enzo"))
    for z in zlist:
        e_dm = getz(data["Enzo"], z, "dm"); e_b = getz(data["Enzo"], z, "baryon")
        if e_dm is None: continue
        kref = e_dm[0]; line = f"{z:7.1f}"
        for lab in present:
            if lab == "Enzo": continue
            parts = []
            for comp, eref in (("dm", e_dm), ("baryon", e_b)):
                d = getz(data[lab], z, comp)
                if d is None or eref is None: parts.append("--"); continue
                Pi = interp(kref, d[0], d[1]); Pe = interp(kref, eref[0], eref[1])
                parts.append(f"{band(Pi/Pe):.3f}")
            line += f" {parts[0]+'|'+parts[1]:>13}"
        print(line)
    print()

# ---------- PNG 1: P(k) panels at low/mid/high z ----------
if present and zlist:
    zsel = [zlist[0], zlist[len(zlist)//2], zlist[-1]]
    fig, axes = plt.subplots(2, len(zsel), figsize=(4*len(zsel), 8), squeeze=False)
    for col, z in enumerate(zsel):
        for row, comp in enumerate(("dm", "baryon")):
            ax = axes[row][col]; yv = []
            for (lab, sh, fn, sty, c) in CODES:
                d = getz(data[lab], z, comp)
                if d is None: continue
                ax.loglog(d[0], d[1], sty, ms=3, color=c, label=lab, lw=1)
                yv += list(d[1][d[1] > 0])
            ll = getz(lin, z, comp)
            if ll is not None:
                ax.loglog(ll[0], ll[1], "-", color="k", lw=1.5, label="CICASS-lin")
                yv += list(ll[1][ll[1] > 0])
            if yv: ax.set_ylim(min(yv)/1.6, max(yv)*1.6)
            ax.set_title(f"{comp} z={z:.0f}"); ax.set_xlabel("k [h/Mpc]")
            if col == 0: ax.set_ylabel("P(k) [(Mpc/h)^3]")
            ax.legend(fontsize=7)
    fig.suptitle(f"CICASS streaming ICs ({TAG}) — Enzo / RAMSES / Arepo / CICASS-linear")
    fig.tight_layout(); fig.savefig(R + f"cicass_4way_pk{TAG}.png", dpi=140)
    print("wrote", R + f"cicass_4way_pk{TAG}.png")

# ---------- PNG 2: large-scale growth vs z ; PNG 3: baryon/DM vs z ----------
def ls_amp(B, z, comp):
    d = getz(B, z, comp)
    if d is None: return np.nan
    k, P = d; m = (k > 0) & (P > 0)
    return float(np.sqrt(np.exp(np.mean(np.log(P[m][:LSBINS]))))) if m.sum() else np.nan

fig2, ax2 = plt.subplots(1, 2, figsize=(11, 4.5))
for comp, ax in zip(("dm", "baryon"), ax2):
    zz = np.array(zlist)
    for (lab, sh, fn, sty, c) in CODES:
        if data[lab] is None: continue
        amp = np.array([ls_amp(data[lab], z, comp) for z in zz])
        ax.plot(1+zz, amp, sty, color=c, label=lab, ms=4)
    if lin:
        ax.plot(1+zz, [ls_amp(lin, z, comp) for z in zz], "k-", lw=1.5, label="CICASS-lin")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.invert_xaxis()
    ax.set_xlabel("1+z"); ax.set_ylabel(f"large-scale sqrt(P) [{comp}]")
    ax.set_title(f"{comp} growth (LSBINS={LSBINS})"); ax.legend(fontsize=8)
fig2.tight_layout(); fig2.savefig(R + f"cicass_4way_growth{TAG}.png", dpi=140)
print("wrote", R + f"cicass_4way_growth{TAG}.png")

fig3, ax3 = plt.subplots(figsize=(6, 4.5))
zz = np.array(zlist)
for (lab, sh, fn, sty, c) in CODES:
    if data[lab] is None: continue
    rb = np.array([ls_amp(data[lab], z, "baryon")/ls_amp(data[lab], z, "dm") for z in zz])
    ax3.plot(1+zz, rb, sty, color=c, label=lab, ms=4)
if lin:
    ax3.plot(1+zz, [ls_amp(lin, z, "baryon")/ls_amp(lin, z, "dm") for z in zz], "k-", lw=1.5, label="CICASS-lin")
ax3.set_xscale("log"); ax3.invert_xaxis()
ax3.set_xlabel("1+z"); ax3.set_ylabel("baryon/DM large-scale amplitude")
ax3.set_title("Baryon catch-up vs DM (streaming suppression)"); ax3.legend(fontsize=8)
fig3.tight_layout(); fig3.savefig(R + f"cicass_4way_baryon_dm{TAG}.png", dpi=140)
print("wrote", R + f"cicass_4way_baryon_dm{TAG}.png")
