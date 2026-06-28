#!/usr/bin/env python3
"""5-way CICASS comparison at 64^3 (fixed resolution, no AMR):
   Enzo-CPU (native C++) / Enzo-GPU (PPM+Poisson kernels) / RAMSES (CUDA) /
   Arepo (64-rank MPI moving mesh)  vs  CICASS linear theory.

Nearest-z matching (Arepo overshoots targets by <2%, negligible for P(k)).
Reports large-scale (lowest LSBINS k) band-averaged P(k) ratio vs linear for DM
and baryon at each output z, the Enzo GPU-vs-CPU agreement (a correctness check),
and writes a summary P(k) panel PNG. Headless."""
import numpy as np, os, sys
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

R = os.path.join(os.path.dirname(__file__), "..", "..", "..", "reports", "multicode") + "/"
LSBINS = int(os.environ.get("LSBINS", "4"))
SSBINS = int(os.environ.get("SSBINS", "8"))
TAG = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("CIC_TAG", "_c64")
G = TAG + "g"  # Enzo-GPU tag

def load(fn):
    B, cur, ks, Ps = {}, None, [], []
    def flush():
        if cur and ks: B[cur] = (np.array(ks), np.array(Ps))
    if not os.path.exists(fn):
        print("MISSING", fn); return B
    for line in open(fn):
        s = line.strip()
        if not s or s.startswith("#"): continue
        if s.startswith("@"):
            flush(); p = s.split(); cur = (float(p[1].split("=")[1]), p[2]); ks, Ps = [], []
        else:
            a, b = s.split()[:2]; ks.append(float(a)); Ps.append(float(b))
    flush(); return B

# Canonical Enzo for the comparison is now GPU (PPMKernels f32, matched ICs) → reads {TAG}.dat.
# Enzo-CPU (native f64) is included ONLY if a separate {TAG}cpu.dat exists (else skipped); the
# old pre-IC-match GPU run {TAG}g is no longer used.
CODES = [("Enzo-GPU", f"cicass_highz_pk{TAG}.dat"),
         ("RAMSES",   f"cicass_ramses_pk{TAG}.dat"),
         ("Arepo",    f"cicass_arepo_pk{TAG}.dat")]
if os.path.exists(R + f"cicass_highz_pk{TAG}cpu.dat"):
    CODES.insert(1, ("Enzo-CPU", f"cicass_highz_pk{TAG}cpu.dat"))
D = {name: load(R+fn) for name, fn in CODES}
L = load(R+"cicass_linear_pk.dat")

def zs(B, tag): return sorted({z for (z,t) in B if t == tag})
def get(B, z, tag):
    zz = zs(B, tag)
    if not zz: return None
    zn = min(zz, key=lambda x: abs(x-z)); return zn, B[(zn, tag)]
def interp(kto, kfr, Pfr):
    m = (kfr > 0) & (Pfr > 0)
    return np.exp(np.interp(np.log(kto), np.log(kfr[m]), np.log(Pfr[m])))
def band(code_kP, lin_kP, nb, lo=True):
    k, P = code_kP; lk, lP = lin_kP
    Pl = interp(k, lk, lP)
    sel = (P > 0) & (Pl > 0)
    idx = np.where(sel)[0]
    if len(idx) == 0: return np.nan
    idx = idx[:nb] if lo else idx[-nb:]
    return np.exp(np.mean(np.log(P[idx]/Pl[idx])))

ZL = zs(L, "dm")
for tag in ("dm", "baryon"):
    print(f"\n================  {tag.upper()}: code P(k) / CICASS-linear  (large-scale, {LSBINS} lowest-k bins)  ================")
    print(f"{'z':>7} " + " ".join(f"{n:>10}" for n,_ in CODES))
    for z in ZL:
        lin = get(L, z, tag)
        row = f"{z:7.1f} "
        for name,_ in CODES:
            g = get(D[name], z, tag)
            if g is None or lin is None: row += f"{'--':>10} "; continue
            row += f"{band(g[1], lin[1], LSBINS):10.3f} "
        print(row)

# Enzo GPU-vs-CPU agreement (correctness of the GPU kernels vs native C++) — only if CPU present
if "Enzo-CPU" in D:
 print("\n================  Enzo-GPU / Enzo-CPU agreement (large-scale | small-scale)  ================")
 print(f"{'z':>7} {'DM ls':>8} {'DM ss':>8} {'bar ls':>8} {'bar ss':>8}")
 for z in ZL:
    out = f"{z:7.1f} "
    for tag in ("dm","baryon"):
        gc = get(D["Enzo-CPU"], z, tag); gg = get(D["Enzo-GPU"], z, tag)
        for lo in (True, False):
            if gc and gg:
                k,P = gg[1]; kc,Pc = gc[1]; Pci = interp(k, kc, Pc)
                sel = (P>0)&(Pci>0); idx = np.where(sel)[0]
                idx = idx[:LSBINS] if lo else idx[-SSBINS:]
                r = np.exp(np.mean(np.log(P[idx]/Pci[idx]))) if len(idx) else np.nan
                out += f"{r:8.3f} "
            else: out += f"{'--':>8} "
    print(out)

# summary plot: DM + baryon P(k) at z=20 for all codes + linear
fig, ax = plt.subplots(1, 2, figsize=(12,5))
for j, tag in enumerate(("dm","baryon")):
    kmin, kmax = np.inf, 0.0
    for name,_ in CODES:
        g = get(D[name], 20.0, tag)
        if g: kmin = min(kmin, g[1][0].min()); kmax = max(kmax, g[1][0].max())
    lin = get(L, 20.0, tag)
    if lin and np.isfinite(kmin):
        m = (lin[1][0] >= kmin) & (lin[1][0] <= kmax)
        ax[j].loglog(lin[1][0][m], lin[1][1][m], 'k-', lw=2, label="CICASS-linear")
    for name,_ in CODES:
        g = get(D[name], 20.0, tag)
        if g: ax[j].loglog(g[1][0], g[1][1], '.-', ms=4, label=f"{name} (z={g[0]:.1f})")
    if np.isfinite(kmin): ax[j].set_xlim(kmin/1.15, kmax*1.15)
    ax[j].set_title(f"{tag.upper()} P(k) at z=20"); ax[j].set_xlabel("k [h/Mpc]")
    ax[j].set_ylabel("P(k) [(Mpc/h)³]"); ax[j].legend(fontsize=8); ax[j].grid(alpha=0.3)
fig.tight_layout(); fig.savefig(R+f"cicass_5way_z20{TAG}.png", dpi=140)
print("\nwrote", R+f"cicass_5way_z20{TAG}.png")

# ALL codes at ALL redshifts — one panel per output z, vs the proper CICASS
# per-z 2-fluid linear prediction (transfer.x PRINT_PK, NOT growth-scaled ICs).
COL = {"Enzo-CPU":"C0","Enzo-GPU":"C1","RAMSES":"C2","Arepo":"C3"}
STY = {"Enzo-CPU":"o-","Enzo-GPU":"s-","RAMSES":"^-","Arepo":"v-"}
zord = sorted(ZL, reverse=True)                 # high-z first
nz = len(zord); ncol = 4; nrow = (nz + ncol - 1)//ncol
for tag in ("dm", "baryon"):
    fig, axs = plt.subplots(nrow, ncol, figsize=(4.2*ncol, 3.4*nrow), squeeze=False)
    for a in axs.flat: a.set_visible(False)
    for i, z in enumerate(zord):
        a = axs.flat[i]; a.set_visible(True)
        # k-range where the SIMULATIONS have data (union of code k-grids); the linear
        # curve and axis are clipped to this — don't show k where only cicass_linear exists.
        kmin, kmax = np.inf, 0.0
        for name,_ in CODES:
            g = get(D[name], z, tag)
            if g: kmin = min(kmin, g[1][0].min()); kmax = max(kmax, g[1][0].max())
        lin = get(L, z, tag)
        if lin and np.isfinite(kmin):
            m = (lin[1][0] >= kmin) & (lin[1][0] <= kmax)
            a.loglog(lin[1][0][m], lin[1][1][m], 'k-', lw=2.2, label="CICASS-linear", zorder=5)
        for name,_ in CODES:
            g = get(D[name], z, tag)
            if g: a.loglog(g[1][0], g[1][1], STY[name], color=COL[name], ms=3, lw=1,
                           label=f"{name}" + ("" if abs(g[0]-z)<0.05 else f" (z={g[0]:.0f})"))
        if np.isfinite(kmin): a.set_xlim(kmin/1.15, kmax*1.15)
        a.set_title(f"z = {z:.0f}", fontsize=11)
        a.grid(alpha=0.3, which="both")
        if i % ncol == 0: a.set_ylabel("P(k) [(Mpc/h)³]")
        if i // ncol == nrow-1 or i+ncol >= nz: a.set_xlabel("k [h/Mpc]")
    # one shared legend in the first visible empty slot or upper area
    h, l = axs.flat[0].get_legend_handles_labels()
    if nz < nrow*ncol:
        la = axs.flat[nz]; la.set_visible(True); la.axis("off"); la.legend(h, l, loc="center", fontsize=11)
    else:
        fig.legend(h, l, loc="upper center", ncol=5, fontsize=10)
    res = TAG.replace("_c","").replace("g","")
    fig.suptitle(f"{tag.upper()} P(k) — all codes at all output z vs proper CICASS 2-fluid linear "
                 f"(transfer.x), {res}³ fixed-res", fontsize=13)
    fig.tight_layout(rect=[0,0,1,0.97])
    fn = R + f"cicass_allz_{tag}{TAG}.png"; fig.savefig(fn, dpi=130); print("wrote", fn)
