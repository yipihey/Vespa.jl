#!/usr/bin/env python3
"""Cell-by-cell comparison of x_HII and T_gas between Enzo and RAMSES on one CICASS
realization (chem runs).  cellcmp dump = Int64 N, then ρ, x_HII, f_H2, f_HD, T (f64 N³).
Both on the same N³ grid (DM r(k)=1 confirms alignment).  Reports per-field median
ratio, cell correlation, and scatter at matched redshifts; writes scatter plots.
"""
import numpy as np, os, glob, re
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

R = (os.environ.get("VESPA_RUN_DIR") or os.getcwd()) + "/"  # run dir on scratch/archive (set VESPA_RUN_DIR)

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

TAG = os.environ.get("CIC_TAG", "")
enz = zof("enzo_cellcmp" + TAG); ram = zof("ramses_cellcmp" + TAG); arp = zof("arepo_cellcmp" + TAG)
print(f"Enzo cellcmp z: {sorted(enz)}\nRAMSES cellcmp z: {sorted(ram)}\nArepo cellcmp z: {sorted(arp)}\n")

def stats(a, b):
    m = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    if m.sum() < 10: return None
    r = b[m] / a[m]
    cc = np.corrcoef(np.log(a[m]), np.log(b[m]))[0, 1]
    return np.median(r), np.exp(np.std(np.log(r))), cc, m.sum()

# Enzo is the reference; compare RAMSES and Arepo against it cell-by-cell (nearest-z match).
others = [("RAM", ram), ("ARP", arp)]
others = [(nm, zd) for nm, zd in others if zd]
hdr = f"{'z(E)':>6} |"
for nm, _ in others:
    hdr += f" {'xHII '+nm+'/E':>11} {'sc':>5} {'cc':>5} | {'T '+nm+'/E':>9} {'sc':>5} {'cc':>5} |"
print(hdr)
pairs = []
for zE in sorted(enz, reverse=True):
    E = load_cc(enz[zE]); row = f"{zE:6d} |"; rec = {"zE": zE, "E": E}
    ok = False
    for nm, zd in others:
        zO = min(zd, key=lambda x: abs(x - zE)); O = load_cc(zd[zO])
        sx = stats(E["xHII"], O["xHII"]); sT = stats(E["T"], O["T"]); rec[nm] = O
        if sx and sT:
            row += f" {sx[0]:11.3f} {sx[1]:5.2f} {sx[2]:5.2f} | {sT[0]:9.3f} {sT[1]:5.2f} {sT[2]:5.2f} |"
            ok = True
        else:
            row += f" {'--':>11} {'--':>5} {'--':>5} | {'--':>9} {'--':>5} {'--':>5} |"
    if ok: print(row); pairs.append(rec)

# scatter plots at a few z: rows = (x_HII, T), one column per selected z; RAMSES & Arepo vs Enzo.
if pairs:
    sel = [pairs[0], pairs[len(pairs)//2], pairs[-1]]
    fig, ax = plt.subplots(2, len(sel), figsize=(4*len(sel), 8), squeeze=False)
    colmap = {"RAM": "tab:blue", "ARP": "tab:green"}
    for c, rec in enumerate(sel):
        E = rec["E"]; zE = rec["zE"]
        for row, key, lab in ((0, "xHII", "x_HII"), (1, "T", "T [K]")):
            a = E[key]
            allv = [a]
            for nm, _ in others:
                if nm not in rec: continue
                b = rec[nm][key]; m = np.isfinite(a)&np.isfinite(b)&(a>0)&(b>0)
                if m.sum() < 10: continue
                ss = np.random.default_rng(0).choice(np.where(m)[0], min(3000, m.sum()), replace=False)
                ax[row][c].loglog(a[ss], b[ss], ".", ms=1, alpha=0.3, color=colmap[nm], label=nm)
                allv += [a[ss], b[ss]]
            allf = np.concatenate([v[np.isfinite(v)&(v>0)] for v in allv])
            lo, hi = allf.min(), allf.max(); ax[row][c].plot([lo, hi], [lo, hi], "r-", lw=0.8)
            ax[row][c].set_title(f"{lab}  z={zE}"); ax[row][c].set_xlabel(f"Enzo {lab}")
            if c == 0: ax[row][c].set_ylabel(f"other {lab}"); ax[row][c].legend(fontsize=7, markerscale=6)
    fig.tight_layout(); out = R + "cicass_cellcmp.png"; fig.savefig(out, dpi=140)
    print("\nwrote", out)
