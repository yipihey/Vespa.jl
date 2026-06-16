#!/usr/bin/env python
"""Enzo vs RAMSES on ONE CICASS realization, auto-discovering the output redshifts
(robust to whatever z's a z=1000->20 run actually lands on).

Left:  density P(k) (gas + DM) for both codes at the lowest common z, + CICASS
       linear DM (the IC realization grown by D(a)^2, written by the Enzo run).
Right: mode-by-mode cross-correlation r(k) of the DM density and the gravitational
       potential between the two codes at a few matched z -- same realization => r->1
       on shared scales (the consistency test).
Also prints a text table of DM/gas P(k) ratios + r(k) so it works headless.
"""
import numpy as np, glob, re, os, sys
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

R = os.path.join(os.path.dirname(__file__), "..", "..", "..", "reports", "multicode") + "/"
N = int(os.environ.get("CIC_NGRID", "128")); L = float(os.environ.get("CIC_BOX", "0.128"))

def load_pk(fn):
    B = {}; cur = None; ks = []; Ps = []
    def flush():
        nonlocal cur, ks, Ps
        if cur and ks: B[cur] = (np.array(ks), np.array(Ps))
        ks, Ps = [], []
    if not os.path.exists(fn): return B
    for line in open(fn):
        line = line.strip()
        if not line or line.startswith("#"): continue
        if line.startswith("@"):
            flush(); p = line.split(); cur = (float(p[1].split("=")[1]), p[2])
        else:
            a, b = line.split(); ks.append(float(a)); Ps.append(float(b))
    flush(); return B

def load_bin(fn):
    d = np.fromfile(fn, dtype=np.float64)
    return d[:N**3].reshape(N, N, N), d[N**3:2*N**3].reshape(N, N, N)   # (phi, dm-density)

def zs_of(prefix, envkey):    # explicit env list (this run's outputs) else glob all dumps
    v = os.environ.get(envkey, "")
    if v.strip():
        return sorted(int(x) for x in v.split(","))
    out = []
    for f in glob.glob(R + prefix + "_z*.bin"):
        m = re.search(r"_z(\d+)\.bin$", f)
        if m: out.append(int(m.group(1)))
    return sorted(set(out))

kf = np.fft.fftfreq(N) * N
KX, KY, KZ = np.meshgrid(kf, kf, kf, indexing="ij"); km = np.sqrt(KX**2 + KY**2 + KZ**2)
def xcorr_k(a, b, nb=14):
    fa = np.fft.fftn(a - a.mean()); fb = np.fft.fftn(b - b.mean())
    kb = np.linspace(1, N // 2, nb + 1); out = []
    for i in range(nb):
        m = (km >= kb[i]) & (km < kb[i + 1])
        cr = np.real(fa[m] * np.conj(fb[m])).sum()
        na = (np.abs(fa[m])**2).sum(); nv = (np.abs(fb[m])**2).sum()
        out.append((0.5 * (kb[i] + kb[i + 1]) * 2 * np.pi / L, cr / np.sqrt(na * nv) if na * nv > 0 else 0))
    return np.array(out)

enz = load_pk(R + "cicass_highz_pk.dat"); ram = load_pk(R + "cicass_ramses_pk.dat")
ez = zs_of("enzo_fields", "ENZO_ZS"); rz = zs_of("ramses_fields", "RAMSES_ZS")
print("Enzo field-dump z:", ez); print("RAMSES field-dump z:", rz)
if not ez or not rz:
    print("missing field dumps; runs incomplete?"); sys.exit(1)

# match each Enzo z to the nearest RAMSES z
pairs = [(e, min(rz, key=lambda r: abs(r - e))) for e in ez]
pairs = [(e, r) for (e, r) in pairs if abs(e - r) / max(e, 1) < 0.05]   # within 5%
print("matched (enzo z, ramses z):", pairs)

fig, (axp, axc) = plt.subplots(1, 2, figsize=(13, 5.6))
# --- left: density P(k) at the LOWEST common z (most evolved) ---
def nearest_pk_block(B, z, tag):
    cand = [(zz, t) for (zz, t) in B if t == tag]
    if not cand: return None
    zz = min(cand, key=lambda c: abs(c[0] - z))[0]; return B[(zz, tag)], zz
elo, rlo = pairs[0]   # lowest z = most evolved (pairs sorted by ascending z)
for B, z, c, base in [(enz, elo, "C0", "Enzo"), (ram, rlo, "C1", "RAMSES")]:
    for tag, ls in (("dm", "-"), ("baryon", "--")):
        r = nearest_pk_block(B, z, tag)
        if r: (k, P), zz = r; axp.loglog(k, P, ls, color=c, lw=1.8, alpha=0.85, label=f"{base} {tag} z{zz:.0f}")
r = nearest_pk_block(enz, elo, "theory_dm")
if r: (k, P), zz = r; axp.loglog(k, P, ":", color="k", lw=1.4, alpha=0.7, label=f"linear DM z{zz:.0f}")
axp.set_xlabel(r"$k\ [h\,{\rm Mpc}^{-1}]$"); axp.set_ylabel(r"$P(k)\ [(h^{-1}{\rm Mpc})^3]$")
axp.set_title(f"Density P(k) at z$\\approx${elo:.0f}"); axp.legend(fontsize=8); axp.grid(which="both", alpha=0.15)

# --- right: cross-correlation r(k) at a few matched z ---
print("\n# cross-correlation r(k) Enzo x RAMSES (DM density, potential):")
colors = ["C3", "C2", "C0", "C4", "C5", "C1", "C6"]
sel = pairs if len(pairs) <= 4 else [pairs[0], pairs[len(pairs)//2], pairs[-1]]
for i, (e, r) in enumerate(sel):
    pe, de = load_bin(R + f"enzo_fields_z{e}.bin"); pr, dr = load_bin(R + f"ramses_fields_z{r}.bin")
    rd = xcorr_k(de, dr); rp = xcorr_k(pe, pr); c = colors[i % len(colors)]
    axc.semilogx(rd[:, 0], rd[:, 1], "-o", color=c, ms=3, label=f"DM z={e}")
    axc.semilogx(rp[:, 0], rp[:, 1], "--s", color=c, ms=3, mfc="none", label=f"phi z={e}")
    print(f"  z={e:4d}: DM  r(k) median={np.median(rd[:,1]):.3f}  large-scale(k<{rd[3,0]:.0f})={rd[:3,1].mean():.3f}"
          f"  | phi r(k) median={np.median(rp[:,1]):.3f}")
axc.axhline(1, color="0.7", lw=0.8); axc.set_ylim(-0.05, 1.05)
axc.set_xlabel(r"$k\ [h\,{\rm Mpc}^{-1}]$"); axc.set_ylabel("cross-correlation $r(k)$ Enzo $\\times$ RAMSES")
axc.set_title("Mode-by-mode consistency (one realization)"); axc.legend(fontsize=7, ncol=2, loc="lower left"); axc.grid(which="both", alpha=0.15)
fig.suptitle("Enzo (GPU gravity+chem+particles) vs RAMSES on one CICASS realization", fontsize=12)
fig.tight_layout(); out = R + "enzo_ramses_consistency.png"; fig.savefig(out, dpi=140); print("\nwrote", out)
