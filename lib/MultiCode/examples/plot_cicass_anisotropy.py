#!/usr/bin/env python3
"""Measure the v_bc streaming ANISOTROPY P(k, costh) in the simulations and compare
to CICASS linear theory slice-by-slice.

The streaming is along x, so costh = |k_x|/|k| (k̂·v̂_bc).  We FFT the simulation
overdensity (gas or DM) from the saved 3D fields (enzo_xspec / ramses_xspec dumps:
Int64 N, then ρ_b[N³], ρ_d[N³], Julia column-major = x fastest), bin |δ_k|² in
(|k|, costh) cells, and overlay CICASS's per-costh linear blocks
(baryon_mu<costh>, dm_mu<costh> from cicass_linear_pk.dat).

Anisotropy signature: gas power suppressed for costh→1 (modes along the stream),
unsuppressed for costh→0 (perpendicular).  DM should be ~isotropic.

Run:  ENZO_ZS=... RAMSES_ZS=... /usr/bin/python3 plot_cicass_anisotropy.py
"""
import numpy as np, os, glob, re
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

R = os.path.join(os.path.dirname(__file__), "..", "..", "..", "reports", "multicode") + "/"
N = int(os.environ.get("CIC_NGRID", "128")); L = float(os.environ.get("CIC_BOX", "0.128"))
# costh bin edges → centers matching CICASS nodes (0.125,0.375,0.625,0.875)
MU_EDGES = np.array([0.0, 0.25, 0.5, 0.75, 1.0]); MU_CEN = 0.5*(MU_EDGES[:-1]+MU_EDGES[1:])

def load_xspec(fn):
    raw = np.fromfile(fn, dtype=np.float64)
    n = int(np.frombuffer(raw[:1].tobytes(), dtype=np.int64)[0])
    rb = raw[1:1+n**3].reshape((n, n, n), order="F")     # x fastest = axis0
    rd = raw[1+n**3:1+2*n**3].reshape((n, n, n), order="F")
    return n, rb, rd

def pk_mu(field):
    """P(k,costh) on (|k|, costh) bins; returns (kcen, P[nk,nmu])."""
    d = field / field.mean() - 1.0
    dk = np.fft.rfftn(d); pk3 = (dk*np.conj(dk)).real / field.size**2 * L**3
    kf = 2*np.pi/L
    kx = np.fft.fftfreq(N, d=1.0/N)[:, None, None]
    ky = np.fft.fftfreq(N, d=1.0/N)[None, :, None]
    kz = np.fft.rfftfreq(N, d=1.0/N)[None, None, :]
    kmag = np.sqrt(kx**2+ky**2+kz**2)
    mu = np.where(kmag > 0, np.abs(kx)/np.maximum(kmag, 1e-12), 0.0) * np.ones_like(kmag)
    kmag_h = kmag * kf                                    # h/Mpc
    kbins = np.arange(1, N//2) * kf
    kcen = 0.5*(kbins[:-1]+kbins[1:])
    P = np.full((len(kcen), len(MU_CEN)), np.nan)
    ik = np.digitize(kmag_h.ravel(), kbins) - 1
    imu = np.digitize(mu.ravel(), MU_EDGES) - 1
    pkr = pk3.ravel()
    for a in range(len(kcen)):
        for b in range(len(MU_CEN)):
            sel = (ik == a) & (imu == b)
            if sel.sum() > 0: P[a, b] = pkr[sel].mean()
    return kcen, P

def load_lin(fn):
    B = {}; cur = None; ks = []; Ps = []
    for line in open(fn):
        s = line.strip()
        if not s or s.startswith("#"): continue
        if s.startswith("@"):
            if cur and ks: B[cur] = (np.array(ks), np.array(Ps))
            p = s.split(); cur = (float(p[1].split("=")[1]), p[2]); ks = []; Ps = []
        else:
            a, b = s.split()[:2]; ks.append(float(a)); Ps.append(float(b))
    if cur and ks: B[cur] = (np.array(ks), np.array(Ps))
    return B

def zs_of(prefix):
    out = []
    for f in glob.glob(R + prefix + "_z*.bin"):
        m = re.search(r"_z(\d+)\.bin$", f)
        if m and "run_z" not in f and not re.search(r"_z\d+[a-z]", os.path.basename(f)):
            out.append(int(m.group(1)))
    return sorted(set(out))

lin = load_lin(R + "cicass_linear_pk.dat")
lin_z = sorted({z for (z, t) in lin})

def lin_slice(z, comp, mu):
    tag = f"{comp}_mu{mu:.3f}"
    zz = [zz for (zz, t) in lin if t == tag]
    if not zz: return None
    znear = min(zz, key=lambda x: abs(x - z))
    return lin[(znear, tag)]

def measure(prefix, envkey):
    zs = [int(x) for x in os.environ.get(envkey, "").split(",") if x] or zs_of(prefix)
    rows = []
    for z in zs:
        fn = R + f"{prefix}_z{z}.bin"
        if not os.path.exists(fn): continue
        n, rb, rd = load_xspec(fn)
        kc, Pg = pk_mu(rb); _, Pd = pk_mu(rd)
        rows.append((z, kc, Pg, Pd))
    return rows

def aniso_table(name, rows):
    print(f"\n{name} gas P(k,costh) anisotropy ratio [P(costh~0.88)/P(costh~0.12)], small-scale band:")
    print(f"{'z':>6} | {'sim gas':>9} {'CICASS-lin':>11} | {'sim dm':>8} {'CICASS dm':>10}")
    for (z, kc, Pg, Pd) in rows:
        ss = slice(len(kc)*2//3, len(kc))
        def aniso(P):
            hi = np.nanmean(P[ss, -1]); lo = np.nanmean(P[ss, 0]); return hi/lo if lo > 0 else np.nan
        def lin_aniso(comp):
            hi = lin_slice(z, comp, MU_CEN[-1]); lo = lin_slice(z, comp, MU_CEN[0])
            if hi is None or lo is None: return np.nan
            kk = kc[ss]
            Ph = np.exp(np.interp(np.log(kk), np.log(hi[0]), np.log(hi[1])))
            Pl = np.exp(np.interp(np.log(kk), np.log(lo[0]), np.log(lo[1])))
            return np.mean(Ph)/np.mean(Pl)
        print(f"{z:6d} | {aniso(Pg):9.3f} {lin_aniso('baryon'):11.3f} | {aniso(Pd):8.3f} {lin_aniso('dm'):10.3f}")

print(f"CICASS-lin z: {lin_z}")
TAG = os.environ.get("CIC_TAG", "")
codes = [("ENZO", measure("enzo_xspec" + TAG, "ENZO_ZS")),
         ("RAMSES", measure("ramses_xspec" + TAG, "RAMSES_ZS")),
         ("AREPO", measure("arepo_xspec" + TAG, "AREPO_ZS"))]
codes = [(nm, rows) for nm, rows in codes if rows]   # skip codes with no dumps
for name, rows in codes:
    if rows: aniso_table(name, rows)

# 2-row figure: gas P(k,costh) slices, Enzo (top) vs RAMSES (bottom), at 3 redshifts.
nz = 3
fig, axes = plt.subplots(len(codes), nz, figsize=(5*nz, 4.0*len(codes)), squeeze=False)
for r, (name, rows) in enumerate(codes):
    if not rows:
        continue
    zsel = [rows[0], rows[len(rows)//2], rows[-1]]
    for col, (z, kc, Pg, Pd) in enumerate(zsel):
        ax = axes[r][col]
        klo, khi = kc[0], kc[-1]; yvals = []
        for b, mu in enumerate(MU_CEN):
            c = plt.cm.viridis(b/(len(MU_CEN)-1))
            ax.loglog(kc, Pg[:, b], "-", color=c, label=f"sim costh~{mu:.2f}")
            v = Pg[:, b]; yvals += list(v[np.isfinite(v) & (v > 0)])
            ls = lin_slice(z, "baryon", mu)
            if ls is not None:
                ax.loglog(ls[0], ls[1], ":", color=c, lw=1)
                m = (ls[0] >= klo) & (ls[0] <= khi) & (ls[1] > 0)
                yvals += list(ls[1][m])
        if yvals:
            ax.set_ylim(min(yvals)/1.6, max(yvals)*1.6)   # snug: ~0.2 dex pad each side
        ax.set_title(f"{name} gas P(k,costh) z={z}  (—sim  ⋯CICASS)")
        ax.set_xlabel("k [h/Mpc]"); ax.set_xlim(klo, khi)
        if col == 0: ax.set_ylabel("P(k,costh) [(Mpc/h)^3]")
        ax.legend(fontsize=6)
fig.tight_layout(); out = R + "cicass_anisotropy.png"; fig.savefig(out, dpi=140)
print("\nwrote", out)
