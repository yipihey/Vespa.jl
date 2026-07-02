#!/usr/bin/env python3
r"""Publication-style Δ²(k,μ) panel grid: CICASS 2-fluid LINEAR THEORY vs the SIMULATION.

Reproduces the "Linear Theory vs. Simulation" figure — a grid of μ columns × redshift rows, each panel
overlaying the linear Δ²_c (CDM, blue) and Δ²_b (baryon, red) on the measured simulation Δ² (black open
circles with cosmic-variance error bars).  The v_bc streaming shows as the μ-dependent baryon suppression
(stream-parallel modes μ→1 pressure-suppressed; perpendicular μ→0 not).

Data:
  * sim   — `<pkmu>.h5` from CIC_PK=1 (on-device P(k,μ)): per-z groups `z<zzz.z>/` with `k[nk]` and
            `(nmu,nk)` blocks `gas_delta` (baryon δ), `dm_delta` (CDM δ), `Nmodes` (mode counts).
  * linear— `cicass_linear_pk.dat` (transfer.x PRINT_PK): `@ z=<z> <tag>` blocks of `k[h/Mpc] P[(Mpc/h)^3]`,
            tags `dm`/`baryon` (angle-avg) + per-costh `dm_mu<costh>`/`baryon_mu<costh>` (costh=k̂·v̂_bc).
Both store P(k); the plot shows Δ²(k) = k³P/(2π²).

Run:
  PKMU=/path/to/full512_pkmu.h5 LIN=/path/to/cicass_linear_pk.dat \
    ZSEL=200,100,50 BOX=0.4 NPART=512 VBC=1 OUT=cicass_pkmu_panels.png \
    <anaconda python3> plot_cicass_pkmu_panels.py
"""
import os, numpy as np, h5py
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter

# ── LaTeX-serif look (falls back to cm mathtext if no system LaTeX) ─────────────────────────────
plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm", "font.size": 11,
    "axes.linewidth": 0.8, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True, "xtick.minor.visible": True, "ytick.minor.visible": True,
})

R      = (os.environ.get("VESPA_RUN_DIR") or os.getcwd()) + "/"
PKMU   = os.environ.get("PKMU", R + "cicass_pkmu.h5")
LIN    = os.environ.get("LIN",  R + "cicass_linear_pk.dat")
OUT    = os.environ.get("OUT",  R + "cicass_pkmu_panels.png")
BOX    = float(os.environ.get("BOX", "0.4"))          # Mpc/h (title only)
NPART  = os.environ.get("NPART", "512")               # per-dim particle/grid count (title only)
VBC    = os.environ.get("VBC", "1")                   # v_bc in units of σ_vbc (title only)
BLUE, RED, BLACK = "#1f4fd8", "#e11", "0.15"
SIMSTYLE = {
    "dm":  dict(color=BLUE, marker="o", label=r"Simulation $\Delta_c^2$"),
    "gas": dict(color=RED,  marker="s", label=r"Simulation $\Delta_b^2$"),
}
TWO_PI2 = 2.0 * np.pi**2

def d2(k, P):                                          # Δ²(k) = k³ P / (2π²)
    return k**3 * P / TWO_PI2

# ── linear theory: `@ z=<z> <tag>` blocks → B[(z,tag)] = (k, P) ─────────────────────────────────
def load_lin(fn):
    B, cur, ks, Ps = {}, None, [], []
    for line in open(fn):
        s = line.strip()
        if not s or s.startswith("#"): continue
        if s.startswith("@"):
            if cur and ks: B[cur] = (np.array(ks), np.array(Ps))
            p = s.split(); cur = (float(p[1].split("=")[1]), p[2]); ks, Ps = [], []
        else:
            a, b = s.split()[:2]; ks.append(float(a)); Ps.append(float(b))
    if cur and ks: B[cur] = (np.array(ks), np.array(Ps))
    return B

def lin_costh_nodes(B, comp):
    xs = sorted({float(t.split("mu")[1]) for (z, t) in B if t.startswith(comp + "_mu")})
    return xs

def lin_slice(B, z, comp, mucen):
    """Δ²(k) linear for `comp` (dm|baryon) at the costh node nearest μ-bin center `mucen`, z nearest."""
    nodes = lin_costh_nodes(B, comp)
    tag = f"{comp}_mu{min(nodes, key=lambda x: abs(x-mucen)):.3f}" if nodes else comp
    zz = [z0 for (z0, t) in B if t == tag]
    if not zz: return None
    k, P = B[(min(zz, key=lambda x: abs(x-z)), tag)]
    return k, d2(k, P)

# ── simulation P(k,μ) from the CIC_PK=1 HDF5 ────────────────────────────────────────────────────
def load_sim(fn):
    out = {}
    with h5py.File(fn, "r") as f:
        for g in f:
            z = float(g[1:])                          # "z050.0" → 50.0
            k = f[g]["k"][:]
            out[z] = dict(k=k, gas=f[g]["gas_delta"][:], dm=f[g]["dm_delta"][:],
                          nmodes=f[g]["Nmodes"][:])
    return out

def main():
    B   = load_lin(LIN)
    sim = load_sim(PKMU)
    zsel = [float(z) for z in os.environ.get("ZSEL", "").split(",") if z] or \
           sorted(sim, reverse=True)[:3]
    zsel = [min(sim, key=lambda x: abs(x-z)) for z in zsel]            # snap to available
    nmu  = next(iter(sim.values()))["gas"].shape[0]
    edges = np.linspace(0.0, 1.0, nmu + 1); cens = 0.5 * (edges[:-1] + edges[1:])
    nrow, ncol = len(zsel), nmu

    fig, axes = plt.subplots(nrow, ncol, figsize=(2.55*ncol, 2.35*nrow),
                             sharex=True, sharey="row", squeeze=False)
    fig.subplots_adjust(wspace=0.0, hspace=0.0, left=0.085, right=0.945, top=0.90, bottom=0.155)

    for i, z in enumerate(zsel):
        s = sim[z]; k = s["k"]; good = np.isfinite(k) & (k > 0)
        for j in range(ncol):
            ax = axes[i][j]
            # sim Δ² — thin connecting line through open circles + cosmic-variance bars (σ_P/P=√(2/N))
            for comp in ("dm", "gas"):
                P = s[comp][j]; nm = s["nmodes"][j].astype(float)
                m = good & np.isfinite(P) & (P > 0) & (nm > 0)
                D = d2(k[m], P[m]); err = D * np.sqrt(2.0 / nm[m])
                st = SIMSTYLE[comp]
                ax.plot(k[m], D, "-", color=st["color"], lw=0.6, alpha=0.65, zorder=4)
                ax.errorbar(k[m], D, yerr=err, fmt=st["marker"], ms=3.0, mfc="none",
                            mec=st["color"], mew=0.7, ecolor=st["color"], elinewidth=0.6,
                            capsize=1.4, capthick=0.6, ls="none", alpha=0.8, zorder=5)
            # linear theory: Δ²_c (blue), Δ²_b (red) at this μ column
            lc = lin_slice(B, z, "dm", cens[j])
            lb = lin_slice(B, z, "baryon", cens[j])
            if lc is not None: ax.plot(lc[0], lc[1], "-", color=BLUE, lw=2.4, zorder=3)
            if lb is not None: ax.plot(lb[0], lb[1], "-", color=RED,  lw=2.4, zorder=2)

            ax.set_xscale("log"); ax.set_yscale("log")
            ax.grid(True, which="major", color="0.85", lw=0.6, zorder=0)
            ax.grid(True, which="minor", color="0.93", lw=0.4, zorder=0)
            ax.xaxis.set_minor_formatter(NullFormatter()); ax.yaxis.set_minor_formatter(NullFormatter())
            ax.tick_params(which="both", labelsize=8)
            if i == 0:                                                 # boxed |μ| header
                ax.text(0.5, 1.04, rf"$|\mu|\in[{edges[j]:.2f},\ {edges[j+1]:.2f}]$",
                        transform=ax.transAxes, ha="center", va="bottom", fontsize=8.5,
                        bbox=dict(boxstyle="round,pad=0.2", fc="0.94", ec="0.6", lw=0.6))
            if j == ncol - 1:                                          # right-side z label
                ax.text(1.045, 0.5, rf"$z={z:.0f}$", transform=ax.transAxes, rotation=-90,
                        ha="left", va="center", fontsize=11)
        # per-row y-limit from the linear DM top and sim floor
        axes[i][0].set_xlim(k[good][0]*0.9, k[good][-1]*1.1)

    fig.text(0.5, 0.955,
             rf"$\bf Linear\ Theory$ vs. $\bf Simulation$ "
             rf"($N=2\times{NPART}^3,\ L={BOX:g}$ Mpc$/h,\ v_{{\rm bc}}={VBC}\sigma$)",
             ha="center", va="bottom", fontsize=13)
    fig.text(0.5, 0.075, r"$k\ \ [h/\mathrm{cMpc}]$", ha="center", fontsize=12)
    fig.text(0.022, 0.5, r"$\Delta^2(k,\mu)$", va="center", rotation="vertical", fontsize=13)

    from matplotlib.lines import Line2D
    leg = [Line2D([0],[0], color=BLUE, lw=2.4, label=r"Linear $\Delta_c^2$"),
           Line2D([0],[0], color=RED,  lw=2.4, label=r"Linear $\Delta_b^2$"),
           Line2D([0],[0], color=BLUE, marker="o", mfc="none", lw=0.6,
                  label=SIMSTYLE["dm"]["label"]),
           Line2D([0],[0], color=RED, marker="s", mfc="none", lw=0.6,
                  label=SIMSTYLE["gas"]["label"])]
    fig.legend(handles=leg, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.008), fontsize=11, columnspacing=2.5, handlelength=1.8)

    fig.savefig(OUT, dpi=150)
    print("wrote", OUT, f"({nrow}×{ncol} panels, z={zsel})")

if __name__ == "__main__":
    main()
