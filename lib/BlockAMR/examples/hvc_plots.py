#!/usr/bin/env python3
"""Figures for hvc_cloud_2d.jl runs (after hvc_snap_export.jl).

    python3 hvc_plots.py OUTDIR RUN_DIR...

Writes into OUTDIR: panels_<tag>.png (density + temperature of the whole tunnel at
four times), timeseries.png (cold mass, cloud velocity and position vs t/t_cc for
every run), and movie_<tag>.mp4 (finest-level zoom window following the cold gas,
density | temperature) when ffmpeg is available.
"""
import sys, os, glob, subprocess, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

out = sys.argv[1]; runs = sys.argv[2:]
os.makedirs(out, exist_ok=True)


def tag(run):
    b = os.path.basename(run.rstrip("/"))
    return b.split("-hvc2d-")[1].rsplit("-", 1)[0] if "-hvc2d-" in b else b


def params(run):
    p = {}
    for line in open(os.path.join(run, "params.toml")):
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            p[k.strip()] = v.strip().strip('"')
    return p


def meta(run, k):
    m = {}
    for line in open(os.path.join(run, "npy", f"meta_{k}.txt")):
        a = line.split()
        m[a[0]] = a[1:] if len(a) > 2 else a[1]
    return m


def load(run, which, fld, k):
    return np.load(os.path.join(run, "npy", f"{which}_{fld}_{k}.npy"))



for run in runs:
    t = tag(run); p = params(run)
    ks = sorted(f[5:9] for f in os.listdir(os.path.join(run, "npy")) if f.startswith("meta_"))
    if not ks:
        continue
    pick = [ks[int(round(i * (len(ks) - 1) / 3))] for i in range(4)]
    fig, axs = plt.subplots(4, 2, figsize=(16, 9.5), constrained_layout=True)
    for row, k in enumerate(pick):
        m = meta(run, k)
        R = load(run, "full", "R", k); T = load(run, "full", "T", k)
        x0, y0, dx = float(m["full_x0"]), float(m["full_y0"]), float(m["full_dx"])
        ext = [x0 * 100, (x0 + R.shape[0] * dx) * 100, y0 * 100, (y0 + R.shape[1] * dx) * 100]   # pc
        im0 = axs[row, 0].imshow(R.T, origin="lower", extent=ext, norm=LogNorm(1e-4, 3), cmap="viridis")
        im1 = axs[row, 1].imshow(T.T, origin="lower", extent=ext, norm=LogNorm(10, 3e6), cmap="inferno")
        axs[row, 0].set_ylabel(f"t = {float(m['t_cc']):.2f} t_cc ({float(m['t_myr']):.1f} Myr)\ny [pc]")
    axs[0, 0].set_title(r"density [m$_H$ cm$^{-3}$]"); axs[0, 1].set_title("temperature [K]")
    for a in axs[-1]:
        a.set_xlabel("x [pc]")
    fig.colorbar(im0, ax=axs[:, 0], shrink=0.6); fig.colorbar(im1, ax=axs[:, 1], shrink=0.6)
    fig.suptitle(f"{t}: χ = {float(p['chi']):.0f}, T_c = {float(p['T_cloud']):.0f} K, β = {p['beta']}, "
                 f"table {'FG2011 shielded' if 'shielded' in p['table'] else 'FG2011 thin'}", fontsize=13)
    fig.savefig(os.path.join(out, f"panels_{t}.png"), dpi=110)
    plt.close(fig)

    # movie: zoom window (finest level), density | temperature
    fr = os.path.join(out, f"frames_{t}"); os.makedirs(fr, exist_ok=True)
    for i, k in enumerate(ks):
        m = meta(run, k)
        R = load(run, "zoom", "R", k); T = load(run, "zoom", "T", k)
        x0, y0, dx = float(m["zoom_x0"]), float(m["zoom_y0"]), float(m["zoom_dx"])
        ext = [x0 * 100, (x0 + R.shape[0] * dx) * 100, y0 * 100, (y0 + R.shape[1] * dx) * 100]
        fig, axs = plt.subplots(1, 2, figsize=(14, 3.9), constrained_layout=True)
        axs[0].imshow(R.T, origin="lower", extent=ext, norm=LogNorm(1e-4, 3), cmap="viridis")
        axs[1].imshow(T.T, origin="lower", extent=ext, norm=LogNorm(10, 3e6), cmap="inferno")
        axs[0].set_title(f"{t}  density   t = {float(m['t_cc']):.2f} t_cc ({float(m['t_myr']):.1f} Myr)")
        axs[1].set_title("temperature")
        for a in axs:
            a.set_xlabel("x [pc]")
        fig.savefig(os.path.join(fr, f"f_{i:04d}.png"), dpi=90)
        plt.close(fig)
    if subprocess.call(["which", "ffmpeg"], stdout=subprocess.DEVNULL) == 0:
        subprocess.call(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "6", "-i",
                         os.path.join(fr, "f_%04d.png"), "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                         "-pix_fmt", "yuv420p", os.path.join(out, f"movie_{t}.mp4")])

# time series, all runs together
fig, axs = plt.subplots(1, 3, figsize=(16, 4.4), constrained_layout=True)
for run in runs:
    rows = list(csv.DictReader(open(os.path.join(run, "timeseries.csv"))))
    if not rows:
        continue
    tc = np.array([float(r["t_cc"]) for r in rows]); mc = np.array([float(r["mcold"]) for r in rows])
    vx = np.array([float(r["vxc"]) for r in rows]); xc = np.array([float(r["xc"]) for r in rows])
    vw = float(params(run)["v_wind_kms"]) / 100.0
    axs[0].plot(tc, mc / mc[0], label=tag(run)); axs[1].plot(tc, vx / vw); axs[2].plot(tc, (xc - xc[0]) * 100)
axs[0].set_ylabel("cold mass (T < 2e4 K) / initial"); axs[1].set_ylabel("cold-gas velocity / v_wind")
axs[2].set_ylabel("cold-gas displacement [pc]")
for a in axs:
    a.set_xlabel("t / t_cc"); a.grid(alpha=0.3)
axs[0].legend()
fig.savefig(os.path.join(out, "timeseries.png"), dpi=110)
print("figures in", out)
