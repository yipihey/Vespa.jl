#!/usr/bin/env python3

import argparse
import csv
import html
import subprocess
import tempfile
from pathlib import Path

KNOWN_LABELS = [
    "standard-muscl",
    "standard-muscl-hll",
    "standard-localppm",
    "standard-localppm-hll",
    "powerfoam-muscl",
    "powerfoam-muscl-hll",
    "powerfoam-localppm",
    "powerfoam-localppm-hll",
]

COLORS = {
    "standard-muscl": "#4c78a8",
    "standard-muscl-hll": "#d62728",
    "standard-localppm": "#f58518",
    "standard-localppm-hll": "#8c1d40",
    "powerfoam-muscl": "#54a24b",
    "powerfoam-muscl-hll": "#008b8b",
    "powerfoam-localppm": "#b279a2",
    "powerfoam-localppm-hll": "#5b2c83",
    "analytic": "#111111",
}


def case_labels(root):
    labels = [
        path.name
        for path in root.iterdir()
        if path.is_dir() and (path / "param.txt").exists() and (path / "output").is_dir()
    ]
    rank = {label: i for i, label in enumerate(KNOWN_LABELS)}
    return sorted(labels, key=lambda label: (rank.get(label, len(rank)), label))


def compile_profiler(dest):
    src = Path(__file__).with_name("profile_noh_snapshot.c")
    exe = dest / "profile_noh_snapshot"
    if exe.exists() and exe.stat().st_mtime >= src.stat().st_mtime:
        return exe
    subprocess.run(["h5cc", str(src), "-O2", "-lm", "-o", str(exe)], check=True)
    return exe


def find_snapshot(case, snapnum):
    direct = case / "output" / f"snap_{snapnum:03d}.hdf5"
    if direct.exists():
        return direct
    split = case / "output" / f"snapdir_{snapnum:03d}" / f"snap_{snapnum:03d}.0.hdf5"
    if split.exists():
        return split
    raise FileNotFoundError(f"no snapshot {snapnum} under {case}")


def list_snapshots(case):
    direct = sorted((case / "output").glob("snap_*.hdf5"))
    out = []
    for path in direct:
        try:
            out.append((int(path.stem.split("_")[-1]), path))
        except ValueError:
            pass
    if out:
        return out
    for snapdir in sorted((case / "output").glob("snapdir_*")):
        try:
            num = int(snapdir.name.split("_")[-1])
        except ValueError:
            continue
        path = snapdir / f"snap_{num:03d}.0.hdf5"
        if path.exists():
            out.append((num, path))
    return out


def run_profiles(root, snapnum, nbins):
    profiler = compile_profiler(root)
    cells_csv = root / "noh_cell_values.csv"
    bins_csv = root / "noh_radial_bins.csv"
    metrics_csv = root / "noh_metrics.csv"
    cells_csv.write_text("case,r,density,density_ref,vrad,utherm\n", encoding="utf-8")
    bins_csv.write_text("case,r,density,density_ref,density_scatter,count\n", encoding="utf-8")
    metrics_csv.write_text(
        "case,time,shock_radius,l1_density,l2_density,postshock_mean,postshock_std,mass,energy,volume_sum\n",
        encoding="utf-8",
    )
    for label in case_labels(root):
        snap = find_snapshot(root / label, snapnum)
        subprocess.run(
            [str(profiler), str(snap), label, str(cells_csv), str(bins_csv), str(metrics_csv), str(nbins)],
            check=True,
        )
    return cells_csv, bins_csv, metrics_csv


def run_evolution(root, nbins):
    profiler = compile_profiler(root)
    labels = case_labels(root)
    snaps = list_snapshots(root / labels[0])
    out = root / "noh_evolution_metrics.csv"
    fields = [
        "snapshot",
        "case",
        "time",
        "shock_radius",
        "l1_density",
        "l2_density",
        "postshock_mean",
        "postshock_std",
        "mass",
        "energy",
        "volume_sum",
    ]
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            for num, _ in snaps:
                for label in labels:
                    cells_tmp = tmp / "cells.csv"
                    bins_tmp = tmp / "bins.csv"
                    metrics_tmp = tmp / "metrics.csv"
                    cells_tmp.write_text("case,r,density,density_ref,vrad,utherm\n", encoding="utf-8")
                    bins_tmp.write_text("case,r,density,density_ref,density_scatter,count\n", encoding="utf-8")
                    metrics_tmp.write_text(
                        "case,time,shock_radius,l1_density,l2_density,postshock_mean,postshock_std,mass,energy,volume_sum\n",
                        encoding="utf-8",
                    )
                    subprocess.run(
                        [
                            str(profiler),
                            str(find_snapshot(root / label, num)),
                            label,
                            str(cells_tmp),
                            str(bins_tmp),
                            str(metrics_tmp),
                            str(nbins),
                        ],
                        check=True,
                    )
                    with metrics_tmp.open("r", encoding="utf-8") as metrics_handle:
                        row = list(csv.DictReader(metrics_handle))[-1]
                    row = {"snapshot": num, **row}
                    writer.writerow(row)
    return out


def read_csv(path):
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_profile_svg(path, cells, bins):
    width, height = 980, 620
    left, right, top, bottom = 78, 32, 34, 58
    rmax = 0.92
    ymax = 22.0

    def sx(x):
        return left + min(x, rmax) / rmax * (width - left - right)

    def sy(y):
        return height - bottom - max(0.0, min(y, ymax)) / ymax * (height - top - bottom)

    labels = [label for label in KNOWN_LABELS if any(row["case"] == label for row in cells)]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="22" text-anchor="middle" font-family="sans-serif" font-size="15">AREPO Noh 2D radial density cell values</text>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#222"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#222"/>',
        f'<text x="{width/2}" y="{height-14}" text-anchor="middle" font-family="sans-serif" font-size="13">r</text>',
        f'<text x="18" y="{height/2}" transform="rotate(-90 18 {height/2})" text-anchor="middle" font-family="sans-serif" font-size="13">density</text>',
    ]
    for tick in range(10):
        x = 0.1 * tick
        px = sx(x)
        parts.append(f'<line x1="{px:.1f}" y1="{height-bottom}" x2="{px:.1f}" y2="{height-bottom+5}" stroke="#222"/>')
        parts.append(f'<text x="{px:.1f}" y="{height-bottom+21}" text-anchor="middle" font-family="sans-serif" font-size="11">{x:.1f}</text>')
    for tick in range(6):
        y = 4.0 * tick
        py = sy(y)
        parts.append(f'<line x1="{left-5}" y1="{py:.1f}" x2="{left}" y2="{py:.1f}" stroke="#222"/>')
        parts.append(f'<text x="{left-9}" y="{py+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11">{y:.0f}</text>')

    for label in labels:
        color = COLORS[label]
        brow = [row for row in bins if row["case"] == label and float(row["r"]) <= rmax]
        if brow:
            upper = []
            lower = []
            for row in brow:
                r = float(row["r"])
                mean = float(row["density"])
                scatter = float(row["density_scatter"])
                upper.append(f'{sx(r):.2f},{sy(mean + scatter):.2f}')
                lower.append(f'{sx(r):.2f},{sy(mean - scatter):.2f}')
            band = " ".join(upper + list(reversed(lower)))
            parts.append(f'<polygon points="{band}" fill="{color}" fill-opacity="0.12" stroke="none"/>')

        rows = [row for row in cells if row["case"] == label and float(row["r"]) <= rmax]
        for row in rows:
            parts.append(
                f'<circle cx="{sx(float(row["r"])):.2f}" cy="{sy(float(row["density"])):.2f}" '
                f'r="0.8" fill="{color}" fill-opacity="0.10"/>'
            )
        points = " ".join(f'{sx(float(row["r"])):.2f},{sy(float(row["density"])):.2f}' for row in brow)
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.8" points="{points}"/>')

    analytic = [row for row in bins if row["case"] == labels[0] and float(row["r"]) <= rmax]
    points = " ".join(f'{sx(float(row["r"])):.2f},{sy(float(row["density_ref"])):.2f}' for row in analytic)
    parts.append(f'<polyline fill="none" stroke="{COLORS["analytic"]}" stroke-width="2.2" stroke-dasharray="7 5" points="{points}"/>')

    lx, ly = width - 245, top + 22
    legend = labels + ["analytic"]
    for i, label in enumerate(legend):
        y = ly + 22 * i
        color = COLORS[label]
        dash = ' stroke-dasharray="7 5"' if label == "analytic" else ""
        parts.append(f'<line x1="{lx}" y1="{y}" x2="{lx+25}" y2="{y}" stroke="{color}" stroke-width="3"{dash}/>')
        parts.append(f'<text x="{lx+34}" y="{y+4}" font-family="sans-serif" font-size="12">{label}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_evolution_svg(path, rows):
    width, height = 960, 620
    left, right, top, bottom = 76, 28, 34, 56
    gap = 44
    panel_h = (height - top - bottom - gap) / 2
    labels = [label for label in KNOWN_LABELS if any(r["case"] == label for r in rows)]
    times = [float(r["time"]) for r in rows]
    tmin, tmax = min(times), max(times)

    def panel(metric, y0, title):
        vals = [float(r[metric]) for r in rows]
        ymin, ymax = min(vals), max(vals)
        if ymin == ymax:
            ymin -= 1
            ymax += 1
        pad = 0.08 * (ymax - ymin)
        ymin -= pad
        ymax += pad

        def sx(t):
            return left + (t - tmin) / (tmax - tmin) * (width - left - right)

        def sy(v):
            return y0 + panel_h - (v - ymin) / (ymax - ymin) * panel_h

        parts = [
            f'<text x="{left}" y="{y0-10}" font-family="sans-serif" font-size="14">{html.escape(title)}</text>',
            f'<line x1="{left}" y1="{y0+panel_h}" x2="{width-right}" y2="{y0+panel_h}" stroke="#222"/>',
            f'<line x1="{left}" y1="{y0}" x2="{left}" y2="{y0+panel_h}" stroke="#222"/>',
        ]
        for i in range(5):
            v = ymin + (ymax - ymin) * i / 4
            y = sy(v)
            parts.append(f'<line x1="{left-5}" y1="{y:.1f}" x2="{left}" y2="{y:.1f}" stroke="#222"/>')
            parts.append(f'<text x="{left-9}" y="{y+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11">{v:.3g}</text>')
        for i in range(5):
            t = tmin + (tmax - tmin) * i / 4
            x = sx(t)
            parts.append(f'<line x1="{x:.1f}" y1="{y0+panel_h}" x2="{x:.1f}" y2="{y0+panel_h+5}" stroke="#222"/>')
            parts.append(f'<text x="{x:.1f}" y="{y0+panel_h+20}" text-anchor="middle" font-family="sans-serif" font-size="11">{t:.2g}</text>')
        for label in labels:
            rr = [r for r in rows if r["case"] == label]
            pts = " ".join(f'{sx(float(r["time"])):.2f},{sy(float(r[metric])):.2f}' for r in rr)
            color = COLORS[label]
            parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{pts}"/>')
            for r in rr:
                parts.append(f'<circle cx="{sx(float(r["time"])):.2f}" cy="{sy(float(r[metric])):.2f}" r="2.3" fill="{color}"/>')
        return parts

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="22" text-anchor="middle" font-family="sans-serif" font-size="15">AREPO Noh 2D analytic error metrics</text>',
    ]
    parts.extend(panel("l1_density", top + 20, "Volume-weighted relative L1 density error, r < 0.8"))
    parts.extend(panel("postshock_std", top + 20 + panel_h + gap, "Postshock cell density standard deviation"))
    lx, ly = width - 235, top + 35
    for i, label in enumerate(labels):
        y = ly + 22 * i
        color = COLORS[label]
        parts.append(f'<line x1="{lx}" y1="{y}" x2="{lx+25}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{lx+34}" y="{y+4}" font-family="sans-serif" font-size="12">{label}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--snapshot", default=3, type=int)
    parser.add_argument("--nbins", default=180, type=int)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.all:
        metrics_csv = run_evolution(args.root, args.nbins)
        svg = args.root / "noh_evolution_metrics.svg"
        write_evolution_svg(svg, read_csv(metrics_csv))
        print(f"wrote {metrics_csv}")
        print(f"wrote {svg}")
    else:
        cells_csv, bins_csv, metrics_csv = run_profiles(args.root, args.snapshot, args.nbins)
        svg = args.root / "noh_radial_density_cells.svg"
        write_profile_svg(svg, read_csv(cells_csv), read_csv(bins_csv))
        print(f"wrote {cells_csv}")
        print(f"wrote {bins_csv}")
        print(f"wrote {metrics_csv}")
        print(f"wrote {svg}")


if __name__ == "__main__":
    main()
