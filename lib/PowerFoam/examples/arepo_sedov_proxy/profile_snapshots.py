#!/usr/bin/env python3

import argparse
import csv
import subprocess
import tempfile
from pathlib import Path


KNOWN_LABELS = [
    "standard-muscl",
    "standard-localppm",
    "standard-shockfollow-muscl",
    "standard-shockfollow-localppm",
    "semi-muscl",
    "semi-localppm",
    "semi-shockfollow-muscl",
    "semi-shockfollow-localppm",
]

COLORS = {
    "standard-muscl": "#4c78a8",
    "standard-localppm": "#f58518",
    "standard-shockfollow-muscl": "#2f5597",
    "standard-shockfollow-localppm": "#b35c00",
    "semi-muscl": "#54a24b",
    "semi-localppm": "#b279a2",
    "semi-shockfollow-muscl": "#1f7a4d",
    "semi-shockfollow-localppm": "#7e4c8a",
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
    src = Path(__file__).with_name("profile_arepo_snapshot.c")
    exe = dest / "profile_arepo_snapshot"
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
    if direct:
        out = []
        for path in direct:
            stem = path.stem
            try:
                out.append((int(stem.split("_")[-1]), path))
            except ValueError:
                continue
        return out
    out = []
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
    labels = case_labels(root)
    profile_csv = root / "radial_profiles.csv"
    metrics_csv = root / "metrics.csv"
    profile_csv.write_text("case,r,density,density_scatter,count\n", encoding="utf-8")
    metrics_csv.write_text(
        "case,time,shock_radius,peak_density,shell_scatter,shell_density_std,mass,energy,pressure_max,volume_sum\n",
        encoding="utf-8",
    )
    for label in labels:
        snap = find_snapshot(root / label, snapnum)
        subprocess.run(
            [str(profiler), str(snap), label, str(profile_csv), str(metrics_csv), str(nbins)],
            check=True,
        )
    return profile_csv, metrics_csv


def run_evolution(root, nbins):
    profiler = compile_profiler(root)
    labels = case_labels(root)
    if not labels:
        raise FileNotFoundError(f"no case directories under {root}")
    snaps = list_snapshots(root / labels[0])
    if not snaps:
        raise FileNotFoundError(f"no snapshots under {root / labels[0] / 'output'}")
    for label in labels[1:]:
        other = [num for num, _ in list_snapshots(root / label)]
        nums = [num for num, _ in snaps]
        if other != nums:
            raise RuntimeError(f"snapshot numbers for {label} differ from baseline: {other} != {nums}")

    out = root / "evolution_metrics.csv"
    fields = [
        "snapshot",
        "case",
        "time",
        "shock_radius",
        "peak_density",
        "shell_scatter",
        "shell_density_std",
        "mass",
        "energy",
        "pressure_max",
        "volume_sum",
    ]
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for num, _ in snaps:
                for label in labels:
                    snap = find_snapshot(root / label, num)
                    profile_tmp = tmp / "profile.csv"
                    metrics_tmp = tmp / "metrics.csv"
                    profile_tmp.write_text("case,r,density,density_scatter,count\n", encoding="utf-8")
                    metrics_tmp.write_text(
                        "case,time,shock_radius,peak_density,shell_scatter,shell_density_std,mass,energy,pressure_max,volume_sum\n",
                        encoding="utf-8",
                    )
                    subprocess.run(
                        [str(profiler), str(snap), label, str(profile_tmp), str(metrics_tmp), str(nbins)],
                        check=True,
                    )
                    with metrics_tmp.open("r", encoding="utf-8") as metrics_handle:
                        rows = list(csv.DictReader(metrics_handle))
                    row = {"snapshot": num}
                    row.update(rows[-1])
                    writer.writerow(row)
    return out


def read_profiles(path):
    profiles = {}
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            profiles.setdefault(row["case"], {"r": [], "rho": []})
            profiles[row["case"]]["r"].append(float(row["r"]))
            profiles[row["case"]]["rho"].append(float(row["density"]))
    return profiles


def write_svg(path, profiles):
    width, height = 900, 520
    left, right, top, bottom = 70, 30, 35, 55
    ymax = max(max(p["rho"]) for p in profiles.values() if p["rho"]) * 1.08
    ymax = max(ymax, 1.1)

    def sx(x):
        return left + x / 0.5 * (width - left - right)

    def sy(y):
        return height - bottom - y / ymax * (height - top - bottom)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#222"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#222"/>',
        f'<text x="{width/2}" y="22" text-anchor="middle" font-family="sans-serif" font-size="15">AREPO Sedov proxy radial density</text>',
        f'<text x="{width/2}" y="{height-12}" text-anchor="middle" font-family="sans-serif" font-size="13">r</text>',
        f'<text x="18" y="{height/2}" transform="rotate(-90 18 {height/2})" text-anchor="middle" font-family="sans-serif" font-size="13">density</text>',
    ]
    for tick in range(6):
        x = 0.1 * tick
        px = sx(x)
        parts.append(f'<line x1="{px:.1f}" y1="{height-bottom}" x2="{px:.1f}" y2="{height-bottom+5}" stroke="#222"/>')
        parts.append(f'<text x="{px:.1f}" y="{height-bottom+20}" text-anchor="middle" font-family="sans-serif" font-size="11">{x:.1f}</text>')
    for tick in range(5):
        y = ymax * tick / 4
        py = sy(y)
        parts.append(f'<line x1="{left-5}" y1="{py:.1f}" x2="{left}" y2="{py:.1f}" stroke="#222"/>')
        parts.append(f'<text x="{left-9}" y="{py+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11">{y:.1f}</text>')

    labels = [label for label in KNOWN_LABELS if label in profiles]
    labels.extend(sorted(label for label in profiles if label not in labels))
    for label in labels:
        p = profiles[label]
        points = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in zip(p["r"], p["rho"]))
        color = COLORS.get(label, "#333333")
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{points}"/>')

    lx, ly = width - 230, top + 15
    for i, label in enumerate(labels):
        color = COLORS.get(label, "#333333")
        y = ly + 22 * i
        parts.append(f'<line x1="{lx}" y1="{y}" x2="{lx+25}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{lx+34}" y="{y+4}" font-family="sans-serif" font-size="12">{label}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def read_evolution(path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            parsed = {"snapshot": int(row["snapshot"]), "case": row["case"]}
            for key, value in row.items():
                if key not in ("snapshot", "case"):
                    parsed[key] = float(value)
            rows.append(parsed)
    return rows


def write_evolution_svg(path, rows):
    width, height = 960, 620
    left, right, top, bottom = 72, 30, 34, 56
    gap = 44
    panel_h = (height - top - bottom - gap) / 2
    labels = [label for label in KNOWN_LABELS if any(r["case"] == label for r in rows)]
    labels.extend(sorted({r["case"] for r in rows} - set(labels)))
    times = [r["time"] for r in rows]
    tmin, tmax = min(times), max(times)

    def panel(metric, y0, title):
        vals = [r[metric] for r in rows]
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
            f'<text x="{left}" y="{y0-10}" font-family="sans-serif" font-size="14">{title}</text>',
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
            parts.append(f'<text x="{x:.1f}" y="{y0+panel_h+20}" text-anchor="middle" font-family="sans-serif" font-size="11">{t:.3g}</text>')
        for label in labels:
            rr = [r for r in rows if r["case"] == label]
            pts = " ".join(f'{sx(r["time"]):.2f},{sy(r[metric]):.2f}' for r in rr)
            color = COLORS.get(label, "#333333")
            parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{pts}"/>')
            for r in rr:
                parts.append(f'<circle cx="{sx(r["time"]):.2f}" cy="{sy(r[metric]):.2f}" r="2.2" fill="{color}"/>')
        return parts

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="22" text-anchor="middle" font-family="sans-serif" font-size="15">AREPO Sedov proxy evolution metrics</text>',
    ]
    parts.extend(panel("shock_radius", top + 20, "Shock radius"))
    parts.extend(panel("shell_scatter", top + 20 + panel_h + gap, "Mass-weighted shell scatter"))
    lx, ly = width - 250, top + 32
    for i, label in enumerate(labels):
        y = ly + 22 * i
        color = COLORS.get(label, "#333333")
        parts.append(f'<line x1="{lx}" y1="{y}" x2="{lx+25}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{lx+34}" y="{y+4}" font-family="sans-serif" font-size="12">{label}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--snapshot", default=1, type=int)
    parser.add_argument("--nbins", default=160, type=int)
    parser.add_argument("--all", action="store_true", help="Analyze all snapshots into evolution_metrics.csv")
    args = parser.parse_args()

    if args.all:
        metrics_csv = run_evolution(args.root, args.nbins)
        svg = args.root / "evolution_metrics.svg"
        write_evolution_svg(svg, read_evolution(metrics_csv))
        print(f"wrote {metrics_csv}")
        print(f"wrote {svg}")
    else:
        profile_csv, metrics_csv = run_profiles(args.root, args.snapshot, args.nbins)
        svg = args.root / "radial_density.svg"
        write_svg(svg, read_profiles(profile_csv))
        print(f"wrote {profile_csv}")
        print(f"wrote {metrics_csv}")
        print(f"wrote {svg}")


if __name__ == "__main__":
    main()
