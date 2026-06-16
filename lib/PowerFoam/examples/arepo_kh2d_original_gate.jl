using Dates
using Printf

const GAMMA = 5 / 3
const AREPO_DIR = get(ENV, "AREPO_DIR", "/Users/tabel/Projects/arepo")
const EXAMPLE = joinpath(AREPO_DIR, "examples", "kh_2d_lecoanet")
const OUTBASE = joinpath(@__DIR__, "out", "arepo_kh2d_original_gate")

parse_arg(i, default, T) = length(ARGS) >= i ? parse(T, ARGS[i]) : default

const NX = parse_arg(1, 32, Int)
const TFINAL = parse_arg(2, 0.1, Float64)
const DRAT = parse_arg(3, 1.0, Float64)
const ANALYSIS_NX = parse_arg(4, max(64, 2 * NX), Int)
const RUN_TAG = replace(@sprintf("N%d_t%.4g_drat%.3g", NX, TFINAL, DRAT), "." => "p")
const OUTDIR = joinpath(OUTBASE, RUN_TAG)

const RUNS = [
    (; label = "arepo_hll",
       executable = get(ENV, "AREPO_KH_HLL_EXE",
                        joinpath(AREPO_DIR, "build_kh2d_hll", "Arepo"))),
    (; label = "arepo_hll_ppm",
       executable = get(ENV, "AREPO_KH_HLL_PPM_EXE",
                        joinpath(AREPO_DIR, "build_kh2d_hll_ppm", "Arepo"))),
]

function python_cmd()
    for exe in (get(ENV, "AREPO_PYTHON", ""), joinpath(AREPO_DIR, ".venv", "bin", "python"),
                "python3", "python")
        isempty(exe) && continue
        try
            run(pipeline(Cmd([exe, "-c", "import h5py, numpy, matplotlib"]);
                         stdout = devnull, stderr = devnull))
            return exe
        catch
        end
    end
    error("no Python with h5py/numpy/matplotlib found; set AREPO_PYTHON")
end

function set_param(text, key, value)
    line = @sprintf("%-38s %s", key, value)
    pattern = Regex("(?m)^" * key * "\\s+.*\$")
    return occursin(pattern, text) ? replace(text, pattern => line) : text * "\n" * line * "\n"
end

function stage_case(label)
    isdir(EXAMPLE) || error("AREPO KH example not found at $EXAMPLE")
    dir = joinpath(OUTDIR, "runs", label)
    isdir(dir) && rm(dir; recursive = true, force = true)
    mkpath(joinpath(dir, "output"))
    param = joinpath(dir, "param.txt")
    text = read(joinpath(EXAMPLE, "param.txt"), String)
    text = set_param(text, "TimeOfFirstSnapshot", "0.0")
    text = set_param(text, "TimeMax", @sprintf("%.12g", TFINAL))
    text = set_param(text, "TimeBetSnapshot", @sprintf("%.12g", max(TFINAL / 2, eps(Float64))))
    text = set_param(text, "MaxSizeTimestep", @sprintf("%.12g", min(0.02, TFINAL / 4)))
    text = set_param(text, "NumFilesPerSnapshot", "1")
    text = set_param(text, "NumFilesWrittenInParallel", "1")
    write(param, text)
    return dir
end

function run_original_case(runinfo, py)
    isfile(runinfo.executable) || error("missing AREPO executable for $(runinfo.label): $(runinfo.executable)")
    dir = stage_case(runinfo.label)
    run(pipeline(Cmd([py, joinpath(EXAMPLE, "create.py"), dir,
                      string(NX), @sprintf("%.12g", DRAT)]);
                 stdout = devnull))
    log_path = joinpath(dir, "arepo.log")
    open(log_path, "w") do log
        run(pipeline(Cmd(Cmd([runinfo.executable, "param.txt"]); dir);
                     stdout = log, stderr = log))
    end
    open(joinpath(dir, "analysis.log"), "w") do log
        run(pipeline(Cmd([py, joinpath(EXAMPLE, "check.py"), dir, string(ANALYSIS_NX)]);
                     stdout = log, stderr = log))
    end
    metrics = joinpath(dir, "analysis", "kh_metrics.csv")
    isfile(metrics) || error("KH analysis produced no metrics for $(runinfo.label)")
    field_csv = export_final_fields(dir, py)
    return (; runinfo.label, dir, log_path, metrics, field_csv)
end

function final_snapshot_path(dir)
    output = joinpath(dir, "output")
    snaps = sort(filter(p -> occursin(r"snap_[0-9]+\.hdf5$", basename(p)),
                        readdir(output; join = true)))
    isempty(snaps) && error("no AREPO snapshots found in $output")
    return snaps[end]
end

function export_final_fields(dir, py)
    snap = final_snapshot_path(dir)
    out = joinpath(dir, "analysis", "final_fields.csv")
    script = raw"""
import csv
import h5py
import numpy as np
import sys

gamma = float(sys.argv[3])
with h5py.File(sys.argv[1], "r") as f:
    g = f["PartType0"]
    n = len(g["Coordinates"])
    coords = np.asarray(g["Coordinates"])
    vel = np.asarray(g["Velocities"])
    rho = np.asarray(g["Density"])
    ids = np.asarray(g["ParticleIDs"]) if "ParticleIDs" in g else np.arange(1, n + 1)
    if "Masses" in g:
        mass = np.asarray(g["Masses"])
    elif "Volume" in g:
        mass = rho * np.asarray(g["Volume"])
    else:
        mass = np.ones(n) / n
    if "Volume" in g:
        volume = np.asarray(g["Volume"])
    else:
        volume = mass / rho
    if "Pressure" in g:
        pressure = np.asarray(g["Pressure"])
        u = pressure / ((gamma - 1.0) * rho)
    elif "InternalEnergy" in g:
        u = np.asarray(g["InternalEnergy"])
        pressure = (gamma - 1.0) * rho * u
    else:
        raise RuntimeError("snapshot has neither Pressure nor InternalEnergy")
    order = np.argsort(ids)
    time = f["Header"].attrs.get("Time", np.nan)

with open(sys.argv[2], "w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["label", "id", "t", "x", "y", "volume", "rho",
                     "vx", "vy", "pressure", "mass", "mx", "my",
                     "energy_density", "energy"])
    for i in order:
        vx = vel[i, 0]
        vy = vel[i, 1]
        kinetic = 0.5 * (vx * vx + vy * vy)
        energy = mass[i] * (u[i] + kinetic)
        writer.writerow(["arepo", int(ids[i]), time, coords[i, 0], coords[i, 1],
                         volume[i], rho[i], vx, vy, pressure[i], mass[i],
                         mass[i] * vx, mass[i] * vy, energy / volume[i],
                         energy])
"""
    run(Cmd([py, "-c", script, snap, out, @sprintf("%.17g", GAMMA)]))
    isfile(out) || error("failed to export final fields from $snap")
    return out
end

function parse_number(x)
    try
        return parse(Float64, x)
    catch
        return NaN
    end
end

function read_metrics(path)
    lines = readlines(path)
    length(lines) >= 2 || error("metrics file has no rows: $path")
    header = split(lines[1], ',')
    rows = NamedTuple[]
    for line in lines[2:end]
        isempty(strip(line)) && continue
        vals = split(line, ',')
        d = Dict{Symbol,Float64}()
        for (k, v) in zip(header, vals)
            d[Symbol(k)] = parse_number(v)
        end
        push!(rows, (; (k => d[k] for k in keys(d))...))
    end
    return rows
end

function final_row(metrics_path)
    rows = read_metrics(metrics_path)
    isempty(rows) && error("no metric rows in $metrics_path")
    return rows[end]
end

csvquote(v) = "\"" * replace(string(v), "\"" => "\"\"") * "\""

function write_combined_csv(path, results)
    open(path, "w") do io
        println(io, "label,t,mixed_area,vertical_ke,enstrophy,pressure_wiggle,symmetry_error,rho_min,rho_max,p_min,p_max")
        for r in results
            for row in read_metrics(r.metrics)
                vals = (r.label, row.t, row.mixed_area, row.vertical_ke, row.enstrophy,
                        row.pressure_wiggle, row.symmetry_error, row.rho_min,
                        row.rho_max, row.p_min, row.p_max)
                println(io, join((csvquote(v) for v in vals), ","))
            end
        end
    end
end

function metric_diff(a, b, key)
    return getproperty(b, key) - getproperty(a, key)
end

function write_report(path, results, combined_csv)
    finals = Dict(r.label => final_row(r.metrics) for r in results)
    status = all(r -> begin
            f = finals[r.label]
            isfinite(f.rho_min) && isfinite(f.p_min) && f.rho_min > 0 && f.p_min > 0
        end, results) ? "passed" : "failed"
    open(path, "w") do io
        println(io, "# AREPO 2-D KH Original-Code Gate")
        println(io)
        println(io, "This gate runs the stock AREPO `kh_2d_lecoanet` problem with")
        println(io, "existing original-code binaries, then compares HLL and HLL+PPM")
        println(io, "diagnostics. It establishes the original-code reference for the")
        println(io, "PowerFoam 2-D periodic KH parity gate that comes next.")
        println(io)
        @printf(io, "- status: %s\n", status)
        @printf(io, "- generated: %s\n", Dates.format(now(), dateformat"yyyy-mm-dd HH:MM:SS"))
        @printf(io, "- nx x ny: %d x %d\n", NX, 2 * NX)
        @printf(io, "- t_final: %.12g\n", TFINAL)
        @printf(io, "- density contrast parameter: %.12g\n", DRAT)
        @printf(io, "- analysis grid: %d x %d\n", ANALYSIS_NX, 2 * ANALYSIS_NX)
        @printf(io, "- combined metrics: `%s`\n", relpath(combined_csv, dirname(path)))
        println(io)
        println(io, "## Final Metrics")
        println(io)
        println(io, "| run | t | mixed area | vertical KE | enstrophy | pressure wiggle | symmetry error | rho min | rho max | p min | p max |")
        println(io, "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for r in results
            f = finals[r.label]
            @printf(io, "| %s | %.12g | %.12g | %.12g | %.12g | %.12g | %.12g | %.12g | %.12g | %.12g | %.12g |\n",
                    r.label, f.t, f.mixed_area, f.vertical_ke, f.enstrophy,
                    f.pressure_wiggle, f.symmetry_error, f.rho_min, f.rho_max,
                    f.p_min, f.p_max)
        end
        if haskey(finals, "arepo_hll") && haskey(finals, "arepo_hll_ppm")
            a = finals["arepo_hll"]
            b = finals["arepo_hll_ppm"]
            println(io)
            println(io, "## Solver Difference")
            println(io)
            println(io, "| metric | HLL+PPM - HLL |")
            println(io, "| --- | ---: |")
            for key in (:mixed_area, :vertical_ke, :enstrophy, :pressure_wiggle, :symmetry_error)
                @printf(io, "| %s | %.12g |\n", String(key), metric_diff(a, b, key))
            end
        end
        println(io)
        println(io, "## Run Artifacts")
        println(io)
        println(io, "| run | directory | metrics | final fields | log |")
        println(io, "| --- | --- | --- | --- | --- |")
        for r in results
            @printf(io, "| %s | `%s` | `%s` | `%s` | `%s` |\n",
                    r.label, relpath(r.dir, dirname(path)),
                    relpath(r.metrics, dirname(path)),
                    relpath(r.field_csv, dirname(path)),
                    relpath(r.log_path, dirname(path)))
        end
        println(io)
        println(io, "## Next Gate")
        println(io)
        println(io, "Run PowerFoam's 2-D periodic/reconstructed KH path from the same")
        println(io, "initial condition and compare scalar metrics plus the normalized")
        println(io, "`final_fields.csv` artifact against `arepo_hll`.")
    end
    return status
end

function main()
    mkpath(OUTDIR)
    py = python_cmd()
    results = [run_original_case(r, py) for r in RUNS]
    combined_csv = joinpath(OUTDIR, "kh_metrics_combined.csv")
    write_combined_csv(combined_csv, results)
    report = joinpath(OUTDIR, "README.md")
    status = write_report(report, results, combined_csv)
    @printf("wrote %s\n", report)
    @printf("wrote %s\n", combined_csv)
    for r in results
        f = final_row(r.metrics)
        @printf("%-16s t=%.6g mixed=%.6g KEy=%.6g rho=[%.6g, %.6g]\n",
                r.label, f.t, f.mixed_area, f.vertical_ke, f.rho_min, f.rho_max)
    end
    status == "passed" || exit(1)
end

main()
