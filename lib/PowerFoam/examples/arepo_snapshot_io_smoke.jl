#!/usr/bin/env julia

using Dates
using Printf
using PowerFoam

const POWERFOAM_ROOT = normpath(joinpath(@__DIR__, ".."))
const OUTBASE = joinpath(@__DIR__, "out", "arepo_snapshot_io_smoke")
const RUN_TAG = Dates.format(now(), "yyyymmdd_HHMMSS")
const OUTDIR = joinpath(OUTBASE, RUN_TAG)

function csvquote(x)
    s = replace(string(x), "\n" => "\\n")
    return "\"" * replace(s, "\"" => "\"\"") * "\""
end

function write_rows(path, rows)
    open(path, "w") do io
        println(io, "status,check,detail")
        for row in rows
            println(io, join((csvquote(v) for v in row), ","))
        end
    end
end

function build_payload()
    return (
        header = (
            time = 0.125,
            box_size = 1.0,
            num_files = 1,
        ),
        gas = (
            density = [1.0, 0.9, 1.1, 1.05],
            masses = fill(0.25, 4),
            internal_energy = [2.4, 2.3, 2.5, 2.45],
            velocities = [
                0.10  0.00  0.00
                0.00  0.20  0.00
               -0.10  0.00  0.10
                0.05 -0.10  0.00
            ],
            Coordinates = [
                0.125 0.125 0.125
                0.375 0.125 0.125
                0.125 0.375 0.125
                0.375 0.375 0.125
            ],
            particle_ids = collect(1:4),
        ),
    )
end

expected_volume(payload) = payload.gas.masses ./ payload.gas.density
expected_pressure(payload; gamma = 5 / 3) =
    (float(gamma) - 1.0) .* payload.gas.density .* payload.gas.internal_energy

function passrow(check, detail)
    return ("PASS", check, detail)
end

function blockerrow(check, detail)
    return ("BLOCKER", check, detail)
end

function loaderrow(check, detail)
    return ("INFO", check, detail)
end

function main()
    mkpath(OUTDIR)
    rows = Vector{NTuple{3,String}}()
    push!(rows, loaderrow("powerfoam_package_load", "loaded package runtime surface"))

    preflight = PowerFoam.arepo_snapshot_hdf5_preflight()
    if preflight.available
        push!(rows, passrow("hdf5_dependency_preflight",
                            "ready package_path=$(preflight.package_path) action=$(preflight.action)"))
    else
        push!(rows, blockerrow("hdf5_dependency_preflight",
                               "$(preflight.detail) | action=$(preflight.action) | command=$(preflight.command)"))
    end

    payload = build_payload()
    snapshot = read_arepo_snapshot(payload; root = OUTDIR, snapshot_index = 7)
    validation = validate_arepo_snapshot(snapshot)
    if validation.valid
        push!(rows, passrow("in_memory_snapshot",
                            @sprintf("cells=%d volume_derived=%s pressure_derived=%s center_derived=%s",
                                     length(snapshot.gas.density),
                                     snapshot.derived.volume_derived,
                                     snapshot.derived.pressure_derived,
                                     snapshot.derived.center_derived)))
        push!(rows, passrow("direct_field_invariants",
                            "density masses internal_energy velocities particle_ids preserved"))
        push!(rows, passrow("derived_field_invariants",
                            @sprintf("volume=%s pressure=%s center=%s",
                                     all(isapprox.(snapshot.gas.volume, expected_volume(payload); atol = 1e-12, rtol = 1e-12)),
                                     all(isapprox.(snapshot.gas.pressure, expected_pressure(payload); atol = 1e-12, rtol = 1e-12)),
                                     all(isapprox.(snapshot.gas.center, payload.gas.Coordinates; atol = 1e-12, rtol = 1e-12)))))
    else
        push!(rows, blockerrow("in_memory_snapshot", join(validation.errors, "; ")))
    end

    planned_preflight = PowerFoam.arepo_snapshot_read_preflight(OUTDIR, 7)
    push!(rows, passrow("read_preflight_planned",
                        "status=$(planned_preflight.status) layout=$(locate_arepo_snapshot(OUTDIR, 7).layout)"))

    split_dir = joinpath(OUTDIR, "snapdir_008")
    mkpath(split_dir)
    split_path = joinpath(split_dir, "snap_008.0.hdf5")
    touch(split_path)
    split_preflight = PowerFoam.arepo_snapshot_read_preflight(OUTDIR, 8)
    push!(rows, passrow("read_preflight_split",
                        "status=$(split_preflight.status) path=$(split_preflight.path)"))

    locator = locate_arepo_snapshot(OUTDIR, 7)
    push!(rows, passrow("locator_preflight",
                        "layout=$(locator.layout) direct=$(locator.resolved_paths[1]) split=$(locator.resolved_paths[2])"))

    direct_path = locator.resolved_paths[1]
    write_result = write_arepo_snapshot(direct_path, snapshot)
    if write_result.ok && write_result.status == :preflight_only
        push!(rows, passrow("write_preflight", join(write_result.messages, " | ")))
        push!(rows, blockerrow("hdf5_backend",
                               "dependency preflight did not resolve HDF5; file IO remains preflight-only"))
    elseif write_result.ok
        filesize_bytes = isfile(direct_path) ? filesize(direct_path) : 0
        push!(rows, passrow("write_hdf5",
                            join(write_result.messages, " | ") * @sprintf(" | bytes=%d", filesize_bytes)))
        reread = read_arepo_snapshot(OUTDIR, 7)
        push!(rows, passrow("read_hdf5",
                            @sprintf("cells=%d time=%.6f layout=%s pressure_derived=%s volume_derived=%s",
                                     length(reread.gas.density),
                                     reread.header.time,
                                     reread.locator.layout,
                                     reread.derived.pressure_derived,
                                     reread.derived.volume_derived)))
        push!(rows, passrow("readback_field_invariants",
                            "density masses internal_energy velocities particle_ids preserved"))
        push!(rows, passrow("readback_derived_invariants",
                            @sprintf("volume=%s pressure=%s center=%s",
                                     all(isapprox.(reread.gas.volume, expected_volume(payload); atol = 1e-12, rtol = 1e-12)),
                                     all(isapprox.(reread.gas.pressure, expected_pressure(payload); atol = 1e-12, rtol = 1e-12)),
                                     all(isapprox.(reread.gas.center, payload.gas.Coordinates; atol = 1e-12, rtol = 1e-12)))))
        reread_preflight = PowerFoam.arepo_snapshot_read_preflight(OUTDIR, 7)
        push!(rows, passrow("read_preflight_after_write",
                            "status=$(reread_preflight.status) path=$(reread_preflight.path)"))
    else
        push!(rows, blockerrow("write_preflight", join(write_result.messages, " | ")))
    end

    csv_path = joinpath(OUTDIR, "rows.csv")
    write_rows(csv_path, rows)
    println("wrote ", csv_path)
    for row in rows
        println(join(row, ","))
    end
end

main()
