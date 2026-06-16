#!/usr/bin/env julia

using Dates
using Printf
using PowerFoam

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
const OUTBASE = joinpath(@__DIR__, "out", "arepo_gravity_solver_registry_smoke")
const RUN_TAG = Dates.format(now(), "yyyymmdd_HHMMSS")
const OUTDIR = joinpath(OUTBASE, RUN_TAG)

csvquote(v) = "\"" * replace(string(v), "\"" => "\"\"") * "\""

function write_rows_csv(path, rows)
    open(path, "w") do io
        println(io, "name,family,status,backend,periodic,cosmological,notes")
        for row in rows
            vals = (row.name, row.family, row.status, row.backend,
                    row.periodic, row.cosmological, join(row.notes, " | "))
            println(io, join((csvquote(v) for v in vals), ","))
        end
    end
end

function write_report(path, rows, csv_path)
    open(path, "w") do io
        println(io, "# AREPO Gravity Solver Registry Smoke")
        println(io)
        println(io, "- timestamp: `", RUN_TAG, "`")
        println(io, "- rows: `", length(rows), "`")
        println(io, "- csv: `", relpath(csv_path, OUTDIR), "`")
        println(io)
        println(io, "| name | family | status | backend | periodic | cosmological |")
        println(io, "| --- | --- | --- | --- | --- | --- |")
        for row in rows
            println(io, "| ", row.name, " | ", row.family, " | ", row.status,
                    " | ", row.backend, " | ", row.periodic, " | ",
                    row.cosmological, " |")
        end
        println(io)
        println(io, "This smoke is a readiness table, not a physics-parity claim.")
        println(io, "Direct tiny-N and root PM have executable component gates; tree,")
        println(io, "cosmological PM, and coupled gas self-gravity remain planned rows.")
    end
end

function main()
    mkpath(OUTDIR)
    rows = arepo_gravity_solver_registry(; probe_pm = true)
    csv_path = joinpath(OUTDIR, "gravity_solver_registry.csv")
    report_path = joinpath(OUTDIR, "README.md")
    write_rows_csv(csv_path, rows)
    write_report(report_path, rows, csv_path)
    println("AREPO gravity solver registry smoke")
    for row in rows
        @printf("%-24s %-10s %-14s %-16s periodic=%s cosmological=%s\n",
                row.name, row.family, row.status, row.backend,
                row.periodic, row.cosmological)
    end
    println("wrote: ", relpath(csv_path, REPO_ROOT))
    println("wrote: ", relpath(report_path, REPO_ROOT))
end

main()
