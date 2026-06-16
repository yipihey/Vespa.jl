using Dates
using Printf

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
const OUTBASE = joinpath(@__DIR__, "out", "arepo_hydro_problem_registry_smoke")
const RUN_TAG = Dates.format(now(), "yyyymmdd_HHMMSS")
const OUTDIR = joinpath(OUTBASE, RUN_TAG)

struct HydroProblemRow
    key::String
    dim::Int
    class::String
    arepo_reference::String
    current_surface::String
    runnable_path::Union{Nothing,String}
    status::String
    next_gate::String
end

function rel(path::AbstractString)
    return relpath(path, REPO_ROOT)
end

function maybe_rel(path::Union{Nothing,String})
    path === nothing && return ""
    return rel(path)
end

function example_path(parts...)
    path = joinpath(@__DIR__, parts...)
    return isfile(path) ? path : nothing
end

function proxy_path(parts...)
    path = joinpath(@__DIR__, parts...)
    return ispath(path) ? path : nothing
end

const PROBLEMS = HydroProblemRow[
    HydroProblemRow(
        "kh2d", 2, "Kelvin-Helmholtz instability",
        "examples/kh_2d_lecoanet/check.py",
        "AREPO original gate plus PowerFoam compare gate",
        example_path("powerfoam_kh2d_compare_gate.jl"),
        "runnable",
        "Promote the compare gate from smoke-scale to a Q0 final-field and integral-metric gate."
    ),
    HydroProblemRow(
        "noh2d", 2, "strong cylindrical shock",
        "examples/noh_2d/check.py",
        "proxy-only AREPO tables and archived metrics",
        proxy_path("arepo_noh_proxy", "generate_tables.jl"),
        "proxy-only",
        "Convert the proxy material into an executable 2-D final-field and radial-profile parity gate."
    ),
    HydroProblemRow(
        "noh3d", 3, "strong spherical shock",
        "examples/noh_3d/check.py",
        "stock AREPO staging plus PowerFoam diagnostic compare",
        example_path("arepo_noh3d_smoke_gate.jl"),
        "runnable",
        "Freeze repeated-run tolerances and promote the existing smoke gate to Q1."
    ),
    HydroProblemRow(
        "gresho", 2, "vortex balance",
        "examples/gresho_2d/check.py",
        "periodic vortex calibration gate with radial profile and rotational proxy diagnostics",
        example_path("arepo_gresho2d_gate.jl"),
        "runnable",
        "Freeze repeated-run tolerances or add an original-AREPO profile comparison before promotion."
    ),
    HydroProblemRow(
        "wave", 1, "linear wave",
        "examples/acoustic_wave_1d/check.py and examples/wave_1d/check.py",
        "reference-only in planning; no local executable gate yet",
        nothing,
        "planned",
        "Build a thin-3D or dedicated 1-D shim with amplitude and phase diagnostics."
    ),
    HydroProblemRow(
        "turbulence", 3, "decaying subsonic turbulence",
        "examples/bauer_springel_turbulence_3d/check.py",
        "bridge/component gate stack plus registry matrix",
        example_path("arepo_standard_problem_matrix.jl"),
        "runnable",
        "Close the native production pass sequence, then promote the standard-problem row to R0."
    ),
]

function csvquote(x)
    s = string(x)
    return "\"" * replace(s, "\"" => "\"\"") * "\""
end

function validate_registry(rows)
    seen = Set{String}()
    for row in rows
        row.key in seen && error("duplicate hydro problem key: $(row.key)")
        push!(seen, row.key)
        isempty(strip(row.arepo_reference)) && error("missing AREPO reference for $(row.key)")
        if row.status == "runnable" && row.runnable_path === nothing
            error("runnable row $(row.key) is missing a local path")
        end
    end
    return nothing
end

function write_csv(path, rows)
    open(path, "w") do io
        println(io, "problem,dim,class,arepo_reference,current_surface,runnable_path,status,next_gate")
        for row in rows
            vals = (
                row.key,
                string(row.dim),
                row.class,
                row.arepo_reference,
                row.current_surface,
                maybe_rel(row.runnable_path),
                row.status,
                row.next_gate,
            )
            println(io, join((csvquote(v) for v in vals), ","))
        end
    end
end

function write_report(path, rows)
    open(path, "w") do io
        println(io, "# AREPO Hydro Problem Registry Smoke")
        println(io)
        println(io, "This is a lightweight registry artifact for the hydro standard")
        println(io, "problems that matter for the current `Arepo.jl` rewrite parity")
        println(io, "plan. It does not run a simulation; it only verifies that the")
        println(io, "problem list is explicit and that existing local drivers are")
        println(io, "discoverable by path.")
        println(io)
        @printf(io, "- generated: %s\n", Dates.format(now(), dateformat"yyyy-mm-dd HH:MM:SS"))
        @printf(io, "- registry rows: %d\n", length(rows))
        @printf(io, "- runnable rows: %d\n", count(r -> r.status == "runnable", rows))
        @printf(io, "- proxy-only rows: %d\n", count(r -> r.status == "proxy-only", rows))
        @printf(io, "- planned rows: %d\n", count(r -> r.status == "planned", rows))
        println(io)
        println(io, "## Coverage")
        println(io)
        println(io, "| problem | dim | status | local path | next gate |")
        println(io, "| --- | ---: | --- | --- | --- |")
        for row in rows
            local_path = row.runnable_path === nothing ? "-" : "`$(maybe_rel(row.runnable_path))`"
            @printf(io, "| %s | %d | %s | %s | %s |\n",
                    row.key, row.dim, row.status, local_path, row.next_gate)
        end
        println(io)
        println(io, "## Promotion Rule")
        println(io)
        println(io, "A standard problem should not advance beyond registry status until")
        println(io, "it has one documented command, one stable artifact directory under")
        println(io, "`lib/PowerFoam/examples/out/`, and one explicit pass/fail metric")
        println(io, "derived from the upstream AREPO example check.")
        println(io)
        println(io, "## Next Conversion")
        println(io)
        println(io, "Convert `noh2d` next. It already has local proxy material under")
        println(io, "`lib/PowerFoam/examples/arepo_noh_proxy/`, covers the strongest")
        println(io, "remaining 2-D shock hole in the parity ladder, and is a cleaner")
        println(io, "first executable gate than starting fresh with wave.")
    end
end

function main()
    validate_registry(PROBLEMS)
    mkpath(OUTDIR)
    csv_path = joinpath(OUTDIR, "registry.csv")
    report_path = joinpath(OUTDIR, "README.md")
    write_csv(csv_path, PROBLEMS)
    write_report(report_path, PROBLEMS)
    @printf("wrote %s\n", report_path)
    @printf("wrote %s\n", csv_path)
    for row in PROBLEMS
        local_path = row.runnable_path === nothing ? "-" : maybe_rel(row.runnable_path)
        @printf("%-12s %-10s %s\n", row.key, row.status, local_path)
    end
end

main()
