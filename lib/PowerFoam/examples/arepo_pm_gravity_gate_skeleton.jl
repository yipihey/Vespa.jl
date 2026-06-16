#!/usr/bin/env julia

using Dates
using Printf

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
const OUTBASE = joinpath(@__DIR__, "out", "arepo_pm_gravity_gate_skeleton")
const RUN_TAG = Dates.format(now(), "yyyymmdd_HHMMSS")
const OUTDIR = joinpath(OUTBASE, RUN_TAG)

using PowerFoam

const POISSONKERNELS_PROBE = PowerFoam.probe_poissonkernels_monorepo()
const HAVE_POISSONKERNELS = POISSONKERNELS_PROBE.pm_module !== nothing

function ensure_outdir()
    mkpath(OUTDIR)
    return OUTDIR
end

function format_field(x)
    if x === nothing
        return ""
    elseif x isa AbstractFloat
        return @sprintf("%.16e", x)
    elseif x isa Integer
        return string(x)
    else
        return replace(string(x), '\n' => ' ')
    end
end

function write_rows_csv(path, rows)
    open(path, "w") do io
        println(io, "category,label,status,value,reference,delta,note")
        for row in rows
            fields = (
                row.category,
                row.label,
                row.status,
                format_field(row.value),
                format_field(row.reference),
                format_field(row.delta),
                replace(row.note, '"' => '\''),
            )
            println(io, join(("\"$(field)\"" for field in fields), ","))
        end
    end
end

function write_readme(path, result)
    open(path, "w") do io
        println(io, "# AREPO PM Gravity Tiny-N Preflight")
        println(io)
        println(io, "- timestamp: `", RUN_TAG, "`")
        println(io, "- PoissonKernels available: `", HAVE_POISSONKERNELS, "`")
        println(io, "- PM Green's function: `", result.greens, "`")
        println(io, "- fixture: 4 equal-mass particles in `[0,1)^3` at `Npm=16` cell centers")
        println(io)
        println(io, "## Numeric rows")
        println(io)
        println(io, "| category | label | status | value | reference | delta |")
        println(io, "| --- | --- | --- | ---: | ---: | ---: |")
        for row in result.rows
            println(io, "| ", row.category, " | ", row.label, " | ", row.status, " | ",
                    format_field(row.value), " | ", format_field(row.reference), " | ",
                    format_field(row.delta), " |")
        end
        println(io)
        println(io, "## Notes")
        println(io)
        if HAVE_POISSONKERNELS
            println(io, "- The PM chain executed through deposit, FFT solve, periodic ghost fill, gradient, and particle interpolation.")
            self_rows = filter(row -> row.category == "pm_self_control" &&
                                      row.label == "one_particle_max_abs_accel",
                               result.rows)
            if !isempty(self_rows)
                println(io, "- The one-particle periodic PM self-force control reports max acceleration `",
                        format_field(first(self_rows).value), "`.")
            end
        else
            println(io, "- The run fell back to direct image-sum diagnostics only because `using PoissonKernels` failed.")
            println(io, "- Load error: `", sprint(showerror, POISSONKERNELS_PROBE.error), "`")
        end
        println(io, "- The `direct_oracle` rows use a finite symmetric image sum with zero net-force projection.")
        println(io, "- The PM-vs-direct rows are diagnostic until the finite oracle is certified against the production periodic convention.")
        println(io, "- The `direct_diag` rows remain raw finite image-sum diagnostics for convergence context.")
    end
end

function print_summary(result)
    println("AREPO PM gravity tiny periodic preflight")
    println("run tag: ", RUN_TAG)
    println("PoissonKernels available: ", HAVE_POISSONKERNELS)
    println()
    println(" category           | label                           | status   | value")
    println("------------------- | ------------------------------- | -------- | ------------------")
    for row in result.rows
        val = row.value === nothing ? row.note : format_field(row.value)
        @printf("%-19s | %-31s | %-8s | %s\n",
                row.category, row.label, row.status, val)
    end
end

function main()
    pkmod = POISSONKERNELS_PROBE.pm_module
    result = run_arepo_pm_gravity_preflight(pkmod; pk_probe = POISSONKERNELS_PROBE)
    outdir = ensure_outdir()
    csv_path = joinpath(outdir, "preflight_rows.csv")
    readme_path = joinpath(outdir, "README.md")
    write_rows_csv(csv_path, result.rows)
    write_readme(readme_path, result)
    print_summary(result)
    println()
    println("wrote: ", relpath(csv_path, REPO_ROOT))
    println("wrote: ", relpath(readme_path, REPO_ROOT))
end

main()
