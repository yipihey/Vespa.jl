using Dates
using Printf
using PowerFoam

const DEFAULT_NX = 32
const DEFAULT_NY = 8
const DEFAULT_TFINAL = 0.05
const DEFAULT_SOLVER = "hll"
const DEFAULT_CFL = 0.25
const DEFAULT_AMPLITUDE = 1e-3
const DEFAULT_MODE = 1
const OUTBASE = joinpath(@__DIR__, "out", "arepo_soundwave2d_gate")

parse_arg(i, default, T) = length(ARGS) >= i ? parse(T, ARGS[i]) : default
parse_arg(i, default, ::Type{String}) = length(ARGS) >= i ? ARGS[i] : default

const NX = parse_arg(1, DEFAULT_NX, Int)
const NY = parse_arg(2, DEFAULT_NY, Int)
const TFINAL = parse_arg(3, DEFAULT_TFINAL, Float64)
const RIEMANN = Symbol(lowercase(parse_arg(4, DEFAULT_SOLVER, String)))
const RUN_TAG = pf_soundwave2d_run_tag(NX, NY, TFINAL, RIEMANN)
const OUTDIR = joinpath(OUTBASE, RUN_TAG)

csvquote(v) = "\"" * replace(string(v), "\"" => "\"\"") * "\""
rel(path) = relpath(path, OUTDIR)

function write_metrics_csv(path, rows)
    open(path, "w") do io
        println(io, "label,step,t,mass,energy,mx,my,mass_rel_drift,energy_rel_drift,rho_min,rho_max,p_min,p_max,rho_l1,rho_l2,vx_l1,vx_l2,pressure_l1,pressure_l2,rho_mode_amp,rho_exact_mode_amp,rho_mode_amp_ratio,rho_mode_phase_error")
        for row in rows
            vals = (row.label, row.step, row.t, row.mass, row.energy, row.mx, row.my,
                    row.mass_rel_drift, row.energy_rel_drift, row.rho_min, row.rho_max,
                    row.p_min, row.p_max, row.rho_l1, row.rho_l2, row.vx_l1, row.vx_l2,
                    row.pressure_l1, row.pressure_l2, row.rho_mode_amp,
                    row.rho_exact_mode_amp, row.rho_mode_amp_ratio,
                    row.rho_mode_phase_error)
            println(io, join((csvquote(v) for v in vals), ","))
        end
    end
end

function write_profile_csv(path, rows)
    open(path, "w") do io
        println(io, "x,y,rho,rho_exact,vx,vx_exact,pressure,pressure_exact")
        for row in rows
            vals = (row.x, row.y, row.rho, row.rho_exact, row.vx, row.vx_exact,
                    row.pressure, row.pressure_exact)
            println(io, join((csvquote(v) for v in vals), ","))
        end
    end
end

function write_log(path, run)
    open(path, "w") do io
        println(io, "# PowerFoam sound-wave 2D executable gate log")
        @printf(io, "generated=%s\n", Dates.format(now(), dateformat"yyyy-mm-dd HH:MM:SS"))
        @printf(io, "nx=%d\n", run.nx)
        @printf(io, "ny=%d\n", run.ny)
        @printf(io, "t_final=%.12g\n", run.t_final)
        @printf(io, "riemann=%s\n", run.riemann)
        @printf(io, "status=%s\n", run.status)
        @printf(io, "numerics_ok=%s\n", run.numerics_ok)
        @printf(io, "steps=%d\n", length(run.logs))
        for row in run.logs
            @printf(io,
                    "step=%d dt=%.12g t=%.12g mass_rel_drift=%.12g energy_rel_drift=%.12g rho_l2=%.12g vx_l2=%.12g pressure_l2=%.12g rho_mode_amp_ratio=%.12g rho_mode_phase_error=%.12g\n",
                    row.step, row.dt, row.t, row.mass_rel_drift,
                    row.energy_rel_drift, row.rho_l2, row.vx_l2, row.pressure_l2,
                    row.rho_mode_amp_ratio, row.rho_mode_phase_error)
        end
    end
end

function write_report(path, run, metrics_csv, profile_csv, log_path)
    final = run.final_metric
    open(path, "w") do io
        println(io, "# PowerFoam Sound-Wave 2D Executable Gate")
        println(io)
        println(io, "This is a small periodic smooth-wave calibration rung for the")
        println(io, "AREPO rewrite surface. It advances a low-amplitude 2-D acoustic")
        println(io, "mode with the existing PowerFoam 2-D hydro API and records")
        println(io, "conservation, exact-solution error, and Fourier-mode diagnostics.")
        println(io)
        println(io, "The gate is intentionally labeled `calibration-PENDING` until it")
        println(io, "is tied either to an upstream AREPO wave reference or to a frozen")
        println(io, "problem-specific tolerance surface derived from repeated runs.")
        println(io)
        @printf(io, "- generated: %s\n", Dates.format(now(), dateformat"yyyy-mm-dd HH:MM:SS"))
        @printf(io, "- status: `%s`\n", run.status)
        @printf(io, "- numerics sane: `%s`\n", run.numerics_ok ? "yes" : "no")
        @printf(io, "- grid: `%d x %d`\n", run.nx, run.ny)
        @printf(io, "- t_final: `%.12g`\n", run.t_final)
        @printf(io, "- solver: `%s`\n", run.riemann)
        @printf(io, "- diagnostics: `%s`, `%s`, `%s`\n",
                rel(metrics_csv), rel(profile_csv), rel(log_path))
        println(io)
        println(io, "## Final Diagnostics")
        println(io)
        println(io, "| metric | value |")
        println(io, "| --- | ---: |")
        @printf(io, "| mass rel drift | %.12g |\n", final.mass_rel_drift)
        @printf(io, "| energy rel drift | %.12g |\n", final.energy_rel_drift)
        @printf(io, "| rho L1 | %.12g |\n", final.rho_l1)
        @printf(io, "| rho L2 | %.12g |\n", final.rho_l2)
        @printf(io, "| vx L1 | %.12g |\n", final.vx_l1)
        @printf(io, "| vx L2 | %.12g |\n", final.vx_l2)
        @printf(io, "| pressure L1 | %.12g |\n", final.pressure_l1)
        @printf(io, "| pressure L2 | %.12g |\n", final.pressure_l2)
        @printf(io, "| rho mode amplitude ratio | %.12g |\n", final.rho_mode_amp_ratio)
        @printf(io, "| rho mode phase error | %.12g |\n", final.rho_mode_phase_error)
        println(io)
        println(io, "## Gate Label")
        println(io)
        if run.numerics_ok
            println(io, "`calibration-PENDING`: the short periodic run stayed")
            println(io, "positive, nearly conservative, and retained a sensible first-mode")
            println(io, "signal, but the problem still needs a frozen pass surface.")
        else
            println(io, "`run-FAIL`: the smooth-wave rung failed one of its")
            println(io, "minimal numerical sanity checks and should not be used as a")
            println(io, "calibration baseline.")
        end
        println(io)
        println(io, "## Package Surface")
        println(io)
        println(io, "The helper functions are loaded through `using PowerFoam`.")
    end
end

function main()
    mkpath(OUTDIR)
    run = pf_soundwave2d_run(; nx = NX, ny = NY, t_final = TFINAL,
                             cfl = DEFAULT_CFL, amplitude = DEFAULT_AMPLITUDE,
                             mode = DEFAULT_MODE, riemann = RIEMANN)
    metrics_csv = joinpath(OUTDIR, "metrics_powerfoam.csv")
    profile_csv = joinpath(OUTDIR, "profile_powerfoam.csv")
    log_path = joinpath(OUTDIR, "powerfoam.log")
    report_path = joinpath(OUTDIR, "README.md")
    write_metrics_csv(metrics_csv, run.history)
    write_profile_csv(profile_csv, run.profile_rows)
    write_log(log_path, run)
    write_report(report_path, run, metrics_csv, profile_csv, log_path)
    @printf("status=%s\n", run.status)
    @printf("metrics=%s\n", metrics_csv)
    @printf("profile=%s\n", profile_csv)
    @printf("log=%s\n", log_path)
    @printf("report=%s\n", report_path)
    return run.numerics_ok ? 0 : 1
end

exit(main())
