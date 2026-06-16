using Dates
using Printf
using PowerFoam

const DEFAULT_NX = 32
const DEFAULT_NY = 32
const DEFAULT_TFINAL = 0.02
const DEFAULT_SOLVER = "hll"
const DEFAULT_CFL = 0.18
const DEFAULT_NBINS = 24
const OUTBASE = joinpath(@__DIR__, "out", "arepo_gresho2d_gate")

parse_arg(i, default, T) = length(ARGS) >= i ? parse(T, ARGS[i]) : default
parse_arg(i, default, ::Type{String}) = length(ARGS) >= i ? ARGS[i] : default

const NX = parse_arg(1, DEFAULT_NX, Int)
const NY = parse_arg(2, DEFAULT_NY, Int)
const TFINAL = parse_arg(3, DEFAULT_TFINAL, Float64)
const RIEMANN = Symbol(lowercase(parse_arg(4, DEFAULT_SOLVER, String)))
const RUN_TAG = pf_gresho2d_run_tag(NX, NY, TFINAL, RIEMANN)
const OUTDIR = joinpath(OUTBASE, RUN_TAG)

csvquote(v) = "\"" * replace(string(v), "\"" => "\"\"") * "\""
rel(path) = relpath(path, OUTDIR)

function write_metrics_csv(path, rows)
    open(path, "w") do io
        println(io, "label,step,t,mass,energy,mx,my,mass_rel_drift,energy_rel_drift,rho_min,rho_max,p_min,p_max,vt_l1,vt_l2,vt_peak_ratio,vt_peak_radius,vt_peak_radius_error,analysis_radius")
        for row in rows
            vals = (row.label, row.step, row.t, row.mass, row.energy, row.mx, row.my,
                    row.mass_rel_drift, row.energy_rel_drift, row.rho_min, row.rho_max,
                    row.p_min, row.p_max, row.vt_l1, row.vt_l2, row.vt_peak_ratio,
                    row.vt_peak_radius, row.vt_peak_radius_error, row.analysis_radius)
            println(io, join((csvquote(v) for v in vals), ","))
        end
    end
end

function write_profile_csv(path, rows)
    open(path, "w") do io
        println(io, "bin,r_inner,r_outer,r_mid,count,volume,mass,rho_mean,vt_mean,vt_exact_mean,vt_abs_error,pressure_mean,pressure_exact_mean")
        for row in rows
            vals = (row.bin, row.r_inner, row.r_outer, row.r_mid, row.count,
                    row.volume, row.mass, row.rho_mean, row.vt_mean,
                    row.vt_exact_mean, row.vt_abs_error, row.pressure_mean,
                    row.pressure_exact_mean)
            println(io, join((csvquote(v) for v in vals), ","))
        end
    end
end

function write_log(path, run)
    open(path, "w") do io
        println(io, "# PowerFoam Gresho 2D executable gate log")
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
                    "step=%d dt=%.12g t=%.12g mass_rel_drift=%.12g energy_rel_drift=%.12g rho_min=%.12g rho_max=%.12g p_min=%.12g p_max=%.12g vt_l2=%.12g vt_peak_ratio=%.12g\n",
                    row.step, row.dt, row.t, row.mass_rel_drift, row.energy_rel_drift,
                    row.rho_min, row.rho_max, row.p_min, row.p_max, row.vt_l2,
                    row.vt_peak_ratio)
        end
    end
end

function write_report(path, run, metrics_csv, profile_csv, log_path)
    final = run.final_metric
    open(path, "w") do io
        println(io, "# PowerFoam Gresho 2D Executable Gate")
        println(io)
        println(io, "This is a small periodic vortex calibration rung for the AREPO")
        println(io, "rewrite surface. It advances the bounded Gresho vortex initial")
        println(io, "state in Julia and records conservation, positivity, and a radial")
        println(io, "rotational-profile proxy.")
        println(io)
        println(io, "The gate is intentionally labeled `calibration-PENDING` until it")
        println(io, "is tied either to an upstream AREPO Gresho profile reference or to")
        println(io, "a frozen local tolerance surface derived from repeated runs.")
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
        @printf(io, "| rho min | %.12g |\n", final.rho_min)
        @printf(io, "| rho max | %.12g |\n", final.rho_max)
        @printf(io, "| p min | %.12g |\n", final.p_min)
        @printf(io, "| p max | %.12g |\n", final.p_max)
        @printf(io, "| vtheta L1 | %.12g |\n", final.vt_l1)
        @printf(io, "| vtheta L2 | %.12g |\n", final.vt_l2)
        @printf(io, "| vtheta peak ratio | %.12g |\n", final.vt_peak_ratio)
        @printf(io, "| vtheta peak radius | %.12g |\n", final.vt_peak_radius)
        @printf(io, "| vtheta peak radius error | %.12g |\n", final.vt_peak_radius_error)
        println(io)
        println(io, "## Gate Label")
        println(io)
        if run.numerics_ok
            println(io, "`calibration-PENDING`: the short periodic vortex run stayed")
            println(io, "positive and nearly conservative, but this first executable")
            println(io, "rung still needs an upstream AREPO comparison or a frozen local")
            println(io, "tolerance surface before it can move beyond calibration.")
        else
            println(io, "`run-FAIL`: the executable rung did not satisfy its minimal sanity")
            println(io, "checks, so the gate should not be treated as a usable calibration baseline.")
        end
        println(io)
        println(io, "## Package Surface")
        println(io)
        println(io, "The Gresho2D helper functions are loaded through `using PowerFoam`.")
    end
end

function main()
    mkpath(OUTDIR)
    run = pf_gresho2d_run(; nx = NX, ny = NY, t_final = TFINAL,
                          cfl = DEFAULT_CFL, nbins = DEFAULT_NBINS,
                          riemann = RIEMANN)
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
