using Dates
using Printf
using PowerFoam

const DEFAULT_N = 24
const DEFAULT_TFINAL = 0.2
const DEFAULT_NBINS = 24
const DEFAULT_SOLVER = "hll"
const DEFAULT_CFL = 0.18
const DEFAULT_DOMAIN_RADIUS = 3.0
const OUTBASE = joinpath(@__DIR__, "out", "arepo_noh2d_gate")

parse_arg(i, default, T) = length(ARGS) >= i ? parse(T, ARGS[i]) : default
parse_arg(i, default, ::Type{String}) = length(ARGS) >= i ? ARGS[i] : default

const N = parse_arg(1, DEFAULT_N, Int)
const TFINAL = parse_arg(2, DEFAULT_TFINAL, Float64)
const NBINS = parse_arg(3, DEFAULT_NBINS, Int)
const RIEMANN = Symbol(lowercase(parse_arg(4, DEFAULT_SOLVER, String)))
const RUN_TAG = pf_noh2d_run_tag(N, TFINAL, RIEMANN)
const OUTDIR = joinpath(OUTBASE, RUN_TAG)

csvquote(v) = "\"" * replace(string(v), "\"" => "\"\"") * "\""
rel(path) = relpath(path, OUTDIR)

function write_metrics_csv(path, rows)
    open(path, "w") do io
        println(io, "label,step,t,mass,energy,mx,my,mass_rel_drift,energy_rel_drift,rho_min,rho_max,p_min,p_max,shock_radius_proxy,analytic_shock_radius,shock_radius_error,analysis_radius")
        for row in rows
            vals = (row.label, row.step, row.t, row.mass, row.energy, row.mx, row.my,
                    row.mass_rel_drift, row.energy_rel_drift, row.rho_min, row.rho_max,
                    row.p_min, row.p_max, row.shock_radius_proxy, row.analytic_shock_radius,
                    row.shock_radius_error, row.analysis_radius)
            println(io, join((csvquote(v) for v in vals), ","))
        end
    end
end

function write_radial_bins_csv(path, rows)
    open(path, "w") do io
        println(io, "bin,r_inner,r_outer,r_mid,count,volume,mass,rho_mean,pressure_mean,vrad_mean")
        for row in rows
            vals = (row.bin, row.r_inner, row.r_outer, row.r_mid, row.count,
                    row.volume, row.mass, row.rho_mean, row.pressure_mean, row.vrad_mean)
            println(io, join((csvquote(v) for v in vals), ","))
        end
    end
end

function write_log(path, run)
    open(path, "w") do io
        println(io, "# PowerFoam Noh2D executable gate log")
        @printf(io, "generated=%s\n", Dates.format(now(), dateformat"yyyy-mm-dd HH:MM:SS"))
        @printf(io, "n_side=%d\n", run.n_side)
        @printf(io, "t_final=%.12g\n", run.t_final)
        @printf(io, "riemann=%s\n", run.riemann)
        @printf(io, "status=%s\n", run.status)
        @printf(io, "numerics_ok=%s\n", run.numerics_ok)
        @printf(io, "steps=%d\n", length(run.logs))
        for row in run.logs
            @printf(io,
                    "step=%d dt=%.12g t=%.12g mass_rel_drift=%.12g energy_rel_drift=%.12g rho_min=%.12g rho_max=%.12g p_min=%.12g p_max=%.12g shock_radius_proxy=%.12g\n",
                    row.step, row.dt, row.t, row.mass_rel_drift, row.energy_rel_drift,
                    row.rho_min, row.rho_max, row.p_min, row.p_max, row.shock_radius_proxy)
        end
    end
end

function write_report(path, run, metrics_csv, radial_csv, log_path)
    final = run.final_metric
    open(path, "w") do io
        println(io, "# PowerFoam Noh2D Executable Gate")
        println(io)
        println(io, "This is the first executable PowerFoam standard-problem gate for")
        println(io, "the AREPO rewrite surface. It generates a small bounded 2-D Noh")
        println(io, "initial condition in Julia, advances it with the existing 2-D")
        println(io, "PowerFoam hydro API, and writes numeric diagnostics.")
        println(io)
        println(io, "The gate is executable today, but its physics label remains")
        println(io, "`calibration-PENDING` until the minimal closed-boundary run is")
        println(io, "tied to the planned original-AREPO and analytic pass thresholds.")
        println(io)
        @printf(io, "- generated: %s\n", Dates.format(now(), dateformat"yyyy-mm-dd HH:MM:SS"))
        @printf(io, "- status: `%s`\n", run.status)
        @printf(io, "- numerics sane: `%s`\n", run.numerics_ok ? "yes" : "no")
        @printf(io, "- grid: `%d x %d`\n", run.n_side, run.n_side)
        @printf(io, "- t_final: `%.12g`\n", run.t_final)
        @printf(io, "- solver: `%s`\n", run.riemann)
        @printf(io, "- domain: `[-%.1f, %.1f]^2`\n", run.domain_radius, run.domain_radius)
        @printf(io, "- diagnostics: `%s`, `%s`, `%s`\n",
                rel(metrics_csv), rel(radial_csv), rel(log_path))
        println(io)
        println(io, "## Final Diagnostics")
        println(io)
        println(io, "| metric | value |")
        println(io, "| --- | ---: |")
        @printf(io, "| mass | %.12g |\n", final.mass)
        @printf(io, "| energy | %.12g |\n", final.energy)
        @printf(io, "| mass rel drift | %.12g |\n", final.mass_rel_drift)
        @printf(io, "| energy rel drift | %.12g |\n", final.energy_rel_drift)
        @printf(io, "| rho min | %.12g |\n", final.rho_min)
        @printf(io, "| rho max | %.12g |\n", final.rho_max)
        @printf(io, "| p min | %.12g |\n", final.p_min)
        @printf(io, "| p max | %.12g |\n", final.p_max)
        @printf(io, "| shock radius proxy | %.12g |\n", final.shock_radius_proxy)
        @printf(io, "| analytic shock radius (t/3) | %.12g |\n", final.analytic_shock_radius)
        @printf(io, "| shock radius proxy error | %.12g |\n", final.shock_radius_error)
        println(io)
        println(io, "## Gate Label")
        println(io)
        if run.numerics_ok
            println(io, "`calibration-PENDING`: the short run stayed positive and conservative")
            println(io, "and formed a compressed central region, but this first executable")
            println(io, "rung is still missing the original-AREPO and analytic pass surface.")
        else
            println(io, "`run-FAIL`: the executable rung did not satisfy its minimal sanity")
            println(io, "checks, so the gate should not be treated as a usable calibration baseline.")
        end
        println(io)
        println(io, "## Package Surface")
        println(io)
        println(io, "The Noh2D helper functions are loaded through `using PowerFoam`.")
    end
end

function main()
    mkpath(OUTDIR)
    run = pf_noh2d_run(; n_side = N, t_final = TFINAL, nbins = NBINS,
                       cfl = DEFAULT_CFL, domain_radius = DEFAULT_DOMAIN_RADIUS,
                       riemann = RIEMANN)
    metrics_csv = joinpath(OUTDIR, "metrics_powerfoam.csv")
    radial_csv = joinpath(OUTDIR, "radial_bins_powerfoam.csv")
    log_path = joinpath(OUTDIR, "powerfoam.log")
    report_path = joinpath(OUTDIR, "README.md")
    write_metrics_csv(metrics_csv, run.history)
    write_radial_bins_csv(radial_csv, run.radial_bins)
    write_log(log_path, run)
    write_report(report_path, run, metrics_csv, radial_csv, log_path)
    @printf("status=%s\n", run.status)
    @printf("metrics=%s\n", metrics_csv)
    @printf("radial_bins=%s\n", radial_csv)
    @printf("log=%s\n", log_path)
    @printf("report=%s\n", report_path)
    return run.numerics_ok ? 0 : 1
end

exit(main())
