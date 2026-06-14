using Printf

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
const GEOMETRY_GATE = joinpath(@__DIR__, "arepo_geometry_gate_3d.jl")
const GEOMETRY_OUT = joinpath(@__DIR__, "out", "arepo_geometry_gate_3d")
const OUTBASE = joinpath(@__DIR__, "out", "arepo_solver_matrix_3d")

parse_arg(i, default, T) = length(ARGS) >= i ? parse(T, ARGS[i]) : default
parse_list_arg(i, default, T) =
    split(length(ARGS) >= i ? ARGS[i] : default, ",") .|> x -> parse(T, strip(x))

const N = parse_arg(1, 12, Int)
const DT = parse_arg(2, 0.001, Float64)
const STEP_COUNTS = parse_list_arg(3, "8", Int)
const SOLVERS = Symbol.(split(length(ARGS) >= 4 ? ARGS[4] : "hll,llf", ","))
const REUSE_EXISTING = lowercase(get(ENV, "POWERFOAM_REUSE_GEOMETRY", "false")) in
                       ("1", "true", "yes")
const STEP_TAG = length(STEP_COUNTS) == 1 ? @sprintf("n%d", only(STEP_COUNTS)) :
                 "n" * join(STEP_COUNTS, "-")
const RUN_TAG = replace(@sprintf("N%d_dt%.3g_%s_%s", N, DT, STEP_TAG,
                                 join(String.(SOLVERS), "-")), "." => "p")
const OUTDIR = joinpath(OUTBASE, RUN_TAG)

function solver_tag(solver::Symbol, nsteps::Int)
    tag = nsteps == 1 ? @sprintf("N%d_dt%.3g_%s", N, DT, solver) :
          @sprintf("N%d_dt%.3g_n%d_%s", N, DT, nsteps, solver)
    return replace(tag, "." => "p")
end

function ensure_geometry_gate!(solver::Symbol, nsteps::Int)
    metrics = joinpath(GEOMETRY_OUT, solver_tag(solver, nsteps), "metrics.csv")
    report = joinpath(GEOMETRY_OUT, solver_tag(solver, nsteps), "README.md")
    if REUSE_EXISTING && isfile(metrics) && isfile(report)
        @printf("reusing AREPO geometry gate for solver=%s steps=%d\n", solver, nsteps)
        return (; solver, nsteps, metrics, report)
    end
    args = String.(Base.julia_cmd().exec)
    project = Base.active_project()
    project === nothing || push!(args, "--project=$(dirname(project))")
    append!(args, [GEOMETRY_GATE, string(N), string(DT), String(solver), string(nsteps)])
    cmd = Cmd(Cmd(args); dir = REPO_ROOT)
    @printf("running AREPO geometry gate for solver=%s steps=%d\n", solver, nsteps)
    run(cmd)
    isfile(metrics) || error("missing metrics for solver=$solver steps=$nsteps at $metrics")
    isfile(report) || error("missing report for solver=$solver steps=$nsteps at $report")
    return (; solver, nsteps, metrics, report)
end

function parse_metrics(path)
    lines = readlines(path)
    rows = NamedTuple[]
    for line in lines[2:end]
        isempty(strip(line)) && continue
        f = split(line, ",")
        length(f) == 14 || error("unexpected metrics row in $path: $line")
        push!(rows, (; label = String(f[1]),
                     step = parse(Int, f[2]),
                     time = parse(Float64, f[3]),
                     mass = parse(Float64, f[4]),
                     mx = parse(Float64, f[5]),
                     my = parse(Float64, f[6]),
                     mz = parse(Float64, f[7]),
                     energy = parse(Float64, f[8]),
                     vrms = parse(Float64, f[9]),
                     mach_rms = parse(Float64, f[10]),
                     density_rms = parse(Float64, f[11]),
                     rho_min = parse(Float64, f[12]),
                     rho_max = parse(Float64, f[13]),
                     pmin = parse(Float64, f[14])))
    end
    return rows
end

function final_row(rows, label)
    candidates = filter(r -> r.label == label, rows)
    isempty(candidates) && error("no rows with label=$label")
    return candidates[argmax(getfield.(candidates, :step))]
end

function initial_row(rows)
    candidates = filter(r -> r.label == "AREPO init in PowerFoam table", rows)
    isempty(candidates) && error("no initial row")
    return first(candidates)
end

function summarize_case(case)
    rows = parse_metrics(case.metrics)
    init = initial_row(rows)
    cpu = final_row(rows, "PowerFoam CPU")
    metal = final_row(rows, "PowerFoam Metal")
    return (; case.solver, case.nsteps, case.metrics, case.report, init, cpu, metal,
            cpu_dmass = cpu.mass - init.mass,
            cpu_denergy = cpu.energy - init.energy,
            metal_dmass = metal.mass - init.mass,
            metal_denergy = metal.energy - init.energy)
end

baseline_for(summaries, nsteps) = first(filter(s -> s.nsteps == nsteps, summaries))
summary_for(summaries, nsteps, solver) =
    first(filter(s -> s.nsteps == nsteps && s.solver == solver, summaries))

function write_summary_csv(path, summaries)
    open(path, "w") do io
        println(io, "nsteps,solver,backend,step,time,vrms,mach_rms,density_rms,rho_min,rho_max,pmin,mass,energy,dmass,denergy,delta_vrms_vs_step_baseline,delta_density_rms_vs_step_baseline,delta_pmin_vs_step_baseline")
        for s in summaries
            baseline = baseline_for(summaries, s.nsteps).metal
            for (backend, row, dmass, denergy) in
                (("CPU Float32", s.cpu, s.cpu_dmass, s.cpu_denergy),
                 ("Metal Float32", s.metal, s.metal_dmass, s.metal_denergy))
                @printf(io, "%d,%s,%s,%d,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g\n",
                        s.nsteps, s.solver, backend, row.step, row.time, row.vrms,
                        row.mach_rms, row.density_rms, row.rho_min, row.rho_max,
                        row.pmin, row.mass, row.energy, dmass, denergy,
                        row.vrms - baseline.vrms,
                        row.density_rms - baseline.density_rms,
                        row.pmin - baseline.pmin)
            end
        end
    end
end

function relpath_from_repo(path)
    return relpath(path, REPO_ROOT)
end

function write_report(path, summaries)
    open(path, "w") do io
        println(io, "# AREPO moving-mesh GPU solver-choice matrix")
        println(io)
        println(io, "This campaign reruns the AREPO 3-D turbulence geometry gate for")
        println(io, "each Julia Riemann solver choice on the same staged AREPO initial")
        println(io, "condition. Each case exports AREPO's live moving Voronoi face rings,")
        println(io, "runs the PowerFoam face-table hydro path on CPU and Metal, and")
        println(io, "keeps the per-solver geometry gate report as the detailed artifact.")
        println(io)
        @printf(io, "- N: %d^3 cells\n", N)
        @printf(io, "- dt: %.8g\n", DT)
        @printf(io, "- step counts: `%s`\n", join(STEP_COUNTS, "`, `"))
        @printf(io, "- solvers: `%s`\n", join(String.(SOLVERS), "`, `"))
        println(io)
        println(io, "## Metal final-state comparison")
        println(io)
        println(io, "| steps | solver | vrms | mach_rms | density_rms | rho_min | rho_max | pmin | mass drift | energy drift |")
        println(io, "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for s in summaries
            r = s.metal
            @printf(io, "| %d | %s | %.8g | %.8g | %.8g | %.8g | %.8g | %.8g | %.9g | %.9g |\n",
                    s.nsteps, s.solver, r.vrms, r.mach_rms, r.density_rms, r.rho_min,
                    r.rho_max, r.pmin, s.metal_dmass, s.metal_denergy)
        end
        println(io)
        println(io, "## Metal deltas relative to first solver at each step count")
        println(io)
        @printf(io, "Baseline solver: `%s`.\n", first(SOLVERS))
        println(io)
        println(io, "| steps | solver | dvrms | dmach_rms | ddensity_rms | drho_min | drho_max | dpmin |")
        println(io, "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for s in summaries
            baseline = baseline_for(summaries, s.nsteps).metal
            r = s.metal
            @printf(io, "| %d | %s | %.9g | %.9g | %.9g | %.9g | %.9g | %.9g |\n",
                    s.nsteps, s.solver, r.vrms - baseline.vrms,
                    r.mach_rms - baseline.mach_rms,
                    r.density_rms - baseline.density_rms,
                    r.rho_min - baseline.rho_min,
                    r.rho_max - baseline.rho_max,
                    r.pmin - baseline.pmin)
        end
        println(io)
        println(io, "## Observed trend")
        println(io)
        last_steps = maximum(STEP_COUNTS)
        if length(SOLVERS) > 1
            for solver in SOLVERS[2:end]
                early = summary_for(summaries, minimum(STEP_COUNTS), solver).metal
                early_base = baseline_for(summaries, minimum(STEP_COUNTS)).metal
                late = summary_for(summaries, last_steps, solver).metal
                late_base = baseline_for(summaries, last_steps).metal
                @printf(io, "- `%s` relative to `%s`: dvrms grows from %.9g at %d steps to %.9g at %d steps; ddensity_rms grows from %.9g to %.9g; dpmin moves from %.9g to %.9g.\n",
                        solver, first(SOLVERS),
                        early.vrms - early_base.vrms, minimum(STEP_COUNTS),
                        late.vrms - late_base.vrms, last_steps,
                        early.density_rms - early_base.density_rms,
                        late.density_rms - late_base.density_rms,
                        early.pmin - early_base.pmin,
                        late.pmin - late_base.pmin)
            end
        else
            println(io, "Only one solver was requested, so no cross-solver trend is available.")
        end
        println(io)
        println(io, "## Per-solver artifacts")
        println(io)
        println(io, "| steps | solver | detailed report | metrics |")
        println(io, "| ---: | --- | --- | --- |")
        for s in summaries
            @printf(io, "| %d | %s | `%s` | `%s` |\n",
                    s.nsteps, s.solver, relpath_from_repo(s.report),
                    relpath_from_repo(s.metrics))
        end
        println(io)
        println(io, "## Interpretation boundary")
        println(io)
        println(io, "This is a GPU solver-choice comparison on AREPO's moving face table,")
        println(io, "including the reconstructed predictor gate inside each detailed report.")
        println(io, "It is not yet a full end-to-end native GPU AREPO replacement, because")
        println(io, "the production-size 3-D mesh rebuild in the geometry gate is still hosted")
        println(io, "by AREPO's native tessellator. The native Julia periodic rebuild currently")
        println(io, "covers the small-mesh contract gate only.")
    end
end

function main()
    mkpath(OUTDIR)
    cases = [ensure_geometry_gate!(solver, nsteps)
             for nsteps in STEP_COUNTS for solver in SOLVERS]
    summaries = summarize_case.(cases)
    csv = joinpath(OUTDIR, "solver_summary.csv")
    report = joinpath(OUTDIR, "README.md")
    write_summary_csv(csv, summaries)
    write_report(report, summaries)
    @printf("wrote %s\n", csv)
    @printf("wrote %s\n", report)
    for s in summaries
        baseline = baseline_for(summaries, s.nsteps).metal
        @printf("n=%d %s Metal: vrms=%.6g density_rms=%.6g pmin=%.6g dvrms=%.4g mass_drift=%.4g energy_drift=%.4g\n",
                s.nsteps, s.solver, s.metal.vrms, s.metal.density_rms,
                s.metal.pmin, s.metal.vrms - baseline.vrms,
                s.metal_dmass, s.metal_denergy)
    end
end

main()
