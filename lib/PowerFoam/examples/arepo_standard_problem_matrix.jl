using Dates
using Printf

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
const OUTBASE = joinpath(@__DIR__, "out", "arepo_standard_problem_matrix")
const RUN_MODE = Symbol(lowercase(get(ENV, "POWERFOAM_STANDARD_RUN", "report")))
const RUN_TAG = string(Dates.format(now(), "yyyymmdd_HHMMSS"), "_", RUN_MODE)
const OUTDIR = joinpath(OUTBASE, RUN_TAG)
const AREPO_LIB_PATH = get(ENV, "AREPO_LIB", "/Users/tabel/Projects/arepo/libarepo.dylib")
const AREPO_LIB_LOAD = "push!(LOAD_PATH, \"/Users/tabel/Projects/Arepo.jl/lib/ArepoLib\")"

struct StandardProblem
    name::String
    dim::Int
    class::String
    arepo_example::String
    powerfoam_surface::String
    current_gate::String
    next_gate::String
    status::String
end

const PROBLEMS = StandardProblem[
    StandardProblem(
        "decaying_subsonic_turbulence_3d", 3, "periodic smooth turbulence",
        "bauer_springel_turbulence_3d",
        "3-D moving Voronoi mesh, gradients, predictor, HLL/LLF flux, hierarchy",
        "N4 native-row replay HLL/LLF; N4/N8 scheduler gates",
        "native production run_step! final-field parity at N8/N12",
        "certified component parity"),
    StandardProblem(
        "multirung_timestep_fixture_3d", 3, "controlled hierarchy stress",
        "bauer_springel_turbulence_3d with IC energy perturbation",
        "AREPO-style hydro bins, gravity-limited effective bins, active lists",
        "N8 HLL multirung hierarchy gate, 3 steps",
        "couple partial drift/rebuild/update to certified scheduler",
        "certified scheduler parity"),
    StandardProblem(
        "noh_3d", 3, "strong spherical shock",
        "noh_3d",
        "positivity, strong-shock update, mesh compression, radial diagnostics",
        "stock AREPO one-step plus fixed, moving full-sync, traced hierarchy-probe, and per-cell final-field PowerFoam comparisons with explicit tolerances",
        "route traced hierarchy output into the same final-field tolerance gate after the active update becomes the production Noh row",
        "traced moving hierarchy probe plus field tolerance diagnostics certified"),
    StandardProblem(
        "noh_2d", 2, "strong cylindrical shock",
        "noh_2d",
        "2-D bounded/moving mesh, artificial viscosity, positivity",
        "executable PowerFoam bounded-mesh gate with conservation and radial diagnostics",
        "tie the executable rung to original AREPO and analytic pass thresholds",
        "executable calibration gate"),
    StandardProblem(
        "sound_wave_2d", 2, "smooth acoustic mode",
        "wave_1d / acoustic_wave_1d semantics on a 2-D periodic mesh",
        "2-D periodic hydro update, low-Mach propagation, Fourier retention",
        "executable PowerFoam periodic-wave gate with exact-solution and mode diagnostics",
        "freeze repeated-run tolerances or hook the rung to an upstream AREPO reference surface",
        "executable calibration gate"),
    StandardProblem(
        "gresho_2d", 2, "vortex balance",
        "gresho_2d",
        "angular momentum, pressure-gradient balance, mesh regularization",
        "executable PowerFoam periodic-vortex gate with radial profile and rotational proxy diagnostics",
        "freeze repeated-run tolerances or add an upstream AREPO profile comparison",
        "executable calibration gate"),
    StandardProblem(
        "sedov_2d", 2, "blast wave",
        "sedov proxy generated locally, not stock public example",
        "2-D blast profile, energy conservation, positivity",
        "PowerFoam/AREPO proxy table generator exists",
        "turn proxy into executable 2-D radial shock/profile parity gate",
        "proxy only"),
    StandardProblem(
        "contact_blob_2d", 2, "pressure-equilibrium advection",
        "contact_blob_2d",
        "contact preservation, moving-vs-static mesh diffusion, scalar transport",
        "AREPO moving/static regression exists in sibling checkout",
        "add PowerFoam 2-D periodic compact/reconstructed final-field parity",
        "AREPO-only regression"),
    StandardProblem(
        "kh_2d_lecoanet", 2, "Kelvin-Helmholtz instability",
        "kh_2d_lecoanet",
        "shear growth, contact handling, reconstruction robustness",
        "AREPO HLL/HLL+PPM reference plus normalized final-field export and PowerFoam fixed-periodic/moving-reconstructed field comparison with explicit tolerances",
        "extend field comparison to HLL+PPM/LLF solver-choice rows",
        "direct periodic moving reconstructed field tolerance diagnostics certified"),
    StandardProblem(
        "shearing_sinusoid_2d", 2, "smooth shear advection",
        "shearing_sinusoid_2d",
        "smooth-wave dissipation, limiter behavior, mesh motion",
        "AREPO example available",
        "add amplitude/harmonic-distortion metric gate",
        "not yet certified"),
    StandardProblem(
        "yee_2d", 2, "vortex advection",
        "yee_2d",
        "smooth vortex preservation and advection error",
        "AREPO example available",
        "add vortex profile and phase-error parity gate",
        "not yet certified"),
    StandardProblem(
        "acoustic_wave_1d", 1, "linear wave",
        "acoustic_wave_1d / wave_1d",
        "low-Mach wave speed and amplitude decay",
        "AREPO example available; PowerFoam currently focuses on 2-D/3-D",
        "add thin-3-D or dedicated 1-D shim before claiming 1-D parity",
        "not yet certified"),
]

mutable struct GateRun
    name::String
    command::String
    report::String
    status::String
    detail::String
end

function rel(path)
    return relpath(path, REPO_ROOT)
end

function julia_expr(expr; env = Dict{String,String}())
    args = String.(Base.julia_cmd().exec)
    project = Base.active_project()
    project === nothing || push!(args, "--project=$(dirname(project))")
    append!(args, ["-e", expr])
    cmd = Cmd(Cmd(args); dir = REPO_ROOT)
    return isempty(env) ? cmd : addenv(cmd, env...)
end

csvquote(v) = "\"" * replace(v, "\"" => "\"\"") * "\""

function run_gate!(runs, name, expr, report;
                   env = Dict{String,String}(),
                   execute = RUN_MODE in (:quick, :extended))
    cmd = julia_expr(expr; env)
    status = "not run"
    detail = ""
    if execute
        try
            run(cmd)
            status = isfile(report) ? "passed" : "ran: missing report"
        catch err
            status = "failed"
            detail = sprint(showerror, err)
        end
    elseif RUN_MODE in (:report, :quick, :extended)
        status = isfile(report) ? "available" : "not run"
    else
        error("unsupported POWERFOAM_STANDARD_RUN=$(RUN_MODE); use report, quick, or extended")
    end
    push!(runs, GateRun(name, string(cmd), report, status, detail))
    return runs
end

function quick_gate_runs()
    runs = GateRun[]
    trace_env = Dict(
        "POWERFOAM_REPLAY_ROWS" => "native",
        "POWERFOAM_REPLAY_GEOMETRY" => "native",
        "POWERFOAM_REPLAY_FACE_VELOCITY" => "native",
        "POWERFOAM_REPLAY_UPDATE_TARGETS" => "native_mesh",
        "POWERFOAM_REPLAY_NATIVE_DT_SOURCE" => "snapshot_time",
    )
    for solver in ("hll", "llf")
        expr = string(AREPO_LIB_LOAD,
                      "; ARGS=[\"4\",\"0.001\",\"", solver, "\",\"1\"]; ",
                      "include(\"lib/PowerFoam/examples/arepo_trace_replay_gate_3d.jl\")")
        report = joinpath(@__DIR__, "out", "arepo_trace_replay_gate_3d",
                          "N4_dt0p001_$(solver)", "README.md")
        run_gate!(runs, "3-D native-row replay $(uppercase(solver))", expr,
                  report; env = trace_env)
    end

    expr_decay = string(AREPO_LIB_LOAD,
                        "; ARGS=[\"4\",\"0.001\",\"hll\",\"1\"]; ",
                        "include(\"lib/PowerFoam/examples/arepo_hierarchy_gate_3d.jl\")")
    report_decay = joinpath(@__DIR__, "out", "arepo_hierarchy_gate_3d",
                            "N4_dt0p001_hll", "README.md")
    run_gate!(runs, "3-D decay scheduler HLL", expr_decay, report_decay)

    expr_multi = string(AREPO_LIB_LOAD,
                        "; ARGS=[\"8\",\"0.001\",\"hll\",\"3\"]; ",
                        "include(\"lib/PowerFoam/examples/arepo_hierarchy_gate_3d.jl\")")
    report_multi = joinpath(@__DIR__, "out", "arepo_hierarchy_gate_3d",
                            "N8_dt0p001_n3_hll_multirung", "README.md")
    run_gate!(runs, "3-D multirung scheduler HLL", expr_multi, report_multi;
              env = Dict("POWERFOAM_HIERARCHY_FIXTURE" => "multirung"))

    expr_noh = string(AREPO_LIB_LOAD,
                      "; ARGS=[\"1\",\"hll\",\"48\"]; ",
                      "include(\"lib/PowerFoam/examples/arepo_noh3d_smoke_gate.jl\")")
    report_noh = joinpath(@__DIR__, "out", "arepo_noh3d_smoke_gate",
                          "stock_n30_n1_hll", "README.md")
    run_gate!(runs, "3-D stock Noh smoke + moving diagnostic HLL", expr_noh, report_noh;
              execute = RUN_MODE == :extended)

    expr_kh = string("ARGS=[\"32\",\"0.1\",\"1.0\",\"64\"]; ",
                     "include(\"lib/PowerFoam/examples/arepo_kh2d_original_gate.jl\")")
    report_kh = joinpath(@__DIR__, "out", "arepo_kh2d_original_gate",
                         "N32_t0p1_drat1", "README.md")
    run_gate!(runs, "2-D original KH HLL/HLL+PPM", expr_kh, report_kh;
              execute = RUN_MODE == :extended)

    expr_pf_kh = string("ARGS=[\"32\",\"0.1\",\"1.0\",\"0.18\",\"hll\",\"32\",\"0.1\"]; ",
                        "include(\"lib/PowerFoam/examples/powerfoam_kh2d_compare_gate.jl\")")
    report_pf_kh = joinpath(@__DIR__, "out", "powerfoam_kh2d_compare_gate",
                            "N32_t0p1_drat1_hll", "README.md")
    run_gate!(runs, "2-D PowerFoam KH fixed+moving HLL", expr_pf_kh, report_pf_kh;
              execute = RUN_MODE == :extended)

    expr_noh2d = string("ARGS=[\"24\",\"0.2\",\"24\",\"hll\"]; ",
                        "include(\"lib/PowerFoam/examples/arepo_noh2d_gate.jl\")")
    report_noh2d = joinpath(@__DIR__, "out", "arepo_noh2d_gate",
                            "N24_t0p2_hll", "README.md")
    run_gate!(runs, "2-D PowerFoam Noh executable gate", expr_noh2d, report_noh2d)

    expr_wave2d = string("ARGS=[\"32\",\"8\",\"0.05\",\"hll\"]; ",
                         "include(\"lib/PowerFoam/examples/arepo_soundwave2d_gate.jl\")")
    report_wave2d = joinpath(@__DIR__, "out", "arepo_soundwave2d_gate",
                             "Nx32_Ny8_t0p05_hll", "README.md")
    run_gate!(runs, "2-D PowerFoam sound-wave executable gate", expr_wave2d, report_wave2d)

    expr_gresho2d = string("ARGS=[\"32\",\"32\",\"0.02\",\"hll\"]; ",
                           "include(\"lib/PowerFoam/examples/arepo_gresho2d_gate.jl\")")
    report_gresho2d = joinpath(@__DIR__, "out", "arepo_gresho2d_gate",
                               "Nx32_Ny32_t0p02_hll", "README.md")
    run_gate!(runs, "2-D PowerFoam Gresho executable gate", expr_gresho2d, report_gresho2d)
    return runs
end

function write_problem_csv(path)
    open(path, "w") do io
        println(io, "name,dim,class,arepo_example,powerfoam_surface,current_gate,next_gate,status")
        for p in PROBLEMS
            vals = (p.name, string(p.dim), p.class, p.arepo_example,
                    p.powerfoam_surface, p.current_gate, p.next_gate, p.status)
            println(io, join((csvquote(v) for v in vals), ","))
        end
    end
end

function write_gate_csv(path, runs)
    open(path, "w") do io
        println(io, "name,status,report,detail")
        for r in runs
            vals = (r.name, r.status, rel(r.report), r.detail)
            println(io, join((csvquote(v) for v in vals), ","))
        end
    end
end

function write_report(path, runs)
    open(path, "w") do io
        println(io, "# AREPO Standard Problem Validation Matrix")
        println(io)
        println(io, "This artifact tracks the path from component parity to a usable")
        println(io, "PowerFoam validation suite covering standard AREPO hydro problems.")
        println(io, "It separates certified executable gates from proxy-only or planned")
        println(io, "problem gates, so we do not overstate physics parity.")
        println(io)
        @printf(io, "- generated: %s\n", Dates.format(now(), dateformat"yyyy-mm-dd HH:MM:SS"))
        @printf(io, "- run mode: `%s`\n", RUN_MODE)
        @printf(io, "- AREPO library: `%s`\n", AREPO_LIB_PATH)
        println(io)

        println(io, "## Executable Gates")
        println(io)
        println(io, "| gate | status | report |")
        println(io, "| --- | --- | --- |")
        for r in runs
            report = isfile(r.report) ? rel(r.report) : rel(r.report)
            @printf(io, "| %s | %s | `%s` |\n", r.name, r.status, report)
        end
        println(io)

        println(io, "## Problem Coverage")
        println(io)
        println(io, "| problem | dim | class | current status | next gate |")
        println(io, "| --- | ---: | --- | --- | --- |")
        for p in PROBLEMS
            @printf(io, "| %s | %d | %s | %s | %s |\n",
                    p.name, p.dim, p.class, p.status, p.next_gate)
        end
        println(io)

        println(io, "## Execution Policy")
        println(io)
        println(io, "- `POWERFOAM_STANDARD_RUN=report` only refreshes this matrix from")
        println(io, "  existing artifacts.")
        println(io, "- `POWERFOAM_STANDARD_RUN=quick` reruns the currently certified small")
        println(io, "  3-D gates: HLL/LLF native-row replay, default decay scheduler, and")
        println(io, "  the opt-in N8 multirung scheduler fixture.")
        println(io, "- `POWERFOAM_STANDARD_RUN=extended` also reruns the stock AREPO")
        println(io, "  3-D Noh smoke/import gate and the original-code 2-D KH")
        println(io, "  HLL/HLL+PPM reference gate, then the PowerFoam fixed-periodic")
        println(io, "  and true-periodic moving-reconstructed 2-D KH comparison gate.")
        println(io, "- New standard problems should enter this matrix first as a row, then")
        println(io, "  graduate to an executable gate with a stable report path and a small")
        println(io, "  default resolution.")
        println(io)

        println(io, "## Next Implementation Target")
        println(io)
        println(io, "The Noh/KH field-difference tables now carry explicit tolerance")
        println(io, "status columns and opt-in strict parity switches, and the new")
        println(io, "2-D sound-wave rung provides a lightweight periodic smooth-flow")
        println(io, "calibration path. The next highest-value target is graduating")
        println(io, "the remaining 2-D Sedov/Gresho/contact standard problems from")
        println(io, "proxy or planned rows to executable field/profile gates.")
    end
end

function main()
    mkpath(OUTDIR)
    runs = quick_gate_runs()
    problem_csv = joinpath(OUTDIR, "problem_matrix.csv")
    gate_csv = joinpath(OUTDIR, "gate_runs.csv")
    report = joinpath(OUTDIR, "README.md")
    write_problem_csv(problem_csv)
    write_gate_csv(gate_csv, runs)
    write_report(report, runs)
    @printf("wrote %s\n", report)
    @printf("wrote %s\n", problem_csv)
    @printf("wrote %s\n", gate_csv)
    for r in runs
        @printf("%-34s %s\n", r.name, r.status)
    end
end

main()
