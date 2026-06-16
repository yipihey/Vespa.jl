using Printf

struct GateRow
    lane::String
    gate::String
    driver::String
    level::String
    artifact::String
    note::String
end

const ROWS = GateRow[
    GateRow("unit", "PowerFoam tests", "lib/PowerFoam/test/runtests.jl",
            "R0", "Pkg.test summary",
            "fast blocking invariant and conservation lane"),
    GateRow("runtime", "Runtime hydro smoke", "lib/PowerFoam/examples/arepo_runtime_hydro_smoke.jl",
            "D1", "examples/out/arepo_runtime_hydro_smoke/.../README.md",
            "prebuilt geometry through ArepoProblemSpec and PowerFoam hydro"),
    GateRow("runtime", "Runtime moving-mesh smoke", "lib/PowerFoam/examples/arepo_runtime_moving_mesh_smoke.jl",
            "D1", "examples/out/arepo_runtime_moving_mesh_smoke/.../README.md",
            "periodic moving-mesh uniform-flow smoke through exported PowerFoam APIs"),
    GateRow("runtime", "Runtime param-spec smoke", "lib/PowerFoam/examples/arepo_runtime_param_spec_smoke.jl",
            "D1", "examples/out/arepo_runtime_param_spec_smoke/.../README.md",
            "package-facing param/config parse -> normalized runtime view -> ArepoProblemSpec smoke"),
    GateRow("runtime", "Runtime standard-problem dispatch", "lib/PowerFoam/examples/arepo_runtime_scaffold_smoke.jl",
            "D1", "stdout only",
            "bounded Noh2D and sound-wave 2-D dispatch through ArepoProblemSpec"),
    GateRow("bridge", "Initial-state parity", "lib/PowerFoam/examples/arepo_initial_state_gate_3d.jl",
            "Q1", "examples/out/arepo_initial_state_gate_3d/.../README.md",
            "CPU Float64 first, CPU/Metal Float32 second"),
    GateRow("bridge", "Geometry parity", "lib/PowerFoam/examples/arepo_geometry_gate_3d.jl",
            "Q1", "examples/out/arepo_geometry_gate_3d/.../README.md",
            "topology, volume, area, normals, centers"),
    GateRow("bridge", "Gradient parity", "lib/PowerFoam/examples/arepo_gradient_parity_3d.jl",
            "Q1", "examples/out/arepo_gradient_parity_3d/.../README.md",
            "live AREPO gradient comparison"),
    GateRow("bridge", "Mesh-velocity parity", "lib/PowerFoam/examples/arepo_mesh_velocity_gate_3d.jl",
            "D1", "examples/out/arepo_mesh_velocity_gate_3d/.../README.md",
            "VelVertex reconstruction against AREPO"),
    GateRow("bridge", "Scheduler parity", "lib/PowerFoam/examples/arepo_hierarchy_gate_3d.jl",
            "Q1", "examples/out/arepo_hierarchy_gate_3d/.../README.md",
            "controlled fixture is close to blocker quality"),
    GateRow("bridge", "Face-trace parity", "lib/PowerFoam/examples/arepo_face_trace_gate_3d.jl",
            "Q1", "examples/out/arepo_face_trace_gate_3d/.../README.md",
            "all active traced rows"),
    GateRow("bridge", "Trace replay parity", "lib/PowerFoam/examples/arepo_trace_replay_gate_3d.jl",
            "Q1", "examples/out/arepo_trace_replay_gate_3d/.../README.md",
            "conserved update replay against AREPO"),
    GateRow("bridge", "Native rebuild parity", "lib/PowerFoam/examples/arepo_native_rebuild_trace_gate_3d.jl",
            "D1", "examples/out/arepo_native_rebuild_trace_gate_3d/.../README.md",
            "diagnostic until native pass sequence is closed"),
    GateRow("bridge", "Tessellator matrix", "lib/PowerFoam/examples/arepo_tessellator_rebuild_gate_matrix.jl",
            "D1", "stdout only",
            "planning/report wrapper over native rebuild runs; topology-only mesh prototype is not production hydro CSR yet"),
    GateRow("bridge", "Tessellator compact parity", "lib/PowerFoam/test/runtests.jl",
            "R0", "Pkg.test summary",
            "N4/N8-like reciprocal-canonical compact-face symmetry with positive placeholder and injected face geometry"),
    GateRow("bridge", "One-step gap diagnostic", "lib/PowerFoam/examples/arepo_one_step_gap_3d.jl",
            "D1", "examples/out/arepo_one_step_gap_3d/.../README.md",
            "native-piece gap finder until the pass sequence closes"),
    GateRow("bridge", "Preflux smoke", "lib/PowerFoam/examples/arepo_preflux_smoke_gate_3d.jl",
            "D1", "examples/out/arepo_preflux_smoke_gate_3d/.../README.md",
            "basic availability of pre-flux snapshots and bridge fields"),
    GateRow("hydro-standard", "Noh 3-D", "lib/PowerFoam/examples/arepo_noh3d_smoke_gate.jl",
            "Q0", "examples/out/arepo_noh3d_smoke_gate/.../README.md",
            "best current problem-level promotion target"),
    GateRow("hydro-standard", "KH 2-D AREPO refs", "lib/PowerFoam/examples/arepo_kh2d_original_gate.jl",
            "D1", "examples/out/arepo_kh2d_original_gate/.../README.md",
            "reference producer, not final blocker"),
    GateRow("hydro-standard", "KH 2-D compare", "lib/PowerFoam/examples/powerfoam_kh2d_compare_gate.jl",
            "Q0", "examples/out/powerfoam_kh2d_compare_gate/.../README.md",
            "promote after moving rung is beyond smoke scale"),
    GateRow("hydro-standard", "Noh 2-D proxy readiness", "lib/PowerFoam/examples/arepo_noh2d_proxy_gate.jl",
            "D1", "examples/out/arepo_noh2d_proxy_gate/.../README.md",
            "executable shell for existing proxy artifacts, not final physics parity"),
    GateRow("hydro-standard", "Noh 2-D executable gate", "lib/PowerFoam/examples/arepo_noh2d_gate.jl",
            "D1", "examples/out/arepo_noh2d_gate/N24_t0p2_hll/README.md",
            "first executable bounded PowerFoam Noh2D rung; calibration pending"),
    GateRow("hydro-standard", "Sound-wave 2-D executable gate", "lib/PowerFoam/examples/arepo_soundwave2d_gate.jl",
            "D1", "examples/out/arepo_soundwave2d_gate/Nx32_Ny8_t0p05_hll/README.md",
            "periodic smooth acoustic wave with amplitude and phase diagnostics"),
    GateRow("hydro-standard", "Gresho 2-D executable gate", "lib/PowerFoam/examples/arepo_gresho2d_gate.jl",
            "D1", "examples/out/arepo_gresho2d_gate/Nx32_Ny32_t0p02_hll/README.md",
            "periodic vortex balance with rotational-profile diagnostics; calibration pending"),
    GateRow("hydro-standard", "Noh 2-D proxy", "lib/PowerFoam/examples/arepo_noh_proxy/",
            "D1", "examples/arepo_noh_proxy/results/README.md",
            "convert proxy into executable gate"),
    GateRow("hydro-standard", "Sedov 2-D proxy", "lib/PowerFoam/examples/arepo_sedov_proxy/",
            "D1", "examples/arepo_sedov_proxy/",
            "convert proxy into executable gate"),
    GateRow("backend", "3-D CPU vs Metal turbulence", "lib/PowerFoam/examples/turbulence_gpu_parity_3d.jl",
            "D1", "examples/out/turbulence_gpu_parity_3d/.../README.md",
            "backend parity only after CPU physics is stable"),
    GateRow("backend", "2-D CPU vs Metal turbulence", "lib/PowerFoam/examples/turbulence_gpu_parity_2d.jl",
            "D1", "examples/out/turbulence_gpu_parity_2d/.../README.md",
            "non-blocking today"),
    GateRow("backend", "Tessellator backend probe", "lib/PowerFoam/examples/tessellator_backend_parity_probe.jl",
            "D1", "examples/out/tessellator_backend_parity_probe/.../README.md",
            "CPU primitive probe; Metal is optional and skipped if unavailable"),
    GateRow("gravity", "Direct gravity smoke", "lib/PowerFoam/examples/arepo_gravity_direct_smoke.jl",
            "D1", "examples/out/arepo_gravity_direct_smoke/.../README.md",
            "tiny-N direct-force oracle and action-reaction check"),
    GateRow("gravity", "PM direct convention smoke", "lib/PowerFoam/examples/arepo_pm_direct_convention_smoke.jl",
            "D1", "stdout only",
            "finite periodic image-sum diagnostic for PM comparison convention"),
    GateRow("gravity", "PM gravity numeric preflight", "lib/PowerFoam/examples/arepo_pm_gravity_gate_skeleton.jl",
            "D1", "examples/out/arepo_pm_gravity_gate_skeleton/.../README.md",
            "numeric periodic image-sum fixture plus monorepo PoissonKernels PM chain"),
    GateRow("cosmology", "Gravity-only box reference", "examples/cosmo_box_gravity_only_3d/check.py",
            "D0", "n/a",
            "AREPO reference observable only; no PowerFoam gate yet"),
    GateRow("cosmology", "Gravity-only zoom reference", "examples/cosmo_zoom_gravity_only_3d/check.py",
            "D0", "n/a",
            "AREPO reference observable only; no PowerFoam gate yet"),
    GateRow("cosmology", "Star-formation box reference", "examples/cosmo_box_star_formation_3d/check.py",
            "D0", "n/a",
            "AREPO reference observable only; no PowerFoam gate yet"),
    GateRow("io", "IO runtime surface smoke", "lib/PowerFoam/examples/arepo_io_runtime_surface_smoke.jl",
            "D1", "examples/out/arepo_io_runtime_surface_smoke/.../README.md",
            "planned package-level IO symbols and missing module check"),
    GateRow("io", "Snapshot IO runtime smoke", "lib/PowerFoam/examples/arepo_snapshot_io_smoke.jl",
            "D1", "examples/out/arepo_snapshot_io_smoke/.../rows.csv",
            "typed gas snapshot plus HDF5 write/read round trip when dependency is available"),
    GateRow("io", "Parameter parser unit surface", "lib/PowerFoam/src/arepo_io_parameters.jl",
            "D1", "Pkg.test summary",
            "dependency-free param/config parser covered by unit tests"),
    GateRow("io", "Parameter parser smoke", "lib/PowerFoam/examples/arepo_parameter_parser_smoke.jl",
            "D1", "examples/out/arepo_parameter_parser_smoke/.../README.md",
            "package-facing param/config parser example with artifact output"),
    GateRow("perf", "Standard-problem rollup", "lib/PowerFoam/examples/arepo_standard_problem_matrix.jl",
            "D1", "examples/out/arepo_standard_problem_matrix/.../README.md",
            "summary/dashboard, not a blocker"),
]

const LEVEL_ORDER = Dict("D0" => 0, "D1" => 1, "Q0" => 2, "Q1" => 3, "R0" => 4, "R1" => 5)
const DEFAULT_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))

struct GateStatus
    runnable::Bool
    artifact_ready::Bool
    driver_kind::String
    evidence_status::String
end

function parse_args(args)
    summary_only = false
    table_only = false
    root = DEFAULT_ROOT
    observed_pass = String[]
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--summary-only"
            summary_only = true
        elseif arg == "--table-only"
            table_only = true
        elseif startswith(arg, "--observed-pass=")
            push!(observed_pass, normpath(arg[length("--observed-pass=")+1:end]))
        elseif arg == "--observed-pass"
            i += 1
            i > length(args) && error("--observed-pass requires a driver path")
            push!(observed_pass, normpath(args[i]))
        elseif startswith(arg, "--root=")
            root = normpath(arg[length("--root=")+1:end])
        elseif arg == "--root"
            i += 1
            i > length(args) && error("--root requires a path")
            root = normpath(args[i])
        else
            error("unknown argument: $arg")
        end
        i += 1
    end
    summary_only && table_only && error("choose at most one of --summary-only and --table-only")
    return (; summary_only, table_only, root, observed_pass)
end

driver_path(root, row::GateRow) = normpath(joinpath(root, row.driver))

function artifact_anchor(root, row::GateRow)
    cleaned = replace(row.artifact, r"/\.\.\..*$" => "")
    cleaned = replace(cleaned, r"/+$" => "")
    if isempty(cleaned) || cleaned == "Pkg.test summary" || cleaned == "stdout only" || cleaned == "n/a"
        return nothing
    elseif startswith(cleaned, "lib/")
        return normpath(joinpath(root, cleaned))
    else
        return normpath(joinpath(root, "lib/PowerFoam", cleaned))
    end
end

function gate_status(root, row::GateRow, observed_pass)
    path = driver_path(root, row)
    kind = isfile(path) ? "file" : isdir(path) ? "directory" : "missing"
    runnable = kind == "file"
    artifact = artifact_anchor(root, row)
    artifact_ready = artifact !== nothing && (isfile(artifact) || isdir(artifact))
    observed = path in observed_pass
    evidence_status = if observed && artifact_ready
        "last_observed_pass"
    elseif runnable
        "exists"
    else
        "not_run"
    end
    return GateStatus(runnable, artifact_ready, kind, evidence_status)
end

status_label(status::GateStatus) = status.runnable ? "runnable" : "planned"
artifact_label(status::GateStatus) = status.artifact_ready ? "present" : "pending"

function print_summary(rows, statuses)
    println("## Status Summary")
    println()
    total = length(rows)
    runnable = count(s -> s.runnable, statuses)
    planned = total - runnable
    ready = count(s -> s.artifact_ready, statuses)
    pending = total - ready
    observed_pass = count(s -> s.evidence_status == "last_observed_pass", statuses)
    exists_only = count(s -> s.evidence_status == "exists", statuses)
    not_run = count(s -> s.evidence_status == "not_run", statuses)
    println("- total gates: $total")
    println("- runnable now: $runnable")
    println("- planned only: $planned")
    println("- artifact anchors present: $ready")
    println("- artifact anchors pending: $pending")
    println("- evidence `last_observed_pass`: $observed_pass")
    println("- evidence `exists`: $exists_only")
    println("- evidence `not_run`: $not_run")
    println()

    print_group_summary("By level", [row.level for row in rows], statuses)
    println()
    print_group_summary("By lane", [row.lane for row in rows], statuses)
end

function print_group_summary(title, group_keys, statuses)
    println("### $title")
    println()
    println("| group | runnable | planned | artifact-ready | observed-pass | exists-only | not-run | total |")
    println("| --- | --- | --- | --- | --- | --- | --- | --- |")
    groups = Dict{String, NTuple{7, Int}}()
    for (key, status) in zip(group_keys, statuses)
        run = status.runnable ? 1 : 0
        plan = status.runnable ? 0 : 1
        art = status.artifact_ready ? 1 : 0
        obs = status.evidence_status == "last_observed_pass" ? 1 : 0
        exists = status.evidence_status == "exists" ? 1 : 0
        notrun = status.evidence_status == "not_run" ? 1 : 0
        total = 1
        prev = get(groups, key, (0, 0, 0, 0, 0, 0, 0))
        groups[key] = (prev[1] + run, prev[2] + plan, prev[3] + art,
                       prev[4] + obs, prev[5] + exists, prev[6] + notrun, prev[7] + total)
    end
    ordered = sort!(collect(keys(groups)); by = key -> group_sort_key(title, key))
    for key in ordered
        run, plan, art, obs, exists, notrun, total = groups[key]
        @printf("| %s | %d | %d | %d | %d | %d | %d | %d |\n",
                key, run, plan, art, obs, exists, notrun, total)
    end
end

function group_sort_key(title, key)
    if title == "By level"
        return (get(LEVEL_ORDER, key, typemax(Int)), key)
    end
    return key
end

function print_table(rows, statuses)
    println("## Gate Table")
    println()
    println("| lane | gate | level | status | evidence | artifact-ready | driver-kind | driver | note |")
    println("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for (row, status) in zip(rows, statuses)
        @printf("| %s | %s | %s | %s | %s | %s | %s | `%s` | %s |\n",
                row.lane, row.gate, row.level, status_label(status), status.evidence_status,
                artifact_label(status), status.driver_kind, row.driver, row.note)
    end
end

function normalize_observed_pass(root, observed_pass)
    normalized = String[]
    for item in observed_pass
        candidate = startswith(item, root) ? normpath(item) : normpath(joinpath(root, item))
        push!(normalized, candidate)
    end
    return normalized
end

function main(args=ARGS)
    opts = parse_args(args)
    observed_pass = normalize_observed_pass(opts.root, opts.observed_pass)
    statuses = [gate_status(opts.root, row, observed_pass) for row in ROWS]

    println("# AREPO Rewrite Verification Gate Table")
    println()
    println("_Root: `$(opts.root)`_")
    println()
    println("_Status is derived from driver-path existence: existing `.jl` files are `runnable`; directories or missing paths stay `planned`._")
    println()
    println("_Evidence is conservative: `last_observed_pass` is only assigned to drivers named via `--observed-pass` and only when their artifact anchor exists. Other runnable files are `exists`; planned/non-file entries stay `not_run`._")
    println()

    opts.table_only || print_summary(ROWS, statuses)
    if !opts.summary_only && !opts.table_only
        println()
    end
    opts.summary_only || print_table(ROWS, statuses)
end

main()
