using Printf

const SCRIPT = joinpath(@__DIR__, "native_moving_solver_matrix_3d.jl")
const OUTBASE = joinpath(@__DIR__, "out", "native_moving_solver_matrix_3d")
const GATE_OUTDIR = joinpath(@__DIR__, "out", "plane_cull_gate_3d")

parse_arg(i, default, T) = length(ARGS) >= i ? parse(T, ARGS[i]) : default

const N = parse_arg(1, 4, Int)
const DT = parse_arg(2, 0.001, Float64)
const STEPS = string(parse_arg(3, 2, Int))
const SOLVER = length(ARGS) >= 4 ? ARGS[4] : "hll"
const SEARCH_RADIUS = string(parse_arg(5, 1, Int))
const ORDER = length(ARGS) >= 6 ? ARGS[6] : "reconstruct"
const REBUILD = get(ENV, "POWERFOAM_REBUILD", "gpu_compact")
const BACKEND = get(ENV, "POWERFOAM_BACKEND", "metal")
const FIELD_ATOL = parse(Float64, get(ENV, "POWERFOAM_PLANE_CULL_ATOL", "0.0"))

function run_native(plane_cull::Bool)
    mkpath(GATE_OUTDIR)
    log_path = joinpath(GATE_OUTDIR, plane_cull ? "plane_cull_true.log" :
                                      "plane_cull_false.log")
    project = Base.active_project()
    project_arg = project === nothing ? "--project=@." : "--project=$(dirname(project))"
    cmd = `$(Base.julia_cmd()) $project_arg $SCRIPT $N $DT $STEPS $SOLVER $SEARCH_RADIUS $ORDER`
    env = copy(ENV)
    env["POWERFOAM_BACKEND"] = BACKEND
    env["POWERFOAM_REBUILD"] = REBUILD
    env["POWERFOAM_PLANE_CULL"] = plane_cull ? "true" : "false"
    env["POWERFOAM_CANDIDATE_TIER"] = get(ENV, "POWERFOAM_CANDIDATE_TIER", "full")
    env["POWERFOAM_PERF_WARMUP"] = get(ENV, "POWERFOAM_PERF_WARMUP", "false")
    env["POWERFOAM_DIAGNOSTICS"] = get(ENV, "POWERFOAM_DIAGNOSTICS", "final")
    env["POWERFOAM_WRITE_FINAL_FIELDS"] = "true"
    env["POWERFOAM_MESH_WORK_STATS"] = get(ENV, "POWERFOAM_MESH_WORK_STATS", "false")
    env["POWERFOAM_SYNC_TIMING"] = get(ENV, "POWERFOAM_SYNC_TIMING", "false")
    env["POWERFOAM_MESH_PROFILE"] = get(ENV, "POWERFOAM_MESH_PROFILE", "false")
    output = read(setenv(cmd, env), String)
    write(log_path, output)
    m = match(r"wrote (.*solver_summary\.csv)", output)
    m === nothing && error("native matrix did not report solver_summary.csv; see $log_path")
    return dirname(m.captures[1]), log_path
end

function read_final_state(path)
    isfile(path) || error("missing final state CSV: $path")
    lines = readlines(path)
    length(lines) > 1 || error("empty final state CSV: $path")
    data = Matrix{Float64}(undef, length(lines) - 1, 5)
    for (row, line) in enumerate(lines[2:end])
        fields = split(line, ',')
        length(fields) == 6 || error("malformed final state row in $path: $line")
        for j in 1:5
            data[row, j] = parse(Float64, fields[j + 1])
        end
    end
    return data
end

function compare_state_files(a_path, b_path)
    a = read_final_state(a_path)
    b = read_final_state(b_path)
    size(a) == size(b) || return (; D = Inf, Mx = Inf, My = Inf, Mz = Inf, E = Inf)
    names = (:D, :Mx, :My, :Mz, :E)
    values = map(1:5) do j
        maximum(abs.(a[:, j] .- b[:, j]))
    end
    return NamedTuple{names}(Tuple(values))
end

function write_gate_report(path, cull_dir, nocull_dir, cpu_diffs, metal_diffs)
    open(path, "w") do io
        println(io, "# 3-D Plane-Cull Equivalence Gate")
        println(io)
        @printf(io, "- N: `%d^3`\n", N)
        @printf(io, "- dt: `%.9g`\n", DT)
        @printf(io, "- steps: `%s`\n", STEPS)
        @printf(io, "- solver: `%s`\n", SOLVER)
        @printf(io, "- order: `%s`\n", ORDER)
        @printf(io, "- rebuild: `%s`\n", REBUILD)
        @printf(io, "- backend request: `%s`\n", BACKEND)
        @printf(io, "- tolerance: `%.9g`\n", FIELD_ATOL)
        println(io)
        println(io, "| backend | D | Mx | My | Mz | E |")
        println(io, "| --- | ---: | ---: | ---: | ---: | ---: |")
        @printf(io, "| CPU | %.9g | %.9g | %.9g | %.9g | %.9g |\n",
                cpu_diffs.D, cpu_diffs.Mx, cpu_diffs.My, cpu_diffs.Mz, cpu_diffs.E)
        @printf(io, "| Metal | %.9g | %.9g | %.9g | %.9g | %.9g |\n",
                metal_diffs.D, metal_diffs.Mx, metal_diffs.My, metal_diffs.Mz,
                metal_diffs.E)
        println(io)
        println(io, "Cull output: `$cull_dir`")
        println(io, "No-cull output: `$nocull_dir`")
    end
    return path
end

function main()
    cull_dir, _ = run_native(true)
    nocull_dir, _ = run_native(false)
    cpu_name = @sprintf("final_state_steps%s_%s_cpu.csv", STEPS, SOLVER)
    metal_name = @sprintf("final_state_steps%s_%s_metal.csv", STEPS, SOLVER)
    cpu_diffs = compare_state_files(joinpath(cull_dir, cpu_name),
                                    joinpath(nocull_dir, cpu_name))
    metal_diffs = compare_state_files(joinpath(cull_dir, metal_name),
                                      joinpath(nocull_dir, metal_name))
    report = write_gate_report(joinpath(GATE_OUTDIR, "README.md"), cull_dir,
                               nocull_dir, cpu_diffs, metal_diffs)
    maxdiff = maximum((cpu_diffs.D, cpu_diffs.Mx, cpu_diffs.My, cpu_diffs.Mz,
                       cpu_diffs.E, metal_diffs.D, metal_diffs.Mx,
                       metal_diffs.My, metal_diffs.Mz, metal_diffs.E))
    @printf("plane-cull gate maxdiff %.9g; wrote %s\n", maxdiff, report)
    maxdiff <= FIELD_ATOL ||
        error(@sprintf("plane-cull gate exceeded tolerance %.9g with maxdiff %.9g",
                       FIELD_ATOL, maxdiff))
end

main()
