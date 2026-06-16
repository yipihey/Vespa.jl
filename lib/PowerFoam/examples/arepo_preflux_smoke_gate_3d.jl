using Printf

const _REQUEST_METAL = lowercase(get(ENV, "POWERFOAM_BACKEND", "metal")) == "metal"
const _METAL_IMPORT_ERROR = Ref{Any}(nothing)
if _REQUEST_METAL
    try
        @eval using Metal
    catch err
        _METAL_IMPORT_ERROR[] = err
    end
end

using PowerFoam

parse_arg(i, default, T) = length(ARGS) >= i ? parse(T, ARGS[i]) : default

const PRE_OUTBASE = joinpath(@__DIR__, "out", "arepo_preflux_smoke_gate_3d")
const PRE_N = parse_arg(1, 4, Int)
const PRE_DT = parse_arg(2, 0.001, Float64)
const PRE_RIEMANN = Symbol(length(ARGS) >= 3 ? ARGS[3] : "hll")
const PRE_NSTEPS = parse_arg(4, 1, Int)
const PRE_RUN_TAG = PRE_NSTEPS == 1 ?
                    @sprintf("N%d_dt%.3g_%s", PRE_N, PRE_DT, PRE_RIEMANN) :
                    @sprintf("N%d_dt%.3g_n%d_%s", PRE_N, PRE_DT,
                             PRE_NSTEPS, PRE_RIEMANN)
const PRE_OUTDIR = joinpath(PRE_OUTBASE, replace(PRE_RUN_TAG, "." => "p"))
const PRE_AREPOLIB_IMPORT_ERROR = Ref{Any}(nothing)

try
    @eval import ArepoLib
catch err
    PRE_AREPOLIB_IMPORT_ERROR[] = err
end

function _pre_bridge_available()
    return isdefined(Main, :ArepoLib) &&
           isdefined(ArepoLib, :get_hydro_preflux_states_3d)
end

function _trace_bridge_available()
    return isdefined(Main, :ArepoLib) &&
           isdefined(ArepoLib, :get_hydro_face_traces_3d)
end

function _pre_arepo_libpath()
    if isdefined(Main, :ArepoLib)
        return ArepoLib.libpath()
    end
    return "unavailable: ArepoLib package is not in the active Julia environment"
end

function _arepo_solver_name(riemann::Symbol)
    riemann == :hll && return "HLL"
    riemann == :llf && return "LLF"
    riemann == :hllc && return "HLLC"
    riemann == :exact && return "Exact"
    error("unsupported AREPO Riemann solver for preflux smoke gate: $riemann")
end

function _set_param_line(text::String, key::AbstractString, value::AbstractString)
    line = @sprintf("%-50s %s", key, value)
    occursin(Regex("(?m)^$(key)\\s"), text) ?
    replace(text, Regex("(?m)^$(key)\\s+.*\$") => line) :
    string(text, endswith(text, "\n") ? "" : "\n", line, "\n")
end

function normalize_param_for_linked_arepo(text)
    lines = split(text, '\n'; keepempty = true)
    keep = String[]
    for line in lines
        if occursin(r"^SofteningComovingType[2-5]\s", line) ||
           occursin(r"^SofteningMaxPhysType[2-5]\s", line)
            continue
        end
        push!(keep, line)
    end
    text = join(keep, "\n")
    text = replace(text,
                   r"(?m)^SofteningTypeOfPartType1\s+.*$" => "SofteningTypeOfPartType1              1",
                   r"(?m)^SofteningTypeOfPartType2\s+.*$" => "SofteningTypeOfPartType2              1",
                   r"(?m)^SofteningTypeOfPartType3\s+.*$" => "SofteningTypeOfPartType3              1",
                   r"(?m)^SofteningTypeOfPartType4\s+.*$" => "SofteningTypeOfPartType4              1",
                   r"(?m)^SofteningTypeOfPartType5\s+.*$" => "SofteningTypeOfPartType5              1")
    if !occursin(r"(?m)^MinimumComovingHydroSoftening\s", text)
        text *= "\nMinimumComovingHydroSoftening         0.001\n"
    end
    if !occursin(r"(?m)^AdaptiveHydroSofteningSpacing\s", text)
        text *= "AdaptiveHydroSofteningSpacing         1.2\n"
    end
    return text
end

const _AREPO_DIR = "/Users/tabel/Projects/arepo"
const _EXAMPLE = joinpath(_AREPO_DIR, "examples", "bauer_springel_turbulence_3d")

function python_cmd()
    for exe in (get(ENV, "AREPO_PYTHON", ""), joinpath(_AREPO_DIR, ".venv", "bin", "python"),
                "python3", "python")
        isempty(exe) && continue
        try
            run(pipeline(Cmd([exe, "-c", "import h5py, numpy"]);
                         stdout = devnull, stderr = devnull))
            return exe
        catch
        end
    end
    error("no Python with h5py/numpy found; set AREPO_PYTHON")
end

function stage_arepo_case(n; riemann::Symbol = PRE_RIEMANN)
    isdir(_EXAMPLE) || error("AREPO turbulence example not found at $_EXAMPLE")
    dir = mktempdir()
    param = joinpath(dir, "param.txt")
    cp(joinpath(_EXAMPLE, "param_decay.txt"), param)
    text = read(param, String)
    text = replace(text,
                   r"(?m)^TimeOfFirstSnapshot\s+.*$" => "TimeOfFirstSnapshot                               2",
                   r"(?m)^TimeBetSnapshot\s+.*$" => "TimeBetSnapshot                                   2")
    text = normalize_param_for_linked_arepo(text)
    text = _set_param_line(text, "HydroRiemannSolver", _arepo_solver_name(riemann))
    write(param, text)
    mkpath(joinpath(dir, "output"))
    py = python_cmd()
    run(pipeline(`$py $(joinpath(_EXAMPLE, "create.py")) $dir unused $n 271`;
                 stdout = devnull))
    isfile(joinpath(dir, "IC.hdf5")) || error("AREPO create.py produced no IC.hdf5")
    return dir
end

function arepo_initial_export(dir)
    p = ArepoLib.precision_bytes()
    p.ndim == 3 || error("AREPO_LIB must point to a 3-D build; got ndim=$(p.ndim)")
    h = cd(() -> ArepoLib.init("param.txt"), dir)
    try
        return (; h, ng = ArepoLib.info(h).numgas, dir)
    catch
        ArepoLib.finalize(h)
        rethrow()
    end
end

function _snapshot_mass_sum(snapshot)
    mass = snapshot.conserved[:, 1]
    return sum(mass)
end

function _snapshot_unique_ids(snapshot)
    return length(unique(snapshot.ids)) == length(snapshot.ids)
end

function _trace_pass_indices(trace)
    return unique(Int.(trace.pass_index))
end

function _snapshot_pass_indices(snapshots)
    return [Int(s.pass_index) for s in snapshots]
end

function _write_preflux_report(path; status, step_status = nothing,
                               snapshots = nothing, trace = nothing,
                               trace_passes = nothing)
    open(path, "w") do io
        println(io, "# AREPO Preflux Smoke Gate")
        println(io)
        println(io, "This gate exercises `ArepoLib.get_hydro_preflux_states_3d` on")
        println(io, "a single N4 HLL AREPO step and checks the bridge data for basic")
        println(io, "sanity conditions.")
        println(io)
        @printf(io, "- AREPO library: `%s`\n", _pre_arepo_libpath())
        @printf(io, "- N: %d^3\n", PRE_N)
        @printf(io, "- Riemann solver: %s\n", PRE_RIEMANN)
        @printf(io, "- AREPO step status: %s\n", string(step_status))
        @printf(io, "- status: %s\n", status)
        println(io)
        if snapshots === nothing
            if PRE_AREPOLIB_IMPORT_ERROR[] !== nothing
                println(io, "The active Julia environment does not expose")
                println(io, "`ArepoLib`, so the preflux bridge could not be queried.")
            else
                println(io, "The AREPO bridge does not yet expose")
                println(io, "`get_hydro_preflux_states_3d`.")
            end
            return
        end
        @printf(io, "- snapshots: %d\n", length(snapshots))
        @printf(io, "- face-trace bridge available: %s\n", string(trace !== nothing))
        if trace_passes !== nothing
            @printf(io, "- trace pass indices: %s\n", join(trace_passes, ", "))
        end
        println(io)
        println(io, "| snapshot | pass | cells | ids unique | mass sum | volume sum |")
        println(io, "| ---: | ---: | ---: | ---: | ---: | ---: |")
        for (i, snap) in pairs(snapshots)
            @printf(io, "| %d | %d | %d | %s | %.12g | %.12g |\n",
                    i, snap.pass_index, length(snap.ids),
                    string(_snapshot_unique_ids(snap)),
                    _snapshot_mass_sum(snap), sum(snap.volume))
        end
    end
end

function main_preflux()
    mkpath(PRE_OUTDIR)
    report = joinpath(PRE_OUTDIR, "README.md")
    if !(_pre_bridge_available())
        _write_preflux_report(report; status = "skipped: missing AREPO preflux bridge")
        @printf("wrote %s\n", report)
        @printf("skipped: ArepoLib.get_hydro_preflux_states_3d is not available\n")
        return
    end
    dir = stage_arepo_case(PRE_N; riemann = PRE_RIEMANN)
    exported = arepo_initial_export(dir)
    try
        step_status = ArepoLib.run_step!(exported.h)
        snapshots = ArepoLib.get_hydro_preflux_states_3d(exported.h)
        trace = _trace_bridge_available() ? ArepoLib.get_hydro_face_traces_3d(exported.h) : nothing
        snapshot_passes = _snapshot_pass_indices(snapshots)
        trace_passes = trace === nothing ? nothing : _trace_pass_indices(trace)
        pass_match = trace_passes === nothing ? true : snapshot_passes == trace_passes
        snapshot_count_ok = !isempty(snapshots)
        ids_unique_ok = all(_snapshot_unique_ids(s) for s in snapshots)
        mass_ok = all(isfinite(_snapshot_mass_sum(s)) && _snapshot_mass_sum(s) > 0 for s in snapshots)
        volume_ok = all(isapprox(sum(s.volume), 1.0; atol = 1e-12, rtol = 1e-10) for s in snapshots)
        status = snapshot_count_ok && pass_match && ids_unique_ok && mass_ok && volume_ok ?
                 "passed" : "failed"
        _write_preflux_report(report; status, step_status, snapshots, trace, trace_passes)
        @printf("wrote %s\n", report)
        @printf("AREPO step status=%s\n", step_status)
        @printf("preflux %s: snapshots=%d pass_match=%s ids_unique=%s mass_ok=%s volume_ok=%s\n",
                status, length(snapshots), string(pass_match), string(ids_unique_ok),
                string(mass_ok), string(volume_ok))
        if trace_passes !== nothing
            @printf("trace passes=%s snapshot passes=%s\n",
                    join(trace_passes, ","), join(snapshot_passes, ","))
        end
    finally
        ArepoLib.finalize(exported.h)
    end
end

main_preflux()
