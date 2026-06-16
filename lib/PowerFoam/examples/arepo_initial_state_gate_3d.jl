const _REQUEST_METAL = lowercase(get(ENV, "POWERFOAM_BACKEND", "metal")) == "metal"
const _METAL_IMPORT_ERROR = Ref{Any}(nothing)
if _REQUEST_METAL
    try
        @eval using Metal
    catch err
        _METAL_IMPORT_ERROR[] = err
    end
end

using KernelAbstractions
using PowerFoam
using Printf
using Statistics
using ArepoLib

const GAMMA = 5 / 3
const AREPO_DIR = get(ENV, "AREPO_DIR", "/Users/tabel/Projects/arepo")
const EXAMPLE = joinpath(AREPO_DIR, "examples", "bauer_springel_turbulence_3d")
const OUTBASE = joinpath(@__DIR__, "out", "arepo_initial_state_gate_3d")

parse_arg(i, default, T) = length(ARGS) >= i ? parse(T, ARGS[i]) : default

const N = parse_arg(1, 4, Int)
const RTOL64 = parse_arg(2, 5e-13, Float64)
const ATOL64 = parse_arg(3, 5e-14, Float64)
const OUTDIR = joinpath(OUTBASE, @sprintf("N%d", N))

function maybe_metal_backend()
    _REQUEST_METAL || return nothing
    if _METAL_IMPORT_ERROR[] !== nothing
        @warn "Metal unavailable; running CPU-only initial-state gate" err = _METAL_IMPORT_ERROR[]
        return nothing
    end
    return Metal.MetalBackend()
end

function python_cmd()
    for exe in (get(ENV, "AREPO_PYTHON", ""), joinpath(AREPO_DIR, ".venv", "bin", "python"),
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

function stage_arepo_case(n)
    isdir(EXAMPLE) || error("AREPO turbulence example not found at $EXAMPLE")
    dir = mktempdir()
    param = joinpath(dir, "param.txt")
    cp(joinpath(EXAMPLE, "param_decay.txt"), param)
    text = read(param, String)
    text = replace(text,
                   r"(?m)^TimeOfFirstSnapshot\s+.*$" => "TimeOfFirstSnapshot                               2",
                   r"(?m)^TimeBetSnapshot\s+.*$" => "TimeBetSnapshot                                   2")
    text = normalize_param_for_linked_arepo(text)
    write(param, text)
    mkpath(joinpath(dir, "output"))
    py = python_cmd()
    run(pipeline(`$py $(joinpath(EXAMPLE, "create.py")) $dir unused $n 271`;
                 stdout = devnull))
    isfile(joinpath(dir, "IC.hdf5")) || error("AREPO create.py produced no IC.hdf5")
    return dir
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

function arepo_initial_export(dir)
    p = ArepoLib.precision_bytes()
    p.ndim == 3 || error("AREPO_LIB must point to a 3-D build; got ndim=$(p.ndim)")
    h = cd(() -> ArepoLib.init("param.txt"), dir)
    try
        ng = ArepoLib.info(h).numgas
        geo = ArepoLib.get_voronoi_3d(h)
        vol = ArepoLib.get_cell_field(h, :volume)
        rho = ArepoLib.get_cell_field(h, :rho)
        pressure = ArepoLib.get_cell_field(h, :pressure)
        center = ArepoLib.get_cell_field(h, :center)
        vel = ArepoLib.get_particle_field(h, :vel)[1:ng, :]
        return (; h, dir, ng, geo, vol, rho, pressure, center, vel,
                box = ArepoLib.box_size(h), lib = ArepoLib.libpath())
    catch
        ArepoLib.finalize(h)
        rethrow()
    end
end

function powerfoam_state(exported; T = Float64)
    geom = arepo_voronoi_mesh_arrays_3d(exported.geo, exported.vol;
                                        T, cell_velocity = exported.vel)
    state = euler_state_3d(geom; rho = exported.rho,
                           vx = exported.vel[:, 1],
                           vy = exported.vel[:, 2],
                           vz = exported.vel[:, 3],
                           pressure = exported.pressure,
                           gamma = GAMMA, T)
    return geom, state
end

maxabs(a, b) = maximum(abs.(Array(a) .- Array(b)))

function primitive_diffs(prim, exported)
    return (; rho = maxabs(prim.rho, exported.rho),
            vx = maxabs(prim.vx, exported.vel[:, 1]),
            vy = maxabs(prim.vy, exported.vel[:, 2]),
            vz = maxabs(prim.vz, exported.vel[:, 3]),
            pressure = maxabs(prim.pressure, exported.pressure))
end

function primitive_diffs_float32(prim, exported)
    return (; rho = maxabs(prim.rho, Float32.(exported.rho)),
            vx = maxabs(prim.vx, Float32.(exported.vel[:, 1])),
            vy = maxabs(prim.vy, Float32.(exported.vel[:, 2])),
            vz = maxabs(prim.vz, Float32.(exported.vel[:, 3])),
            pressure = maxabs(prim.pressure, Float32.(exported.pressure)))
end

function diagnostics(state, geom)
    prim = conserved_to_primitive_3d(state; gamma = GAMMA)
    totals = total_conserved_3d(state, geom)
    v2 = prim.vx .* prim.vx .+ prim.vy .* prim.vy .+ prim.vz .* prim.vz
    cs2 = GAMMA .* prim.pressure ./ prim.rho
    return (; totals.mass, totals.mx, totals.my, totals.mz, totals.energy,
            vrms = sqrt(mean(v2)), mach_rms = sqrt(mean(v2 ./ cs2)),
            density_rms = std(prim.rho) / mean(prim.rho),
            rho_min = minimum(prim.rho), rho_max = maximum(prim.rho),
            pmin = minimum(prim.pressure), pmax = maximum(prim.pressure))
end

function backend_roundtrip(exported, be)
    geom64, state64 = powerfoam_state(exported; T = Float64)
    geom = to_backend(be, geom64; T = Float32)
    state = to_backend(be, state64; T = Float32)
    work = primitive_work_3d(state)
    conserved_to_primitive_3d!(work, state; gamma = GAMMA)
    return primitive_to_arrays_3d(work), diagnostics(state, geom)
end

function compare_backend_primitives(a, b)
    return (; rho = maxabs(a.rho, b.rho),
            vx = maxabs(a.vx, b.vx),
            vy = maxabs(a.vy, b.vy),
            vz = maxabs(a.vz, b.vz),
            pressure = maxabs(a.pressure, b.pressure))
end

function print_diff_row(io, label, d)
    @printf(io, "| %s | %.12g | %.12g | %.12g | %.12g | %.12g |\n",
            label, d.rho, d.vx, d.vy, d.vz, d.pressure)
end

function print_diag(io, label, d)
    @printf(io, "| %s | %.12g | %.12g | %.12g | %.12g | %.12g | %.12g | %.12g | %.12g |\n",
            label, d.mass, d.mx, d.my, d.mz, d.energy, d.vrms, d.mach_rms, d.density_rms)
end

function write_report(path, exported, cpu64_diffs, cpu64_diag,
                      cpu32_diffs, cpu32_diag, metal32_diffs, metal32_diag,
                      cpu_metal_diffs)
    open(path, "w") do io
        println(io, "# AREPO 3-D Initial-State Gate")
        println(io)
        println(io, "This gate initializes the stock AREPO 3-D turbulence IC through")
        println(io, "`ArepoLib`, imports the live AREPO primitives and Voronoi volumes,")
        println(io, "constructs a PowerFoam `EulerState3D`, and checks primitive/conserved")
        println(io, "round-trip parity before any hydro update is attempted.")
        println(io)
        @printf(io, "- AREPO library: `%s`\n", exported.lib)
        @printf(io, "- staged case: `%s`\n", exported.dir)
        @printf(io, "- N: %d^3\n", N)
        @printf(io, "- cells: %d\n", exported.ng)
        @printf(io, "- faces: %d\n", length(exported.geo.nv))
        @printf(io, "- volume sum: %.12g\n", sum(exported.vol))
        println(io)
        println(io, "## Primitive Round Trip")
        println(io)
        println(io, "| path | rho | vx | vy | vz | pressure |")
        println(io, "| --- | ---: | ---: | ---: | ---: | ---: |")
        print_diff_row(io, "CPU Float64 vs AREPO", cpu64_diffs)
        print_diff_row(io, "KA CPU Float32 vs Float32(AREPO)", cpu32_diffs)
        if metal32_diffs !== nothing
            print_diff_row(io, "Metal Float32 vs Float32(AREPO)", metal32_diffs)
            print_diff_row(io, "Metal Float32 vs KA CPU Float32", cpu_metal_diffs)
        end
        println(io)
        println(io, "## Conserved Diagnostics")
        println(io)
        println(io, "| path | mass | mx | my | mz | energy | vrms | mach rms | density rms |")
        println(io, "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        print_diag(io, "CPU Float64", cpu64_diag)
        print_diag(io, "KA CPU Float32", cpu32_diag)
        if metal32_diag !== nothing
            print_diag(io, "Metal Float32", metal32_diag)
        end
    end
end

function within(d, atol, rtol)
    all(getfield(d, f) <= atol + rtol for f in (:rho, :vx, :vy, :vz, :pressure))
end

function main()
    mkpath(OUTDIR)
    dir = stage_arepo_case(N)
    exported = arepo_initial_export(dir)
    try
        geom64, state64 = powerfoam_state(exported; T = Float64)
        prim64 = conserved_to_primitive_3d(state64; gamma = GAMMA)
        cpu64_diffs = primitive_diffs(prim64, exported)
        cpu64_diag = diagnostics(state64, geom64)

        cpu32_prim, cpu32_diag = backend_roundtrip(exported, KernelAbstractions.CPU())
        cpu32_diffs = primitive_diffs_float32(cpu32_prim, exported)

        metal = maybe_metal_backend()
        metal32_prim = nothing
        metal32_diag = nothing
        metal32_diffs = nothing
        cpu_metal_diffs = nothing
        if metal !== nothing
            try
                metal32_prim, metal32_diag = backend_roundtrip(exported, metal)
                metal32_diffs = primitive_diffs_float32(metal32_prim, exported)
                cpu_metal_diffs = compare_backend_primitives(metal32_prim, cpu32_prim)
            catch err
                @warn "Metal initial-state round trip skipped" err
            end
        end

        report = joinpath(OUTDIR, "README.md")
        write_report(report, exported, cpu64_diffs, cpu64_diag,
                     cpu32_diffs, cpu32_diag, metal32_diffs, metal32_diag,
                     cpu_metal_diffs)
        @printf("wrote %s\n", report)
        @printf("CPU Float64 primitive maxdiffs: rho %.6g vx %.6g vy %.6g vz %.6g p %.6g\n",
                cpu64_diffs.rho, cpu64_diffs.vx, cpu64_diffs.vy,
                cpu64_diffs.vz, cpu64_diffs.pressure)
        if !within(cpu64_diffs, ATOL64, RTOL64)
            error("CPU Float64 initial-state parity exceeded tolerance; see $report")
        end
    finally
        ArepoLib.finalize(exported.h)
    end
end

main()
