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
const AREPO_DIR = "/Users/tabel/Projects/arepo"
const EXAMPLE = joinpath(AREPO_DIR, "examples", "bauer_springel_turbulence_3d")
const OUTBASE = joinpath(@__DIR__, "out", "arepo_gradient_parity_3d")

parse_arg(i, default, T) = length(ARGS) >= i ? parse(T, ARGS[i]) : default

const N = parse_arg(1, 8, Int)
const RTOL = parse_arg(2, 5e-10, Float64)
const ATOL = parse_arg(3, 5e-11, Float64)
const OUTDIR = joinpath(OUTBASE, @sprintf("N%d", N))

function maybe_metal_backend()
    _REQUEST_METAL || return nothing
    if _METAL_IMPORT_ERROR[] !== nothing
        @warn "Metal unavailable; running CPU-only gradient parity" err = _METAL_IMPORT_ERROR[]
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
    cp(joinpath(EXAMPLE, "param_decay.txt"), joinpath(dir, "param.txt"))
    param = joinpath(dir, "param.txt")
    write(param, normalize_param_for_linked_arepo(read(param, String)))
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

maxdiff(a, b) = maximum(abs.(Array(a) .- Array(b)))
maxreldiff(a, b) = maximum(abs.(Array(a) .- Array(b)) ./ max.(abs.(Array(a)), 1.0))

function cpu_gradients(exported)
    conn = gradient_connections_3d(exported.conn; T = Float64)
    out = hydro_gradient_work_3d(exported.rho)
    calculate_gradients_3d!(out, conn, exported.rho,
                            exported.vel[:, 1], exported.vel[:, 2], exported.vel[:, 3],
                            exported.pressure, exported.center;
                            box_size = exported.box, gamma = GAMMA,
                            sound_speed = exported.csnd)
    return hydro_gradients_to_arrays(out)
end

function metal_gradients(exported)
    be = maybe_metal_backend()
    be === nothing && return nothing
    conn = gradient_connections_3d(exported.conn; T = Float64)
    scratch = PowerFoam._backend_zeros(be, Float32, exported.ng)
    out = hydro_gradient_work_3d(scratch)
    calculate_gradients_3d!(out, conn, exported.rho,
                            @view(exported.vel[:, 1]), @view(exported.vel[:, 2]),
                            @view(exported.vel[:, 3]), exported.pressure, exported.center;
                            box_size = exported.box, gamma = GAMMA,
                            sound_speed = exported.csnd)
    return hydro_gradients_to_arrays(out)
end

function arepo_initial_export(dir)
    p = ArepoLib.precision_bytes()
    p.ndim == 3 || error("AREPO_LIB must point to a 3-D build; got ndim=$(p.ndim)")
    h = cd(() -> ArepoLib.init("param.txt"), dir)
    try
        ng = ArepoLib.info(h).numgas
        return (; h, dir, ng,
                rho = ArepoLib.get_cell_field(h, :rho),
                pressure = ArepoLib.get_cell_field(h, :pressure),
                csnd = ArepoLib.get_cell_field(h, :csnd),
                center = ArepoLib.get_cell_field(h, :center),
                surfacearea = ArepoLib.get_cell_field(h, :surfacearea),
                vel = ArepoLib.get_particle_field(h, :vel)[1:ng, :],
                cgrad = ArepoLib.get_hydro_gradients(h),
                conn = ArepoLib.get_gradient_connections_3d(h),
                box = ArepoLib.box_size(h))
    catch
        ArepoLib.finalize(h)
        rethrow()
    end
end

function component_metrics(cgrad, jgrad)
    jvel_t = permutedims(jgrad.dvel, (1, 3, 2))
    return (
        drho_abs = maxdiff(cgrad.drho, jgrad.drho),
        drho_rel = maxreldiff(cgrad.drho, jgrad.drho),
        drho_cmax = maximum(abs.(cgrad.drho)),
        drho_jmax = maximum(abs.(jgrad.drho)),
        dvel_abs = maxdiff(cgrad.dvel, jgrad.dvel),
        dvel_rel = maxreldiff(cgrad.dvel, jgrad.dvel),
        dvel_transpose_abs = maxdiff(cgrad.dvel, jvel_t),
        dvel_cmax = maximum(abs.(cgrad.dvel)),
        dvel_jmax = maximum(abs.(jgrad.dvel)),
        dpress_abs = maxdiff(cgrad.dpress, jgrad.dpress),
        dpress_rel = maxreldiff(cgrad.dpress, jgrad.dpress),
        dpress_cmax = maximum(abs.(cgrad.dpress)),
        dpress_jmax = maximum(abs.(jgrad.dpress)),
    )
end

function write_report(path, exported, cpu_metrics, gpu_metrics)
    open(path, "w") do io
        println(io, "# AREPO 3-D gradient parity gate")
        println(io)
        println(io, "This initializes the stock AREPO 3-D turbulence setup through")
        println(io, "`ArepoLib`, exports the exact `calculate_gradients()` connection")
        println(io, "rows from the C mesh, and recomputes the hydro gradients with a")
        println(io, "KernelAbstractions Julia kernel.")
        println(io)
        @printf(io, "- AREPO library: `%s`\n", ArepoLib.libpath())
        @printf(io, "- N: %d^3\n", N)
        @printf(io, "- cells: %d\n", exported.ng)
        @printf(io, "- accepted gradient connections: %d\n", length(exported.conn.area))
        @printf(io, "- mean accepted connections/cell: %.6g\n",
                length(exported.conn.area) / exported.ng)
        @printf(io, "- surface area min/max: %.8g / %.8g\n",
                minimum(exported.surfacearea), maximum(exported.surfacearea))
        println(io)
        println(io, "## C vs Julia KA CPU")
        println(io)
        println(io, "| field | max abs diff | max relative diff | max abs C | max abs KA |")
        println(io, "| --- | ---: | ---: | ---: | ---: |")
        @printf(io, "| density gradient | %.12g | %.12g | %.12g | %.12g |\n",
                cpu_metrics.drho_abs, cpu_metrics.drho_rel, cpu_metrics.drho_cmax, cpu_metrics.drho_jmax)
        @printf(io, "| velocity gradient | %.12g | %.12g | %.12g | %.12g |\n",
                cpu_metrics.dvel_abs, cpu_metrics.dvel_rel, cpu_metrics.dvel_cmax, cpu_metrics.dvel_jmax)
        @printf(io, "\nVelocity tensor transpose check: max abs diff %.12g\n", cpu_metrics.dvel_transpose_abs)
        @printf(io, "| pressure gradient | %.12g | %.12g | %.12g | %.12g |\n",
                cpu_metrics.dpress_abs, cpu_metrics.dpress_rel, cpu_metrics.dpress_cmax, cpu_metrics.dpress_jmax)
        if gpu_metrics !== nothing
            println(io)
            println(io, "## C vs Julia KA Metal")
            println(io)
            println(io, "| field | max abs diff | max relative diff | max abs C | max abs KA |")
            println(io, "| --- | ---: | ---: | ---: | ---: |")
            @printf(io, "| density gradient | %.12g | %.12g | %.12g | %.12g |\n",
                    gpu_metrics.drho_abs, gpu_metrics.drho_rel, gpu_metrics.drho_cmax, gpu_metrics.drho_jmax)
            @printf(io, "| velocity gradient | %.12g | %.12g | %.12g | %.12g |\n",
                    gpu_metrics.dvel_abs, gpu_metrics.dvel_rel, gpu_metrics.dvel_cmax, gpu_metrics.dvel_jmax)
            @printf(io, "\nVelocity tensor transpose check: max abs diff %.12g\n", gpu_metrics.dvel_transpose_abs)
            @printf(io, "| pressure gradient | %.12g | %.12g | %.12g | %.12g |\n",
                    gpu_metrics.dpress_abs, gpu_metrics.dpress_rel, gpu_metrics.dpress_cmax, gpu_metrics.dpress_jmax)
        end
        println(io)
        println(io, "The Julia kernel uses AREPO's native connection rows, face areas,")
        println(io, "face centers, mirrored neighbor centers, neighbor primitives, and")
        println(io, "the same limiter face walk exported from the live AREPO mesh.")
        println(io, "The least-squares solve follows AREPO's diagonal-pivot Gaussian")
        println(io, "elimination order so this is a production-gradient parity gate,")
        println(io, "not a simplified proxy.")
    end
end

function main()
    mkpath(OUTDIR)
    dir = stage_arepo_case(N)
    exported = arepo_initial_export(dir)
    try
        jcpu = cpu_gradients(exported)
        cpu_metrics = component_metrics(exported.cgrad, jcpu)
        jgpu = metal_gradients(exported)
        gpu_metrics = jgpu === nothing ? nothing : component_metrics(exported.cgrad, jgpu)
        write_report(joinpath(OUTDIR, "README.md"), exported, cpu_metrics, gpu_metrics)
        @printf("wrote %s\n", joinpath(OUTDIR, "README.md"))
        @printf("cells=%d connections=%d\n", exported.ng, length(exported.conn.area))
        @printf("CPU C-vs-KA: drho %.6g dvel %.6g dpress %.6g\n",
                cpu_metrics.drho_abs, cpu_metrics.dvel_abs, cpu_metrics.dpress_abs)
        if cpu_metrics.drho_abs > ATOL + RTOL * max(1.0, maximum(abs.(exported.cgrad.drho))) ||
           cpu_metrics.dvel_abs > ATOL + RTOL * max(1.0, maximum(abs.(exported.cgrad.dvel))) ||
           cpu_metrics.dpress_abs > ATOL + RTOL * max(1.0, maximum(abs.(exported.cgrad.dpress)))
            error("CPU gradient parity exceeded tolerance; see $(joinpath(OUTDIR, "README.md"))")
        end
    finally
        ArepoLib.finalize(exported.h)
    end
end

main()
