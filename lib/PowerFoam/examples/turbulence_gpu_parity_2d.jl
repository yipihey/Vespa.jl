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
using LinearAlgebra
using PowerFoam
using Printf
using Random
using Statistics

const GAMMA = 5 / 3
const BASE_OUTDIR = joinpath(@__DIR__, "out", "turbulence_gpu_parity_2d")
const AREPO_METRICS = "/Users/tabel/Projects/arepo/run/decay_runtime_hll_N12/analysis/metrics.csv"

function parse_arg(i, default, T)
    return length(ARGS) >= i ? parse(T, ARGS[i]) : default
end

const N = parse_arg(1, 12, Int)
const MACH = parse_arg(2, 5.0, Float64)
const TFINAL = parse_arg(3, 0.02, Float64)
const CFL = parse_arg(4, 0.18, Float64)
const RIEMANN = Symbol(length(ARGS) >= 5 ? ARGS[5] : "hll")
const BOUNDARY = Symbol(length(ARGS) >= 6 ? ARGS[6] : "clamp")
const ORDER = Symbol(length(ARGS) >= 7 ? ARGS[7] : "first")
ORDER in (:first, :reconstruct) || error("ORDER must be first or reconstruct")
const RUN_TAG = @sprintf("N%d_M%.3g_t%.3g_%s_%s_%s", N, MACH, TFINAL, RIEMANN,
                         BOUNDARY, ORDER)
const OUTDIR = joinpath(BASE_OUTDIR, replace(RUN_TAG, "." => "p"))

function maybe_metal_backend()
    _REQUEST_METAL || return nothing
    if _METAL_IMPORT_ERROR[] !== nothing
        @warn "Metal unavailable; running CPU-only parity check" err = _METAL_IMPORT_ERROR[]
        return nothing
    end
    return Metal.MetalBackend()
end

function grid_points(n)
    pts = Matrix{Float64}(undef, n * n, 2)
    q = 1
    for j in 1:n, i in 1:n
        pts[q, 1] = (i - 0.5) / n
        pts[q, 2] = (j - 0.5) / n
        q += 1
    end
    return pts
end

function solenoidal_modes_2d(points; mach, gamma, seed = 271, kmin = 2, kmax = 3)
    rng = MersenneTwister(seed)
    ncell = size(points, 1)
    vx = zeros(Float64, ncell)
    vy = zeros(Float64, ncell)
    for kx in -kmax:kmax, ky in -kmax:kmax
        kx == 0 && ky == 0 && continue
        kmag = hypot(kx, ky)
        (kmin <= kmag <= kmax) || continue
        phase = 2pi * rand(rng)
        amp = randn(rng) / (kmag * kmag)
        ex = -ky / kmag
        ey = kx / kmag
        @inbounds for i in 1:ncell
            arg = 2pi * (kx * points[i, 1] + ky * points[i, 2]) + phase
            s = amp * cos(arg)
            vx[i] += ex * s
            vy[i] += ey * s
        end
    end
    vx .-= mean(vx)
    vy .-= mean(vy)
    vrms = sqrt(mean(vx .* vx .+ vy .* vy))
    isfinite(vrms) && vrms > 0 || error("turbulence initializer produced zero velocity")
    cs = sqrt(gamma * (1 / gamma) / 1)
    scale = mach * cs / vrms
    return vx .* scale, vy .* scale
end

function make_initial_state(n; mach)
    pts = grid_points(n)
    mesh = power_diagram(PowerSites2D(pts))
    vx, vy = solenoidal_modes_2d(pts; mach, gamma = GAMMA)
    state = euler_state_2d(mesh; rho = 1.0, vx, vy, pressure = 1 / GAMMA,
                           gamma = GAMMA)
    return mesh, state
end

function diagnostics(label, state, geom, time, step)
    prim = conserved_to_primitive_2d(state; gamma = GAMMA)
    totals = total_conserved_2d(state, geom)
    vrms = sqrt(mean(prim.vx .* prim.vx .+ prim.vy .* prim.vy))
    density_rms = std(prim.rho) / mean(prim.rho)
    pmin = minimum(prim.pressure)
    return (; label, step, time, cells = length(prim.rho), mass = totals.mass,
            mx = totals.mx, my = totals.my, energy = totals.energy, vrms,
            density_rms, rho_min = minimum(prim.rho), rho_max = maximum(prim.rho),
            pmin)
end

function stable_dt(state, geom; cfl)
    area = Array(geom.volume)
    dx = sqrt(max(minimum(area), eps(Float64)))
    wavespeed = max_signal_speed_2d(state; gamma = GAMMA)
    return cfl * dx / wavespeed
end

function run_case(label, be; n, mach, tfinal, cfl, riemann, boundary, order)
    mesh, host_state = make_initial_state(n; mach)
    geom_host = arepo_mesh_arrays(mesh; T = Float64)
    state = to_backend(be, host_state; T = Float32)
    geom = to_backend(be, geom_host; T = Float32)
    rows = [diagnostics(label, state, geom, 0.0, 0)]
    t = 0.0
    step = 0
    while t < tfinal - 1e-12
        dt = min(stable_dt(state, geom; cfl), tfinal - t)
        prim = conserved_to_primitive_2d(state; gamma = GAMMA)
        vmesh = hcat(prim.vx, prim.vy)
        moved = if order == :reconstruct
            moving_mesh_reconstructed_step_2d!(state, mesh; dt, gamma = GAMMA,
                                               mesh_velocity = vmesh, riemann,
                                               backend = be, boundary)
        else
            moving_mesh_step_2d!(state, mesh; dt, gamma = GAMMA,
                                 mesh_velocity = vmesh, riemann,
                                 backend = be, boundary)
        end
        mesh = moved.mesh
        geom = moved.geom
        t += dt
        step += 1
        push!(rows, diagnostics(label, state, geom, t, step))
        any(!isfinite, Array(state.D)) && error("$label produced non-finite density")
        minimum(conserved_to_primitive_2d(state; gamma = GAMMA).pressure) > 0 ||
            error("$label produced non-positive pressure")
    end
    return rows, state, geom
end

function max_abs_diff(a, b)
    aa = Array(a)
    bb = Array(b)
    length(aa) == length(bb) || return NaN
    return maximum(abs.(aa .- bb))
end

function compare_states(cpu_state, gpu_state)
    return (; D = max_abs_diff(cpu_state.D, gpu_state.D),
            Mx = max_abs_diff(cpu_state.Mx, gpu_state.Mx),
            My = max_abs_diff(cpu_state.My, gpu_state.My),
            E = max_abs_diff(cpu_state.E, gpu_state.E))
end

function read_arepo_metrics(path)
    isfile(path) || return String[]
    return readlines(path)
end

function write_csv(path, rows)
    open(path, "w") do io
        println(io, "label,step,time,cells,mass,mx,my,energy,vrms,density_rms,rho_min,rho_max,pmin")
        for r in rows
            @printf(io, "%s,%d,%.9g,%d,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g\n",
                    r.label, r.step, r.time, r.cells, r.mass, r.mx, r.my, r.energy,
                    r.vrms, r.density_rms, r.rho_min, r.rho_max, r.pmin)
        end
    end
end

function write_report(path, cpu_rows, gpu_rows, diffs, arepo_lines, gpu_enabled)
    cpu_final = cpu_rows[end]
    gpu_final = isempty(gpu_rows) ? nothing : gpu_rows[end]
    open(path, "w") do io
        println(io, "# PowerFoam turbulence GPU parity, 2-D")
        println(io)
        println(io, "This is a small 2-D bounded Voronoi/ALE parity check for the current Julia rewrite.")
        println(io, "It is not yet an exact comparison to the original AREPO turbulence run, because the")
        println(io, "available AREPO turbulence setup is a 3-D periodic moving-Voronoi box while PowerFoam")
        println(io, "currently has the AREPO-shaped hydro kernels only for 2-D bounded meshes.")
        println(io)
        @printf(io, "- N: %d x %d cells\n", N, N)
        @printf(io, "- Mach: %.6g\n", MACH)
        @printf(io, "- t_final: %.6g\n", TFINAL)
        @printf(io, "- CFL: %.6g\n", CFL)
        @printf(io, "- Riemann solver: %s\n", RIEMANN)
        @printf(io, "- Bounded-mesh boundary mode: %s\n", BOUNDARY)
        @printf(io, "- Hydro order: %s\n", ORDER)
        println(io)
        println(io, "## Julia rewrite results")
        println(io)
        println(io, "| backend | steps | vrms | density_rms | rho_min | rho_max | pmin | mass | energy |")
        println(io, "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        @printf(io, "| CPU Float32 | %d | %.8g | %.8g | %.8g | %.8g | %.8g | %.8g | %.8g |\n",
                cpu_final.step, cpu_final.vrms, cpu_final.density_rms,
                cpu_final.rho_min, cpu_final.rho_max, cpu_final.pmin,
                cpu_final.mass, cpu_final.energy)
        if gpu_final !== nothing
            @printf(io, "| Metal Float32 | %d | %.8g | %.8g | %.8g | %.8g | %.8g | %.8g | %.8g |\n",
                    gpu_final.step, gpu_final.vrms, gpu_final.density_rms,
                    gpu_final.rho_min, gpu_final.rho_max, gpu_final.pmin,
                    gpu_final.mass, gpu_final.energy)
        else
            println(io, "| Metal Float32 | not run | | | | | | | |")
        end
        println(io)
        println(io, "## CPU/GPU field differences at final time")
        println(io)
        if gpu_enabled
            println(io, "| field | max_abs_diff |")
            println(io, "| --- | ---: |")
            @printf(io, "| D | %.9g |\n", diffs.D)
            @printf(io, "| Mx | %.9g |\n", diffs.Mx)
            @printf(io, "| My | %.9g |\n", diffs.My)
            @printf(io, "| E | %.9g |\n", diffs.E)
        else
            println(io, "Metal was not available in this Julia environment, so only the CPU leg ran.")
        end
        println(io)
        println(io, "## Existing original AREPO reference on disk")
        println(io)
        if isempty(arepo_lines)
            println(io, "No AREPO metrics file found at `$AREPO_METRICS`.")
        else
            println(io, "The nearest small AREPO metrics file is:")
            println(io)
            println(io, "```csv")
            foreach(line -> println(io, line), arepo_lines)
            println(io, "```")
            println(io)
            println(io, "That run is the 3-D Bauer/Springel-style Mach 0.3 decay case, so it is a")
            println(io, "sanity reference for expected diagnostic scale, not an exact parity target")
            println(io, "for the 2-D GPU rewrite.")
        end
        println(io)
        println(io, "## Next parity gate")
        println(io)
        println(io, "The next required implementation step for same-result AREPO parity is a 3-D")
        println(io, "periodic moving-mesh face table plus the same face-centric ALE HLL/LLF kernels")
        println(io, "over 3-D normals and conserved variables. The current harness is useful as a")
        println(io, "GPU correctness gate while that 3-D mesh path is added.")
    end
end

function main()
    mkpath(OUTDIR)
    cpu_be = KernelAbstractions.CPU()
    @printf("PowerFoam 2-D turbulence parity: N=%d Mach=%.3g tfinal=%.3g riemann=%s\n",
            N, MACH, TFINAL, RIEMANN)
    cpu_rows, cpu_state, cpu_geom = run_case("cpu-f32", cpu_be; n = N, mach = MACH,
                                             tfinal = TFINAL, cfl = CFL,
                                             riemann = RIEMANN,
                                             boundary = BOUNDARY,
                                             order = ORDER)
    gpu_be = maybe_metal_backend()
    gpu_rows = NamedTuple[]
    diffs = (; D = NaN, Mx = NaN, My = NaN, E = NaN)
    if gpu_be !== nothing
        gpu_rows, gpu_state, gpu_geom = run_case("metal-f32", gpu_be; n = N, mach = MACH,
                                                 tfinal = TFINAL, cfl = CFL,
                                                 riemann = RIEMANN,
                                                 boundary = BOUNDARY,
                                                 order = ORDER)
        diffs = compare_states(cpu_state, gpu_state)
        _ = gpu_geom
    end
    rows = vcat(cpu_rows, gpu_rows)
    write_csv(joinpath(OUTDIR, "metrics.csv"), rows)
    write_report(joinpath(OUTDIR, "README.md"), cpu_rows, gpu_rows, diffs,
                 read_arepo_metrics(AREPO_METRICS), gpu_be !== nothing)
    @printf("wrote %s\n", joinpath(OUTDIR, "metrics.csv"))
    @printf("wrote %s\n", joinpath(OUTDIR, "README.md"))
    final = cpu_rows[end]
    @printf("CPU final: steps=%d vrms=%.6g density_rms=%.6g rho=[%.6g, %.6g]\n",
            final.step, final.vrms, final.density_rms, final.rho_min, final.rho_max)
    if gpu_be !== nothing
        gfinal = gpu_rows[end]
        @printf("GPU final: steps=%d vrms=%.6g density_rms=%.6g rho=[%.6g, %.6g]\n",
                gfinal.step, gfinal.vrms, gfinal.density_rms, gfinal.rho_min, gfinal.rho_max)
        @printf("field max abs diffs: D=%.4g Mx=%.4g My=%.4g E=%.4g\n",
                diffs.D, diffs.Mx, diffs.My, diffs.E)
    end
end

main()
