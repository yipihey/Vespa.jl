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
const BASE_OUTDIR = joinpath(@__DIR__, "out", "turbulence_gpu_parity_3d")
const AREPO_METRICS = "/Users/tabel/Projects/arepo/run/decay_runtime_hll_N12/analysis/metrics.csv"

parse_arg(i, default, T) = length(ARGS) >= i ? parse(T, ARGS[i]) : default

const N = parse_arg(1, 12, Int)
const MACH = parse_arg(2, 5.0, Float64)
const TFINAL = parse_arg(3, 0.02, Float64)
const CFL = parse_arg(4, 0.18, Float64)
const RIEMANN = Symbol(length(ARGS) >= 5 ? ARGS[5] : "hll")
const RUN_TAG = @sprintf("N%d_M%.3g_t%.3g_%s", N, MACH, TFINAL, RIEMANN)
const OUTDIR = joinpath(BASE_OUTDIR, replace(RUN_TAG, "." => "p"))

function maybe_metal_backend()
    _REQUEST_METAL || return nothing
    if _METAL_IMPORT_ERROR[] !== nothing
        @warn "Metal unavailable; running CPU-only parity check" err = _METAL_IMPORT_ERROR[]
        return nothing
    end
    return Metal.MetalBackend()
end

function grid_points_3d(n)
    pts = Matrix{Float64}(undef, n^3, 3)
    q = 1
    for k in 1:n, j in 1:n, i in 1:n
        pts[q, 1] = (i - 0.5) / n
        pts[q, 2] = (j - 0.5) / n
        pts[q, 3] = (k - 0.5) / n
        q += 1
    end
    return pts
end

function solenoidal_modes_3d(points; mach, gamma, seed = 271, kmin = 2, kmax = 3)
    rng = MersenneTwister(seed)
    ncell = size(points, 1)
    vx = zeros(Float64, ncell)
    vy = zeros(Float64, ncell)
    vz = zeros(Float64, ncell)
    for kx in -kmax:kmax, ky in -kmax:kmax, kz in -kmax:kmax
        kx == 0 && ky == 0 && kz == 0 && continue
        kmag = sqrt(kx * kx + ky * ky + kz * kz)
        (kmin <= kmag <= kmax) || continue
        phase = 2pi * rand(rng)
        ax, ay, az = randn(rng), randn(rng), randn(rng)
        dotak = ax * kx + ay * ky + az * kz
        ax -= dotak * kx / (kmag * kmag)
        ay -= dotak * ky / (kmag * kmag)
        az -= dotak * kz / (kmag * kmag)
        amp = randn(rng) / (kmag * kmag)
        @inbounds for i in 1:ncell
            arg = 2pi * (kx * points[i, 1] + ky * points[i, 2] +
                         kz * points[i, 3]) + phase
            s = amp * cos(arg)
            vx[i] += ax * s
            vy[i] += ay * s
            vz[i] += az * s
        end
    end
    vx .-= mean(vx)
    vy .-= mean(vy)
    vz .-= mean(vz)
    vrms = sqrt(mean(vx .* vx .+ vy .* vy .+ vz .* vz))
    isfinite(vrms) && vrms > 0 || error("turbulence initializer produced zero velocity")
    cs = sqrt(gamma * (1 / gamma) / 1)
    scale = mach * cs / vrms
    return vx .* scale, vy .* scale, vz .* scale
end

function make_initial_state(n; mach)
    geom = cartesian_periodic_mesh_arrays_3d(n; T = Float64)
    pts = grid_points_3d(n)
    vx, vy, vz = solenoidal_modes_3d(pts; mach, gamma = GAMMA)
    state = euler_state_3d(geom; rho = 1.0, vx, vy, vz, pressure = 1 / GAMMA,
                           gamma = GAMMA)
    return geom, state
end

function diagnostics(label, state, geom, time, step)
    prim = conserved_to_primitive_3d(state; gamma = GAMMA)
    totals = total_conserved_3d(state, geom)
    vrms = sqrt(mean(prim.vx .* prim.vx .+ prim.vy .* prim.vy .+ prim.vz .* prim.vz))
    density_rms = std(prim.rho) / mean(prim.rho)
    return (; label, step, time, cells = length(prim.rho), mass = totals.mass,
            mx = totals.mx, my = totals.my, mz = totals.mz, energy = totals.energy,
            vrms, density_rms, rho_min = minimum(prim.rho),
            rho_max = maximum(prim.rho), pmin = minimum(prim.pressure))
end

function stable_dt(state, geom; cfl)
    dx = cbrt(minimum(Array(geom.volume)))
    cfl * dx / max_signal_speed_3d(state; gamma = GAMMA)
end

function run_case(label, be; n, mach, tfinal, cfl, riemann)
    host_geom, host_state = make_initial_state(n; mach)
    geom = to_backend(be, host_geom; T = Float32)
    state = to_backend(be, host_state; T = Float32)
    rows = [diagnostics(label, state, geom, 0.0, 0)]
    t = 0.0
    step = 0
    while t < tfinal - 1e-12
        dt = min(stable_dt(state, geom; cfl), tfinal - t)
        finite_volume_step_3d!(state, geom; dt, gamma = GAMMA, riemann)
        t += dt
        step += 1
        push!(rows, diagnostics(label, state, geom, t, step))
        any(!isfinite, Array(state.D)) && error("$label produced non-finite density")
        minimum(conserved_to_primitive_3d(state; gamma = GAMMA).pressure) > 0 ||
            error("$label produced non-positive pressure")
    end
    return rows, state, geom
end

max_abs_diff(a, b) = maximum(abs.(Array(a) .- Array(b)))

function compare_states(cpu_state, gpu_state)
    return (; D = max_abs_diff(cpu_state.D, gpu_state.D),
            Mx = max_abs_diff(cpu_state.Mx, gpu_state.Mx),
            My = max_abs_diff(cpu_state.My, gpu_state.My),
            Mz = max_abs_diff(cpu_state.Mz, gpu_state.Mz),
            E = max_abs_diff(cpu_state.E, gpu_state.E))
end

read_arepo_metrics(path) = isfile(path) ? readlines(path) : String[]

function write_csv(path, rows)
    open(path, "w") do io
        println(io, "label,step,time,cells,mass,mx,my,mz,energy,vrms,density_rms,rho_min,rho_max,pmin")
        for r in rows
            @printf(io, "%s,%d,%.9g,%d,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g\n",
                    r.label, r.step, r.time, r.cells, r.mass, r.mx, r.my, r.mz,
                    r.energy, r.vrms, r.density_rms, r.rho_min, r.rho_max, r.pmin)
        end
    end
end

function write_report(path, cpu_rows, gpu_rows, diffs, arepo_lines, gpu_enabled)
    cpu_final = cpu_rows[end]
    gpu_final = isempty(gpu_rows) ? nothing : gpu_rows[end]
    open(path, "w") do io
        println(io, "# PowerFoam turbulence GPU parity, 3-D Cartesian")
        println(io)
        println(io, "This is the first 3-D periodic face-table hydro gate for the Julia rewrite.")
        println(io, "It uses a Cartesian periodic Voronoi-equivalent mesh, not the full moving")
        println(io, "AREPO Voronoi tessellation yet. The same `ArepoMeshArrays3D` layout is")
        println(io, "intended for AREPO's exported 3-D face rings.")
        println(io)
        @printf(io, "- N: %d^3 cells\n", N)
        @printf(io, "- Mach: %.6g\n", MACH)
        @printf(io, "- t_final: %.6g\n", TFINAL)
        @printf(io, "- CFL: %.6g\n", CFL)
        @printf(io, "- Riemann solver: %s\n", RIEMANN)
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
            @printf(io, "| Mz | %.9g |\n", diffs.Mz)
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
            println(io, "That is the moving 3-D AREPO HLL run. This Cartesian-gate result")
            println(io, "should not be interpreted as final same-result parity yet.")
        end
        println(io)
        println(io, "## Next parity gate")
        println(io)
        println(io, "Use `arepo_geometry_gate_3d.jl` to feed `ArepoLib.get_voronoi_3d`")
        println(io, "face rings into `arepo_voronoi_mesh_arrays_3d`, then add the")
        println(io, "remaining AREPO production-hydro pieces: reconstruction, predictor,")
        println(io, "timestep hierarchy, and mesh rebuild.")
    end
end

function main()
    mkpath(OUTDIR)
    cpu_be = KernelAbstractions.CPU()
    @printf("PowerFoam 3-D turbulence parity: N=%d Mach=%.3g tfinal=%.3g riemann=%s\n",
            N, MACH, TFINAL, RIEMANN)
    cpu_rows, cpu_state, cpu_geom = run_case("cpu-f32", cpu_be; n = N, mach = MACH,
                                             tfinal = TFINAL, cfl = CFL,
                                             riemann = RIEMANN)
    gpu_be = maybe_metal_backend()
    gpu_rows = NamedTuple[]
    diffs = (; D = NaN, Mx = NaN, My = NaN, Mz = NaN, E = NaN)
    if gpu_be !== nothing
        gpu_rows, gpu_state, gpu_geom = run_case("metal-f32", gpu_be; n = N, mach = MACH,
                                                 tfinal = TFINAL, cfl = CFL,
                                                 riemann = RIEMANN)
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
        @printf("field max abs diffs: D=%.4g Mx=%.4g My=%.4g Mz=%.4g E=%.4g\n",
                diffs.D, diffs.Mx, diffs.My, diffs.Mz, diffs.E)
    end
end

main()
