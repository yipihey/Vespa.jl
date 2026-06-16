using Dates
using PowerFoam
using Printf

const GAMMA = 5 / 3
const N = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 3
const DT = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 1.0e-2
const OUTBASE = joinpath(@__DIR__, "out", "arepo_runtime_moving_mesh_smoke")
const RUN_TAG = Dates.format(now(), "yyyymmdd_HHMMSS")
const OUTDIR = joinpath(OUTBASE, RUN_TAG)
const REQUIRED_EXPORTS = (
    :local_periodic_voronoi_mesh_arrays_3d,
    :euler_state_3d,
    :moving_mesh_step_3d!,
    :advect_generators_3d,
    :total_conserved_3d,
    :conserved_to_primitive_3d,
)

function missing_exports()
    Symbol[sym for sym in REQUIRED_EXPORTS if !isdefined(PowerFoam, sym)]
end

function cartesian_generators_3d(n)
    h = 1.0 / n
    pts = Matrix{Float64}(undef, n^3, 3)
    idx = 1
    for k in 0:n-1, j in 0:n-1, i in 0:n-1
        pts[idx, 1] = (i + 0.5) * h
        pts[idx, 2] = (j + 0.5) * h
        pts[idx, 3] = (k + 0.5) * h
        idx += 1
    end
    return pts
end

function uniform_mesh_velocity(ncells)
    vx, vy, vz = 0.125, -0.05, 0.075
    return repeat(reshape([vx, vy, vz], 1, 3), ncells, 1)
end

function maxabsdiff(a, b)
    return maximum(abs.(a .- b))
end

function primitive_maxdiff(before, after)
    return (
        rho = maxabsdiff(before.rho, after.rho),
        vx = maxabsdiff(before.vx, after.vx),
        vy = maxabsdiff(before.vy, after.vy),
        vz = maxabsdiff(before.vz, after.vz),
        pressure = maxabsdiff(before.pressure, after.pressure),
    )
end

function total_drift(before, after)
    return (
        mass = after.mass - before.mass,
        mx = after.mx - before.mx,
        my = after.my - before.my,
        mz = after.mz - before.mz,
        energy = after.energy - before.energy,
    )
end

function max_generator_displacement(before, after)
    return maximum(sqrt.(sum((after .- before) .^ 2; dims = 2)))
end

function write_report(path, summary)
    open(path, "w") do io
        println(io, "# AREPO Runtime Moving-Mesh Smoke")
        println(io)
        @printf(io, "- generated: %s\n", Dates.format(now(), dateformat"yyyy-mm-dd HH:MM:SS"))
        @printf(io, "- N: %d\n", summary.n)
        @printf(io, "- cells: %d\n", summary.cells)
        @printf(io, "- dt: %.6g\n", summary.dt)
        @printf(io, "- missing exports: %s\n",
                isempty(summary.missing) ? "none" : join(string.(summary.missing), ", "))
        println(io)
        if !isempty(summary.missing)
            println(io, "## Blocker")
            println(io)
            println(io, "The exported runtime surface is missing the calls above, so this smoke")
            println(io, "example cannot build a periodic moving-mesh hydro step without source edits.")
            return
        end
        println(io, "## Moving-Mesh Result")
        println(io)
        println(io, "| metric | value |")
        println(io, "| --- | ---: |")
        @printf(io, "| max generator displacement | %.12g |\n", summary.max_disp)
        @printf(io, "| max advect vs step point diff | %.12g |\n", summary.max_point_err)
        @printf(io, "| mass drift | %.12g |\n", summary.drift.mass)
        @printf(io, "| mx drift | %.12g |\n", summary.drift.mx)
        @printf(io, "| my drift | %.12g |\n", summary.drift.my)
        @printf(io, "| mz drift | %.12g |\n", summary.drift.mz)
        @printf(io, "| energy drift | %.12g |\n", summary.drift.energy)
        @printf(io, "| rho max diff | %.12g |\n", summary.pdiff.rho)
        @printf(io, "| vx max diff | %.12g |\n", summary.pdiff.vx)
        @printf(io, "| vy max diff | %.12g |\n", summary.pdiff.vy)
        @printf(io, "| vz max diff | %.12g |\n", summary.pdiff.vz)
        @printf(io, "| pressure max diff | %.12g |\n", summary.pdiff.pressure)
        println(io)
        println(io, "This slice uses only exported PowerFoam APIs. The mesh is a tiny periodic")
        println(io, "3-D Voronoi mesh built from Cartesian generator positions, and the gas")
        println(io, "velocity is matched to the prescribed mesh velocity so the ALE update is")
        println(io, "a uniform-flow moving-mesh smoke rather than a heavy hydro run.")
    end
end

function main()
    mkpath(OUTDIR)
    missing = missing_exports()
    report = joinpath(OUTDIR, "README.md")
    if !isempty(missing)
        write_report(report, (; n = N, cells = N^3, dt = DT, missing))
        @printf("wrote %s\n", report)
        @printf("BLOCKER missing exported PowerFoam calls: %s\n", join(string.(missing), ", "))
        return
    end

    domain = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))
    points0 = cartesian_generators_3d(N)
    vmesh = uniform_mesh_velocity(size(points0, 1))
    mesh0 = local_periodic_voronoi_mesh_arrays_3d(
        points0;
        domain,
        T = Float64,
        cell_velocity = vmesh,
        bins_per_axis = N,
        search_radius = 1,
    )
    state = euler_state_3d(
        mesh0.geom;
        rho = 1.25,
        vx = vmesh[:, 1],
        vy = vmesh[:, 2],
        vz = vmesh[:, 3],
        pressure = 0.9,
        gamma = GAMMA,
        T = Float64,
    )
    totals_before = total_conserved_3d(state, mesh0.geom)
    prim_before = conserved_to_primitive_3d(state; gamma = GAMMA)
    expected_points = advect_generators_3d(points0, vmesh, DT, domain; boundary = :periodic)
    moved = moving_mesh_step_3d!(
        state,
        points0;
        dt = DT,
        gamma = GAMMA,
        mesh_velocity = vmesh,
        domain,
        boundary = :periodic,
        rebuild = :local,
        local_bins_per_axis = mesh0.bins_per_axis,
        local_search_radius = mesh0.search_radius,
        riemann = :hll,
        T = Float64,
    )
    totals_after = total_conserved_3d(state, moved.geom)
    prim_after = conserved_to_primitive_3d(state; gamma = GAMMA)
    drift = total_drift(totals_before, totals_after)
    pdiff = primitive_maxdiff(prim_before, prim_after)
    max_point_err = maxabsdiff(expected_points, moved.points)
    max_disp = max_generator_displacement(points0, moved.points)

    tol_total = 1.0e-11
    tol_prim = 1.0e-10
    maximum(abs, collect(drift)) <= tol_total ||
        error("moving-mesh smoke conserved drift exceeded tolerance: $(drift)")
    maximum(pdiff) <= tol_prim ||
        error("moving-mesh smoke primitive drift exceeded tolerance: $(pdiff)")
    max_point_err <= eps(Float64) ||
        error("moving-mesh point advection mismatch: $(max_point_err)")

    summary = (; n = N, cells = size(points0, 1), dt = DT, missing,
               max_disp, max_point_err, drift, pdiff)
    write_report(report, summary)
    @printf("wrote %s\n", report)
    @printf("moving-mesh smoke cells=%d dt=%.3e max_disp=%.3e point_err=%.3e\n",
            summary.cells, summary.dt, summary.max_disp, summary.max_point_err)
    @printf("drift mass=%.3e mx=%.3e my=%.3e mz=%.3e energy=%.3e\n",
            drift.mass, drift.mx, drift.my, drift.mz, drift.energy)
    @printf("primitive maxdiff rho=%.3e vx=%.3e vy=%.3e vz=%.3e pressure=%.3e\n",
            pdiff.rho, pdiff.vx, pdiff.vy, pdiff.vz, pdiff.pressure)
end

main()
