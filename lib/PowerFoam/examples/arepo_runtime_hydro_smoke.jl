using Dates
using PowerFoam
using Printf

const GAMMA = 5 / 3
const N = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 4
const DT = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 1.0e-3
const OUTBASE = joinpath(@__DIR__, "out", "arepo_runtime_hydro_smoke")
const RUN_TAG = Dates.format(now(), "yyyymmdd_HHMMSS")
const OUTDIR = joinpath(OUTBASE, RUN_TAG)

function make_spec(n)
    return arepo_problem_spec(
        :cartesian_hydro_smoke_3d;
        dimensionality = 3,
        domain = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
        periodic = (true, true, true),
        gas_cell_count = n^3,
        particle_count = 0,
        physics = (hydro = true, tessellation = false, gravity = false),
        initial_conditions = (
            family = :uniform_flow,
            geometry = :cartesian_periodic,
            mesh_source = :prebuilt_cartesian_arrays,
        ),
        metadata = (
            label = "Prebuilt 3-D Cartesian hydro smoke",
            mesh = :cartesian_periodic_mesh_arrays_3d,
            state = :euler_state_3d,
        ),
    )
end

function uniform_state(mesh)
    return euler_state_3d(
        mesh;
        rho = 1.25,
        vx = 0.2,
        vy = -0.1,
        vz = 0.05,
        pressure = 0.9,
        gamma = GAMMA,
        T = Float64,
    )
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

function check_invariants(smoke, totals_before, totals_after, prim_before, prim_after)
    smoke.eligible || error("smoke classification should be eligible, got $(smoke.status)")
    drift = total_drift(totals_before, totals_after)
    pdiff = primitive_maxdiff(prim_before, prim_after)
    tol = 1.0e-12
    maximum(abs, collect(drift)) <= tol ||
        error("conserved totals drift exceeded tolerance: $(drift)")
    maximum(pdiff) <= tol ||
        error("primitive drift exceeded tolerance: $(pdiff)")
    minimum(prim_after.rho) > 0 || error("rho became non-positive")
    minimum(prim_after.pressure) > 0 || error("pressure became non-positive")
    return drift, pdiff
end

function write_report(path, spec, smoke, totals_before, totals_after, drift, pdiff)
    open(path, "w") do io
        println(io, "# AREPO Runtime Hydro Smoke")
        println(io)
        @printf(io, "- generated: %s\n", Dates.format(now(), dateformat"yyyy-mm-dd HH:MM:SS"))
        @printf(io, "- spec: %s\n", spec.name)
        @printf(io, "- dimensionality: %d\n", spec.dimensionality)
        @printf(io, "- gas cells: %d\n", spec.gas_cell_count)
        @printf(io, "- periodic: %s\n", repr(spec.periodic))
        @printf(io, "- smoke status: %s\n", string(smoke.status))
        @printf(io, "- eligible: %s\n", string(smoke.eligible))
        @printf(io, "- requirements: %s\n", join(string.(smoke.requirements), ", "))
        println(io)
        println(io, "## Conserved Totals")
        println(io)
        println(io, "| stage | mass | mx | my | mz | energy |")
        println(io, "| --- | ---: | ---: | ---: | ---: | ---: |")
        @printf(io, "| before | %.16g | %.16g | %.16g | %.16g | %.16g |\n",
                totals_before.mass, totals_before.mx, totals_before.my,
                totals_before.mz, totals_before.energy)
        @printf(io, "| after | %.16g | %.16g | %.16g | %.16g | %.16g |\n",
                totals_after.mass, totals_after.mx, totals_after.my,
                totals_after.mz, totals_after.energy)
        @printf(io, "| drift | %.16g | %.16g | %.16g | %.16g | %.16g |\n",
                drift.mass, drift.mx, drift.my, drift.mz, drift.energy)
        println(io)
        println(io, "## Primitive Max-Abs Drift")
        println(io)
        println(io, "| rho | vx | vy | vz | pressure |")
        println(io, "| ---: | ---: | ---: | ---: | ---: |")
        @printf(io, "| %.3e | %.3e | %.3e | %.3e | %.3e |\n",
                pdiff.rho, pdiff.vx, pdiff.vy, pdiff.vz, pdiff.pressure)
        println(io)
        println(io, "This slice uses a prebuilt periodic Cartesian mesh with `tessellation=false`")
        println(io, "and advances one uniform-flow `finite_volume_step_3d!` update through the")
        println(io, "current exported `PowerFoam` APIs.")
    end
end

function main()
    mkpath(OUTDIR)
    spec = make_spec(N)
    smoke = classify_ka_hydro_smoke(spec)
    mesh = cartesian_periodic_mesh_arrays_3d(N; T = Float64)
    state = uniform_state(mesh)
    totals_before = total_conserved_3d(state, mesh)
    prim_before = conserved_to_primitive_3d(state; gamma = GAMMA)
    finite_volume_step_3d!(state, mesh; dt = DT, gamma = GAMMA, riemann = :hll)
    totals_after = total_conserved_3d(state, mesh)
    prim_after = conserved_to_primitive_3d(state; gamma = GAMMA)
    drift, pdiff = check_invariants(smoke, totals_before, totals_after,
                                    prim_before, prim_after)
    report = joinpath(OUTDIR, "README.md")
    write_report(report, spec, smoke, totals_before, totals_after, drift, pdiff)
    @printf("wrote %s\n", report)
    @printf("spec=%s smoke_status=%s eligible=%s cells=%d dt=%.3e\n",
            spec.name, smoke.status, smoke.eligible, spec.gas_cell_count, DT)
    @printf("drift mass=%.3e mx=%.3e my=%.3e mz=%.3e energy=%.3e\n",
            drift.mass, drift.mx, drift.my, drift.mz, drift.energy)
    @printf("primitive maxdiff rho=%.3e vx=%.3e vy=%.3e vz=%.3e pressure=%.3e\n",
            pdiff.rho, pdiff.vx, pdiff.vy, pdiff.vz, pdiff.pressure)
end

main()
