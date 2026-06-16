"""
    ArepoGravitySolverSpec

Small package-level registry row for gravity solver readiness.  This does not
execute a gravity step; it makes the current direct/PM/tree/cosmology surface
machine-readable for runtime gates and planning reports.
"""
struct ArepoGravitySolverSpec
    name::Symbol
    family::Symbol
    status::Symbol
    backend::Symbol
    periodic::Bool
    cosmological::Bool
    notes::Vector{String}
end

"""
    ArepoDirectGravityParticleState

Tiny-N particle payload for the direct-gravity runtime slice.  The fields stay
typed and explicit so the scaffold can carry a frozen particle snapshot through
the runtime dispatch path without pulling in the PM/tree machinery.
"""
struct ArepoDirectGravityParticleState{X<:AbstractVector,
                                       Y<:AbstractVector,
                                       Z<:AbstractVector,
                                       M<:AbstractVector,
                                       VX<:AbstractVector,
                                       VY<:AbstractVector,
                                       VZ<:AbstractVector}
    x::X
    y::Y
    z::Z
    m::M
    vx::VX
    vy::VY
    vz::VZ
end

"""
    ArepoDirectGravityResult

Typed payload returned by the bounded direct-gravity runtime slice.
`before` stores the metadata-provided particle state, `after` stores the
frozen one-step kick-drift output when requested, and `accelerations` captures
the direct force evaluation at the `before` positions.
"""
struct ArepoDirectGravityResult{P<:ArepoDirectGravityParticleState,
                                A<:NamedTuple}
    before::P
    after::P
    accelerations::A
    potential_energy::Float64
    momentum_residual::NamedTuple{(:x, :y, :z),Tuple{Float64,Float64,Float64}}
    max_abs_accel::Float64
    softening::Float64
    G::Float64
    dt::Float64
    advanced::Bool
end

function _arepo_direct_gravity_scalar_vector(template::AbstractVector,
                                             value::Real,
                                             particle_count::Integer)
    return fill(promote_type(eltype(template), Float64)(value), Int(particle_count))
end

function _arepo_direct_gravity_vector_or_default(particles, name::Symbol,
                                                 particle_count::Integer;
                                                 default = nothing,
                                                 required::Bool = false)
    value = hasproperty(particles, name) ? getproperty(particles, name) : default
    if value === nothing
        required &&
            error("arepo_direct_gravity_runtime_state: metadata.particles is missing required field $(name)")
        return nothing
    end
    length(value) == particle_count ||
        error("arepo_direct_gravity_runtime_state: particles.$(name) length must match particle_count=$(particle_count)")
    return value
end

function arepo_direct_gravity_particle_state(particles, particle_count::Integer)
    x = _arepo_direct_gravity_vector_or_default(particles, :x, particle_count;
                                                required = true)
    y = _arepo_direct_gravity_vector_or_default(particles, :y, particle_count;
                                                required = true)
    z = _arepo_direct_gravity_vector_or_default(particles, :z, particle_count;
                                                required = true)
    m = _arepo_direct_gravity_vector_or_default(particles, :m, particle_count;
                                                required = true)
    vx = _arepo_direct_gravity_vector_or_default(
        particles, :vx, particle_count;
        default = _arepo_direct_gravity_scalar_vector(x, 0.0, particle_count))
    vy = _arepo_direct_gravity_vector_or_default(
        particles, :vy, particle_count;
        default = _arepo_direct_gravity_scalar_vector(y, 0.0, particle_count))
    vz = _arepo_direct_gravity_vector_or_default(
        particles, :vz, particle_count;
        default = _arepo_direct_gravity_scalar_vector(z, 0.0, particle_count))
    return ArepoDirectGravityParticleState(x, y, z, m, vx, vy, vz)
end

function arepo_direct_gravity_particle_state(particles::ArepoDirectGravityParticleState,
                                             particle_count::Integer)
    length(particles.x) == particle_count ||
        error("arepo_direct_gravity_runtime_state: particles.x length must match particle_count=$(particle_count)")
    length(particles.y) == particle_count ||
        error("arepo_direct_gravity_runtime_state: particles.y length must match particle_count=$(particle_count)")
    length(particles.z) == particle_count ||
        error("arepo_direct_gravity_runtime_state: particles.z length must match particle_count=$(particle_count)")
    length(particles.m) == particle_count ||
        error("arepo_direct_gravity_runtime_state: particles.m length must match particle_count=$(particle_count)")
    length(particles.vx) == particle_count ||
        error("arepo_direct_gravity_runtime_state: particles.vx length must match particle_count=$(particle_count)")
    length(particles.vy) == particle_count ||
        error("arepo_direct_gravity_runtime_state: particles.vy length must match particle_count=$(particle_count)")
    length(particles.vz) == particle_count ||
        error("arepo_direct_gravity_runtime_state: particles.vz length must match particle_count=$(particle_count)")
    return particles
end

function _arepo_direct_gravity_momentum_residual(m, ax, ay, az)
    return (
        x = sum(m .* ax),
        y = sum(m .* ay),
        z = sum(m .* az),
    )
end

function _arepo_direct_gravity_max_abs_accel(ax, ay, az)
    return maximum(sqrt.(ax .* ax .+ ay .* ay .+ az .* az))
end

function _arepo_direct_gravity_runtime_result(before::ArepoDirectGravityParticleState;
                                              G::Real = 1.0,
                                              softening::Real = 0.0,
                                              dt::Real = 0.0,
                                              advanced::Bool = false)
    if advanced
        step = arepo_direct_gravity_kick_drift_step(before.x, before.y, before.z,
                                                    before.m, before.vx, before.vy,
                                                    before.vz; dt = dt,
                                                    G = G, softening = softening)
        after = ArepoDirectGravityParticleState(step.x, step.y, step.z, step.m,
                                                step.vx, step.vy, step.vz)
        accelerations = (ax = step.ax, ay = step.ay, az = step.az)
        return ArepoDirectGravityResult(before, after,
                                        accelerations, float(step.potential_energy),
                                        step.momentum_residual,
                                        float(step.max_abs_accel),
                                        float(softening), float(G), float(step.dt),
                                        true)
    end

    oracle = arepo_direct_gravity_oracle(before.x, before.y, before.z, before.m;
                                         G = G, softening = softening)
    accelerations = (ax = oracle.ax, ay = oracle.ay, az = oracle.az)
    residual = _arepo_direct_gravity_momentum_residual(before.m,
                                                       oracle.ax, oracle.ay, oracle.az)
    max_abs = _arepo_direct_gravity_max_abs_accel(oracle.ax, oracle.ay, oracle.az)
    return ArepoDirectGravityResult(
        before, before, accelerations, float(oracle.potential_energy), residual,
        float(max_abs), float(softening), float(G), float(dt), false)
end

function arepo_direct_gravity_runtime_state(spec::ArepoProblemSpec;
                                            backend::Symbol = :ka,
                                            device = nothing,
                                            options::ArepoRunOptions = ArepoRunOptions())
    get(spec.physics, :gravity, false) || return nothing
    spec.particle_count > 0 || return nothing
    solver = Symbol(get(spec.metadata, :gravity_solver, :direct_tiny_n))
    solver in (:direct, :direct_tiny_n, :direct_open) || return nothing
    particles = get(spec.metadata, :particles, nothing)
    particles === nothing && return nothing

    particle_state = arepo_direct_gravity_particle_state(particles, spec.particle_count)
    softening = float(get(spec.metadata, :softening,
                          get(spec.metadata, :gravity_softening, 0.0)))
    G = float(get(spec.metadata, :G, 1.0))
    advanced = Bool(get(spec.metadata, :advance_gravity,
                        get(spec.metadata, :kick_drift, false)))
    dt = float(get(spec.metadata, :dt,
                   options.final_time > options.start_time ?
                   options.final_time - options.start_time : 0.0))
    if advanced && dt == 0.0
        error("arepo_direct_gravity_runtime_state: advance_gravity=true requires a positive dt")
    end

    result = _arepo_direct_gravity_runtime_result(particle_state;
                                                  G = G, softening = softening,
                                                  dt = dt, advanced = advanced)
    diagnostics = String[
        "Executed bounded direct-gravity runtime slice from metadata-provided particles.",
        "particle_count=$(spec.particle_count), advanced=$(advanced), dt=$(result.dt), softening=$(result.softening), G=$(result.G).",
        "net_force=$(result.momentum_residual), max_abs_accel=$(result.max_abs_accel), potential_energy=$(result.potential_energy).",
    ]
    payload = (
        mesh = nothing,
        hydro = nothing,
        gravity = result,
        outputs = NamedTuple(),
        backend_request = backend,
        device_request = device,
        direct_gravity_particles = particle_state,
    )
    status = advanced ? :gravity_advanced : :gravity_direct
    return arepo_runtime_state_3d(spec; backend, device, options,
                                  diagnostics, unsupported = Symbol[],
                                  status, payload)
end

function _arepo_pm_gravity_fixture_from_state(spec::ArepoProblemSpec,
                                              particle_state::ArepoDirectGravityParticleState)
    xspan = spec.domain[1][2] - spec.domain[1][1]
    yspan = spec.domain[2][2] - spec.domain[2][1]
    zspan = spec.domain[3][2] - spec.domain[3][1]
    boxsize = float(get(spec.metadata, :boxsize,
                        get(spec.metadata, :BoxSize,
                            maximum((xspan, yspan, zspan)))))
    Npm = Int(get(spec.metadata, :Npm, get(spec.metadata, :npm, 16)))
    ng = Int(get(spec.metadata, :ng, get(spec.metadata, :pm_ghost_depth, 3)))
    Npm > 0 || error("arepo_pm_gravity_runtime_state: Npm must be positive")
    ng >= 1 || error("arepo_pm_gravity_runtime_state: ng must be positive")
    boxsize > 0 || error("arepo_pm_gravity_runtime_state: boxsize must be positive")
    return (
        Npm = Npm,
        boxsize = boxsize,
        ng = ng,
        x = particle_state.x,
        y = particle_state.y,
        z = particle_state.z,
        m = particle_state.m,
        vx = particle_state.vx,
        vy = particle_state.vy,
        vz = particle_state.vz,
    )
end

function arepo_pm_gravity_runtime_state(spec::ArepoProblemSpec;
                                        backend::Symbol = :ka,
                                        device = nothing,
                                        options::ArepoRunOptions = ArepoRunOptions())
    get(spec.physics, :gravity, false) || return nothing
    spec.particle_count > 0 || return nothing
    solver = Symbol(get(spec.metadata, :gravity_solver, :direct_tiny_n))
    solver in (:pm, :periodic_pm, :periodic_pm_root) || return nothing
    particles = get(spec.metadata, :particles, nothing)
    particles === nothing && return nothing

    particle_state = arepo_direct_gravity_particle_state(particles, spec.particle_count)
    fixture = _arepo_pm_gravity_fixture_from_state(spec, particle_state)
    greens = Symbol(get(spec.metadata, :greens, :spectral))
    probe = probe_poissonkernels_monorepo()
    if probe.pm_module === nothing
        diagnostics = String[
            "Requested periodic PM gravity runtime slice, but PoissonKernels could not be loaded.",
            sprint(showerror, probe.error),
        ]
        payload = (
            mesh = nothing,
            hydro = nothing,
            gravity = nothing,
            outputs = NamedTuple(),
            backend_request = backend,
            device_request = device,
            direct_gravity_particles = particle_state,
        )
        return arepo_runtime_state_3d(spec; backend, device, options,
                                      diagnostics,
                                      unsupported = [:pm_gravity_dependency],
                                      status = :unsupported,
                                      payload)
    end

    workspace = arepo_pm_gravity_workspace(; fixture = fixture, greens = greens)
    result = arepo_pm_gravity!(workspace, probe.pm_module,
                               fixture.x, fixture.y, fixture.z, fixture.m,
                               fixture.vx, fixture.vy, fixture.vz)
    diagnostics = String[
        "Executed bounded periodic PM gravity runtime slice from metadata-provided particles.",
        "particle_count=$(spec.particle_count), Npm=$(fixture.Npm), ng=$(fixture.ng), boxsize=$(fixture.boxsize), greens=$(greens).",
        "mass_sum=$(result.mass_sum), rhs_sum=$(result.rhs_sum), net_force=$(result.net_force), max_abs_accel=$(result.max_abs_accel).",
        "This runtime branch evaluates PM acceleration only; cosmological KDK advancement is not wired yet.",
    ]
    payload = (
        mesh = nothing,
        hydro = nothing,
        gravity = result,
        outputs = NamedTuple(),
        backend_request = backend,
        device_request = device,
        direct_gravity_particles = particle_state,
        pm_gravity = result,
    )
    return arepo_runtime_state_3d(spec; backend, device, options,
                                  diagnostics, unsupported = Symbol[],
                                  status = :gravity_pm,
                                  payload)
end

"""
    arepo_gravity_solver_registry(; probe_pm=false)

Return the current PowerFoam gravity solver readiness table.  `probe_pm=true`
checks whether the repo-local `PoissonKernels` PM module is loadable.
"""
function arepo_gravity_solver_registry(; probe_pm::Bool = false)
    pm_status = :component_ready
    pm_notes = String[
        "periodic PM preflight exists via PoissonKernels deposit/FFT/ghost/interp chain",
        "bounded PM workspace/result helpers expose the reusable deposit->solve->ghost-fill->interp surface",
        "finite direct-oracle comparison is diagnostic until production periodic convention is certified",
    ]
    if probe_pm
        probe = probe_poissonkernels_monorepo()
        if probe.pm_module === nothing
            pm_status = :blocked
            push!(pm_notes, "PoissonKernels load failed: $(sprint(showerror, probe.error))")
        else
            pm_status = :runtime_ready
            push!(pm_notes, "PoissonKernels resolved from $(something(probe.pkg_entry, "unknown path"))")
        end
    end

    return ArepoGravitySolverSpec[
        ArepoGravitySolverSpec(
            :direct_tiny_n, :direct, :component_ready, :ka_cpu,
            false, false,
            ["open-boundary O(N^2) direct force and potential oracle",
             "softened pair force is covered by unit tests",
             "periodic=true remains intentionally outside this open-boundary API"]),
        ArepoGravitySolverSpec(
            :periodic_pm_root, :pm, pm_status, :poissonkernels,
            true, false, pm_notes),
        ArepoGravitySolverSpec(
            :tree_short_range, :tree, :planned, :none,
            false, false,
            ["production tree/short-range gravity is not implemented in PowerFoam yet"]),
        ArepoGravitySolverSpec(
            :cosmological_pm, :pm, :planned, :none,
            true, true,
            ["scale-factor aware PM and KDK cosmological integration are not wired yet"]),
        ArepoGravitySolverSpec(
            :hydro_self_gravity, :coupled, :planned, :none,
            true, true,
            ["gas+particle source coupling and hydro gravity source terms remain future work"]),
    ]
end

function arepo_gravity_solver_status(name::Union{Symbol,AbstractString};
                                     probe_pm::Bool = false)
    target = Symbol(name)
    matches = filter(row -> row.name == target,
                     arepo_gravity_solver_registry(; probe_pm))
    isempty(matches) &&
        error("arepo_gravity_solver_status: unknown gravity solver $(target)")
    return only(matches)
end
