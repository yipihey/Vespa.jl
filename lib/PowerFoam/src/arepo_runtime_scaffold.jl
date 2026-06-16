"""
    ArepoRunOptions

Lightweight runtime controls for the future pure-Julia AREPO-style driver.
These options are intentionally small and dependency-free so this file can be
`include`d before the main runtime/export wiring is finalized.
"""
struct ArepoRunOptions
    start_time::Float64
    final_time::Float64
    max_steps::Int
    cfl::Float64
    output_interval::Int
    record_diagnostics::Bool
    allow_unsupported::Bool
end

"""
    ArepoHydroSmokeAssessment

Small diagnostic record describing whether a problem spec could plausibly route
through a future pure-KA hydro smoke harness. This is intentionally advisory:
it does not execute kernels or validate tessellation/hydro implementations.
"""
struct ArepoHydroSmokeAssessment
    eligible::Bool
    status::Symbol
    reasons::Vector{String}
    requirements::Vector{Symbol}
end

"""
    ArepoRunOptions(; kwargs...)

Keyword constructor with conservative defaults for scaffold-only runs.
"""
function ArepoRunOptions(; start_time::Real = 0.0,
                         final_time::Real = 0.0,
                         max_steps::Integer = 0,
                         cfl::Real = 0.4,
                         output_interval::Integer = 0,
                         record_diagnostics::Bool = true,
                         allow_unsupported::Bool = true)
    max_steps >= 0 || error("ArepoRunOptions: max_steps must be nonnegative")
    output_interval >= 0 || error("ArepoRunOptions: output_interval must be nonnegative")
    cfl > 0 || error("ArepoRunOptions: cfl must be positive")
    float(final_time) >= float(start_time) ||
        error("ArepoRunOptions: final_time must be >= start_time")
    return ArepoRunOptions(float(start_time), float(final_time), Int(max_steps),
                           float(cfl), Int(output_interval),
                           record_diagnostics, allow_unsupported)
end

"""
    ArepoProblemSpec

Small immutable description of an AREPO-like problem setup.  The fields are
chosen to be broad enough for hydro, tessellation, and gravity orchestration
without committing to any heavy runtime payload yet.
"""
struct ArepoProblemSpec{P<:NamedTuple,I<:NamedTuple,M<:NamedTuple}
    name::Symbol
    dimensionality::Int
    domain::NTuple{3,NTuple{2,Float64}}
    periodic::NTuple{3,Bool}
    gas_cell_count::Int
    particle_count::Int
    physics::P
    initial_conditions::I
    metadata::M
end

"""
    ArepoProblemSpec(name; kwargs...)

Construct an AREPO-style problem description with normalized 3-D domain and
periodicity metadata.  Lower-dimensional problems are embedded in the 3-D
runtime shape by padding inactive axes with `[0, 1]` and `false`.
"""
function ArepoProblemSpec(name::Union{Symbol,AbstractString};
                          dimensionality::Integer = 3,
                          domain = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                          periodic = (true, true, true),
                          gas_cell_count::Integer = 0,
                          particle_count::Integer = 0,
                          physics::NamedTuple = (hydro = true,
                                                 tessellation = true,
                                                 gravity = false),
                          initial_conditions::NamedTuple = NamedTuple(),
                          metadata::NamedTuple = NamedTuple())
    dim = Int(dimensionality)
    1 <= dim <= 3 || error("ArepoProblemSpec: dimensionality must be in 1:3")
    gas_cell_count >= 0 || error("ArepoProblemSpec: gas_cell_count must be nonnegative")
    particle_count >= 0 || error("ArepoProblemSpec: particle_count must be nonnegative")
    return ArepoProblemSpec(Symbol(name), dim,
                            _normalize_domain3d(domain),
                            _normalize_periodic3d(periodic),
                            Int(gas_cell_count), Int(particle_count),
                            physics, initial_conditions, metadata)
end

"""
    ArepoRuntimeState3D

Mutable scaffold state returned by `arepo_run_scaffold`.  The placeholders are
explicit so later integration can swap in live mesh, hydro, gravity, and output
substates without changing the top-level driver shape.
"""
mutable struct ArepoRuntimeState3D{S,D,P}
    spec::S
    backend::Symbol
    device::D
    options::ArepoRunOptions
    time::Float64
    step::Int
    status::Symbol
    diagnostics::Vector{String}
    unsupported::Vector{Symbol}
    payload::P
end

"""
    arepo_problem_spec(name; kwargs...)

Convenience wrapper around `ArepoProblemSpec(...)`.
"""
arepo_problem_spec(name; kwargs...) = ArepoProblemSpec(name; kwargs...)

"""
    arepo_runtime_state_3d(spec; backend=:ka, device=nothing,
                           options=ArepoRunOptions(),
                           diagnostics=String[],
                           unsupported=Symbol[],
                           status=:initialized,
                           payload=(;))

Build a mutable runtime state without executing any physics.
"""
function arepo_runtime_state_3d(spec::ArepoProblemSpec;
                                backend::Symbol = :ka,
                                device = nothing,
                                options::ArepoRunOptions = ArepoRunOptions(),
                                diagnostics::Vector{String} = String[],
                                unsupported::Vector{Symbol} = Symbol[],
                                status::Symbol = :initialized,
                                payload::NamedTuple = (;))
    return ArepoRuntimeState3D(spec, backend, device, options, options.start_time,
                               0, status, copy(diagnostics), copy(unsupported),
                               payload)
end

"""
    arepo_run_scaffold(spec; backend=:ka, device=nothing,
                       options=ArepoRunOptions())

Stub entrypoint for the future pure-Julia AREPO runtime.  It accepts a problem
specification and returns a state object whose diagnostics explain which pieces
of the real runtime are still unsupported.
"""
function arepo_run_scaffold(spec::ArepoProblemSpec;
                            backend::Symbol = :ka,
                            device = nothing,
                            options::ArepoRunOptions = ArepoRunOptions())
    standard = _try_arepo_standard_problem_run(spec, backend, device, options)
    standard === nothing || return standard

    diagnostics, unsupported = _scaffold_diagnostics(spec, backend, device, options)
    smoke = classify_ka_hydro_smoke(spec; backend, device)
    status = isempty(unsupported) ? :ready : :unsupported
    payload = (
        mesh = nothing,
        hydro = nothing,
        gravity = nothing,
        outputs = NamedTuple(),
        backend_request = backend,
        device_request = device,
        ka_hydro_smoke = smoke,
    )
    push!(diagnostics,
          "Pure-KA hydro smoke classification: status=$(smoke.status), eligible=$(smoke.eligible), requirements=$(smoke.requirements).")
    append!(diagnostics, ("  smoke: " * reason for reason in smoke.reasons))
    return arepo_runtime_state_3d(spec; backend, device, options,
                                  diagnostics, unsupported, status, payload)
end

function _try_arepo_standard_problem_run(spec::ArepoProblemSpec, backend::Symbol,
                                         device, options::ArepoRunOptions)
    backend == :ka || return nothing
    get(spec.physics, :hydro, false) || return nothing
    get(spec.physics, :gravity, false) && return nothing

    if spec.name in (:noh2d, :noh_2d)
        spec.dimensionality == 2 || return nothing
        n_side = Int(get(spec.metadata, :n_side,
                        spec.gas_cell_count > 0 ?
                        max(1, round(Int, sqrt(spec.gas_cell_count))) : 24))
        t_final = get(spec.metadata, :t_final,
                      options.final_time > options.start_time ? options.final_time : 0.2)
        nbins = Int(get(spec.metadata, :nbins, n_side))
        cfl = get(spec.metadata, :cfl, options.cfl)
        gamma = get(spec.metadata, :gamma, PF_NOH2D_DEFAULT_GAMMA)
        rho0 = get(spec.metadata, :rho0, 1.0)
        p0 = get(spec.metadata, :p0, 1e-4)
        vrad = get(spec.metadata, :vrad, 1.0)
        domain_radius = get(spec.metadata, :domain_radius,
                            max(abs(spec.domain[1][1]), abs(spec.domain[1][2]),
                                abs(spec.domain[2][1]), abs(spec.domain[2][2])))
        riemann = Symbol(get(spec.metadata, :riemann, :hll))
        max_steps = options.max_steps > 0 ? options.max_steps :
                    Int(get(spec.metadata, :max_steps, 10_000))

        run = pf_noh2d_run(; n_side, t_final, nbins, cfl, gamma, rho0, p0,
                           vrad, domain_radius, riemann, max_steps)
        diagnostics = String[
            "Executed package-owned PowerFoam Noh2D standard-problem runtime path.",
            "Noh2D status=$(run.status), steps=$(length(run.logs)), final_time=$(run.final_metric.t).",
            "mass_rel_drift=$(run.final_metric.mass_rel_drift), energy_rel_drift=$(run.final_metric.energy_rel_drift), rho_max=$(run.final_metric.rho_max).",
        ]
        payload = (
            mesh = run.geom,
            hydro = run.state,
            gravity = nothing,
            outputs = (standard_problem = :noh2d,),
            backend_request = backend,
            device_request = device,
            ka_hydro_smoke = classify_ka_hydro_smoke(spec; backend, device),
            standard_problem = run,
        )
        status = run.numerics_ok ? :calibration_pending : :failed
        return arepo_runtime_state_3d(spec; backend, device, options,
                                      diagnostics, unsupported = Symbol[],
                                      status, payload)
    elseif spec.name in (:soundwave2d, :soundwave2d_scaffold,
                         :sound_wave_2d, :sound_wave2d)
        spec.dimensionality == 2 || return nothing
        nx = Int(get(spec.metadata, :nx,
                    spec.gas_cell_count > 0 ?
                    max(1, round(Int, sqrt(spec.gas_cell_count))) : 32))
        ny = Int(get(spec.metadata, :ny,
                    spec.gas_cell_count > 0 ?
                    max(1, cld(spec.gas_cell_count, nx)) : 8))
        t_final = get(spec.metadata, :t_final,
                      options.final_time > options.start_time ? options.final_time : 0.05)
        cfl = get(spec.metadata, :cfl, options.cfl)
        gamma = get(spec.metadata, :gamma, PF_SOUNDWAVE2D_DEFAULT_GAMMA)
        rho0 = get(spec.metadata, :rho0, 1.0)
        p0 = get(spec.metadata, :p0, 1.0)
        amplitude = get(spec.metadata, :amplitude, 1e-3)
        mode = Int(get(spec.metadata, :mode, 1))
        riemann = Symbol(get(spec.metadata, :riemann, :hll))
        max_steps = options.max_steps > 0 ? options.max_steps :
                    Int(get(spec.metadata, :max_steps, 10_000))
        xlim = get(spec.metadata, :xlim, (0.0, 1.0))
        ylim = get(spec.metadata, :ylim, (0.0, 1.0))

        run = pf_soundwave2d_run(; nx, ny, t_final, cfl, gamma, rho0, p0,
                                 amplitude, mode, riemann, max_steps, xlim, ylim)
        diagnostics = String[
            "Executed package-owned PowerFoam sound-wave 2D standard-problem runtime path.",
            "soundwave2d status=$(run.status), steps=$(length(run.logs)), final_time=$(run.final_metric.t).",
            "mass_rel_drift=$(run.final_metric.mass_rel_drift), energy_rel_drift=$(run.final_metric.energy_rel_drift), rho_mode_amp_ratio=$(run.final_metric.rho_mode_amp_ratio).",
        ]
        payload = (
            mesh = run.geom,
            hydro = run.state,
            gravity = nothing,
            outputs = (standard_problem = :soundwave2d,),
            backend_request = backend,
            device_request = device,
            ka_hydro_smoke = classify_ka_hydro_smoke(spec; backend, device),
            standard_problem = run,
        )
        status = run.numerics_ok ? :calibration_pending : :failed
        return arepo_runtime_state_3d(spec; backend, device, options,
                                      diagnostics, unsupported = Symbol[],
                                      status, payload)
    elseif spec.name in (:gresho2d, :gresho2d_scaffold, :gresho_2d)
        spec.dimensionality == 2 || return nothing
        nx = Int(get(spec.metadata, :nx,
                    spec.gas_cell_count > 0 ?
                    max(1, round(Int, sqrt(spec.gas_cell_count))) : 32))
        ny = Int(get(spec.metadata, :ny,
                    spec.gas_cell_count > 0 ?
                    max(1, cld(spec.gas_cell_count, nx)) : nx))
        t_final = get(spec.metadata, :t_final,
                      options.final_time > options.start_time ? options.final_time : 0.02)
        cfl = get(spec.metadata, :cfl, options.cfl)
        gamma = get(spec.metadata, :gamma, PF_GRESHO2D_DEFAULT_GAMMA)
        rho0 = get(spec.metadata, :rho0, 1.0)
        center = get(spec.metadata, :center, (0.5, 0.5))
        riemann = Symbol(get(spec.metadata, :riemann, :hll))
        max_steps = options.max_steps > 0 ? options.max_steps :
                    Int(get(spec.metadata, :max_steps, 10_000))
        nbins = Int(get(spec.metadata, :nbins, max(12, min(nx, ny))))
        xlim = get(spec.metadata, :xlim, (0.0, 1.0))
        ylim = get(spec.metadata, :ylim, (0.0, 1.0))

        run = pf_gresho2d_run(; nx, ny, t_final, cfl, gamma, rho0, center,
                              riemann, max_steps, nbins, xlim, ylim)
        diagnostics = String[
            "Executed package-owned PowerFoam Gresho 2D standard-problem runtime path.",
            "gresho2d status=$(run.status), steps=$(length(run.logs)), final_time=$(run.final_metric.t).",
            "mass_rel_drift=$(run.final_metric.mass_rel_drift), energy_rel_drift=$(run.final_metric.energy_rel_drift), vt_peak_ratio=$(run.final_metric.vt_peak_ratio).",
        ]
        payload = (
            mesh = run.geom,
            hydro = run.state,
            gravity = nothing,
            outputs = (standard_problem = :gresho2d,),
            backend_request = backend,
            device_request = device,
            ka_hydro_smoke = classify_ka_hydro_smoke(spec; backend, device),
            standard_problem = run,
        )
        status = run.numerics_ok ? :calibration_pending : :failed
        return arepo_runtime_state_3d(spec; backend, device, options,
                                      diagnostics, unsupported = Symbol[],
                                      status, payload)
    else
        return nothing
    end
end

"""
    classify_ka_hydro_smoke(spec; backend=:ka, device=nothing)

Classify whether a problem spec fits the narrow shape of a future pure
KernelAbstractions hydro smoke harness. The result is diagnostic-only and is
meant to help the orchestrator decide whether a lightweight hydro-only path is
worth attempting before full runtime integration exists.
"""
function classify_ka_hydro_smoke(spec::ArepoProblemSpec;
                                 backend::Symbol = :ka,
                                 device = nothing)
    reasons = String[]
    requirements = Symbol[:runtime_loop_stub, :hydro_state_builder]
    eligible = true
    status = :eligible

    if backend != :ka
        eligible = false
        status = :needs_ka_backend
        push!(reasons, "backend=$(backend) is outside the narrow pure-KA smoke path")
    else
        push!(reasons, "backend=:ka matches the planned pure-KA smoke entrypoint")
    end

    if !get(spec.physics, :hydro, false)
        eligible = false
        status = :missing_hydro
        push!(reasons, "physics.hydro=false leaves no hydro phase to smoke-test")
    else
        push!(reasons, "physics.hydro=true enables a hydro-only orchestration target")
    end

    if spec.dimensionality ∉ (2, 3)
        eligible = false
        status = :unsupported_dimensionality
        push!(reasons, "dimensionality=$(spec.dimensionality) is outside the current 2-D/3-D smoke target")
    else
        push!(reasons, "dimensionality=$(spec.dimensionality) fits the current 2-D/3-D smoke target")
    end

    if spec.gas_cell_count <= 0
        eligible = false
        status = :missing_gas_cells
        push!(reasons, "gas_cell_count=$(spec.gas_cell_count) gives no gas state to construct")
    else
        push!(reasons, "gas_cell_count=$(spec.gas_cell_count) is sufficient for a synthetic hydro state")
    end

    if spec.particle_count > 0 || get(spec.physics, :gravity, false)
        eligible = false
        status = :mixed_physics
        push!(reasons, "particle/gravity content would force the smoke harness beyond hydro-only scope")
    else
        push!(reasons, "no particle/gravity coupling is requested")
    end

    if get(spec.physics, :tessellation, true)
        push!(requirements, :tessellation_adapter)
        push!(reasons, "tessellation=true means the smoke path still needs a mesh adapter, even if it stays lightweight")
    else
        push!(reasons, "tessellation=false permits a prebuilt-geometry smoke path")
    end

    if get(spec.metadata, :requires_restart, false)
        eligible = false
        status = :restart_only
        push!(reasons, "metadata.requires_restart=true is outside a first-pass fresh-start smoke harness")
    end

    if device === nothing
        push!(reasons, "device is unspecified, which is acceptable for a shape-only classifier")
    else
        push!(reasons, "device=$(repr(device)) is recorded for later backend/device wiring")
    end

    if eligible && :tessellation_adapter in requirements
        status = :eligible_with_tessellation_adapter
    elseif eligible
        status = :eligible
    end

    return ArepoHydroSmokeAssessment(eligible, status, reasons, unique(requirements))
end

function _normalize_domain3d(domain)
    values = collect(domain)
    1 <= length(values) <= 3 ||
        error("ArepoProblemSpec: domain must provide between 1 and 3 axes")
    out = ntuple(3) do i
        if i <= length(values)
            axis = values[i]
            length(axis) == 2 ||
                error("ArepoProblemSpec: each domain axis must have exactly 2 bounds")
            lo = float(axis[1])
            hi = float(axis[2])
            hi >= lo || error("ArepoProblemSpec: domain upper bound must be >= lower bound")
            (lo, hi)
        else
            (0.0, 1.0)
        end
    end
    return out
end

function _normalize_periodic3d(periodic)
    values = Bool.(collect(periodic))
    1 <= length(values) <= 3 ||
        error("ArepoProblemSpec: periodic must provide between 1 and 3 axes")
    return ntuple(i -> i <= length(values) ? values[i] : false, 3)
end

function _scaffold_diagnostics(spec::ArepoProblemSpec, backend::Symbol, device,
                               options::ArepoRunOptions)
    unsupported = Symbol[
        :runtime_loop,
        :parameter_parser,
        :snapshot_io,
        :output_policy,
        :time_integration,
        :timestep_hierarchy,
    ]
    diagnostics = String[
        "arepo_run_scaffold is a planning stub: no mesh build, hydro step, gravity step, or output write occurs yet.",
        "The scaffold accepts backend=$(backend) and device=$(repr(device)) for API shaping only; no KernelAbstractions dispatch is performed here.",
        "Problem $(spec.name) is normalized to a 3-D runtime envelope with dimensionality=$(spec.dimensionality).",
        "Gas cells=$(spec.gas_cell_count), particles=$(spec.particle_count), periodic=$(spec.periodic), domain=$(spec.domain).",
        "Options request start_time=$(options.start_time), final_time=$(options.final_time), max_steps=$(options.max_steps), cfl=$(options.cfl).",
    ]

    if get(spec.physics, :tessellation, false)
        push!(unsupported, :tessellation)
        push!(diagnostics, "Tessellation integration is not wired: future versions should route to PowerFoam 3-D Voronoi/Delaunay builders and rebuild policies.")
    end
    if get(spec.physics, :hydro, false)
        push!(unsupported, :hydro)
        push!(diagnostics, "Hydro integration is not wired: future versions should call PowerFoam face-state, flux, and ALE update paths through an AREPO-style run loop.")
    end
    if get(spec.physics, :gravity, false) || spec.particle_count > 0
        push!(unsupported, :gravity)
        push!(diagnostics, "Gravity/cosmology integration is not wired: future versions should attach PM/tree or direct-force phases plus kick-drift-kick scheduling.")
    end
    if backend != :ka
        push!(unsupported, :backend_selector)
        push!(diagnostics, "Only the high-level :ka API shape is sketched here; bridge/native backend switching belongs in the later orchestrator.")
    end
    if !options.allow_unsupported
        push!(diagnostics, "allow_unsupported=false was requested, but this scaffold still returns diagnostics instead of throwing so callers can inspect planned gaps.")
    end
    return diagnostics, unique(unsupported)
end
