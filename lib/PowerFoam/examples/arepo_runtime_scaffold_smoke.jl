using PowerFoam

function smoke_specs()
    kh2d = arepo_problem_spec(
        :kh2d_scaffold;
        dimensionality = 2,
        domain = ((0.0, 1.0), (0.0, 1.0)),
        periodic = (true, true),
        gas_cell_count = 4096,
        particle_count = 0,
        physics = (hydro = true, tessellation = true, gravity = false),
        initial_conditions = (
            family = :kelvin_helmholtz,
            solver = :hll,
            mesh_motion = :moving,
        ),
        metadata = (
            label = "KH2D-like periodic hydro smoke candidate",
            target_path = :pure_ka_hydro_smoke,
        ),
    )

    noh3d = arepo_problem_spec(
        :noh3d_scaffold;
        dimensionality = 3,
        domain = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
        periodic = (false, false, false),
        gas_cell_count = 32768,
        particle_count = 0,
        physics = (hydro = true, tessellation = true, gravity = false),
        initial_conditions = (
            family = :noh,
            solver = :hll,
            geometry = :spherical_inflow,
        ),
        metadata = (
            label = "Noh3D-like strong-shock hydro smoke candidate",
            target_path = :pure_ka_hydro_smoke,
        ),
    )

    soundwave2d = arepo_problem_spec(
        :soundwave2d_scaffold;
        dimensionality = 2,
        domain = ((0.0, 1.0), (0.0, 1.0)),
        periodic = (true, true),
        gas_cell_count = 32,
        particle_count = 0,
        physics = (hydro = true, tessellation = true, gravity = false),
        initial_conditions = (
            family = :sound_wave,
            solver = :hll,
            geometry = :periodic_acoustic_mode,
        ),
        metadata = (
            label = "SoundWave2D-like periodic acoustic smoke candidate",
            target_path = :soundwave2d_standard_problem,
            nx = 8,
            ny = 4,
            t_final = 0.01,
            cfl = 0.25,
            amplitude = 1e-3,
            riemann = :hll,
        ),
    )

    gresho2d = arepo_problem_spec(
        :gresho2d_scaffold;
        dimensionality = 2,
        domain = ((0.0, 1.0), (0.0, 1.0)),
        periodic = (true, true),
        gas_cell_count = 1024,
        particle_count = 0,
        physics = (hydro = true, tessellation = true, gravity = false),
        initial_conditions = (
            family = :gresho_vortex,
            solver = :hll,
            geometry = :periodic_vortex_balance,
        ),
        metadata = (
            label = "Gresho2D-like periodic vortex smoke candidate",
            target_path = :gresho2d_standard_problem,
            nx = 16,
            ny = 16,
            t_final = 0.01,
            cfl = 0.18,
            nbins = 16,
            center = (0.5, 0.5),
            riemann = :hll,
        ),
    )

    return (kh2d, noh3d, soundwave2d, gresho2d)
end

function print_spec_summary(spec)
    smoke = classify_ka_hydro_smoke(spec)
    options = spec.name == :soundwave2d_scaffold ?
              ArepoRunOptions(final_time = 0.01, max_steps = 10_000, cfl = 0.25) :
              spec.name == :gresho2d_scaffold ?
              ArepoRunOptions(final_time = 0.01, max_steps = 10_000, cfl = 0.18) :
              ArepoRunOptions(max_steps = 1, cfl = 0.4)
    state = arepo_run_scaffold(spec; options)

    println("spec=$(spec.name)")
    println("  dim=$(spec.dimensionality) gas=$(spec.gas_cell_count) particles=$(spec.particle_count) periodic=$(spec.periodic)")
    println("  smoke_status=$(smoke.status) eligible=$(smoke.eligible) requirements=$(join(string.(smoke.requirements), ", "))")
    for reason in smoke.reasons
        println("    - ", reason)
    end
    println("  scaffold_status=$(state.status) unsupported=$(join(string.(state.unsupported), ", "))")
    println("  scaffold_diag_count=$(length(state.diagnostics))")
    if haskey(state.payload, :standard_problem)
        run = state.payload.standard_problem
        println("  standard_problem_status=$(run.status) final_time=$(run.final_metric.t)")
    end
    println()
end

function main()
    println("AREPO runtime scaffold smoke")
    println("============================")
    println()
    for spec in smoke_specs()
        print_spec_summary(spec)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
