using Test

ENV["MHD_BACKEND"] = "cpu"
include("pmf_dm_lattice_bench.jl")

@testset "PMF checkpoint endpoint extension" begin
    mktempdir() do dir
        path = joinpath(dir, "checkpoint.jls")
        stored = (N=4, zstart=3.0e6, zend=1.0e5, cfl=0.1)
        expected = (N=4, zstart=3.0e6, zend=500.0, cfl=0.1)
        source = Float32[1, 2, 3]
        _write_pmf_checkpoint(path, stored, (cycle=7,), ["field" => source])
        destination = zeros(Float32, 3)
        runtime = _restore_pmf_checkpoint!(path, expected, ["field" => destination])
        @test runtime.cycle == 7
        @test destination == source

        incompatible = (N=4, zstart=3.0e6, zend=500.0, cfl=0.2)
        @test_throws ErrorException _restore_pmf_checkpoint!(
            path, incompatible, ["field" => zeros(Float32, 3)])

        destination .= 0
        runtime = _restore_pmf_checkpoint!(
            path, incompatible, ["field" => destination];
            allowed_config_changes=(:cfl,))
        @test runtime.cycle == 7
        @test destination == source

        physical_change = (N=8, zstart=3.0e6, zend=500.0, cfl=0.2)
        @test_throws ErrorException _restore_pmf_checkpoint!(
            path, physical_change, ["field" => zeros(Float32, 3)];
            allowed_config_changes=(:cfl,))
    end
end

@testset "nonlocal table checkpoint migration guard" begin
    mktempdir() do dir
        function write_table(path, redshifts)
            open(path, "w") do io
                println(io, "redshift\tk_mpc")
                for z in redshifts
                    println(io, "$z\t100")
                end
            end
        end
        old_table = joinpath(dir, "old.tsv")
        new_table = joinpath(dir, "new.tsv")
        write_table(old_table, (700.0, 1200.0, 1700.0))
        write_table(new_table, (700.0, 1200.0, 1700.0))
        stored = (N=4, zend=500.0, chem_nonlocal_table=old_table)
        expected = (N=4, zend=500.0, chem_nonlocal_table=new_table)
        source = Float32[1, 2, 3]

        safe_path = joinpath(dir, "safe.jls")
        _write_pmf_checkpoint(
            safe_path, stored, (cycle=7, a=z_to_a(1e5)), ["field" => source]
        )
        destination = zeros(Float32, 3)
        runtime = _restore_pmf_checkpoint!(
            safe_path, expected, ["field" => destination];
            allowed_config_changes=(:chem_nonlocal_table,),
            allowed_config_validator=_checkpoint_nonlocal_table_change_is_safe,
        )
        @test runtime.cycle == 7
        @test destination == source

        unsafe_path = joinpath(dir, "unsafe.jls")
        _write_pmf_checkpoint(
            unsafe_path, stored, (cycle=8, a=z_to_a(1100.0)), ["field" => source]
        )
        @test_throws ErrorException _restore_pmf_checkpoint!(
            unsafe_path, expected, ["field" => zeros(Float32, 3)];
            allowed_config_changes=(:chem_nonlocal_table,),
            allowed_config_validator=_checkpoint_nonlocal_table_change_is_safe,
        )
    end
end

@testset "Jedamzik cosmological MHD scaling" begin
    a_ref = 1 / 4501
    a = 1 / 1501
    h = 0.67
    vunit = 8.0e5
    lbox = 0.713 * 3.0857e21

    legacy_clock = _hydro_time_per_tau(a, h, vunit, lbox)
    transformed_clock = _hydro_time_per_tau(a, h, vunit, lbox, a_ref)
    @test transformed_clock / legacy_clock ≈ sqrt(a_ref / a)

    dtphys = 1.25e10
    legacy_dt = _hydro_code_dt(dtphys, 0.0, vunit, lbox, a)
    transformed_dt = _hydro_code_dt(dtphys, 0.0, vunit, lbox, a, a_ref)
    @test transformed_dt / legacy_dt ≈ sqrt(a_ref / a)
    @test _cosmological_velocity_unit(vunit, a, a_ref) ≈
          vunit * sqrt(a_ref / a)

    # Integrating the extra H~/2 term exactly gives v~ ∝ a^(-1/2).
    @test _hubble_momentum_decay(a_ref, a, a_ref) ≈ sqrt(a_ref / a)
    @test _hubble_momentum_decay(a_ref, a, NaN) == 1.0

    # At the normalization epoch the transformed and legacy code units agree.
    @test _hydro_time_per_tau(a_ref, h, vunit, lbox, a_ref) ≈
          _hydro_time_per_tau(a_ref, h, vunit, lbox)
    @test _hydro_code_dt(dtphys, 0.0, vunit, lbox, a_ref, a_ref) ≈
          _hydro_code_dt(dtphys, 0.0, vunit, lbox, a_ref)
end

@testset "imported PMF rescaling" begin
    mktempdir() do dir
        N = 4
        dims = (N, N, N)
        fill_value = (1.0f0, 2.0f0, 3.0f0)
        for (name, value) in zip(("bx.f32", "by.f32", "bz.f32"), fill_value)
            _write_f32_array_atomic(joinpath(dir, name), fill(Float32(value), dims))
        end
        be = MHDKernels.backend(:cpu)
        state = MHDKernels.allocate_state(be, Float32, dims; dx=1/N, gamma=5/3,
                                          riemann=:hll, recon=:ppm)
        target = 0.25f0
        measured = _import_initial_b_field!(be, dir, state; N=N, p0=1.0,
                                             target_brms=target)
        @test isapprox(measured, target; rtol=5e-6)
        host = MHDKernels.fields_to_host(state)
        @test all(iszero, host[2]) && all(iszero, host[3]) && all(iszero, host[4])
        expected_energy = 1.0f0 / (Float32(5/3) - 1.0f0) + 0.5f0 * target^2
        @test all(value -> isapprox(value, expected_energy; rtol=1e-5), host[5])
    end
end

@testset "joint PMF handoff import" begin
    mktempdir() do dir
        N = 4
        dims = (N, N, N)
        be = MHDKernels.backend(:cpu)
        state = MHDKernels.allocate_state(be, Float32, dims; dx=1/N, gamma=5/3,
                                          riemann=:hll, recon=:ppm)
        expected = [fill(Float32(i), dims) for i in 1:9]
        expected[1] .= 1.25f0
        expected[5] .= 8.0f0
        for i in eachindex(expected)
            _write_f32_array_atomic(joinpath(dir, "mhd_U_$i.f32"), expected[i])
        end
        measured = _import_initial_mhd_state!(be, dir, state; N=N)
        @test isfinite(measured) && measured > 0
        @test isfinite(_device_pressure_mean!(be, state, state.scratch[1]))
        for i in eachindex(expected)
            @test Array(state.U[i]) == vec(expected[i])
        end

        particles = ntuple(i -> zeros(Float32, N^3), 6)
        particle_names = ("particle_px", "particle_py", "particle_pz",
                          "particle_vx", "particle_vy", "particle_vz")
        for (i, name) in enumerate(particle_names)
            _write_f32_array_atomic(joinpath(dir, "$name.f32"),
                                    fill(Float32(i) / 10, N^3))
        end
        chemistry = ntuple(i -> zeros(Float32, N^3), 3)
        for (i, name) in enumerate(("chem_eint", "chem_HII", "chem_H2I"))
            _write_f32_array_atomic(joinpath(dir, "$name.f32"),
                                    fill(Float32(i) / 100, N^3))
        end
        photon = MHDKernels.allocate_photon_moment_state(state)
        photon_expected = reshape(ComplexF32.(1:length(photon.hat_batch),
                                              -1:-1:-length(photon.hat_batch)),
                                  size(photon.hat_batch))
        open(joinpath(dir, "photon_hat_batch.c32"), "w") do io
            write(io, photon_expected)
        end

        _import_initial_aux_state!(dir; N=N, particles=particles,
                                   chemistry=chemistry, photons=photon)
        for i in 1:6
            @test particles[i] == fill(Float32(i) / 10, N^3)
        end
        for i in 1:3
            @test chemistry[i] == fill(Float32(i) / 100, N^3)
        end
        _write_f32_array_atomic(joinpath(dir, "chem_HeII.f32"),
                                fill(0.04f0, N^3))
        chemistry_helium = ntuple(i -> zeros(Float32, N^3), 4)
        _import_initial_aux_state!(dir; N=N, particles=particles,
                                   chemistry=chemistry_helium, photons=photon)
        for i in 1:4
            @test chemistry_helium[i] == fill(Float32(i) / 100, N^3)
        end
        @test Array(photon.hat_batch) == photon_expected
    end
end

@testset "cosmological helium thermodynamic closure" begin
    h = 0.674
    Ob = 0.05366655
    fh = 0.7546
    vunit = 8.0e5
    @test _PMF_T_CMB == ChemistryKernels.comp2_cmb(0.0)
    for z in (1.0e6, 3200.0, 2999.0)
        helium = _equilibrium_helium_state(1.0, z, h, Ob, fh)
        @test 1 <= helium.xe_total <= 1 + 2(1 - fh) / (4fh)
        @test 0 <= helium.xheii <= (1 - fh) / (4fh)
        @test 0 <= helium.heii_mass_fraction <= 1 - fh
        @test _drag_electron_fraction(1.0, helium.xheii, z, h, Ob, fh, true) ≈
              helium.xe_total rtol=2e-10

        temperature = _PMF_T_CMB * (1 + z)
        particles = fh * (1 + helium.xe_total) + (1 - fh) / 4
        expected = particles * ChemistryKernels.KBOLTZ * temperature /
                   (ChemistryKernels.MH * vunit^2)
        @test _cosmological_ionized_p0(z, fh, vunit, h, Ob) ≈ expected rtol=2e-14
    end

    # The device diagnostic receives dimensionless code density.  Its Saha
    # reconstruction must use the physical density unit or He III electrons
    # are lost at high redshift and the inferred temperature is biased high.
    N = 4
    z = 1.0e6
    be = MHDKernels.backend(:cpu)
    state = MHDKernels.allocate_state(
        be, Float32, (N, N, N); dx=1 / N, gamma=5 / 3,
    )
    helium = _equilibrium_helium_state(1.0, z, h, Ob, fh)
    fill!(state.U[1], 1f0)
    foreach(field -> fill!(field, 0f0), state.U[2:4])
    foreach(field -> fill!(field, 0f0), state.U[6:8])
    fill!(state.U[5], Float32(_cosmological_ionized_p0(z, fh, vunit, h, Ob) /
                              (state.γ - 1)))
    HII = fill(Float32(fh), N^3)
    H2I = zeros(Float32, N^3)
    HeII = fill(Float32(helium.heii_mass_fraction), N^3)
    stats = _device_temperature_stats!(
        be, state, state.scratch[1], HII, H2I, HeII,
        _density_unit_cgs(z, h, Ob), vunit^2, z, fh,
    )
    @test stats[1] ≈ _PMF_T_CMB * (1 + z) rtol=2e-5
    @test stats[2] ≈ stats[1] rtol=2e-6
    @test stats[3] ≈ stats[1] rtol=2e-6
end
