using Test

@testset "nonlocal Lyman-alpha recombination transport" begin
    N = 16
    backend_name = Symbol(get(ENV, "MHD_TEST_BACKEND", "cpu"))
    be = backend(backend_name)
    s = allocate_state(be, Float32, (N, N, N); dx=1 / N)
    fh = 0.76f0
    xe_mean = 0.1f0
    density_amplitude = 0.01f0
    ionization_amplitude = 0.005f0
    velocity_amplitude = 0.02f0
    rho_host = zeros(Float32, N^3)
    mx_host = zeros(Float32, N^3)
    hii_host = zeros(Float32, N^3)
    @inbounds for k in 1:N, j in 1:N, i in 1:N
        c = i + N * ((j - 1) + N * (k - 1))
        phase = 2f0 * Float32(pi) * (i - 1) / N
        rho = 1f0 + density_amplitude * cos(phase)
        rho_host[c] = rho
        mx_host[c] = rho * velocity_amplitude * sin(phase)
        hii_host[c] = fh * rho * (xe_mean + ionization_amplitude * cos(phase))
    end
    copyto!(s.U[1], rho_host)
    copyto!(s.U[2], mx_host)
    HII = to_device(be, hii_host, Float32)
    H2I = device_zeros(be, Float32, (N^3,))
    workspace = allocate_radiation_workspace(s)
    redshifts = expm1.(range(log1p(500.0), log1p(2000.0), length=3))
    wavenumbers = exp.(range(log(1.0), log(100.0), length=3))
    density_rate = fill(-2.0e-12, 3, 3)
    ionization_rate = fill(-3.0e-12, 3, 3)
    velocity_rate = fill(4.0e-12, 3, 3)
    table = NonlocalRecombinationTable(be, redshifts, wavenumbers,
        density_rate, ionization_rate, velocity_rate; precision=Float32)
    dt_seconds = 1.0e8
    applied = apply_nonlocal_recombination!(HII, H2I, s, workspace, table;
        redshift=redshifts[2], xe_mean=xe_mean, dt_seconds=dt_seconds,
        fundamental_k_mpc=10.0, theta_over_aH_conversion=1.0,
        hydrogen_mass_fraction=fh)
    @test applied
    q = -3.0e-12 * dt_seconds
    phi = expm1(q) / -3.0e-12
    theta_amplitude = velocity_amplitude * sinpi(2 / N) / (1 / N)
    expected_amplitude = exp(q) * ionization_amplitude + phi * (
        -2.0e-12 * density_amplitude + 4.0e-12 * theta_amplitude)
    xe = to_host(HII) ./ (fh .* rho_host)
    expected = Float32[
        xe_mean + expected_amplitude * cospi(2f0 * (i - 1) / N)
        for k in 1:N for j in 1:N for i in 1:N
    ]
    @test maximum(abs, xe .- expected) < 8f-7
    @test abs(sum(xe) / length(xe) - xe_mean) < 2f-7

    before = to_host(HII)
    @test !apply_nonlocal_recombination!(HII, H2I, s, workspace, table;
        redshift=3000.0, xe_mean=xe_mean, dt_seconds=dt_seconds,
        fundamental_k_mpc=10.0, theta_over_aH_conversion=0.0,
        hydrogen_mass_fraction=fh)
    @test to_host(HII) == before

    # Below the first tabulated k, the differential response must vanish as k^2.
    hii_lowk_host = similar(hii_host)
    @inbounds for k in 1:N, j in 1:N, i in 1:N
        c = i + N * ((j - 1) + N * (k - 1))
        phase = 2f0 * Float32(pi) * (i - 1) / N
        hii_lowk_host[c] = fh * rho_host[c] *
                           (xe_mean + ionization_amplitude * cos(phase))
    end
    HII_lowk = to_device(be, hii_lowk_host, Float32)
    low_scale = (0.1 / first(wavenumbers))^2
    @test apply_nonlocal_recombination!(HII_lowk, H2I, s, workspace, table;
        redshift=redshifts[2], xe_mean=xe_mean, dt_seconds=dt_seconds,
        fundamental_k_mpc=0.1, theta_over_aH_conversion=1.0,
        hydrogen_mass_fraction=fh)
    q_low = low_scale * -3.0e-12 * dt_seconds
    phi_low = expm1(q_low) / (low_scale * -3.0e-12)
    expected_low_amplitude = exp(q_low) * ionization_amplitude + phi_low * low_scale * (
        -2.0e-12 * density_amplitude + 4.0e-12 * theta_amplitude)
    xe_low = to_host(HII_lowk) ./ (fh .* rho_host)
    expected_low = Float32[
        xe_mean + expected_low_amplitude * cospi(2f0 * (i - 1) / N)
        for k in 1:N for j in 1:N for i in 1:N
    ]
    @test maximum(abs, xe_low .- expected_low) < 8f-7

    # The fourth real-field slot carries delta(T_b)/T_b through the same batched
    # transform, so temperature response adds no FFT or full-grid workspace.
    s_temperature = allocate_state(be, Float32, (N, N, N); dx=1 / N)
    temperature_reference = 3000f0
    temperature_amplitude = 0.003f0
    velocity_unit2 = 1.0f12
    temperature_rate = fill(5.0e-12, 3, 3)
    table_temperature = NonlocalRecombinationTable(
        be, redshifts, wavenumbers, density_rate, ionization_rate, velocity_rate;
        temperature_rate=temperature_rate, precision=Float32,
    )
    rho_temperature = ones(Float32, N^3)
    hii_temperature = fill(fh * xe_mean, N^3)
    energy_temperature = similar(rho_temperature)
    particles_per_mass_h = fh + fh * xe_mean + (1f0 - fh) / 4f0
    @inbounds for k in 1:N, j in 1:N, i in 1:N
        c = i + N * ((j - 1) + N * (k - 1))
        phase = 2f0 * Float32(pi) * (i - 1) / N
        temperature = temperature_reference *
                      (1f0 + temperature_amplitude * cos(phase))
        specific_e = temperature * 1.380649f-16 * particles_per_mass_h /
                     ((Float32(s_temperature.γ) - 1f0) * 1.6735575f-24 *
                      velocity_unit2)
        energy_temperature[c] = specific_e
    end
    copyto!(s_temperature.U[1], rho_temperature)
    copyto!(s_temperature.U[5], energy_temperature)
    HII_temperature = to_device(be, hii_temperature, Float32)
    H2I_temperature = device_zeros(be, Float32, (N^3,))
    workspace_temperature = allocate_radiation_workspace(s_temperature)
    @test apply_nonlocal_recombination!(
        HII_temperature, H2I_temperature, s_temperature, workspace_temperature,
        table_temperature; redshift=redshifts[2], xe_mean=xe_mean,
        dt_seconds=dt_seconds, fundamental_k_mpc=10.0,
        theta_over_aH_conversion=0.0, hydrogen_mass_fraction=fh,
        velocity_unit2=velocity_unit2,
        temperature_reference=temperature_reference,
    )
    phi_temperature = expm1(q) / -3.0e-12
    expected_temperature_amplitude = phi_temperature * 5.0e-12 *
                                     temperature_amplitude
    xe_temperature = to_host(HII_temperature) ./ fh
    expected_temperature = Float32[
        xe_mean + expected_temperature_amplitude * cospi(2f0 * (i - 1) / N)
        for k in 1:N for j in 1:N for i in 1:N
    ]
    @test maximum(abs, xe_temperature .- expected_temperature) < 8f-7
    @test table_temperature.has_temperature

    mktemp() do path, io
        println(io, join(("redshift", "k_mpc", "delta_density_rate_s",
                          "delta_ionization_rate_s",
                          "delta_velocity_divergence_rate_s"), '\t'))
        for z in redshifts, k in wavenumbers
            println(io, join((z, k, -2e-12, -3e-12, 4e-12), '\t'))
        end
        close(io)
        loaded = load_nonlocal_recombination_table(path, be; precision=Float32)
        @test loaded.nz == 3
        @test loaded.nk == 3
        @test isapprox(loaded.zlog_min, Float32(log1p(first(redshifts))); rtol=2f-6)
        @test isapprox(loaded.logk_max, Float32(log(last(wavenumbers))); rtol=2f-6)
        @test !loaded.has_temperature
    end

    mktemp() do path, io
        println(io, join(("redshift", "k_mpc", "delta_density_rate_s",
                          "delta_ionization_rate_s",
                          "delta_velocity_divergence_rate_s",
                          "delta_temperature_rate_s",
                          "local_velocity_divergence_rate_s"), '\t'))
        for z in redshifts, k in wavenumbers
            println(io, join((z, k, -2e-12, -3e-12, 4e-12, 5e-12, -1e-12), '\t'))
        end
        close(io)
        loaded = load_nonlocal_recombination_table(path, be; precision=Float32)
        @test loaded.has_temperature
        @test all(isapprox.(to_host(loaded.local_velocity_divergence_rate),
                            -1f-12; rtol=2f-6))
    end

    @test_throws ErrorException NonlocalRecombinationTable(
        be, redshifts, wavenumbers, density_rate, -ionization_rate,
        velocity_rate; precision=Float32)
end
