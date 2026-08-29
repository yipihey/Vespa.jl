using MHDKernels, Test

@testset "bounded passive species transport" begin
    backend_name = Symbol(get(ENV, "MHD_TEST_BACKEND", "cpu"))
    be = backend(backend_name)
    precisions = backend_name === :metal ? (Float32,) : (Float32, Float64)
    for T in precisions
        n = 16
        s = allocate_state(be, T, (n, n, n); dx=inv(T(n)))
        rho = reshape(T[1 + 0.2sin(2pi * (i - 1) / n) for
                        i in 1:n, j in 1:n, k in 1:n], :)
        copyto!(s.U[1], rho)
        s.U[2] .= T(0.2) .* s.U[1]
        fill!(s.U[3], zero(T))
        fill!(s.U[4], zero(T))
        q1 = T(0.3)
        q2 = T(1.0e-4)
        species1 = copy(s.U[1])
        species2 = copy(s.U[1])
        helium = copy(s.U[1])
        species1 .*= q1
        species2 .*= q2
        helium .*= T(0.08)
        conserved_species_to_fractions!(
            species1, species2, s.U[1]; maximum_total=T(0.76)
        )
        advect_species_fractions!(
            species1, species2, s.scratch[5], s.scratch[9], s, T(0.01);
            maximum_total=T(0.76),
        )
        conserved_species_to_fraction!(helium, s.U[1]; maximum_total=T(0.24))
        advect_species_fraction!(helium, s.scratch[4], s, T(0.01);
                                 maximum_total=T(0.24))
        result1 = to_host(species1) ./ to_host(s.U[1])
        result2 = to_host(species2) ./ to_host(s.U[1])
        tolerance = T === Float32 ? 2e-7 : 2e-14
        @test maximum(abs.(result1 .- q1)) < tolerance
        @test maximum(abs.(result2 .- q2)) < tolerance
        @test maximum(abs.(to_host(helium) ./ to_host(s.U[1]) .- T(0.08))) < tolerance
    end

    n = 32
    T = backend_name === :metal ? Float32 : Float64
    s = allocate_state(be, T, (n, n, n); dx=inv(T(n)))
    fill!(s.U[1], one(T))
    fill!(s.U[2], T(0.25))
    fill!(s.U[3], zero(T))
    fill!(s.U[4], zero(T))
    species1 = to_device(be, reshape(T[0.2 + 0.1sin(2pi * (i - 1) / n) for
                                      i in 1:n, j in 1:n, k in 1:n], :), T)
    species2 = to_device(be, fill(T(1.0e-3), n^3), T)
    conserved_species_to_fractions!(species1, species2, s.U[1])
    dt = T(0.5) / n
    advect_species_fractions!(
        species1, species2, s.scratch[5], s.scratch[9], s, dt
    )
    expected = T[0.2 + 0.1sin(2pi * ((i - 1) - 0.125) / n) for
                 i in 1:n, j in 1:n, k in 1:n]
    @test maximum(abs.(to_host(species1) .- vec(expected))) < 6e-4
    @test minimum(species1) >= 0
    @test maximum(species1 .+ species2) <= 1

    centered = to_device(be, reshape(T[-3e-8 + 2e-8sin(2pi * (i - 1) / n) for
                                      i in 1:n, j in 1:n, k in 1:n], :), T)
    advect_passive_scalar!(centered, s.scratch[4], s, dt)
    centered_expected = T[-3e-8 + 2e-8sin(2pi * ((i - 1) - 0.125) / n) for
                          i in 1:n, j in 1:n, k in 1:n]
    centered_host = to_host(centered)
    @test minimum(centered_host) < 0
    @test maximum(centered_host) > -3e-8
    @test maximum(abs.(centered_host .- vec(centered_expected))) < 2e-10

    function compressible_transport_drift(n)
        Tcomp = backend_name === :metal ? Float32 : Float64
        s = allocate_state(
            be, Tcomp, (n, n, n); dx=1 / n, gamma=5 / 3,
            recon=:ppm, riemann=:hlld,
        )
        rho_host = zeros(Tcomp, n^3)
        mx_host = zeros(Tcomp, n^3)
        energy_host = zeros(Tcomp, n^3)
        species1_host = zeros(Tcomp, n^3)
        species2_host = zeros(Tcomp, n^3)
        @inbounds for k in 1:n, j in 1:n, i in 1:n
            c = i + n * ((j - 1) + n * (k - 1))
            x = 2pi * (i - 0.5) / n
            rho = 1 + 0.08cos(x)
            vx = 0.04sin(x)
            q1 = 0.2 + 0.03sin(x + 0.4)
            q2 = 1e-3 * (1 + 0.2cos(x - 0.2))
            rho_host[c] = rho
            mx_host[c] = rho * vx
            energy_host[c] = one(Tcomp) / (s.γ - one(Tcomp)) +
                             Tcomp(0.5) * rho * vx^2
            species1_host[c] = rho * q1
            species2_host[c] = rho * q2
        end
        copyto!(s.U[1], rho_host)
        copyto!(s.U[2], mx_host)
        copyto!(s.U[5], energy_host)
        species1 = to_device(be, species1_host, Tcomp)
        species2 = to_device(be, species2_host, Tcomp)
        before = (sum(species1), sum(species2))
        conserved_species_to_fractions!(species1, species2, s.U[1])
        dt = 0.1 / n
        step!(s, dt; ch=1.5, glm_cr=1,
              integrator=backend_name === :metal ? :cube : :ref)
        advect_species_fractions!(
            species1, species2, s.scratch[5], s.scratch[9], s, dt
        )
        after = (sum(species1), sum(species2))
        fractions = to_host(species1) ./ to_host(s.U[1])
        drift = maximum(abs(after[i] / before[i] - 1) for i in 1:2)
        drift, extrema(fractions)
    end

    drift16, bounds16 = compressible_transport_drift(16)
    drift32, bounds32 = compressible_transport_drift(32)
    @test drift32 < drift16
    @test drift32 < (backend_name === :metal ? 3e-5 : 2e-5)
    @test 0.16 < bounds16[1] < bounds16[2] < 0.24
    @test 0.16 < bounds32[1] < bounds32[2] < 0.24
end
