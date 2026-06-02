# Phase C1 — Enzo-compatible cosmology units + background expansion a(t), with NO
# hydro/gravity coupling. Oracles pin the unit system and the Friedmann integrator
# against *independent* physics / closed-form solutions (not a re-statement of the
# transcribed formulas):
#   U1  DensityUnits / LengthUnits vs first-principles ρ̄=3Ω_mH₀²/8πG, comoving box
#   U2  GravitationalConstant_code = 4πG·ρunit·tunit² ≈ 1 (the "4πG=1" normalization)
#   U3  redshift scalings of the unit factors
#   E1  EdS: a(t) = (3/2)^{1/3} t^{2/3}, t(a=1) = √(2/3)
#   E2  flat ΛCDM: a(t) = (Ω_m/Ω_Λ)^{1/3} sinh^{2/3}(√(3Ω_Λ/2Ω_m)·t)
#   E3  a(z_i)=1 for z_i≠0; redshift round-trips and is monotonic

const G_CGS   = 6.67428e-8
const MPC     = 3.0857e24
const H0_CGS_PER_h = 100.0 * 1.0e5 / MPC          # 100 km/s/Mpc in s⁻¹, per unit h

@testset "U1: density/length units vs first-principles physics" begin
    # DensityUnits must equal the mean matter density ρ̄ = 3Ω_mH₀²/(8πG)·(1+z)³,
    # computed from CGS constants — an independent check of the 1.8788e-29 literal.
    c = Cosmology(; OmegaMatter = 0.3, OmegaLambda = 0.7, HubbleConstantNow = 0.7,
                  ComovingBoxSize = 64.0, InitialRedshift = 20.0, FinalRedshift = 0.0)
    for z in (20.0, 5.0, 0.0)
        a = scale_factor_at_redshift(c, z)
        u = cosmology_units(c, a)
        H0 = c.HubbleConstantNow * H0_CGS_PER_h
        ρbar = 3 * c.OmegaMatter * H0^2 / (8π * G_CGS) * (1 + z)^3
        @test isapprox(u.density, ρbar; rtol = 2e-3)
        # LengthUnits = proper size of the comoving box / (1+z).
        Lcomoving_cm = MPC * c.ComovingBoxSize / c.HubbleConstantNow
        @test isapprox(u.length, Lcomoving_cm / (1 + z); rtol = 1e-12)
    end
end

@testset "U2: GravitationalConstant_code ≈ 1 (4πG=1 normalization)" begin
    for (Ωm, h, box, zi) in ((1.0, 0.5, 1.0, 0.0), (0.3, 0.7, 64.0, 20.0),
                             (0.27, 0.71, 8.0, 99.0))
        c = Cosmology(; OmegaMatter = Ωm, OmegaLambda = 1 - Ωm, HubbleConstantNow = h,
                      ComovingBoxSize = box, InitialRedshift = zi, FinalRedshift = 0.0)
        @test isapprox(gravitational_constant_code(c), 1.0; rtol = 2e-3)
    end
end

@testset "U3: redshift scalings of the unit factors" begin
    c = Cosmology(; OmegaMatter = 0.3, OmegaLambda = 0.7, HubbleConstantNow = 0.7,
                  ComovingBoxSize = 10.0, InitialRedshift = 10.0, FinalRedshift = 0.0)
    a1 = scale_factor_at_redshift(c, 10.0); a2 = scale_factor_at_redshift(c, 0.0)
    u1 = cosmology_units(c, a1); u2 = cosmology_units(c, a2)
    # density ∝ (1+z)³, length ∝ 1/(1+z); velocity/time/temperature use z_i (fixed).
    @test isapprox(u1.density / u2.density, (11.0 / 1.0)^3; rtol = 1e-12)
    @test isapprox(u1.length / u2.length, 1.0 / 11.0; rtol = 1e-12)
    @test u1.velocity == u2.velocity
    @test u1.time == u2.time
    @test u1.temperature == u2.temperature
end

@testset "E1: EdS expansion a(t) = (3/2)^{1/3} t^{2/3}" begin
    c = Cosmology(; OmegaMatter = 1.0, OmegaLambda = 0.0, HubbleConstantNow = 0.5,
                  ComovingBoxSize = 1.0, InitialRedshift = 0.0, FinalRedshift = 0.0)
    @test isapprox(c.t_initial, sqrt(2 / 3); rtol = 1e-6)          # t(a=1) = √(2/3)
    for t in (0.4, 0.8165, 1.5, 4.0)
        a, dadt = expansion_at(c, t)
        @test isapprox(a, (1.5)^(1 / 3) * t^(2 / 3); rtol = 1e-6)
        @test isapprox(dadt, sqrt(2 / 3) / sqrt(a); rtol = 1e-6)   # da/dt = √(2/3)·a^{-1/2}
    end
end

@testset "E2: flat ΛCDM a(t) closed form" begin
    Ωm = 0.3; ΩΛ = 0.7
    c = Cosmology(; OmegaMatter = Ωm, OmegaLambda = ΩΛ, HubbleConstantNow = 0.7,
                  ComovingBoxSize = 1.0, InitialRedshift = 0.0, FinalRedshift = -0.9)
    A = (Ωm / ΩΛ)^(1 / 3); C = sqrt(3 * ΩΛ / (2 * Ωm))
    for t in (0.3, 0.8, c.t_initial, 1.5, 2.2)              # a stays within a physical range
        a, _ = expansion_at(c, t)
        @test isapprox(a, A * sinh(C * t)^(2 / 3); rtol = 1e-5)
    end
end

@testset "E3: a(z_i)=1, redshift round-trip + monotonic" begin
    c = Cosmology(; OmegaMatter = 0.3, OmegaLambda = 0.7, HubbleConstantNow = 0.7,
                  ComovingBoxSize = 64.0, InitialRedshift = 24.0, FinalRedshift = 0.0)
    @test isapprox(c.a, 1.0; rtol = 1e-10)                         # cached a at z_i
    @test isapprox(redshift(c), 24.0; rtol = 1e-9)
    tprev = -Inf; aprev = -Inf
    for z in (24.0, 10.0, 3.0, 1.0, 0.0)
        a = scale_factor_at_redshift(c, z)
        t = time_from_scale_factor(c, a)
        ar, _ = expansion_at(c, t)                                 # invert back
        @test isapprox(ar, a; rtol = 1e-8)
        @test t > tprev && a > aprev                               # both increase with cosmic time
        tprev = t; aprev = a
    end
    # tfinal span (z_i → z_f) is positive and lands at z_f.
    @test cosmology_tfinal(c) > 0
    af, _ = expansion_at(c, c.t_initial + cosmology_tfinal(c))
    @test isapprox(redshift_at_scale_factor(c, af), 0.0; atol = 1e-6)
end
