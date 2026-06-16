using ChemistryKernels, KernelAbstractions, Test
try; @eval using Metal; catch; end
include(joinpath(@__DIR__, "harness.jl"))

# Load rates_cmb and cooling_compton in an isolated module so the include is
# self-contained (same pattern as test_rates_atomic.jl / test_rates_h2.jl).
module UnitCMB
  using ChemistryKernels, KernelAbstractions
  using ChemistryKernels: MH, TINY, TEV_PER_K
  include(joinpath(@__DIR__, "..", "src", "rates_cmb.jl"))
  include(joinpath(@__DIR__, "..", "src", "cooling_compton.jl"))
end

@testset "rates_cmb literal" begin
    # Verify k27, k28, and comp2 against the literal formulas from
    # solve_chemistry.c:158-160 and cool1d_multi_g.F:198-199.
    # There is no grackle oracle for these (they live in solve_chemistry.c, not
    # rate_functions.c), so the reference IS the literal formula.
    for z in (10.0, 50.0, 100.0, 300.0, 1000.0, 1100.0)
        Trad = 2.73 * (1 + z)
        # k27: H- + γ_CMB → H + e  (solve_chemistry.c:158)
        @test isapprox(UnitCMB.k27_cmb(Trad), 1.1e-1 * Trad^2.13 * exp(-8823.0 / Trad); rtol=1e-12)
        # k28: H2+ + γ_CMB → H + H+  (solve_chemistry.c:160)
        @test isapprox(UnitCMB.k28_cmb(Trad), 1.63e7 * exp(-32400.0 / Trad); rtol=1e-12)
        # comp2: CMB temperature (Fixsen 2009: T_CMB,0 = 2.725 K; grackle
        # cool1d_multi_g.F uses 2.73 — we use the physical value for CAMB consistency)
        @test isapprox(UnitCMB.comp2_cmb(z), 2.725 * (1 + z); rtol=1e-12)
    end

    # comp1: Compton coupling coefficient scaling (cool1d_multi_g.F:198)
    for z in (10.0, 50.0, 300.0, 1000.0)
        @test isapprox(UnitCMB.comp1_cmb(z), UnitCMB.COMPA * (1.0 + z)^4; rtol=1e-12)
    end
    # Verify the COMPA constant itself matches the grackle value (rate_functions.c:1312)
    @test UnitCMB.COMPA == 5.65e-36

    # Metal f32 parity for k27 over a Trad grid (skip if no Metal backend).
    # Use a floor of 1e-30 in the denominator so that both-zero entries
    # (k27 underflows to 0f0 at the lowest z) give 0/1e-30 = 0, not 0/0 = NaN.
    if ChemistryKernels.has_backend(:metal)
        Ts = collect(2.73 .* (1.0 .+ (10.0:50.0:1100.0)))
        cpu = UnitCMB.k27_cmb_grid(:cpu,   Float32, Ts)
        gpu = UnitCMB.k27_cmb_grid(:metal, Float32, Ts)
        maxrel = maximum(@. abs(gpu - cpu) / max(abs(cpu), Float32(1e-30)))
        @test maxrel < 1e-5
    end
end
