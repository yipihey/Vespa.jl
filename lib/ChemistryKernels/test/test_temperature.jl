using ChemistryKernels, KernelAbstractions, Test
try; @eval using Metal; catch; end
include(joinpath(@__DIR__, "oracle.jl")); using .ChemOracle
include(joinpath(@__DIR__, "harness.jl"))

# Reduced-network temperature: ChemistryKernels.temperature_from_reduced (via the
# temperature_grid launcher) vs grackle's calculate_temperature (chem_temperature
# oracle, density_units=length_units=time_units=1 ⇒ code units ≡ physical CGS).
@testset "temperature vs grackle" begin
    # Use a realistic density_units (cosmological, ~ρ̄_b) with length_units =
    # time_units = 1 (⇒ velocity_units = 1, so e_code ≡ e_cgs and the oracle's
    # mass-density-based `number_density` floors `tiny`≈1e-20 stay negligible in
    # CODE units exactly as in production — at density_units=1 those absolute
    # floors would swamp realistically small ρ and corrupt grackle's mmw).  T is
    # independent of density_units (P/nd cancels it), so we feed the oracle code
    # densities ρ_cgs/DU and compare to the Julia kernel in physical CGS.
    DU = 1.0e-24
    rc = ChemOracle.temperature_init!(; a_value = 1.0/(1+200.0), fh = 0.76,
                                      density_units = DU, length_units = 1.0,
                                      time_units = 1.0)
    @test rc == 1

    # Sweep (rho, eint, x_HII, x_H2) so the run spans the H2 variable-γ branch:
    #   x_H2 up to 0.1 (nH2/n_other ≫ 1e-3) and a range of eint that drives T
    #   through the x = 6100/T < 10 (T ≳ 610 K) regime where γ2 departs from 5/3.
    rhos  = [1.0e-26, 1.0e-24, 1.0e-22, 1.0e-20]      # g/cm³ (physical CGS)
    eints = [1.0e8, 1.0e9, 1.0e10, 1.0e11, 1.0e12, 1.0e13]   # erg/g
    xHIIs = [1.0e-4, 1.0e-2, 1.0e-1]
    xH2s  = [0.0, 1.0e-3, 1.0e-2, 1.0e-1]
    fh    = 0.76

    rho = Float64[]; eint = Float64[]; HII = Float64[]; H2I = Float64[]
    for r in rhos, e in eints, xi in xHIIs, x2 in xH2s
        # keep HI = fh·ρ − HII − H2I positive
        (xi + x2) < 0.9 || continue
        push!(rho, r); push!(eint, e)
        push!(HII, xi * fh * r); push!(H2I, x2 * fh * r)
    end
    # oracle takes code-unit densities (ρ/DU); e_code ≡ e_cgs since v_units = 1.
    ref = ChemOracle.temperature(rho ./ DU, eint, HII ./ DU, H2I ./ DU;
                                 a_value = 1.0/(1+200.0))

    run(name, T) = ChemistryKernels.temperature_grid(name, T, rho, eint, HII, H2I; fh = fh)
    layerA!("temperature", run(:cpu, Float64), ref)

    if metal_ready()
        # f32 floor: the γ-correction's exp/x²-ratio is mild; 1e-4 is comfortable.
        tol = (rtol = 1.0e-4, atol = 1.0e-3)
        cpuf = run(:cpu, Float32); gpuf = run(:metal, Float32)
        @check("temperature [B:cpu≡metal f32]", gpuf, cpuf, tol)
        @check("temperature [C:metal-f32 vs grackle]", gpuf, ref, tol)
    end
end
