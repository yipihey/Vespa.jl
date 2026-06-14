using ChemistryKernels, Test
using ChemistryKernels: cooling_edot, ceHI, ciHI, reHII, brem,
    GAHI, GAH2, GAHe, GAHp, GAel, H2LTE, HDlte, HDlow, comp1_cmb, comp2_cmb, MH, TINY
include(joinpath(@__DIR__, "oracle.jl")); using .ChemOracle

@testset "edot: term-by-term transcription" begin
    # cooling_edot must equal an INDEPENDENT re-assembly from the (already
    # oracle-verified) Wave-1 coefficients — catches a wrong coefficient/species
    # /sign in the assembly. Pure arithmetic ⇒ 1e-12.
    nHI, nHII, nde = 0.6, 0.05, 0.05
    nHeI, nH2, nHD = 0.06, 1.0e-3, 1.0e-6
    T, z = 3000.0, 30.0
    Tc = comp2_cmb(z)
    atomic = (ceHI(T)+ciHI(T))*nHI*nde + reHII(T)*nHII*nde + brem(T)*nHII*nde
    galdl = GAHI(T)*nHI + GAH2(T)*nH2 + GAHe(T)*nHeI + GAHp(T)*nHII + GAel(T)*nde
    cool_gas = H2LTE(T)/(1 + H2LTE(T)/galdl)
    galdl_c = GAHI(Tc)*nHI + GAH2(Tc)*nH2 + GAHe(Tc)*nHeI + GAHp(Tc)*nHII + GAel(Tc)*nde
    cool_cmb = H2LTE(Tc)/(1 + H2LTE(Tc)/galdl_c)
    h2 = nH2*(cool_gas - cool_cmb)
    hd = T > Tc ? nHD*HDlte(T)/(1 + (HDlte(T)/nHI)/max(HDlow(T),TINY)) : 0.0
    compton = comp1_cmb(z)*(T - Tc)*nde
    ref = -(atomic + h2 + hd + compton)
    @test isapprox(cooling_edot(nHI,nHII,nHeI,nde,nH2,nHD,T,z), ref; rtol=1e-12)

    # Compton-only sanity: hot dilute fully-ionized gas at high z (H2/HD/atomic
    # negligible) ⇒ edot ≈ −comp1·(T−Tc)·n_e and < 0 (cooling).
    e_hot = cooling_edot(1e-30, 0.1, 1e-30, 0.1, 1e-30, 0.0, 1.0e4, 1000.0)
    @test e_hot < 0
    @test isapprox(e_hot, -comp1_cmb(1000.0)*(1.0e4 - comp2_cmb(1000.0))*0.1; rtol=1e-3)
end

@testset "edot: assembled vs grackle cooling_time" begin
    # Sanity-check the H2/HD/Compton assembly (the dom-factor bookkeeping that the
    # term-by-term test can't see) against grackle's calculate_cooling_time.
    # ė_vol = ρ·e/t_cool [erg cm⁻³ s⁻¹]. Tolerance is loose (~3%): grackle here
    # uses interpolated rates and includes minor channels we omit at these
    # densities (CIE, H2-formation heating) — the tight gate is the Wave-5 one-zone.
    DU, fh = 1.0e-24, 0.76
    rc = ChemOracle.temperature_init!(; a_value=1.0/31, fh=fh,
                                      density_units=DU, length_units=1.0, time_units=1.0)
    @test rc == 1

    # (rho_code, eint[erg/g], xHII, xH2, z) — Compton-, atomic-, and H2-dominated.
    # H2 case must sit ABOVE T_cmb (T≈3000 K here) so it's genuine line cooling,
    # not the T<T_cmb recombination/CMB-heating regime (where grackle's CMB-
    # recombination correction, outside this assembly, dominates).
    cases = [(1.0, 5.0e11, 0.9, 0.0, 1000.0),    # high-z, hot, ionized: Compton
             (1.0, 2.0e12, 0.5, 0.0, 50.0),      # T~few×10⁴: atomic Lyα + brem
             (1.0, 3.0e11, 1.0e-4, 0.1, 20.0)]   # warm molecular (T≈3000): H2 line
    for (rc_, e, xHII, xH2, z) in cases
        a = 1.0/(1+z)
        rho_code = [rc_]; eint = [e]
        HII = [xHII*fh*rc_]; H2I = [xH2*fh*rc_]
        T  = ChemOracle.temperature(rho_code, eint, HII, H2I; a_value=a)[1]
        tc = ChemOracle.cooling_time(rho_code, eint, HII, H2I; a_value=a)[1]
        rho_cgs = rc_*DU
        edot_oracle = rho_cgs * e / tc          # e_cgs ≡ e (v_units=1)

        nHII = xHII*fh*rc_*DU/MH
        nH2  = xH2*fh*rc_*DU/(2*MH)
        nHI  = max((fh*rc_ - xHII*fh*rc_ - xH2*fh*rc_)*DU/MH, TINY)
        nHeI = (1-fh)*rho_cgs/(4*MH)
        edot_jl = cooling_edot(nHI, nHII, nHeI, nHII, nH2, 0.0, T, z)

        @test isapprox(edot_jl, edot_oracle; rtol=3e-2)
    end
end
