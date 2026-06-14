# equilibrium.jl — algebraic equilibrium intermediaries of the v2026 reduced
# network: H⁻ (HM), H2⁺ (H2II) and D⁺ (DII).  In the reduced model these three
# species are NOT advected; each is held at the steady state of its own fast
# formation/destruction balance (grackle flags equilibrium_h2_intermediates and
# equilibrium_deuterium), so the host carries only HII, H2I and HDI.
#
# Direct transcription of solve_rate_cool_g.F:2450-2525 (the `ieq==1` / `ideut==1`
# branches).  SPECIES CONVENTION = grackle's network densities, i.e. the same
# mass-equivalent code units the rate network uses: yHI=n_HI, yHII=n_HII, yde=n_e,
# yHM=n_HM, yHeI=n_HeI, and the molecular fields carry a factor of 2/3 so that
#   yH2I  = 2·n(H2 molecule),  yH2II = 2·n(H2⁺),  yHDI = 3·n(HD)
# — hence the literal `/2` and `/3` factors below match the Fortran verbatim.  The
# overall scale cancels in each equilibrium RATIO, so the same functions work on
# physical CGS number densities provided the molecular fields keep that 2×/3×
# convention (boundary.jl supplies them that way).  Pure & allocation-free.
#
# Photo-rates: k27 (H⁻+γ) and k28 (H2⁺+γ) are the CMB rates (rates_cmb.jl); the
# Lyman–Werner / UV terms k29,k30 (H2 dissociation) and k24 (D photo-ion.) are
# zero in the no-radiation primordial network and default to 0.

export equilibrium_HM, equilibrium_H2II, equilibrium_DII

# A scale-free tiny floor mirroring the Fortran `tiny` guard on the denominator.
const _EQ_TINY = 1.0e-20

"""
    equilibrium_HM(yHI, yHII, yde, yH2II, k7, k8, k14, k15, k16, k17, k19, k27)

H⁻ equilibrium abundance: `HM = (k7·HI·de) / Σ(destruction)` with destruction by
k8/k15 (HI), k16/k17 (HII), k14 (e), k19 (H2⁺/2) and the CMB photodetachment k27.
(solve_rate_cool_g.F:2450-2459.) Pure.
"""
@inline function equilibrium_HM(yHI, yHII, yde, yH2II,
                                k7, k8, k14, k15, k16, k17, k19, k27)
    R = typeof(yHI)
    scoef = k7 * yHI * yde
    acoef = (k8 + k15) * yHI + (k16 + k17) * yHII +
            k14 * yde + k19 * yH2II / R(2) + k27
    return scoef / max(acoef, R(_EQ_TINY))
end

"""
    equilibrium_H2II(yHI, yHII, yH2I, yde, yHM, k9, k10, k11, k17, k18, k19, k28;
                     k29 = 0, k30 = 0)

H2⁺ equilibrium abundance (solve_rate_cool_g.F:2465-2475):
`H2II = 2·(k9·HI·HII + k11·H2I/2·HII + k17·HM·HII + k29·H2I) /
        (k10·HI + k18·de + k19·HM + k28 + k30)`.
k28 is the CMB photodissociation; k29 (LW, numerator) and k30 default to 0. Pure.
"""
@inline function equilibrium_H2II(yHI, yHII, yH2I, yde, yHM,
                                  k9, k10, k11, k17, k18, k19, k28;
                                  k29 = zero(typeof(yHI)),
                                  k30 = zero(typeof(yHI)))
    R = typeof(yHI)
    num = R(2) * (k9 * yHI * yHII + k11 * yH2I / R(2) * yHII +
                  k17 * yHM * yHII + k29 * yH2I)
    den = k10 * yHI + k18 * yde + k19 * yHM + (k28 + k30)
    return num / max(den, R(_EQ_TINY))
end

"""
    equilibrium_DII(yDI, yde, yHI, yHII, yH2I, yHDI, k1, k2, k50, k51, k52, k53;
                    k24 = 0)

D⁺ equilibrium abundance (solve_rate_cool_g.F:2509-2520):
`DII = (k1·DI·de + k50·HII·DI + 2·k53·HII·HDI/3 + k24·DI) /
       (k2·de + k51·HI + k52·H2I/2)`.
k24 is D photo-ionization (0 without radiation). Pure.
"""
@inline function equilibrium_DII(yDI, yde, yHI, yHII, yH2I, yHDI,
                                 k1, k2, k50, k51, k52, k53;
                                 k24 = zero(typeof(yDI)))
    R = typeof(yDI)
    scoef = k1 * yDI * yde + k50 * yHII * yDI +
            R(2) * k53 * yHII * yHDI / R(3) + k24 * yDI
    acoef = k2 * yde + k51 * yHI + k52 * yH2I / R(2)
    return scoef / max(acoef, R(_EQ_TINY))
end
