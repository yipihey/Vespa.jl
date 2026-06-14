# edot.jl — net radiative cooling/heating rate of the v2026 reduced network.
#
# Assembles the volumetric energy-change rate ė [erg cm⁻³ s⁻¹] (negative ⇒
# cooling) from the Wave-1 cooling coefficients and physical number densities,
# exactly as cool1d_multi_g.F builds `edot`, specialized to the reduced model:
#   helium forced neutral ⇒ every He cooling channel (ce/ci/re of He, and the
#   He terms of bremsstrahlung) is ∝ n_HeII or n_HeIII = tiny ≈ 0 and DROPS OUT,
#   leaving only HI/HII/e atomic cooling, H2, HD, and CMB-Compton.
#
# Written in physical CGS (number densities [cm⁻³], T [K]); grackle's `dom`/
# `coolunit` code-unit bookkeeping cancels in the physical form (the H2/HD
# two-level functions and Compton are textbook).  Pure & allocation-free.
#
# Reference lines (cool1d_multi_g.F):
#   atomic       417-447   (ceHI/ciHI·n_HI·n_e, reHII·n_HII·n_e, brem·n_HII·n_e)
#   H2 (GP2008)  490-549   (galdl low-density limit, H2LTE high-density, CMB floor)
#   HD           681-711   (two-level with collider n_HI, CMB-gated)
#   Compton      1053-1063 (comp1·(T−T_cmb)·n_e)

export cooling_edot

"""
    cooling_edot(nHI, nHII, nHeI, nde, nH2, nHD, T, z; ih2optical=false, nH=nothing)

Net volumetric energy rate ė [erg cm⁻³ s⁻¹] (cooling ⇒ negative) for the reduced
network at gas temperature `T` [K] and redshift `z`. Number densities are
physical [cm⁻³]; `nH2`/`nHD` are H2 and HD *molecule* densities. The H2 optical-
depth fudge (grackle's `h2_optical_depth_approximation`, default off) needs the
total H-nucleus density `nH`. Pure.
"""
@inline function cooling_edot(nHI, nHII, nHeI, nde, nH2, nHD, T, z;
                              ih2optical::Bool = false, nH = nothing)
    R   = typeof(T)
    one_ = one(R)
    Tc  = comp2_cmb(R(z))                          # T_cmb = 2.73(1+z)

    # ── atomic (He neutral ⇒ only HI/HII/e survive) ──────────────────────────
    atomic = (ceHI(T) + ciHI(T)) * nHI * nde +
             reHII(T) * nHII * nde +
             brem(T)  * nHII * nde

    # ── H2 (Galli-Palla 2008 two-level: 1/Λ = 1/Λ_LTE + 1/Λ_lowρ), CMB floor ─
    galdl = GAHI(T) * nHI + GAH2(T) * nH2 + GAHe(T) * nHeI +
            GAHp(T) * nHII + GAel(T) * nde
    h2lte = H2LTE(T)
    cool_gas = h2lte / (one_ + h2lte / galdl)
    galdl_c = GAHI(Tc) * nHI + GAH2(Tc) * nH2 + GAHe(Tc) * nHeI +
              GAHp(Tc) * nHII + GAel(Tc) * nde
    h2lte_c  = H2LTE(Tc)
    cool_cmb = h2lte_c / (one_ + h2lte_c / galdl_c)
    fudge = one_
    if ih2optical && nH !== nothing
        fudge = min((R(nH) / R(8.0e9))^R(-0.45), one_)   # 0.76·ρ·dom = n_H
    end
    h2 = fudge * nH2 * (cool_gas - cool_cmb)

    # ── HD (two-level, collider = HI, CMB-gated as in grackle) ───────────────
    hd = zero(R)
    if T > Tc
        hdlte  = HDlte(T)
        hdlte1 = hdlte / nHI
        hdlow1 = max(HDlow(T), R(TINY))
        hd = nHD * hdlte / (one_ + hdlte1 / hdlow1)
    end

    # ── CMB Compton (cooling if T>T_cmb, heating if T<T_cmb) ─────────────────
    compton = comp1_cmb(R(z)) * (T - Tc) * nde

    return -(atomic + h2 + hd + compton)
end
