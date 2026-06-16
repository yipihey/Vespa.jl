# network_step.jl — ONE linearly-implicit backward-Euler sweep of the v2026
# reduced primordial+D network: one backward-Euler sweep of the network, with
#   H⁻, H₂⁺, D⁺ as algebraic-equilibrium intermediaries; helium in collisional-
#   radiative ionisation equilibrium (or, optionally, advected He⁺ — see the He
#   block below); nₑ from charge conservation; no dust / self-shielding (only the
#   CMB photo-rates k27,k28 and any supplied external photoionisation survive).
#
# Each provisional Xⁿ⁺¹ = (s·dt + Xⁿ)/(1 + a·dt) with s = formation, a =
# destruction frequency.  Species use the network's mass-equivalent convention
# (same as equilibrium.jl): yHI=n_HI, yHII=n_HII, yde=n_e, yHM=n_HM, yHeI=n_HeI, and
# yH2I=2·n(H2), yH2II=2·n(H2⁺), yHDI=3·n(HD) — so every literal /2,/3,2× is
# consistent with that convention.  Pure & allocation-free (AD-friendly); the
# per-cell KA launcher lives in solve.jl.
#
# Gauss-Seidel ordering: HIp/HIIp/dep/H2Ip from the OLD state; HMp from OLD;
# H2IIp from the NEW provisionals (uses dep for its e⁻ destruction); the
# deuterium block from OLD; and the charge-conservation nₑ at the end uses the
# NEW HII but the OLD HM/H2II.

export network_step

# The species floor (absolute, code units — negligible vs real abundances in any
# cosmological density_units; see the temperature.jl test note).
const _NET_TINY = 1.0e-20

"""
    network_step(d, fh, yHI, yHII, yde, yH2I, yHM, yH2II, yDI, yDII, yHDI, K, dt;
                 deuterium = false)

One backward-Euler sweep. `d` = total density (same units as the species), `fh` =
hydrogen mass fraction, `K` = NamedTuple of rate coefficients `k1..k58` plus the
CMB photo-rates `k27`,`k28` (all in the network's per-density-unit convention),
`dt` = the subcycle step. Returns a NamedTuple of the updated species. Pure.
"""
@inline function network_step(d, fh, yHI, yHII, yde, yH2I, yHM, yH2II,
                              yDI, yDII, yHDI, K, dt; deuterium::Bool = false,
                              yHeII_in = nothing, yHeIII_in = nothing,
                              GamHeI = 0.0, GamHeII = 0.0)
    R    = typeof(yHI)
    tiny = R(_NET_TINY)
    two  = R(2); half = R(0.5); four = R(4)

    # Helium ionisation.  Two modes (both in the mass-equivalent ×4 convention,
    # yHeX = 4·n(HeX)):
    #   • Equilibrium (default, yHeII_in===nothing): collisional-radiative ionisation
    #     equilibrium — every up/down channel balanced.  For each stage the ratio of
    #     successive ions is  (collisional ion·nₑ + photo) / (recombination·nₑ):
    #         n_HeII /n_HeI  = K.she1/nₑ + k3/k4 + Γ_HeI /(k4·nₑ)
    #         n_HeIII/n_HeII = K.she2/nₑ + k5/k6 + Γ_HeII/(k6·nₑ)
    #     The radiation/CMB term is the Saha factor (K.she*, detailed balance at
    #     T_rad — exact at z≳3000 & z≲1700, ~3% low only in the He⁺→He⁰ freeze-out);
    #     k3/k5 = collisional ionisation, k4/k6 = recombination (Abel/Hui-Gnedin, at
    #     T_matter); Γ = optional external photoionisation rate [s⁻¹] (e.g. a UV
    #     background, 0 by default).  Cold CMB + cold matter ⇒ He neutral ⇒ no cost
    #     at late times; hot gas ⇒ collisional-ionisation equilibrium.  Solved
    #     semi-implicitly off the OLD nₑ.
    #   • Evolved (yHeII_in given): the caller (evolve_cell_mixing) has integrated
    #     He⁺ with the full rate equation (incl. the He I radiative-transfer
    #     freeze-out); we just consume its yHeII/yHeIII for the electron balance.
    nHe4 = (one(R) - R(fh)) * d                  # total He in ×4 convention
    if yHeII_in === nothing
        _, nHeII, nHeIII = helium_equilibrium(K.she1, K.she2, K.k3, K.k4, K.k5, K.k6,
                                              max(yde, tiny), nHe4 / four;
                                              GamHeI = GamHeI, GamHeII = GamHeII)
        yHeII  = four * nHeII                     # back to ×4 convention
        yHeIII = four * nHeIII
    else
        yHeII   = R(yHeII_in)
        yHeIII  = R(yHeIII_in)
    end
    yHeI = max(nHe4 - yHeII - yHeIII, tiny)      # neutral He (×4)

    k1=K.k1; k2=K.k2; k3=K.k3; k4=K.k4; k5=K.k5; k6=K.k6; k7=K.k7; k8=K.k8
    k9=K.k9; k10=K.k10; k11=K.k11; k12=K.k12; k13=K.k13; k14=K.k14; k15=K.k15
    k16=K.k16; k17=K.k17; k18=K.k18; k19=K.k19; k22=K.k22; k57=K.k57; k58=K.k58
    k27=K.k27; k28=K.k28; k_beta1s=K.k_beta1s

    # ── (C) HI / HII / e⁻ / H2 with molecular terms ──────────────────────────
    # H⁻ and H₂⁺ are fast algebraic-equilibrium intermediaries. We evaluate them
    # from the OLD state FIRST and substitute the equilibrium H₂⁺ (H2IIeq) into the
    # HI/HII/e⁻/H2 source terms — INCLUDING the CMB photodissociation return k28
    # (H₂⁺+γ → HI + HII) in the HI/HII equations.
    #
    # DELIBERATE EXTENSION beyond the original network (Abel, Anninos, Zhang &
    # Norman 1997): it drops the H₂⁺ photodissociation (k28) return to HI/HII and
    # uses the lagged H₂⁺ density because H₂⁺ is trace at z<100, its galaxy-
    # formation regime.  During recombination (z≈1000-1200), however, the radiative
    # association k9 (HI+HII→H₂⁺) reaches ~1.5% of the net recombination rate; the
    # CMB then photodissociates that H₂⁺ straight back (k28≈330 s⁻¹), so the cycle
    # is very nearly null for HII.  Without crediting the k28/k10 return, the k9
    # term leaks HII and biases x_e ~1-1.5% low at z≈1000-1100.  We add it for the
    # recombination epoch.  Closing the cycle (the only net HII sink via H₂⁺ is the
    # dissociative k18·de branch) recovers the full network to <0.25% of HyRec
    # across z=700-1100.  H₂⁺ being trace at low z, the change is negligible in the
    # original galaxy-formation regime.
    HMp    = equilibrium_HM(yHI, yHII, yde, yH2II, k7, k8, k14, k15, k16, k17, k19, k27)
    H2IIeq = equilibrium_H2II(yHI, yHII, yH2I, yde, HMp,
                              k9, k10, k11, k17, k18, k19, k28)
    nH2II  = H2IIeq / two          # n(H₂⁺); H2IIeq carries the 2× mass-equiv convention

    # 1) HI  (+ β₁s CMB photoionisation of H(1s); + k28 H₂⁺ photodissociation return)
    sc = k2*yHII*yde + two*k13*yHI*yH2I/two + k11*yHII*yH2I/two +
         two*k12*yde*yH2I/two + k14*yHM*yde + k15*yHM*yHI +
         two*k16*yHM*yHII + two*k18*H2IIeq*yde/two + k19*H2IIeq*yHM/two +
         k28*nH2II
    ac = k1*yde + k7*yde + k8*yHM + k9*yHII + k10*H2IIeq/two +
         two*k22*yHI^2 + k57*yHI + k58*yHeI/four + k_beta1s
    HIp = (sc*dt + yHI) / (one(R) + ac*dt)

    # 2) HII  (+ β₁s source; + k28 H₂⁺ photodissociation return)
    sc = k1*yHI*yde + k10*H2IIeq*yHI/two + k57*yHI*yHI + k58*yHI*yHeI/four +
         k_beta1s*yHI + k28*nH2II
    ac = k2*yde + k9*yHI + k11*yH2I/two + k16*yHM + k17*yHM
    HIIp = (sc*dt + yHII) / (one(R) + ac*dt)

    # 3) e⁻ provisional — used ONLY for downstream consistency
    sc = k8*yHM*yHI + k15*yHM*yHI + k17*yHM*yHII + k57*yHI*yHI + k58*yHI*yHeI/four +
         k_beta1s*yHI
    ac = -(k1*yHI - k2*yHII + k3*yHeI/four - k6*yHeIII/four +
           k5*yHeII/four - k4*yHeII/four + k14*yHM - k7*yHI - k18*H2IIeq/two)
    dep = (sc*dt + yde) / (one(R) + ac*dt)

    # 7) H2  (formation via the H₂⁺ and H⁻ channels uses the equilibrium H₂⁺)
    sc = two*(k8*yHM*yHI + k10*H2IIeq*yHI/two + k19*H2IIeq*yHM/two + k22*yHI*yHI^2)
    ac = k13*yHI + k11*yHII + k12*yde
    H2Ip = (sc*dt + yH2I) / (one(R) + ac*dt)

    # 8,9) store the consistent old-state equilibrium H₂⁺ (H⁻ already in HMp above)
    H2IIp = H2IIeq

    # ── (D) deuterium (OLD state) ────────────────────────────────────────────
    if deuterium
        k50=K.k50; k51=K.k51; k52=K.k52; k53=K.k53; k54=K.k54; k55=K.k55; k56=K.k56
        three = R(3)
        # 1) DI
        sc = k2*yDII*yde + k51*yDII*yHI + two*k55*yHDI*yHI/three
        ac = k1*yde + k50*yHII + k54*yH2I/two + k56*yHM
        DIp = (sc*dt + yDI) / (one(R) + ac*dt)
        # 2) DII equilibrium (OLD)
        DIIp = equilibrium_DII(yDI, yde, yHI, yHII, yH2I, yHDI,
                               k1, k2, k50, k51, k52, k53)
        # 3) HDI (OLD DI,DII)
        sc = three*(k52*yDII*yH2I/two/two + k54*yDI*yH2I/two/two +
                    two*k56*yDI*yHM/two)
        ac = k53*yHII + k55*yHI
        HDIp = (sc*dt + yHDI) / (one(R) + ac*dt)
    else
        DIp = yDI; DIIp = yDII; HDIp = yHDI
    end

    # ── (E) field assignment + charge-conservation nₑ ────────────────────────
    HI_n   = max(HIp,  tiny)
    HII_n  = max(HIIp, tiny)
    # nₑ from charge conservation: NEW HII, He neutral, but OLD HM/H2II.
    de_n   = HII_n + yHeII/four + yHeIII/two - yHM + yH2II/two
    HM_n   = max(HMp,   tiny)
    H2I_n  = max(H2Ip,  tiny)
    H2II_n = max(H2IIp, tiny)
    DI_n   = max(DIp,  tiny)
    DII_n  = max(DIIp, tiny)
    HDI_n  = max(HDIp, tiny)

    return (; yHI = HI_n, yHII = HII_n, yde = de_n, yH2I = H2I_n,
            yHM = HM_n, yH2II = H2II_n, yDI = DI_n, yDII = DII_n, yHDI = HDI_n)
end
