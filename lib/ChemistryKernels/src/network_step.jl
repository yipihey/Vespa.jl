# network_step.jl — ONE linearly-implicit backward-Euler sweep of the v2026
# reduced primordial+D network: the per-subcycle update grackle performs in
# step_rate_g (solve_rate_cool_g.F:2128-2604), specialized to the reduced flags
#   ieq=1 (H⁻,H2⁺,D⁺ algebraic),  ineutralhe=1 (helium forced neutral, nₑ from
#   charge conservation),  no radiation / dust / self-shielding (only the CMB
#   photo-rates k27,k28 survive).
#
# Each provisional Xⁿ⁺¹ = (s·dt + Xⁿ)/(1 + a·dt) with s = formation, a =
# destruction frequency.  Species use grackle's MASS-EQUIVALENT convention (same
# as equilibrium.jl): yHI=n_HI, yHII=n_HII, yde=n_e, yHM=n_HM, yHeI=n_HeI, and
# yH2I=2·n(H2), yH2II=2·n(H2⁺), yHDI=3·n(HD) — so every literal /2,/3,2× matches
# the Fortran verbatim.  Pure & allocation-free (AD-friendly); the per-cell KA
# launcher lives in solve.jl.
#
# Gauss-Seidel ordering faithfully reproduced (it matters for bit-parity with the
# reduced lib in Wave 5): HIp/HIIp/dep/H2Ip from the OLD state; HMp from OLD;
# H2IIp from the NEW provisionals (uses dep for its e⁻ destruction); the
# deuterium block from OLD; and the charge-conservation nₑ at the end uses the
# NEW HII but the OLD HM/H2II (exactly solve_rate_cool_g.F:2566-2570).

export network_step

# grackle's `tiny` species floor (absolute, code units — negligible vs real
# abundances in any cosmological density_units; see the temperature.jl test note).
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
                              yDI, yDII, yHDI, K, dt; deuterium::Bool = false)
    R    = typeof(yHI)
    tiny = R(_NET_TINY)
    two  = R(2); half = R(0.5); four = R(4)

    # helium forced neutral
    yHeI   = (one(R) - R(fh)) * d
    yHeII  = tiny
    yHeIII = tiny

    k1=K.k1; k2=K.k2; k3=K.k3; k4=K.k4; k5=K.k5; k6=K.k6; k7=K.k7; k8=K.k8
    k9=K.k9; k10=K.k10; k11=K.k11; k12=K.k12; k13=K.k13; k14=K.k14; k15=K.k15
    k16=K.k16; k17=K.k17; k18=K.k18; k19=K.k19; k22=K.k22; k57=K.k57; k58=K.k58
    k27=K.k27; k28=K.k28

    # ── (C) HI / HII / e⁻ / H2 with molecular terms (OLD state) ──────────────
    # 1) HI  (solve_rate_cool_g.F:2339-2360)
    sc = k2*yHII*yde + two*k13*yHI*yH2I/two + k11*yHII*yH2I/two +
         two*k12*yde*yH2I/two + k14*yHM*yde + k15*yHM*yHI +
         two*k16*yHM*yHII + two*k18*yH2II*yde/two + k19*yH2II*yHM/two
    ac = k1*yde + k7*yde + k8*yHM + k9*yHII + k10*yH2II/two +
         two*k22*yHI^2 + k57*yHI + k58*yHeI/four
    HIp = (sc*dt + yHI) / (one(R) + ac*dt)

    # 2) HII  (:2384-2398)
    sc = k1*yHI*yde + k10*yH2II*yHI/two + k57*yHI*yHI + k58*yHI*yHeI/four
    ac = k2*yde + k9*yHI + k11*yH2I/two + k16*yHM + k17*yHM
    HIIp = (sc*dt + yHII) / (one(R) + ac*dt)

    # 3) e⁻ provisional  (:2401-2424) — used ONLY for the H2⁺ equilibrium below
    sc = k8*yHM*yHI + k15*yHM*yHI + k17*yHM*yHII + k57*yHI*yHI + k58*yHI*yHeI/four
    ac = -(k1*yHI - k2*yHII + k3*yHeI/four - k6*yHeIII/four +
           k5*yHeII/four - k4*yHeII/four + k14*yHM - k7*yHI - k18*yH2II/two)
    dep = (sc*dt + yde) / (one(R) + ac*dt)

    # 7) H2  (:2427-2445)
    sc = two*(k8*yHM*yHI + k10*yH2II*yHI/two + k19*yH2II*yHM/two + k22*yHI*yHI^2)
    ac = k13*yHI + k11*yHII + k12*yde
    H2Ip = (sc*dt + yH2I) / (one(R) + ac*dt)

    # 8) H⁻ equilibrium (OLD HI,de,H2II,HII)  (:2450-2459)
    HMp = equilibrium_HM(yHI, yHII, yde, yH2II, k7, k8, k14, k15, k16, k17, k19, k27)

    # 9) H2⁺ equilibrium (NEW provisionals + dep)  (:2465-2475)
    H2IIp = equilibrium_H2II(HIp, HIIp, H2Ip, dep, HMp,
                             k9, k10, k11, k17, k18, k19, k28)

    # ── (D) deuterium (OLD state)  (:2484-2536) ──────────────────────────────
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

    # ── (E) field assignment + charge-conservation nₑ  (:2549-2581) ──────────
    HI_n   = max(HIp,  tiny)
    HII_n  = max(HIIp, tiny)
    # nₑ from charge conservation: NEW HII, He neutral, but OLD HM/H2II (Fortran).
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
