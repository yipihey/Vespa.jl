# cooling_metal.jl — atomic/ionic METAL-LINE cooling (C, O, Si, Fe), fast & AD-ready.
#
# Physics provenance
#   Fine-structure (T ≲ 1e4 K) collisional rates: Glover & Jappsen (2007), ported
#   from Enzo input/metal_cooling.pro (and cross-checked against the MetalLineCooling
#   draft / its validated Python port).  The neutral↔singly-ionized partition of each
#   element is set in instantaneous statistical equilibrium but DRIVEN BY THE LIVE
#   non-equilibrium electron density n_e (and n_HI, n_HII) from the primordial network
#   (Enzo calc_equil_c_si; the scheme Abel suggested for metal_cool.dat).  Level
#   populations are solved in statistical equilibrium with the CMB radiation field.
#   Fe is tracked as an INDEPENDENT element so a non-solar [α/Fe] cools correctly.
#
# Design (matches the house style: pure @inline, precision-generic R=typeof(T),
#   allocation-free, no Dict / no Vector / no `\`; monomorphic — each ion is its own
#   function, summed explicitly).  Three layers, all linear in element abundance:
#     L1  per-ion emissivity PER EMITTING ION  ε̃  [erg s⁻¹ cm³]  (abundance-free)
#     L2  live-n_e ion fraction f_{X,+}                          (abundance-free)
#     L3  Λ = Σ_X n_H·a_X·f_stage·ε̃   with a_X = n(X)/n_H per cell
#   Because Λ is linear in a_X, yield-channel templates (II/Ia/AGB) mix by linear
#   combination of the abundance vector (done host-side; see `metal_abund`).
#
#   Closed-form level solves (no allocation, differentiable):
#     • 2-level (C II, Si II): n1/n0 = r_up/r_dn          (IDL calc_ratio1)
#     • 3-level (C I, O I, Si I): exact 3×3 SE solve        (IDL calc_ratio2)
#     • 5-level Fe II: adjacent-J ladder ⇒ tridiagonal ⇒ birth-death chain, so the
#       steady state is detailed balance per link and the populations are a
#       normalized product of up/down ratios (no matrix).  Weak ΔJ>1 Fe II lines are
#       dropped from the solve (consistent with the factor-2 Fe⁺–H rate uncertainty).
#
# High-T handoff: fine-structure cooling is smoothly tapered to 0 over 1e4–2e4 K
#   (smootherstep) rather than hard-cut — keeps Λ C² (good for implicit/AD solvers)
#   while still NOT inventing solar-ratio cooling above 1e4 K (that regime needs an
#   ion-by-ion table, deferred).  Below 1e4 K the taper is exactly 1.
#
# Units: CGS.  `metal_cooling_rate` returns Λ ≥ 0 [erg s⁻¹ cm⁻³]; edot.jl subtracts it.

# hc/k_B in µm·K (so E_ul/k_B [K] = _HCK_UMK / λ_µm); avoids carrying HP/CL into the
# hot path — line energies are compile-time constants.
const _HCK_UMK = 1.43877687e4
@inline _EK(lam_um) = _HCK_UMK / lam_um

# CMB photon occupation 1/(e^{E/Trad}−1), exponent-clamped (cf. _LOG_DHUGE) so it is
# overflow-safe in Float32 and →0 (not Inf/NaN) for cold CMB / high-energy lines.
@inline _nbar(EK, Trad) = inv(expm1(min(oftype(Trad, _LOG_DHUGE), EK / Trad)))

# Net per-emitting-ion emissivity of one line [erg s⁻¹ cm³]:
#   dE·[ n_u·A·(1+n̄) − n_l·A·n̄·(g_u/g_l) ]   (spontaneous+stimulated down − absorption up)
# Vanishes as T_gas→T_CMB (level pops → Boltzmann at T_rad).  May be <0 if T<T_CMB
# (CMB pumping = heating), which composes correctly with the energy update.
@inline function _emiss(n_u, n_l, A, EK, nbar, gratio)
    R = typeof(n_u)
    dE = R(KBOLTZ) * R(EK)
    return dE * (n_u * A * (one(R) + nbar) - n_l * A * nbar * gratio)
end

# ── 2-level closed form (C II, Si II) ────────────────────────────────────────
# qdn = Σ_partners q_ul(T)·n_partner [s⁻¹]; A, EK for the single line; g0,g1 weights.
@inline function _cool2(qdn, A, EK, g0, g1, Tg, Trad)
    R = typeof(Tg)
    Ar = R(A); EKr = R(EK); gr = R(g1) / R(g0)
    nbar = _nbar(EKr, Trad)
    qup  = qdn * gr * exp(-min(R(_LOG_DHUGE), EKr / Tg))
    rdn  = qdn + Ar * (one(R) + nbar)
    rup  = qup + Ar * nbar * gr
    r    = rup / rdn
    n1   = r / (one(R) + r); n0 = one(R) - n1
    return _emiss(n1, n0, Ar, EKr, nbar, gr)
end

# ── 3-level closed form (C I, O I, Si I) — IDL calc_ratio2 ───────────────────
# qXY are the DOWNWARD collisional sums for links (1→0),(2→0),(2→1) [s⁻¹].
# `_cool3_lines` returns the 3 per-line emissivities (ε10,ε20,ε21); `_cool3` sums
# them in the same left-to-right order (scalar value bit-identical).
@inline function _cool3_lines(q10, q20, q21,
                        A10, E10, A20, E20, A21, E21,
                        g0, g1, g2, Tg, Trad)
    R = typeof(Tg)
    A10r=R(A10); A20r=R(A20); A21r=R(A21)
    E10r=R(E10); E20r=R(E20); E21r=R(E21)
    g0r=R(g0); g1r=R(g1); g2r=R(g2)
    g10=g1r/g0r; g20=g2r/g0r; g21=g2r/g1r
    nb10=_nbar(E10r,Trad); nb20=_nbar(E20r,Trad); nb21=_nbar(E21r,Trad)
    # upward collisional via detailed balance
    q01 = q10*g10*exp(-min(R(_LOG_DHUGE), E10r/Tg))
    q02 = q20*g20*exp(-min(R(_LOG_DHUGE), E20r/Tg))
    q12 = q21*g21*exp(-min(R(_LOG_DHUGE), E21r/Tg))
    # total up/down rates (collisional + radiative incl. CMB stimulated/absorption)
    r10 = q10 + A10r*(one(R)+nb10);  r01 = q01 + A10r*nb10*g10
    r20 = q20 + A20r*(one(R)+nb20);  r02 = q02 + A20r*nb20*g20
    r21 = q21 + A21r*(one(R)+nb21);  r12 = q12 + A21r*nb21*g21
    # calc_ratio2 (metal_cooling.pro:44-68)
    a1 = r01 + r02; a2 = -r10; a3 = -r20
    b1 = r01;       b2 = -(r10 + r12); b3 = r21
    n2 = -a1*(a1*b2 - b1*a2) /
         ((a1 - a2)*(a1*b3 - b1*a3) - (a1 - a3)*(a1*b2 - b1*a2))
    n1 = a1/(a1 - a2) - ((a1 - a3)/(a1 - a2))*n2
    n0 = one(R) - n1 - n2
    return (_emiss(n1, n0, A10r, E10r, nb10, g10),
            _emiss(n2, n0, A20r, E20r, nb20, g20),
            _emiss(n2, n1, A21r, E21r, nb21, g21))
end
@inline function _cool3(q10, q20, q21, A10, E10, A20, E20, A21, E21, g0, g1, g2, Tg, Trad)
    t = _cool3_lines(q10, q20, q21, A10, E10, A20, E20, A21, E21, g0, g1, g2, Tg, Trad)
    return t[1] + t[2] + t[3]
end

# ── 5-level adjacent-J ladder (Fe II) — birth-death chain, exact closed form ──
# Each link k couples levels (k-1)↔k only ⇒ zero net flux per link ⇒
#   n_k / n_{k-1} = r_up_k / r_dn_k.  Populations = normalized cumulative product.
# Inputs: 4 links, each its downward collisional sum qdn_k and line (A,EK,g_lo,g_hi).
@inline function _coolladder4_lines(qd1,qd2,qd3,qd4,
                              A1,EK1,A2,EK2,A3,EK3,A4,EK4,
                              g0,g1,g2,g3,g4, Tg, Trad)
    R = typeof(Tg)
    @inline function _ratio(qdn, A, EK, glo, ghi)
        Ar=R(A); EKr=R(EK); gr=R(ghi)/R(glo)
        nbar=_nbar(EKr,Trad)
        qup = qdn*gr*exp(-min(R(_LOG_DHUGE), EKr/Tg))
        rdn = qdn + Ar*(one(R)+nbar)
        rup = qup + Ar*nbar*gr
        return rup/rdn, nbar, Ar, EKr, gr
    end
    r1,nb1,Ar1,EK1r,gr1 = _ratio(qd1,A1,EK1,g0,g1)
    r2,nb2,Ar2,EK2r,gr2 = _ratio(qd2,A2,EK2,g1,g2)
    r3,nb3,Ar3,EK3r,gr3 = _ratio(qd3,A3,EK3,g2,g3)
    r4,nb4,Ar4,EK4r,gr4 = _ratio(qd4,A4,EK4,g3,g4)
    p0=one(R); p1=p0*r1; p2=p1*r2; p3=p2*r3; p4=p3*r4
    s = p0+p1+p2+p3+p4
    n0=p0/s; n1=p1/s; n2=p2/s; n3=p3/s; n4=p4/s
    return (_emiss(n1,n0,Ar1,EK1r,nb1,gr1), _emiss(n2,n1,Ar2,EK2r,nb2,gr2),
            _emiss(n3,n2,Ar3,EK3r,nb3,gr3), _emiss(n4,n3,Ar4,EK4r,nb4,gr4))
end
@inline function _coolladder4(qd1,qd2,qd3,qd4, A1,EK1,A2,EK2,A3,EK3,A4,EK4,
                              g0,g1,g2,g3,g4, Tg, Trad)
    t = _coolladder4_lines(qd1,qd2,qd3,qd4, A1,EK1,A2,EK2,A3,EK3,A4,EK4, g0,g1,g2,g3,g4, Tg, Trad)
    return t[1] + t[2] + t[3] + t[4]
end

@inline _T2(T) = T / oftype(T, 100.0)   # T/100, the GJ07 fit variable

# =============================================================================
# CARBON — C I (3-level) + C II (2-level)   (GJ07 fits, metal_cooling.pro)
# =============================================================================
@inline cI_q21_oH2(T) = (R=typeof(T); R(8.7e-11) - R(6.6e-11)*exp(-T/R(218.3)) + R(6.6e-11)*exp(-R(2.0)*T/R(218.3)))
@inline cI_q21_pH2(T) = (R=typeof(T); R(7.9e-11) - R(8.7e-11)*exp(-T/R(126.4)) + R(1.3e-10)*exp(-R(2.0)*T/R(126.4)))
@inline cI_q21_HI(T)  = (R=typeof(T); R(1.6e-10)*_T2(T)^R(0.14))
@inline cI_q21_HII(T) = (R=typeof(T); T <= R(5000.0) ? (R(9.6e-11) - R(1.8e-14)*T + R(1.9e-18)*T^2)*T^R(0.45) : R(8.9e-10)*T^R(0.117))
@inline cI_q21_e(T)   = (R=typeof(T); lt=log(T); T <= R(1000.0) ?
    R(2.88e-6)/sqrt(T)*exp(R(-9.25141) - R(7.73782e-1)*lt + R(3.61184e-1)*lt^2 - R(1.50892e-2)*lt^3 - R(6.56325e-4)*lt^4) :
    R(2.88e-6)/sqrt(T)*exp(R(-4.446e2) - R(2.27913e2)*lt + R(4.2595e1)*lt^2 - R(3.4762)*lt^3 + R(1.0508e-1)*lt^4))
@inline cI_q31_oH2(T) = (R=typeof(T); R(1.2e-10) - R(6.1e-11)*exp(-T/R(387.3)))
@inline cI_q31_pH2(T) = (R=typeof(T); R(1.1e-10) - R(8.6e-11)*exp(-T/R(223.0)) + R(8.7e-11)*exp(-R(2.0)*T/R(126.4)))
@inline cI_q31_HI(T)  = (R=typeof(T); R(9.2e-11)*_T2(T)^R(0.26))
@inline cI_q31_HII(T) = (R=typeof(T); T <= R(5000.0) ? (R(3.1e-12) - R(6.0e-16)*T + R(3.9e-20)*T^2)*T : R(2.3e-9)*T^R(0.0965))
@inline cI_q31_e(T)   = (R=typeof(T); lt=log(T); T <= R(1000.0) ?
    R(1.73e-6)/sqrt(T)*exp(R(-7.69735) - R(1.30745)*lt + R(0.697638)*lt^2 - R(0.111338)*lt^3 + R(0.705277e-2)*lt^4) :
    R(1.73e-6)/sqrt(T)*exp(R(-3.50609e2) - R(1.87474e2)*lt + R(3.61803e1)*lt^2 - R(3.03283)*lt^3 + R(9.38138e-2)*lt^4))
@inline cI_q32_oH2(T) = (R=typeof(T); R(2.9e-10) - R(1.9e-10)*exp(-T/R(348.9)))
@inline cI_q32_pH2(T) = (R=typeof(T); R(2.7e-10) - R(2.6e-10)*exp(-T/R(250.7)) + R(1.8e-10)*exp(-R(2.0)*T/R(250.7)))
@inline cI_q32_HI(T)  = (R=typeof(T); R(2.9e-10)*_T2(T)^R(0.26))
@inline cI_q32_HII(T) = (R=typeof(T); T <= R(5000.0) ? (R(1.0e-10) - R(2.2e-14)*T + R(1.7e-18)*T^2)*T^R(0.70) : R(9.2e-9)*T^R(0.0535))
@inline cI_q32_e(T)   = (R=typeof(T); lt=log(T); T <= R(1000.0) ?
    R(1.73e-6)/sqrt(T)*exp(R(-7.4387) - R(0.57443)*lt + R(0.358264)*lt^2 - R(3.19268)*lt^3 + R(9.78573e-2)*lt^4) :
    R(1.73e-6)/sqrt(T)*exp(R(-3.86186e2) - R(2.02192e2)*lt + R(3.85049e1)*lt^2 - R(3.19268)*lt^3 + R(9.78573e-2)*lt^4))

# Per-emitting-ion C I emissivity [erg s⁻¹ cm³]; lines 1→0 609.2µm, 2→0 229.9µm, 2→1 369.0µm
@inline function _cool_CI_lines(T, Trad, nHI, nHII, nH2o, nH2p, nde)
    q10 = cI_q21_oH2(T)*nH2o + cI_q21_pH2(T)*nH2p + cI_q21_HI(T)*nHI + cI_q21_HII(T)*nHII + cI_q21_e(T)*nde
    q20 = cI_q31_oH2(T)*nH2o + cI_q31_pH2(T)*nH2p + cI_q31_HI(T)*nHI + cI_q31_HII(T)*nHII + cI_q31_e(T)*nde
    q21 = cI_q32_oH2(T)*nH2o + cI_q32_pH2(T)*nH2p + cI_q32_HI(T)*nHI + cI_q32_HII(T)*nHII + cI_q32_e(T)*nde
    return _cool3_lines(q10, q20, q21,
                  7.9e-8, _EK(609.2), 2.1e-14, _EK(229.9), 2.1e-7, _EK(369.0),
                  1.0, 3.0, 5.0, T, Trad)
end
@inline _cool_CI(T,Trad,nHI,nHII,nH2o,nH2p,nde) =
    (t=_cool_CI_lines(T,Trad,nHI,nHII,nH2o,nH2p,nde); t[1]+t[2]+t[3])

@inline cII_q21_oH2(T) = (R=typeof(T); T <= R(250.0) ? (R(4.7e-10) + R(4.6e-13)*T) : R(5.85e-10)*T^R(0.07))
@inline cII_q21_pH2(T) = (R=typeof(T); T <= R(250.0) ? (R(2.5e-10)*T^R(0.12))     : R(4.85e-10)*T^R(0.07))
@inline cII_q21_HI(T)  = (R=typeof(T); T <= R(2000.0) ? R(8.0e-10)*_T2(T)^R(0.07) : R(3.1e-10)*_T2(T)^R(0.385))
@inline cII_q21_e(T)   = (R=typeof(T); T <= R(2000.0) ? R(3.86e-7)*_T2(T)^R(-0.5) : R(2.43e-7)*_T2(T)^R(-0.345))

@inline function _cool_CII(T, Trad, nHI, nH2o, nH2p, nde)
    q = cII_q21_oH2(T)*nH2o + cII_q21_pH2(T)*nH2p + cII_q21_HI(T)*nHI + cII_q21_e(T)*nde
    return _cool2(q, 2.3e-6, _EK(157.7), 2.0, 4.0, T, Trad)
end

# =============================================================================
# OXYGEN — O I (3-level; ground ³P₂). O I H⁺ routing corrected (q31_HII on 2→0).
# =============================================================================
@inline oI_q21_oH2(T) = (R=typeof(T); R(2.7e-11)*T^R(0.362))
@inline oI_q21_pH2(T) = (R=typeof(T); R(3.46e-11)*T^R(0.316))
@inline oI_q21_HI(T)  = (R=typeof(T); R(9.2e-11)*_T2(T)^R(0.67))
@inline oI_q21_e(T)   = (R=typeof(T); R(5.12e-10)*T^R(-0.075))
@inline oI_q21_HII(T) = (R=typeof(T); T <= R(194.0) ? R(6.38e-11)*T^R(0.40) : T <= R(3686.0) ? R(7.75e-12)*T^R(0.80) : R(2.65e-10)*T^R(0.37))
@inline oI_q31_oH2(T) = (R=typeof(T); R(5.49e-11)*T^R(0.317))
@inline oI_q31_pH2(T) = (R=typeof(T); R(7.07e-11)*T^R(0.268))
@inline oI_q31_HI(T)  = (R=typeof(T); R(4.3e-11)*_T2(T)^R(0.80))
@inline oI_q31_e(T)   = (R=typeof(T); R(4.86e-10)*T^R(-0.026))
@inline oI_q31_HII(T) = (R=typeof(T); T <= R(511.0) ? R(6.10e-13)*T^R(1.10) : T <= R(7510.0) ? R(2.12e-12)*T^R(0.90) : R(4.49e-10)*T^R(0.30))
@inline oI_q32_oH2(T) = (R=typeof(T); R(2.74e-14)*T^R(1.060))
@inline oI_q32_pH2(T) = (R=typeof(T); R(3.33e-15)*T^R(1.360))
@inline oI_q32_HI(T)  = (R=typeof(T); R(1.1e-10)*_T2(T)^R(0.44))
@inline oI_q32_e(T)   = (R=typeof(T); R(1.08e-14)*T^R(0.926))
@inline oI_q32_HII(T) = (R=typeof(T); T <= R(2090.0) ? R(2.03e-11)*T^R(0.56) : R(3.43e-10)*T^R(0.19))

# lines 1→0 63.1µm, 2→0 44.2µm, 2→1 145.6µm; ground ³P₂(g5) > ³P₁(g3) > ³P₀(g1)
@inline function _cool_OI_lines(T, Trad, nHI, nHII, nH2o, nH2p, nde)
    q10 = oI_q21_oH2(T)*nH2o + oI_q21_pH2(T)*nH2p + oI_q21_HI(T)*nHI + oI_q21_e(T)*nde + oI_q21_HII(T)*nHII
    q20 = oI_q31_oH2(T)*nH2o + oI_q31_pH2(T)*nH2p + oI_q31_HI(T)*nHI + oI_q31_e(T)*nde + oI_q31_HII(T)*nHII
    q21 = oI_q32_oH2(T)*nH2o + oI_q32_pH2(T)*nH2p + oI_q32_HI(T)*nHI + oI_q32_e(T)*nde + oI_q32_HII(T)*nHII
    return _cool3_lines(q10, q20, q21,
                  8.9e-5, _EK(63.1), 1.3e-10, _EK(44.2), 1.8e-5, _EK(145.6),
                  5.0, 3.0, 1.0, T, Trad)
end
@inline _cool_OI(T,Trad,nHI,nHII,nH2o,nH2p,nde) =
    (t=_cool_OI_lines(T,Trad,nHI,nHII,nH2o,nH2p,nde); t[1]+t[2]+t[3])

# =============================================================================
# SILICON — Si I (3-level; H, H⁺ only) + Si II (2-level)
# =============================================================================
@inline siI_q21_HI(T)  = (R=typeof(T); R(3.5e-10)*_T2(T)^R(-0.03))
@inline siI_q21_HII(T) = (R=typeof(T); R(7.2e-9))
@inline siI_q31_HI(T)  = (R=typeof(T); R(1.7e-11)*_T2(T)^R(0.17))
@inline siI_q31_HII(T) = (R=typeof(T); R(7.2e-9))
@inline siI_q32_HI(T)  = (R=typeof(T); R(5.0e-10)*_T2(T)^R(0.17))
@inline siI_q32_HII(T) = (R=typeof(T); R(2.2e-8))

@inline function _cool_SiI_lines(T, Trad, nHI, nHII, nH2o, nH2p, nde)
    q10 = siI_q21_HI(T)*nHI + siI_q21_HII(T)*nHII
    q20 = siI_q31_HI(T)*nHI + siI_q31_HII(T)*nHII
    q21 = siI_q32_HI(T)*nHI + siI_q32_HII(T)*nHII
    return _cool3_lines(q10, q20, q21,
                  8.4e-6, _EK(129.6), 2.4e-10, _EK(44.8), 4.2e-5, _EK(68.4),
                  1.0, 3.0, 5.0, T, Trad)
end
@inline _cool_SiI(T,Trad,nHI,nHII,nH2o,nH2p,nde) =
    (t=_cool_SiI_lines(T,Trad,nHI,nHII,nH2o,nH2p,nde); t[1]+t[2]+t[3])

@inline siII_q21_HI(T) = (R=typeof(T); R(4.95e-10)*_T2(T)^R(0.24))
@inline siII_q21_e(T)  = (R=typeof(T); R(1.2e-6)*_T2(T)^R(-0.5))

@inline function _cool_SiII(T, Trad, nHI, nde)
    q = siII_q21_HI(T)*nHI + siII_q21_e(T)*nde
    return _cool2(q, 2.2e-4, _EK(34.8), 2.0, 4.0, T, Trad)
end

# =============================================================================
# IRON — Fe II ground ⁶D fine structure, adjacent-J ladder (5 levels)
# g = 2J+1 = 10,8,6,4,2 for J = 9/2,7/2,5/2,3/2,1/2.  Links 1→0 25.99µm,
# 2→1 35.35µm, 3→2 51.30µm, 4→3 87.40µm.  q_ul(e) = 8.629e-6·Υ/(g_u·√T)
# (Zhang & Pradhan Υ); Fe⁺–H APPROXIMATE (FEII_HI_APPROX, factor-2; matters only
# in neutral-dominated gas) ~ (Υ/Υ₂₁)·8e-10·(T/100)^0.17.
#
# ACCURACY (measured vs the draft's full 5-level matrix solve, which keeps the
# ΔJ=2 lines): the tridiagonal ladder is near-exact (rel <2e-4) for T≲50 K — the
# cold-gas regime where Fe II 26µm is a dominant coolant — and runs ~12% low at
# 300 K, ~30% low at T≳1000 K (the ladder omits ΔJ=2 collisional pumping of the
# upper levels).  Accepted: at T≳1000 K Fe II is a minor metal coolant and the
# whole channel tapers out by 1e4 K, and the Fe⁺–H rates are factor-2 placeholders
# anyway.  For high-T Fe II fidelity, swap _cool_FeII for a full 5-level solve.
# =============================================================================
@inline feII_qe(ups, gu, T) = (R=typeof(T); R(8.629e-6)*R(ups)/(R(gu)*sqrt(T)))
@inline feII_qHI(ups, T)    = (R=typeof(T); R(ups/9.0)*R(8.0e-10)*_T2(T)^R(0.17))

@inline function _cool_FeII_lines(T, Trad, nHI, nde)
    # link k: (Υ, g_upper); g_lo/g_hi per link from (10,8,6,4,2)
    qd1 = feII_qe(9.0, 8.0, T)*nde + feII_qHI(9.0, T)*nHI   # 1→0
    qd2 = feII_qe(5.5, 6.0, T)*nde + feII_qHI(5.5, T)*nHI   # 2→1
    qd3 = feII_qe(2.7, 4.0, T)*nde + feII_qHI(2.7, T)*nHI   # 3→2
    qd4 = feII_qe(1.0, 2.0, T)*nde + feII_qHI(1.0, T)*nHI   # 4→3
    return _coolladder4_lines(qd1, qd2, qd3, qd4,
                        2.13e-3, _EK(25.99), 1.57e-3, _EK(35.35),
                        7.18e-4, _EK(51.30), 1.96e-4, _EK(87.40),
                        10.0, 8.0, 6.0, 4.0, 2.0, T, Trad)
end
@inline _cool_FeII(T,Trad,nHI,nde) =
    (t=_cool_FeII_lines(T,Trad,nHI,nde); t[1]+t[2]+t[3]+t[4])

# =============================================================================
# Layer 2 — live-n_e ion balance (Enzo calc_equil_c_si; Fe added).  f₊ = r/(1+r),
# r = (k_ci·n_e + k_ctf·n_HII) / (k_rec·n_e + k_ctr·n_HI).  O locked neutral.
# =============================================================================
@inline cC_rec(T) = (R=typeof(T); T <= R(7950.0) ? R(4.67e-12)*(T/R(300.0))^R(-0.6) :
                     T <= R(21140.0) ? R(1.23e-17)*(T/R(300.0))^R(2.49)*exp(R(21845.6)/T) :
                                       R(9.62e-8)*(T/R(300.0))^R(-1.37)*exp(-R(115786.2)/T))
@inline cC_ci(T)  = (R=typeof(T); u=R(11.26)/(T/R(11605.0)); R(6.85e-8)/(R(0.193)+u)*u^R(0.25)*exp(-u))
@inline cC_ctf(T) = (R=typeof(T); R(3.9e-16)*T^R(0.213))
@inline cC_ctr(T) = (R=typeof(T); R(6.08e-14)*(T/R(1e4))^R(1.96)*exp(-R(1.7e5)/T))
@inline _fion_C(T, nde, nHI, nHII) = (R=typeof(T);
    r = (cC_ci(T)*nde + cC_ctf(T)*nHII) / (cC_rec(T)*nde + cC_ctr(T)*nHI); r/(one(R)+r))

@inline cSi_rec(T) = (R=typeof(T); T <= R(2000.0) ? R(7.5e-12)*(T/R(300.0))^R(-0.55) :
                      T <= R(1e4) ? R(4.86e-12)*(T/R(300.0))^R(-0.32) : R(9.08e-14)*(T/R(300.0))^R(0.818))
@inline cSi_ci(T)  = (R=typeof(T); u=R(8.2)/(T/R(11605.0)); R(1.88e-7)*(one(R)+sqrt(u))/(R(0.376)+u)*u^R(0.25)*exp(-u))
@inline cSi_ctf(T) = (R=typeof(T); T <= R(1e4) ? R(5.88e-13)*T^R(0.848) : R(1.45e-13)*T)
@inline _fion_Si(T, nde, nHI, nHII) = (R=typeof(T);
    r = (cSi_ci(T)*nde + cSi_ctf(T)*nHII) / (cSi_rec(T)*nde); r/(one(R)+r))   # no reverse-CT (Enzo)

@inline cFe_rec(T) = (R=typeof(T); R(1.42e-12)*(T/R(1e4))^R(-0.891) + R(1.0e-11)*(T/R(1e4))^R(-1.5)*exp(-R(9.1e4)/T))
@inline cFe_ci(T)  = (R=typeof(T); u=R(7.90)/(T/R(11605.0)); R(2.0e-8)/(R(0.50)+u)*u^R(0.25)*exp(-u))
@inline cFe_ctf(T) = (R=typeof(T); R(1.0e-9))
@inline cFe_ctr(T) = (R=typeof(T); R(1.3e-9)*exp(-R(3.0e4)/T))
@inline _fion_Fe(T, nde, nHI, nHII) = (R=typeof(T);
    r = (cFe_ci(T)*nde + cFe_ctf(T)*nHII) / (cFe_rec(T)*nde + cFe_ctr(T)*nHI); r/(one(R)+r))

# =============================================================================
# Layer 3 — per-cell element abundances n(X)/n_H and linear yield-template mixing
# =============================================================================
"""
    MetalAbundances{T}(C, O, Si, Fe)

Per-cell number abundances n(X)/n_H for the tracked metals. isbits & parametric on
`T` (Dual-safe ⇒ ∂Λ/∂abundance differentiable). Zero ⇒ no metal cooling.
"""
struct MetalAbundances{T}
    C::T
    O::T
    Si::T
    Fe::T
end
MetalAbundances{T}() where {T} = MetalAbundances{T}(zero(T), zero(T), zero(T), zero(T))
# Promoting constructor: tolerate mixed arg types (e.g. one Dual abundance for
# ∂Λ/∂a via ForwardDiff, the rest Float64).
MetalAbundances(C, O, Si, Fe) = MetalAbundances(promote(C, O, Si, Fe)...)
@inline _no_metals(::Type{T}) where {T} = MetalAbundances{T}()
@inline _has_metals(a::MetalAbundances) = !(iszero(a.C) & iszero(a.O) & iszero(a.Si) & iszero(a.Fe))

# Solar (Asplund-ish; Enzo metal_cool.dat values for C/O/Si, Fe added) n(X)/n_H.
const METAL_SOLAR = (C=2.69e-4, O=4.90e-4, Si=3.24e-5, Fe=3.16e-5)
# Schematic yield-channel patterns (qualitative [α/Fe]; swap in Nomoto/Karakas for production).
const METAL_CCSN  = (C=1.9e-4, O=6.5e-4, Si=3.6e-5, Fe=1.3e-5)   # α-enhanced, Fe-poor
const METAL_IA    = (C=3.0e-6, O=1.4e-4, Si=1.5e-4, Fe=2.2e-4)   # Fe-peak, α-poor
const METAL_AGB   = (C=4.5e-4, O=2.0e-4, Si=5.0e-6, Fe=2.0e-6)   # C-rich

"""
    metal_abund(; solar=0, ccsn=0, ia=0, agb=0) -> MetalAbundances

Linear combination of the yield-channel templates, Σ_t w_t · pattern_t. Use host-side
to turn separately-tracked II/Ia/AGB (and a solar floor) into the per-cell element
abundance vector the kernel consumes — [α/Fe] then varies cell-to-cell with no change
to the atomic physics (Λ is linear in each abundance).
"""
@inline function metal_abund(; solar=0.0, ccsn=0.0, ia=0.0, agb=0.0)
    w(f) = solar*getfield(METAL_SOLAR,f) + ccsn*getfield(METAL_CCSN,f) +
           ia*getfield(METAL_IA,f) + agb*getfield(METAL_AGB,f)
    return MetalAbundances(w(:C), w(:O), w(:Si), w(:Fe))
end

# Smootherstep taper: 1 for T≤1e4, →0 over 1e4–2e4 K (C²; keeps Λ differentiable).
@inline function _hot_taper(T)
    R = typeof(T)
    x = clamp((R(2.0e4) - T) / R(1.0e4), zero(R), one(R))
    return x*x*x*(x*(x*R(6.0) - R(15.0)) + R(10.0))
end

# =============================================================================
# Assembler — total metal-line cooling [erg s⁻¹ cm⁻³], ≥0 (edot.jl subtracts it)
# =============================================================================
"""
    metal_cooling_rate(T, z, nHI, nHII, nde, nH2, nH, ab::MetalAbundances) -> Λ ≥ 0

Volumetric atomic/ionic metal-line cooling [erg s⁻¹ cm⁻³]. Sums C(I/II), O(I),
Si(I/II), Fe(II) fine-structure lines, each weighted by n_H·a_X·f_stage (live-n_e ion
balance). Smoothly tapered to 0 over 1e4–2e4 K. Pure, allocation-free, differentiable
in T and in `ab`.
"""
@inline function metal_cooling_rate(T, z, nHI, nHII, nde, nH2, nH, ab::MetalAbundances{R}) where {R}
    _has_metals(ab) || return zero(R)          # zero abundances ⇒ exactly 0, zero-cost
    T >= R(2.0e4) && return zero(R)
    Trad = comp2_cmb(R(z))
    nH2o = R(0.75)*nH2; nH2p = R(0.25)*nH2
    one_ = one(R)

    fC  = _fion_C(T, nde, nHI, nHII)
    ΛC  = ab.C * ((one_ - fC)*_cool_CI(T, Trad, nHI, nHII, nH2o, nH2p, nde) +
                  fC*_cool_CII(T, Trad, nHI, nH2o, nH2p, nde))
    ΛO  = ab.O * _cool_OI(T, Trad, nHI, nHII, nH2o, nH2p, nde)            # O locked neutral
    fSi = _fion_Si(T, nde, nHI, nHII)
    ΛSi = ab.Si * ((one_ - fSi)*_cool_SiI(T, Trad, nHI, nHII, nH2o, nH2p, nde) +
                   fSi*_cool_SiII(T, Trad, nHI, nde))
    fFe = _fion_Fe(T, nde, nHI, nHII)
    ΛFe = ab.Fe * fFe * _cool_FeII(T, Trad, nHI, nde)                     # Fe⁰ FS not modeled

    Λ = nH * (ΛC + ΛO + ΛSi + ΛFe)
    return T > R(1.0e4) ? Λ * _hot_taper(T) : Λ
end
