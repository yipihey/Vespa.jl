# H₂/H⁻ reaction-rate kernels — table-free, precision-generic.
#
# Each function evaluates the corresponding rate coefficient kN(T) of the
# Abel/Anninos et al. 1997 network with CGS units (units=1.0 convention).
#
# Channel choices assumed:
#   h2_charge_exchange_rate = 1  (Savin 2004)
#   three_body_rate         = 0  (Abel et al.)
#
# Every numeric literal is wrapped in R(...) so the math is fully generic in T.
# `TINY` is imported from ChemistryKernels (= 1e-20).

export k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k22
export k7_grid, k8_grid, k9_grid, k10_grid, k11_grid, k12_grid, k13_grid,
       k14_grid, k15_grid, k16_grid, k17_grid, k18_grid, k19_grid, k22_grid

# ── k7: HI + e → HM + photon ─────────────────────────────────────────────────
# Stancil, Lepp & Dalgarno (1998), based on Wishart (1979).
@inline function k7(T::Real)
    R = typeof(T)
    return R(3.0e-16) * (T / R(300.0))^R(0.95) * exp(-T / R(9.32e3))
end
@scalarkernel k7

# ── k8: HI + HM → H2I* + e ───────────────────────────────────────────────────
# Kreckel et al. (2010, Science, 329, 69).
@inline function k8(T::Real)
    R = typeof(T)
    num = R(9.8493e-2)
    return R(1.35e-9) * (T^num
            + R(3.2852e-1) * T^R(5.5610e-1)
            + R(2.771e-7)  * T^R(2.1826)) /
           (R(1.0)
            + R(6.191e-3)  * T^R(1.0461)
            + R(8.9712e-11)* T^R(3.0424)
            + R(3.2576e-14)* T^R(3.7741))
end
@scalarkernel k8

# ── k9: HI + HII → H2II + photon ────────────────────────────────────────────
# Latif et al. (2015, MNRAS, 446, 3163): valid for 1 < T < 32000 K.
@inline function k9(T::Real)
    R = typeof(T)
    if T < R(30.0)
        return R(2.10e-20) * (T / R(30.0))^R(-0.15)
    else
        Tk = min(T, R(3.2e4))
        lTk = log10(Tk)
        return R(10.0)^(R(-18.20) - R(3.194) * lTk
                        + R(1.786) * lTk^R(2) - R(0.2072) * lTk^R(3))
    end
end
@scalarkernel k9

# ── k10: H2II + HI → H2I* + HII ──────────────────────────────────────────────
@inline function k10(T::Real)
    R = typeof(T)
    return R(6.0e-10)
end
@scalarkernel k10

# ── k11: H2I + HII → H2II + HI ───────────────────────────────────────────────
# Default: h2_charge_exchange_rate = 1 → Savin (2004) fit in logT (Kelvin).
@inline function k11(T::Real)
    R = typeof(T)
    Tev = T / R(11605.0)
    if Tev > R(0.3)
        lT = log(T)
        return exp(-R(21237.15) / T) * (
                 R(-3.3232183e-07)
               + R(3.3735382e-07)  * lT
               - R(1.4491368e-07)  * lT^R(2)
               + R(3.4172805e-08)  * lT^R(3)
               - R(4.7813720e-09)  * lT^R(4)
               + R(3.9731542e-10)  * lT^R(5)
               - R(1.8171411e-11)  * lT^R(6)
               + R(3.5311932e-13)  * lT^R(7))
    else
        return R(TINY)
    end
end
@scalarkernel k11

# ── k12: H2I + e → 2HI + e ───────────────────────────────────────────────────
# Trevisan & Tennyson (2002, Plasma Phys. Cont. Fus., 44, 1263).
@inline function k12(T::Real)
    R = typeof(T)
    Tev = T / R(11605.0)
    if Tev > R(0.3)
        return R(4.4886e-9) * T^R(0.109127) * exp(-R(101858.0) / T)
    else
        return R(TINY)
    end
end
@scalarkernel k12

# ── k13: H2I + HI → 3HI ──────────────────────────────────────────────────────
# Default: three_body_rate = 0 → Abel et al. (1997) inverse dissociation rate.
@inline function k13(T::Real)
    R = typeof(T)
    Tev = T / R(11605.0)
    if Tev > R(0.3)
        return R(1.0670825e-10) * Tev^R(2.012) /
               (exp(R(4.463) / Tev) * (R(1.0) + R(0.2472) * Tev)^R(3.512))
    else
        return R(TINY)
    end
end
@scalarkernel k13

# ── k14: HM + e → HI + 2e ────────────────────────────────────────────────────
# Degree-8 exp-poly fit in ln(T_ev).
@inline function k14(T::Real)
    R = typeof(T)
    Tev = T / R(11605.0)
    if Tev > R(0.04)
        lTev = log(Tev)
        return exp(R(-18.01849334273)
                 + R(2.360852208681)     * lTev
                 - R(0.2827443061704)    * lTev^R(2)
                 + R(0.01623316639567)   * lTev^R(3)
                 - R(0.03365012031362999)* lTev^R(4)
                 + R(0.01178329782711)   * lTev^R(5)
                 - R(0.001656194699504)  * lTev^R(6)
                 + R(0.0001068275202678) * lTev^R(7)
                 - R(2.631285809207e-6)  * lTev^R(8))
    else
        return R(TINY)
    end
end
@scalarkernel k14

# ── k15: HM + HI → 2HI + e ───────────────────────────────────────────────────
# Degree-9 exp-poly fit in ln(T_ev) for Tev > 0.1; power law below.
@inline function k15(T::Real)
    R = typeof(T)
    Tev = T / R(11605.0)
    if Tev > R(0.1)
        lTev = log(Tev)
        return exp(R(-20.37260896533324)
                 + R(1.139449335841631)   * lTev
                 - R(0.1421013521554148)  * lTev^R(2)
                 + R(0.00846445538663)    * lTev^R(3)
                 - R(0.0014327641212992)  * lTev^R(4)
                 + R(0.0002012250284791)  * lTev^R(5)
                 + R(0.0000866396324309)  * lTev^R(6)
                 - R(0.00002585009680264) * lTev^R(7)
                 + R(2.4555011970392e-6)  * lTev^R(8)
                 - R(8.06838246118e-8)    * lTev^R(9))
    else
        return R(2.56e-9) * Tev^R(1.78186)
    end
end
@scalarkernel k15

# ── k16: HM + HI → 2HI ───────────────────────────────────────────────────────
# Croft et al. (1999), based on Fussen & Kubach (1986).
@inline function k16(T::Real)
    R = typeof(T)
    return R(2.4e-6) * (R(1.0) + T / R(2.0e4)) / sqrt(T)
end
@scalarkernel k16

# ── k17: HM + HI → H2I + e ───────────────────────────────────────────────────
@inline function k17(T::Real)
    R = typeof(T)
    if T > R(1.0e4)
        return R(4.0e-4) * T^R(-1.4) * exp(-R(15100.0) / T)
    else
        return R(1.0e-8) * T^R(-0.4)
    end
end
@scalarkernel k17

# ── k18: H2I + e → 2HI ───────────────────────────────────────────────────────
@inline function k18(T::Real)
    R = typeof(T)
    if T > R(617.0)
        return R(1.32e-6) * T^R(-0.76)
    else
        return R(1.0e-8)
    end
end
@scalarkernel k18

# ── k19: H2I + HM → H2I + HI ────────────────────────────────────────────────
@inline function k19(T::Real)
    R = typeof(T)
    return R(5.0e-7) * sqrt(R(100.0) / T)
end
@scalarkernel k19

# ── k22: 2HI + HI → H2I + HI ────────────────────────────────────────────────
# Default: three_body_rate = 0.
@inline function k22(T::Real)
    R = typeof(T)
    if T <= R(300.0)
        return R(1.3e-32) * (T / R(300.0))^R(-0.38)
    else
        return R(1.3e-32) * (T / R(300.0))^R(-1.0)
    end
end
@scalarkernel k22
