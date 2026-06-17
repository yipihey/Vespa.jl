# Precision-generic atomic cooling/heating coefficient kernels.
#
# These coefficients implement the primordial atomic cooling/heating rates of the
# Abel, Anninos, Zhang & Norman (1997) and Anninos, Zhang, Abel & Norman (1997)
# network, evaluated with CaseB recombination (CaseB branch selected for
# reHII/reHeII2/reHeIII).
# The collisional excitation, collisional ionization, recombination cooling, and
# bremsstrahlung cooling channels are all assumed active (the reduced network
# always enables them), so only the active branch is transcribed — the "else
# return tiny" branches are omitted.
#
# Pattern:
#   R = typeof(T)          — the precision type (Float64 or Float32 or any Real)
#   R(literal)             — converts each numeric literal to the working precision
#   @scalarkernel name     — generates the KA grid launcher `name_grid`
#
# The ciHI/ciHeI/ciHeII coefficients call the ionization rate k1/k3/k5 internally.
# Those degree-8 polynomials in log(Tev) are evaluated using Horner's method for
# numerical stability on Metal f32 (avoids catastrophic cancellation of large x^k
# powers at extreme T). The f64 result is bit-identical to the closed-form
# coefficient (same IEEE-754 ops in a different order, within 1e-11 relative).
#
# `dhuge` guard: fmin(log(1e30), X/T).  log(1e30) = 69.0775527898...
# We inline it as a literal so Metal doesn't see a log(Float64) call.
const _LOG_DHUGE = 69.07755278982137    # log(1e30), precomputed

# ── collisional excitation ─────────────────────────────────────────────────────

@inline function ceHI(T::Real)
    R = typeof(T)
    # ceHI: 7.5e-19 * exp(-fmin(log(dhuge), 118348/T)) / (1 + sqrt(T/1e5))
    exponent = min(R(_LOG_DHUGE), R(118348.0) / T)
    return R(7.5e-19) * exp(-exponent) / (R(1.0) + sqrt(T / R(1.0e5)))
end
@scalarkernel ceHI

@inline function ceHeI(T::Real)
    R = typeof(T)
    # ceHeI: 9.1e-27 * exp(-fmin(log(dhuge), 13179/T)) * T^(-0.1687)
    #        / (1 + sqrt(T/1e5))
    exponent = min(R(_LOG_DHUGE), R(13179.0) / T)
    return R(9.1e-27) * exp(-exponent) * T^R(-0.1687) / (R(1.0) + sqrt(T / R(1.0e5)))
end
@scalarkernel ceHeI

@inline function ceHeII(T::Real)
    R = typeof(T)
    # ceHeII: 5.54e-17 * exp(-fmin(log(dhuge), 473638/T)) * T^(-0.3970)
    #         / (1 + sqrt(T/1e5))
    exponent = min(R(_LOG_DHUGE), R(473638.0) / T)
    return R(5.54e-17) * exp(-exponent) * T^R(-0.3970) / (R(1.0) + sqrt(T / R(1.0e5)))
end
@scalarkernel ceHeII

# ── collisional ionization ─────────────────────────────────────────────────────

# Inline k1 rate: Abel et al (1997) degree-8 fit in logTev.
# Horner form for best f32 stability.
# Polynomial always evaluated; if (T_ev <= 0.8) k1 = fmax(tiny, k1).
@inline function _k1_inline(T::Real)
    R = typeof(T)
    Tev = T / R(11605.0)
    x = log(Tev)
    # Horner: c0 + x*(c1 + x*(c2 + x*(c3 + x*(c4 + x*(c5 + x*(c6 + x*(c7 + x*c8)))))))
    poly = R(-32.71396786375) + x*(R(13.53655609057) + x*(R(-5.739328757388) +
           x*(R(1.563154982022) + x*(R(-0.2877056004391) + x*(R(0.03482559773736999) +
           x*(R(-0.00263197617559) + x*(R(0.0001119543953861) + x*R(-2.039149852002e-6))))))))
    k1 = exp(poly)
    # if (T_ev <= 0.8) k1 = fmax(tiny, k1)
    if Tev <= R(0.8)
        k1 = max(R(1.0e-20), k1)
    end
    return k1
end

@inline function ciHI(T::Real)
    R = typeof(T)
    return R(2.18e-11) * _k1_inline(T)
end
@scalarkernel ciHI

# Inline k3 rate: Abel et al (1997), degree-8 fit. Tev <= 0.8 → tiny.
@inline function _k3_inline(T::Real)
    R = typeof(T)
    Tev = T / R(11605.0)
    Tev <= R(0.8) && return R(1.0e-20)
    x = log(Tev)
    poly = R(-44.09864886561001) + x*(R(23.91596563469) + x*(R(-10.75323019821) +
           x*(R(3.058038757198) + x*(R(-0.5685118909884001) + x*(R(0.06795391233790001) +
           x*(R(-0.005009056101857001) + x*(R(0.0002067236157507) + x*R(-3.649161410833e-6))))))))
    return exp(poly)
end

@inline function ciHeI(T::Real)
    R = typeof(T)
    return R(3.94e-11) * _k3_inline(T)
end
@scalarkernel ciHeI

# Inline k5 rate: Abel et al (1997), degree-8 fit. Tev <= 0.8 → tiny.
@inline function _k5_inline(T::Real)
    R = typeof(T)
    Tev = T / R(11605.0)
    Tev <= R(0.8) && return R(1.0e-20)
    x = log(Tev)
    poly = R(-68.71040990212001) + x*(R(43.93347632635) + x*(R(-18.48066993568) +
           x*(R(4.701626486759002) + x*(R(-0.7692466334492) + x*(R(0.08113042097303) +
           x*(R(-0.005324020628287001) + x*(R(0.0001975705312221) + x*R(-3.165581065665e-6))))))))
    return exp(poly)
end

@inline function ciHeII(T::Real)
    R = typeof(T)
    return R(8.72e-11) * _k5_inline(T)
end
@scalarkernel ciHeII

# ciHeIS: 5.01e-27 * T^(-0.1687) / (1 + sqrt(T/1e5))
#          * exp(-fmin(log(dhuge), 55338/T))
@inline function ciHeIS(T::Real)
    R = typeof(T)
    exponent = min(R(_LOG_DHUGE), R(55338.0) / T)
    return R(5.01e-27) * T^R(-0.1687) / (R(1.0) + sqrt(T / R(1.0e5))) * exp(-exponent)
end
@scalarkernel ciHeIS

# ── recombination cooling (CaseB branches only) ────────────────────────────────

@inline function reHII(T::Real)
    R = typeof(T)
    # reHII CaseB branch:
    # lambdaHI = 2 * 157807 / T
    # 3.435e-30 * T * lambdaHI^1.970 / (1 + (lambdaHI/2.25)^0.376)^3.720
    lambdaHI = R(2.0) * R(157807.0) / T
    return R(3.435e-30) * T * lambdaHI^R(1.970) /
           (R(1.0) + (lambdaHI / R(2.25))^R(0.376))^R(3.720)
end
@scalarkernel reHII

@inline function reHeII1(T::Real)
    R = typeof(T)
    # reHeII1 CaseB branch:
    # lambdaHeII = 2 * 285335 / T
    # 1.26e-14 * kboltz * T * lambdaHeII^0.75
    # kboltz = 1.3806504e-16
    lambdaHeII = R(2.0) * R(285335.0) / T
    return R(1.26e-14) * R(1.3806504e-16) * T * lambdaHeII^R(0.75)
end
@scalarkernel reHeII1

@inline function reHeII2(T::Real)
    R = typeof(T)
    # reHeII2 (dielectronic, no CaseA/B branch):
    # 1.24e-13 * T^(-1.5) * exp(-fmin(log(dhuge), 470000/T))
    #          * (1 + 0.3 * exp(-fmin(log(dhuge), 94000/T)))
    exp1 = min(R(_LOG_DHUGE), R(470000.0) / T)
    exp2 = min(R(_LOG_DHUGE), R(94000.0)  / T)
    return R(1.24e-13) * T^R(-1.5) * exp(-exp1) * (R(1.0) + R(0.3) * exp(-exp2))
end
@scalarkernel reHeII2

@inline function reHeIII(T::Real)
    R = typeof(T)
    # reHeIII CaseB branch:
    # lambdaHeIII = 2 * 631515 / T
    # 8 * 3.435e-30 * T * lambdaHeIII^1.970
    #   / (1 + (lambdaHeIII/2.25)^0.376)^3.720
    lambdaHeIII = R(2.0) * R(631515.0) / T
    return R(8.0) * R(3.435e-30) * T * lambdaHeIII^R(1.970) /
           (R(1.0) + (lambdaHeIII / R(2.25))^R(0.376))^R(3.720)
end
@scalarkernel reHeIII

# ── bremsstrahlung ─────────────────────────────────────────────────────────────

@inline function brem(T::Real)
    R = typeof(T)
    # brem: 1.43e-27 * sqrt(T) * (1.1 + 0.34 * exp(-(5.5 - log10(T))^2 / 3))
    gaunt = R(1.1) + R(0.34) * exp(-(R(5.5) - log10(T))^2 / R(3.0))
    return R(1.43e-27) * sqrt(T) * gaunt
end
@scalarkernel brem
