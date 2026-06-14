# rates_atomic.jl — precision-generic atomic H/He reaction-rate kernels.
#
# Each function is a pure scalar formula of temperature T [K], returning the
# reaction rate coefficient in CGS (cm³/s), matching grackle's
# `kN_rate(T, 1.0, cd)` with CaseBRecombination=1 to floating-point round-off.
#
# All functions run as real device kernels via @scalarkernel (CPU f64/f32 +
# Metal f32).  k1/k3/k5 are degree-8 exp(poly(logTev)) Abel-1997 fits whose f32
# evaluation cancels only at T > ~1e8 K (outside the primordial regime); the
# verification ladder checks the f32 layers over T ≤ 1e8 K accordingly.
#
# Rules throughout: R = typeof(T); every numeric literal in R(...); fractional
# exponents in R(...) (e.g. T^R(-1.5)); coefficients copied verbatim from
# grackle/src/clib/rate_functions.c.

# ── k1 : HI + e → HII + 2e ───────────────────────────────────────────────────
# Fit: Abel et al. (1997), degree-8 exp(poly(logTev)).
# Branch: always evaluate poly; at Tev≤0.8 return max(tiny, poly_val)
# (matches grackle's fmax(tiny, k1) at Tev≤0.8 — does NOT early-return,
# since the boundary T = 9284 K lies in the log-spaced tgrid and the poly
# value there is 2.23e-16, not tiny).
@inline function k1(T::Real)
    R = typeof(T)
    Tev = T / R(11605.0)
    x = log(Tev)
    val = exp(evalpoly(x, (R(-32.71396786375), R(13.53655609057), R(-5.739328757388),
              R(1.563154982022), R(-0.2877056004391), R(0.03482559773736999),
              R(-0.00263197617559), R(0.0001119543953861), R(-2.039149852002e-6))))
    return Tev <= R(0.8) ? max(R(1e-20), val) : val
end
@scalarkernel k1

# ── k3 : HeI + e → HeII + 2e ─────────────────────────────────────────────────
# Fit: Abel et al. (1997), degree-8 exp(poly(logTev)).
# Branch: Tev≤0.8 → tiny (grackle returns tiny, not fmax here).
@inline function k3(T::Real)
    R = typeof(T)
    Tev = T / R(11605.0)
    Tev <= R(0.8) && return R(1e-20)
    x = log(Tev)
    return exp(evalpoly(x, (R(-44.09864886561001), R(23.91596563469), R(-10.75323019821),
               R(3.058038757198), R(-0.5685118909884001), R(0.06795391233790001),
               R(-0.005009056101857001), R(0.0002067236157507), R(-3.649161410833e-6))))
end
@scalarkernel k3

# ── k5 : HeII + e → HeIII + 2e ───────────────────────────────────────────────
# Fit: Abel et al. (1997), degree-8 exp(poly(logTev)).
# Branch: Tev≤0.8 → tiny.
@inline function k5(T::Real)
    R = typeof(T)
    Tev = T / R(11605.0)
    Tev <= R(0.8) && return R(1e-20)
    x = log(Tev)
    return exp(evalpoly(x, (R(-68.71040990212001), R(43.93347632635), R(-18.48066993568),
               R(4.701626486759002), R(-0.7692466334492), R(0.08113042097303),
               R(-0.005324020628287001), R(0.0001975705312221), R(-3.165581065665e-6))))
end
@scalarkernel k5

# ── k2 : HII + e → HI + photon  (CaseB recombination) ───────────────────────
# Fit: Hui & Gnedin (1997) CaseB. Power-law: f32 accurate to ~5e-7 everywhere.
# Note: grackle returns tiny for T≥1e9, but evaluating the formula for all T
# avoids a Metal f32 quantization artifact (Float32(1e9-1) = 1.0f9 = 1e9,
# which would falsely trigger the branch and return tiny instead of ~1.46e-19).
@inline function k2(T::Real)
    R = typeof(T)
    return R(4.881357e-6) * T^R(-1.5) * (R(1.0) + R(1.14813e2)*T^R(-0.407))^R(-2.242)
end
@scalarkernel k2

# ── k4 : HeII + e → HeI + photon  (CaseB recombination) ─────────────────────
# Fit: Hui & Gnedin (1997) CaseB. No branch in the CaseB path.
@inline function k4(T::Real)
    R = typeof(T)
    return R(1.26e-14) * (R(5.7067e5) / T)^R(0.75)
end
@scalarkernel k4

# ── k6 : HeIII + e → HeII + photon  (CaseB recombination) ───────────────────
# Fit: Hui & Gnedin (1997) CaseB. Same quantization note as k2.
@inline function k6(T::Real)
    R = typeof(T)
    return R(7.8155e-5) * T^R(-1.5) * (R(1.0) + R(2.0189e2)*T^R(-0.407))^R(-2.242)
end
@scalarkernel k6

# ── k57 : HI + HI → HII + HI + e  (collisional ionisation) ──────────────────
# Fit: Lenzuni, Chernoff & Salpeter (1991); cross-sections: Gealy & van Zyl (1987).
# Branch: T≤3e3 → tiny.
@inline function k57(T::Real)
    R = typeof(T)
    T <= R(3.0e3) && return R(1e-20)
    return R(1.2e-17) * T^R(1.2) * exp(-R(1.578e5) / T)
end
@scalarkernel k57

# ── k58 : HI + HeI → HII + HeI + e  (collisional ionisation) ────────────────
# Fit: Lenzuni, Chernoff & Salpeter (1991); cross-sections: van Zyl, Le & Amme (1981).
# Branch: T≤3e3 → tiny.
@inline function k58(T::Real)
    R = typeof(T)
    T <= R(3.0e3) && return R(1e-20)
    return R(1.75e-17) * T^R(1.3) * exp(-R(1.578e5) / T)
end
@scalarkernel k58
