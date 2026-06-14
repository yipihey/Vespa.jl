# Deuterium reaction rates k50–k56 (precision-generic, table-free).
# Formulas transcribed exactly from grackle/src/clib/rate_functions.c.
# Units = CGS (units=1.0 convention — divide by units is a no-op here).
# Every literal is wrapped in R(...) = typeof(T)(...) so the function
# compiles to the correct precision on any floating-point type T.

export k50, k50_grid
export k51, k51_grid
export k52, k52_grid
export k53, k53_grid
export k54, k54_grid
export k55, k55_grid
export k56, k56_grid

# ── k50  HII + DI --> HI + DII  (Savin 2002) ────────────────────────────────
@inline function k50(T::Real)
    R = typeof(T)
    if T <= R(2.0e5)
        return R(2.0e-10) * T^R(0.402) * exp(-R(3.71e1)/T) -
               R(3.31e-17) * T^R(1.48)
    else
        return R(2.5e-8) * (T / R(2.0e5))^R(0.402)
    end
end
@scalarkernel k50

# ── k51  HI + DII --> HII + DI  (Savin 2002) ────────────────────────────────
@inline function k51(T::Real)
    R = typeof(T)
    return R(2.06e-10) * T^R(0.396) * exp(-R(3.30e1)/T) +
           R(2.03e-9)  * T^R(-0.332)
end
@scalarkernel k51

# ── k52  H2I + DII --> HDI + HII  (Galli & Palla 2002 / Gerlich 1982) ───────
@inline function k52(T::Real)
    R  = typeof(T)
    l  = log10(T)
    if T <= R(1.0e4)
        return R(1.0e-9) * (R(0.417) + R(0.846)*l - R(0.137)*l^2)
    else
        return R(1.609e-9)
    end
end
@scalarkernel k52

# ── k53  HDI + HII --> H2I + DII  (Galli & Palla 2002 / Gerlich 1982) ───────
@inline function k53(T::Real)
    R = typeof(T)
    return R(1.1e-9) * exp(-R(4.88e2)/T)
end
@scalarkernel k53

# ── k54  H2I + DI --> HDI + HI  (Clark et al 2011 / Mielke et al 2003) ──────
# Polynomial in log10(T) = log(T)/log(10), evaluated via Horner's method.
# Using log(T)*inv_ln10 rather than log10(T) keeps CPU↔Metal f32 agreement within
# RTOL_B = 1e-6 (Metal's log10 diverges from CPU libm more than its log does).
@inline function k54(T::Real)
    R = typeof(T)
    # log10(T) = log(T) / log(10); bake in log(10) as a constant
    l = log(T) * R(0.4342944819032518)   # ≡ log10(T)
    if T <= R(2.0e3)
        # Horner form of:
        #   -5.64737e1 + 5.88886·l + 7.19692·l² + 2.25069·l³
        #              - 2.16903·l⁴ + 3.17887e-1·l⁵
        expt = muladd(l, muladd(l, muladd(l, muladd(l, muladd(l,
                      R(3.17887e-1),
                      -R(2.16903)),
                      R(2.25069)),
                      R(7.19692)),
                      R(5.88886)),
                      -R(5.64737e1))
        return exp(expt * R(2.302585092994046))  # 10^expt = exp(ln(10)*expt)
    else
        return R(3.17e-10) * exp(-R(5.207e3)/T)
    end
end
@scalarkernel k54

# ── k55  HDI + HI --> H2I + DI  (Galli & Palla 2002 / Shavitt 1959) ─────────
@inline function k55(T::Real)
    R = typeof(T)
    if T <= R(2.0e2)
        return R(1.08e-22)
    else
        return R(5.25e-11) * exp(-R(4.43e3)/T + R(1.739e5)/T^2)
    end
end
@scalarkernel k55

# ── k56  DI + HM --> HDI + e  (Miller et al 2012; identical to k8) ──────────
# k8: HI + HM --> H2I* + e  (Kreckel et al 2010, Science 329 69)
@inline function k56(T::Real)
    R = typeof(T)
    num = R(1.35e-9) * (T^R(9.8493e-2)  +
                         R(3.2852e-1) * T^R(5.5610e-1) +
                         R(2.771e-7)  * T^R(2.1826))
    den = R(1.0) + R(6.191e-3)  * T^R(1.0461) +
                   R(8.9712e-11) * T^R(3.0424) +
                   R(3.2576e-14) * T^R(3.7741)
    return num / den
end
@scalarkernel k56
