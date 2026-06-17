# HD cooling-coefficient kernels — table-free, precision-generic.
#
# HD cooling coefficients HDlte and HDlow of the Abel/Anninos et al. 1997 network.
#
# HDlte: Coppola et al. 2011 LTE fit (T in [10, 3e4]).
# HDlow: Wrathmall, Gusdorf & Flower (2007) HD-H collisional excitation fit (T in [10, 6e3]).
#
# Every numeric literal is wrapped in R(...) for precision-genericity.
# log10(T) is computed as log(T)*R(0.4342944819032518) and 10^x as
# exp(x*R(2.302585092994046)) to avoid f64 log10 calls on Metal f32.
# Polynomial evaluated via Horner's method for f32 stability on Metal.

export HDlte, HDlow
export HDlte_grid, HDlow_grid

# ── HDlte: HD LTE cooling rate (Coppola et al. 2011) ─────────────────────────
# Constrain T to [10, 3e4].
#   HDlte = -55.5725 + 56.649*log10(tm) - 37.9102*log10(tm)^2 + 12.698*log10(tm)^3
#            - 2.02424*log10(tm)^4 + 0.122393*log10(tm)^5
#   return pow(10, min(HDlte, 0)) / units
# NOTE: polynomial is in log10(tm) directly (NOT log10(tm/1e3)).
@inline function HDlte(T::Real)
    R = typeof(T)
    tm  = max(T, R(10.0))
    tm  = min(tm, R(3.0e4))
    ltm = log(tm) * R(0.4342944819032518)   # log10(tm)
    # Horner: c0 + ltm*(c1 + ltm*(c2 + ltm*(c3 + ltm*(c4 + ltm*c5))))
    poly = R(-55.5725) + ltm*(R(56.649) + ltm*(R(-37.9102) +
           ltm*(R(12.698) + ltm*(R(-2.02424) + ltm*R(0.122393)))))
    return exp(min(poly, R(0.0)) * R(2.302585092994046))
end
@scalarkernel HDlte

# ── HDlow: HD low-density cooling rate (Wrathmall, Gusdorf & Flower 2007) ─────
# Constrain T to [10, 6e3]; lt3 = log10(tm/1e3).
#   HDlow = -23.175780 + 1.5035261*lt3 + 0.40871403*lt3^2 + 0.17849311*lt3^3
#            - 0.077291388*lt3^4 + 0.10031326*lt3^5
#   return pow(10, HDlow) / units
@inline function HDlow(T::Real)
    R = typeof(T)
    tm  = max(T, R(1.0e1))
    tm  = min(tm, R(6.0e3))
    lt3 = log(tm / R(1.0e3)) * R(0.4342944819032518)
    # Horner
    poly = R(-23.175780) + lt3*(R(1.5035261) + lt3*(R(0.40871403) +
           lt3*(R(0.17849311) + lt3*(R(-0.077291388) + lt3*R(0.10031326)))))
    return exp(poly * R(2.302585092994046))
end
@scalarkernel HDlow
