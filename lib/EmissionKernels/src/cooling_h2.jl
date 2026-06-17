# H₂ cooling-coefficient kernels — table-free, precision-generic.
#
# H₂ collisional de-excitation cooling coefficients GAHI, GAH2, GAHe, GAHp, GAel,
# H2LTE of the Abel/Anninos et al. 1997 network, using the Glover & Abel 2008
# fits.
#
# GAHI: H₂–H cooling uses the Lique 2015 rate.
#
# Temperature range clamping and branch structure are kept exactly as specified —
# do NOT simplify.
#
# Every numeric literal is wrapped in R(...) for precision-genericity.
# log10(T) is computed as log(T)*R(0.4342944819032518) and 10^x as
# exp(x*R(2.302585092994046)) to avoid f64 log10 calls on Metal f32.
# Polynomial in lt3 evaluated via Horner's method for f32 stability on Metal.

export GAHI, GAH2, GAHe, GAHp, GAel, H2LTE
export GAHI_grid, GAH2_grid, GAHe_grid, GAHp_grid, GAel_grid, H2LTE_grid

# ── GAHI: H₂ cooling by H collisions (Lique 2015) ────────────────────────────
# Constrain T to [10, 1e4]; lt3 = log10(tm/1e3).
# Returns 0 if tm < 100.
# pow(10, -24.07950609 + 4.54182810*lt3 - 2.40206896*lt3^2
#               - 0.75355292*lt3^3 + 4.69258178*lt3^4 - 2.79573574*lt3^5
#               - 3.14766075*lt3^6 + 2.50751333*lt3^7) / units
@inline function GAHI(T::Real)
    R = typeof(T)
    tm  = max(T, R(10.0))
    tm  = min(tm, R(1.0e4))
    tm < R(1.0e2) && return R(0.0)
    lt3 = log(tm / R(1.0e3)) * R(0.4342944819032518)
    # Horner: c0 + lt3*(c1 + lt3*(c2 + lt3*(c3 + lt3*(c4 + lt3*(c5 + lt3*(c6 + lt3*c7))))))
    poly = R(-24.07950609) + lt3*(R(4.54182810) + lt3*(R(-2.40206896) +
           lt3*(R(-0.75355292) + lt3*(R(4.69258178) + lt3*(R(-2.79573574) +
           lt3*(R(-3.14766075) + lt3*R(2.50751333)))))))
    return exp(poly * R(2.302585092994046))
end
@scalarkernel GAHI

# ── GAH2: H₂ cooling by H₂ collisions (Glover & Abel 2008) ───────────────────
# Constrain T to [10, 1e4]; lt3 = log10(tm/1e3).
# pow(10, -23.962112 + 2.09433740*lt3 - 0.77151436*lt3^2
#               + 0.43693353*lt3^3 - 0.14913216*lt3^4 - 0.033638326*lt3^5) / units
@inline function GAH2(T::Real)
    R = typeof(T)
    tm  = max(T, R(10.0))
    tm  = min(tm, R(1.0e4))
    lt3 = log(tm / R(1.0e3)) * R(0.4342944819032518)
    poly = R(-23.962112) + lt3*(R(2.09433740) + lt3*(R(-0.77151436) +
           lt3*(R(0.43693353) + lt3*(R(-0.14913216) + lt3*R(-0.033638326)))))
    return exp(poly * R(2.302585092994046))
end
@scalarkernel GAH2

# ── GAHe: H₂ cooling by He collisions (Glover & Abel 2008) ───────────────────
# Constrain T to [10, 1e4]; lt3 = log10(tm/1e3).
# pow(10, -23.689237 + 2.1892372*lt3 - 0.81520438*lt3^2
#               + 0.29036281*lt3^3 - 0.16596184*lt3^4 + 0.19191375*lt3^5) / units
@inline function GAHe(T::Real)
    R = typeof(T)
    tm  = max(T, R(10.0))
    tm  = min(tm, R(1.0e4))
    lt3 = log(tm / R(1.0e3)) * R(0.4342944819032518)
    poly = R(-23.689237) + lt3*(R(2.1892372) + lt3*(R(-0.81520438) +
           lt3*(R(0.29036281) + lt3*(R(-0.16596184) + lt3*R(0.19191375)))))
    return exp(poly * R(2.302585092994046))
end
@scalarkernel GAHe

# ── GAHp: H₂ cooling by H⁺ collisions (Glover & Abel 2008) ──────────────────
# Constrain T to [10, 1e4]; lt3 = log10(tm/1e3).
# pow(10, -22.089523 + 1.5714711*lt3 + 0.015391166*lt3^2
#               - 0.23619985*lt3^3 - 0.51002221*lt3^4 + 0.32168730*lt3^5) / units
@inline function GAHp(T::Real)
    R = typeof(T)
    tm  = max(T, R(10.0))
    tm  = min(tm, R(1.0e4))
    lt3 = log(tm / R(1.0e3)) * R(0.4342944819032518)
    poly = R(-22.089523) + lt3*(R(1.5714711) + lt3*(R(0.015391166) +
           lt3*(R(-0.23619985) + lt3*(R(-0.51002221) + lt3*R(0.32168730)))))
    return exp(poly * R(2.302585092994046))
end
@scalarkernel GAHp

# ── GAel: H₂ cooling by electron collisions (Glover & Abel 2008) ─────────────
# Constrain T to [10, 1e4]; lt3 = log10(tm/1e3).
# Returns 0 if tm < 100.
# Two polynomial branches: 100 ≤ tm < 500 (degree 8), tm ≥ 500 (degree 8).
# Branch boundary: (tm < 100.0) → 0, (tm < 500.0) → first poly, else second.
@inline function GAel(T::Real)
    R = typeof(T)
    tm  = max(T, R(10.0))
    tm  = min(tm, R(1.0e4))
    tm < R(100.0) && return R(0.0)
    lt3 = log(tm / R(1.0e3)) * R(0.4342944819032518)
    if tm < R(500.0)
        # pow(10, -21.928796 + 16.815730*lt3 + 96.743155*lt3^2 + 343.19180*lt3^3
        #         + 734.71651*lt3^4 + 983.67576*lt3^5 + 801.81247*lt3^6
        #         + 364.14446*lt3^7 + 70.609154*lt3^8) / units
        poly = R(-21.928796) + lt3*(R(16.815730) + lt3*(R(96.743155) +
               lt3*(R(343.19180) + lt3*(R(734.71651) + lt3*(R(983.67576) +
               lt3*(R(801.81247) + lt3*(R(364.14446) + lt3*R(70.609154))))))))
    else
        # pow(10, -22.921189 + 1.6802758*lt3 + 0.93310622*lt3^2 + 4.0406627*lt3^3
        #         - 4.7274036*lt3^4 - 8.8077017*lt3^5 + 8.9167183*lt3^6
        #         + 6.4380698*lt3^7 - 6.3701156*lt3^8) / units
        poly = R(-22.921189) + lt3*(R(1.6802758) + lt3*(R(0.93310622) +
               lt3*(R(4.0406627) + lt3*(R(-4.7274036) + lt3*(R(-8.8077017) +
               lt3*(R(8.9167183) + lt3*(R(6.4380698) + lt3*R(-6.3701156))))))))
    end
    return exp(poly * R(2.302585092994046))
end
@scalarkernel GAel

# ── H2LTE: H₂ LTE cooling rate (Glover & Abel 2008) ─────────────────────────
# Constrain T to [10, 1e4]; lt3 = log10(tm/1e3).
# tm < 100: simple power-law extrapolation 7e-27 * tm^1.5 * exp(-512/tm).
# tm >= 100: degree-8 polynomial in lt3.
# pow(10, -20.584225 + 5.0194035*lt3 - 1.5738805*lt3^2 - 4.7155769*lt3^3
#               + 2.4714161*lt3^4 + 5.4710750*lt3^5 - 3.9467356*lt3^6
#               - 2.2148338*lt3^7 + 1.8161874*lt3^8) / units
@inline function H2LTE(T::Real)
    R = typeof(T)
    tm  = max(T, R(10.0))
    tm  = min(tm, R(1.0e4))
    if tm < R(1.0e2)
        return R(7.0e-27) * tm^R(1.5) * exp(R(-512.0) / tm)
    end
    lt3 = log(tm / R(1.0e3)) * R(0.4342944819032518)
    poly = R(-20.584225) + lt3*(R(5.0194035) + lt3*(R(-1.5738805) +
           lt3*(R(-4.7155769) + lt3*(R(2.4714161) + lt3*(R(5.4710750) +
           lt3*(R(-3.9467356) + lt3*(R(-2.2148338) + lt3*R(1.8161874))))))))
    return exp(poly * R(2.302585092994046))
end
@scalarkernel H2LTE
