# Exact 1D Riemann solver for the ideal-gas Euler equations (Toro, "Riemann
# Solvers and Numerical Methods for Fluid Dynamics", ch. 4). Used purely as the
# analytic oracle for the Sod shock tube — not in the time loop.
#
# States are 1D primitive tuples `(ρ, u, p)`. `exact_riemann_sample` returns the
# self-similar solution at similarity coordinate `s = (x - x0) / t`.

@inline _sound(ρ, p, γ) = sqrt(γ * p / ρ)

# Pressure function and derivative for one side (Toro eqs. 4.6–4.7).
function _pressure_fn(p, ρK, pK, cK, γ)
    if p > pK                                   # shock
        AK = 2 / ((γ + 1) * ρK)
        BK = (γ - 1) / (γ + 1) * pK
        f = (p - pK) * sqrt(AK / (p + BK))
        df = sqrt(AK / (BK + p)) * (1 - (p - pK) / (2 * (BK + p)))
    else                                        # rarefaction
        f = 2 * cK / (γ - 1) * ((p / pK)^((γ - 1) / (2γ)) - 1)
        df = 1 / (ρK * cK) * (p / pK)^(-(γ + 1) / (2γ))
    end
    return f, df
end

"Solve for star-region pressure and velocity `(p*, u*)` by Newton iteration."
function _star_state(WL, WR, γ; tol = 1e-12, maxit = 100)
    ρL, uL, pL = WL
    ρR, uR, pR = WR
    cL = _sound(ρL, pL, γ)
    cR = _sound(ρR, pR, γ)

    p = max(tol, 0.5 * (pL + pR))               # initial guess
    for _ in 1:maxit
        fL, dfL = _pressure_fn(p, ρL, pL, cL, γ)
        fR, dfR = _pressure_fn(p, ρR, pR, cR, γ)
        f = fL + fR + (uR - uL)
        p_new = p - f / (dfL + dfR)
        p_new = max(tol, p_new)
        if abs(p_new - p) / (0.5 * (p_new + p)) < tol
            p = p_new
            break
        end
        p = p_new
    end
    fL, _ = _pressure_fn(p, ρL, pL, cL, γ)
    fR, _ = _pressure_fn(p, ρR, pR, cR, γ)
    u = 0.5 * (uL + uR) + 0.5 * (fR - fL)
    return p, u
end

"""
    exact_riemann_sample(WL, WR, γ, s) -> (ρ, u, p)

Exact solution of the 1D Riemann problem with left/right primitive states
`WL = (ρL, uL, pL)`, `WR = (ρR, uR, pR)`, sampled at similarity speed `s = x/t`.
"""
function exact_riemann_sample(WL, WR, γ, s)
    ρL, uL, pL = WL
    ρR, uR, pR = WR
    cL = _sound(ρL, pL, γ)
    cR = _sound(ρR, pR, γ)
    pstar, ustar = _star_state(WL, WR, γ)

    if s <= ustar
        # Left of the contact discontinuity.
        if pstar > pL                            # left shock
            SL = uL - cL * sqrt((γ + 1) / (2γ) * pstar / pL + (γ - 1) / (2γ))
            if s <= SL
                return (ρL, uL, pL)
            else
                ρstar = ρL * (pstar / pL + (γ - 1) / (γ + 1)) /
                        ((γ - 1) / (γ + 1) * pstar / pL + 1)
                return (ρstar, ustar, pstar)
            end
        else                                     # left rarefaction
            SHL = uL - cL
            cstar = cL * (pstar / pL)^((γ - 1) / (2γ))
            STL = ustar - cstar
            if s <= SHL
                return (ρL, uL, pL)
            elseif s >= STL
                ρstar = ρL * (pstar / pL)^(1 / γ)
                return (ρstar, ustar, pstar)
            else                                 # inside the fan
                u = 2 / (γ + 1) * (cL + (γ - 1) / 2 * uL + s)
                c = 2 / (γ + 1) * (cL + (γ - 1) / 2 * (uL - s))
                ρ = ρL * (c / cL)^(2 / (γ - 1))
                p = pL * (c / cL)^(2γ / (γ - 1))
                return (ρ, u, p)
            end
        end
    else
        # Right of the contact discontinuity.
        if pstar > pR                            # right shock
            SR = uR + cR * sqrt((γ + 1) / (2γ) * pstar / pR + (γ - 1) / (2γ))
            if s >= SR
                return (ρR, uR, pR)
            else
                ρstar = ρR * (pstar / pR + (γ - 1) / (γ + 1)) /
                        ((γ - 1) / (γ + 1) * pstar / pR + 1)
                return (ρstar, ustar, pstar)
            end
        else                                     # right rarefaction
            SHR = uR + cR
            cstar = cR * (pstar / pR)^((γ - 1) / (2γ))
            STR = ustar + cstar
            if s >= SHR
                return (ρR, uR, pR)
            elseif s <= STR
                ρstar = ρR * (pstar / pR)^(1 / γ)
                return (ρstar, ustar, pstar)
            else                                 # inside the fan
                u = 2 / (γ + 1) * (-cR + (γ - 1) / 2 * uR + s)
                c = 2 / (γ + 1) * (cR - (γ - 1) / 2 * (uR - s))
                ρ = ρR * (c / cR)^(2 / (γ - 1))
                p = pR * (c / cR)^(2γ / (γ - 1))
                return (ρ, u, p)
            end
        end
    end
end
