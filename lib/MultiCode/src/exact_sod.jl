# ── exact Riemann solution for the Sod comparison (Toro §4) ───────────────────
#
# Self-contained ideal-gas exact Riemann solver: star-region pressure by
# Newton iteration on the pressure function, then self-similar sampling in
# ξ = (x − x₀)/t.  This is the shared oracle all three codes are compared
# against; its own correctness is cross-checked by three independent codes
# converging to it in the report.

"""
    exact_sod(ξ; rhoL=1.0, uL=0.0, pL=1.0, rhoR=0.125, uR=0.0, pR=0.1, gamma=1.4)
        -> (; rho, u, p)

Exact ideal-gas Riemann solution sampled at the similarity coordinate
`ξ = (x - x₀)/t`.
"""
function exact_sod(ξ::Real; rhoL = 1.0, uL = 0.0, pL = 1.0,
                   rhoR = 0.125, uR = 0.0, pR = 0.1, gamma = 1.4)
    γ = gamma
    aL = sqrt(γ * pL / rhoL); aR = sqrt(γ * pR / rhoR)
    g1 = (γ - 1) / (2γ); g2 = (γ + 1) / (2γ); g3 = 2γ / (γ - 1)
    g4 = 2 / (γ - 1); g5 = 2 / (γ + 1); g6 = (γ - 1) / (γ + 1)
    g7 = (γ - 1) / 2

    # pressure function for one side: shock (p>pK) or rarefaction (p≤pK)
    function fK(p, rhoK, pK, aK)
        if p > pK
            AK = g5 / rhoK; BK = g6 * pK
            f = (p - pK) * sqrt(AK / (p + BK))
            df = sqrt(AK / (BK + p)) * (1 - (p - pK) / (2 * (BK + p)))
            return f, df
        else
            f = g4 * aK * ((p / pK)^g1 - 1)
            df = (1 / (rhoK * aK)) * (p / pK)^(-g2)
            return f, df
        end
    end

    # star pressure: Newton from the PV (two-rarefaction) guess
    du = uR - uL
    p = max(1e-12, ((aL + aR - g7 * du) / (aL / pL^g1 + aR / pR^g1))^g3)
    for _ in 1:60
        fL, dL = fK(p, rhoL, pL, aL)
        fR, dR = fK(p, rhoR, pR, aR)
        dp = (fL + fR + du) / (dL + dR)
        p = max(1e-12, p - dp)
        abs(dp) < 1e-14 * p && break
    end
    pstar = p
    ustar = 0.5 * (uL + uR) + 0.5 * (fK(pstar, rhoR, pR, aR)[1] - fK(pstar, rhoL, pL, aL)[1])

    # sample at ξ
    if ξ <= ustar                                   # left of the contact
        if pstar > pL                               # left shock
            sL = uL - aL * sqrt(g2 * pstar / pL + g1)
            if ξ <= sL
                return (rho = rhoL, u = uL, p = pL)
            else
                ρ = rhoL * ((pstar / pL + g6) / (g6 * pstar / pL + 1))
                return (rho = ρ, u = ustar, p = pstar)
            end
        else                                        # left rarefaction
            shL = uL - aL
            astar = aL * (pstar / pL)^g1
            stL = ustar - astar
            if ξ <= shL
                return (rho = rhoL, u = uL, p = pL)
            elseif ξ >= stL
                ρ = rhoL * (pstar / pL)^(1 / γ)
                return (rho = ρ, u = ustar, p = pstar)
            else                                    # inside the fan
                u = g5 * (aL + g7 * uL + ξ)
                a = g5 * (aL + g7 * (uL - ξ))
                ρ = rhoL * (a / aL)^g4
                return (rho = ρ, u = u, p = pL * (a / aL)^g3)
            end
        end
    else                                            # right of the contact
        if pstar > pR                               # right shock
            sR = uR + aR * sqrt(g2 * pstar / pR + g1)
            if ξ >= sR
                return (rho = rhoR, u = uR, p = pR)
            else
                ρ = rhoR * ((pstar / pR + g6) / (g6 * pstar / pR + 1))
                return (rho = ρ, u = ustar, p = pstar)
            end
        else                                        # right rarefaction
            shR = uR + aR
            astar = aR * (pstar / pR)^g1
            stR = ustar + astar
            if ξ >= shR
                return (rho = rhoR, u = uR, p = pR)
            elseif ξ <= stR
                ρ = rhoR * (pstar / pR)^(1 / γ)
                return (rho = ρ, u = ustar, p = pstar)
            else
                u = g5 * (-aR + g7 * uR + ξ)
                a = g5 * (aR - g7 * (uR - ξ))
                ρ = rhoR * (a / aR)^g4
                return (rho = ρ, u = u, p = pR * (a / aR)^g3)
            end
        end
    end
end
