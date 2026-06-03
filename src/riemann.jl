# Approximate Riemann solvers (kernels, ADR P1): stateless functions returning
# the conserved-flux vector across a face for given left/right primitive states.
#
# `dim` selects the face normal (1→x, 2→y, 3→z); the normal momentum/velocity is
# component `dim`, the others are tangential. This keeps a single implementation
# valid for directionally-split sweeps in any dimension.

"Physical flux of conserved variables for primitive state `W`, normal to `dim`."
@inline function euler_flux(W::NTuple{5,T}, γ, dim::Int) where {T}
    ρ, vx, vy, vz, p = W
    un = (vx, vy, vz)[dim]
    E = p / (T(γ) - one(T)) + T(0.5) * ρ * (vx * vx + vy * vy + vz * vz)
    m = (ρ * vx, ρ * vy, ρ * vz)
    pflux = (dim == 1 ? p : zero(T), dim == 2 ? p : zero(T), dim == 3 ? p : zero(T))
    return (ρ * un,
            m[1] * un + pflux[1],
            m[2] * un + pflux[2],
            m[3] * un + pflux[3],
            (E + p) * un)
end

@inline _add(a::NTuple{5,T}, b::NTuple{5,T}) where {T} = ntuple(i -> a[i] + b[i], 5)
@inline _sub(a::NTuple{5,T}, b::NTuple{5,T}) where {T} = ntuple(i -> a[i] - b[i], 5)
@inline _scale(s, a::NTuple{5,T}) where {T} = ntuple(i -> s * a[i], 5)

"""
    hllc_flux(WL, WR, γ, dim)

HLLC approximate Riemann flux (Toro) between left/right primitive states across a
face normal to `dim`. Resolves contact and shear; reduces to the exact result on
isolated waves.
"""
@inline function hllc_flux(WL::NTuple{5,T}, WR::NTuple{5,T}, γ, dim::Int) where {T}
    ρL, ρR = WL[1], WR[1]
    pL, pR = WL[5], WR[5]
    unL = (WL[2], WL[3], WL[4])[dim]
    unR = (WR[2], WR[3], WR[4])[dim]
    g = T(γ)                          # γ at the field precision (homogeneous-T wave speeds)
    cL = sqrt(g * pL / ρL)
    cR = sqrt(g * pR / ρR)

    # Davis wave-speed estimates.
    SL = min(unL - cL, unR - cR)
    SR = max(unL + cL, unR + cR)

    UL = prim2cons(WL, γ)
    UR = prim2cons(WR, γ)
    FL = euler_flux(WL, γ, dim)
    FR = euler_flux(WR, γ, dim)

    if SL >= 0
        return FL
    elseif SR <= 0
        return FR
    end

    # Contact-wave speed.
    Sstar = (pR - pL + ρL * unL * (SL - unL) - ρR * unR * (SR - unR)) /
            (ρL * (SL - unL) - ρR * (SR - unR))

    if Sstar >= 0
        return _hllc_star(WL, UL, FL, SL, Sstar, dim, γ)
    else
        return _hllc_star(WR, UR, FR, SR, Sstar, dim, γ)
    end
end

# Star-region flux  F*K = FK + SK (U*K - UK)   (Toro eq. 10.73).
@inline function _hllc_star(W::NTuple{5,T}, U::NTuple{5,T}, F::NTuple{5,T},
                            SK, Sstar, dim::Int, γ) where {T}
    ρ = W[1]
    un = (W[2], W[3], W[4])[dim]
    p = W[5]
    E = U[5]
    factor = ρ * (SK - un) / (SK - Sstar)

    # Velocity vector with the normal component replaced by Sstar.
    v = (W[2], W[3], W[4])
    vstar = ntuple(d -> d == dim ? Sstar : v[d], 3)
    Estar = E / ρ + (Sstar - un) * (Sstar + p / (ρ * (SK - un)))

    Ustar = (factor,
             factor * vstar[1],
             factor * vstar[2],
             factor * vstar[3],
             factor * Estar)
    return _add(F, _scale(SK, _sub(Ustar, U)))
end
