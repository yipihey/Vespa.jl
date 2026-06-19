# Sedov–Taylor point-blast wave as source code (ADR-0001, P9).
#
# A large thermal energy E0 is deposited in a small central region of an
# otherwise cold, uniform medium; the result is a self-similar, circular
# (in 2D) blast wave whose shock radius grows as
#
#     R(t) = ξ₀ (E0 t² / ρ₀)^{1/(2+ν)}        (ν = #dimensions)
#
# This is the canonical AMR stress test: a sharp shock expands across the
# domain, so the refined region must track it dynamically. The spec runs on any
# backend; on HGBackend with a RefinementPolicy the mesh follows the shock.

using Vespa

"""
    sedov_problem(; n=64, E0=1.0, ρ0=1.0, p0=1e-5, γ=1.4,
                    r0=nothing, tfinal=0.05) -> Problem

2D Sedov blast on the unit square `[0,1]²` centered at `(0.5, 0.5)`, `n` base
cells per axis, reflecting boundaries. The blast energy `E0` is spread over a hot
spot of radius `r0` (default ≈ 2 base cells) as a raised pressure
`p_hot = (γ-1) E0 / area`, on a cold background `(ρ0, p0)` at rest.
"""
function sedov_problem(; n::Integer = 64, E0::Real = 1.0, ρ0::Real = 1.0,
                       p0::Real = 1e-5, γ::Real = 1.4,
                       r0 = nothing, tfinal::Real = 0.05)
    dx = 1.0 / n
    rhot = r0 === nothing ? 2.0 * dx : Float64(r0)
    area = π * rhot^2
    p_hot = (γ - 1) * E0 / area          # uniform internal energy density in the spot
    xc, yc = 0.5, 0.5
    function init(x, y, z)
        r = hypot(x - xc, y - yc)
        p = r <= rhot ? p_hot : Float64(p0)
        return (Float64(ρ0), 0.0, 0.0, 0.0, p)
    end
    return Problem(; name = "Sedov2D",
                   dims = (Int(n), Int(n)),
                   domain = ((0.0, 1.0), (0.0, 1.0)),
                   γ = Float64(γ),
                   bcs = Reflecting(),
                   init = init,
                   tfinal = Float64(tfinal),
                   cfl = 0.3)
end

"""
    sedov_shock_radius(ρ0, E0, γ, t; ν=2) -> Float64

Self-similar Sedov shock radius `R = ξ₀ (E0 t² / ρ0)^{1/(2+ν)}`. The constant ξ₀
is O(1) and γ-dependent; for validation we use the **growth exponent** (R ∝
t^{2/(2+ν)}) rather than the absolute constant, so this returns the shape factor
with ξ₀ = 1.
"""
sedov_shock_radius(ρ0, E0, γ, t; ν = 2) = (E0 * t^2 / ρ0)^(1 / (2 + ν))
