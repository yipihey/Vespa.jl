# masked_cg! — conjugate gradients on a MASKED 7-point Dirichlet system, the
# KA-kernelized irregular-domain solve (ADR-0006 Next-7).
#
# The system (RAMSES's fine-level Poisson problem on an arbitrary refined
# region, ghosts folded into the RHS by the caller):
#
#     (A·x)[c] = m[c]·( 6·x[c] − Σ_nbr m[nbr]·x[nbr] )      — SPD on the mask
#
# `m` is the covered mask as a FIELD (1 inside the region, 0 outside), so the
# kernel is branch-free and precision-generic; every CG vector is zero outside
# the mask by construction (x starts 0, every update carries the m factor), so
# the reductions are plain `dot`/`sum` — no masked reduction kernel needed.
#
# One source, two devices: CPU f64 certifies against the host CG (and the
# RAMSES oracle); Metal runs f32 with the stagnation guard catching the f32
# residual floor (the same idiom as `vcycle_solve!`).

using LinearAlgebra: dot

@kernel function _masked_apply_k!(out, @Const(x), @Const(m))
    gi, gj, gk = @index(Global, NTuple)            # ndrange covers the interior
    i = gi + 1; j = gj + 1; k = gk + 1
    T = eltype(out)
    @inbounds out[i, j, k] = m[i, j, k] * (T(6) * x[i, j, k] -
        m[i-1, j, k] * x[i-1, j, k] - m[i+1, j, k] * x[i+1, j, k] -
        m[i, j-1, k] * x[i, j-1, k] - m[i, j+1, k] * x[i, j+1, k] -
        m[i, j, k-1] * x[i, j, k-1] - m[i, j, k+1] * x[i, j, k+1])
end

"""
    masked_cg!(x, b, m; rtol=1e-12, maxiter=2000, stagnation=0.999)
        -> (x, iters, relres)

Solve the masked 7-point Dirichlet system `A·x = b` in place (see file
header).  `x`, `b`, `m` are same-backend 3-D arrays (CPU or Metal); `b` must
be zero outside the mask (the caller folds the Dirichlet ghost contributions
into it) and `x` enters as the zero initial guess.  Stops on the relative
residual `‖r‖²/‖r₀‖² ≤ rtol²` or stagnation (the f32 floor).
"""
function masked_cg!(x::AbstractArray{T,3}, b::AbstractArray{T,3},
                    m::AbstractArray{T,3}; rtol::Real = 1e-12,
                    maxiter::Integer = 2000, stagnation::Real = 0.999) where {T}
    be = KA.get_backend(x)
    nint = size(x) .- 2
    all(>(0), nint) || error("masked_cg!: arrays need a 1-cell halo (got $(size(x)))")
    r = copy(b); p = copy(b)
    Ap = similar(x); fill!(Ap, zero(T))
    rr = T(dot(r, r))
    rr0 = rr
    rr0 == zero(T) && return (x, 0, zero(T))
    iters = 0
    stag = T(stagnation)
    while rr > T(rtol)^2 * rr0 && iters < maxiter
        _masked_apply_k!(be)(Ap, p, m; ndrange = nint)
        KA.synchronize(be)
        pAp = T(dot(p, Ap))
        pAp > zero(T) || break                       # f32 breakdown guard
        alpha = rr / pAp
        @. x += alpha * p
        @. r -= alpha * Ap
        rr2 = T(dot(r, r))
        iters += 1
        rr2 >= stag * rr && (rr = rr2; break)        # stagnated (f32 floor)
        beta = rr2 / rr
        rr = rr2
        @. p = r + beta * p
    end
    KA.synchronize(be)
    return (x, iters, sqrt(rr / rr0))
end
