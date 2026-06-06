# mg_calc_defect — the (negative) residual -(Lu - f) and its L2 norm.
# Port of src/enzo/mg_calc_defect.F (the 2nd-order 7-point `#else` branch, 3-D).
#
# The defect is zeroed on the width-1 OUTER ring (the boundary, mg_calc_defect.F
# :361-374): we `fill!(defect, 0)` then write only the interior 2:dim-1 cube, so
# the untouched ring stays exactly 0 — identical to the explicit zeroing.
#
# Norm (mg_calc_defect.F:379-387): sum of defect² over ALL cells (the ring is 0,
# so it contributes nothing), then sqrt(sum)/(dim1·dim2·dim3) — the divisor is the
# FULL padded product, not the active count. Fortran accumulates in REAL*8; on the
# CPU-f64 path `sum(abs2, ·)` matches up to FP reassociation (well within RTOL_A).

@kernel function _mg_defect_kernel!(defect, @Const(sol), @Const(rhs), h3)
    gi, gj, gk = @index(Global, NTuple)        # ndrange covers the interior 2:dim-1
    i = gi + 1; j = gj + 1; k = gk + 1
    T = eltype(defect)
    # mirrors mg_calc_defect.F:355-359 (left-to-right sum, ·h3, then + rhs)
    @inbounds defect[i, j, k] = h3 * (
        sol[i-1, j, k] + sol[i+1, j, k] +
        sol[i, j-1, k] + sol[i, j+1, k] +
        sol[i, j, k-1] + sol[i, j, k+1] -
        T(6) * sol[i, j, k]) + rhs[i, j, k]
end

"""
    mg_calc_defect!(defect, sol, rhs; compute_norm=true) -> norm

Fill `defect` with the negative residual `-(L·sol - rhs)` of the 2nd-order
7-point Poisson operator (boundary ring zeroed). When `compute_norm=true` (the
default) also return its L2 norm `sqrt(Σ defect²)/(dim1·dim2·dim3)`; when
`compute_norm=false` skip the (GPU→host blocking) reduction and return `zero(T)`
— used in the V-cycle down-leg, where only the defect *array* is needed (it gets
restricted) and the norm would be discarded. All arrays are 3-D device arrays.
"""
function mg_calc_defect!(defect::AbstractArray{T,3}, sol::AbstractArray{T,3},
                         rhs::AbstractArray{T,3}; compute_norm::Bool = true) where {T}
    be = KA.get_backend(sol)
    d1, d2, d3 = size(sol)
    # h-factors in Enzo's exact order (mg_calc_defect.F:315-317):
    #   h1 = -(d1-1); h2 = h1·(d2-1); h3 = h2·(d3-1)
    h1 = -T(d1 - 1)
    h2 = h1 * T(d2 - 1)
    h3 = h2 * T(d3 - 1)
    fill!(defect, zero(T))
    nint = (d1 - 2, d2 - 2, d3 - 2)
    if nint[1] > 0 && nint[2] > 0 && nint[3] > 0
        _mg_defect_kernel!(be)(defect, sol, rhs, h3; ndrange = nint)
    end
    # compute_norm=false: leave `defect` on device for the next kernel (the
    # restrict), queue-ordered — no host stall. compute_norm=true: `sum` reads a
    # host scalar, which synchronizes implicitly.
    compute_norm || return zero(T)
    s = sum(abs2, defect)              # device reduction (host scalar) — the costly part
    return sqrt(s) / (d1 * d2 * d3)
end
