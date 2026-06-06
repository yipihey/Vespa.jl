# mg_relax — multigrid relaxation with the differenced Poisson operator.
# Port of src/enzo/mg_relax.F (the 2nd-order 7-point `#else` branch, 3-D).
#
# Gauss-Seidel is order-dependent, so it is done as a RED then BLACK sweep: the
# 7-point stencil only couples a cell to its opposite-colour neighbours, so within
# one colour every update is independent (race-free), and "all red, then all
# black" is exactly the serial sweep. Enzo's 3-D path uses a CACHE_AWARE fusion
# (mg_relax.F:289-341) whose NET result is plain red-then-black — red on slab k is
# computed before any black neighbour reads it — so two kernel launches with a
# sync between reproduce it bit-for-bit (certified by test_relax.jl).
#
# Colour: pass-1 (red, updated first) is the set with (i+j+k) EVEN — the first
# cell Enzo touches is (2,2,2). We launch red (parity 0) then black (parity 1).

@kernel function _mg_relax_kernel!(sol, @Const(rhs), redblack::Int, h3, coef3)
    gi, gj, gk = @index(Global, NTuple)        # ndrange covers the interior 2:dim-1
    i = gi + 1; j = gj + 1; k = gk + 1
    @inbounds if (i + j + k) % 2 == redblack
        # mirrors mg_relax.F:313-317 / :350-354 (left-to-right sum, then ·coef3)
        sol[i, j, k] = coef3 * (
            sol[i-1, j, k] + sol[i+1, j, k] +
            sol[i, j-1, k] + sol[i, j+1, k] +
            sol[i, j, k-1] + sol[i, j, k+1] -
            h3 * rhs[i, j, k])
    end
end

"""
    mg_relax!(sol, rhs) -> sol

One Gauss-Seidel relaxation of the 2nd-order 7-point Poisson operator on the
interior of `sol` (in place), using right-hand side `rhs`. `sol`/`rhs` are 3-D
device arrays `(dim1,dim2,dim3)`. Red sweep, sync, black sweep, sync — the
certified equivalent of Enzo's serial `mg_relax`.
"""
function mg_relax!(sol::AbstractArray{T,3}, rhs::AbstractArray{T,3}) where {T}
    be = KA.get_backend(sol)
    d1, d2, d3 = size(sol)
    # Cumulative h-factors, computed in Enzo's exact order (mg_relax.F:246-248):
    #   h1 = 1/(d1-1); h2 = h1/(d2-1); h3 = h2/(d3-1)   (NOT 1/((d1-1)(d2-1)(d3-1)))
    h1 = one(T) / T(d1 - 1)
    h2 = h1 / T(d2 - 1)
    h3 = h2 / T(d3 - 1)
    coef3 = one(T) / T(6)
    nint = (d1 - 2, d2 - 2, d3 - 2)
    (nint[1] < 1 || nint[2] < 1 || nint[3] < 1) && return sol  # nothing interior
    # No per-kernel synchronize: KA orders kernels on the backend queue, so the
    # black sweep runs after the red sweep (which it reads) without a host stall.
    # The only sync boundary is a host read (the residual norm / to_host) — the
    # same single-source pattern PPMKernels uses. This avoids a ~190 µs Metal
    # command-buffer drain after every launch (the dominant V-cycle overhead).
    _mg_relax_kernel!(be)(sol, rhs, 0, h3, coef3; ndrange = nint)  # red
    _mg_relax_kernel!(be)(sol, rhs, 1, h3, coef3; ndrange = nint)  # black (queue-ordered after red)
    return sol
end
