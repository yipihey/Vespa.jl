# mg_prolong — prolong (interpolate) a coarse field onto a fine field.
# Port of src/enzo/mg_prolong.F (3-D body, lines 93-128): trilinear interpolation
# over the FULL destination grid.
#
# This writes `dest = interpolated value` (the pure prolong, matching the Fortran
# oracle). The V-cycle's "prolong-and-add" is done by the driver: prolong into a
# scratch buffer, then `Sol .+= scratch` (exactly MultigridSolver.C:182-189).
#
# Constants are Enzo's verbatim: half = 0.5001 (note: mg_prolong2 uses 0.50001 —
# DIFFERENT), edge_d = REAL(sdim_d) - half (the UNSHIFTED sdim, not sdim-1). The
# clamp keeps i1 and i1+1 in [1, sdim].

@kernel function _mg_prolong_kernel!(dest, @Const(src), fact1, fact2, fact3,
                                     half, edge1, edge2, edge3,
                                     sd1::Int, sd2::Int, sd3::Int)
    i, j, k = @index(Global, NTuple)           # ndrange = full dest (dd1,dd2,dd3)
    T = eltype(dest)
    @inbounds begin
        x = min(max(T(i-1) * fact1 + T(0.5), half), edge1)
        i1 = unsafe_trunc(Int, x + T(0.5)); dx = T(i1) + T(0.5) - x
        y = min(max(T(j-1) * fact2 + T(0.5), half), edge2)
        j1 = unsafe_trunc(Int, y + T(0.5)); dy = T(j1) + T(0.5) - y
        z = min(max(T(k-1) * fact3 + T(0.5), half), edge3)
        k1 = unsafe_trunc(Int, z + T(0.5)); dz = T(k1) + T(0.5) - z
        # mirrors mg_prolong.F:108-124 (8-term trilinear, left-to-right)
        dest[i, j, k] =
            src[i1,   j1,   k1]   * dx * dy * dz +
            src[i1+1, j1,   k1]   * (one(T) - dx) * dy * dz +
            src[i1,   j1+1, k1]   * dx * (one(T) - dy) * dz +
            src[i1+1, j1+1, k1]   * (one(T) - dx) * (one(T) - dy) * dz +
            src[i1,   j1,   k1+1] * dx * dy * (one(T) - dz) +
            src[i1+1, j1,   k1+1] * (one(T) - dx) * dy * (one(T) - dz) +
            src[i1,   j1+1, k1+1] * dx * (one(T) - dy) * (one(T) - dz) +
            src[i1+1, j1+1, k1+1] * (one(T) - dx) * (one(T) - dy) * (one(T) - dz)
    end
end

"""
    mg_prolong!(dest, src) -> dest

Prolong (trilinearly interpolate) the coarse field `src` onto the fine field
`dest` (3-D device arrays); `dest` is overwritten. `size(src)` is the coarse grid,
`size(dest)` the fine grid.
"""
function mg_prolong!(dest::AbstractArray{T,3}, src::AbstractArray{T,3}) where {T}
    be = KA.get_backend(dest)
    sd = size(src); dd = size(dest)
    fact1 = T(sd[1] - 1) / T(dd[1] - 1)
    fact2 = T(sd[2] - 1) / T(dd[2] - 1)
    fact3 = T(sd[3] - 1) / T(dd[3] - 1)
    half = T(0.5001)
    edge1 = T(sd[1]) - half; edge2 = T(sd[2]) - half; edge3 = T(sd[3]) - half
    _mg_prolong_kernel!(be)(dest, src, fact1, fact2, fact3, half, edge1, edge2, edge3,
                            sd[1], sd[2], sd[3]; ndrange = dd)
    return dest                          # queue-ordered; sync only at host reads
end
