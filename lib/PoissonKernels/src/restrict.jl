# mg_restrict — restrict (project) a fine field onto a coarse field.
# Port of src/enzo/mg_restrict.F (3-D body, lines 117-187).
#
# The interior is a 27-point quadratic restriction (SQUARED weights) scaled by
# coef3 = 0.52. The Fortran writes the boundary faces with nearest-neighbour
# copies in three passes (i-faces, j-faces, k-faces); those index regions are
# DISJOINT (i-faces only at interior j,k; j-faces at j∈{1,ddim2}; k-faces at
# k∈{1,ddim3}), so a single full-grid kernel with the priority k-face → j-face →
# i-face → interior reproduces the exact result — no overlap, no order ambiguity.
#
# Index helpers match Fortran exactly: x = (i-1)·fact + 0.5; i1 = int(x)+1 (int
# truncates toward zero, x>0 here); the clamped variant min(max(int(..)+1,1),sdim)
# is used on the j/k faces precisely where Enzo uses it.

@kernel function _mg_restrict_kernel!(dest, @Const(src), fact1, fact2, fact3, coef3,
                                      sd1::Int, sd2::Int, sd3::Int,
                                      dd1::Int, dd2::Int, dd3::Int)
    i, j, k = @index(Global, NTuple)           # ndrange = full dest (dd1,dd2,dd3)
    T = eltype(dest)
    @inbounds begin
        if k == 1 || k == dd3
            # k-face (mg_restrict.F:177-185): clamped i1,j1; literal k-source.
            ksrc = (k == 1) ? 1 : sd3
            i1 = min(max(unsafe_trunc(Int, T(i-1) * fact1 + T(0.5)) + 1, 1), sd1)
            j1 = min(max(unsafe_trunc(Int, T(j-1) * fact2 + T(0.5)) + 1, 1), sd2)
            dest[i, j, k] = src[i1, j1, ksrc]
        elseif j == 1 || j == dd2
            # j-face (mg_restrict.F:170-174): clamped i1, unclamped k1, literal j.
            jsrc = (j == 1) ? 1 : sd2
            i1 = min(max(unsafe_trunc(Int, T(i-1) * fact1 + T(0.5)) + 1, 1), sd1)
            k1 = unsafe_trunc(Int, T(k-1) * fact3 + T(0.5)) + 1
            dest[i, j, k] = src[i1, jsrc, k1]
        elseif i == 1 || i == dd1
            # i-face (mg_restrict.F:167-168): literal i, unclamped j1,k1.
            isrc = (i == 1) ? 1 : sd1
            j1 = unsafe_trunc(Int, T(j-1) * fact2 + T(0.5)) + 1
            k1 = unsafe_trunc(Int, T(k-1) * fact3 + T(0.5)) + 1
            dest[i, j, k] = src[isrc, j1, k1]
        else
            # interior 27-point quadratic (mg_restrict.F:131-165)
            x = T(i-1) * fact1 + T(0.5); i1 = unsafe_trunc(Int, x) + 1
            y = T(j-1) * fact2 + T(0.5); j1 = unsafe_trunc(Int, y) + 1
            z = T(k-1) * fact3 + T(0.5); k1 = unsafe_trunc(Int, z) + 1
            dxm = T(0.5) * (T(i1) - x)^2; dxp = T(0.5) * (one(T) + x - T(i1))^2; dx0 = one(T) - dxp - dxm
            dym = T(0.5) * (T(j1) - y)^2; dyp = T(0.5) * (one(T) + y - T(j1))^2; dy0 = one(T) - dyp - dym
            dzm = T(0.5) * (T(k1) - z)^2; dzp = T(0.5) * (one(T) + z - T(k1))^2; dz0 = one(T) - dzp - dzm
            v = src[i1-1, j1-1, k1-1] * dxm * dym * dzm +
                src[i1,   j1-1, k1-1] * dx0 * dym * dzm +
                src[i1+1, j1-1, k1-1] * dxp * dym * dzm +
                src[i1-1, j1,   k1-1] * dxm * dy0 * dzm +
                src[i1,   j1,   k1-1] * dx0 * dy0 * dzm +
                src[i1+1, j1,   k1-1] * dxp * dy0 * dzm +
                src[i1-1, j1+1, k1-1] * dxm * dyp * dzm +
                src[i1,   j1+1, k1-1] * dx0 * dyp * dzm +
                src[i1+1, j1+1, k1-1] * dxp * dyp * dzm +
                src[i1-1, j1-1, k1]   * dxm * dym * dz0 +
                src[i1,   j1-1, k1]   * dx0 * dym * dz0 +
                src[i1+1, j1-1, k1]   * dxp * dym * dz0 +
                src[i1-1, j1,   k1]   * dxm * dy0 * dz0 +
                src[i1,   j1,   k1]   * dx0 * dy0 * dz0 +
                src[i1+1, j1,   k1]   * dxp * dy0 * dz0 +
                src[i1-1, j1+1, k1]   * dxm * dyp * dz0 +
                src[i1,   j1+1, k1]   * dx0 * dyp * dz0 +
                src[i1+1, j1+1, k1]   * dxp * dyp * dz0 +
                src[i1-1, j1-1, k1+1] * dxm * dym * dzp +
                src[i1,   j1-1, k1+1] * dx0 * dym * dzp +
                src[i1+1, j1-1, k1+1] * dxp * dym * dzp +
                src[i1-1, j1,   k1+1] * dxm * dy0 * dzp +
                src[i1,   j1,   k1+1] * dx0 * dy0 * dzp +
                src[i1+1, j1,   k1+1] * dxp * dy0 * dzp +
                src[i1-1, j1+1, k1+1] * dxm * dyp * dzp +
                src[i1,   j1+1, k1+1] * dx0 * dyp * dzp +
                src[i1+1, j1+1, k1+1] * dxp * dyp * dzp
            dest[i, j, k] = coef3 * v
        end
    end
end

"""
    mg_restrict!(dest, src) -> dest

Restrict the fine field `src` onto the coarse field `dest` (3-D device arrays) via
Enzo's 27-point quadratic restriction (interior) plus nearest-neighbour boundary
faces. `size(src)` is the fine grid, `size(dest)` the coarse grid.
"""
function mg_restrict!(dest::AbstractArray{T,3}, src::AbstractArray{T,3}) where {T}
    be = KA.get_backend(dest)
    sd = size(src); dd = size(dest)
    fact1 = T(sd[1] - 1) / T(dd[1] - 1)
    fact2 = T(sd[2] - 1) / T(dd[2] - 1)
    fact3 = T(sd[3] - 1) / T(dd[3] - 1)
    coef3 = T(0.52)
    _mg_restrict_kernel!(be)(dest, src, fact1, fact2, fact3, coef3,
                             sd[1], sd[2], sd[3], dd[1], dd[2], dd[3]; ndrange = dd)
    return dest                          # queue-ordered; sync only at host reads
end
