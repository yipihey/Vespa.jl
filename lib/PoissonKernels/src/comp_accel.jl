# comp_accel — difference the potential to get the acceleration g = -∇φ.
# Port of src/enzo/comp_accel.F (the 2nd-order `#else` branch, 3-D, lines 287-303).
#
# `start1/2/3` are the (ghost) offsets of the destination into the source field;
# `iflag` controls the staggering: iflag=1 is the symmetric/face-centred difference
# (fact = -1/(2·del)), iflag=0 the one-sided difference (fact = -1/del). The same
# convention the PPM/MUSCL hydro source term consumes (grx/gry/grz).

@kernel function _comp_accel_kernel!(d1f, d2f, d3f, @Const(src),
                                     f1, f2, f3, iflag::Int, s1::Int, s2::Int, s3::Int)
    i, j, k = @index(Global, NTuple)           # ndrange = full dest (dd1,dd2,dd3)
    @inbounds begin
        # mirrors comp_accel.F:291-299
        d1f[i, j, k] = f1 * (src[i+s1+iflag, j+s2, k+s3] - src[i+s1-1, j+s2, k+s3])
        d2f[i, j, k] = f2 * (src[i+s1, j+s2+iflag, k+s3] - src[i+s1, j+s2-1, k+s3])
        d3f[i, j, k] = f3 * (src[i+s1, j+s2, k+s3+iflag] - src[i+s1, j+s2, k+s3-1])
    end
end

"""
    comp_accel!(d1, d2, d3, src; iflag, start, del) -> (d1, d2, d3)

Finite-difference the potential `src` into the three acceleration components
`d1,d2,d3` (3-D device arrays, all the same destination shape). `iflag` ∈ {0,1}
selects the staggering, `start = (s1,s2,s3)` the dest→source offset, and
`del = (dx,dy,dz)` the cell sizes. `g = -∇φ`.
"""
function comp_accel!(d1::AbstractArray{T,3}, d2::AbstractArray{T,3}, d3::AbstractArray{T,3},
                     src::AbstractArray{T,3}; iflag::Integer, start, del) where {T}
    be = KA.get_backend(d1)
    dd = size(d1)
    f1 = -one(T) / (T(iflag + 1) * T(del[1]))
    f2 = -one(T) / (T(iflag + 1) * T(del[2]))
    f3 = -one(T) / (T(iflag + 1) * T(del[3]))
    _comp_accel_kernel!(be)(d1, d2, d3, src, f1, f2, f3,
                            Int(iflag), Int(start[1]), Int(start[2]), Int(start[3]); ndrange = dd)
    return d1, d2, d3                    # queue-ordered; sync only at host reads
end
