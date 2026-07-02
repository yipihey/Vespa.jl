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

# Gather a patch's ghosted nd³ block out of a global ncell³ field `G` by PERIODIC index
# wrap, into the FLAT output vector `out` (length nd³, linear l = i + (j-1)·nd1 +
# (k-1)·nd1·nd2), converting to `eltype(out)`.  The device counterpart of MultiCode's
# host segment-copy scatter — keeps the per-patch accel gather on the GPU (no host
# round-trip / per-patch upload), which scales better when there are many patches.
@kernel function _gather_periodic_k!(out, @Const(G), o1::Int, o2::Int, o3::Int, ng::Int,
                                     nd1::Int, nd2::Int, nc1::Int, nc2::Int, nc3::Int)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        gx = mod(o1 + (i - ng - 1), nc1) + 1
        gy = mod(o2 + (j - ng - 1), nc2) + 1
        gz = mod(o3 + (k - ng - 1), nc3) + 1
        out[i + (j - 1) * nd1 + (k - 1) * nd1 * nd2] = eltype(out)(G[gx, gy, gz])
    end
end

# KDK gas gravity kick that DIFFERENCES the potential inline (no stored accel field):
# for each interior cell g = −∇φ (2-point central difference, iflag=1 / fact = −1/(2dx)),
# then the KE-consistent momentum kick dS = ρ·g·c, ΔTau = (S·dS + ½dS²)/ρ.  `φ,D,S1..,Tau`
# are the patch's FLAT length-`n1·n2·n3` fields; the central difference reads the ng-ghost
# halo so the interior is exact.  Identical result to comp_accel!→stored-accel→kick, but
# stores only φ.
@kernel function _grav_kick_phi_kernel!(@Const(φ), D, S1, S2, S3, Tau,
                                        n1::Int, n2::Int, ng::Int, hc, c)
    li, lj, lk = @index(Global, NTuple)            # over the pdim³ interior
    @inbounds begin
        T = eltype(D)
        idx = (li+ng) + n1*((lj+ng)-1) + n1*n2*((lk+ng)-1)
        sx = 1; sy = n1; sz = n1*n2
        gx = -hc*(φ[idx+sx]-φ[idx-sx]); gy = -hc*(φ[idx+sy]-φ[idx-sy]); gz = -hc*(φ[idx+sz]-φ[idx-sz])
        d = D[idx]
        dS1 = d*gx*c; dS2 = d*gy*c; dS3 = d*gz*c
        Tau[idx] += ((S1[idx]*dS1 + S2[idx]*dS2 + S3[idx]*dS3) + T(0.5)*(dS1*dS1+dS2*dS2+dS3*dS3)) / d
        S1[idx] += dS1; S2[idx] += dS2; S3[idx] += dS3
    end
end

"""
    grav_kick_from_potential!(φ, D,S1,S2,S3,Tau; dims, ng, dx, halfdt) -> nothing

Apply a half KDK gravity kick to the gas, computing `g = −∇φ` on the fly (central
difference of the patch potential `φ`) instead of reading a stored acceleration field.
`φ,D,S1,S2,S3,Tau` are flat `prod(dims)` arrays (the patch with its `ng`-ghost halo);
`dx` is the cell width and `halfdt = ½dt`.  Updates only the interior cells.
"""
function grav_kick_from_potential!(φ, D, S1, S2, S3, Tau; dims, ng::Integer, dx::Real, halfdt::Real)
    be = KA.get_backend(D); T = eltype(D)
    pdim = ntuple(d -> dims[d] - 2*Int(ng), 3)
    _grav_kick_phi_kernel!(be)(φ, D, S1, S2, S3, Tau, Int(dims[1]), Int(dims[2]), Int(ng),
                               T(0.5)/T(dx), T(halfdt); ndrange = pdim)
    return nothing
end

# Ghost-FREE gravity kick: reads the GLOBAL potential `φg` (nc³, no halo) with periodic wrap, and
# writes the gas arrays which are ALSO ghost-free nc³ (the FVGK-canonical g.R blocks).  This removes
# the per-patch ghosted φ-block copy entirely.  `Tau` may be GE_SCALE-lifted (energy slot stored
# ×gesc so cold eint survives f16): the KE increment is scaled by `gesc` before adding.  All inner
# math is done in Float32 so f16 D/S/Tau lose no precision in the (E−KE)-free momentum kick.
@kernel function _grav_kick_global_phi_kernel!(@Const(φg), D, S1, S2, S3, Tau,
                                               n1::Int, n2::Int, n3::Int, hc::Float32, c::Float32, gesc::Float32, msc::Float32)
    i, j, k = @index(Global, NTuple)               # over the full nc³ (1-based)
    @inbounds begin
        idx = i + n1*(j-1) + n1*n2*(k-1)
        ip = i == n1 ? 1 : i+1; im = i == 1 ? n1 : i-1
        jp = j == n2 ? 1 : j+1; jm = j == 1 ? n2 : j-1
        kp = k == n3 ? 1 : k+1; km = k == 1 ? n3 : k-1
        gx = -hc*(Float32(φg[ip + n1*(j-1) + n1*n2*(k-1)]) - Float32(φg[im + n1*(j-1) + n1*n2*(k-1)]))
        gy = -hc*(Float32(φg[i + n1*(jp-1) + n1*n2*(k-1)]) - Float32(φg[i + n1*(jm-1) + n1*n2*(k-1)]))
        gz = -hc*(Float32(φg[i + n1*(j-1) + n1*n2*(kp-1)]) - Float32(φg[i + n1*(j-1) + n1*n2*(km-1)]))
        d = Float32(D[idx])
        # MOM_SCALE-aware: S is stored LIFTED (×msc, FVGK dedup).  Un-lift to true momentum, add the physical
        # increment dS=ρ·g·dt in true units (KE increment ∝ true S·dS, no msc²), then re-lift on store.  msc=1 ⇒ no-op.
        imsc = 1f0/msc
        dS1 = d*gx*c; dS2 = d*gy*c; dS3 = d*gz*c
        s1 = Float32(S1[idx])*imsc; s2 = Float32(S2[idx])*imsc; s3 = Float32(S3[idx])*imsc
        dKE = ((s1*dS1 + s2*dS2 + s3*dS3) + 0.5f0*(dS1*dS1+dS2*dS2+dS3*dS3)) / d
        Tau[idx] += eltype(Tau)(dKE * gesc)
        S1[idx] = eltype(S1)((s1 + dS1)*msc); S2[idx] = eltype(S2)((s2 + dS2)*msc); S3[idx] = eltype(S3)((s3 + dS3)*msc)
    end
end

"""
    grav_kick_from_global_potential!(φg, D,S1,S2,S3,Tau; nc, dx, halfdt, gesc=1) -> nothing

Half KDK gravity kick reading the GLOBAL `nc³` potential `φg` (no halo, periodic wrap) and writing
the ghost-free gas arrays in place — the FVGK-canonical layout, with no per-patch φ-block copy.
`gesc` (default 1) scales the energy increment when `Tau` is GE_SCALE-lifted f16 storage.
"""
function grav_kick_from_global_potential!(φg, D, S1, S2, S3, Tau; nc::NTuple{3,Int},
                                          dx::Real, halfdt::Real, gesc::Real=1, msc::Real=1)
    be = KA.get_backend(D)
    _grav_kick_global_phi_kernel!(be)(φg, D, S1, S2, S3, Tau, nc[1], nc[2], nc[3],
                                      Float32(0.5)/Float32(dx), Float32(halfdt), Float32(gesc), Float32(msc); ndrange = nc)
    return nothing
end

"""
    gather_periodic_block!(out, G, o, ng, nd, nc) -> out

Fill the flat device vector `out` (length `prod(nd)`) with the patch block of the global
field `G` (3-D device, `nc` cells/axis) whose interior origin is `o` (0-based per axis),
including the `ng` ghost layer, by periodic wrap.  `out`'s element type may differ from
`G`'s (converted in the kernel).  GPU device gather; queue-ordered.
"""
function gather_periodic_block!(out::AbstractVector, G::AbstractArray{<:Any,3},
                                o::NTuple{3,Int}, ng::Int, nd::NTuple{3,Int}, nc::NTuple{3,Int})
    be = KA.get_backend(out)
    _gather_periodic_k!(be)(out, G, o[1], o[2], o[3], ng, nd[1], nd[2], nc[1], nc[2], nc[3];
                            ndrange = nd)
    return out
end
