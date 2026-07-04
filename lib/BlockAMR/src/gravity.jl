# gravity.jl — KDK half-kicks from the TOPGRID potential (Phase-5 gravity model:
# particles and the Poisson solve stay on the topgrid; blocks only ever READ φ).
#
# Per-block positions reach the device as the exact split representation
#   x_topgrid = i0 (Int32 topgrid cell) + fr (f32 dyadic fraction) + (c+½)·2^−l
# computed on the host from the UInt128 origin (i0 = origin >> l exactly; fr is
# f32-exact for l ≤ 24 and < 2^−24 topgrid cells off beyond — far below the
# interpolation truncation error of a topgrid-resolution potential).  No f64 on
# the device, no absolute float coordinates anywhere.
#
# Acceleration: central difference of trilinearly-interpolated φ (h = 1 topgrid
# cell, each φ(x±h e_d) itself trilinear, periodic) — continuous across topgrid
# cell faces and consistent with the house CIC/Enzo convention.  Kick is
# KE-consistent: Tau += (S·ΔS + ½|ΔS|²)/ρ (matches patchgrid _grav_kick!).

"Attach per-block device geometry (topgrid cell + fraction) to `lev.tabs`."
function sync_block_geometry!(lev::Level)
    hi0 = zeros(Int32, 3 * lev.cap)
    hfr = zeros(Float32, 3 * lev.cap)
    l = lev.l
    for s in lev.live
        m = lev.meta[s]
        for d in 1:3
            hi0[3(Int(s)-1)+d] = Int32(m.origin[d] >> l)
            hfr[3(Int(s)-1)+d] = Float32((m.origin[d] & ((UInt128(1) << l) - 1)) /
                                         exp2(l))
        end
    end
    lev.tabs[:gi0] = to_device(lev.be, hi0, Int32)
    lev.tabs[:gfr] = to_device(lev.be, hfr, Float32)
    return nothing
end

# trilinear φ at topgrid position (x,y,z) in cell units (cell centers at q+½), periodic
@inline function _phi_tl(φ, x::Float32, y::Float32, z::Float32,
                         n1::Int32, n2::Int32, n3::Int32)
    xc = x - 0.5f0; yc = y - 0.5f0; zc = z - 0.5f0
    i0 = unsafe_trunc(Int32, floor(xc)); fx = xc - Float32(i0)
    j0 = unsafe_trunc(Int32, floor(yc)); fy = yc - Float32(j0)
    k0 = unsafe_trunc(Int32, floor(zc)); fz = zc - Float32(k0)
    ia = mod(i0, n1); ib = mod(i0 + Int32(1), n1)
    ja = mod(j0, n2); jb = mod(j0 + Int32(1), n2)
    ka = mod(k0, n3); kb = mod(k0 + Int32(1), n3)
    @inbounds begin
        c000 = Float32(φ[(ka*n2+ja)*n1+ia+1]); c100 = Float32(φ[(ka*n2+ja)*n1+ib+1])
        c010 = Float32(φ[(ka*n2+jb)*n1+ia+1]); c110 = Float32(φ[(ka*n2+jb)*n1+ib+1])
        c001 = Float32(φ[(kb*n2+ja)*n1+ia+1]); c101 = Float32(φ[(kb*n2+ja)*n1+ib+1])
        c011 = Float32(φ[(kb*n2+jb)*n1+ia+1]); c111 = Float32(φ[(kb*n2+jb)*n1+ib+1])
    end
    return (1-fx)*((1-fy)*((1-fz)*c000 + fz*c001) + fy*((1-fz)*c010 + fz*c011)) +
           fx   *((1-fy)*((1-fz)*c100 + fz*c101) + fy*((1-fz)*c110 + fz*c111))
end

@kernel function _grav_kick_k!(S1, S2, S3, Tau, @Const(D), @Const(φ),
                               @Const(live_d), @Const(gi0), @Const(gfr),
                               @Const(Dsc), @Const(Ssc), @Const(Esc),
                               halfdt::Float32, invh::Float32, hlev::Float32,
                               n1::Int32, n2::Int32, n3::Int32,
                               B::Int32, ng::Int32, nd::Int32, stride::Int32)
    t = @index(Global)
    t0 = Int32(t) - Int32(1)
    B3 = B * B * B
    bi = t0 ÷ B3; c = t0 % B3
    @inbounds begin
        slot = live_d[bi + Int32(1)]
        base = (slot - Int32(1)) * stride
        dsc = Dsc[slot]; ssc = Ssc[slot]; esc = Esc[slot]
        ci = c % B; cj = (c ÷ B) % B; ck = c ÷ (B * B)
        # topgrid-cell-unit position of the cell CENTER (relative arithmetic, f32)
        x = Float32(gi0[3*(slot-Int32(1))+Int32(1)]) + gfr[3*(slot-Int32(1))+Int32(1)] +
            (Float32(ci) + 0.5f0) * hlev
        y = Float32(gi0[3*(slot-Int32(1))+Int32(2)]) + gfr[3*(slot-Int32(1))+Int32(2)] +
            (Float32(cj) + 0.5f0) * hlev
        z = Float32(gi0[3*(slot-Int32(1))+Int32(3)]) + gfr[3*(slot-Int32(1))+Int32(3)] +
            (Float32(ck) + 0.5f0) * hlev
        # a = −∇φ, central difference over ±1 topgrid cell (invh = 1/(2·cellsize_code))
        ax = -( _phi_tl(φ, x+1f0, y, z, n1,n2,n3) - _phi_tl(φ, x-1f0, y, z, n1,n2,n3) ) * invh
        ay = -( _phi_tl(φ, x, y+1f0, z, n1,n2,n3) - _phi_tl(φ, x, y-1f0, z, n1,n2,n3) ) * invh
        az = -( _phi_tl(φ, x, y, z+1f0, n1,n2,n3) - _phi_tl(φ, x, y, z-1f0, n1,n2,n3) ) * invh
        idx = base + _lidx(ci + ng, cj + ng, ck + ng, nd)
        ρ  = Float32(D[idx]) * dsc
        s1 = Float32(S1[idx]) * ssc; s2 = Float32(S2[idx]) * ssc; s3 = Float32(S3[idx]) * ssc
        d1 = ρ * ax * halfdt; d2 = ρ * ay * halfdt; d3 = ρ * az * halfdt
        τ  = Float32(Tau[idx]) * esc +
             (s1 * d1 + s2 * d2 + s3 * d3 + 0.5f0 * (d1*d1 + d2*d2 + d3*d3)) / max(ρ, 1f-30)
        S1[idx]  = _narrow(eltype(S1), (s1 + d1) / ssc)
        S2[idx]  = _narrow(eltype(S2), (s2 + d2) / ssc)
        S3[idx]  = _narrow(eltype(S3), (s3 + d3) / ssc)
        Tau[idx] = _narrow(eltype(Tau), τ / esc)
    end
end

"""
    grav_kick_level!(hier, l, φ, halfdt)

KE-consistent momentum half-kick of every level-`l` active cell from the topgrid
potential `φ` (flat device array, nbase³, code units).  `halfdt` in code time.
Call `sync_block_geometry!` after any topology change (build_level_tables! does).
"""
function grav_kick_level!(hier::AMRHierarchy, l::Int, φ, halfdt::Real)
    lev = hier.levels[l + 1]
    isempty(lev.live) && return nothing
    haskey(lev.tabs, :gi0) || sync_block_geometry!(lev)
    n1, n2, n3 = Int32.(hier.nbase)
    hcell = hier.box / hier.nbase[1]              # topgrid cell size, code length
    invh = Float32(1.0 / (2.0 * hcell))
    n = length(lev.live) * lev.B^3
    _grav_kick_k!(lev.be)(lev.S1, lev.S2, lev.S3, lev.Tau, lev.D, φ,
                          lev.live_d, lev.tabs[:gi0], lev.tabs[:gfr],
                          lev.Dsc, lev.Ssc, lev.Esc,
                          Float32(halfdt), invh, Float32(exp2(-l)),
                          n1, n2, n3, Int32(lev.B), Int32(lev.ng), Int32(lev.nd),
                          Int32(lev.stride); ndrange = n)
    return nothing
end
