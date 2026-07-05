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
                          lev.live_d, _tabi(lev, :gi0), _tabf(lev, :gfr),
                          lev.Dsc, lev.Ssc, lev.Esc,
                          Float32(halfdt), invh, Float32(exp2(-l)),
                          n1, n2, n3, Int32(lev.B), Int32(lev.ng), Int32(lev.nd),
                          Int32(lev.stride); ndrange = n)
    return nothing
end

# ══ native level gravity: batched red-black Poisson on the block union ════════
# Topgrid φ stays an external (FFT) solve; every AMR level l ≥ 1 solves
#   ∇²φ_l = source_coef · ρ_total   (7-point, h = dx_l)
# on the UNION of its blocks: Dirichlet boundary values prolonged trilinearly
# from the parent potential ONCE per solve, red-black Gauss–Seidel sweeps fused
# over all blocks with sibling ghost exchange per sweep (true same-level
# coupling — not independent per-block solves).  Block origins are even, so the
# local (i+j+k) parity IS the global parity: RB is consistent across the union.
# Warm-started from the previous φ (or the parent prolongation on fresh blocks).

@kernel function _phi_from_global_k!(phi, @Const(φg), @Const(live_d), @Const(gi0),
                                     n1::Int32, n2::Int32, n3::Int32,
                                     ng::Int32, nd::Int32, stride::Int32)
    t = @index(Global)
    t0 = Int32(t) - Int32(1)
    n3d = nd * nd * nd
    bi = t0 ÷ n3d; c = t0 % n3d
    @inbounds begin
        slot = live_d[bi + Int32(1)]
        u = c % nd; v = (c ÷ nd) % nd; w_ = c ÷ (nd * nd)
        gi = mod(gi0[3*(slot-Int32(1))+Int32(1)] + u - ng, n1)
        gj = mod(gi0[3*(slot-Int32(1))+Int32(2)] + v - ng, n2)
        gk = mod(gi0[3*(slot-Int32(1))+Int32(3)] + w_ - ng, n3)
        phi[(slot - Int32(1)) * stride + c + Int32(1)] = Float32(φg[(gk*n2+gj)*n1+gi+1])
    end
end

"Fill a level-0 block pool's φ (incl. ghosts) from the global topgrid potential."
function phi_from_global!(hier::AMRHierarchy, φg)
    lev = hier.levels[1]
    isempty(lev.live) && return nothing
    haskey(lev.tabs, :gi0) || sync_block_geometry!(lev)
    n = length(lev.live) * lev.nd^3
    _phi_from_global_k!(lev.be)(lev.phi, φg, lev.live_d, _tabi(lev, :gi0),
                                Int32.(hier.nbase)..., Int32(lev.ng), Int32(lev.nd),
                                Int32(lev.stride); ndrange = n)
    return nothing
end

# rhs = h²·coef·ρ_phys (+ h²·coef·ρ_ext interpolated from a topgrid field — the
# DM density living on the topgrid enters the fine source this way)
@kernel function _grav_rhs_k!(rhs, @Const(D), @Const(Dsc), @Const(live_d),
                              @Const(gi0), @Const(gfr), ext, hasext::Bool,
                              dmp, hasdm::Bool,
                              h2coef::Float32, rho0::Float32, hlev::Float32,
                              n1::Int32, n2::Int32, n3::Int32,
                              B::Int32, ng::Int32, nd::Int32, stride::Int32)
    t = @index(Global)
    t0 = Int32(t) - Int32(1)
    B3 = B * B * B
    bi = t0 ÷ B3; c = t0 % B3
    @inbounds begin
        slot = live_d[bi + Int32(1)]
        base = (slot - Int32(1)) * stride
        ci = c % B; cj = (c ÷ B) % B; ck = c ÷ (B * B)
        idx = base + _lidx(ci + ng, cj + ng, ck + ng, nd)
        ρ = Float32(D[idx]) * Dsc[slot]
        hasdm && (ρ += Float32(dmp[idx]))
        if hasext
            x = Float32(gi0[3*(slot-Int32(1))+Int32(1)]) + gfr[3*(slot-Int32(1))+Int32(1)] +
                (Float32(ci) + 0.5f0) * hlev
            y = Float32(gi0[3*(slot-Int32(1))+Int32(2)]) + gfr[3*(slot-Int32(1))+Int32(2)] +
                (Float32(cj) + 0.5f0) * hlev
            z = Float32(gi0[3*(slot-Int32(1))+Int32(3)]) + gfr[3*(slot-Int32(1))+Int32(3)] +
                (Float32(ck) + 0.5f0) * hlev
            ρ += _phi_tl(ext, x, y, z, n1, n2, n3)
        end
        rhs[idx] = h2coef * (ρ - rho0)
    end
end

# one red-black half-sweep over every block's interior (global parity = local)
@kernel function _rb_relax_k!(phi, @Const(rhs), @Const(live_d), parity::Int32,
                              B::Int32, ng::Int32, nd::Int32, stride::Int32)
    t = @index(Global)
    t0 = Int32(t) - Int32(1)
    B3 = B * B * B
    bi = t0 ÷ B3; c = t0 % B3
    @inbounds begin
        slot = live_d[bi + Int32(1)]
        base = (slot - Int32(1)) * stride
        ci = c % B; cj = (c ÷ B) % B; ck = c ÷ (B * B)
        if ((ci + cj + ck) & Int32(1)) == parity
            i = ci + ng; j = cj + ng; k = ck + ng
            idx = base + _lidx(i, j, k, nd)
            phi[idx] = (phi[idx - Int32(1)] + phi[idx + Int32(1)] +
                        phi[idx - nd] + phi[idx + nd] +
                        phi[idx - nd*nd] + phi[idx + nd*nd] - rhs[idx]) * (1.0f0 / 6.0f0)
        end
    end
end

@kernel function _residual_k!(res, @Const(phi), @Const(rhs), @Const(live_d),
                              B::Int32, ng::Int32, nd::Int32, stride::Int32)
    t = @index(Global)
    t0 = Int32(t) - Int32(1)
    B3 = B * B * B
    bi = t0 ÷ B3; c = t0 % B3
    @inbounds begin
        slot = live_d[bi + Int32(1)]
        base = (slot - Int32(1)) * stride
        i = c % B + ng; j = (c ÷ B) % B + ng; k = c ÷ (B * B) + ng
        idx = base + _lidx(i, j, k, nd)
        res[Int32(t)] = abs(phi[idx - Int32(1)] + phi[idx + Int32(1)] +
                            phi[idx - nd] + phi[idx + nd] +
                            phi[idx - nd*nd] + phi[idx + nd*nd] -
                            6.0f0 * phi[idx] - rhs[idx])
    end
end

"""
    solve_gravity_level!(hier, l; source_coef, nsweep = 60, rho_ext = nothing,
                         rho_mean = 0, init = :warm) -> residual∞

Solve ∇²φ = source_coef·(ρ_gas − rho_mean [+ ρ_ext interpolated from the topgrid
field `rho_ext`]) on level `l`'s block union.  l ≥ 1: Dirichlet boundaries
prolonged trilinearly from the parent φ pool.  l = 0: fully periodic (the base
tiling covers the box, so the sibling exchange IS the periodic BC) — pass the
mean density in `rho_mean` so the source is solvable.  `init = :parent` seeds
the interior from the parent; `:warm` keeps the existing φ.  Returns the final
∞-norm residual (h²-scaled units).
"""
function solve_gravity_level!(hier::AMRHierarchy, l::Int; source_coef::Real,
                              nsweep::Int = 60, rho_ext = nothing,
                              rho_mean::Real = 0, init::Symbol = :warm,
                              use_dm::Bool = false)
    lev = hier.levels[l + 1]
    plev = l >= 1 ? hier.levels[l] : lev
    isempty(lev.live) && return 0.0f0
    haskey(lev.tabs, :gi0) || sync_block_geometry!(lev)
    n  = length(lev.live) * lev.B^3
    h  = level_dx(hier, l)
    n1, n2, n3 = Int32.(hier.nbase)
    _grav_rhs_k!(lev.be)(lev.rhs, lev.D, lev.Dsc, lev.live_d,
                         _tabi(lev, :gi0), _tabf(lev, :gfr),
                         rho_ext === nothing ? lev.rhs : rho_ext,
                         rho_ext !== nothing,
                         isempty(lev.dm) ? lev.rhs : lev.dm,     # dm lazy: dummy when absent
                         use_dm && !isempty(lev.dm),
                         Float32(h^2 * source_coef), Float32(rho_mean), Float32(exp2(-l)),
                         n1, n2, n3, Int32(lev.B), Int32(lev.ng), Int32(lev.nd),
                         Int32(lev.stride); ndrange = n)
    sib = _tabt(lev, :sib)
    pro = l >= 1 ? _tabt(lev, :pro) : RectJobTable()
    if pro.total > 0                                       # Dirichlet BCs (once)
        _rect_prolong_tl_k!(lev.be)(lev.phi, plev.phi, pro.jobs, pro.cellstart,
                                    Int32(pro.njobs), Int32(lev.nd),
                                    Int32(lev.stride); ndrange = pro.total)
        # the shell fill also touched sibling-covered ghosts — restore fine data
        # immediately so a warm restart resumes from the converged state
        _run_copy!(lev.be, sib, lev.phi, lev.phi, lev.nd, lev.stride)
    end
    if init === :parent && l >= 1                          # interior seed
        tab = to_device_table(lev.be, build_prolong_jobs(lev, plev; interior = true))
        _run_prolong!(lev.be, tab, lev.phi, plev.phi, plev.phi, 0.0f0,
                      lev.nd, lev.stride)
    end
    for _ in 1:nsweep
        _rb_relax_k!(lev.be)(lev.phi, lev.rhs, lev.live_d, Int32(0), Int32(lev.B),
                             Int32(lev.ng), Int32(lev.nd), Int32(lev.stride); ndrange = n)
        _rb_relax_k!(lev.be)(lev.phi, lev.rhs, lev.live_d, Int32(1), Int32(lev.B),
                             Int32(lev.ng), Int32(lev.nd), Int32(lev.stride); ndrange = n)
        _run_copy!(lev.be, sib, lev.phi, lev.phi, lev.nd, lev.stride)
    end
    res = device_zeros(lev.be, Float32, (n,))
    _residual_k!(lev.be)(res, lev.phi, lev.rhs, lev.live_d, Int32(lev.B),
                         Int32(lev.ng), Int32(lev.nd), Int32(lev.stride); ndrange = n)
    return maximum(res)
end

# KDK kick from the LEVEL's own φ pool (block-local central difference)
@kernel function _grav_kick_pool_k!(S1, S2, S3, Tau, @Const(D), @Const(phi),
                                    @Const(live_d), @Const(Dsc), @Const(Ssc),
                                    @Const(Esc), halfdt::Float32, inv2h::Float32,
                                    B::Int32, ng::Int32, nd::Int32, stride::Int32)
    t = @index(Global)
    t0 = Int32(t) - Int32(1)
    B3 = B * B * B
    bi = t0 ÷ B3; c = t0 % B3
    @inbounds begin
        slot = live_d[bi + Int32(1)]
        base = (slot - Int32(1)) * stride
        dsc = Dsc[slot]; ssc = Ssc[slot]; esc = Esc[slot]
        i = c % B + ng; j = (c ÷ B) % B + ng; k = c ÷ (B * B) + ng
        idx = base + _lidx(i, j, k, nd)
        ax = -(Float32(phi[idx + Int32(1)]) - Float32(phi[idx - Int32(1)])) * inv2h
        ay = -(Float32(phi[idx + nd])       - Float32(phi[idx - nd]))       * inv2h
        az = -(Float32(phi[idx + nd*nd])    - Float32(phi[idx - nd*nd]))    * inv2h
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

"Momentum half-kick of level `l` from ITS OWN φ pool (central difference)."
function grav_kick_level_pool!(hier::AMRHierarchy, l::Int, halfdt::Real)
    lev = hier.levels[l + 1]
    isempty(lev.live) && return nothing
    inv2h = Float32(1.0 / (2.0 * level_dx(hier, l)))
    n = length(lev.live) * lev.B^3
    _grav_kick_pool_k!(lev.be)(lev.S1, lev.S2, lev.S3, lev.Tau, lev.D, lev.phi,
                               lev.live_d, lev.Dsc, lev.Ssc, lev.Esc,
                               Float32(halfdt), inv2h, Int32(lev.B), Int32(lev.ng),
                               Int32(lev.nd), Int32(lev.stride); ndrange = n)
    return nothing
end

# scatter a level-0 pool field (D by default, scale-decoded) into a global
# nbase³ f32 array — the gas side of the topgrid FFT source.  Level-0 blocks
# tile disjointly: no atomics.
@kernel function _global_from_level0_k!(g, @Const(F), @Const(sc), @Const(live_d),
                                        @Const(gi0), n1::Int32, n2::Int32, n3::Int32,
                                        B::Int32, ng::Int32, nd::Int32, stride::Int32)
    t = @index(Global)
    t0 = Int32(t) - Int32(1)
    B3 = B * B * B
    bi = t0 ÷ B3; c = t0 % B3
    @inbounds begin
        slot = live_d[bi + Int32(1)]
        base = (slot - Int32(1)) * stride
        ci = c % B; cj = (c ÷ B) % B; ck = c ÷ (B * B)
        gi = gi0[3*(slot-Int32(1))+Int32(1)] + ci
        gj = gi0[3*(slot-Int32(1))+Int32(2)] + cj
        gk = gi0[3*(slot-Int32(1))+Int32(3)] + ck
        idx = base + _lidx(ci + ng, cj + ng, ck + ng, nd)
        g[(gk * n2 + gj) * n1 + gi + Int32(1)] = Float32(F[idx]) * sc[slot]
    end
end

"Gather level-0 field `f` (default :D, scale-decoded) into global array `g` (nbase³)."
function global_from_level0!(hier::AMRHierarchy, g; f::Symbol = :D)
    lev = hier.levels[1]
    isempty(lev.live) && return g
    haskey(lev.tabs, :gi0) || sync_block_geometry!(lev)
    sc = f === :D ? lev.Dsc : (f in (:S1, :S2, :S3) ? lev.Ssc : lev.Esc)
    n = length(lev.live) * lev.B^3
    _global_from_level0_k!(lev.be)(g, getfield(lev, f), sc, lev.live_d,
                                   _tabi(lev, :gi0), Int32.(hier.nbase)...,
                                   Int32(lev.B), Int32(lev.ng), Int32(lev.nd),
                                   Int32(lev.stride); ndrange = n)
    return g
end

# ── Compton momentum drag ─────────────────────────────────────────────────────
# Damp each cell's peculiar velocity toward the GLOBAL mass-weighted bulk by
# f = exp(−Γ/H·Δln a) (frame-agnostic — never damps the coherent streaming
# bulk), keeping the KE bookkeeping exact in Tau.  Mirrors the patchgrid
# compton_drag_patches! semantics on the block pools, scale-aware.
@kernel function _compton_drag_k!(S1, S2, S3, Tau, @Const(D), @Const(live_d),
                                  @Const(Dsc), @Const(Ssc), @Const(Esc),
                                  f::Float32, vbx::Float32, vby::Float32, vbz::Float32,
                                  B::Int32, ng::Int32, nd::Int32, stride::Int32)
    t = @index(Global)
    t0 = Int32(t) - Int32(1)
    B3 = B * B * B
    bi = t0 ÷ B3; c = t0 % B3
    @inbounds begin
        slot = live_d[bi + Int32(1)]
        base = (slot - Int32(1)) * stride
        dsc = Dsc[slot]; ssc = Ssc[slot]; esc = Esc[slot]
        i = c % B + ng; j = (c ÷ B) % B + ng; k = c ÷ (B * B) + ng
        idx = base + _lidx(i, j, k, nd)
        ρ  = max(Float32(D[idx]) * dsc, 1.0f-30)
        s1 = Float32(S1[idx]) * ssc; s2 = Float32(S2[idx]) * ssc; s3 = Float32(S3[idx]) * ssc
        ke0 = (s1*s1 + s2*s2 + s3*s3) / (2.0f0 * ρ)
        n1 = ρ*vbx + (s1 - ρ*vbx) * f
        n2 = ρ*vby + (s2 - ρ*vby) * f
        n3 = ρ*vbz + (s3 - ρ*vbz) * f
        ke1 = (n1*n1 + n2*n2 + n3*n3) / (2.0f0 * ρ)
        S1[idx]  = _narrow(eltype(S1), n1 / ssc)
        S2[idx]  = _narrow(eltype(S2), n2 / ssc)
        S3[idx]  = _narrow(eltype(S3), n3 / ssc)
        Tau[idx] = _narrow(eltype(Tau), (Float32(Tau[idx]) * esc + (ke1 - ke0)) / esc)
    end
end

"""
    compton_drag!(hier, f; scratch) -> nothing

Apply Compton drag with factor `f = exp(−Γ/H·Δln a)` to every level: the global
mass-weighted bulk velocity is computed from the level-0 composite (device
gathers + reductions via `scratch`, a global nbase³ f32 array), then each
level's cells damp toward it (KE-exact in Tau).
"""
function compton_drag!(hier::AMRHierarchy, f::Real; scratch)
    lev0 = hier.levels[1]
    isempty(lev0.live) && return nothing
    global_from_level0!(hier, scratch; f = :D);  M  = Float64(sum(scratch))
    global_from_level0!(hier, scratch; f = :S1); p1 = Float64(sum(scratch))
    global_from_level0!(hier, scratch; f = :S2); p2 = Float64(sum(scratch))
    global_from_level0!(hier, scratch; f = :S3); p3 = Float64(sum(scratch))
    vb = (Float32(p1 / M), Float32(p2 / M), Float32(p3 / M))
    for l in 0:length(hier.levels)-1
        lev = hier.levels[l + 1]
        isempty(lev.live) && continue
        n = length(lev.live) * lev.B^3
        _compton_drag_k!(lev.be)(lev.S1, lev.S2, lev.S3, lev.Tau, lev.D, lev.live_d,
                                 lev.Dsc, lev.Ssc, lev.Esc, Float32(f), vb...,
                                 Int32(lev.B), Int32(lev.ng), Int32(lev.nd),
                                 Int32(lev.stride); ndrange = n)
    end
    return nothing
end
