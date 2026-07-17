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
                               scap::Float32,
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
        # absolute post-kick velocity cap (the BAM_SMAXCAP twin for kicks):
        # kicks bypass the hydro epilogue guards — at a refinement-saturated
        # cell ∇φ is unbounded and ρ·g·dt overflows the f16 momentum window
        # (stored > 65504 → Inf → smax = Inf).  Clamp the POST-kick velocity
        # per component; the τ update uses the ACTUAL applied deltas, so the
        # energy stays consistent with the momentum actually added.
        # NONFINITE-safe: clamp(NaN) = NaN — a NaN ∇φ would sail through a
        # plain clamp into the pools.  A nonfinite kick keeps the OLD velocity.
        begin
            invρc = 1.0f0 / max(ρ, 1f-30)
            v1 = (s1 + d1) * invρc; v2 = (s2 + d2) * invρc; v3 = (s3 + d3) * invρc
            scap > 0.0f0 && (v1 = clamp(v1, -scap, scap); v2 = clamp(v2, -scap, scap);
                             v3 = clamp(v3, -scap, scap))
            d1 = ifelse(isfinite(v1), v1 * ρ - s1, 0.0f0)
            d2 = ifelse(isfinite(v2), v2 * ρ - s2, 0.0f0)
            d3 = ifelse(isfinite(v3), v3 * ρ - s3, 0.0f0)
        end
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
    scap = Float32(parse(Float64, get(ENV, "BAM_SMAXCAP", "0")))
    _grav_kick_k!(lev.be)(lev.S1, lev.S2, lev.S3, lev.Tau, lev.D, φ,
                          lev.live_d, _tabi(lev, :gi0), _tabf(lev, :gfr),
                          lev.Dsc, lev.Ssc, lev.Esc,
                          Float32(halfdt), invh, Float32(exp2(-l)), scap,
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

# gather |rhs| over interior cells into a compact (n,) buffer for ‖rhs‖∞
@kernel function _absrhs_k!(out, @Const(rhs), @Const(live_d),
                            B::Int32, ng::Int32, nd::Int32, stride::Int32)
    t = @index(Global)
    t0 = Int32(t) - Int32(1)
    B3 = B * B * B
    bi = t0 ÷ B3; c = t0 % B3
    @inbounds begin
        slot = live_d[bi + Int32(1)]
        i = c % B + ng; j = (c ÷ B) % B + ng; k = c ÷ (B * B) + ng
        out[Int32(t)] = abs(rhs[(slot - Int32(1)) * stride + _lidx(i, j, k, nd)])
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

`restol > 0` enables residual-based early exit: the loop breaks once the ∞-norm
residual drops below `restol·‖rhs‖∞` (the source norm — the standard Enzo/RAMSES
convergence test, NOT relative to the initial residual, which an already-converged
warm φ cannot reduce further).  Checked before the first sweep and every
`check_every` sweeps thereafter; `nsweep` is the hard cap.  This is the deep-
subcycled fast path — a warm-started φ under 2:1 subcycling is usually already
near-converged (`res/‖rhs‖ < restol` on entry ⇒ zero sweeps), so a fixed sweep
budget is wasted work.
"""
function solve_gravity_level!(hier::AMRHierarchy, l::Int; source_coef::Real,
                              nsweep::Int = 60, rho_ext = nothing,
                              rho_mean::Real = 0, init::Symbol = :warm,
                              use_dm::Bool = false, residual::Bool = true,
                              restol::Real = 0.0, check_every::Int = 4,
                              bc_phi = nothing)
    lev = hier.levels[l + 1]
    plev = l >= 1 ? hier.levels[l] : lev
    # bc_phi: override the parent φ used for the Dirichlet ring (+ :parent init) —
    # subcycled children pass the TIME-INTERPOLATED parent state (repic dyadic blend)
    bcφ = bc_phi === nothing ? plev.phi : bc_phi
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
        _rect_prolong_tl_k!(lev.be)(lev.phi, bcφ, pro.jobs, pro.cellstart,
                                    Int32(pro.njobs), Int32(lev.nd),
                                    Int32(lev.stride); ndrange = pro.total)
        # the shell fill also touched sibling-covered ghosts — restore fine data
        # immediately so a warm restart resumes from the converged state
        _run_copy!(lev.be, sib, lev.phi, lev.phi, lev.nd, lev.stride)
    end
    if init === :parent && l >= 1                          # interior seed
        tab = to_device_table(lev.be, build_prolong_jobs(lev, plev; interior = true))
        _run_prolong!(lev.be, tab, lev.phi, bcφ, bcφ, 0.0f0,
                      lev.nd, lev.stride)
    end
    # per-sweep exchange: FACES ONLY (the 7-point stencil never reads
    # edge/corner ghosts; the full 26-rect shell per sweep was ~20% of all GPU
    # time at 128³) — one full-shell exchange after the loop restores the
    # corners for trilinear consumers (particle gather).
    sibf = _tabt(lev, :sibf)
    # residual-based early exit (subcycled deep levels warm-start nearly
    # converged — a warm φ that already meets restol skips the whole sweep
    # budget).  resbuf doubles as the final-diagnostic scratch.
    resbuf = (restol > 0 || residual) ? device_zeros(lev.be, Float32, (n,)) : nothing
    rtarget = 0.0f0
    if restol > 0 && nsweep > 0
        _absrhs_k!(lev.be)(resbuf, lev.rhs, lev.live_d, Int32(lev.B),
                           Int32(lev.ng), Int32(lev.nd), Int32(lev.stride); ndrange = n)
        rtarget = Float32(restol) * maximum(resbuf)      # restol·‖rhs‖∞
    end
    for s in 1:nsweep
        # convergence test BEFORE sweeping: an already-converged warm φ exits at 0
        if rtarget > 0 && (s == 1 || (s - 1) % check_every == 0)
            _residual_k!(lev.be)(resbuf, lev.phi, lev.rhs, lev.live_d, Int32(lev.B),
                                 Int32(lev.ng), Int32(lev.nd), Int32(lev.stride); ndrange = n)
            maximum(resbuf) <= rtarget && break
        end
        _rb_relax_k!(lev.be)(lev.phi, lev.rhs, lev.live_d, Int32(0), Int32(lev.B),
                             Int32(lev.ng), Int32(lev.nd), Int32(lev.stride); ndrange = n)
        _rb_relax_k!(lev.be)(lev.phi, lev.rhs, lev.live_d, Int32(1), Int32(lev.B),
                             Int32(lev.ng), Int32(lev.nd), Int32(lev.stride); ndrange = n)
        _run_copy!(lev.be, sibf, lev.phi, lev.phi, lev.nd, lev.stride)
    end
    nsweep > 0 && _run_copy!(lev.be, sib, lev.phi, lev.phi, lev.nd, lev.stride)
    residual || return 0.0f0                     # skip the diagnostic reduce +
    res = resbuf                                  # D→H sync on the hot path
    _residual_k!(lev.be)(res, lev.phi, lev.rhs, lev.live_d, Int32(lev.B),
                         Int32(lev.ng), Int32(lev.nd), Int32(lev.stride); ndrange = n)
    return maximum(res)
end

# KDK kick from the LEVEL's own φ pool (block-local central difference)
@kernel function _grav_kick_pool_k!(S1, S2, S3, Tau, @Const(D), @Const(phi),
                                    @Const(live_d), @Const(Dsc), @Const(Ssc),
                                    @Const(Esc), halfdt::Float32, inv2h::Float32,
                                    scap::Float32,
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
        # absolute post-kick velocity cap (the BAM_SMAXCAP twin for kicks):
        # kicks bypass the hydro epilogue guards — at a refinement-saturated
        # cell ∇φ is unbounded and ρ·g·dt overflows the f16 momentum window
        # (stored > 65504 → Inf → smax = Inf).  Clamp the POST-kick velocity
        # per component; the τ update uses the ACTUAL applied deltas, so the
        # energy stays consistent with the momentum actually added.
        # NONFINITE-safe: clamp(NaN) = NaN — a NaN ∇φ would sail through a
        # plain clamp into the pools.  A nonfinite kick keeps the OLD velocity.
        begin
            invρc = 1.0f0 / max(ρ, 1f-30)
            v1 = (s1 + d1) * invρc; v2 = (s2 + d2) * invρc; v3 = (s3 + d3) * invρc
            scap > 0.0f0 && (v1 = clamp(v1, -scap, scap); v2 = clamp(v2, -scap, scap);
                             v3 = clamp(v3, -scap, scap))
            d1 = ifelse(isfinite(v1), v1 * ρ - s1, 0.0f0)
            d2 = ifelse(isfinite(v2), v2 * ρ - s2, 0.0f0)
            d3 = ifelse(isfinite(v3), v3 * ρ - s3, 0.0f0)
        end
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
    scap = Float32(parse(Float64, get(ENV, "BAM_SMAXCAP", "0")))
    n = length(lev.live) * lev.B^3
    _grav_kick_pool_k!(lev.be)(lev.S1, lev.S2, lev.S3, lev.Tau, lev.D, lev.phi,
                               lev.live_d, lev.Dsc, lev.Ssc, lev.Esc,
                               Float32(halfdt), inv2h, scap, Int32(lev.B), Int32(lev.ng),
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
    # containment guard: a local nonfinite cell must NOT globalize through the
    # bulk (one NaN in the gathers poisons EVERY S/Tau on every level — the
    # bamr256L10 z=7 failure mode).  Skip the drag; the local fault stays
    # local and diagnosable.
    if !(isfinite(vb[1]) && isfinite(vb[2]) && isfinite(vb[3]) && isfinite(f))
        @warn "compton_drag!: nonfinite bulk/factor — drag skipped" vb f maxlog = 10
        return nothing
    end
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

"""
    refresh_phi_ghosts!(hier, l)

One Dirichlet prolong + full sibling exchange on level `l`'s φ ghost shell —
the same ring maintenance `solve_gravity_level!` performs, without sweeps.
Called coarse→fine at the END of `regrid!`: the fine-first W-cycle makes a
child level's FIRST solve read the parent φ ring before the parent's own
solve ever exchanges, and a parent created in the same regrid carries a
recycled (dead-block) ring — trilinear Dirichlet has no limiter, so that
garbage became ~250× accelerations at the fresh skins (the bamr256L10
step-213 K1 detonation).
"""
function refresh_phi_ghosts!(hier::AMRHierarchy, l::Int)
    lev = hier.levels[l + 1]
    isempty(lev.live) && return nothing
    if l >= 1
        plev = hier.levels[l]
        pro = _tabt(lev, :pro)
        if pro.total > 0
            _rect_prolong_tl_k!(lev.be)(lev.phi, plev.phi, pro.jobs, pro.cellstart,
                                        Int32(pro.njobs), Int32(lev.nd),
                                        Int32(lev.stride); ndrange = pro.total)
        end
    end
    sib = _tabt(lev, :sib)
    _run_copy!(lev.be, sib, lev.phi, lev.phi, lev.nd, lev.stride)
    return nothing
end


# ── masked multigrid level solve (RAMSES multigrid_fine scheme) ────────────────
#
# Plain RB-GS propagates boundary/long-wavelength information only ~√nsweep cells
# per solve — on zoom unions 100+ cells across it NEVER converges (measured: the
# level φ reached 0.44/0.05/0.006 of the converged solution at L1/L2/L3; fine
# particles felt no infall and zoom halos were pushed apart while the topgrid-FFT
# control collapsed correctly).  An unmasked periodic coarse solve amplifies (the
# near-ring boundary-layer defect gets an unpinned global response) — the cure,
# as in RAMSES multigrid_fine, is a MASKED V-cycle: every coarse level carries
# the union's Dirichlet mask (corrections vanish outside the footprint) and the
# smoother only updates masked cells.
#
# BlockAMR twist: instead of per-MG-level block tables, the union is gathered
# into a DENSE grid over its bounding box (+1-cell halo).  Zoom unions are
# compact (~100³ cells ≈ few MB), so the pyramid of dense mask/e/r arrays is
# tiny, the kernels are trivial 3-D KA kernels, and the whole V-cycle is
# self-contained.  Boundary values (the Dirichlet ring prolonged from the
# parent) live in unmasked cells adjacent to the union; corrections use e = 0
# there.  Coarse masks are conservative (a coarse cell is interior only if ALL
# 8 children are) — the correction misses a one-cell rim per coarse level,
# which the fine smoothing absorbs.

@kernel function _mgd_gather_int_k!(φd, rhsd, maskd, @Const(phi), @Const(rhs),
                                    @Const(live_d), @Const(rel0),
                                    nx::Int32, ny::Int32,
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
        # dense coords: rel block origin (+1 halo, +1 1-based).  rel0 is indexed
        # by LIST POSITION (bi), not slot — with Schwarz overlap a slot can be
        # a member of several sub-clusters, each with its own dense origin.
        di = rel0[3*bi+Int32(1)] + ci + Int32(2)
        dj = rel0[3*bi+Int32(2)] + cj + Int32(2)
        dk = rel0[3*bi+Int32(3)] + ck + Int32(2)
        g = (dk - Int32(1)) * (nx * ny) + (dj - Int32(1)) * nx + di
        φd[g]   = Float32(phi[idx])
        rhsd[g] = Float32(rhs[idx])
        maskd[g] = 1.0f0
    end
end

@kernel function _mgd_gather_ring_k!(φd, @Const(maskd), @Const(phi), @Const(live_d),
                                     @Const(rel0), nx::Int32, ny::Int32, nz::Int32,
                                     B::Int32, ng::Int32, nd::Int32, stride::Int32)
    t = @index(Global)
    t0 = Int32(t) - Int32(1)
    n3d = nd * nd * nd
    bi = t0 ÷ n3d; c = t0 % n3d
    @inbounds begin
        slot = live_d[bi + Int32(1)]
        u = c % nd; v = (c ÷ nd) % nd; w_ = c ÷ (nd * nd)
        # ghost-frame cells only (interiors were pass 1)
        inint = (u >= ng) & (u < B + ng) & (v >= ng) & (v < B + ng) &
                (w_ >= ng) & (w_ < B + ng)
        if !inint
            di = rel0[3*bi+Int32(1)] + (u - ng) + Int32(2)
            dj = rel0[3*bi+Int32(2)] + (v - ng) + Int32(2)
            dk = rel0[3*bi+Int32(3)] + (w_ - ng) + Int32(2)
            if di >= Int32(1) && di <= nx && dj >= Int32(1) && dj <= ny &&
               dk >= Int32(1) && dk <= nz
                g = (dk - Int32(1)) * (nx * ny) + (dj - Int32(1)) * nx + di
                # union-exterior cells only: these hold the parent-prolonged
                # Dirichlet values (benign write races — same parent source)
                maskd[g] == 0.0f0 && (φd[g] = Float32(phi[(slot-Int32(1))*stride + c + Int32(1)]))
            end
        end
    end
end

@kernel function _mgd_relax_k!(φd, @Const(rhsd), @Const(maskd), parity::Int32,
                               nx::Int32, ny::Int32, nz::Int32)
    t = @index(Global)
    t0 = Int32(t) - Int32(1)
    i = t0 % nx; j = (t0 ÷ nx) % ny; k = t0 ÷ (nx * ny)
    @inbounds if i > Int32(0) && i < nx - Int32(1) && j > Int32(0) && j < ny - Int32(1) &&
                 k > Int32(0) && k < nz - Int32(1) &&
                 ((i + j + k) & Int32(1)) == parity
        g = k * (nx * ny) + j * nx + i + Int32(1)
        if maskd[g] != 0.0f0
            φd[g] = (φd[g-Int32(1)] + φd[g+Int32(1)] + φd[g-nx] + φd[g+nx] +
                     φd[g-nx*ny] + φd[g+nx*ny] - rhsd[g]) * (1.0f0 / 6.0f0)
        end
    end
end

@kernel function _mgd_resid_k!(rd, @Const(φd), @Const(rhsd), @Const(maskd),
                               nx::Int32, ny::Int32, nz::Int32)
    t = @index(Global)
    t0 = Int32(t) - Int32(1)
    i = t0 % nx; j = (t0 ÷ nx) % ny; k = t0 ÷ (nx * ny)
    @inbounds begin
        g = k * (nx * ny) + j * nx + i + Int32(1)
        r = 0.0f0
        if i > Int32(0) && i < nx - Int32(1) && j > Int32(0) && j < ny - Int32(1) &&
           k > Int32(0) && k < nz - Int32(1) && maskd[g] != 0.0f0
            r = rhsd[g] - (φd[g-Int32(1)] + φd[g+Int32(1)] + φd[g-nx] + φd[g+nx] +
                           φd[g-nx*ny] + φd[g+nx*ny] - 6.0f0 * φd[g])
        end
        rd[g] = r
    end
end

# rc = 4·⟨r⟩ over the 2³ children (h² rescale); mask_c = 1 if ANY child is masked
# (expansive — a conservative all-8 mask erodes one cell per coarse level and
# starves long wavelengths of coarse correction entirely; the slight over-reach
# past the rim is a Dirichlet-0 correction absorbed by post-smoothing).
# Coarse dims (ncx,ncy,ncz) with 1-cell halo like the fine.
@kernel function _mgd_restrict_k!(rc, mc, @Const(rf), @Const(mf),
                                  ncx::Int32, ncy::Int32, ncz::Int32,
                                  nx::Int32, ny::Int32, nz::Int32)
    t = @index(Global)
    t0 = Int32(t) - Int32(1)
    i = t0 % ncx; j = (t0 ÷ ncx) % ncy; k = t0 ÷ (ncx * ncy)
    @inbounds begin
        gc = k * (ncx * ncy) + j * ncx + i + Int32(1)
        acc = 0.0f0; m = 0.0f0
        if i > Int32(0) && i < ncx - Int32(1) && j > Int32(0) && j < ncy - Int32(1) &&
           k > Int32(0) && k < ncz - Int32(1)
            fi = Int32(2) * i - Int32(1); fj = Int32(2) * j - Int32(1); fk = Int32(2) * k - Int32(1)
            for dk in Int32(0):Int32(1), dj in Int32(0):Int32(1), di in Int32(0):Int32(1)
                ii = fi + di; jj = fj + dj; kk = fk + dk
                if (ii >= Int32(0)) & (ii < nx) & (jj >= Int32(0)) & (jj < ny) &
                   (kk >= Int32(0)) & (kk < nz)
                    gf = kk * (nx * ny) + jj * nx + ii + Int32(1)
                    acc += rf[gf]                     # r = 0 at unmasked cells already
                    m = max(m, mf[gf])
                end
            end
        end
        rc[gc] = m != 0.0f0 ? 0.5f0 * acc : 0.0f0     # (4/8)·Σ = 4·⟨r⟩
        mc[gc] = m
    end
end

# e_f += ω·e_c (piecewise-constant injection, under-relaxed) at masked fine cells.
# ω < 1 damps the two-grid over-correction (measured ≈1.6× with PC injection on
# the masked union → undamped cycles RING at ρ≈0.62; ω≈0.65 makes them monotone).
@kernel function _mgd_prolongadd_k!(ef, @Const(ec), @Const(mf), ω::Float32,
                                    nx::Int32, ny::Int32,
                                    ncx::Int32, ncy::Int32, ncz::Int32)
    t = @index(Global)
    t0 = Int32(t) - Int32(1)
    i = t0 % nx; j = (t0 ÷ nx) % ny; k = t0 ÷ (nx * ny)
    @inbounds begin
        g = k * (nx * ny) + j * nx + i + Int32(1)
        if mf[g] != 0.0f0
            ci = (i + Int32(1)) ÷ Int32(2); cj = (j + Int32(1)) ÷ Int32(2); ck = (k + Int32(1)) ÷ Int32(2)
            if (ci < ncx) & (cj < ncy) & (ck < ncz)
                gc = ck * (ncx * ncy) + cj * ncx + ci + Int32(1)
                ef[g] += ω * ec[gc]
            end
        end
    end
end

@kernel function _mgd_scatter_k!(phi, @Const(φd), @Const(live_d), @Const(rel0),
                                 nx::Int32, ny::Int32,
                                 B::Int32, ng::Int32, nd::Int32, stride::Int32)
    t = @index(Global)
    t0 = Int32(t) - Int32(1)
    B3 = B * B * B
    bi = t0 ÷ B3; c = t0 % B3
    @inbounds begin
        slot = live_d[bi + Int32(1)]
        base = (slot - Int32(1)) * stride
        ci = c % B; cj = (c ÷ B) % B; ck = c ÷ (B * B)
        di = rel0[3*bi+Int32(1)] + ci + Int32(2)
        dj = rel0[3*bi+Int32(2)] + cj + Int32(2)
        dk = rel0[3*bi+Int32(3)] + ck + Int32(2)
        g = (dk - Int32(1)) * (nx * ny) + (dj - Int32(1)) * nx + di
        phi[base + _lidx(ci + ng, cj + ng, ck + ng, nd)] = φd[g]
    end
end

# one masked V-cycle on the dense pyramid (arrays pre-allocated in `pyr`)
function _mgd_vcycle!(be, pyr, lvl::Int, nu1::Int, nu2::Int, nub::Int; ω::Float32 = 0.65f0)
    φ, rhs, m, r, dims = pyr[lvl]
    nx, ny, nz = Int32.(dims)
    n = prod(dims)
    if lvl == length(pyr)                       # coarsest: smooth hard
        for _ in 1:nub
            _mgd_relax_k!(be)(φ, rhs, m, Int32(0), nx, ny, nz; ndrange = n)
            _mgd_relax_k!(be)(φ, rhs, m, Int32(1), nx, ny, nz; ndrange = n)
        end
        return nothing
    end
    for _ in 1:nu1                              # pre-smooth
        _mgd_relax_k!(be)(φ, rhs, m, Int32(0), nx, ny, nz; ndrange = n)
        _mgd_relax_k!(be)(φ, rhs, m, Int32(1), nx, ny, nz; ndrange = n)
    end
    _mgd_resid_k!(be)(r, φ, rhs, m, nx, ny, nz; ndrange = n)
    φc, rhsc, mc, rc, dimsc = pyr[lvl + 1]
    ncx, ncy, ncz = Int32.(dimsc)
    _mgd_restrict_k!(be)(rhsc, mc, r, m, ncx, ncy, ncz, nx, ny, nz;
                         ndrange = prod(dimsc))
    fill!(φc, 0.0f0)                            # correction: e = 0 everywhere incl. ring
    _mgd_vcycle!(be, pyr, lvl + 1, nu1, nu2, nub; ω)   # deeper stages undamped
    _mgd_prolongadd_k!(be)(φ, φc, m, lvl == 1 ? ω : 1.0f0, nx, ny, ncx, ncy, ncz;
                           ndrange = n)
    for _ in 1:nu2                              # post-smooth
        _mgd_relax_k!(be)(φ, rhs, m, Int32(0), nx, ny, nz; ndrange = n)
        _mgd_relax_k!(be)(φ, rhs, m, Int32(1), nx, ny, nz; ndrange = n)
    end
    return nothing
end

"""
    solve_gravity_level_mg!(hier, l; source_coef, rho_mean=0, use_dm=false,
                            ncycle=6, nu1=3, nu2=3, nub=80, init=:warm) -> defect∞

Masked-multigrid version of [`solve_gravity_level!`](@ref) for `l ≥ 1` — the
RAMSES `multigrid_fine` scheme on a dense bounding-box grid of the block union
(see the section comment).  Same equation, same Dirichlet-from-parent ring, but
CONVERGES on arbitrarily large unions: each V-cycle contracts the error at all
wavelengths.  Returns the ∞-norm defect (h²-scaled) after the last cycle.
"""
function solve_gravity_level_mg!(hier::AMRHierarchy, l::Int; source_coef::Real,
                                 rho_mean::Real = 0, use_dm::Bool = false,
                                 rho_ext = nothing, init::Symbol = :warm,
                                 ncycle::Int = 6, nu1::Int = 3, nu2::Int = 3,
                                 nub::Int = 80, ω::Real = 0.65, bc_phi = nothing,
                                 mgeps::Real = 0)
    l >= 1 || error("solve_gravity_level_mg!: level 0 is the global FFT — solve it there")
    lev = hier.levels[l + 1]
    isempty(lev.live) && return 0.0f0
    # rhs + Dirichlet ring (+ optional interior seed), no sweeps — the existing path
    solve_gravity_level!(hier, l; source_coef, rho_mean, use_dm, rho_ext,
                         nsweep = 0, init, residual = false, bc_phi)
    # cached union geometry + dense pyramids (topology-keyed: build_level_tables!
    # deletes :mgd at every rebuild, so a static mesh pays the host pass ONCE).
    # The union is split into CONNECTED CLUSTERS (grid-hash union-find, link radius
    # 2B cells) and each cluster gets its OWN bounding-box pyramid: stray adaptive
    # blocks far from the main zoom rectangle otherwise blow the single bbox up to
    # a box-scale dense grid (the 14.8 GiB OOM).  Disjoint clusters are independent
    # Dirichlet problems — solving them separately IS the one-way scheme.
    cache = get(lev.tabs, :mgd, nothing)
    if cache === nothing
        P = Int128(lev.P[1])
        anc = lev.meta[lev.live[1]].origin
        nlive = length(lev.live)
        rel = Matrix{Int64}(undef, 3, nlive)
        for (n_, s) in enumerate(lev.live)
            m = lev.meta[s]
            for d in 1:3
                rel[d, n_] = Int64(mod(Int128(m.origin[d]) - Int128(anc[d]) + P ÷ 2, P) - P ÷ 2)
            end
        end
        # union-find over blocks, linked when within 2B cells (grid-hash buckets)
        parent = collect(1:nlive)
        find(i) = (while parent[i] != i; parent[i] = parent[parent[i]]; i = parent[i]; end; i)
        link = 2 * lev.B
        buck = Dict{NTuple{3,Int64},Vector{Int}}()
        for n_ in 1:nlive
            k = ntuple(d -> fld(rel[d, n_], link), 3)
            push!(get!(buck, k, Int[]), n_)
        end
        for n_ in 1:nlive
            k = ntuple(d -> fld(rel[d, n_], link), 3)
            for dz in -1:1, dy in -1:1, dx in -1:1
                nb = get(buck, (k[1]+dx, k[2]+dy, k[3]+dz), nothing)
                nb === nothing && continue
                for m_ in nb
                    if maximum(abs.(rel[:, n_] .- rel[:, m_])) <= link
                        ra = find(n_); rb = find(m_)
                        ra != rb && (parent[ra] = rb)
                    end
                end
            end
        end
        groups = Dict{Int,Vector{Int}}()
        for n_ in 1:nlive
            push!(get!(groups, find(n_), Int[]), n_)
        end
        # Over-budget connected clusters are RECURSIVELY BISECTED (median block
        # along the longest bbox axis) into shrink-wrapped sub-clusters — a
        # union that percolates across the whole zoom sphere (haloA: L2 at
        # 850×1010×818, 5% fill → 14 GiB dense; L3 would be ~90 GiB) cannot get
        # one dense pyramid.  Each part is its own Dirichlet solve WITH
        # OVERLAP: it also gathers (and masks as interior) the component's
        # blocks within `ovl` cells of its own bbox, but scatters ONLY its own
        # blocks — overlapping Schwarz.  The 1-cell halo alone converges at
        # ~0.9/pass (useless); a ~B-cell overlap contracts strongly, and the
        # per-substep warm start makes the seam lag a small multiple of the
        # per-substep φ drift.  Ring cells outside the overlap take the
        # neighbor part's pre-solve φ via the sibling ghost exchange.
        # Budget: BAM_MGCLUSTER_GIB (default 24, own-bbox) — splitting is a
        # LAST-RESORT scaling fallback: the seam rings lag the neighbor part's
        # φ by one substep, which measurably perturbs a fast-deepening
        # potential (haloA: split-MG shifted L7 formation z=24.75→24.55 and
        # fed the λ-ratchet); prefer one dense pyramid whenever it fits
        # (managed memory pages it).  Overlap: BAM_MGOVERLAP cells (default B).
        # rel origins are PER-CLUSTER, indexed by list position (a slot can be
        # a member of several parts).
        clusters = Vector{Tuple{Any,Int,Int,NTuple{3,Int},Any,Vector{Any}}}()
        budget = Int64(round(parse(Float64, get(ENV, "BAM_MGCLUSTER_GIB", "24.0")) * (Int64(1) << 30)))
        ovl = clamp(parse(Int, get(ENV, "BAM_MGOVERLAP", string(lev.B))), 0, 4 * lev.B)
        for g0 in values(groups)
            stack = Vector{Vector{Int}}([g0])
            while !isempty(stack)
                mem = pop!(stack)
                lo = ntuple(d -> minimum(rel[d, n_] for n_ in mem), 3)
                hi = ntuple(d -> maximum(rel[d, n_] for n_ in mem) + lev.B, 3)
                dims = ntuple(d -> Int(hi[d] - lo[d]) + 2, 3)
                if prod(dims) * 4 * 5 > budget && length(mem) > 1
                    ax = argmax(dims)
                    srt = sort(mem; by = n_ -> rel[ax, n_])
                    h = length(srt) ÷ 2
                    push!(stack, srt[1:h]); push!(stack, srt[h+1:end])
                    continue
                end
                # overlap borrow: component blocks whose extent intersects the
                # own bbox grown by ovl (skip when the part IS the component)
                ext = Int[]
                if length(mem) < length(g0) && ovl > 0
                    inmem = BitSet(mem)
                    for n_ in g0
                        n_ in inmem && continue
                        hit = true
                        for d in 1:3
                            (rel[d, n_] < hi[d] + ovl && rel[d, n_] + lev.B > lo[d] - ovl) ||
                                (hit = false; break)
                        end
                        hit && push!(ext, n_)
                    end
                end
                all_ = isempty(ext) ? mem : vcat(mem, ext)
                glo = ntuple(d -> minimum(rel[d, n_] for n_ in all_), 3)
                ghi = ntuple(d -> maximum(rel[d, n_] for n_ in all_) + lev.B, 3)
                gdims = ntuple(d -> Int(ghi[d] - glo[d]) + 2, 3)
                relsub = Vector{Int32}(undef, 3 * length(all_))
                for (i_, n_) in enumerate(all_), d in 1:3
                    relsub[3*(i_-1)+d] = Int32(rel[d, n_] - glo[d])
                end
                livesub = to_device(lev.be, Int32[lev.live[n_] for n_ in all_], Int32)
                relsub_d = to_device(lev.be, relsub, Int32)
                pyr = Vector{Any}()
                dd = gdims
                while true
                    push!(pyr, (device_zeros(lev.be, Float32, (prod(dd),)),
                                device_zeros(lev.be, Float32, (prod(dd),)),
                                device_zeros(lev.be, Float32, (prod(dd),)),
                                device_zeros(lev.be, Float32, (prod(dd),)), dd))
                    minimum(dd) <= 6 && break
                    dd = ntuple(d -> cld(dd[d] - 2, 2) + 2, 3)   # halve interior, keep halo
                end
                push!(clusters, (livesub, length(all_), length(mem), gdims, relsub_d, pyr))
            end                                   # while !isempty(stack)
        end                                       # for g0
        cache = clusters
        lev.tabs[:mgd] = cache
    end
    clusters = cache
    defect = 0.0f0
    for (livesub, nall, nown, dims, relsub, pyr) in clusters
        φd, rhsd, maskd, rd, _ = pyr[1]
        fill!(φd, 0.0f0); fill!(rhsd, 0.0f0); fill!(maskd, 0.0f0)
        nx, ny, nz = Int32.(dims)
        _mgd_gather_int_k!(lev.be)(φd, rhsd, maskd, lev.phi, lev.rhs, livesub, relsub,
                                   nx, ny, Int32(lev.B), Int32(lev.ng), Int32(lev.nd),
                                   Int32(lev.stride); ndrange = nall * lev.B^3)
        _mgd_gather_ring_k!(lev.be)(φd, maskd, lev.phi, livesub, relsub, nx, ny, nz,
                                    Int32(lev.B), Int32(lev.ng), Int32(lev.nd),
                                    Int32(lev.stride); ndrange = nall * lev.nd^3)
        # MONOTONICITY GUARD: on spongy percolated unions (thousands of
        # scattered Jeans-driven blocks) the masked V-cycle correction can
        # AMPLIFY instead of contract — a slow, silent divergence (haloA: φ
        # grew to ~1e11, 16 decades above physical, over ~15 root steps; the
        # rhs-implied solution is ~1e4).  Every 4th solve: measure the defect
        # before/after and revert non-contracting corrections.  Every solve:
        # ONE fused reduction (max|φ| with nonfinite→Inf) enforcing the
        # absolute sanity window — the anti-runaway backstop at ~1/6 the
        # guard cost (the growth we chased was ~×1.2/root-step, so a 4-solve
        # cadence catches divergence within a root step).  Contract-or-freeze,
        # never amplify.  Proper cure (Krylov-wrapped V-cycle / better sponge
        # coarse masks) is future solver work.
        # RAMSES-style ε EARLY EXIT (multigrid_fine_commons: iterate until
        # err < epsilon): with per-substep warm starts and λ_J resolved by
        # hundreds of cells, φ changes by ~1/N_J per fine substep — the warm
        # start is already converged on most solves.  Test the residual FIRST
        # and skip the cycles (and the scatter — the pool already holds this
        # φd) when r0 ≤ ε·max|rhs|.  mgeps=0 disables (always cycle).
        _mgd_resid_k!(lev.be)(rd, φd, rhsd, maskd, nx, ny, nz; ndrange = prod(dims))
        r0 = maximum(abs, rd)
        if mgeps > 0 && isfinite(r0) && r0 <= mgeps * maximum(abs, rhsd)
            defect = max(defect, r0)
            continue
        end
        for _ in 1:ncycle
            _mgd_vcycle!(lev.be, pyr, 1, nu1, nu2, nub; ω = Float32(ω))
        end
        # MONOTONICITY GUARD (contract-or-freeze; see the divergence note
        # above): with r0 always in hand, check every cycled solve.
        mxφ = mapreduce(x -> isfinite(x) ? abs(x) : Inf32, max, φd; init = 0.0f0)
        okc = mxφ < 1.0f8                       # nonfinite folded to Inf → fails
        r1 = 0.0f0
        if okc
            _mgd_resid_k!(lev.be)(rd, φd, rhsd, maskd, nx, ny, nz; ndrange = prod(dims))
            r1 = maximum(abs, rd)
            okc = isfinite(r1) && r1 <= r0 * 1.001f0
        end
        if okc
            defect = max(defect, r1)
            # scatter ONLY the part's own blocks (own-first in livesub) —
            # overlap blocks belong to (and are written by) their own part
            _mgd_scatter_k!(lev.be)(lev.phi, φd, livesub, relsub, nx, ny,
                                    Int32(lev.B), Int32(lev.ng), Int32(lev.nd),
                                    Int32(lev.stride); ndrange = nown * lev.B^3)
        else
            defect = max(defect, isfinite(r0) ? r0 : 0.0f0)
            get(ENV, "BAM_NANSTAGE", "0") == "1" &&
                (println("MGDIAG REVERT l=$l dims=$dims nall=$nall: r0=$r0 r1=$r1 max|φ|=$mxφ");
                 flush(stdout))
        end
    end
    _run_copy!(lev.be, _tabt(lev, :sib), lev.phi, lev.phi, lev.nd, lev.stride)
    return defect
end

# ══ MG2: block-topology multigrid (RAMSES-spirit, no big dense pyramids) ══════
#
# The dense-bbox MG above gathers the WHOLE union into a dense grid whose
# volume scales with the bounding box, not the block count — 15-20× waste on
# sponges, and at high Jeans resolution the per-level pyramids (10+ GB each)
# exceed the device and page over PCIe on EVERY substep solve.  MG2 keeps all
# heavy work on the RESIDENT block pools:
#
#   smooth (RB on pools)  →  signed residual (pools, compact)  →  restrict to
#   a SMALL dense coarse grid at ÷cfac (position-based atomic averaging —
#   alignment-free under free block offsets; L10's 800³ bbox becomes 200³ =
#   32 MB)  →  existing masked-MG V-cycles solve the correction (Dirichlet-0
#   halo)  →  trilinear prolong-add back to the pools  →  smooth.
#
# Traffic per cycle ≈ (2·nsm+3) passes over the level's own cells at device
# bandwidth; the only dense object is bbox/cfac³.  The cfac jump (4×) leaves
# the 2–4Δx error band to the smoother — nsm≈4 RB pairs cover it; long
# wavelengths ride the coarse solve.  ε early-exit and an absolute correction
# guard as in the dense path.

# signed residual rhs − Lφ into a compact live-cell buffer (restriction source)
@kernel function _residual_s_k!(res, @Const(phi), @Const(rhs), @Const(live_d),
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
        res[Int32(t)] = rhs[idx] - (phi[idx - Int32(1)] + phi[idx + Int32(1)] +
                                    phi[idx - nd] + phi[idx + nd] +
                                    phi[idx - nd*nd] + phi[idx + nd*nd] -
                                    6.0f0 * phi[idx])
    end
end

# restrict the compact residual to the ÷cfac dense coarse grid (atomic mean·cfac²:
# w = cfac²/cfac³ = 1/cfac maps pool-units h²r to coarse-units H²r)
@kernel function _mg2_restrict_k!(rc, mc, @Const(res), @Const(rel0),
                                  w::Float32, cfac::Int32,
                                  nx::Int32, ny::Int32, B::Int32)
    t = @index(Global)
    t0 = Int32(t) - Int32(1)
    B3 = B * B * B
    bi = t0 ÷ B3; c = t0 % B3
    @inbounds begin
        i = c % B; j = (c ÷ B) % B; k = c ÷ (B * B)
        gx = (rel0[3*bi+Int32(1)] + i) ÷ cfac + Int32(2)   # +1 halo, +1 one-based
        gy = (rel0[3*bi+Int32(2)] + j) ÷ cfac + Int32(2)
        gz = (rel0[3*bi+Int32(3)] + k) ÷ cfac + Int32(2)
        g = (gz - Int32(1)) * (nx * ny) + (gy - Int32(1)) * nx + gx
        KA.@atomic rc[g] += w * res[Int32(t)]
        mc[g] = 1.0f0                                       # benign write race
    end
end

# trilinear prolong-add of the dense coarse correction into the pool φ
@kernel function _mg2_prolongadd_k!(phi, @Const(ec), @Const(live_d), @Const(rel0),
                                    cfac::Int32, nx::Int32, ny::Int32, nz::Int32,
                                    B::Int32, ng::Int32, nd::Int32, stride::Int32)
    t = @index(Global)
    t0 = Int32(t) - Int32(1)
    B3 = B * B * B
    bi = t0 ÷ B3; c = t0 % B3
    @inbounds begin
        slot = live_d[bi + Int32(1)]
        base = (slot - Int32(1)) * stride
        i = c % B; j = (c ÷ B) % B; k = c ÷ (B * B)
        # coarse-grid index space: coarse cell cc's center sits at index cc+2
        x = (Float32(rel0[3*bi+Int32(1)] + i) + 0.5f0) / Float32(cfac) + 1.5f0
        y = (Float32(rel0[3*bi+Int32(2)] + j) + 0.5f0) / Float32(cfac) + 1.5f0
        z = (Float32(rel0[3*bi+Int32(3)] + k) + 0.5f0) / Float32(cfac) + 1.5f0
        ix = unsafe_trunc(Int32, floor(x)); fx = x - Float32(ix)
        iy = unsafe_trunc(Int32, floor(y)); fy = y - Float32(iy)
        iz = unsafe_trunc(Int32, floor(z)); fz = z - Float32(iz)
        ix = clamp(ix, Int32(1), nx - Int32(1)); iy = clamp(iy, Int32(1), ny - Int32(1))
        iz = clamp(iz, Int32(1), nz - Int32(1))
        E(a, b, cc) = ec[(cc - Int32(1)) * (nx * ny) + (b - Int32(1)) * nx + a]
        e = (1-fx)*(1-fy)*(1-fz)*E(ix,   iy,   iz  ) + fx*(1-fy)*(1-fz)*E(ix+Int32(1), iy,   iz  ) +
            (1-fx)*   fy *(1-fz)*E(ix,   iy+Int32(1), iz  ) + fx*   fy *(1-fz)*E(ix+Int32(1), iy+Int32(1), iz  ) +
            (1-fx)*(1-fy)*   fz *E(ix,   iy,   iz+Int32(1)) + fx*(1-fy)*   fz *E(ix+Int32(1), iy,   iz+Int32(1)) +
            (1-fx)*   fy *   fz *E(ix,   iy+Int32(1), iz+Int32(1)) + fx*   fy *   fz *E(ix+Int32(1), iy+Int32(1), iz+Int32(1))
        idx = base + _lidx(i + ng, j + ng, k + ng, nd)
        phi[idx] += e
    end
end

"""
    solve_gravity_level_mg2!(hier, l; source_coef, ...) -> defect

Block-topology multigrid level solve: RB smoothing + residual on the RESIDENT
block pools, one small dense coarse grid (union bbox ÷ `cfac`, masked-MG
bottom, Dirichlet-0 correction halo), trilinear prolong-add.  Replaces the
dense-bbox path wherever the pyramid would not fit the device (high Jeans
resolution) or the union is a sponge (percolated Jeans levels).  `mgeps > 0`
enables the RAMSES-style relative-residual early exit.
"""
function solve_gravity_level_mg2!(hier::AMRHierarchy, l::Int; source_coef::Real,
                                  rho_mean::Real = 0, use_dm::Bool = false,
                                  rho_ext = nothing, ncycle::Int = 2, nsm::Int = 4,
                                  cfac::Int = 4, nub::Int = 60, mgeps::Real = 0,
                                  bc_phi = nothing)
    l >= 1 || error("solve_gravity_level_mg2!: level 0 is the global FFT")
    lev = hier.levels[l + 1]
    isempty(lev.live) && return 0.0f0
    # rhs + Dirichlet ring (+ sibling ghosts), no sweeps — the existing pre-pass
    solve_gravity_level!(hier, l; source_coef, rho_mean, use_dm, rho_ext,
                         nsweep = 0, residual = false, bc_phi)
    n = length(lev.live) * lev.B^3
    cache = get(lev.tabs, :mg2, nothing)
    if cache === nothing
        P = Int128(lev.P[1])
        anc = lev.meta[lev.live[1]].origin
        nlive = length(lev.live)
        rel = Matrix{Int64}(undef, 3, nlive)
        for (n_, s) in enumerate(lev.live)
            m = lev.meta[s]
            for d in 1:3
                rel[d, n_] = Int64(mod(Int128(m.origin[d]) - Int128(anc[d]) + P ÷ 2, P) - P ÷ 2)
            end
        end
        lo = ntuple(d -> minimum(view(rel, d, :)), 3)
        hi = ntuple(d -> maximum(view(rel, d, :)) + lev.B, 3)
        dc = ntuple(d -> Int(cld(hi[d] - lo[d], cfac)) + 2, 3)
        rel0 = Vector{Int32}(undef, 3 * nlive)
        for n_ in 1:nlive, d in 1:3
            rel0[3*(n_-1)+d] = Int32(rel[d, n_] - lo[d])
        end
        pyr = Vector{Any}()
        dd = dc
        while true
            push!(pyr, (device_zeros(lev.be, Float32, (prod(dd),)),
                        device_zeros(lev.be, Float32, (prod(dd),)),
                        device_zeros(lev.be, Float32, (prod(dd),)),
                        device_zeros(lev.be, Float32, (prod(dd),)), dd))
            minimum(dd) <= 6 && break
            dd = ntuple(d -> cld(dd[d] - 2, 2) + 2, 3)
        end
        cache = (to_device(lev.be, rel0, Int32), dc, pyr,
                 device_zeros(lev.be, Float32, (n,)))
        lev.tabs[:mg2] = cache
    end
    rel0_d, dc, pyr, resb = cache
    sib = _tabt(lev, :sib); sibf = _tabt(lev, :sibf)
    nx, ny, nz = Int32.(dc)
    φc, rc, mc, _, _ = pyr[1]
    B = Int32(lev.B); ng = Int32(lev.ng); ndv = Int32(lev.nd); st = Int32(lev.stride)
    smooth! = () -> begin
        _rb_relax_k!(lev.be)(lev.phi, lev.rhs, lev.live_d, Int32(0), B, ng, ndv, st;
                             ndrange = n)
        _rb_relax_k!(lev.be)(lev.phi, lev.rhs, lev.live_d, Int32(1), B, ng, ndv, st;
                             ndrange = n)
        _run_copy!(lev.be, sibf, lev.phi, lev.phi, lev.nd, lev.stride)
    end
    defect = 0.0f0
    for cyc in 1:ncycle
        if mgeps > 0 && cyc == 1
            _absrhs_k!(lev.be)(resb, lev.rhs, lev.live_d, B, ng, ndv, st; ndrange = n)
            rn = maximum(resb)
            _residual_s_k!(lev.be)(resb, lev.phi, lev.rhs, lev.live_d, B, ng, ndv, st;
                                   ndrange = n)
            defect = maximum(abs, resb)
            defect <= mgeps * rn && return defect
        end
        for _ in 1:nsm; smooth!(); end
        _residual_s_k!(lev.be)(resb, lev.phi, lev.rhs, lev.live_d, B, ng, ndv, st;
                               ndrange = n)
        fill!(φc, 0.0f0); fill!(rc, 0.0f0); fill!(mc, 0.0f0)
        _mg2_restrict_k!(lev.be)(rc, mc, resb, rel0_d, Float32(1 / cfac), Int32(cfac),
                                 nx, ny, B; ndrange = n)
        _mgd_vcycle!(lev.be, pyr, 1, 2, 2, nub)
        _mgd_vcycle!(lev.be, pyr, 1, 2, 2, nub)
        # absolute correction guard (the anti-runaway backstop; RB smoothing is
        # non-expansive, so this is the only injection point to bound)
        mxe = mapreduce(x -> isfinite(x) ? abs(x) : Inf32, max, φc; init = 0.0f0)
        if mxe < 1.0f8
            _mg2_prolongadd_k!(lev.be)(lev.phi, φc, lev.live_d, rel0_d, Int32(cfac),
                                       nx, ny, nz, B, ng, ndv, st; ndrange = n)
            _run_copy!(lev.be, sib, lev.phi, lev.phi, lev.nd, lev.stride)
        end
        for _ in 1:nsm; smooth!(); end
    end
    _run_copy!(lev.be, sib, lev.phi, lev.phi, lev.nd, lev.stride)
    _residual_s_k!(lev.be)(resb, lev.phi, lev.rhs, lev.live_d, B, ng, ndv, st;
                           ndrange = n)
    return maximum(abs, resb)
end
