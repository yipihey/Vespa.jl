# ct.jl — constrained-transport MHD on the block hierarchy (`physics = :mhd,
# divb = :ct`), 2-D.
#
# Scheme: NestedMHD's CT mode (core/src/kernels.jl + physics.jl; the CT fix of
# Solaceonx/GLMMHDTurb@sweep-fixes): MUSCL-Hancock on cell-centred primitives
# with B_cc = face averages, one Riemann solve per face with Bn = the FACE value
# (F[Bn] = 0, no ψ), Gardiner & Stone (2005) CT-contact upwinded corner EMFs
# (reference EMF from the predictor velocity and the start-of-step field), face
# update B ← B − λ·curl E.  In 2-D the only EMF is E_z at cell corners; Bz is a
# cell-centred conserved quantity (no z faces).
#
# Layout.  FBx[idx] is Bx on the LOWER x-face of stored cell idx (FBy likewise),
# so a block's own faces are x-faces i ∈ [ng, ng+B] (j active) and y-faces
# j ∈ [ng, ng+B] (i active); the hi faces i = ng+B coincide with the right
# neighbour's i = ng.  Both blocks evolve a shared face from bitwise-identical
# corner EMFs (their scratch inputs are ghost copies of each other), and the
# sibling exchange re-syncs it from the neighbour after every coarse correction.
# A face with no same-level neighbour beyond it (C/F or domain edge) is owned by
# the block.  The cell pools Bx, By hold the face averages (active cells: from
# the faces after every update; ghosts: the usual cell ghost machinery).
#
# One level step = four launches: rec (half-step prims + slopes + reference EMF)
# → face fluxes (both directions, every face the corner EMFs need) → corner E_z
# → update (cells by flux divergence, faces by curl E, B_cc by averaging).
#
# AMR coupling (∇·B at round-off on every level — Stokes):
#   * cell reflux as GLM (lanes ρ, ρv, E, Bz from the face-flux scratch);
#   * E_z registers at coarse C/F corners (2-D: a coarse corner coincides with ONE
#     fine corner): Σ w_f E_f − w_c E_c, applied to the uncovered coarse faces that
#     touch a C/F corner;
#   * coarse faces on or inside the fine footprint are restricted (mean of the two
#     fine sub-faces), written to the face's OWNER block, then the coarse level's
#     faces are sibling-synced and B_cc recomputed;
#   * C/F ghost faces: coarse face (or mid-plane mean of the two coarse faces) +
#     MonCen transverse slope at ±¼ (NestedMHD prolong_faces), owned faces skipped;
#   * NEW blocks (regrid — NestedMHD's patches are static, this part is BlockAMR's):
#     faces on coarse faces prolonged sub-face-sum preserving, faces shared with an
#     existing fine block copied from it, interior mid-plane faces from a
#     Tóth–Roe-type solve per coarse cell ⇒ every new fine cell is exactly
#     solenoidal; E is corrected for the new B_cc;
#   * domain BCs: the wall face is owned (evolved); ghost faces mirror about it,
#     :reflecting negates the normal face, :inflow sets the inflow field.

using .MHDPhys: riemann_x, rot_to, rot_flux_from, moncen

# cell-centred primitive for CT (ψ ≡ 0); dual-energy thermal part (mhd.jl)
@inline function _ct_prim(::Type{T}, D, S1, S2, S3, E, Ge, Bx, By, Bz, idx, dsc, ssc, esc, gsc,
                          gamma::T, smallr::T, pfl::T, eta1::T, dev::Val) where {T}
    u = _mload_de(T, D, S1, S2, S3, E, Ge, Bx, By, Bz, Bz, idx, dsc, ssc, esc, gsc, eta1, dev)
    cons2prim((u[1], u[2], u[3], u[4], u[5], u[6], u[7], u[8], zero(T)), gamma, smallr, pfl)
end

# CT Riemann in direction `dir`: Bn is the single face value; F[Bn] = 0, ψ ≡ 0.
@fastmath @inline function _ct_riemann(rv::Val, Lq::NTuple{9,T}, Rq::NTuple{9,T}, dir::Int,
                                       gamma::T, smallr::T, pfl::T, bface::T) where {T}
    L0 = rot_to(Lq, dir); R0 = rot_to(Rq, dir)
    L = (max(L0[1],smallr), L0[2],L0[3],L0[4], max(L0[5],pfl), bface, L0[7],L0[8], L0[9])
    R = (max(R0[1],smallr), R0[2],R0[3],R0[4], max(R0[5],pfl), bface, R0[7],R0[8], R0[9])
    f = riemann_x(rv, L, R, gamma, bface, zero(T), zero(T), one(T))
    return rot_flux_from(f, dir)
end

# Gardiner & Stone (2005) CT-contact edge EMF (eqs. 41-45) — NestedMHD gs05.
@inline function _gs05(Ea0::T,Eam::T,Eb0::T,Ebm::T,mA0::T,mAm::T,mB0::T,mBm::T,
                       Er00::T,Erm0::T,Er0m::T,Ermm::T) where {T}
    up(m,lower,upper) = ifelse(m > zero(T), lower, ifelse(m < zero(T), upper, T(0.5)*(lower+upper)))
    A1 = up(mA0, Erm0-Ebm, Er00-Eb0)
    A3 = up(mAm, Ebm-Ermm, Eb0-Er0m)
    B1 = up(mB0, Er0m-Eam, Er00-Ea0)
    B3 = up(mBm, Eam-Ermm, Ea0-Erm0)
    T(0.25)*((Ea0+Eam+Eb0+Ebm) + (A3-A1) + (B3-B1))
end

# scratch lanes (lane-major, nt = cap·stride per lane)
# offsets (lane = off + v): mh 1:9, sx 10:18, sy 19:27, Fx 29:37, Fy 38:46;
# single lanes: the GS05 reference E_z (28) and the corner E_z (47); dual energy:
# the entropy flux of the x faces (48) and y faces (49), the advected K (50) and
# its minmod slopes (51 x, 52 y — see mhd.jl `_de_flux`)
const _CL_MH, _CL_SX, _CL_SY, _CL_FX, _CL_FY = 0, 9, 18, 28, 37
const _CL_ER, _CL_EZ = 28, 47
const _CL_GX, _CL_GY, _CL_K = 48, 49, 50
const _CL_SH = 53                                   # shock flag (mhd.jl shock switch)
const _CLANES = 53

function _ctrec(lev::Level)
    T = eltype(lev.D)
    n = _CLANES * lev.cap * lev.stride
    r = get(lev.tabs, :mrec, nothing)
    if !(r isa AbstractVector{T}) || length(r) != n
        r = device_zeros(lev.be, T, (n,))
        lev.tabs[:mrec] = r
    end
    return r
end

@inline _li2(i::Int32, j::Int32, nd::Int32) = j * nd + i + Int32(1)     # 2-D stored index

# ── pass 1: half-step prims + half slopes + GS05 reference EMF ────────────────
@kernel function _ct_rec_k!(rec, @Const(D), @Const(S1), @Const(S2), @Const(S3), @Const(E),
                            @Const(Ge), @Const(Bx), @Const(By), @Const(Bz),
                            @Const(live_d), @Const(Dsc), @Const(Ssc), @Const(Esc), @Const(Gsc),
                            gamma::T, smallr::T, pfl::T, dtdx::T, eta1::T, dev::Val{DE},
                            sw::Bool, seps::T,
                            B::Int32, ng::Int32, nd::Int32, stride::Int32, nt::Int) where {T,DE}
    t = @index(Global)
    t0 = Int32(t) - Int32(1)
    R = B + Int32(2); nr = R * R
    bi = t0 ÷ nr; c = t0 % nr
    @inbounds begin
        slot = live_d[bi + Int32(1)]
        base = (slot - Int32(1)) * stride
        dsc = Dsc[slot]; ssc = Ssc[slot]; esc = Esc[slot]; gsc = Gsc[slot]
        i = c % R + ng - Int32(1); j = c ÷ R + ng - Int32(1)
        @inline P(ii, jj) = _ct_prim(T, D, S1, S2, S3, E, Ge, Bx, By, Bz,
                                     base + _li2(ii, jj, nd), dsc, ssc, esc, gsc, gamma, smallr,
                                     pfl, eta1, dev)
        m0 = P(i, j)
        xm = P(i - Int32(1), j); xp = P(i + Int32(1), j); ym = P(i, j - Int32(1)); yp = P(i, j + Int32(1))
        shk = sw && _shock_flag(m0, xm, xp, ym, yp, m0, m0, false, seps)
        sx = shk ? prim_slope_mm(xm, m0, xp) : prim_slope(xm, m0, xp)
        sy = shk ? prim_slope_mm(ym, m0, yp) : prim_slope(ym, m0, yp)
        mh = cons2prim(hancock2(m0, sx, sy, dtdx, gamma), gamma, smallr, pfl)
        if mh[1] <= smallr || mh[5] <= pfl
            mh = m0
        end
        g = Int(base + _li2(i, j, nd))
        Base.Cartesian.@nexprs 9 v -> begin
            rec[(_CL_MH + v - 1) * nt + g] = mh[v]
            rec[(_CL_SX + v - 1) * nt + g] = sx[v]
            rec[(_CL_SY + v - 1) * nt + g] = sy[v]
        end
        # E_z reference: predictor velocity × start-of-step field (sweep-fixes 7256715)
        rec[(_CL_ER - 1) * nt + g] = mh[3] * m0[6] - mh[2] * m0[7]
        rec[(_CL_SH - 1) * nt + g] = shk ? one(T) : zero(T)
        if DE                                   # advected entropy: K_half + half slopes (mhd.jl)
            @inline Kc(ii, jj) = _de_Kc(T, D, Ge, base + _li2(ii, jj, nd), dsc, gsc, gamma)
            k0 = Kc(i, j)
            skx = _de_hs(Kc(i - Int32(1), j), k0, Kc(i + Int32(1), j))
            sky = _de_hs(Kc(i, j - Int32(1)), k0, Kc(i, j + Int32(1)))
            rec[(_CL_K - 1) * nt + g] = k0 - dtdx * (m0[2] * skx + m0[3] * sky)
            rec[_CL_K * nt + g] = skx
            rec[(_CL_K + 1) * nt + g] = sky
        end
    end
end

# ── pass 2: face fluxes (x: i ∈ [ng, ng+B], j ∈ [ng−1, ng+B]; y transposed) ───
@kernel function _ct_flux_k!(rec, @Const(FBx), @Const(FBy), @Const(live_d),
                             gamma::T, smallr::T, pfl::T, sw::Bool,
                             B::Int32, ng::Int32, nd::Int32, stride::Int32, nt::Int,
                             rv::Val) where {T}
    t = @index(Global)
    t0 = Int32(t) - Int32(1)
    R = B + Int32(2); nr = R * R
    bi = t0 ÷ nr; c = t0 % nr
    @inbounds begin
        slot = live_d[bi + Int32(1)]
        base = (slot - Int32(1)) * stride
        i = c % R + ng - Int32(1); j = c ÷ R + ng - Int32(1)
        gi = base + _li2(i, j, nd); g = Int(gi)
        mr = _rget(rec, _CL_MH, g, nt)
        if i >= ng
            gl = g - 1
            ml = _rget(rec, _CL_MH, gl, nt)
            L = iflo(padd(ml, _rget(rec, _CL_SX, gl, nt), one(T)), ml, smallr, pfl)
            Rq = iflo(padd(mr, _rget(rec, _CL_SX, g, nt), -one(T)), mr, smallr, pfl)
            f = (sw && (_rl(rec, _CL_SH, gl, nt) + _rl(rec, _CL_SH, g, nt) > zero(T))) ?
                _ct_riemann(Val(:hll), L, Rq, 1, gamma, smallr, pfl, T(FBx[gi])) :
                _ct_riemann(rv, L, Rq, 1, gamma, smallr, pfl, T(FBx[gi]))
            Base.Cartesian.@nexprs 9 v -> (rec[(_CL_FX + v - 1) * nt + g] = f[v])
            rec[(_CL_GX - 1) * nt + g] = _de_flux(rec, _CL_K, 1, gl, g, nt, f[1])
        end
        if j >= ng
            gl = g - Int(nd)
            ml = _rget(rec, _CL_MH, gl, nt)
            L = iflo(padd(ml, _rget(rec, _CL_SY, gl, nt), one(T)), ml, smallr, pfl)
            Rq = iflo(padd(mr, _rget(rec, _CL_SY, g, nt), -one(T)), mr, smallr, pfl)
            f = (sw && (_rl(rec, _CL_SH, gl, nt) + _rl(rec, _CL_SH, g, nt) > zero(T))) ?
                _ct_riemann(Val(:hll), L, Rq, 2, gamma, smallr, pfl, T(FBy[gi])) :
                _ct_riemann(rv, L, Rq, 2, gamma, smallr, pfl, T(FBy[gi]))
            Base.Cartesian.@nexprs 9 v -> (rec[(_CL_FY + v - 1) * nt + g] = f[v])
            rec[(_CL_GY - 1) * nt + g] = _de_flux(rec, _CL_K, 2, gl, g, nt, f[1])
        end
    end
end

@inline _rl(rec, lane::Int, g::Int, nt::Int) = @inbounds rec[(lane - 1) * nt + g]

# corner E_z at (i−½, j−½) = lower-left corner of stored cell (i, j) — NestedMHD ez
@inline function _ct_ez(rec, g::Int, nd::Int, nt::Int)
    @inline fx(c, q) = _rl(rec, _CL_FX + c, q, nt)
    @inline fy(c, q) = _rl(rec, _CL_FY + c, q, nt)
    @inline er(q) = _rl(rec, _CL_ER, q, nt)
    _gs05(-fx(7, g), -fx(7, g - nd), fy(6, g), fy(6, g - 1),
          fx(1, g), fx(1, g - nd), fy(1, g), fy(1, g - 1),
          er(g), er(g - 1), er(g - nd), er(g - nd - 1))
end

# ── pass 3: corner EMFs, i, j ∈ [ng, ng+B] ────────────────────────────────────
@kernel function _ct_emf_k!(rec, @Const(live_d), B::Int32, ng::Int32, nd::Int32,
                            stride::Int32, nt::Int)
    t = @index(Global)
    t0 = Int32(t) - Int32(1)
    R = B + Int32(1); nr = R * R
    bi = t0 ÷ nr; c = t0 % nr
    @inbounds begin
        slot = live_d[bi + Int32(1)]
        base = (slot - Int32(1)) * stride
        i = c % R + ng; j = c ÷ R + ng
        g = Int(base + _li2(i, j, nd))
        rec[(_CL_EZ - 1) * nt + g] = _ct_ez(rec, g, Int(nd), nt)
    end
end

# ── pass 4: update cells (flux divergence) + faces (curl E) + B_cc ─────────────
@kernel function _ct_update_k!(Do_, S1o_, S2o_, S3o_, Eo_, Geo_, Bxo_, Byo_, Bzo_, FBxo_, FByo_,
                               @Const(D), @Const(S1), @Const(S2), @Const(S3), @Const(E),
                               @Const(Ge), @Const(Bx), @Const(By), @Const(Bz), @Const(FBx), @Const(FBy),
                               @Const(rec), @Const(live_d),
                               @Const(Dsc), @Const(Ssc), @Const(Esc), @Const(Gsc),
                               gamma::T, smallr::T, pfl::T, dtdx::T, eta1::T, eta2::T,
                               dev::Val{DE},
                               B::Int32, ng::Int32, nd::Int32, stride::Int32, nt::Int) where {T,DE}
    t = @index(Global)
    t0 = Int32(t) - Int32(1)
    B2 = B * B
    bi = t0 ÷ B2; c = t0 % B2
    @inbounds begin
        slot = live_d[bi + Int32(1)]
        base = (slot - Int32(1)) * stride
        dsc = Dsc[slot]; ssc = Ssc[slot]; esc = Esc[slot]; gsc = Gsc[slot]
        i = c % B + ng; j = c ÷ B + ng
        idx = base + _li2(i, j, nd); g = Int(idx); n = Int(nd)
        u0 = _mload(T, D, S1, S2, S3, E, Bx, By, Bz, Bz, idx, dsc, ssc, esc)
        @inline fx(cc, q) = _rl(rec, _CL_FX + cc, q, nt)
        @inline fy(cc, q) = _rl(rec, _CL_FY + cc, q, nt)
        @inline dU(cc) = dtdx * ((fx(cc, g) - fx(cc, g + 1)) + (fy(cc, g) - fy(cc, g + n)))
        r1 = u0[1] + dU(1); r2 = u0[2] + dU(2); r3 = u0[3] + dU(3); r4 = u0[4] + dU(4)
        r5 = u0[5] + dU(5); r8 = u0[8] + dU(8)
        # faces: lower x-face (i), upper (i+1); lower y-face (j), upper (j+1)
        @inline ez(q) = _rl(rec, _CL_EZ, q, nt)
        bxl = T(FBx[idx])     - dtdx * (ez(g + n) - ez(g))
        bxh = T(FBx[idx + 1]) - dtdx * (ez(g + 1 + n) - ez(g + 1))
        byl = T(FBy[idx])     + dtdx * (ez(g + 1) - ez(g))
        byh = T(FBy[idx + nd]) + dtdx * (ez(g + n + 1) - ez(g + n))
        bxc = T(0.5) * (bxl + bxh); byc = T(0.5) * (byl + byh)
        # dual energy: advected entropy density (mhd.jl) → Ge at the new density
        ge0 = DE ? T(Ge[idx]) * T(gsc) : zero(T)
        gadv = zero(T)
        if DE                 # K clamped to the cell's 3×3 neighbourhood range (mhd.jl)
            kmin = _rl(rec, _CL_K, g, nt); kmax = kmin
            for dj in -1:1, di in -1:1
                kv = _rl(rec, _CL_K, g + di + dj * n, nt)
                kmin = min(kmin, kv); kmax = max(kmax, kv)
            end
            gadv = _de_ge_lim(_de_rhoK(ge0, u0[1], gamma) + dtdx *
                              ((_rl(rec, _CL_GX, g, nt) - _rl(rec, _CL_GX, g + 1, nt)) +
                               (_rl(rec, _CL_GY, g, nt) - _rl(rec, _CL_GY, g + n, nt))),
                              r1, kmin, kmax, gamma)
        end
        if r1 < smallr        # positivity floor (NestedMHD): static floored cell, B kept;
            # dual energy: keep the cell's entropy at the floor density, pressure capped
            # at the face neighbours' minimum (mhd.jl)
            gadv = DE ? max(min(_de_rhoK(ge0, u0[1], gamma) / max(u0[1], floatmin(T)) * smallr^gamma,
                                _de_pnb(rec, g, 1, n, 0, nt)) /
                            (gamma - one(T)), pfl / (gamma - one(T))) : pfl / (gamma - one(T))
            r1 = smallr; r2 = zero(T); r3 = zero(T); r4 = zero(T)
            r5 = gadv + T(0.5) * (bxc*bxc + byc*byc + r8*r8)
        end
        ok = isfinite(r1) & isfinite(r2) & isfinite(r3) & isfinite(r4) & isfinite(r5) &
             isfinite(r8) & isfinite(gadv)
        if !ok
            r1 = u0[1]; r2 = u0[2]; r3 = u0[3]; r4 = u0[4]; r5 = u0[5]; r8 = u0[8]
            gadv = ge0
        end
        gnew = zero(T)
        if DE
            demax = _de_demax(T, E, base, i, j, Int32(0), nd, esc, Int32(0))
            r5, gnew = _de_close(r1, r2, r3, r4, r5, bxc, byc, r8, gadv, demax, eta1, eta2,
                                 pfl / (gamma - one(T)))
        end
        q = cons2prim((r1, r2, r3, r4, r5, bxc, byc, r8, zero(T)), gamma, smallr, pfl)
        Do_[idx]  = _narrow(eltype(Do_), r1 / T(dsc))
        S1o_[idx] = _narrow(eltype(S1o_), r2 / T(ssc))
        S2o_[idx] = _narrow(eltype(S2o_), r3 / T(ssc))
        S3o_[idx] = _narrow(eltype(S3o_), r4 / T(ssc))
        Eo_[idx]  = _narrow(eltype(Eo_), r5 / T(esc))
        Geo_[idx] = _narrow(eltype(Geo_), (DE ? gnew : q[5] / (gamma - one(T))) / T(gsc))
        Bxo_[idx] = bxc; Byo_[idx] = byc; Bzo_[idx] = r8
        FBxo_[idx] = bxl; FByo_[idx] = byl
        i == ng + B - Int32(1) && (FBxo_[idx + 1] = bxh)     # the block's hi faces
        j == ng + B - Int32(1) && (FByo_[idx + nd] = byh)
    end
end

"""
    ct_level!(hier, l, λ)

One CT MUSCL-Hancock substep of level `l` (2-D): R → O (the caller swaps).
Requires current R ghosts.  The scratch (fluxes, corner EMFs) stays valid for
the C/F captures that follow.
"""
function ct_level!(hier::AMRHierarchy, l::Int, λ::Real)
    lev = hier.levels[l + 1]
    isempty(lev.live) && return nothing
    T = eltype(lev.D); γ, sr, pfl = _mpar(hier)
    rec = _ctrec(lev); nt = lev.cap * lev.stride
    nb = length(lev.live); B = lev.B
    a = (Int32(B), Int32(lev.ng), Int32(lev.nd), Int32(lev.stride), nt)
    g6 = gasfields(lev); dev, η1, η2 = _mde(hier)
    _ct_rec_k!(lev.be)(rec, g6..., mfields(lev)[1:3]..., lev.live_d, lev.Dsc, lev.Ssc, lev.Esc,
                       lev.Gsc, γ, sr, pfl, T(λ), η1, dev, hier.mhd.sw, T(hier.mhd.seps), a...;
                       ndrange = nb * (B + 2)^2)
    _ct_flux_k!(lev.be)(rec, cfields(lev)..., lev.live_d, γ, sr, pfl, hier.mhd.sw, a...,
                        Val(hier.mhd.riemann);
                        ndrange = nb * (B + 2)^2)
    _ct_emf_k!(lev.be)(rec, lev.live_d, a...; ndrange = nb * (B + 1)^2)
    _ct_update_k!(lev.be)(gasfields_o(lev)..., mfields_o(lev)[1:3]..., cfields_o(lev)...,
                          g6..., mfields(lev)[1:3]..., cfields(lev)..., rec, lev.live_d,
                          lev.Dsc, lev.Ssc, lev.Esc, lev.Gsc, γ, sr, pfl, T(λ), η1, η2, dev, a...;
                          ndrange = nb * B^2)
    return nothing
end

# ── B_cc from the faces (active cells of every live block) ────────────────────
@kernel function _ct_bcc_k!(Bx, By, @Const(FBx), @Const(FBy), @Const(live_d),
                            B::Int32, ng::Int32, nd::Int32, stride::Int32)
    t = @index(Global)
    t0 = Int32(t) - Int32(1)
    B2 = B * B
    bi = t0 ÷ B2; c = t0 % B2
    @inbounds begin
        slot = live_d[bi + Int32(1)]
        idx = (slot - Int32(1)) * stride + _li2(c % B + ng, c ÷ B + ng, nd)
        h = eltype(Bx)(0.5)
        Bx[idx] = h * (FBx[idx] + FBx[idx + 1])
        By[idx] = h * (FBy[idx] + FBy[idx + nd])
    end
end

function ct_bcc!(lev::Level)
    nb = length(lev.live); nb == 0 && return nothing
    _ct_bcc_k!(lev.be)(lev.Bx, lev.By, lev.FBx, lev.FBy, lev.live_d, Int32(lev.B),
                       Int32(lev.ng), Int32(lev.nd), Int32(lev.stride); ndrange = nb * lev.B^2)
    return nothing
end

# ── C/F ghost-face prolongation (NestedMHD prolong_faces), :pro job table ─────
# Fine ghost cell (di, dj) ← parent cell (si, sj) with parity (px, py).  The fine
# face is the LOWER `ax`-face of that cell: on a coarse face (even parity along
# ax) → that coarse face, else the mid-plane → mean of the cell's two coarse
# faces; each + MonCen transverse slope at the fine sub-face offset ±¼.
@kernel function _ct_prolong_face_k!(Ff, @Const(FR), @Const(FO), θ::T, ax::Int32,
                                     @Const(jobs), @Const(cellstart), njobs::Int32,
                                     B::Int32, ng::Int32, nd::Int32, stride::Int32) where {T}
    t = @index(Global)
    t0 = Int32(t) - Int32(1)
    b, ri, rj, rk = _job_decode(jobs, cellstart, njobs, t0)
    @inbounds begin
        di = jobs[b+3] + ri; dj = jobs[b+4] + rj
        # owned faces (the block's own lo..hi faces) are evolved, never prolonged
        dn = ax == Int32(1) ? di : dj; dt = ax == Int32(1) ? dj : di
        owned = (ng <= dn <= ng + B) & (ng <= dt < ng + B)
        if !owned
            si = jobs[b+6] + ((ri + jobs[b+12]) >> 0x01)
            sj = jobs[b+7] + ((rj + jobs[b+13]) >> 0x01)
            pn = ((ax == Int32(1) ? ri + jobs[b+12] : rj + jobs[b+13]) & Int32(1))   # 0: on a coarse face
            pt = ((ax == Int32(1) ? rj + jobs[b+13] : ri + jobs[b+12]) & Int32(1))
            o  = pt == Int32(0) ? T(-0.25) : T(0.25)
            sbase = (jobs[b+2] - Int32(1)) * stride
            un = ax == Int32(1) ? Int32(1) : nd                 # +1 along the normal
            ut = ax == Int32(1) ? nd : Int32(1)                 # +1 transverse
            q0 = sbase + _li2(si, sj, nd)
            @inline S(q) = (one(T) - θ) * T(FR[q]) + θ * T(FO[q])
            @inline face(q) = (v0 = S(q); v0 + o * moncen(v0 - S(q - ut), S(q + ut) - v0))
            v = pn == Int32(0) ? face(q0) : T(0.5) * (face(q0) + face(q0 + un))
            Ff[(jobs[b+1] - Int32(1)) * stride + _li2(di, dj, nd)] = v
        end
    end
end

function ct_fill_face_ghosts!(hier::AMRHierarchy, l::Int; θ::Real = 0, buf::Symbol = :R)
    lev = hier.levels[l + 1]
    dst = buf === :R ? cfields(lev) : cfields_o(lev)
    T = eltype(lev.D)
    if l >= 1
        plev = hier.levels[l]; pro = _tabt(lev, :pro)
        if pro.total > 0
            for (ax, fd, fR, fO) in zip((1, 2), dst, cfields(plev), cfields_o(plev))
                _ct_prolong_face_k!(lev.be)(fd, fR, fO, T(θ), Int32(ax), pro.jobs, pro.cellstart,
                                            Int32(pro.njobs), Int32(lev.B), Int32(lev.ng),
                                            Int32(lev.nd), Int32(lev.stride); ndrange = pro.total)
            end
        end
    end
    ct_face_exchange!(lev, dst)
    return nothing
end

"Authoritative-face sibling exchange of a level's face pools (tables.jl)."
function ct_face_exchange!(lev::Level, dst = cfields(lev))
    _run_copy!(lev.be, _tabt(lev, :sibfx), dst[1], dst[1], lev.nd, lev.stride)
    _run_copy!(lev.be, _tabt(lev, :sibfy), dst[2], dst[2], lev.nd, lev.stride)
    return nothing
end

# ── domain BCs for the face pools (boundary.jl :bc tables; after the cells) ───
# Normal faces (pool normal = BC axis): the wall face is owned; ghosts mirror
# about it (:reflecting negates).  Tangential faces follow the cell rule.
@kernel function _ct_bc_face_k!(F, FT, @Const(val), @Const(jobs), @Const(cellstart), njobs::Int32,
                                nd::Int32, stride::Int32, ng::Int32, B::Int32, n::Int32)
    t = @index(Global)
    t0 = Int32(t) - Int32(1)
    b, ri, rj, rk = _job_decode(jobs, cellstart, njobs, t0)
    @inbounds begin
        slot = jobs[b+1]
        di = jobs[b+3] + ri; dj = jobs[b+4] + rj
        a = jobs[b+12]; side = jobs[b+13]; kind = jobs[b+14]; fid = jobs[b+15]
        g = a == Int32(1) ? di : dj
        base = (slot - Int32(1)) * stride
        wall = side == Int32(1) && g == ng + B && n == a      # the hi wall faces
        if wall
            # owned in the active rows; in a transverse GHOST row with no same-level
            # block supplying it (a C/F corner on the wall) the face belongs to a cell
            # outside the domain and nothing else sets it: close its ghost cell
            # (ng+B−1, row) solenoidally from that cell's other three faces
            tt = a == Int32(1) ? dj : di
            fl = jobs[b+16]
            need = (tt < ng && (fl & Int32(1)) == Int32(0)) ||
                   (tt >= ng + B && (fl & Int32(2)) == Int32(0) && tt + Int32(1) < nd)
            if need
                ci = a == Int32(1) ? di - Int32(1) : di; cj = a == Int32(1) ? dj : dj - Int32(1)
                q = base + _li2(ci, cj, nd)
                ut = a == Int32(1) ? nd : Int32(1)        # transverse +1 (FT pool)
                F[base + _li2(di, dj, nd)] = F[q] - (FT[q + ut] - FT[q])
            end
        else
            sg = if n == a
                kind == BC_REFLECTING ? (side == Int32(0) ? Int32(2) * ng - g : Int32(2) * (ng + B) - g) :
                                        (side == Int32(0) ? ng : ng + B)
            else
                kind == BC_REFLECTING ? (side == Int32(0) ? Int32(2) * ng - Int32(1) - g :
                                                            Int32(2) * (ng + B) - Int32(1) - g) :
                                        (side == Int32(0) ? ng : ng + B - Int32(1))
            end
            si = a == Int32(1) ? sg : di; sj = a == Int32(2) ? sg : dj
            didx = base + _li2(di, dj, nd)
            if kind == BC_INFLOW
                F[didx] = val[fid]
            else
                v = F[base + _li2(si, sj, nd)]
                F[didx] = (kind == BC_REFLECTING && n == a) ? -v : v
            end
        end
    end
end

function ct_fill_face_boundary!(hier::AMRHierarchy, l::Int; buf::Symbol = :R)
    lev = hier.levels[l + 1]
    all(lev.per) && return nothing
    dst = buf === :R ? cfields(lev) : cfields_o(lev)
    vals = lev.tabs[:bcval]
    for a in 1:lev.dim
        tab = get(lev.tabs, Symbol(:bc, a), nothing)
        (tab === nothing || tab.total == 0) && continue
        for (n, F, FT, v) in ((1, dst[1], dst[2], vals[7]), (2, dst[2], dst[1], vals[8]))
            _ct_bc_face_k!(lev.be)(F, FT, v, tab.jobs, tab.cellstart, Int32(tab.njobs), Int32(lev.nd),
                                   Int32(lev.stride), Int32(lev.ng), Int32(lev.B), Int32(n);
                                   ndrange = tab.total)
        end
    end
    return nothing
end

# ── C/F tables (host): face restriction, E_z corners, corrected faces ─────────
# global parent cell (X, Y) → (slot, local i, j) of its owner, or nothing
function _cell_owner(plev::Level, X::Int128, Y::Int128)
    for (d, v) in ((1, X), (2, Y))
        plev.per[d] || (0 <= v < Int128(plev.P[d])) || return nothing
    end
    q = (wrapc(X, plev.P[1]), wrapc(Y, plev.P[2]), UInt128(0))
    s = lattice_owner(plev, q, plev.B)
    s == 0 && return nothing
    o = plev.meta[s].origin
    return (s, Int(q[1] - o[1]) + plev.ng, Int(q[2] - o[2]) + plev.ng)
end

# owner of the `ax`-face at global (X, Y) (lower face of cell (X, Y)); at a
# non-periodic hi domain edge the left/lower block's hi face
function _face_owner(plev::Level, X::Int128, Y::Int128, ax::Int)
    r = _cell_owner(plev, X, Y)
    r !== nothing && return r
    r = ax == 1 ? _cell_owner(plev, X - 1, Y) : _cell_owner(plev, X, Y - 1)
    r === nothing && return nothing
    return ax == 1 ? (r[1], r[2] + 1, r[3]) : (r[1], r[2], r[3] + 1)
end

# a block whose corner range [ng, ng+B]² holds global corner (X, Y)
function _corner_owner(lev::Level, X::Int128, Y::Int128)
    for (dx, dy) in ((0, 0), (-1, 0), (0, -1), (-1, -1))
        r = _cell_owner(lev, X + dx, Y + dy)
        r === nothing || return (r[1], r[2] - dx, r[3] - dy)
    end
    return nothing
end

function build_ct_tables!(hier::AMRHierarchy, l::Int)
    lev = hier.levels[l + 1]; plev = hier.levels[l]
    B = lev.B; Bh = B ÷ 2; ng = lev.ng
    covered(X, Y) = begin                       # parent cell under a level-l block?
        for (d, v) in ((1, X), (2, Y))
            plev.per[d] || (0 <= v < Int128(plev.P[d])) || return true   # beyond the domain
        end
        q = (wrapc(2X, lev.P[1]), wrapc(2Y, lev.P[2]), UInt128(0))
        lattice_owner(lev, q, B) != 0
    end
    res = Int32[]; corners = Dict{NTuple{2,Int128},Int}(); cent = Int32[]
    # corner / face keys wrap ONLY on periodic axes (a hi-wall corner x = P must not
    # fold onto the lo-wall corner x = 0)
    wk(v, d) = plev.per[d] ? Int128(mod(v, Int128(plev.P[d]))) : Int128(v)
    for s in lev.live
        m = lev.meta[s]
        X0 = Int128(m.origin[1]) >> 1; Y0 = Int128(m.origin[2]) >> 1
        # restriction: coarse x-faces X0..X0+Bh × Y0..Y0+Bh−1 (y transposed)
        for (ax, na, nb) in ((1, Bh, Bh - 1), (2, Bh - 1, Bh)), b in 0:nb, a in 0:na
            X = X0 + a; Y = Y0 + b
            o = _face_owner(plev, X, Y, ax)
            o === nothing && continue
            append!(res, Int32[o[1], o[2], o[3], ax, s, ng + 2a, ng + 2b, 0])
        end
        # C/F corners on the footprint boundary
        for b in 0:Bh, a in 0:Bh
            (a in (0, Bh) || b in (0, Bh)) || continue
            X = X0 + a; Y = Y0 + b
            key = (wk(X, 1), wk(Y, 2))
            haskey(corners, key) && continue
            all(covered(X + dx, Y + dy) for dx in (-1, 0), dy in (-1, 0)) && continue
            co = _corner_owner(plev, X, Y); fo = _corner_owner(lev, 2X, 2Y)
            (co === nothing || fo === nothing) && continue
            corners[key] = length(corners) + 1
            append!(cent, Int32[co[1], co[2], co[3], fo[1], fo[2], fo[3], 0, 0])
        end
    end
    # corrected faces: uncovered on both sides, touching a C/F corner
    reg(X, Y) = get(corners, (wk(X, 1), wk(Y, 2)), 0)
    fent = Int32[]; seen = Set{NTuple{3,Int128}}()
    for (X, Y) in keys(corners)
        for (ax, FX, FY) in ((1, X, Y - 1), (1, X, Y), (2, X - 1, Y), (2, X, Y))
            k = (wk(FX, 1), wk(FY, 2), Int128(ax))
            k in seen && continue
            push!(seen, k)
            # the two cells sharing the face
            c1 = ax == 1 ? (FX - 1, FY) : (FX, FY - 1)
            inb(c) = all(plev.per[d] || (0 <= c[d] < Int128(plev.P[d])) for d in 1:2)
            (inb(c1) && covered(c1...)) && continue
            (inb((FX, FY)) && covered(FX, FY)) && continue
            o = _face_owner(plev, FX, FY, ax)
            o === nothing && continue
            lo = reg(FX, FY); hi = ax == 1 ? reg(FX, FY + 1) : reg(FX + 1, FY)
            (lo == 0 && hi == 0) && continue
            append!(fent, Int32[o[1], o[2], o[3], ax, lo, hi, 0, 0])
        end
    end
    T = eltype(lev.D); be = lev.be
    lev.tabs[:ctres]  = (length(res) ÷ 8, to_device(be, res, Int32))
    lev.tabs[:ctcorn] = (length(cent) ÷ 8, to_device(be, cent, Int32))
    lev.tabs[:ctface] = (length(fent) ÷ 8, to_device(be, fent, Int32))
    lev.tabs[:ctreg]  = device_zeros(be, T, (max(length(corners), 1),))
    return nothing
end

# ── C/F kernels ───────────────────────────────────────────────────────────────
@kernel function _ct_restrict_face_k!(FXc, FYc, @Const(FXf), @Const(FYf), @Const(ent),
                                      ndc::Int32, ndf::Int32, stc::Int32, stf::Int32)
    e = @index(Global)
    b = (Int32(e) - Int32(1)) * Int32(8)
    @inbounds begin
        dc = (ent[b+1] - Int32(1)) * stc + _li2(ent[b+2], ent[b+3], ndc)
        fb = (ent[b+5] - Int32(1)) * stf; fi = ent[b+6]; fj = ent[b+7]
        h = eltype(FXc)(0.5)
        if ent[b+4] == Int32(1)
            FXc[dc] = h * (FXf[fb + _li2(fi, fj, ndf)] + FXf[fb + _li2(fi, fj + Int32(1), ndf)])
        else
            FYc[dc] = h * (FYf[fb + _li2(fi, fj, ndf)] + FYf[fb + _li2(fi + Int32(1), fj, ndf)])
        end
    end
end

# reg[c] += w · E_z(at the fine / coarse corner of entry c)   (side: 6 fine, 1 coarse)
@kernel function _ct_cap_emf_k!(reg, @Const(ent), @Const(rec), w::T, off::Int32,
                                nd::Int32, stride::Int32, nt::Int) where {T}
    e = @index(Global)
    b = (Int32(e) - Int32(1)) * Int32(8) + off
    @inbounds begin
        g = Int((ent[b+1] - Int32(1)) * stride + _li2(ent[b+2], ent[b+3], nd))
        reg[e] += w * _rl(rec, _CL_EZ, g, nt)
    end
end

# uncovered coarse faces at C/F corners: B ← B − λ_c·curl(ΔE),  ΔE = reg (0 off-corner)
@kernel function _ct_emf_apply_k!(FX, FY, @Const(ent), @Const(reg), λc::T,
                                  nd::Int32, stride::Int32) where {T}
    e = @index(Global)
    b = (Int32(e) - Int32(1)) * Int32(8)
    @inbounds begin
        idx = (ent[b+1] - Int32(1)) * stride + _li2(ent[b+2], ent[b+3], nd)
        lo = ent[b+5]; hi = ent[b+6]
        Δ = (hi > Int32(0) ? reg[hi] : zero(T)) - (lo > Int32(0) ? reg[lo] : zero(T))
        if ent[b+4] == Int32(1)
            FX[idx] -= λc * Δ          # Bx −= λ (E(top) − E(bottom))
        else
            FY[idx] += λc * Δ          # By += λ (E(right) − E(left))
        end
    end
end

# C/F cell flux captures from the face-flux scratch (lanes ρ, ρv, E, Bz; 6,7,9 = 0)
@kernel function _ct_capture_coarse_k!(reg, @Const(ent), @Const(rec), wc::T, de::Bool,
                                       nd::Int32, stride::Int32, nt::Int) where {T}
    e = @index(Global)
    b = (Int32(e) - Int32(1)) * Int32(16)
    @inbounds begin
        ci = ent[b+2]; cj = ent[b+3]; ax = ent[b+5]; side = ent[b+6]
        # side=+1: the coarse cell's LOW face (its own lower face); −1: its HIGH face
        hi = side == Int32(1) ? Int32(0) : Int32(1)
        i = ci + (ax == Int32(1) ? hi : Int32(0)); j = cj + (ax == Int32(2) ? hi : Int32(0))
        g = Int((ent[b+1] - Int32(1)) * stride + _li2(i, j, nd))
        lane0 = ax == Int32(1) ? _CL_FX : _CL_FY
        lg = ax == Int32(1) ? _CL_GX : _CL_GY           # dual energy: entropy flux in lane 9
        r = (Int(e) - 1) * 9
        Base.Cartesian.@nexprs 9 v -> begin
            f = (v == 6 || v == 7) ? zero(T) :
                v == 9 ? (de ? _rl(rec, lg, g, nt) : zero(T)) : _rl(rec, lane0 + v, g, nt)
            reg[r + v] -= wc * ifelse(isfinite(f), f, zero(T))
        end
    end
end

@kernel function _ct_capture_fine_k!(reg, @Const(ent), @Const(rec), wf::T, de::Bool,
                                     nd::Int32, stride::Int32, nt::Int) where {T}
    e = @index(Global)
    b = (Int32(e) - Int32(1)) * Int32(16)
    @inbounds begin
        fi = ent[b+8]; fj = ent[b+9]; ax = ent[b+5]; side = ent[b+6]
        hi = side == Int32(1) ? Int32(1) : Int32(0)    # inside cell's hi face for side = +1
        lane0 = ax == Int32(1) ? _CL_FX : _CL_FY
        base = (ent[b+7] - Int32(1)) * stride
        i0 = fi + (ax == Int32(1) ? hi : Int32(0)); j0 = fj + (ax == Int32(2) ? hi : Int32(0))
        g1 = Int(base + _li2(i0, j0, nd))
        g2 = Int(base + _li2(i0 + (ax == Int32(1) ? Int32(0) : Int32(1)),
                             j0 + (ax == Int32(1) ? Int32(1) : Int32(0)), nd))
        lg = ax == Int32(1) ? _CL_GX : _CL_GY           # dual energy: entropy flux in lane 9
        q = T(0.5) * wf
        r = (Int(e) - 1) * 9
        Base.Cartesian.@nexprs 9 v -> begin
            if v == 9
                if de
                    f = _rl(rec, lg, g1, nt) + _rl(rec, lg, g2, nt)
                    reg[r + v] += q * ifelse(isfinite(f), f, zero(T))
                end
            elseif !(v == 6 || v == 7)
                f = _rl(rec, lane0 + v, g1, nt) + _rl(rec, lane0 + v, g2, nt)
                reg[r + v] += q * ifelse(isfinite(f), f, zero(T))
            end
        end
    end
end

function ct_capture_fine!(hier::AMRHierarchy, l::Int, w::Real)
    lev = hier.levels[l + 1]
    T = eltype(lev.D); rec = _ctrec(lev); nt = lev.cap * lev.stride
    n = get(lev.tabs, :cfn, 0)::Int
    n > 0 && _ct_capture_fine_k!(lev.be)(lev.tabs[:cfreg], _tabi(lev, :cf), rec, T(w), hier.mhd.de,
                                         Int32(lev.nd), Int32(lev.stride), nt; ndrange = n)
    nc, cent = lev.tabs[:ctcorn]
    nc > 0 && _ct_cap_emf_k!(lev.be)(lev.tabs[:ctreg], cent, rec, T(w), Int32(3),
                                     Int32(lev.nd), Int32(lev.stride), nt; ndrange = nc)
    return nothing
end

function ct_capture_coarse!(hier::AMRHierarchy, l::Int, w::Real)
    lev = hier.levels[l + 1]; plev = hier.levels[l]
    T = eltype(lev.D); rec = _ctrec(plev); nt = plev.cap * plev.stride
    n = get(lev.tabs, :cfn, 0)::Int
    n > 0 && _ct_capture_coarse_k!(lev.be)(lev.tabs[:cfreg], _tabi(lev, :cf), rec, T(w), hier.mhd.de,
                                           Int32(plev.nd), Int32(plev.stride), nt; ndrange = n)
    nc, cent = lev.tabs[:ctcorn]
    nc > 0 && _ct_cap_emf_k!(lev.be)(lev.tabs[:ctreg], cent, rec, -T(w), Int32(0),
                                     Int32(plev.nd), Int32(plev.stride), nt; ndrange = nc)
    return nothing
end

"""
    ct_sync!(hier, l, λc)

After level l's children ran and the cell reflux/restriction of level `l` (the
fine level) into l−1 are done: E_z corrections of the uncovered C/F faces, face
restriction of the footprint, sibling re-sync of the coarse faces, B_cc.
"""
function ct_sync!(hier::AMRHierarchy, l::Int, λc::Real)
    lev = hier.levels[l + 1]; plev = hier.levels[l]
    T = eltype(lev.D)
    nf, fent = lev.tabs[:ctface]
    nf > 0 && _ct_emf_apply_k!(lev.be)(plev.FBx, plev.FBy, fent, lev.tabs[:ctreg], T(λc),
                                       Int32(plev.nd), Int32(plev.stride); ndrange = nf)
    fill!(lev.tabs[:ctreg], zero(T))
    nr, rent = lev.tabs[:ctres]
    nr > 0 && _ct_restrict_face_k!(lev.be)(plev.FBx, plev.FBy, lev.FBx, lev.FBy, rent,
                                           Int32(plev.nd), Int32(lev.nd), Int32(plev.stride),
                                           Int32(lev.stride); ndrange = nr)
    ct_face_exchange!(plev)
    ct_bcc!(plev)
    return nothing
end

# ── divergence-free initialisation of NEW blocks (regrid) ─────────────────────
# Pass A — faces lying on coarse faces (even fine index along the normal; the
# block's lo/hi boundary faces included): coarse face + MonCen transverse slope
# at ±¼ (sub-face sums preserved).  Slope taps stay inside the parent block's
# owned faces (no dependence on possibly stale parent ghosts at regrid time).
@kernel function _ct_new_even_k!(FX, FY, @Const(PFX), @Const(PFY), @Const(news),
                                 @Const(pinfo), B::Int32, ng::Int32, nd::Int32, stride::Int32,
                                 pnd::Int32, pstride::Int32)
    t = @index(Global)
    t0 = Int32(t) - Int32(1)
    nper = (B + Int32(1)) * B                  # faces per orientation per block
    bi = t0 ÷ (Int32(2) * nper); r = t0 % (Int32(2) * nper)
    ax = r < nper ? Int32(1) : Int32(2); r = r % nper
    @inbounds begin
        s = news[bi + Int32(1)]
        ps = pinfo[3bi + Int32(1)]; ox = pinfo[3bi + Int32(2)]; oy = pinfo[3bi + Int32(3)]
        a = r % (B + Int32(1)); b = r ÷ (B + Int32(1))        # a: normal face, b: transverse row
        if (a & Int32(1)) == Int32(0)
            T = eltype(FX)
            pa = a >> 0x01; pb = b >> 0x01
            o = (b & Int32(1)) == Int32(0) ? T(-0.25) : T(0.25)
            # parent stored indices (normal, transverse) → (i, j)
            pn = ng + (ax == Int32(1) ? ox : oy) + pa
            pt = ng + (ax == Int32(1) ? oy : ox) + pb
            PF = ax == Int32(1) ? PFX : PFY
            pbase = (ps - Int32(1)) * pstride
            @inline pidx(nn, tt) = pbase + (ax == Int32(1) ? _li2(nn, tt, pnd) : _li2(tt, nn, pnd))
            v0 = T(PF[pidx(pn, pt)])
            sl = (pt > ng && pt < ng + B - Int32(1)) ?
                 moncen(v0 - T(PF[pidx(pn, pt - Int32(1))]), T(PF[pidx(pn, pt + Int32(1))]) - v0) : zero(T)
            fn = ng + a; ft = ng + b
            didx = (s - Int32(1)) * stride + (ax == Int32(1) ? _li2(fn, ft, nd) : _li2(ft, fn, nd))
            (ax == Int32(1) ? FX : FY)[didx] = v0 + o * sl
        end
    end
end

# Pass B — boundary faces shared with an EXISTING fine block: copy its value.
# job: (dst slot, src slot, axis, side) — side 0: dst lo face ← src hi face
@kernel function _ct_new_share_k!(FX, FY, @Const(jobs), B::Int32, ng::Int32, nd::Int32,
                                  stride::Int32)
    t = @index(Global)
    t0 = Int32(t) - Int32(1)
    e = t0 ÷ B; r = t0 % B
    @inbounds begin
        d = jobs[4e + Int32(1)]; sr = jobs[4e + Int32(2)]
        ax = jobs[4e + Int32(3)]; side = jobs[4e + Int32(4)]
        nd_ = side == Int32(0) ? ng : ng + B; ns = side == Int32(0) ? ng + B : ng
        tt = ng + r
        di = ax == Int32(1) ? _li2(nd_, tt, nd) : _li2(tt, nd_, nd)
        si = ax == Int32(1) ? _li2(ns, tt, nd) : _li2(tt, ns, nd)
        F = ax == Int32(1) ? FX : FY
        F[(d - Int32(1)) * stride + di] = F[(sr - Int32(1)) * stride + si]
    end
end

# Pass C — the four mid-plane faces of every coarse cell (Tóth–Roe-type): with
# the 8 boundary sub-faces fixed, the 4 fine cells are solenoidal and the mid
# x-faces split their (fixed) sum by the mean y-variation of the boundary faces.
@kernel function _ct_new_mid_k!(FX, FY, @Const(news), Bh::Int32, ng::Int32, nd::Int32,
                                stride::Int32)
    t = @index(Global)
    t0 = Int32(t) - Int32(1)
    nc = Bh * Bh
    bi = t0 ÷ nc; c = t0 % nc
    @inbounds begin
        T = eltype(FX)
        s = news[bi + Int32(1)]; base = (s - Int32(1)) * stride
        i0 = ng + Int32(2) * (c % Bh); j0 = ng + Int32(2) * (c ÷ Bh)
        @inline X(i, j) = base + _li2(i, j, nd)
        bxL0 = T(FX[X(i0, j0)]);          bxL1 = T(FX[X(i0, j0 + Int32(1))])
        bxR0 = T(FX[X(i0 + Int32(2), j0)]); bxR1 = T(FX[X(i0 + Int32(2), j0 + Int32(1))])
        byB0 = T(FY[X(i0, j0)]);          byB1 = T(FY[X(i0 + Int32(1), j0)])
        byT0 = T(FY[X(i0, j0 + Int32(2))]); byT1 = T(FY[X(i0 + Int32(1), j0 + Int32(2))])
        h = T(0.5)
        S = bxL0 + bxL1 + byB0 - byT0
        Dd = h * ((bxL0 - bxL1) + (bxR0 - bxR1))
        bxM0 = h * (S + Dd); bxM1 = h * (S - Dd)
        FX[X(i0 + Int32(1), j0)] = bxM0
        FX[X(i0 + Int32(1), j0 + Int32(1))] = bxM1
        FY[X(i0, j0 + Int32(1))] = byB0 + bxL0 - bxM0
        FY[X(i0 + Int32(1), j0 + Int32(1))] = byB1 - bxR0 + bxM0
    end
end

# B_cc of the new blocks from their faces; E corrected for the B_cc change
@kernel function _ct_new_bcc_k!(E, Bx, By, @Const(FX), @Const(FY), @Const(Esc), @Const(news),
                                B::Int32, ng::Int32, nd::Int32, stride::Int32)
    t = @index(Global)
    t0 = Int32(t) - Int32(1)
    B2 = B * B
    bi = t0 ÷ B2; c = t0 % B2
    @inbounds begin
        T = eltype(Bx)
        s = news[bi + Int32(1)]
        idx = (s - Int32(1)) * stride + _li2(c % B + ng, c ÷ B + ng, nd)
        bx = T(0.5) * (FX[idx] + FX[idx + 1]); by = T(0.5) * (FY[idx] + FY[idx + nd])
        dEm = T(0.5) * ((bx*bx + by*by) - (Bx[idx]^2 + By[idx]^2))
        E[idx] = _narrow(eltype(E), (T(E[idx]) * T(Esc[s]) + dEm) / T(Esc[s]))
        Bx[idx] = bx; By[idx] = by
    end
end

"Divergence-free face fill of freshly created level-`lc` blocks (after their cell prolongation)."
function ct_init_new!(hier::AMRHierarchy, lc::Int, news::Vector{Int32})
    lev = hier.levels[lc + 1]; plev = hier.levels[lc]
    isempty(news) && return nothing
    B = lev.B; ng = lev.ng
    pinfo = Int32[]
    for s in news
        m = lev.meta[s]
        append!(pinfo, Int32[m.parent, m.offset[1], m.offset[2]])
    end
    nd_ = to_device(lev.be, news, Int32)
    a = (Int32(B), Int32(ng), Int32(lev.nd), Int32(lev.stride))
    _ct_new_even_k!(lev.be)(lev.FBx, lev.FBy, plev.FBx, plev.FBy, nd_, to_device(lev.be, pinfo, Int32),
                            a..., Int32(plev.nd), Int32(plev.stride);
                            ndrange = length(news) * 2 * (B + 1) * B)
    # existing (old) fine neighbours share boundary faces
    newset = Set(news); jobs = Int32[]
    for s in news, ax in 1:2, side in (0, 1)
        o = lev.meta[s].origin
        no = ntuple(d -> d == ax ? Int128(o[d]) + (side == 0 ? -B : B) : Int128(o[d]), 3)
        (!lev.per[ax] && !(0 <= no[ax] < Int128(lev.P[ax]))) && continue
        key = ntuple(d -> wrapc(no[d], lev.P[d]), 3)
        n = get(lev.byorigin, key, Int32(0))
        (n == 0 || n in newset) && continue
        append!(jobs, Int32[s, n, ax, side])
    end
    if !isempty(jobs)
        _ct_new_share_k!(lev.be)(lev.FBx, lev.FBy, to_device(lev.be, jobs, Int32), a...;
                                 ndrange = (length(jobs) ÷ 4) * B)
    end
    _ct_new_mid_k!(lev.be)(lev.FBx, lev.FBy, nd_, Int32(B ÷ 2), Int32(ng), Int32(lev.nd),
                           Int32(lev.stride); ndrange = length(news) * (B ÷ 2)^2)
    _ct_new_bcc_k!(lev.be)(lev.Tau, lev.Bx, lev.By, lev.FBx, lev.FBy, lev.Esc, nd_, a...;
                           ndrange = length(news) * B^2)
    return nothing
end

# ── host helpers ──────────────────────────────────────────────────────────────
"""
    set_ct_faces_from_Az!(hier, l, Az)

Set level-`l` face fields from a vector potential `Az(x, y)` sampled at cell
corners (Bx = ∂_y A_z, By = −∂_x A_z, discretely): ∇·B = 0 to round-off on every
level, and coarse faces equal restricted fine faces exactly.  Also refreshes
B_cc (and leaves the gas pools to the caller — E must include ½|B_cc|²).
"""
function set_ct_faces_from_Az!(hier::AMRHierarchy, l::Int, Az)
    lev = hier.levels[l + 1]; dx = level_dx(hier, l); T = eltype(lev.D)
    hx = zeros(T, lev.cap * lev.stride); hy = zeros(T, lev.cap * lev.stride)
    nd = lev.nd; ng = lev.ng
    for s in lev.live
        m = lev.meta[s]; base = (Int(s) - 1) * lev.stride
        A(i, j) = Az((Float64(m.origin[1]) + i - ng) * dx, (Float64(m.origin[2]) + j - ng) * dx)
        for j in 0:nd-1, i in 0:nd-1
            j + 1 <= nd && (hx[base + j*nd + i + 1] = (A(i, j + 1) - A(i, j)) / dx)
            hy[base + j*nd + i + 1] = -(A(i + 1, j) - A(i, j)) / dx
        end
    end
    copyto!(lev.FBx, hx); copyto!(lev.FBy, hy)
    ct_bcc!(lev)
    return nothing
end

"max over active cells of |∇·B|·dx / max|B| from the FACE field (CT)."
function max_divb_ct(hier::AMRHierarchy)
    worst = 0.0; bmax = 0.0
    for l in 0:length(hier.levels)-1
        lev = hier.levels[l + 1]
        isempty(lev.live) && continue
        fx = Array(lev.FBx); fy = Array(lev.FBy); nd = lev.nd; ng = lev.ng
        for s in lev.live
            base = (Int(s) - 1) * lev.stride
            for j in ng:ng+lev.B-1, i in ng:ng+lev.B-1
                q = base + j*nd + i + 1
                worst = max(worst, abs((fx[q + 1] - fx[q]) + (fy[q + nd] - fy[q])))
                bmax = max(bmax, abs(fx[q]), abs(fy[q]))
            end
        end
    end
    return worst / max(bmax, floatmin(Float64))
end
