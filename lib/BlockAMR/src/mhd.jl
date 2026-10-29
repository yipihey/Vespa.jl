# mhd.jl — ideal GLM-MHD on the block hierarchy (`physics = :mhd`).
#
# Scheme: the NestedMHD core (github.com/yipihey/NestedTurbulence, core/src/physics.jl,
# itself GLMMHDTurb + the Solaceonx sweep-fixes): PLM with the @fastmath branchless
# MonCen limiter on primitives, a 3-D (2-D: x,y) MUSCL-Hancock half step, one
# Riemann solve per face (HLLD / HLL / LLF, compile-time `Val`), and Dedner mixed
# GLM cleaning (ψ flux pair c_h² Bn*, ψ* ; exponential ψ damping per step).  The
# pointwise physics below is NestedMHD's, verbatim (submodule `MHDPhys`); only the
# data layout and the AMR coupling are BlockAMR's:
#
#   * state = the gas pools with Tau ≡ TOTAL energy (incl. ½B²) + the cell-centred
#     pools Bx, By, Bz, Psi.  Ge is kept = p/(γ−1) (diagnostic: CFL, regrid Jeans,
#     scales) — MHD carries no independent dual-energy channel;
#   * one level step = two launches: `_mhd_rec_k!` stores the half-step primitive +
#     the per-direction half slopes of every cell of the active box grown by one
#     (scratch `tabs[:mrec]`), `_mhd_update_k!` solves the 2·DIM faces of each active
#     cell from that scratch.  Both cells sharing a face solve it from IDENTICAL
#     inputs ⇒ interior faces cancel exactly; the C/F capture kernels read the same
#     scratch ⇒ the reflux registers reproduce the applied flux (9 lanes: ρ, ρv, E,
#     B, ψ — B and ψ are refluxed like any conserved quantity, which keeps ∫B exact
#     across C/F; ∇·B itself is cleaned, not constrained, in this GLM mode);
#   * the single-stage scheme rides the `:ctu` W-cycle path (one ghost fill, one
#     launch pair R→O, fine ½ / coarse 1 capture weights, swap);
#   * c_h is ONE global speed: c_h = glm_ch_scale · s_max / sigdims, s_max the
#     global unsplit signal Σ_d(|v_d|+c_f,d) behind λ — NestedMHD's
#     courant·dx/(3·dt_cfl)·scale with the level-uniform λ = dt/dx of the strict-2:1
#     W-cycle; damping ψ ← ψ·exp(−c_h·dt_l/(glm_cp_coef·box)) per level step.
# Gas, B and ψ are all cell-centred, so ghost prolongation (limited-linear),
# restriction, regrid, domain BCs (reflecting flips the normal B) and checkpoints
# reuse the hydro machinery unchanged.

module MHDPhys
# ── NestedMHD core/src/physics.jl (pointwise ideal MHD) ─────────────────────────
# Conserved order (9): (ρ, ρvx, ρvy, ρvz, E, Bx, By, Bz, ψ)
# Primitive order (9): (ρ, vx, vy, vz, p, Bx, By, Bz, ψ)

@fastmath @inline function cons2prim(c::NTuple{9,T}, gamma::T, smallr::T, pfl::T) where {T}
    rho = max(c[1], smallr)
    ir  = one(T) / rho
    vx = c[2]*ir; vy = c[3]*ir; vz = c[4]*ir
    bx = c[6]; by = c[7]; bz = c[8]
    ekin = T(0.5)*(c[2]*c[2] + c[3]*c[3] + c[4]*c[4])*ir
    emag = T(0.5)*(bx*bx + by*by + bz*bz)
    p = max((gamma - one(T))*(c[5] - ekin - emag), pfl)
    return (rho, vx, vy, vz, p, bx, by, bz, c[9])
end

@inline function prim2cons(q::NTuple{9,T}, gamma::T) where {T}
    rho = q[1]; vx = q[2]; vy = q[3]; vz = q[4]; p = q[5]
    bx = q[6]; by = q[7]; bz = q[8]
    ekin = T(0.5)*rho*(vx*vx + vy*vy + vz*vz)
    emag = T(0.5)*(bx*bx + by*by + bz*bz)
    E = p/(gamma - one(T)) + ekin + emag
    return (rho, rho*vx, rho*vy, rho*vz, E, bx, by, bz, q[9])
end

@fastmath @inline function fast_speed(q::NTuple{9,T}, gamma::T, bn::T) where {T}
    rho = q[1]
    c2 = gamma * q[5] / rho
    b2 = (q[6]*q[6] + q[7]*q[7] + q[8]*q[8]) / rho
    d2 = T(0.5)*(b2 + c2)
    return sqrt(d2 + sqrt(max(d2*d2 - c2*bn*bn/rho, zero(T))))
end

@inline function phys_flux_x(q::NTuple{9,T}, gamma::T) where {T}
    rho = q[1]; vx = q[2]; vy = q[3]; vz = q[4]; p = q[5]
    bx = q[6]; by = q[7]; bz = q[8]
    b2 = bx*bx + by*by + bz*bz
    ptot = p + T(0.5)*b2
    E = p/(gamma - one(T)) + T(0.5)*rho*(vx*vx + vy*vy + vz*vz) + T(0.5)*b2
    vdotb = vx*bx + vy*by + vz*bz
    return (rho*vx,
            rho*vx*vx + ptot - bx*bx,
            rho*vx*vy - bx*by,
            rho*vx*vz - bx*bz,
            (E + ptot)*vx - bx*vdotb,
            zero(T),
            vx*by - vy*bx,
            vx*bz - vz*bx,
            zero(T))
end

@inline function rot_to(q::NTuple{9}, dir::Int)
    dir == 1 && return q
    dir == 2 && return (q[1], q[3], q[4], q[2], q[5], q[7], q[8], q[6], q[9])
    return (q[1], q[4], q[2], q[3], q[5], q[8], q[6], q[7], q[9])
end
@inline function rot_flux_from(f::NTuple{9}, dir::Int)
    dir == 1 && return f
    dir == 2 && return (f[1], f[4], f[2], f[3], f[5], f[8], f[6], f[7], f[9])
    return (f[1], f[3], f[4], f[2], f[5], f[7], f[8], f[6], f[9])
end

@fastmath @inline function moncen(dl::T, dr::T) where {T}
    dc  = T(0.5)*(dl + dr)
    sgn = ifelse(dc >= zero(T), one(T), -one(T))
    val = sgn * min(T(2)*min(abs(dl), abs(dr)), abs(dc))
    return ifelse(dl*dr <= zero(T), zero(T), val)
end

@fastmath @inline prim_slope(L::NTuple{9,T}, M::NTuple{9,T}, R::NTuple{9,T}) where {T} =
    ntuple(i -> T(0.5)*moncen(M[i]-L[i], R[i]-M[i]), 9)

@inline padd(q::NTuple{9,T}, s::NTuple{9,T}, a::T) where {T} = ntuple(i -> q[i] + a*s[i], 9)

@inline dir_flux(q::NTuple{9,T}, dir::Int, gamma::T) where {T} =
    rot_flux_from(phys_flux_x(rot_to(q, dir), gamma), dir)

@fastmath @inline function hancock(m0::NTuple{9,T}, sx, sy, sz, dtdx::T, gamma::T) where {T}
    u0 = prim2cons(m0, gamma)
    h = T(0.5)*dtdx
    fxp = dir_flux(padd(m0, sx, one(T)), 1, gamma); fxm = dir_flux(padd(m0, sx, -one(T)), 1, gamma)
    fyp = dir_flux(padd(m0, sy, one(T)), 2, gamma); fym = dir_flux(padd(m0, sy, -one(T)), 2, gamma)
    fzp = dir_flux(padd(m0, sz, one(T)), 3, gamma); fzm = dir_flux(padd(m0, sz, -one(T)), 3, gamma)
    return ntuple(i -> u0[i] - h*((fxp[i]-fxm[i]) + (fyp[i]-fym[i]) + (fzp[i]-fzm[i])), 9)
end

# 2-D Hancock: the flat z has no flux pair (it cancels exactly in `hancock`)
@fastmath @inline function hancock2(m0::NTuple{9,T}, sx, sy, dtdx::T, gamma::T) where {T}
    u0 = prim2cons(m0, gamma)
    h = T(0.5)*dtdx
    fxp = dir_flux(padd(m0, sx, one(T)), 1, gamma); fxm = dir_flux(padd(m0, sx, -one(T)), 1, gamma)
    fyp = dir_flux(padd(m0, sy, one(T)), 2, gamma); fym = dir_flux(padd(m0, sy, -one(T)), 2, gamma)
    return ntuple(i -> u0[i] - h*((fxp[i]-fxm[i]) + (fyp[i]-fym[i])), 9)
end

@inline function glm_pair(bnL::T, bnR::T, psiL::T, psiR::T, ch::T) where {T}
    bns  = T(0.5)*(bnL + bnR) - T(0.5)*(psiR - psiL)/ch
    psis = T(0.5)*(psiL + psiR) - T(0.5)*ch*(bnR - bnL)
    return bns, psis
end

@fastmath @inline function hll_x(L::NTuple{9,T}, R::NTuple{9,T}, gamma::T, bn::T, fbn::T, fpsi::T) where {T}
    cfL = fast_speed(L, gamma, bn); cfR = fast_speed(R, gamma, bn)
    SL = min(min(L[2]-cfL, R[2]-cfR), zero(T)); SR = max(max(L[2]+cfL, R[2]+cfR), zero(T))
    fL = phys_flux_x(L, gamma); fR = phys_flux_x(R, gamma)
    uL = prim2cons(L, gamma);   uR = prim2cons(R, gamma)
    ihd = one(T)/(SR - SL)
    f = ntuple(i -> (SR*fL[i] - SL*fR[i] + SL*SR*(uR[i]-uL[i]))*ihd, 9)
    return (f[1], f[2], f[3], f[4], f[5], fbn, f[7], f[8], fpsi)
end

@fastmath @inline function llf_x(L::NTuple{9,T}, R::NTuple{9,T}, gamma::T, ch::T, fbn::T, fpsi::T) where {T}
    fL = phys_flux_x(L, gamma); fR = phys_flux_x(R, gamma)
    uL = prim2cons(L, gamma);   uR = prim2cons(R, gamma)
    smax = max(abs(L[2]) + fast_speed(L, gamma, L[6]), abs(R[2]) + fast_speed(R, gamma, R[6]))
    f = ntuple(i -> T(0.5)*(fL[i] + fR[i]) - T(0.5)*smax*(uR[i] - uL[i]), 9)
    return (f[1], f[2], f[3], f[4], f[5], fbn, f[7], f[8], fpsi)
end

# HLLD (Miyoshi & Kusano 2005), with the sweep-fixes degenerate guard |denK| < 1e-4·p*.
# (Only change vs NestedMHD: the inner `star` closure is @inline — see below.)
@fastmath @inline function hlld_x(L::NTuple{9,T}, R::NTuple{9,T}, gamma::T, bn::T, fbn::T, fpsi::T) where {T}
    dL,uL,vL,wL,pL = L[1],L[2],L[3],L[4],L[5]; byL,bzL = L[7],L[8]
    dR,uR,vR,wR,pR = R[1],R[2],R[3],R[4],R[5]; byR,bzR = R[7],R[8]
    h = T(0.5); g1 = gamma - one(T)
    b2L = bn*bn + byL*byL + bzL*bzL; b2R = bn*bn + byR*byR + bzR*bzR
    ptL = pL + h*b2L; ptR = pR + h*b2R
    EL = pL/g1 + h*dL*(uL*uL+vL*vL+wL*wL) + h*b2L
    ER = pR/g1 + h*dR*(uR*uR+vR*vR+wR*wR) + h*b2R
    cfL = fast_speed(L, gamma, bn); cfR = fast_speed(R, gamma, bn)
    SL = min(min(uL, uR) - max(cfL, cfR), zero(T))
    SR = max(max(uL, uR) + max(cfL, cfR), zero(T))
    UL = (dL, dL*uL, dL*vL, dL*wL, EL, bn, byL, bzL, zero(T))
    UR = (dR, dR*uR, dR*vR, dR*wR, ER, bn, byR, bzR, zero(T))
    FL = phys_flux_x((dL,uL,vL,wL,pL,bn,byL,bzL,zero(T)), gamma)
    FR = phys_flux_x((dR,uR,vR,wR,pR,bn,byR,bzR,zero(T)), gamma)
    den = (SR - uR)*dR - (SL - uL)*dL
    SM  = ((SR - uR)*dR*uR - (SL - uL)*dL*uL - ptR + ptL) / den
    pts = ((SR - uR)*dR*ptL - (SL - uL)*dL*ptR + dL*dR*(SR - uR)*(SL - uL)*(uR - uL)) / den
    dLs = max(dL*(SL - uL)/(SL - SM), T(1e-12)); dRs = max(dR*(SR - uR)/(SR - SM), T(1e-12))
    sqdLs = sqrt(dLs); sqdRs = sqrt(dRs)
    SLs = SM - abs(bn)/sqdLs; SRs = SM + abs(bn)/sqdRs
    @inline function star(d, u, v, w, by, bz, S)    # @inline: else a GPU call + stack tuple
        denK = d*(S-u)*(S-SM) - bn*bn
        if abs(denK) < T(1e-4)*pts
            return v, w, by, bz
        end
        fac = bn*(SM - u)/denK
        num = d*(S-u)*(S-u) - bn*bn
        return v - by*fac, w - bz*fac, by*num/denK, bz*num/denK
    end
    vyLs, vzLs, byLs, bzLs = star(dL,uL,vL,wL,byL,bzL,SL)
    vyRs, vzRs, byRs, bzRs = star(dR,uR,vR,wR,byR,bzR,SR)
    vdotbL = uL*bn + vL*byL + wL*bzL; vdotbLs = SM*bn + vyLs*byLs + vzLs*bzLs
    ELs = ((SL - uL)*EL - ptL*uL + pts*SM + bn*(vdotbL - vdotbLs))/(SL - SM)
    vdotbR = uR*bn + vR*byR + wR*bzR; vdotbRs = SM*bn + vyRs*byRs + vzRs*bzRs
    ERs = ((SR - uR)*ER - ptR*uR + pts*SM + bn*(vdotbR - vdotbRs))/(SR - SM)
    ULs = (dLs, dLs*SM, dLs*vyLs, dLs*vzLs, ELs, bn, byLs, bzLs, zero(T))
    URs = (dRs, dRs*SM, dRs*vyRs, dRs*vzRs, ERs, bn, byRs, bzRs, zero(T))
    sgn = ifelse(bn >= zero(T), one(T), -one(T))
    invsum = one(T)/(sqdLs + sqdRs)
    vyss = (sqdLs*vyLs + sqdRs*vyRs + (byRs - byLs)*sgn)*invsum
    vzss = (sqdLs*vzLs + sqdRs*vzRs + (bzRs - bzLs)*sgn)*invsum
    byss = (sqdLs*byRs + sqdRs*byLs + sqdLs*sqdRs*(vyRs - vyLs)*sgn)*invsum
    bzss = (sqdLs*bzRs + sqdRs*bzLs + sqdLs*sqdRs*(vzRs - vzLs)*sgn)*invsum
    vdotbss = SM*bn + vyss*byss + vzss*bzss
    ELss = ELs - sqdLs*(vdotbLs - vdotbss)*sgn
    ERss = ERs + sqdRs*(vdotbRs - vdotbss)*sgn
    ULss = (dLs, dLs*SM, dLs*vyss, dLs*vzss, ELss, bn, byss, bzss, zero(T))
    URss = (dRs, dRs*SM, dRs*vyss, dRs*vzss, ERss, bn, byss, bzss, zero(T))
    if SLs >= zero(T)
        f = ntuple(i -> FL[i] + SL*(ULs[i] - UL[i]), 9)
    elseif SM >= zero(T)
        f = ntuple(i -> FL[i] + SLs*ULss[i] - (SLs - SL)*ULs[i] - SL*UL[i], 9)
    elseif SRs >= zero(T)
        f = ntuple(i -> FR[i] + SRs*URss[i] - (SRs - SR)*URs[i] - SR*UR[i], 9)
    else
        f = ntuple(i -> FR[i] + SR*(URs[i] - UR[i]), 9)
    end
    return (f[1], f[2], f[3], f[4], f[5], fbn, f[7], f[8], fpsi)
end

@inline riemann_x(::Val{:hll},  L, R, gamma, bn, fbn, fpsi, ch) = hll_x(L, R, gamma, bn, fbn, fpsi)
@inline riemann_x(::Val{:hlld}, L, R, gamma, bn, fbn, fpsi, ch) = hlld_x(L, R, gamma, bn, fbn, fpsi)
@inline riemann_x(::Val{:llf},  L, R, gamma, bn, fbn, fpsi, ch) = llf_x(L, R, gamma, ch, fbn, fpsi)

# GLM Riemann in direction `dir`: Dedner pair on (Bn, ψ) → F[Bn] = ψ*, F[ψ] = c_h² Bn*.
@fastmath @inline function riemann_dir(rv::Val, Lq::NTuple{9,T}, Rq::NTuple{9,T}, dir::Int,
                                       gamma::T, ch::T, smallr::T, pfl::T) where {T}
    L0 = rot_to(Lq, dir); R0 = rot_to(Rq, dir)
    bns, psis = glm_pair(L0[6], R0[6], L0[9], R0[9], ch)
    fpsi = ch*ch*bns
    L = (max(L0[1],smallr), L0[2],L0[3],L0[4], max(L0[5],pfl), bns, L0[7],L0[8], L0[9])
    R = (max(R0[1],smallr), R0[2],R0[3],R0[4], max(R0[5],pfl), bns, R0[7],R0[8], R0[9])
    f = riemann_x(rv, L, R, gamma, bns, psis, fpsi, ch)
    return rot_flux_from(f, dir)
end

# unsplit CFL signal Σ_d (|v_d| + c_f,d) over the first `nd` directions
@inline function signal_speed(q::NTuple{9,T}, gamma::T, nd::Int) where {T}
    s = abs(q[2]) + fast_speed(q, gamma, q[6])
    s += abs(q[3]) + fast_speed(q, gamma, q[7])
    nd == 3 && (s += abs(q[4]) + fast_speed(q, gamma, q[8]))
    return s
end

@inline iflo(f::NTuple{9,T}, m::NTuple{9,T}, smallr::T, pfl::T) where {T} =
    (f[1] < smallr || f[5] < pfl) ? m : f
end # module MHDPhys

using .MHDPhys: cons2prim, prim2cons, prim_slope, hancock, hancock2, riemann_dir, signal_speed,
                padd, iflo

# ── pool access ───────────────────────────────────────────────────────────────
# conserved 9-tuple of one cell, physical units, compute type T
@inline function _mload(::Type{T}, D, S1, S2, S3, E, Bx, By, Bz, Ps, idx,
                        dsc, ssc, esc) where {T}
    @inbounds (T(D[idx]) * T(dsc), T(S1[idx]) * T(ssc), T(S2[idx]) * T(ssc),
               T(S3[idx]) * T(ssc), T(E[idx]) * T(esc),
               T(Bx[idx]), T(By[idx]), T(Bz[idx]), T(Ps[idx]))
end

# rec scratch: lane-major, `nt` = cap·stride elements per lane (Int indexing)
@inline _rr(rec, lane::Int, g::Int, nt::Int) = @inbounds rec[(lane - 1) * nt + g]
@inline _rget(rec, off::Int, g::Int, nt::Int) = ntuple(v -> _rr(rec, off + v, g, nt), 9)

"Scratch lanes: half-step primitive (9) + half slopes per active direction (9·DIM)."
_mlanes(dim::Int) = 9 + 9dim

function _mrec(lev::Level)
    T = eltype(lev.D)
    n = _mlanes(lev.dim) * lev.cap * lev.stride
    r = get(lev.tabs, :mrec, nothing)
    if !(r isa AbstractVector{T}) || length(r) != n
        r = device_zeros(lev.be, T, (n,))
        lev.tabs[:mrec] = r
    end
    return r
end

# ── pass 1: MUSCL-Hancock half-step primitive + half slopes ───────────────────
# Every cell of the active box grown by one (the faces of every active cell need
# the scratch on both sides); stencil ±1 more ⇒ reads stay within ng = 2 ghosts.
@kernel function _mhd_rec_k!(rec, @Const(D), @Const(S1), @Const(S2), @Const(S3), @Const(E),
                             @Const(Bx), @Const(By), @Const(Bz), @Const(Ps),
                             @Const(live_d), @Const(Dsc), @Const(Ssc), @Const(Esc),
                             gamma::T, smallr::T, pfl::T, dtdx::T,
                             B::Int32, ng::Int32, nd::Int32, stride::Int32, nt::Int,
                             dimv::Val{DIM}) where {T,DIM}
    t = @index(Global)
    t0 = Int32(t) - Int32(1)
    R = B + Int32(2)
    nr = DIM == 3 ? R * R * R : R * R
    bi = t0 ÷ nr; c = t0 % nr
    kz = _kzv(dimv)
    @inbounds begin
        slot = live_d[bi + Int32(1)]
        base = (slot - Int32(1)) * stride
        dsc = Dsc[slot]; ssc = Ssc[slot]; esc = Esc[slot]
        i = c % R + ng - Int32(1); j = (c ÷ R) % R + ng - Int32(1)
        k = DIM == 3 ? c ÷ (R * R) + ng - Int32(1) : Int32(0)
        @inline P(ii, jj, kk) = cons2prim(_mload(T, D, S1, S2, S3, E, Bx, By, Bz, Ps,
                                         base + _lidx(ii, jj, kk, nd), dsc, ssc, esc),
                                  gamma, smallr, pfl)
        m0 = P(i, j, k)
        sx = prim_slope(P(i - Int32(1), j, k), m0, P(i + Int32(1), j, k))
        sy = prim_slope(P(i, j - Int32(1), k), m0, P(i, j + Int32(1), k))
        if DIM == 3
            sz = prim_slope(P(i, j, k - kz), m0, P(i, j, k + kz))
            mh = cons2prim(hancock(m0, sx, sy, sz, dtdx, gamma), gamma, smallr, pfl)
        else                                    # flat z: no z slope, no z flux pair
            sz = sx
            mh = cons2prim(hancock2(m0, sx, sy, dtdx, gamma), gamma, smallr, pfl)
        end
        if mh[1] <= smallr || mh[5] <= pfl
            mh = m0
        end
        g = Int(base + _lidx(i, j, k, nd))
        # compile-time unrolled stores: a runtime-indexed tuple spills to local memory
        Base.Cartesian.@nexprs 9 v -> begin
            rec[(v - 1) * nt + g] = mh[v]
            rec[(8 + v) * nt + g] = sx[v]
            rec[(17 + v) * nt + g] = sy[v]
            DIM == 3 && (rec[(26 + v) * nt + g] = sz[v])
        end
    end
end

# the GLM flux through the face between scratch cells gL (lower) and gR along `d`
@inline function _mhd_face(rec, gL::Int, gR::Int, d::Int, nt::Int, rv::Val,
                           gamma::T, ch::T, smallr::T, pfl::T) where {T}
    off = 9 * d                                         # slope lanes of direction d
    ml = _rget(rec, 0, gL, nt); mr = _rget(rec, 0, gR, nt)
    L = iflo(padd(ml, _rget(rec, off, gL, nt), one(T)), ml, smallr, pfl)
    Rq = iflo(padd(mr, _rget(rec, off, gR, nt), -one(T)), mr, smallr, pfl)
    return riemann_dir(rv, L, Rq, d, gamma, ch, smallr, pfl)
end

# ── pass 2: conservative update of every active cell, R → O ───────────────────
@kernel function _mhd_update_k!(Do_, S1o_, S2o_, S3o_, Eo_, Geo_, Bxo_, Byo_, Bzo_, Pso_,
                                @Const(D), @Const(S1), @Const(S2), @Const(S3), @Const(E),
                                @Const(Bx), @Const(By), @Const(Bz), @Const(Ps),
                                @Const(rec), @Const(live_d),
                                @Const(Dsc), @Const(Ssc), @Const(Esc), @Const(Gsc),
                                gamma::T, ch::T, smallr::T, pfl::T, dtdx::T, glm_fac::T,
                                B::Int32, ng::Int32, nd::Int32, stride::Int32, nt::Int,
                                rv::Val, dimv::Val{DIM}) where {T,DIM}
    t = @index(Global)
    t0 = Int32(t) - Int32(1)
    B3 = _nblk(B, dimv)
    bi = t0 ÷ B3; c = t0 % B3
    @inbounds begin
        slot = live_d[bi + Int32(1)]
        base = (slot - Int32(1)) * stride
        dsc = Dsc[slot]; ssc = Ssc[slot]; esc = Esc[slot]; gsc = Gsc[slot]
        i, j, k = _cell_ijk(c, B, ng, dimv)
        idx = base + _lidx(i, j, k, nd)
        g = Int(idx)
        u0 = _mload(T, D, S1, S2, S3, E, Bx, By, Bz, Ps, idx, dsc, ssc, esc)
        # flux differences (F_lo − F_hi) per direction, NestedMHD's update order
        sx = Int(base + _lidx(i + Int32(1), j, k, nd)) - g
        fxl = _mhd_face(rec, g - sx, g, 1, nt, rv, gamma, ch, smallr, pfl)
        fxh = _mhd_face(rec, g, g + sx, 1, nt, rv, gamma, ch, smallr, pfl)
        sy = Int(base + _lidx(i, j + Int32(1), k, nd)) - g
        fyl = _mhd_face(rec, g - sy, g, 2, nt, rv, gamma, ch, smallr, pfl)
        fyh = _mhd_face(rec, g, g + sy, 2, nt, rv, gamma, ch, smallr, pfl)
        r = if DIM == 3
            sz = Int(base + _lidx(i, j, k + Int32(1), nd)) - g
            fzl = _mhd_face(rec, g - sz, g, 3, nt, rv, gamma, ch, smallr, pfl)
            fzh = _mhd_face(rec, g, g + sz, 3, nt, rv, gamma, ch, smallr, pfl)
            ntuple(v -> u0[v] + dtdx * ((fxl[v] - fxh[v]) + (fyl[v] - fyh[v]) +
                                        (fzl[v] - fzh[v])), 9)
        else
            ntuple(v -> u0[v] + dtdx * ((fxl[v] - fxh[v]) + (fyl[v] - fyh[v])), 9)
        end
        if r[1] < smallr       # positivity floor (NestedMHD): static floored cell, B kept
            r = (smallr, zero(T), zero(T), zero(T),
                 pfl / (gamma - one(T)) + T(0.5) * (r[6]*r[6] + r[7]*r[7] + r[8]*r[8]),
                 r[6], r[7], r[8], r[9])
        end
        # keep-old-state safety (the hydro epilogue twin): a nonfinite update never
        # poisons the level
        ok = true
        Base.Cartesian.@nexprs 9 v -> (ok &= isfinite(r[v]))
        r = ok ? r : u0
        q = cons2prim(r, gamma, smallr, pfl)
        Do_[idx]  = _narrow(eltype(Do_), r[1] / T(dsc))
        S1o_[idx] = _narrow(eltype(S1o_), r[2] / T(ssc))
        S2o_[idx] = _narrow(eltype(S2o_), r[3] / T(ssc))
        S3o_[idx] = _narrow(eltype(S3o_), r[4] / T(ssc))
        Eo_[idx]  = _narrow(eltype(Eo_), r[5] / T(esc))
        Geo_[idx] = _narrow(eltype(Geo_), q[5] / (gamma - one(T)) / T(gsc))
        Bxo_[idx] = r[6]; Byo_[idx] = r[7]; Bzo_[idx] = r[8]
        Pso_[idx] = r[9] * glm_fac
    end
end

@inline _narrow(::Type{Float64}, v::Float64) = v

# ── CFL signal: Σ_d(|v_d| + c_f,d) per active cell ───────────────────────────
@kernel function _mhd_signal_k!(out, @Const(D), @Const(S1), @Const(S2), @Const(S3), @Const(E),
                                @Const(Bx), @Const(By), @Const(Bz), @Const(Ps),
                                @Const(live_d), @Const(Dsc), @Const(Ssc), @Const(Esc),
                                gamma::T, smallr::T, pfl::T, sigdims::Int32,
                                B::Int32, ng::Int32, nd::Int32, stride::Int32,
                                dimv::Val{DIM}) where {T,DIM}
    t = @index(Global)
    t0 = Int32(t) - Int32(1)
    B3 = _nblk(B, dimv)
    bi = t0 ÷ B3; c = t0 % B3
    @inbounds begin
        slot = live_d[bi + Int32(1)]
        base = (slot - Int32(1)) * stride
        i, j, k = _cell_ijk(c, B, ng, dimv)
        q = cons2prim(_mload(T, D, S1, S2, S3, E, Bx, By, Bz, Ps, base + _lidx(i, j, k, nd),
                             Dsc[slot], Ssc[slot], Esc[slot]), gamma, smallr, pfl)
        out[t] = signal_speed(q, gamma, Int(sigdims))
    end
end

_mpar(hier::AMRHierarchy) = (T = eltype(hier.levels[1].D);
    (T(hier.gamma), T(hier.mhd.smallr), T(hier.mhd.smallr * hier.mhd.smallc^2 / hier.gamma)))

"Max unsplit MHD signal Σ_d(|v_d|+c_f,d) over the active cells of a level."
function mhd_max_signal(hier::AMRHierarchy, lev::Level)
    n = length(lev.live) * _nact(lev)
    n == 0 && return 0.0
    T = eltype(lev.D)
    out = device_zeros(lev.be, T, (n,))
    γ, sr, pfl = _mpar(hier)
    _mhd_signal_k!(lev.be)(out, gasfields(lev)[1:5]..., mfields(lev)..., lev.live_d,
                           lev.Dsc, lev.Ssc, lev.Esc, γ, sr, pfl, Int32(hier.mhd.sigdims),
                           Int32(lev.B), Int32(lev.ng), Int32(lev.nd), Int32(lev.stride),
                           Val(lev.dim); ndrange = n)
    return Float64(maximum(out))
end

"""The global GLM cleaning speed (one c_h for every level), from the CFL λ of the
root step (`hier.λ` = cfl/s_max) — not a shortened step (NestedMHD sweep-fixes 176058b)."""
glm_ch(hier::AMRHierarchy) =
    hier.mhd.glm_ch_scale * (hier.cfl / Float64(hier.λ)) / hier.mhd.sigdims

"""
    mhd_level!(hier, l, λ)

One MUSCL-Hancock GLM-MHD substep of level `l`: R → O (the caller swaps).
Requires current R ghosts (fill_ghosts!).  The scratch stays valid for the
C/F captures that follow (capture_fine!/capture_coarse!).
"""
function mhd_level!(hier::AMRHierarchy, l::Int, λ::Real)
    lev = hier.levels[l + 1]
    isempty(lev.live) && return nothing
    T = eltype(lev.D)
    γ, sr, pfl = _mpar(hier)
    ch = glm_ch(hier)
    dt = Float64(λ) * level_dx(hier, l)
    cp = hier.mhd.glm_cp_coef
    glm_fac = cp > 0 ? T(exp(-ch * dt / (cp * hier.box))) : one(T)
    rec = _mrec(lev); nt = lev.cap * lev.stride
    nb = length(lev.live); R = lev.B + 2
    _mhd_rec_k!(lev.be)(rec, gasfields(lev)[1:5]..., mfields(lev)..., lev.live_d,
                        lev.Dsc, lev.Ssc, lev.Esc, γ, sr, pfl, T(λ),
                        Int32(lev.B), Int32(lev.ng), Int32(lev.nd), Int32(lev.stride), nt,
                        Val(lev.dim); ndrange = nb * R^lev.dim)
    _mhd_update_k!(lev.be)(gasfields_o(lev)..., mfields_o(lev)...,
                           gasfields(lev)[1:5]..., mfields(lev)..., rec, lev.live_d,
                           lev.Dsc, lev.Ssc, lev.Esc, lev.Gsc,
                           γ, T(ch), sr, pfl, T(λ), glm_fac,
                           Int32(lev.B), Int32(lev.ng), Int32(lev.nd), Int32(lev.stride), nt,
                           Val(hier.mhd.riemann), Val(lev.dim); ndrange = nb * _nact(lev))
    return nothing
end

# ── C/F flux registers (9 lanes), read straight from the level scratch ─────────
# Registry entries as reflux.jl.  Coarse: the face of the uncovered coarse cell
# toward the fine block; fine: the 2² (2-D: 2) fine sub-faces of the fine block.
@kernel function _mhd_capture_coarse_k!(reg, @Const(ent), @Const(rec), wc::T, gamma::T, ch::T,
                                        smallr::T, pfl::T, nd::Int32, stride::Int32, nt::Int,
                                        rv::Val) where {T}
    e = @index(Global)
    b = (Int32(e) - Int32(1)) * Int32(16)
    @inbounds begin
        base = (ent[b+1] - Int32(1)) * stride
        ci = ent[b+2]; cj = ent[b+3]; ck = ent[b+4]
        ax = ent[b+5]; side = ent[b+6]
        di = ax == Int32(1) ? Int32(1) : Int32(0)
        dj = ax == Int32(2) ? Int32(1) : Int32(0)
        dk = ax == Int32(3) ? Int32(1) : Int32(0)
        gc = Int(base + _lidx(ci, cj, ck, nd))
        so = Int(base + _lidx(ci + di, cj + dj, ck + dk, nd)) - gc
        # side=+1: the coarse cell's LOW face (fine block below); −1: its HIGH face
        gL = side == Int32(1) ? gc - so : gc
        f = _mhd_face(rec, gL, gL + so, Int(ax), nt, rv, gamma, ch, smallr, pfl)
        r = (Int(e) - 1) * 9
        Base.Cartesian.@nexprs 9 v -> (reg[r + v] -= wc * ifelse(isfinite(f[v]), f[v], zero(T)))
    end
end

@kernel function _mhd_capture_fine_k!(reg, @Const(ent), @Const(rec), wf::T, gamma::T, ch::T,
                                      smallr::T, pfl::T, nd::Int32, stride::Int32, nt::Int,
                                      kz::Int32, rv::Val) where {T}
    e = @index(Global)
    b = (Int32(e) - Int32(1)) * Int32(16)
    @inbounds begin
        base = (ent[b+7] - Int32(1)) * stride
        fi = ent[b+8]; fj = ent[b+9]; fk = ent[b+10]
        ax = ent[b+5]; side = ent[b+6]
        sh = side == Int32(1) ? Int32(0) : Int32(-1)
        di = ax == Int32(1) ? Int32(1) : Int32(0)
        dj = ax == Int32(2) ? Int32(1) : Int32(0)
        dk = ax == Int32(3) ? Int32(1) : Int32(0)
        t1i = ax == Int32(1) ? Int32(0) : Int32(1)
        t1j = ax == Int32(1) ? Int32(1) : Int32(0)
        t2j = ax == Int32(3) ? Int32(1) : Int32(0)
        t2k = ax == Int32(3) ? Int32(0) : Int32(1)
        acc = ntuple(_ -> zero(T), Val(9))
        for s2 in Int32(0):kz, s1 in Int32(0):Int32(1)    # 2-D: t2 is the flat z
            i = fi + sh * di + s1 * t1i
            j = fj + sh * dj + s1 * t1j + s2 * t2j
            k = fk + sh * dk + s2 * t2k
            gL = Int(base + _lidx(i, j, k, nd))
            gR = Int(base + _lidx(i + di, j + dj, k + dk, nd))
            f = _mhd_face(rec, gL, gR, Int(ax), nt, rv, gamma, ch, smallr, pfl)
            acc = _acc9(acc, f, one(T))
        end
        q = (kz == Int32(1) ? T(0.25) : T(0.5)) * wf
        r = (Int(e) - 1) * 9
        Base.Cartesian.@nexprs 9 v -> (reg[r + v] += q * acc[v])
    end
end

# closure-free 9-lane accumulators (a closure over a reassigned tuple boxes it —
# fine on the CPU, an invalid GPU kernel)
@inline _acc9(a::NTuple{9,T}, f::NTuple{9,T}, w::T) where {T} =
    ntuple(v -> a[v] + w * ifelse(isfinite(f[v]), f[v], zero(T)), Val(9))
@inline _reg9(reg, r::Int) = ntuple(v -> @inbounds(reg[r + v]), Val(9))
@inline _add9(a::NTuple{9,T}, b::NTuple{9,T}) where {T} = ntuple(v -> a[v] + b[v], Val(9))

# U_c += λ_c·side·reg grouped by target coarse cell; NestedMHD's positivity guard
# (skip a correction that would leave ρ ≤ 0 or thermal energy ≤ 0).
@kernel function _mhd_reflux_apply_k!(D, S1, S2, S3, E, Ge, Bx, By, Bz, Ps,
                                      @Const(Dsc), @Const(Ssc), @Const(Esc), @Const(Gsc),
                                      @Const(ent), @Const(reg), @Const(perm), @Const(gs),
                                      λc::T, gamma::T, smallr::T, pfl::T,
                                      nd::Int32, stride::Int32) where {T}
    g = @index(Global)
    @inbounds begin
        a = ntuple(_ -> zero(T), Val(9))
        for ii in gs[g]:gs[Int32(g)+Int32(1)] - Int32(1)
            e = perm[ii]
            w = λc * T(ent[(e - Int32(1)) * Int32(16) + Int32(6)])
            a = _acc9(a, _reg9(reg, (Int(e) - 1) * 9), w)
        end
        b = (perm[gs[g]] - Int32(1)) * Int32(16)
        slot = ent[b+1]
        idx = (slot - Int32(1)) * stride + _lidx(ent[b+2], ent[b+3], ent[b+4], nd)
        dsc = Dsc[slot]; ssc = Ssc[slot]; esc = Esc[slot]; gsc = Gsc[slot]
        u = _mload(T, D, S1, S2, S3, E, Bx, By, Bz, Ps, idx, dsc, ssc, esc)
        q = _add9(u, a)
        ρn = q[1]
        eth = q[5] - (q[2]*q[2] + q[3]*q[3] + q[4]*q[4]) / (2ρn) -
              T(0.5) * (q[6]*q[6] + q[7]*q[7] + q[8]*q[8])
        ok = (ρn > zero(T)) & (eth > zero(T))
        Base.Cartesian.@nexprs 9 v -> (ok &= isfinite(q[v]))
        if ok
            D[idx]  = _narrow(eltype(D), q[1] / T(dsc))
            S1[idx] = _narrow(eltype(S1), q[2] / T(ssc))
            S2[idx] = _narrow(eltype(S2), q[3] / T(ssc))
            S3[idx] = _narrow(eltype(S3), q[4] / T(ssc))
            E[idx]  = _narrow(eltype(E), q[5] / T(esc))
            Bx[idx] = q[6]; By[idx] = q[7]; Bz[idx] = q[8]; Ps[idx] = q[9]
            Ge[idx] = _narrow(eltype(Ge),
                              cons2prim(q, gamma, smallr, pfl)[5] / (gamma - one(T)) / T(gsc))
        end
    end
end

function mhd_capture_fine!(hier::AMRHierarchy, l::Int, w::Real, λ::Real)
    lev = hier.levels[l + 1]
    n = get(lev.tabs, :cfn, 0)::Int
    n == 0 && return nothing
    T = eltype(lev.D); γ, sr, pfl = _mpar(hier)
    _mhd_capture_fine_k!(lev.be)(_tabt_any(lev, :cfreg), _tabi(lev, :cf), _mrec(lev),
                                 T(w), γ, T(glm_ch(hier)), sr, pfl, Int32(lev.nd),
                                 Int32(lev.stride), lev.cap * lev.stride, _kz(lev),
                                 Val(hier.mhd.riemann); ndrange = n)
    return nothing
end

function mhd_capture_coarse!(hier::AMRHierarchy, l::Int, w::Real, λ::Real)
    lev = hier.levels[l + 1]; plev = hier.levels[l]
    n = get(lev.tabs, :cfn, 0)::Int
    n == 0 && return nothing
    T = eltype(lev.D); γ, sr, pfl = _mpar(hier)
    _mhd_capture_coarse_k!(lev.be)(_tabt_any(lev, :cfreg), _tabi(lev, :cf), _mrec(plev),
                                   T(w), γ, T(glm_ch(hier)), sr, pfl, Int32(plev.nd),
                                   Int32(plev.stride), plev.cap * plev.stride,
                                   Val(hier.mhd.riemann); ndrange = n)
    return nothing
end

function mhd_reflux_apply!(hier::AMRHierarchy, l::Int, λc::Real)
    lev = hier.levels[l + 1]; plev = hier.levels[l]
    n = get(lev.tabs, :cfn, 0)::Int
    n == 0 && return nothing
    ng_ = lev.tabs[:cfng]::Int
    ng_ == 0 && return nothing
    T = eltype(lev.D); γ, sr, pfl = _mpar(hier)
    reg = _tabt_any(lev, :cfreg)
    _mhd_reflux_apply_k!(lev.be)(gasfields(plev)..., mfields(plev)...,
                                 plev.Dsc, plev.Ssc, plev.Esc, plev.Gsc,
                                 _tabi(lev, :cf), reg, _tabi(lev, :cfperm), _tabi(lev, :cfgs),
                                 T(λc), γ, sr, pfl, Int32(plev.nd), Int32(plev.stride);
                                 ndrange = ng_)
    fill!(reg, zero(T))
    return nothing
end

@inline _tabt_any(lev::Level, k::Symbol) = lev.tabs[k]

# ── host helpers: ICs + diagnostics ───────────────────────────────────────────
"""
    encode_mhd_from_host!(lev, hD, hS1, hS2, hS3, hE, hGe, hBx, hBy, hBz, hPsi)

Set an MHD level from PHYSICAL host pools (pool layout); `hE` is the TOTAL energy
(incl. ½B²), `hGe` = p/(γ−1).  Gas classes go through `encode_from_host!`.
"""
function encode_mhd_from_host!(lev::Level, hD, hS1, hS2, hS3, hE, hGe, hBx, hBy, hBz, hPsi)
    encode_from_host!(lev, hD, hS1, hS2, hS3, hE, hGe)
    T = eltype(lev.D)
    for (dev, h) in zip(mfields(lev), (hBx, hBy, hBz, hPsi))
        copyto!(dev, convert(Array{T}, h))
    end
    return nothing
end

"""
    max_divb(hier) -> Float64

max over every level's active cells of |∇·B|·dx / max|B| — the centred-difference
divergence of the cell-centred field relative to the GLOBAL peak field (a local
normalization is meaningless at field nulls).  Fills R ghosts first.
"""
function max_divb(hier::AMRHierarchy)
    worst = 0.0; bmax = 0.0
    for l in 0:length(hier.levels)-1
        lev = hier.levels[l + 1]
        (isempty(lev.live) || !ismhd(lev)) && continue
        fill_ghosts!(hier, l)
        hb = map(Array, mfields(lev)[1:3])
        nd = lev.nd; ng = lev.ng; kz = hier.dim == 3 ? 1 : 0
        for s in lev.live
            base = (Int(s) - 1) * lev.stride
            ix(i, j, k) = base + (k * nd + j) * nd + i + 1
            for k in (kz == 1 ? (ng:ng+lev.B-1) : (0:0)), j in ng:ng+lev.B-1, i in ng:ng+lev.B-1
                dv = 0.5 * ((hb[1][ix(i+1,j,k)] - hb[1][ix(i-1,j,k)]) +
                            (hb[2][ix(i,j+1,k)] - hb[2][ix(i,j-1,k)]))
                kz == 1 && (dv += 0.5 * (hb[3][ix(i,j,k+1)] - hb[3][ix(i,j,k-1)]))
                bmax = max(bmax, sqrt(hb[1][ix(i,j,k)]^2 + hb[2][ix(i,j,k)]^2 + hb[3][ix(i,j,k)]^2))
                worst = max(worst, abs(dv))
            end
        end
    end
    return worst / max(bmax, floatmin(Float64))
end

"Composite volume integrals ∫B dV (covered coarse cells excluded) — exact under reflux."
function total_bflux(hier::AMRHierarchy)
    tot = zeros(3)
    for l in 0:length(hier.levels)-1
        lev = hier.levels[l + 1]
        (isempty(lev.live) || !ismhd(lev)) && continue
        dV = level_dx(hier, l)^hier.dim
        flev = l + 1 <= length(hier.levels) - 1 ? hier.levels[l + 2] : nothing
        hb = map(Array, mfields(lev)[1:3])
        kg = _ngax(lev, 3); r3 = hier.dim == 3 ? 2 : 1
        for s in lev.live
            m = lev.meta[s]; base = (Int(s) - 1) * lev.stride
            for k in 0:_Bax(lev, 3)-1, j in 0:lev.B-1, i in 0:lev.B-1
                if flev !== nothing && !isempty(flev.live)
                    g = (Int128(m.origin[1]) + i, Int128(m.origin[2]) + j, Int128(m.origin[3]) + k)
                    gf = ntuple(d -> Int128(d == 3 ? r3 : 2) * g[d], 3)
                    isempty(overlapping_blocks(flev, gf, (2, 2, r3))) || continue
                end
                idx = base + ((k + kg) * lev.nd + (j + lev.ng)) * lev.nd + (i + lev.ng) + 1
                for d in 1:3; tot[d] += Float64(hb[d][idx]) * dV; end
            end
        end
    end
    return (tot[1], tot[2], tot[3])
end
