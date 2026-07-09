# kernels.jl — the fused batched hydro kernel + the shared flux chain.
#
# ONE launch advances every live block of a level (thread ↔ (slot, active cell)).
# Scheme: SSP-RK2, PLM (minmod) primitive reconstruction, HLLC — the exact physics
# of the euler_nested reference (HierarchicalGrids examples) so the AMR machinery
# can be validated against it.  Dual energy: pressure comes from Ge ((γ−1)·Ge);
# Ge itself is advected as a mass-flux color (CMA upwind) with a pdV source from
# the HLLC contact speeds; Tau carries the conservative energy flux; an Enzo-style
# selection (Ge ← Tau−KE where well-resolved) closes the pair.
#
# Conservation contract: the face flux is computed by BOTH adjacent cells through
# the SAME `@inline` chain with the SAME stencil arguments — bit-identical values
# on the two sides, so interior faces cancel to per-cell accumulation round-off,
# and the reflux recompute (reflux.jl) reproduces the applied flux exactly.
#
# The update is length-unit-free: `U_new = U_old − λ·Σ(F_hi − F_lo)` with the ONE
# fraction λ = dt/dx passed per launch (a single global constant under strict 2:1
# subcycling).  All math is Float32 registers; loads/stores go through `Float32()`
# / `_narrow()` so the same kernel serves f32 (validation) and f16 (production,
# with per-block scales layered on in the f16 phase).

import ChemistryKernels: encode_log2sp, decode_log2sp

@inline _lidx(i::Int32, j::Int32, k::Int32, nd::Int32) = (k * nd + j) * nd + i + Int32(1)

# load one cell's conserved state as PHYSICAL f32 (per-block class scales un-lift)
@inline function _loadU(D, S1, S2, S3, Tau, Ge, idx::Int32,
                        dsc::Float32, ssc::Float32, esc::Float32, gsc::Float32)
    @inbounds (Float32(D[idx]) * dsc, Float32(S1[idx]) * ssc, Float32(S2[idx]) * ssc,
               Float32(S3[idx]) * ssc, Float32(Tau[idx]) * esc, Float32(Ge[idx]) * gsc)
end

# primitives (ρ,u,v,w,P) with DUAL-ENERGY pressure P = (γ−1)·Ge, floored.
# VACUUM GATE: a cell whose stored density is ≤0 (f16 underflow under an extreme
# per-block scale window) carries no mass; dividing its (quantization-noise)
# momentum by the 1e-30 floor manufactures ~1e30 velocities that overflow kin to
# Inf and detonate the Riemann solve (the bamr256L10 λ-collapse).  Vacuum cells
# get v=0 — the hydro twin of the chem gather's ρ>0 predicate.
@inline function _prim_de(U::NTuple{6,Float32}, γ::Float32)
    ρ = max(U[1], 1.0f-30); inv = ifelse(U[1] > 0.0f0, 1.0f0 / ρ, 0.0f0)
    vx = U[2] * inv; vy = U[3] * inv; vz = U[4] * inv
    kin = 0.5f0 * ρ * (vx * vx + vy * vy + vz * vz)
    P = (γ - 1.0f0) * max(U[6], 1.0f-30)
    return (ρ, vx, vy, vz, max(P, 1.0f-12 * kin))
end

@inline _minmod(a::Float32, b::Float32) = a * b <= 0.0f0 ? 0.0f0 :
                                          (abs(a) < abs(b) ? a : b)

@inline _slopes(Wm::NTuple{5,Float32}, Wc::NTuple{5,Float32}, Wp::NTuple{5,Float32}) =
    ntuple(q -> _minmod(Wc[q] - Wm[q], Wp[q] - Wc[q]), 5)

@inline function _prim2cons(W::NTuple{5,Float32}, γ::Float32)
    ρ, vx, vy, vz, P = W
    E = P / (γ - 1.0f0) + 0.5f0 * ρ * (vx * vx + vy * vy + vz * vz)
    return (ρ, ρ * vx, ρ * vy, ρ * vz, E)
end

@inline function _euler_flux(W::NTuple{5,Float32}, γ::Float32, ax::Int32)
    ρ, vx, vy, vz, P = W
    un = ax == Int32(1) ? vx : (ax == Int32(2) ? vy : vz)
    E = P / (γ - 1.0f0) + 0.5f0 * ρ * (vx * vx + vy * vy + vz * vz)
    return (ρ * un,
            ρ * vx * un + (ax == Int32(1) ? P : 0.0f0),
            ρ * vy * un + (ax == Int32(2) ? P : 0.0f0),
            ρ * vz * un + (ax == Int32(3) ? P : 0.0f0),
            (E + P) * un)
end

@inline function _hllc_star(W::NTuple{5,Float32}, U::NTuple{5,Float32},
                            F::NTuple{5,Float32}, SK::Float32, Ss::Float32,
                            ax::Int32)
    ρ = W[1]
    un = ax == Int32(1) ? W[2] : (ax == Int32(2) ? W[3] : W[4])
    P = W[5]; E = U[5]
    fac = ρ * (SK - un) / (SK - Ss)
    vs1 = ax == Int32(1) ? Ss : W[2]
    vs2 = ax == Int32(2) ? Ss : W[3]
    vs3 = ax == Int32(3) ? Ss : W[4]
    Es  = E / ρ + (Ss - un) * (Ss + P / (ρ * (SK - un)))
    return (F[1] + SK * (fac       - U[1]),
            F[2] + SK * (fac * vs1 - U[2]),
            F[3] + SK * (fac * vs2 - U[3]),
            F[4] + SK * (fac * vs3 - U[4]),
            F[5] + SK * (fac * Es  - U[5]))
end

"HLLC flux + contact speed at a face; identical to the euler_nested reference."
@inline function _hllc(WL::NTuple{5,Float32}, WR::NTuple{5,Float32},
                       γ::Float32, ax::Int32)
    ρL = WL[1]; ρR = WR[1]; PL = WL[5]; PR = WR[5]
    uL = ax == Int32(1) ? WL[2] : (ax == Int32(2) ? WL[3] : WL[4])
    uR = ax == Int32(1) ? WR[2] : (ax == Int32(2) ? WR[3] : WR[4])
    cL = sqrt(γ * PL / ρL); cR = sqrt(γ * PR / ρR)
    SL = @fastmath min(uL - cL, uR - cR)
    SR = @fastmath max(uL + cL, uR + cR)
    Ss = (PR - PL + ρL * uL * (SL - uL) - ρR * uR * (SR - uR)) /
         (ρL * (SL - uL) - ρR * (SR - uR))
    if SL >= 0.0f0
        return _euler_flux(WL, γ, ax), Ss
    elseif SR <= 0.0f0
        return _euler_flux(WR, γ, ax), Ss
    elseif Ss >= 0.0f0
        return _hllc_star(WL, _prim2cons(WL, γ), _euler_flux(WL, γ, ax), SL, Ss, ax), Ss
    else
        return _hllc_star(WR, _prim2cons(WR, γ), _euler_flux(WR, γ, ax), SR, Ss, ax), Ss
    end
end

"""
PLM+HLLC flux at the face between stencil cells 2 and 3 of a 4-cell window
(W1 W2 | W3 W4 along `ax`).  Called by BOTH adjacent cells (and the reflux
recompute) with identical arguments → bit-identical flux.
"""
@inline function _face_flux(W1::NTuple{5,Float32}, W2::NTuple{5,Float32},
                            W3::NTuple{5,Float32}, W4::NTuple{5,Float32},
                            γ::Float32, ax::Int32)
    sL = _slopes(W1, W2, W3)
    sR = _slopes(W2, W3, W4)
    WLf = ntuple(q -> W2[q] + 0.5f0 * sL[q], 5)
    WRf = ntuple(q -> W3[q] - 0.5f0 * sR[q], 5)
    return _hllc(WLf, WRf, γ, ax)
end

# both fluxes of a cell along one axis, from the 5-cell stencil c−2..c+2, plus
# the CMA Ge fluxes (upwind cell-centred specific internal energy on the HLLC
# mass flux) and the contact speeds for the pdV source.
@inline function _axis_update(D, S1, S2, S3, Tau, Ge, base::Int32,
                              i::Int32, j::Int32, k::Int32, nd::Int32,
                              γ::Float32, ax::Int32,
                              dsc::Float32, ssc::Float32, esc::Float32, gsc::Float32)
    di = ax == Int32(1) ? Int32(1) : Int32(0)
    dj = ax == Int32(2) ? Int32(1) : Int32(0)
    dk = ax == Int32(3) ? Int32(1) : Int32(0)
    idx(m) = base + _lidx(i + m * di, j + m * dj, k + m * dk, nd)
    Um2 = _loadU(D, S1, S2, S3, Tau, Ge, idx(Int32(-2)), dsc, ssc, esc, gsc)
    Um1 = _loadU(D, S1, S2, S3, Tau, Ge, idx(Int32(-1)), dsc, ssc, esc, gsc)
    Uc  = _loadU(D, S1, S2, S3, Tau, Ge, idx(Int32(0)), dsc, ssc, esc, gsc)
    Up1 = _loadU(D, S1, S2, S3, Tau, Ge, idx(Int32(1)), dsc, ssc, esc, gsc)
    Up2 = _loadU(D, S1, S2, S3, Tau, Ge, idx(Int32(2)), dsc, ssc, esc, gsc)
    Wm2 = _prim_de(Um2, γ); Wm1 = _prim_de(Um1, γ); Wc = _prim_de(Uc, γ)
    Wp1 = _prim_de(Up1, γ); Wp2 = _prim_de(Up2, γ)
    Flo, Sslo = _face_flux(Wm2, Wm1, Wc, Wp1, γ, ax)
    Fhi, Sshi = _face_flux(Wm1, Wc, Wp1, Wp2, γ, ax)
    # CMA: Ge rides the mass flux with the upwind cell's specific internal energy
    gelo = Flo[1] >= 0.0f0 ? Um1[6] / max(Um1[1], 1.0f-30) : Uc[6]  / max(Uc[1], 1.0f-30)
    gehi = Fhi[1] >= 0.0f0 ? Uc[6]  / max(Uc[1], 1.0f-30) : Up1[6] / max(Up1[1], 1.0f-30)
    dge  = Fhi[1] * gehi - Flo[1] * gelo
    return Flo, Fhi, dge, Sshi - Sslo, Uc
end

# ── the fused SSP-RK2 stage kernel ────────────────────────────────────────────
# OUT = w_old·OLD + (1−w_old)·(IN − λ·ΔF(IN))   per active cell of every block.
#   stage 1: OUT=O, IN=R, OLD=R, w=0        (O = R − λΔF(R))
#   stage 2: OUT=R, IN=O, OLD=R, w=½        (in-place R is safe: only own-cell R read)
@kernel function _rk_stage_k!(Do_, S1o_, S2o_, S3o_, Tauo_, Geo_, spout,
                              @Const(D), @Const(S1), @Const(S2), @Const(S3),
                              @Const(Tau), @Const(Ge), spin,
                              OldD, OldS1, OldS2, OldS3, OldTau, OldGe, spold,
                              w::Float32, @Const(live_d),
                              @Const(Dsc), @Const(Ssc), @Const(Esc), @Const(Gsc),
                              λ::Float32, γ::Float32, η::Float32,
                              B::Int32, ng::Int32, nd::Int32, stride::Int32,
                              ::Val{NS}) where {NS}
    t = @index(Global)
    t0 = Int32(t) - Int32(1)
    B3 = B * B * B
    bi = t0 ÷ B3; c = t0 % B3
    @inbounds begin
        slot = live_d[bi + Int32(1)]
        base = (slot - Int32(1)) * stride
        dsc = Dsc[slot]; ssc = Ssc[slot]; esc = Esc[slot]; gsc = Gsc[slot]
        i = c % B + ng; j = (c ÷ B) % B + ng; k = c ÷ (B * B) + ng
        Fx0, Fx1, dgex, dux, Uc = _axis_update(D, S1, S2, S3, Tau, Ge, base, i, j, k, nd, γ, Int32(1), dsc, ssc, esc, gsc)
        Fy0, Fy1, dgey, duy, _  = _axis_update(D, S1, S2, S3, Tau, Ge, base, i, j, k, nd, γ, Int32(2), dsc, ssc, esc, gsc)
        Fz0, Fz1, dgez, duz, _  = _axis_update(D, S1, S2, S3, Tau, Ge, base, i, j, k, nd, γ, Int32(3), dsc, ssc, esc, gsc)
        P_c = (γ - 1.0f0) * max(Uc[6], 1.0f-30)
        dU1 = (Fx1[1] - Fx0[1]) + (Fy1[1] - Fy0[1]) + (Fz1[1] - Fz0[1])
        dU2 = (Fx1[2] - Fx0[2]) + (Fy1[2] - Fy0[2]) + (Fz1[2] - Fz0[2])
        dU3 = (Fx1[3] - Fx0[3]) + (Fy1[3] - Fy0[3]) + (Fz1[3] - Fz0[3])
        dU4 = (Fx1[4] - Fx0[4]) + (Fy1[4] - Fy0[4]) + (Fz1[4] - Fz0[4])
        dU5 = (Fx1[5] - Fx0[5]) + (Fy1[5] - Fy0[5]) + (Fz1[5] - Fz0[5])
        dU6 = (dgex + dgey + dgez) + P_c * (dux + duy + duz)   # advection + pdV
        idx = base + _lidx(i, j, k, nd)
        nD  = Uc[1] - λ * dU1
        nS1 = Uc[2] - λ * dU2
        nS2 = Uc[3] - λ * dU3
        nS3 = Uc[4] - λ * dU4
        nT  = Uc[5] - λ * dU5
        nG  = Uc[6] - λ * dU6
        wo = w; wi = 1.0f0 - w
        nD  = wo * Float32(OldD[idx])   * dsc + wi * nD
        nS1 = wo * Float32(OldS1[idx])  * ssc + wi * nS1
        nS2 = wo * Float32(OldS2[idx])  * ssc + wi * nS2
        nS3 = wo * Float32(OldS3[idx])  * ssc + wi * nS3
        nT  = wo * Float32(OldTau[idx]) * esc + wi * nT
        nG  = wo * Float32(OldGe[idx])  * gsc + wi * nG
        # dual-energy selection: trust Tau−KE where it is well-resolved
        nD = max(nD, 1.0f-30)
        KE = 0.5f0 * (nS1 * nS1 + nS2 * nS2 + nS3 * nS3) / nD
        (nT - KE) > η * nT && (nG = nT - KE)
        Do_[idx]   = _narrow(eltype(Do_), nD / dsc)
        S1o_[idx]  = _narrow(eltype(S1o_), nS1 / ssc)
        S2o_[idx]  = _narrow(eltype(S2o_), nS2 / ssc)
        S3o_[idx]  = _narrow(eltype(S3o_), nS3 / ssc)
        Tauo_[idx] = _narrow(eltype(Tauo_), nT / esc)
        Geo_[idx]  = _narrow(eltype(Geo_), max(nG, 1.0f-30) / gsc)
        # ── species CMA: X rides the six HLLC mass fluxes, upwinded per face; the
        # conserved quantity is ρX (fractions decode from the absolute u16 codec) ──
        if NS > 0
            ρR = Float32(OldD[idx]) * dsc                 # OLD physical density (R)
            ixm = base + _lidx(i - Int32(1), j, k, nd); ixp = base + _lidx(i + Int32(1), j, k, nd)
            iym = base + _lidx(i, j - Int32(1), k, nd); iyp = base + _lidx(i, j + Int32(1), k, nd)
            izm = base + _lidx(i, j, k - Int32(1), nd); izp = base + _lidx(i, j, k + Int32(1), nd)
            for q in 1:NS
                sq = spin[q]
                Xc  = decode_log2sp(Float32, sq[idx])
                aX  = Fx1[1] * (Fx1[1] >= 0.0f0 ? Xc : decode_log2sp(Float32, sq[ixp])) -
                      Fx0[1] * (Fx0[1] >= 0.0f0 ? decode_log2sp(Float32, sq[ixm]) : Xc) +
                      Fy1[1] * (Fy1[1] >= 0.0f0 ? Xc : decode_log2sp(Float32, sq[iyp])) -
                      Fy0[1] * (Fy0[1] >= 0.0f0 ? decode_log2sp(Float32, sq[iym]) : Xc) +
                      Fz1[1] * (Fz1[1] >= 0.0f0 ? Xc : decode_log2sp(Float32, sq[izp])) -
                      Fz0[1] * (Fz0[1] >= 0.0f0 ? decode_log2sp(Float32, sq[izm]) : Xc)
                ρXold = decode_log2sp(Float32, spold[q][idx]) * ρR
                ρXnew = wo * ρXold + wi * (Xc * Uc[1] - λ * aX)
                spout[q][idx] = encode_log2sp(max(ρXnew, 0.0f0) / max(nD, 1.0f-30))
            end
        end
    end
end

# ── batched CFL reduction ─────────────────────────────────────────────────────
@kernel function _max_signal_k!(out, @Const(D), @Const(S1), @Const(S2), @Const(S3),
                                @Const(Ge), @Const(live_d),
                                @Const(Dsc), @Const(Ssc), @Const(Gsc), γ::Float32,
                                B::Int32, ng::Int32, nd::Int32, stride::Int32)
    t = @index(Global)
    t0 = Int32(t) - Int32(1)
    B3 = B * B * B
    bi = t0 ÷ B3; c = t0 % B3
    @inbounds begin
        slot = live_d[bi + Int32(1)]
        base = (slot - Int32(1)) * stride
        dsc = Dsc[slot]; ssc = Ssc[slot]; gsc = Gsc[slot]
        i = c % B + ng; j = (c ÷ B) % B + ng; k = c ÷ (B * B) + ng
        idx = base + _lidx(i, j, k, nd)
        ρraw = Float32(D[idx]) * dsc
        # vacuum gate (see _prim_de): a stored-ρ≤0 cell carries no signal —
        # v = S/1e-30 garbage must never set the global λ.
        inv = ifelse(ρraw > 0.0f0, 1.0f0 / max(ρraw, 1.0f-30), 0.0f0)
        vx = Float32(S1[idx]) * ssc * inv; vy = Float32(S2[idx]) * ssc * inv
        vz = Float32(S3[idx]) * ssc * inv
        cs = sqrt(γ * (γ - 1.0f0) * max(Float32(Ge[idx]) * gsc, 1.0f-30) * inv)
        out[Int32(t)] = max(abs(vx), max(abs(vy), abs(vz))) + cs
    end
end

"Max |v|+c_s over all active cells of a level (0 for an empty level)."
function max_signal(lev::Level, γ::Real)
    n = length(lev.live) * lev.B^3
    n == 0 && return 0.0f0
    out = device_zeros(lev.be, Float32, (n,))
    _max_signal_k!(lev.be)(out, lev.D, lev.S1, lev.S2, lev.S3, lev.Ge, lev.live_d,
                           lev.Dsc, lev.Ssc, lev.Gsc,
                           Float32(γ), Int32(lev.B), Int32(lev.ng), Int32(lev.nd),
                           Int32(lev.stride); ndrange = n)
    return maximum(out)                       # device reduction
end

"""
    stage_level!(hier, l, λ; w, IN, OUT)

One SSP-RK2 stage over every live block of level `l`:
`OUT = w·R_old + (1−w)·(IN − λ·ΔF(IN))`.  `IN`/`OUT` ∈ {:R, :O}; the old state is
always the R buffers (SSP-RK2's U^n; stage 2 writes R in place, which is safe —
only the cell's own R value is read).
"""
function stage_level!(hier::AMRHierarchy, l::Int, λ::Float32;
                      w::Float32, IN::Symbol, OUT::Symbol, η::Float32 = 1.0f-3)
    lev = hier.levels[l + 1]
    isempty(lev.live) && return nothing
    fin  = IN === :R ? gasfields(lev) : gasfields_o(lev)
    fout = OUT === :R ? gasfields(lev) : gasfields_o(lev)
    fold = gasfields(lev)
    NS = lev.nsp
    sin_  = IN === :R ? ntuple(q -> lev.sp[q], NS)  : ntuple(q -> lev.spo[q], NS)
    sout_ = OUT === :R ? ntuple(q -> lev.sp[q], NS) : ntuple(q -> lev.spo[q], NS)
    sold_ = ntuple(q -> lev.sp[q], NS)
    n = length(lev.live) * lev.B^3
    _rk_stage_k!(lev.be)(fout..., sout_, fin..., sin_, fold..., sold_, w, lev.live_d,
                         lev.Dsc, lev.Ssc, lev.Esc, lev.Gsc, λ,
                         Float32(hier.gamma), η, Int32(lev.B), Int32(lev.ng),
                         Int32(lev.nd), Int32(lev.stride), Val(NS); ndrange = n)
    return nothing
end
