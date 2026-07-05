# kernels_ctu.jl — single-pass CTU hydro: a tiled GPU kernel + a per-cell oracle.
#
# Port of the FVGK `k_ctus_de16s` structure (transpile_cuda.jl) onto the block
# pools: unsplit MUSCL–Hancock with the transverse (CTU-style) predictor —
# every cell gets ONE dU = −λ/2·Σ_dirs(F(face⁺)−F(face⁻)) from PLM face values;
# the corrector solves ONE HLLC per face between dU-evolved predicted states.
# ONE ghost fill + ONE read + ONE write per substep (RK2 needs two of each).
#
# Two kernels, ONE physics chain (the @inlines below):
#   * `_ctu_step_k!` — tiled (GPU-only, cpu=false): workgroup = one 8×8×4 tile
#     of one block; phase 0 stages prims into an f16 SHARED tile in HAT units
#     (ρ̂=ρ/Dsc, u,v,w physical, ê=Ĝe/ρ̂ — per-block po2 windows keep every lane
#     f16-normal; block-uniform scales lift back to physical f32 on read;
#     species stay u16 codes); phase 1 computes dU once per (+1)-halo cell into
#     f32 shared; phase 2 sweeps the three face planes recon-once/Riemann-once.
#   * `_ctu_cell_k!` — per-cell (CPU+GPU): gathers through `_ctu_prims_pool`,
#     which REPLICATES the f16 hat-tile rounding — same inputs, same inlines,
#     same accumulation order.  On GPU the two kernels agree to ~1 ulp (ptxas
#     contracts a*b+c → fma per-kernel by scheduling context; true bitwise
#     equality across differently-structured kernels would require banning
#     contraction).  The validation oracle and the CPU path.
#
# Conservation contract: blocks sharing a face hold bit-identical physical halo
# values (po2 ghost rescale and f16 hat rounding commute across blocks), and
# WITHIN one kernel both sides of a face run the same compiled instructions ⇒
# interior faces cancel exactly.  The reflux capture (`_ctu_face6`) recomputes
# through the same rounding + inlines from a different kernel, so it matches
# the applied flux to fma-contraction round-off (~1 ulp) — far below the C/F
# conservation tolerances.
#
# Deviation from FVGK (documented): the predictor carries NO total-energy lane
# (the corrector's HLLC rebuilds E from the face pressure, so it never reads
# one), pressure comes from the ge lane throughout (the BlockAMR dual-energy
# convention), and the DE epilogue selects Ge ← Tau−KE where resolved WITHOUT
# rewriting Tau (matching the RK2 kernel, not FVGK's E-repair).

# ── the shared CTU physics chain (pure @inline on loaded 5-lane states) ───────
# cell state c = (ρ, u, v, w, e) physical f32; P = (γ−1)·ρ·e on demand.

"(ρ,u,v,w,e) → (ρ,u,v,w,P) with the DE pressure (+ the _prim_de floors)."
@inline function _W5(c::NTuple{5,Float32}, γ::Float32)
    ρ, u, v, w, e = c
    kin = 0.5f0 * ρ * (u * u + v * v + w * w)
    P = (γ - 1.0f0) * max(ρ * e, 1.0f-30)
    return (ρ, u, v, w, max(P, 1.0f-12 * kin))
end

# predictor lane flux: (ρ, ρu, ρv, ρw, ρe) — no E lane.
@inline function _ctu_flux5(c::NTuple{5,Float32}, γ::Float32, ax::Int32)
    W = _W5(c, γ)
    un = ax == Int32(1) ? W[2] : (ax == Int32(2) ? W[3] : W[4])
    m = W[1] * un
    return (m,
            m * W[2] + (ax == Int32(1) ? W[5] : 0.0f0),
            m * W[3] + (ax == Int32(2) ? W[5] : 0.0f0),
            m * W[4] + (ax == Int32(3) ? W[5] : 0.0f0),
            m * c[5])
end

"PLM (minmod) one-sided face value; side −1 = lo face, +1 = hi face."
@inline _recon_one(m::NTuple{5,Float32}, c::NTuple{5,Float32}, p::NTuple{5,Float32},
                   side::Float32) =
    ntuple(q -> c[q] + 0.5f0 * side * _minmod(c[q] - m[q], p[q] - c[q]), 5)

"per-cell CTU predictor dU (5 lanes) = −λ/2·Σ_dirs(F(face⁺)−F(face⁻))."
@inline function _ctu_dU5(xm::NTuple{5,Float32}, xc::NTuple{5,Float32},
                          xp::NTuple{5,Float32},
                          ym::NTuple{5,Float32}, yp::NTuple{5,Float32},
                          zm::NTuple{5,Float32}, zp::NTuple{5,Float32},
                          λ::Float32, γ::Float32)
    fLx = _ctu_flux5(_recon_one(xm, xc, xp, -1.0f0), γ, Int32(1))
    fRx = _ctu_flux5(_recon_one(xm, xc, xp,  1.0f0), γ, Int32(1))
    fLy = _ctu_flux5(_recon_one(ym, xc, yp, -1.0f0), γ, Int32(2))
    fRy = _ctu_flux5(_recon_one(ym, xc, yp,  1.0f0), γ, Int32(2))
    fLz = _ctu_flux5(_recon_one(zm, xc, zp, -1.0f0), γ, Int32(3))
    fRz = _ctu_flux5(_recon_one(zm, xc, zp,  1.0f0), γ, Int32(3))
    return ntuple(q -> ((fRx[q] - fLx[q]) + (fRy[q] - fLy[q]) + (fRz[q] - fLz[q])) *
                       (-0.5f0) * λ, 5)
end

"the ρX predictor lane: −λ/2·Σ(F_m⁺·X⁺ − F_m⁻·X⁻) with PLM'd X per direction."
@inline function _ctu_dUX(xm::NTuple{5,Float32}, xc::NTuple{5,Float32},
                          xp::NTuple{5,Float32},
                          ym::NTuple{5,Float32}, yp::NTuple{5,Float32},
                          zm::NTuple{5,Float32}, zp::NTuple{5,Float32},
                          Xxm::Float32, Xc::Float32, Xxp::Float32,
                          Xym::Float32, Xyp::Float32,
                          Xzm::Float32, Xzp::Float32,
                          λ::Float32, γ::Float32)
    acc = 0.0f0
    sX = _minmod(Xc - Xxm, Xxp - Xc)
    acc += _ctu_flux5(_recon_one(xm, xc, xp,  1.0f0), γ, Int32(1))[1] * (Xc + 0.5f0 * sX) -
           _ctu_flux5(_recon_one(xm, xc, xp, -1.0f0), γ, Int32(1))[1] * (Xc - 0.5f0 * sX)
    sX = _minmod(Xc - Xym, Xyp - Xc)
    acc += _ctu_flux5(_recon_one(ym, xc, yp,  1.0f0), γ, Int32(2))[1] * (Xc + 0.5f0 * sX) -
           _ctu_flux5(_recon_one(ym, xc, yp, -1.0f0), γ, Int32(2))[1] * (Xc - 0.5f0 * sX)
    sX = _minmod(Xc - Xzm, Xzp - Xc)
    acc += _ctu_flux5(_recon_one(zm, xc, zp,  1.0f0), γ, Int32(3))[1] * (Xc + 0.5f0 * sX) -
           _ctu_flux5(_recon_one(zm, xc, zp, -1.0f0), γ, Int32(3))[1] * (Xc - 0.5f0 * sX)
    return acc * (-0.5f0) * λ
end

"predicted face state: one-sided PLM + the cell's dU in (ρ,ρu,ρv,ρw,ρe) lanes."
@inline function _ctu_predicted(m::NTuple{5,Float32}, c::NTuple{5,Float32},
                                p::NTuple{5,Float32}, dU::NTuple{5,Float32},
                                side::Float32)
    Wf = _recon_one(m, c, p, side)
    ρ  = Wf[1]
    U1 = ρ + dU[1]
    U2 = ρ * Wf[2] + dU[2]; U3 = ρ * Wf[3] + dU[3]; U4 = ρ * Wf[4] + dU[4]
    U5 = ρ * Wf[5] + dU[5]
    ρn = max(U1, 1.0f-30); inv = 1.0f0 / ρn
    return (ρn, U2 * inv, U3 * inv, U4 * inv, max(U5, 1.0f-30) * inv)
end

"corrector face flux: HLLC on the predicted prims + the ge lane on the mass flux."
@inline function _ctu_face(L::NTuple{5,Float32}, R::NTuple{5,Float32},
                           γ::Float32, ax::Int32)
    F, _ = _hllc(_W5(L, γ), _W5(R, γ), γ, ax)
    ge = F[1] >= 0.0f0 ? L[5] : R[5]
    return F, F[1] * ge, F[1]
end

"the species corrector lane: predicted-face X = (ρ_face·X_face + dUX)/ρ_pred, upwinded."
@inline function _ctu_faceX(Fm::Float32, ρpL::Float32, ρpR::Float32,
                            ρfL::Float32, ρfR::Float32,
                            XmL::Float32, XcL::Float32, XcR::Float32, XpR::Float32,
                            dUXL::Float32, dUXR::Float32)
    XfL = (ρfL * (XcL + 0.5f0 * _minmod(XcL - XmL, XcR - XcL)) + dUXL) / max(ρpL, 1.0f-30)
    XfR = (ρfR * (XcR - 0.5f0 * _minmod(XcR - XcL, XpR - XcR)) + dUXR) / max(ρpR, 1.0f-30)
    return Fm * (Fm >= 0.0f0 ? XfL : XfR)
end

# ── pool gather with the f16 hat-tile rounding (oracle + capture path) ────────
"tile-rounded physical (ρ,u,v,w,e): the EXACT value the f16 hat tile yields."
@inline function _ctu_prims_pool(D, S1, S2, S3, Tau, Ge, idx::Int32,
                                 dsc::Float32, ssc::Float32, esc::Float32)
    U = _loadU(D, S1, S2, S3, Tau, Ge, idx, dsc, ssc, esc)
    ρ = max(U[1], 1.0f-30); inv = 1.0f0 / ρ
    ρ̂ = Float32(Float16(U[1] / dsc))
    u = Float32(Float16(U[2] * inv)); v = Float32(Float16(U[3] * inv))
    w = Float32(Float16(U[4] * inv))
    ê = Float32(Float16(max(U[6], 1.0f-30) / esc / max(U[1] / dsc, 1.0f-30)))
    return (max(ρ̂ * dsc, 1.0f-30), u, v, w, ê * (esc / dsc))
end

# corrector flux (5 lanes + ge) between local cell (i,j,k) and its +ax neighbour,
# fully recomputed from the pool — the reflux capture chain.
@inline function _ctu_face6(D, S1, S2, S3, Tau, Ge, base::Int32,
                            i::Int32, j::Int32, k::Int32, nd::Int32,
                            λ::Float32, γ::Float32, ax::Int32,
                            dsc::Float32, ssc::Float32, esc::Float32)
    di = ax == Int32(1) ? Int32(1) : Int32(0)
    dj = ax == Int32(2) ? Int32(1) : Int32(0)
    dk = ax == Int32(3) ? Int32(1) : Int32(0)
    P(ii, jj, kk) = _ctu_prims_pool(D, S1, S2, S3, Tau, Ge,
                                    base + _lidx(ii, jj, kk, nd), dsc, ssc, esc)
    cL = P(i, j, k); cR = P(i + di, j + dj, k + dk)
    dUL = _ctu_dU5(P(i - Int32(1), j, k), cL, P(i + Int32(1), j, k),
                   P(i, j - Int32(1), k), P(i, j + Int32(1), k),
                   P(i, j, k - Int32(1)), P(i, j, k + Int32(1)), λ, γ)
    i2 = i + di; j2 = j + dj; k2 = k + dk
    dUR = _ctu_dU5(P(i2 - Int32(1), j2, k2), cR, P(i2 + Int32(1), j2, k2),
                   P(i2, j2 - Int32(1), k2), P(i2, j2 + Int32(1), k2),
                   P(i2, j2, k2 - Int32(1)), P(i2, j2, k2 + Int32(1)), λ, γ)
    L = _ctu_predicted(P(i - di, j - dj, k - dk), cL, cR, dUL, 1.0f0)
    R = _ctu_predicted(cL, cR, P(i2 + di, j2 + dj, k2 + dk), dUR, -1.0f0)
    F, Fge, _ = _ctu_face(L, R, γ, ax)
    return F, Fge
end

# ── the per-cell CTU kernel (CPU+GPU oracle) ──────────────────────────────────
# One thread per active cell; every face flux recomputed through the same chain
# the tiled kernel uses (redundantly — this is the correctness path, not perf).
@kernel function _ctu_cell_k!(Do_, S1o_, S2o_, S3o_, Tauo_, Geo_, spout,
                              @Const(D), @Const(S1), @Const(S2), @Const(S3),
                              @Const(Tau), @Const(Ge), spin,
                              @Const(live_d), @Const(Dsc), @Const(Ssc), @Const(Esc),
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
        dsc = Dsc[slot]; ssc = Ssc[slot]; esc = Esc[slot]
        i = c % B + ng; j = (c ÷ B) % B + ng; k = c ÷ (B * B) + ng
        idx = base + _lidx(i, j, k, nd)
        P(ii, jj, kk) = _ctu_prims_pool(D, S1, S2, S3, Tau, Ge,
                                        base + _lidx(ii, jj, kk, nd), dsc, ssc, esc)
        a1 = 0.0f0; a2 = 0.0f0; a3 = 0.0f0; a4 = 0.0f0; a5 = 0.0f0; a6 = 0.0f0
        aX1 = 0.0f0; aX2 = 0.0f0
        for ax in Int32(1):Int32(3)
            di = ax == Int32(1) ? Int32(1) : Int32(0)
            dj = ax == Int32(2) ? Int32(1) : Int32(0)
            dk = ax == Int32(3) ? Int32(1) : Int32(0)
            # stash the lo face, then add (F_hi − F_lo) in ONE difference — the
            # SAME rounding order the tiled kernel's plane sweep uses
            # (a1 += Fs[hi] − Fs[lo]); two signed adds round differently.
            F1l = 0.0f0; F2l = 0.0f0; F3l = 0.0f0; F4l = 0.0f0; F5l = 0.0f0
            F6l = 0.0f0; fX1l = 0.0f0; fX2l = 0.0f0
            for s in (Int32(-1), Int32(0))                    # lo face (cell−1,cell), hi face
                fi = i + s * di; fj = j + s * dj; fk = k + s * dk
                cL = P(fi, fj, fk); cR = P(fi + di, fj + dj, fk + dk)
                dUL = _ctu_dU5(P(fi - Int32(1), fj, fk), cL, P(fi + Int32(1), fj, fk),
                               P(fi, fj - Int32(1), fk), P(fi, fj + Int32(1), fk),
                               P(fi, fj, fk - Int32(1)), P(fi, fj, fk + Int32(1)), λ, γ)
                gi = fi + di; gj = fj + dj; gk = fk + dk
                dUR = _ctu_dU5(P(gi - Int32(1), gj, gk), cR, P(gi + Int32(1), gj, gk),
                               P(gi, gj - Int32(1), gk), P(gi, gj + Int32(1), gk),
                               P(gi, gj, gk - Int32(1)), P(gi, gj, gk + Int32(1)), λ, γ)
                L = _ctu_predicted(P(fi - di, fj - dj, fk - dk), cL, cR, dUL, 1.0f0)
                R = _ctu_predicted(cL, cR, P(gi + di, gj + dj, gk + dk), dUR, -1.0f0)
                F, Fge, Fm = _ctu_face(L, R, γ, ax)
                if s == Int32(-1)
                    F1l = F[1]; F2l = F[2]; F3l = F[3]; F4l = F[4]; F5l = F[5]
                    F6l = Fge
                else
                    a1 += F[1] - F1l; a2 += F[2] - F2l; a3 += F[3] - F3l
                    a4 += F[4] - F4l; a5 += F[5] - F5l; a6 += Fge - F6l
                end
                if NS > 0
                    ρfL = _recon_one(P(fi - di, fj - dj, fk - dk), cL, cR, 1.0f0)[1]
                    ρfR = _recon_one(cL, cR, P(gi + di, gj + dj, gk + dk), -1.0f0)[1]
                    for q in 1:NS
                        sq = spin[q]
                        XmL = decode_log2sp(Float32, sq[base + _lidx(fi - di, fj - dj, fk - dk, nd)])
                        XcL = decode_log2sp(Float32, sq[base + _lidx(fi, fj, fk, nd)])
                        XcR = decode_log2sp(Float32, sq[base + _lidx(gi, gj, gk, nd)])
                        XpR = decode_log2sp(Float32, sq[base + _lidx(gi + di, gj + dj, gk + dk, nd)])
                        dUXL = _ctu_dUX(P(fi - Int32(1), fj, fk), cL, P(fi + Int32(1), fj, fk),
                                        P(fi, fj - Int32(1), fk), P(fi, fj + Int32(1), fk),
                                        P(fi, fj, fk - Int32(1)), P(fi, fj, fk + Int32(1)),
                                        decode_log2sp(Float32, sq[base + _lidx(fi - Int32(1), fj, fk, nd)]),
                                        XcL,
                                        decode_log2sp(Float32, sq[base + _lidx(fi + Int32(1), fj, fk, nd)]),
                                        decode_log2sp(Float32, sq[base + _lidx(fi, fj - Int32(1), fk, nd)]),
                                        decode_log2sp(Float32, sq[base + _lidx(fi, fj + Int32(1), fk, nd)]),
                                        decode_log2sp(Float32, sq[base + _lidx(fi, fj, fk - Int32(1), nd)]),
                                        decode_log2sp(Float32, sq[base + _lidx(fi, fj, fk + Int32(1), nd)]),
                                        λ, γ)
                        dUXR = _ctu_dUX(P(gi - Int32(1), gj, gk), cR, P(gi + Int32(1), gj, gk),
                                        P(gi, gj - Int32(1), gk), P(gi, gj + Int32(1), gk),
                                        P(gi, gj, gk - Int32(1)), P(gi, gj, gk + Int32(1)),
                                        decode_log2sp(Float32, sq[base + _lidx(gi - Int32(1), gj, gk, nd)]),
                                        XcR,
                                        decode_log2sp(Float32, sq[base + _lidx(gi + Int32(1), gj, gk, nd)]),
                                        decode_log2sp(Float32, sq[base + _lidx(gi, gj - Int32(1), gk, nd)]),
                                        decode_log2sp(Float32, sq[base + _lidx(gi, gj + Int32(1), gk, nd)]),
                                        decode_log2sp(Float32, sq[base + _lidx(gi, gj, gk - Int32(1), nd)]),
                                        decode_log2sp(Float32, sq[base + _lidx(gi, gj, gk + Int32(1), nd)]),
                                        λ, γ)
                        fX = _ctu_faceX(Fm, L[1], R[1], ρfL, ρfR, XmL, XcL, XcR, XpR, dUXL, dUXR)
                        if s == Int32(-1)
                            q == 1 && (fX1l = fX)
                            q == 2 && (fX2l = fX)
                        else
                            q == 1 && (aX1 += fX - fX1l)
                            q == 2 && (aX2 += fX - fX2l)
                        end
                    end
                end
            end
        end
        # epilogue (shared with the tiled kernel by construction)
        U0 = _loadU(D, S1, S2, S3, Tau, Ge, idx, dsc, ssc, esc)
        Wc = P(i, j, k)
        divv = 0.5f0 * ((P(i + Int32(1), j, k)[2] - P(i - Int32(1), j, k)[2]) +
                        (P(i, j + Int32(1), k)[3] - P(i, j - Int32(1), k)[3]) +
                        (P(i, j, k + Int32(1))[4] - P(i, j, k - Int32(1))[4]))
        P_c = (γ - 1.0f0) * max(Wc[1] * Wc[5], 1.0f-30)
        nD  = max(U0[1] - λ * a1, 1.0f-30)
        nS1 = U0[2] - λ * a2
        nS2 = U0[3] - λ * a3
        nS3 = U0[4] - λ * a4
        nT  = U0[5] - λ * a5
        nG  = U0[6] - λ * (a6 + P_c * divv)
        KE = 0.5f0 * (nS1 * nS1 + nS2 * nS2 + nS3 * nS3) / nD
        (nT - KE) > η * nT && (nG = nT - KE)
        Do_[idx]   = _narrow(eltype(Do_), nD / dsc)
        S1o_[idx]  = _narrow(eltype(S1o_), nS1 / ssc)
        S2o_[idx]  = _narrow(eltype(S2o_), nS2 / ssc)
        S3o_[idx]  = _narrow(eltype(S3o_), nS3 / ssc)
        Tauo_[idx] = _narrow(eltype(Tauo_), nT / esc)
        Geo_[idx]  = _narrow(eltype(Geo_), max(nG, 1.0f-30) / esc)
        if NS >= 1
            Xc = decode_log2sp(Float32, spin[1][idx])
            spout[1][idx] = encode_log2sp(max(Xc * U0[1] - λ * aX1, 0.0f0) / nD)
        end
        if NS >= 2
            Xc = decode_log2sp(Float32, spin[2][idx])
            spout[2][idx] = encode_log2sp(max(Xc * U0[1] - λ * aX2, 0.0f0) / nD)
        end
    end
end

# ── the tiled GPU kernel ──────────────────────────────────────────────────────
const _TBX, _TBY, _TBZ = 8, 8, 4
const _WSX, _WSY, _WSZ = _TBX + 4, _TBY + 4, _TBZ + 4          # prim tile (±2)
const _DSX, _DSY, _DSZ = _TBX + 2, _TBY + 2, _TBZ + 2          # dU tile (±1)
const _NT   = _TBX * _TBY * _TBZ
const _WS3  = _WSX * _WSY * _WSZ
const _DS3  = _DSX * _DSY * _DSZ
const _FSMX = max((_TBX + 1) * _TBY * _TBZ, _TBX * (_TBY + 1) * _TBZ,
                  _TBX * _TBY * (_TBZ + 1))

@inline _wsi(lx::Int32, ly::Int32, lz::Int32) =
    (lz * Int32(_WSY) + ly) * Int32(_WSX) + lx        # 0-based
@inline _dsi(lx::Int32, ly::Int32, lz::Int32) =
    (lz * Int32(_DSY) + ly) * Int32(_DSX) + lx        # 0-based

# physical 5-lane state of W-tile cell (lx,ly,lz) from the f16 hat tile
@inline function _tld(Wt, lx::Int32, ly::Int32, lz::Int32,
                      dsc::Float32, esc::Float32)
    q = _wsi(lx, ly, lz) * Int32(5)
    @inbounds (max(Float32(Wt[q+1]) * dsc, 1.0f-30), Float32(Wt[q+2]),
               Float32(Wt[q+3]), Float32(Wt[q+4]), Float32(Wt[q+5]) * (esc / dsc))
end

@inline _xld(Xt, w0::Int32, s::Int, NS::Int) =
    @inbounds decode_log2sp(Float32, Xt[w0 * Int32(NS) + Int32(s)])

@kernel cpu=false function _ctu_step_k!(
        Do_, S1o_, S2o_, S3o_, Tauo_, Geo_, spout,
        @Const(D), @Const(S1), @Const(S2), @Const(S3), @Const(Tau), @Const(Ge), spin,
        @Const(live_d), @Const(Dsc), @Const(Ssc), @Const(Esc),
        λ::Float32, γ::Float32, η::Float32, tpb::Int32,
        B::Int32, ng::Int32, nd::Int32, stride::Int32, ::Val{NS}) where {NS}
    Wt  = @localmem Float16 (_WS3 * 5)
    Xt  = @localmem UInt16 (_WS3 * max(NS, 1))
    dUs = @localmem Float32 (_DS3 * (5 + NS))
    Fs  = @localmem Float32 (_FSMX * (6 + NS))
    tid = Int32(@index(Local, Linear)) - Int32(1)              # 0-based
    g   = Int32(@index(Group, Linear)) - Int32(1)
    bi  = g ÷ tpb; tile = g % tpb
    ntx = B ÷ Int32(_TBX); nty = B ÷ Int32(_TBY)
    tx0 = (tile % ntx) * Int32(_TBX)                           # tile origin, active frame
    ty0 = ((tile ÷ ntx) % nty) * Int32(_TBY)
    tz0 = (tile ÷ (ntx * nty)) * Int32(_TBZ)
    slot = @inbounds live_d[bi + Int32(1)]
    base = (slot - Int32(1)) * stride
    dsc = @inbounds Dsc[slot]; ssc = @inbounds Ssc[slot]; esc = @inbounds Esc[slot]
    NL = Int32(5 + NS)
    # ── phase 0: pool U → physical prims → f16 hat tile (+ u16 species) ──────
    let t = tid
        while t < Int32(_WS3)
            lx = t % Int32(_WSX); ly = (t ÷ Int32(_WSX)) % Int32(_WSY)
            lz = t ÷ Int32(_WSX * _WSY)
            idx = base + _lidx(tx0 + lx + ng - Int32(2), ty0 + ly + ng - Int32(2),
                               tz0 + lz + ng - Int32(2), nd)
            U = _loadU(D, S1, S2, S3, Tau, Ge, idx, dsc, ssc, esc)
            ρ = max(U[1], 1.0f-30); inv = 1.0f0 / ρ
            q = t * Int32(5)
            @inbounds begin
                Wt[q+1] = Float16(U[1] / dsc)
                Wt[q+2] = Float16(U[2] * inv)
                Wt[q+3] = Float16(U[3] * inv)
                Wt[q+4] = Float16(U[4] * inv)
                Wt[q+5] = Float16(max(U[6], 1.0f-30) / esc / max(U[1] / dsc, 1.0f-30))
                for s in 1:NS
                    Xt[t*Int32(NS)+Int32(s)] = spin[s][idx]
                end
            end
            t += Int32(_NT)
        end
    end
    @synchronize
    # ── phase 1: predictor dU per (+1)-halo cell, f32 shared ─────────────────
    let t = tid
        while t < Int32(_DS3)
            dx = t % Int32(_DSX); dy = (t ÷ Int32(_DSX)) % Int32(_DSY)
            dz = t ÷ Int32(_DSX * _DSY)
            lx = dx + Int32(1); ly = dy + Int32(1); lz = dz + Int32(1)  # W coords
            xc = _tld(Wt, lx, ly, lz, dsc, esc)
            xm = _tld(Wt, lx - Int32(1), ly, lz, dsc, esc)
            xp = _tld(Wt, lx + Int32(1), ly, lz, dsc, esc)
            ym = _tld(Wt, lx, ly - Int32(1), lz, dsc, esc)
            yp = _tld(Wt, lx, ly + Int32(1), lz, dsc, esc)
            zm = _tld(Wt, lx, ly, lz - Int32(1), dsc, esc)
            zp = _tld(Wt, lx, ly, lz + Int32(1), dsc, esc)
            dU = _ctu_dU5(xm, xc, xp, ym, yp, zm, zp, λ, γ)
            q = t * NL
            @inbounds begin
                dUs[q+1] = dU[1]; dUs[q+2] = dU[2]; dUs[q+3] = dU[3]
                dUs[q+4] = dU[4]; dUs[q+5] = dU[5]
                for s in 1:NS
                    w0 = _wsi(lx, ly, lz)
                    dUs[q+Int32(5)+Int32(s)] = _ctu_dUX(xm, xc, xp, ym, yp, zm, zp,
                        _xld(Xt, _wsi(lx - Int32(1), ly, lz), s, NS),
                        _xld(Xt, w0, s, NS),
                        _xld(Xt, _wsi(lx + Int32(1), ly, lz), s, NS),
                        _xld(Xt, _wsi(lx, ly - Int32(1), lz), s, NS),
                        _xld(Xt, _wsi(lx, ly + Int32(1), lz), s, NS),
                        _xld(Xt, _wsi(lx, ly, lz - Int32(1)), s, NS),
                        _xld(Xt, _wsi(lx, ly, lz + Int32(1)), s, NS), λ, γ)
                end
            end
            t += Int32(_NT)
        end
    end
    @synchronize
    # ── phase 2: three face-plane sweeps, ΔF per target cell in registers ────
    tx = tid % Int32(_TBX); ty = (tid ÷ Int32(_TBX)) % Int32(_TBY)
    tz = tid ÷ Int32(_TBX * _TBY)
    a1 = 0.0f0; a2 = 0.0f0; a3 = 0.0f0; a4 = 0.0f0; a5 = 0.0f0; a6 = 0.0f0
    aX1 = 0.0f0; aX2 = 0.0f0
    for dir in Int32(1):Int32(3)
        ex = dir == Int32(1) ? Int32(1) : Int32(0)
        ey = dir == Int32(2) ? Int32(1) : Int32(0)
        ez = dir == Int32(3) ? Int32(1) : Int32(0)
        nfx = Int32(_TBX) + ex; nfy = Int32(_TBY) + ey; nfz = Int32(_TBZ) + ez
        nf  = nfx * nfy * nfz
        let p = tid
            while p < nf
                fx = p % nfx; fy = (p ÷ nfx) % nfy; fz = p ÷ (nfx * nfy)
                lx = fx + Int32(2) - ex; ly = fy + Int32(2) - ey; lz = fz + Int32(2) - ez
                mx = lx + ex; my = ly + ey; mz = lz + ez
                dL = _dsi(lx - Int32(1), ly - Int32(1), lz - Int32(1)) * NL
                dR = _dsi(mx - Int32(1), my - Int32(1), mz - Int32(1)) * NL
                @inbounds begin
                    dUL = (dUs[dL+1], dUs[dL+2], dUs[dL+3], dUs[dL+4], dUs[dL+5])
                    dUR = (dUs[dR+1], dUs[dR+2], dUs[dR+3], dUs[dR+4], dUs[dR+5])
                end
                cm = _tld(Wt, lx - ex, ly - ey, lz - ez, dsc, esc)
                cL = _tld(Wt, lx, ly, lz, dsc, esc)
                cR = _tld(Wt, mx, my, mz, dsc, esc)
                cp = _tld(Wt, mx + ex, my + ey, mz + ez, dsc, esc)
                L = _ctu_predicted(cm, cL, cR, dUL, 1.0f0)
                R = _ctu_predicted(cL, cR, cp, dUR, -1.0f0)
                F, Fge, Fm = _ctu_face(L, R, γ, dir)
                q = p * (Int32(6) + Int32(NS))
                @inbounds begin
                    Fs[q+1] = F[1]; Fs[q+2] = F[2]; Fs[q+3] = F[3]
                    Fs[q+4] = F[4]; Fs[q+5] = F[5]; Fs[q+6] = Fge
                    if NS > 0
                        ρfL = _recon_one(cm, cL, cR, 1.0f0)[1]
                        ρfR = _recon_one(cL, cR, cp, -1.0f0)[1]
                        for s in 1:NS
                            Fs[q+Int32(6)+Int32(s)] = _ctu_faceX(Fm, L[1], R[1], ρfL, ρfR,
                                _xld(Xt, _wsi(lx - ex, ly - ey, lz - ez), s, NS),
                                _xld(Xt, _wsi(lx, ly, lz), s, NS),
                                _xld(Xt, _wsi(mx, my, mz), s, NS),
                                _xld(Xt, _wsi(mx + ex, my + ey, mz + ez), s, NS),
                                dUs[dL+Int32(5)+Int32(s)], dUs[dR+Int32(5)+Int32(s)])
                        end
                    end
                end
                p += Int32(_NT)
            end
        end
        @synchronize
        flo = ((tz * nfy + ty) * nfx + tx) * (Int32(6) + Int32(NS))
        fhi = (((tz + ez) * nfy + (ty + ey)) * nfx + (tx + ex)) * (Int32(6) + Int32(NS))
        @inbounds begin
            a1 += Fs[fhi+1] - Fs[flo+1]; a2 += Fs[fhi+2] - Fs[flo+2]
            a3 += Fs[fhi+3] - Fs[flo+3]; a4 += Fs[fhi+4] - Fs[flo+4]
            a5 += Fs[fhi+5] - Fs[flo+5]; a6 += Fs[fhi+6] - Fs[flo+6]
            NS >= 1 && (aX1 += Fs[fhi+7] - Fs[flo+7])
            NS >= 2 && (aX2 += Fs[fhi+8] - Fs[flo+8])
        end
        @synchronize
    end
    # ── epilogue ──────────────────────────────────────────────────────────────
    i = tx0 + tx + ng; j = ty0 + ty + ng; k = tz0 + tz + ng
    idx = base + _lidx(i, j, k, nd)
    @inbounds begin
        U0 = _loadU(D, S1, S2, S3, Tau, Ge, idx, dsc, ssc, esc)
        wx = tx + Int32(2); wy = ty + Int32(2); wz = tz + Int32(2)
        Wc = _tld(Wt, wx, wy, wz, dsc, esc)
        divv = 0.5f0 * ((_tld(Wt, wx + Int32(1), wy, wz, dsc, esc)[2] -
                         _tld(Wt, wx - Int32(1), wy, wz, dsc, esc)[2]) +
                        (_tld(Wt, wx, wy + Int32(1), wz, dsc, esc)[3] -
                         _tld(Wt, wx, wy - Int32(1), wz, dsc, esc)[3]) +
                        (_tld(Wt, wx, wy, wz + Int32(1), dsc, esc)[4] -
                         _tld(Wt, wx, wy, wz - Int32(1), dsc, esc)[4]))
        P_c = (γ - 1.0f0) * max(Wc[1] * Wc[5], 1.0f-30)
        nD  = max(U0[1] - λ * a1, 1.0f-30)
        nS1 = U0[2] - λ * a2
        nS2 = U0[3] - λ * a3
        nS3 = U0[4] - λ * a4
        nT  = U0[5] - λ * a5
        nG  = U0[6] - λ * (a6 + P_c * divv)
        KE = 0.5f0 * (nS1 * nS1 + nS2 * nS2 + nS3 * nS3) / nD
        (nT - KE) > η * nT && (nG = nT - KE)
        Do_[idx]   = _narrow(eltype(Do_), nD / dsc)
        S1o_[idx]  = _narrow(eltype(S1o_), nS1 / ssc)
        S2o_[idx]  = _narrow(eltype(S2o_), nS2 / ssc)
        S3o_[idx]  = _narrow(eltype(S3o_), nS3 / ssc)
        Tauo_[idx] = _narrow(eltype(Tauo_), nT / esc)
        Geo_[idx]  = _narrow(eltype(Geo_), max(nG, 1.0f-30) / esc)
        if NS >= 1
            Xc = decode_log2sp(Float32, spin[1][idx])
            spout[1][idx] = encode_log2sp(max(Xc * U0[1] - λ * aX1, 0.0f0) / nD)
        end
        if NS >= 2
            Xc = decode_log2sp(Float32, spin[2][idx])
            spout[2][idx] = encode_log2sp(max(Xc * U0[1] - λ * aX2, 0.0f0) / nD)
        end
    end
end

"""
    ctu_level!(hier, l, λ; η = 1e-3, tiled = nothing)

One full CTU substep of level `l`: R → O in one launch (the caller swaps
buffers).  `tiled = nothing` auto-selects: the shared-memory kernel on GPU
backends when `B` is (8,8,4)-tileable and `nsp ≤ 2`, the per-cell oracle
otherwise (always on CPU).
"""
function ctu_level!(hier::AMRHierarchy, l::Int, λ::Float32;
                    η::Float32 = 1.0f-3, tiled::Union{Nothing,Bool} = nothing)
    lev = hier.levels[l + 1]
    isempty(lev.live) && return nothing
    B = lev.B; NS = lev.nsp
    @assert NS <= 2 "the CTU kernels carry ≤ 2 species lanes (got $NS)"
    can_tile = hier.besym != :cpu && B % _TBX == 0 && B % _TBY == 0 && B % _TBZ == 0
    usetile = tiled === nothing ? can_tile : (tiled && can_tile)
    sin_  = ntuple(q -> lev.sp[q], NS)
    sout_ = ntuple(q -> lev.spo[q], NS)
    if usetile
        tpb = (B ÷ _TBX) * (B ÷ _TBY) * (B ÷ _TBZ)
        n = length(lev.live) * tpb * _NT
        _ctu_step_k!(lev.be, _NT)(gasfields_o(lev)..., sout_, gasfields(lev)..., sin_,
                                  lev.live_d, lev.Dsc, lev.Ssc, lev.Esc,
                                  λ, Float32(hier.gamma), η, Int32(tpb),
                                  Int32(B), Int32(lev.ng), Int32(lev.nd),
                                  Int32(lev.stride), Val(NS); ndrange = n)
    else
        n = length(lev.live) * B^3
        _ctu_cell_k!(lev.be)(gasfields_o(lev)..., sout_, gasfields(lev)..., sin_,
                             lev.live_d, lev.Dsc, lev.Ssc, lev.Esc,
                             λ, Float32(hier.gamma), η,
                             Int32(B), Int32(lev.ng), Int32(lev.nd),
                             Int32(lev.stride), Val(NS); ndrange = n)
    end
    return nothing
end
