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
    # predictor fallback: the transverse half-step is UNLIMITED — at strong
    # shocks it can drive ρ/ρe near or through zero, and an absolute floor
    # then manufactures ~1e30 velocities that NaN the Riemann solve (the
    # bamr256L10 level-2 seed, 482 cells in one substep).  If the half-step
    # removes >90% of density or internal energy, use the un-evolved face
    # state instead (locally first-order in time — the Enzo/Athena cure).
    if !(U1 > 0.1f0 * ρ) || !(U5 > 0.1f0 * ρ * Wf[5])
        return Wf
    end
    inv = 1.0f0 / U1
    return (U1, U2 * inv, U3 * inv, U4 * inv, U5 * inv)
end

"corrector face flux: HLLC on the predicted prims + the ge lane on the mass flux
+ the contact speed Ss (the pdV face velocity — Enzo flux_twoshock's ub)."
@inline function _ctu_face(L::NTuple{5,Float32}, R::NTuple{5,Float32},
                           γ::Float32, ax::Int32)
    F, Ss = _hllc(_W5(L, γ), _W5(R, γ), γ, ax)
    ge = F[1] >= 0.0f0 ? L[5] : R[5]
    return F, F[1] * ge, F[1], Ss
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

# Enzo pgas2d_dual `demax`: the max total-energy density over the cell and its 6
# face neighbours (input R-buffer, physical units).  The dual-energy resync
# Ge ← Tau−KE is gated on (Tau−KE) > η₂·demax with η₂ ~ 0.1 — only a thermal
# energy that is a significant fraction of the LOCAL energy scale (a resolved
# shock) may overwrite the advected internal energy.  Unresolved sub-grid
# velocity dispersion (restriction/C-F noise, ~1% of E) must NOT become heat —
# the old local gate ((Tau−KE) > 1e-3·Tau) let exactly that through and drove
# the static-level runaway (uncovered coarse cells ×172 in 10 steps).
@inline function _demax6(Tau, base::Int32, i::Int32, j::Int32, k::Int32,
                         nd::Int32, esc::Float32, kz::Int32 = Int32(1))
    m = abs(Float32(Tau[base + _lidx(i, j, k, nd)]))
    m = max(m, abs(Float32(Tau[base + _lidx(i - Int32(1), j, k, nd)])))
    m = max(m, abs(Float32(Tau[base + _lidx(i + Int32(1), j, k, nd)])))
    m = max(m, abs(Float32(Tau[base + _lidx(i, j - Int32(1), k, nd)])))
    m = max(m, abs(Float32(Tau[base + _lidx(i, j + Int32(1), k, nd)])))
    m = max(m, abs(Float32(Tau[base + _lidx(i, j, k - kz, nd)])))    # 2-D: self
    m = max(m, abs(Float32(Tau[base + _lidx(i, j, k + kz, nd)])))
    return m * esc
end

# ── pool gather with the f16 hat-tile rounding (oracle + capture path) ────────
"tile-rounded physical (ρ,u,v,w,e): the EXACT value the f16 hat tile yields."
@inline function _ctu_prims_pool(D, S1, S2, S3, Tau, Ge, idx::Int32,
                                 dsc::Float32, ssc::Float32, esc::Float32, gsc::Float32,
                                 vb1::Float32 = 0.0f0, vb2::Float32 = 0.0f0,
                                 vb3::Float32 = 0.0f0)
    U = _loadU(D, S1, S2, S3, Tau, Ge, idx, dsc, ssc, esc, gsc)
    # vacuum gate — must mirror the tiled kernel's phase-0 load bit-exactly.
    ρ = max(U[1], 1.0f-30)
    inv = ifelse(U[1] > 0.0f0, 1.0f0 / ρ, 0.0f0)
    ρ̂ = Float32(Float16(U[1] / dsc))
    live = ρ̂ > 0.0f0
    # bulk-frame f16 rounding + lab lift — replicates the tile (_tld) exactly
    u = ifelse(live, Float32(Float16(U[2] * inv - vb1)) + vb1, 0.0f0)
    v = ifelse(live, Float32(Float16(U[3] * inv - vb2)) + vb2, 0.0f0)
    w = ifelse(live, Float32(Float16(U[4] * inv - vb3)) + vb3, 0.0f0)
    # internal energy rides ITS OWN scale gsc (U[6]=Ge·gsc) — the tile lift factor
    # is gsc/dsc so the f16 tile windows the true internal energy, not Tau's.
    ê = Float32(Float16(ifelse(U[1] > 0.0f0,
                               max(U[6], 1.0f-30) / gsc /
                               max(U[1] / dsc, 1.0f-30), 0.0f0)))
    return (max(ρ̂ * dsc, 1.0f-30), u, v, w, ê * (gsc / dsc))
end

# corrector flux (5 lanes + ge) between local cell (i,j,k) and its +ax neighbour,
# fully recomputed from the pool — the reflux capture chain.
@inline function _ctu_face6(D, S1, S2, S3, Tau, Ge, base::Int32,
                            i::Int32, j::Int32, k::Int32, nd::Int32,
                            λ::Float32, γ::Float32, ax::Int32,
                            dsc::Float32, ssc::Float32, esc::Float32, gsc::Float32,
                            vb1::Float32 = 0.0f0, vb2::Float32 = 0.0f0,
                            vb3::Float32 = 0.0f0, kz::Int32 = Int32(1))
    di = ax == Int32(1) ? Int32(1) : Int32(0)
    dj = ax == Int32(2) ? Int32(1) : Int32(0)
    dk = ax == Int32(3) ? Int32(1) : Int32(0)
    P(ii, jj, kk) = _ctu_prims_pool(D, S1, S2, S3, Tau, Ge,
                                    base + _lidx(ii, jj, kk, nd), dsc, ssc, esc, gsc,
                                    vb1, vb2, vb3)
    cL = P(i, j, k); cR = P(i + di, j + dj, k + dk)
    dUL = _ctu_dU5(P(i - Int32(1), j, k), cL, P(i + Int32(1), j, k),
                   P(i, j - Int32(1), k), P(i, j + Int32(1), k),
                   P(i, j, k - kz), P(i, j, k + kz), λ, γ)
    i2 = i + di; j2 = j + dj; k2 = k + dk
    dUR = _ctu_dU5(P(i2 - Int32(1), j2, k2), cR, P(i2 + Int32(1), j2, k2),
                   P(i2, j2 - Int32(1), k2), P(i2, j2 + Int32(1), k2),
                   P(i2, j2, k2 - kz), P(i2, j2, k2 + kz), λ, γ)
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
                              @Const(Tau), @Const(Ge), spin, spres,
                              @Const(live_d), @Const(Dsc), @Const(Ssc), @Const(Esc), @Const(Gsc),
                              @Const(Vb),
                              λ::Float32, γ::Float32, η::Float32, scap::Float32,
                              pdv::Float32,
                              B::Int32, ng::Int32, nd::Int32, stride::Int32,
                              ::Val{NS}, dimv::Val{DIM}) where {NS,DIM}
    t = @index(Global)
    t0 = Int32(t) - Int32(1)
    B3 = _nblk(B, dimv)
    bi = t0 ÷ B3; c = t0 % B3
    # 2-D: the predictor's z taps read the cell itself (zero z slope ⇒ the z
    # flux difference is exactly 0) and the corrector sweeps x, y only
    kz = _kzv(dimv)
    @inbounds begin
        slot = live_d[bi + Int32(1)]
        base = (slot - Int32(1)) * stride
        dsc = Dsc[slot]; ssc = Ssc[slot]; esc = Esc[slot]; gsc = Gsc[slot]
        vb1 = Vb[3*(slot-Int32(1))+Int32(1)]           # per-block bulk velocity (BAM_BULKV; 0 = off)
        vb2 = Vb[3*(slot-Int32(1))+Int32(2)]
        vb3 = Vb[3*(slot-Int32(1))+Int32(3)]
        i, j, k = _cell_ijk(c, B, ng, dimv)
        idx = base + _lidx(i, j, k, nd)
        P(ii, jj, kk) = _ctu_prims_pool(D, S1, S2, S3, Tau, Ge,
                                        base + _lidx(ii, jj, kk, nd), dsc, ssc, esc, gsc,
                                        vb1, vb2, vb3)
        a1 = 0.0f0; a2 = 0.0f0; a3 = 0.0f0; a4 = 0.0f0; a5 = 0.0f0; a6 = 0.0f0
        aU = 0.0f0                                    # Σ_ax (Ss_hi − Ss_lo): pdV face-velocity div
        aX1 = 0.0f0; aX2 = 0.0f0
        for ax in Int32(1):Int32(DIM)
            di = ax == Int32(1) ? Int32(1) : Int32(0)
            dj = ax == Int32(2) ? Int32(1) : Int32(0)
            dk = ax == Int32(3) ? Int32(1) : Int32(0)
            # stash the lo face, then add (F_hi − F_lo) in ONE difference — the
            # SAME rounding order the tiled kernel's plane sweep uses
            # (a1 += Fs[hi] − Fs[lo]); two signed adds round differently.
            F1l = 0.0f0; F2l = 0.0f0; F3l = 0.0f0; F4l = 0.0f0; F5l = 0.0f0
            F6l = 0.0f0; Ssl = 0.0f0; fX1l = 0.0f0; fX2l = 0.0f0
            for s in (Int32(-1), Int32(0))                    # lo face (cell−1,cell), hi face
                fi = i + s * di; fj = j + s * dj; fk = k + s * dk
                cL = P(fi, fj, fk); cR = P(fi + di, fj + dj, fk + dk)
                dUL = _ctu_dU5(P(fi - Int32(1), fj, fk), cL, P(fi + Int32(1), fj, fk),
                               P(fi, fj - Int32(1), fk), P(fi, fj + Int32(1), fk),
                               P(fi, fj, fk - kz), P(fi, fj, fk + kz), λ, γ)
                gi = fi + di; gj = fj + dj; gk = fk + dk
                dUR = _ctu_dU5(P(gi - Int32(1), gj, gk), cR, P(gi + Int32(1), gj, gk),
                               P(gi, gj - Int32(1), gk), P(gi, gj + Int32(1), gk),
                               P(gi, gj, gk - kz), P(gi, gj, gk + kz), λ, γ)
                L = _ctu_predicted(P(fi - di, fj - dj, fk - dk), cL, cR, dUL, 1.0f0)
                R = _ctu_predicted(cL, cR, P(gi + di, gj + dj, gk + dk), dUR, -1.0f0)
                F, Fge, Fm, Ssf = _ctu_face(L, R, γ, ax)
                if s == Int32(-1)
                    F1l = F[1]; F2l = F[2]; F3l = F[3]; F4l = F[4]; F5l = F[5]
                    F6l = Fge; Ssl = Ssf
                else
                    a1 += F[1] - F1l; a2 += F[2] - F2l; a3 += F[3] - F3l
                    a4 += F[4] - F4l; a5 += F[5] - F5l; a6 += Fge - F6l
                    aU += Ssf - Ssl
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
                                        P(fi, fj, fk - kz), P(fi, fj, fk + kz),
                                        decode_log2sp(Float32, sq[base + _lidx(fi - Int32(1), fj, fk, nd)]),
                                        XcL,
                                        decode_log2sp(Float32, sq[base + _lidx(fi + Int32(1), fj, fk, nd)]),
                                        decode_log2sp(Float32, sq[base + _lidx(fi, fj - Int32(1), fk, nd)]),
                                        decode_log2sp(Float32, sq[base + _lidx(fi, fj + Int32(1), fk, nd)]),
                                        decode_log2sp(Float32, sq[base + _lidx(fi, fj, fk - kz, nd)]),
                                        decode_log2sp(Float32, sq[base + _lidx(fi, fj, fk + kz, nd)]),
                                        λ, γ)
                        dUXR = _ctu_dUX(P(gi - Int32(1), gj, gk), cR, P(gi + Int32(1), gj, gk),
                                        P(gi, gj - Int32(1), gk), P(gi, gj + Int32(1), gk),
                                        P(gi, gj, gk - kz), P(gi, gj, gk + kz),
                                        decode_log2sp(Float32, sq[base + _lidx(gi - Int32(1), gj, gk, nd)]),
                                        XcR,
                                        decode_log2sp(Float32, sq[base + _lidx(gi + Int32(1), gj, gk, nd)]),
                                        decode_log2sp(Float32, sq[base + _lidx(gi, gj - Int32(1), gk, nd)]),
                                        decode_log2sp(Float32, sq[base + _lidx(gi, gj + Int32(1), gk, nd)]),
                                        decode_log2sp(Float32, sq[base + _lidx(gi, gj, gk - kz, nd)]),
                                        decode_log2sp(Float32, sq[base + _lidx(gi, gj, gk + kz, nd)]),
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
        # epilogue (shared with the tiled kernel by construction).  pdV work
        # from the HLLC face contact-speed divergence aU (Enzo flux_twoshock /
        # the RK2 kernel's Sshi−Sslo) — see the tiled epilogue comment.
        U0 = _loadU(D, S1, S2, S3, Tau, Ge, idx, dsc, ssc, esc, gsc)
        Wc = P(i, j, k)
        P_c = (γ - 1.0f0) * max(Wc[1] * Wc[5], 1.0f-30)
        nD  = max(U0[1] - λ * a1, 1.0f-30)
        nS1 = U0[2] - λ * a2
        nS2 = U0[3] - λ * a3
        nS3 = U0[4] - λ * a4
        nT  = U0[5] - λ * a5
        nG  = U0[6] - λ * (a6 + pdv * P_c * aU)       # pdv: BAM_NO_PDV diag (0 zeros pdV work)
        KE = 0.5f0 * (nS1 * nS1 + nS2 * nS2 + nS3 * nS3) / nD
        demax = _demax6(Tau, base, i, j, k, nd, esc, kz)
        (nT - KE) > η * demax && (nG = nT - KE)   # Enzo eta2 gate (η vs NEIGHBOR-MAX total)
        # extreme-frontier safety: a degenerate Riemann state can still emit a
        # nonfinite lane (~1 cell/Mcell-step at the collapse frontier).  Keep
        # the OLD state for that cell/substep instead of poisoning the level.
        ok = isfinite(nD) & isfinite(nS1) & isfinite(nS2) & isfinite(nS3) &
             isfinite(nT) & isfinite(nG) & isfinite(aX1) & isfinite(aX2) &
             (nD > 0.0f0) & (nT > 0.0f0)   # positivity: huge-but-FINITE fluxes
                                           # must not drive ρ/E through zero
        # CFL self-consistency cap: λ was chosen so (|v|+cs)·λ ≤ cfl·safety
        # (~0.12) for every real cell.  An update whose NEW signal speed
        # exceeds 1/λ (8× headroom) broke the assumption it was built on —
        # a degenerate flux wrote garbage-but-finite v or e.  Keep old state.
        invD = 1.0f0 / max(nD, 1.0f-30)
        vm = max(abs(nS1), max(abs(nS2), abs(nS3))) * invD
        cn = sqrt(γ * (γ - 1.0f0) * max(nG, 1.0f-30) * invD)
        ok &= (vm + cn) * λ < 1.0f0
        # ABSOLUTE signal cap (BAM_SMAXCAP; 0 = off): the λ-relative bound above
        # ESCALATES under pathology — a hot injection shrinks the global λ,
        # which widens 1/λ for the next substep (observed: a zoom-clipped
        # secondary knot ratcheting C/F skin cells from 15 km/s to 1e18 K over
        # ~40 substeps, strangling dτ 470×).  An absolute bound anchored to the
        # problem's physical signal scale breaks the ratchet.
        ok &= (scap <= 0.0f0) | (vm + cn < scap)
        if !ok
            nD = U0[1]; nS1 = U0[2]; nS2 = U0[3]; nS3 = U0[4]
            nT = U0[5]; nG = U0[6]
        end
        # Enzo eta1 (pgas2d_dual eslice update): keep the CONSERVATIVE total where
        # the thermal part is locally resolved ((nT−KE) > η1·nT); rebuild
        # Tau ≡ KE + Ge only in KE-dominated cells, where the stored Tau−KE is
        # cancellation noise.  Exact conservation through resolved shocks; the
        # (deliberate, Enzo-matching) non-conservation is confined to cells with
        # thermal fraction < η1 = 1e-3.
        nG = max(nG, 1.0f-30)
        KEn = 0.5f0 * (nS1 * nS1 + nS2 * nS2 + nS3 * nS3) / max(nD, 1.0f-30)
        nT = ifelse(nT - KEn > 1.0f-3 * nT, nT, KEn + nG)
        Do_[idx]   = _narrow(eltype(Do_), nD / dsc)
        S1o_[idx]  = _narrow(eltype(S1o_), nS1 / ssc)
        S2o_[idx]  = _narrow(eltype(S2o_), nS2 / ssc)
        S3o_[idx]  = _narrow(eltype(S3o_), nS3 / ssc)
        Tauo_[idx] = _narrow(eltype(Tauo_), nT / esc)
        Geo_[idx]  = _narrow(eltype(Geo_), nG / gsc)
        if NS >= 1
            Xc = decode_log2sp(Float32, spin[1][idx])
            spout[1][idx] = ok ?
                _sd_spstore!(spres[1], idx, max(Xc * U0[1] - λ * aX1, 0.0f0),
                             max(nD, 1.0f-30)) :
                spin[1][idx]
        end
        if NS >= 2
            Xc = decode_log2sp(Float32, spin[2][idx])
            spout[2][idx] = ok ?
                _sd_spstore!(spres[2], idx, max(Xc * U0[1] - λ * aX2, 0.0f0),
                             max(nD, 1.0f-30)) :
                spin[2][idx]
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

# 2-D tile: 16×16×1 (same 256 threads), NO z halo — the flat z axis has no
# neighbours, so the predictor's z taps read the cell itself (zero z flux).
const _TBX2, _TBY2 = 16, 16
const _WS2  = (_TBX2 + 4) * (_TBY2 + 4)
const _DS2  = (_TBX2 + 2) * (_TBY2 + 2)
const _FSMX2 = max((_TBX2 + 1) * _TBY2, _TBX2 * (_TBY2 + 1))

# (TBX, TBY, TBZ, hz): tile extent + z-halo unit (W tile ±2hz, dU tile ±hz)
@inline _tgeom(::Val{3}) = (Int32(_TBX), Int32(_TBY), Int32(_TBZ), Int32(1))
@inline _tgeom(::Val{2}) = (Int32(_TBX2), Int32(_TBY2), Int32(1), Int32(0))
@inline _ws3(::Val{3}) = _WS3;   @inline _ws3(::Val{2}) = _WS2
@inline _ds3(::Val{3}) = _DS3;   @inline _ds3(::Val{2}) = _DS2
@inline _fsmx(::Val{3}) = _FSMX; @inline _fsmx(::Val{2}) = _FSMX2

@inline _wsi(lx::Int32, ly::Int32, lz::Int32,
             wsx::Int32 = Int32(_WSX), wsy::Int32 = Int32(_WSY)) =
    (lz * wsy + ly) * wsx + lx                        # 0-based
@inline _dsi(lx::Int32, ly::Int32, lz::Int32,
             dsx::Int32 = Int32(_DSX), dsy::Int32 = Int32(_DSY)) =
    (lz * dsy + ly) * dsx + lx                        # 0-based

# physical 5-lane state of W-tile cell (lx,ly,lz) from the f16 hat tile.
# BAM_BULKV: the tile stores velocities in the per-block BULK FRAME (u − vb) so
# the f16 quantum applies to the small residual velocity, not the supersonic
# bulk (2⁻¹¹·|v_bulk| was ~3× the internal energy at KE/e ~ 2900); the lift adds
# vb back, so everything downstream is in the lab frame as before.  vb = 0
# (BULKV off) is bit-identical to the historic path (x − 0 ≡ x).
@inline function _tld(Wt, lx::Int32, ly::Int32, lz::Int32,
                      dsc::Float32, gsc::Float32,
                      vb1::Float32, vb2::Float32, vb3::Float32,
                      wsx::Int32 = Int32(_WSX), wsy::Int32 = Int32(_WSY))
    q = _wsi(lx, ly, lz, wsx, wsy) * Int32(5)
    # lane 5 is the internal energy hat, lifted by gsc/dsc (its OWN scale).
    # vacuum (ρ̂ ≤ 0) keeps v ≡ 0 exactly (the phase-0 gate staged w = −vb there).
    ρ̂ = @inbounds Float32(Wt[q+1])
    live = ρ̂ > 0.0f0
    @inbounds (max(ρ̂ * dsc, 1.0f-30),
               ifelse(live, Float32(Wt[q+2]) + vb1, 0.0f0),
               ifelse(live, Float32(Wt[q+3]) + vb2, 0.0f0),
               ifelse(live, Float32(Wt[q+4]) + vb3, 0.0f0),
               Float32(Wt[q+5]) * (gsc / dsc))
end

@inline _xld(Xt, w0::Int32, s::Int, NS::Int) =
    @inbounds decode_log2sp(Float32, Xt[w0 * Int32(NS) + Int32(s)])

@kernel cpu=false function _ctu_step_k!(
        Do_, S1o_, S2o_, S3o_, Tauo_, Geo_, spout,
        @Const(D), @Const(S1), @Const(S2), @Const(S3), @Const(Tau), @Const(Ge), spin, spres,
        @Const(live_d), @Const(Dsc), @Const(Ssc), @Const(Esc), @Const(Gsc),
        @Const(Vb),
        λ::Float32, γ::Float32, η::Float32, scap::Float32, pdv::Float32, tpb::Int32,
        B::Int32, ng::Int32, nd::Int32, stride::Int32, ::Val{NS},
        dimv::Val{DIM}) where {NS,DIM}
    Wt  = @localmem Float16 (_ws3(dimv) * 5)
    Xt  = @localmem UInt16 (_ws3(dimv) * max(NS, 1))
    dUs = @localmem Float32 (_ds3(dimv) * (5 + NS))
    Fs  = @localmem Float32 (_fsmx(dimv) * (7 + NS))  # lanes: 5 flux + ge + Ss(+NS)
    TBX, TBY, TBZ, hz = _tgeom(dimv)                   # 3-D: 8,8,4,1 · 2-D: 16,16,1,0
    WSX = TBX + Int32(4); WSY = TBY + Int32(4); WS3 = WSX * WSY * (TBZ + Int32(4) * hz)
    DSX = TBX + Int32(2); DSY = TBY + Int32(2); DS3 = DSX * DSY * (TBZ + Int32(2) * hz)
    NT  = TBX * TBY * TBZ
    tid = Int32(@index(Local, Linear)) - Int32(1)              # 0-based
    g   = Int32(@index(Group, Linear)) - Int32(1)
    bi  = g ÷ tpb; tile = g % tpb
    ntx = B ÷ TBX; nty = B ÷ TBY
    tx0 = (tile % ntx) * TBX                                   # tile origin, active frame
    ty0 = ((tile ÷ ntx) % nty) * TBY
    tz0 = (tile ÷ (ntx * nty)) * TBZ
    slot = @inbounds live_d[bi + Int32(1)]
    base = (slot - Int32(1)) * stride
    dsc = @inbounds Dsc[slot]; ssc = @inbounds Ssc[slot]; esc = @inbounds Esc[slot]; gsc = @inbounds Gsc[slot]
    vb1 = @inbounds Vb[3*(slot-Int32(1))+Int32(1)]     # per-block bulk velocity (BAM_BULKV; 0 = off)
    vb2 = @inbounds Vb[3*(slot-Int32(1))+Int32(2)]
    vb3 = @inbounds Vb[3*(slot-Int32(1))+Int32(3)]
    NL = Int32(5 + NS)
    # ── phase 0: pool U → physical prims → f16 hat tile (+ u16 species) ──────
    let t = tid
        while t < WS3
            lx = t % WSX; ly = (t ÷ WSX) % WSY
            lz = t ÷ (WSX * WSY)
            idx = base + _lidx(tx0 + lx + ng - Int32(2), ty0 + ly + ng - Int32(2),
                               tz0 + lz + hz * (ng - Int32(2)), nd)
            U = _loadU(D, S1, S2, S3, Tau, Ge, idx, dsc, ssc, esc, gsc)
            # vacuum gate (see _prim_de): stored ρ ≤ 0 ⇒ v = 0, ê = 0 — never
            # divide quantization-noise momenta by the density floor.
            ρ = max(U[1], 1.0f-30)
            inv = ifelse(U[1] > 0.0f0, 1.0f0 / ρ, 0.0f0)
            q = t * Int32(5)
            @inbounds begin
                Wt[q+1] = Float16(U[1] / dsc)
                Wt[q+2] = Float16(U[2] * inv - vb1)    # bulk frame (see _tld)
                Wt[q+3] = Float16(U[3] * inv - vb2)
                Wt[q+4] = Float16(U[4] * inv - vb3)
                # internal energy hat on gsc (U[6]=Ge·gsc) → full f16 tile precision
                Wt[q+5] = Float16(ifelse(U[1] > 0.0f0,
                                         max(U[6], 1.0f-30) / gsc /
                                         max(U[1] / dsc, 1.0f-30), 0.0f0))
                for s in 1:NS
                    Xt[t*Int32(NS)+Int32(s)] = spin[s][idx]
                end
            end
            t += NT
        end
    end
    @synchronize
    # ── phase 1: predictor dU per (+1)-halo cell, f32 shared ─────────────────
    let t = tid
        while t < DS3
            dx = t % DSX; dy = (t ÷ DSX) % DSY
            dz = t ÷ (DSX * DSY)
            lx = dx + Int32(1); ly = dy + Int32(1); lz = dz + hz        # W coords
            xc = _tld(Wt, lx, ly, lz, dsc, gsc, vb1, vb2, vb3, WSX, WSY)
            xm = _tld(Wt, lx - Int32(1), ly, lz, dsc, gsc, vb1, vb2, vb3, WSX, WSY)
            xp = _tld(Wt, lx + Int32(1), ly, lz, dsc, gsc, vb1, vb2, vb3, WSX, WSY)
            ym = _tld(Wt, lx, ly - Int32(1), lz, dsc, gsc, vb1, vb2, vb3, WSX, WSY)
            yp = _tld(Wt, lx, ly + Int32(1), lz, dsc, gsc, vb1, vb2, vb3, WSX, WSY)
            zm = _tld(Wt, lx, ly, lz - hz, dsc, gsc, vb1, vb2, vb3, WSX, WSY)
            zp = _tld(Wt, lx, ly, lz + hz, dsc, gsc, vb1, vb2, vb3, WSX, WSY)
            dU = _ctu_dU5(xm, xc, xp, ym, yp, zm, zp, λ, γ)
            q = t * NL
            @inbounds begin
                dUs[q+1] = dU[1]; dUs[q+2] = dU[2]; dUs[q+3] = dU[3]
                dUs[q+4] = dU[4]; dUs[q+5] = dU[5]
                for s in 1:NS
                    w0 = _wsi(lx, ly, lz, WSX, WSY)
                    dUs[q+Int32(5)+Int32(s)] = _ctu_dUX(xm, xc, xp, ym, yp, zm, zp,
                        _xld(Xt, _wsi(lx - Int32(1), ly, lz, WSX, WSY), s, NS),
                        _xld(Xt, w0, s, NS),
                        _xld(Xt, _wsi(lx + Int32(1), ly, lz, WSX, WSY), s, NS),
                        _xld(Xt, _wsi(lx, ly - Int32(1), lz, WSX, WSY), s, NS),
                        _xld(Xt, _wsi(lx, ly + Int32(1), lz, WSX, WSY), s, NS),
                        _xld(Xt, _wsi(lx, ly, lz - hz, WSX, WSY), s, NS),
                        _xld(Xt, _wsi(lx, ly, lz + hz, WSX, WSY), s, NS), λ, γ)
                end
            end
            t += NT
        end
    end
    @synchronize
    # ── phase 2: three face-plane sweeps, ΔF per target cell in registers ────
    tx = tid % TBX; ty = (tid ÷ TBX) % TBY
    tz = tid ÷ (TBX * TBY)
    a1 = 0.0f0; a2 = 0.0f0; a3 = 0.0f0; a4 = 0.0f0; a5 = 0.0f0; a6 = 0.0f0
    aU = 0.0f0                                         # Σ_ax (Ss_hi − Ss_lo): pdV face-velocity div
    aX1 = 0.0f0; aX2 = 0.0f0
    for dir in Int32(1):Int32(DIM)                    # 2-D: no z face sweep
        ex = dir == Int32(1) ? Int32(1) : Int32(0)
        ey = dir == Int32(2) ? Int32(1) : Int32(0)
        ez = dir == Int32(3) ? Int32(1) : Int32(0)
        nfx = TBX + ex; nfy = TBY + ey; nfz = TBZ + ez
        nf  = nfx * nfy * nfz
        let p = tid
            while p < nf
                fx = p % nfx; fy = (p ÷ nfx) % nfy; fz = p ÷ (nfx * nfy)
                lx = fx + Int32(2) - ex; ly = fy + Int32(2) - ey; lz = fz + Int32(2) * hz - ez
                mx = lx + ex; my = ly + ey; mz = lz + ez
                dL = _dsi(lx - Int32(1), ly - Int32(1), lz - hz, DSX, DSY) * NL
                dR = _dsi(mx - Int32(1), my - Int32(1), mz - hz, DSX, DSY) * NL
                @inbounds begin
                    dUL = (dUs[dL+1], dUs[dL+2], dUs[dL+3], dUs[dL+4], dUs[dL+5])
                    dUR = (dUs[dR+1], dUs[dR+2], dUs[dR+3], dUs[dR+4], dUs[dR+5])
                end
                cm = _tld(Wt, lx - ex, ly - ey, lz - ez, dsc, gsc, vb1, vb2, vb3, WSX, WSY)
                cL = _tld(Wt, lx, ly, lz, dsc, gsc, vb1, vb2, vb3, WSX, WSY)
                cR = _tld(Wt, mx, my, mz, dsc, gsc, vb1, vb2, vb3, WSX, WSY)
                cp = _tld(Wt, mx + ex, my + ey, mz + ez, dsc, gsc, vb1, vb2, vb3, WSX, WSY)
                L = _ctu_predicted(cm, cL, cR, dUL, 1.0f0)
                R = _ctu_predicted(cL, cR, cp, dUR, -1.0f0)
                F, Fge, Fm, Ssf = _ctu_face(L, R, γ, dir)
                q = p * (Int32(7) + Int32(NS))
                @inbounds begin
                    Fs[q+1] = F[1]; Fs[q+2] = F[2]; Fs[q+3] = F[3]
                    Fs[q+4] = F[4]; Fs[q+5] = F[5]; Fs[q+6] = Fge
                    Fs[q+7] = Ssf                      # contact speed (pdV face velocity)
                    if NS > 0
                        ρfL = _recon_one(cm, cL, cR, 1.0f0)[1]
                        ρfR = _recon_one(cL, cR, cp, -1.0f0)[1]
                        for s in 1:NS
                            Fs[q+Int32(7)+Int32(s)] = _ctu_faceX(Fm, L[1], R[1], ρfL, ρfR,
                                _xld(Xt, _wsi(lx - ex, ly - ey, lz - ez, WSX, WSY), s, NS),
                                _xld(Xt, _wsi(lx, ly, lz, WSX, WSY), s, NS),
                                _xld(Xt, _wsi(mx, my, mz, WSX, WSY), s, NS),
                                _xld(Xt, _wsi(mx + ex, my + ey, mz + ez, WSX, WSY), s, NS),
                                dUs[dL+Int32(5)+Int32(s)], dUs[dR+Int32(5)+Int32(s)])
                        end
                    end
                end
                p += NT
            end
        end
        @synchronize
        flo = ((tz * nfy + ty) * nfx + tx) * (Int32(7) + Int32(NS))
        fhi = (((tz + ez) * nfy + (ty + ey)) * nfx + (tx + ex)) * (Int32(7) + Int32(NS))
        @inbounds begin
            a1 += Fs[fhi+1] - Fs[flo+1]; a2 += Fs[fhi+2] - Fs[flo+2]
            a3 += Fs[fhi+3] - Fs[flo+3]; a4 += Fs[fhi+4] - Fs[flo+4]
            a5 += Fs[fhi+5] - Fs[flo+5]; a6 += Fs[fhi+6] - Fs[flo+6]
            aU += Fs[fhi+7] - Fs[flo+7]
            NS >= 1 && (aX1 += Fs[fhi+8] - Fs[flo+8])
            NS >= 2 && (aX2 += Fs[fhi+9] - Fs[flo+9])
        end
        @synchronize
    end
    # ── epilogue ──────────────────────────────────────────────────────────────
    i = tx0 + tx + ng; j = ty0 + ty + ng; k = tz0 + tz + hz * ng
    idx = base + _lidx(i, j, k, nd)
    @inbounds begin
        U0 = _loadU(D, S1, S2, S3, Tau, Ge, idx, dsc, ssc, esc, gsc)
        wx = tx + Int32(2); wy = ty + Int32(2); wz = tz + Int32(2) * hz
        Wc = _tld(Wt, wx, wy, wz, dsc, gsc, vb1, vb2, vb3, WSX, WSY)
        # pdV work from the HLLC face contact-speed divergence aU (Enzo
        # flux_twoshock: ges = qc·pcent·(ub_i − ub_{i+1}); the RK2 kernel's
        # Sshi−Sslo).  The former CENTERED cell-velocity divv read C/F ghost
        # prims — prolonged-skin velocity error → spurious compression heating
        # of uncovered coarse cells (the sync-independent runaway channel).
        P_c = (γ - 1.0f0) * max(Wc[1] * Wc[5], 1.0f-30)
        nD  = max(U0[1] - λ * a1, 1.0f-30)
        nS1 = U0[2] - λ * a2
        nS2 = U0[3] - λ * a3
        nS3 = U0[4] - λ * a4
        nT  = U0[5] - λ * a5
        nG  = U0[6] - λ * (a6 + pdv * P_c * aU)       # pdv: BAM_NO_PDV diag (0 zeros pdV work)
        KE = 0.5f0 * (nS1 * nS1 + nS2 * nS2 + nS3 * nS3) / nD
        demax = _demax6(Tau, base, i, j, k, nd, esc, hz)
        (nT - KE) > η * demax && (nG = nT - KE)   # Enzo eta2 gate (η vs NEIGHBOR-MAX total)
        # extreme-frontier safety: a degenerate Riemann state can still emit a
        # nonfinite lane (~1 cell/Mcell-step at the collapse frontier).  Keep
        # the OLD state for that cell/substep instead of poisoning the level.
        ok = isfinite(nD) & isfinite(nS1) & isfinite(nS2) & isfinite(nS3) &
             isfinite(nT) & isfinite(nG) & isfinite(aX1) & isfinite(aX2) &
             (nD > 0.0f0) & (nT > 0.0f0)   # positivity: huge-but-FINITE fluxes
                                           # must not drive ρ/E through zero
        # CFL self-consistency cap: λ was chosen so (|v|+cs)·λ ≤ cfl·safety
        # (~0.12) for every real cell.  An update whose NEW signal speed
        # exceeds 1/λ (8× headroom) broke the assumption it was built on —
        # a degenerate flux wrote garbage-but-finite v or e.  Keep old state.
        invD = 1.0f0 / max(nD, 1.0f-30)
        vm = max(abs(nS1), max(abs(nS2), abs(nS3))) * invD
        cn = sqrt(γ * (γ - 1.0f0) * max(nG, 1.0f-30) * invD)
        ok &= (vm + cn) * λ < 1.0f0
        # ABSOLUTE signal cap (BAM_SMAXCAP; 0 = off): the λ-relative bound above
        # ESCALATES under pathology — a hot injection shrinks the global λ,
        # which widens 1/λ for the next substep (observed: a zoom-clipped
        # secondary knot ratcheting C/F skin cells from 15 km/s to 1e18 K over
        # ~40 substeps, strangling dτ 470×).  An absolute bound anchored to the
        # problem's physical signal scale breaks the ratchet.
        ok &= (scap <= 0.0f0) | (vm + cn < scap)
        if !ok
            nD = U0[1]; nS1 = U0[2]; nS2 = U0[3]; nS3 = U0[4]
            nT = U0[5]; nG = U0[6]
        end
        # Enzo eta1 (pgas2d_dual eslice update): keep the CONSERVATIVE total where
        # the thermal part is locally resolved ((nT−KE) > η1·nT); rebuild
        # Tau ≡ KE + Ge only in KE-dominated cells, where the stored Tau−KE is
        # cancellation noise.  Exact conservation through resolved shocks; the
        # (deliberate, Enzo-matching) non-conservation is confined to cells with
        # thermal fraction < η1 = 1e-3.
        nG = max(nG, 1.0f-30)
        KEn = 0.5f0 * (nS1 * nS1 + nS2 * nS2 + nS3 * nS3) / max(nD, 1.0f-30)
        nT = ifelse(nT - KEn > 1.0f-3 * nT, nT, KEn + nG)
        Do_[idx]   = _narrow(eltype(Do_), nD / dsc)
        S1o_[idx]  = _narrow(eltype(S1o_), nS1 / ssc)
        S2o_[idx]  = _narrow(eltype(S2o_), nS2 / ssc)
        S3o_[idx]  = _narrow(eltype(S3o_), nS3 / ssc)
        Tauo_[idx] = _narrow(eltype(Tauo_), nT / esc)
        Geo_[idx]  = _narrow(eltype(Geo_), nG / gsc)
        if NS >= 1
            Xc = decode_log2sp(Float32, spin[1][idx])
            spout[1][idx] = ok ?
                _sd_spstore!(spres[1], idx, max(Xc * U0[1] - λ * aX1, 0.0f0),
                             max(nD, 1.0f-30)) :
                spin[1][idx]
        end
        if NS >= 2
            Xc = decode_log2sp(Float32, spin[2][idx])
            spout[2][idx] = ok ?
                _sd_spstore!(spres[2], idx, max(Xc * U0[1] - λ * aX2, 0.0f0),
                             max(nD, 1.0f-30)) :
                spin[2][idx]
        end
    end
end

# ── per-block bulk velocity (BAM_BULKV) ──────────────────────────────────────
# Mass-weighted mean velocity per block, accumulated over ACTIVE cells (f32
# atomics; order round-off is harmless — vb is a rounding-frame offset, any
# nearby value works, and tiled/per-cell parity holds because both read the
# SAME computed array).
@kernel function _blockvsum_k!(sums, @Const(D), @Const(S1), @Const(S2), @Const(S3),
                               @Const(live_d),
                               B::Int32, ng::Int32, nd::Int32, stride::Int32,
                               dimv::Val{DIM}) where {DIM}
    t = @index(Global)
    t0 = Int32(t) - Int32(1)
    B3 = _nblk(B, dimv)
    bi = t0 ÷ B3; c = t0 % B3
    @inbounds begin
        slot = live_d[bi + Int32(1)]
        base = (slot - Int32(1)) * stride
        i, j, k = _cell_ijk(c, B, ng, dimv)
        idx = base + _lidx(i, j, k, nd)
        d = Float32(D[idx])
        if d > 0.0f0
            r = 4 * (slot - Int32(1))
            KA.@atomic sums[r+1] += d
            KA.@atomic sums[r+2] += Float32(S1[idx])
            KA.@atomic sums[r+3] += Float32(S2[idx])
            KA.@atomic sums[r+4] += Float32(S3[idx])
        end
    end
end

@kernel function _blockvb_k!(Vb, @Const(sums), @Const(live_d),
                             @Const(Dsc), @Const(Ssc))
    b = @index(Global)
    @inbounds begin
        slot = live_d[b]
        r = 4 * (slot - Int32(1)); q = 3 * (slot - Int32(1))
        sd = sums[r+1]
        f = ifelse(sd > 0.0f0, Ssc[slot] / (Dsc[slot] * max(sd, 1.0f-30)), 0.0f0)
        Vb[q+1] = sums[r+2] * f
        Vb[q+2] = sums[r+3] * f
        Vb[q+3] = sums[r+4] * f
    end
end

"Get-or-create the per-block bulk-velocity table (zeros ⇒ lab frame, bit-identical)."
function _vbtab(lev::Level)
    vbt = get(lev.tabs, :vb, nothing)
    if !(vbt isa AbstractVector{Float32}) || length(vbt) != 3 * lev.cap
        vbt = device_zeros(lev.be, Float32, (3 * lev.cap,))
        lev.tabs[:vb] = vbt
    end
    return vbt
end

"""
    ctu_level!(hier, l, λ; η = 0.1, tiled = nothing)

`η` is the ENZO eta2 dual-energy selection threshold (`BAM_DE_ETA2`, default 0.1,
Bryan et al. 1995 / pgas2d_dual.F): Ge ← Tau−KE only where the implied thermal
energy exceeds η × the MAX total-energy density over the cell + 6 face
neighbours (`_demax6`).  Only a resolved shock passes; unresolved sub-grid
velocity dispersion (~1% of E from restriction / C-F transfer) does not — the
old LOCAL gate at 1e-3 let it through and drove the static-level runaway.

One full CTU substep of level `l`: R → O in one launch (the caller swaps
buffers).  `tiled = nothing` auto-selects: the shared-memory kernel on GPU
backends when `B` is (8,8,4)-tileable and `nsp ≤ 2`, the per-cell oracle
otherwise (always on CPU).
"""
function ctu_level!(hier::AMRHierarchy, l::Int, λ::Float32;
                    η::Float32 = Float32(parse(Float64, get(ENV, "BAM_DE_ETA2", "0.1"))),
                    tiled::Union{Nothing,Bool} = nothing)
    lev = hier.levels[l + 1]
    isempty(lev.live) && return nothing
    scap = Float32(parse(Float64, get(ENV, "BAM_SMAXCAP", "0")))
    pdv  = get(ENV, "BAM_NO_PDV", "0") == "1" ? 0.0f0 : 1.0f0   # diag: zero the pdV work source
    B = lev.B; NS = lev.nsp
    @assert NS <= 2 "the CTU kernels carry ≤ 2 species lanes (got $NS)"
    # the tiled kernel needs ~42 KB threadgroup memory at NS=2 — fine on CUDA,
    # but over Metal's usual 32 KB limit, so it must be OPTED IN per backend
    # (`_TILING`, populated by each GPU extension once validated).  The per-cell
    # oracle runs on every backend; `tiled=true`/`false` force either path.
    # tile: 8×8×4 (3-D) or 16×16×1 (2-D); B must tile it
    tilecap = hier.besym != :cpu && (lev.dim == 3 ?
              (B % _TBX == 0 && B % _TBY == 0 && B % _TBZ == 0) :
              (B % _TBX2 == 0 && B % _TBY2 == 0))
    prefer  = hier.besym in _TILING
    # BAM_CTU_TILED=0: force the per-cell oracle on GPU (tiled-kernel bisect)
    get(ENV, "BAM_CTU_TILED", "1") == "0" && (prefer = false)
    usetile = (tiled === nothing ? prefer : tiled) && tilecap
    sin_  = ntuple(q -> lev.sp[q], NS)
    sout_ = ntuple(q -> lev.spo[q], NS)
    sres_ = ntuple(q -> lev.spr[q], NS)
    # BAM_BULKV=1: refresh the per-block bulk velocity from the R state; the
    # f16 tile then quantizes the RESIDUAL velocity (see _tld).  Off ⇒ zeros
    # (bit-identical to the historic lab-frame path).
    vbt = _vbtab(lev)
    if get(ENV, "BAM_BULKV", "0") == "1"
        nb = length(lev.live)
        sums = device_zeros(lev.be, Float32, (4 * lev.cap,))
        _blockvsum_k!(lev.be)(sums, lev.D, lev.S1, lev.S2, lev.S3, lev.live_d,
                              Int32(B), Int32(lev.ng), Int32(lev.nd),
                              Int32(lev.stride), Val(lev.dim); ndrange = nb * _nact(lev))
        _blockvb_k!(lev.be)(vbt, sums, lev.live_d, lev.Dsc, lev.Ssc; ndrange = nb)
    end
    if usetile
        tpb = lev.dim == 3 ? (B ÷ _TBX) * (B ÷ _TBY) * (B ÷ _TBZ) :
                             (B ÷ _TBX2) * (B ÷ _TBY2)
        n = length(lev.live) * tpb * _NT                      # _NT = 256 in both
        _ctu_step_k!(lev.be, _NT)(gasfields_o(lev)..., sout_, gasfields(lev)..., sin_, sres_,
                                  lev.live_d, lev.Dsc, lev.Ssc, lev.Esc, lev.Gsc, vbt,
                                  λ, Float32(hier.gamma), η, scap, pdv, Int32(tpb),
                                  Int32(B), Int32(lev.ng), Int32(lev.nd),
                                  Int32(lev.stride), Val(NS), Val(lev.dim); ndrange = n)
    else
        n = length(lev.live) * _nact(lev)
        _ctu_cell_k!(lev.be)(gasfields_o(lev)..., sout_, gasfields(lev)..., sin_, sres_,
                             lev.live_d, lev.Dsc, lev.Ssc, lev.Esc, lev.Gsc, vbt,
                             λ, Float32(hier.gamma), η, scap, pdv,
                             Int32(B), Int32(lev.ng), Int32(lev.nd),
                             Int32(lev.stride), Val(NS), Val(lev.dim); ndrange = n)
    end
    return nothing
end
