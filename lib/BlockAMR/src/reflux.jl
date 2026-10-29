# reflux.jl — Berger–Colella flux correction across coarse–fine interfaces,
# by RECOMPUTATION through the same `_face_flux` chain the stage kernel used
# (euler_nested pattern): with the stage input buffer intact (R→O double
# buffering), recomputing a face flux from identical stencil arguments yields the
# bit-identical value the sweep applied — the correction cancels exactly where it
# should.  Registers accumulate, in f32,
#     reg = Σ_stages/substeps [ w_f·(¼ Σ_{2×2} F_fine) − w_c·F_coarse ]
# per coarse interface face-cell (6 lanes: 5 conserved + the CMA Ge flux), and
# `reflux_apply!` adds λ_c·side·reg to the uncovered coarse cell.
#
# Face-registry entry (16 Int32):
#   [1] c_slot  [2..4] c_cell (local 0-based, coarse UNCOVERED cell)
#   [5] axis    [6] side (+1: fine block below the coarse cell along axis)
#   [7] f_slot  [8..10] f_cell (local 0-based low-corner INSIDE fine cell of the
#                               2×2 group adjacent to the face)

"""
    build_cf_faces(lev, plev) -> (n, entries::Vector{Int32})

Enumerate all coarse–fine interface face-cells of fine level `lev` against parent
level `plev`: fine-block outer faces not covered by same-level siblings, grouped
2×2 into coarse face cells, each targeted at the adjacent UNCOVERED coarse cell.
"""
function build_cf_faces(lev::Level, plev::Level)
    ent = Int32[]
    B = lev.B; ng = lev.ng
    B2 = lev.dim == 3 ? B : 1                          # t2 extent (flat z in 2-D)
    g2 = lev.dim == 3 ? 2 : 1                          # fine group along t2
    for s in lev.live
        m = lev.meta[s]
        for ax in 1:lev.dim, side in (Int32(-1), Int32(1))
            t1, t2 = ax == 1 ? (2, 3) : ax == 2 ? (1, 3) : (1, 2)
            slab_lo = ntuple(d -> d == ax ?
                (side == 1 ? Int128(m.origin[d]) + B : Int128(m.origin[d]) - 1) :
                Int128(m.origin[d]), 3)
            # a face ON a non-periodic domain edge has no coarse neighbour
            (!lev.per[ax] && !(0 <= slab_lo[ax] < Int128(lev.P[ax]))) && continue
            covered = falses(B, B2)                     # transverse (t1, t2), 1-based
            for c in lattice_neighbors(lev, m.origin, B)
                cm = lev.meta[c]
                ia = filter(im -> lev.per[ax] || im[3] == 0,
                            axis_images(slab_lo[ax], 1, Int128(cm.origin[ax]), B,
                                        Int128(lev.P[ax])))
                isempty(ia) && continue
                for a1 in axis_images(slab_lo[t1], B, Int128(cm.origin[t1]), B, Int128(lev.P[t1])),
                    a2 in axis_images(slab_lo[t2], B2, Int128(cm.origin[t2]), _Bax(lev, t2),
                                      Int128(lev.P[t2]))
                    (!lev.per[t1] && a1[3] != 0) && continue
                    (!lev.per[t2] && a2[3] != 0) && continue
                    r1 = Int(a1[1] - slab_lo[t1]); r2 = Int(a2[1] - slab_lo[t2])
                    covered[r1+1:r1+a1[2], r2+1:r2+a2[2]] .= true
                end
            end
            for f2 in 0:g2:B2-g2, f1 in 0:2:B-2         # 2×2 (2-D: 2×1) fine groups
                cov = covered[f1+1, f2+1]
                # even origins/extents ⇒ coverage is uniform over each fine group
                @assert all(covered[f1+i, f2+j] == cov for i in 1:2, j in 1:g2)
                cov && continue
                # outside coarse cell (global coarse coords)
                q = Vector{Int128}(undef, 3)
                q[ax] = side == 1 ? (Int128(m.origin[ax]) + B) >> 1 :
                                    (Int128(m.origin[ax]) >> 1) - 1
                q[t1] = (Int128(m.origin[t1]) + f1) >> 1
                q[t2] = t2 <= lev.dim ? (Int128(m.origin[t2]) + f2) >> 1 : Int128(0)
                qw = ntuple(d -> wrapc(q[d], plev.P[d]), 3)
                own = lattice_owner(plev, qw, B)
                own != 0 ||
                    error("C/F face target not owned (nesting violation?)")
                po = plev.meta[own]
                cc = ntuple(d -> Int32(Int(mod(Int128(qw[d]) - Int128(po.origin[d]),
                                               Int128(plev.P[d]))) + _ngax(plev, d)), 3)
                fc = Vector{Int32}(undef, 3)
                fc[ax] = side == 1 ? Int32(ng + B - 1) : Int32(ng)
                fc[t1] = Int32(ng + f1); fc[t2] = Int32(_ngax(lev, t2) + f2)
                append!(ent, Int32[own, cc..., ax, side, s, fc..., 0, 0, 0, 0, 0, 0])
            end
        end
    end
    return length(ent) ÷ 16, ent
end

"Attach the C/F face registry + zeroed f32 register (6 lanes/entry) to `lev.tabs`."
function build_cf_register!(hier::AMRHierarchy, l::Int)
    lev = hier.levels[l + 1]; plev = hier.levels[l]
    n, ent = build_cf_faces(lev, plev)
    lev.tabs[:cfn]   = n
    lev.tabs[:cf]    = to_device(lev.be, ent, Int32)
    # 6 + nsp lanes: 5 conserved + ge + the SPECIES ρX CMA lanes.  Without the
    # species lanes the fine and coarse sides disagree about the species mass
    # crossing every C/F face and the mismatch is never reconciled — measured
    # ΣρX creation at C/F rings (the in-run xe/fH2 fraction pump: species mass
    # made at every level's boundary ring around the collapsing core, ×10 per
    # new level; gas mass exact, max-X bounded, single-level CMA exact).
    lev.tabs[:cfreg] = device_zeros(lev.be, Float32, (max((6 + lev.nsp) * n, 1),))
    # group entries by unique target coarse cell → atomic-free apply
    keyof(i) = (ent[16(i-1)+1], ent[16(i-1)+2], ent[16(i-1)+3], ent[16(i-1)+4])
    perm = sort!(collect(Int32, 1:n); by = keyof)
    gs = Int32[1]
    for i in 2:n
        keyof(Int(perm[i])) != keyof(Int(perm[i-1])) && push!(gs, Int32(i))
    end
    push!(gs, Int32(n + 1))
    lev.tabs[:cfng]   = max(length(gs) - 1, 0)
    lev.tabs[:cfperm] = to_device(lev.be, perm, Int32)
    lev.tabs[:cfgs]   = to_device(lev.be, gs, Int32)
    # per-target-cell f32 species-correction RESIDUAL: the u16 codec quantum
    # (~8e-4 relative) is 60-300× the per-apply ρX correction, so a naive
    # encode drops EVERY correction (measured: reflux lanes bit-inert).  Carry
    # the unrepresentable remainder across applies so corrections integrate.
    lev.tabs[:cfres] = device_zeros(lev.be, Float32,
                                    (max(lev.nsp * max(length(gs) - 1, 0), 1),))
    return nothing
end

# ge value adjacent to a face (specific internal energy of the upwind CELL)
@inline function _geflux(Fm::Float32, GeL::Float32, DL::Float32,
                         GeR::Float32, DR::Float32)
    return Fm >= 0.0f0 ? Fm * GeL / max(DL, 1.0f-30) : Fm * GeR / max(DR, 1.0f-30)
end

# recompute the face flux (5 vars + ge lane) between cells at `a` and `a+1` along
# axis, in block `base` local coords — the same `_face_flux` call the stage made.
@inline function _face6(D, S1, S2, S3, Tau, Ge, base::Int32,
                        i::Int32, j::Int32, k::Int32, nd::Int32,
                        γ::Float32, ax::Int32,
                        dsc::Float32, ssc::Float32, esc::Float32, gsc::Float32)
    di = ax == Int32(1) ? Int32(1) : Int32(0)
    dj = ax == Int32(2) ? Int32(1) : Int32(0)
    dk = ax == Int32(3) ? Int32(1) : Int32(0)
    idx(m) = base + _lidx(i + m * di, j + m * dj, k + m * dk, nd)
    U1 = _loadU(D, S1, S2, S3, Tau, Ge, idx(Int32(-1)), dsc, ssc, esc, gsc)
    U2 = _loadU(D, S1, S2, S3, Tau, Ge, idx(Int32(0)), dsc, ssc, esc, gsc)
    U3 = _loadU(D, S1, S2, S3, Tau, Ge, idx(Int32(1)), dsc, ssc, esc, gsc)
    U4 = _loadU(D, S1, S2, S3, Tau, Ge, idx(Int32(2)), dsc, ssc, esc, gsc)
    F, _ = _face_flux(_prim_de(U1, γ), _prim_de(U2, γ), _prim_de(U3, γ),
                      _prim_de(U4, γ), γ, ax)
    gf = _geflux(F[1], U2[6], U2[1], U3[6], U3[1])
    return F, gf
end

# reg −= wc · F_coarse   (the flux the coarse sweep applied at the interface).
# Species lanes 7..6+NS: ρX = F_mass·X_upwind — the SAME CMA upwinding the RK2
# kernel applied; without these lanes the fine/coarse species-mass mismatch at
# every C/F face is never reconciled (the measured ΣρX creation → xe/fH2 pump).
@kernel function _capture_coarse_k!(reg, @Const(ent),
                                    @Const(D), @Const(S1), @Const(S2), @Const(S3),
                                    @Const(Tau), @Const(Ge), spin,
                                    @Const(Dsc), @Const(Ssc), @Const(Esc), @Const(Gsc),
                                    wc::Float32, γ::Float32, nd::Int32, stride::Int32,
                                    ::Val{NS}) where {NS}
    e = @index(Global)
    b = (Int32(e) - Int32(1)) * Int32(16)
    @inbounds begin
        base = (ent[b+1] - Int32(1)) * stride
        dsc = Dsc[ent[b+1]]; ssc = Ssc[ent[b+1]]; esc = Esc[ent[b+1]]; gsc = Gsc[ent[b+1]]
        ci = ent[b+2]; cj = ent[b+3]; ck = ent[b+4]
        ax = ent[b+5]; side = ent[b+6]
        # side=+1 → the coarse cell's LOW face: face between (q−1, q) → window at q−1
        # side=−1 → HIGH face: between (q, q+1) → window at q
        sh = side == Int32(1) ? Int32(-1) : Int32(0)
        i = ci + (ax == Int32(1) ? sh : Int32(0))
        j = cj + (ax == Int32(2) ? sh : Int32(0))
        k = ck + (ax == Int32(3) ? sh : Int32(0))
        F, gf = _face6(D, S1, S2, S3, Tau, Ge, base, i, j, k, nd, γ, ax, dsc, ssc, esc, gsc)
        # a degenerate Riemann solve (f16-vacuum extremes) yields a nonfinite
        # flux; the hydro epilogue kept the old state there, so the matching
        # register contribution is ZERO — never let NaN into the register.
        ok = isfinite(F[1]) & isfinite(F[2]) & isfinite(F[3]) &
             isfinite(F[4]) & isfinite(F[5]) & isfinite(gf)
        r = (Int32(e) - Int32(1)) * (Int32(6) + Int32(NS))
        reg[r+1] -= wc * ifelse(ok, F[1], 0.0f0)
        reg[r+2] -= wc * ifelse(ok, F[2], 0.0f0)
        reg[r+3] -= wc * ifelse(ok, F[3], 0.0f0)
        reg[r+4] -= wc * ifelse(ok, F[4], 0.0f0)
        reg[r+5] -= wc * ifelse(ok, F[5], 0.0f0)
        reg[r+6] -= wc * ifelse(ok, gf,   0.0f0)
        if NS > 0
            di = ax == Int32(1) ? Int32(1) : Int32(0)
            dj = ax == Int32(2) ? Int32(1) : Int32(0)
            dk = ax == Int32(3) ? Int32(1) : Int32(0)
            iL = base + _lidx(i, j, k, nd)
            iR = base + _lidx(i + di, j + dj, k + dk, nd)
            for q in 1:NS
                XL = decode_log2sp(Float32, spin[q][iL])
                XR = decode_log2sp(Float32, spin[q][iR])
                fX = F[1] * (F[1] >= 0.0f0 ? XL : XR)
                reg[r+Int32(6)+Int32(q)] -= wc * ifelse(ok & isfinite(fX), fX, 0.0f0)
            end
        end
    end
end

# CTU variants: same registry walk, the flux recomputed through the CTU chain
# (`_ctu_face6` replicates the f16 hat-tile rounding, so it reproduces the
# tiled/per-cell kernels' applied flux bit-exactly).  λ is the predictor's.
@kernel function _capture_coarse_ctu_k!(reg, @Const(ent),
                                        @Const(D), @Const(S1), @Const(S2), @Const(S3),
                                        @Const(Tau), @Const(Ge),
                                        @Const(Dsc), @Const(Ssc), @Const(Esc), @Const(Gsc),
                                        @Const(Vb),
                                        wc::Float32, λ::Float32, γ::Float32,
                                        nd::Int32, stride::Int32, nspreg::Int32, kz::Int32)
    e = @index(Global)
    b = (Int32(e) - Int32(1)) * Int32(16)
    @inbounds begin
        base = (ent[b+1] - Int32(1)) * stride
        dsc = Dsc[ent[b+1]]; ssc = Ssc[ent[b+1]]; esc = Esc[ent[b+1]]; gsc = Gsc[ent[b+1]]
        vq = 3 * (ent[b+1] - Int32(1))
        vb1 = Vb[vq+1]; vb2 = Vb[vq+2]; vb3 = Vb[vq+3]   # block's bulk frame (BAM_BULKV)
        ci = ent[b+2]; cj = ent[b+3]; ck = ent[b+4]
        ax = ent[b+5]; side = ent[b+6]
        sh = side == Int32(1) ? Int32(-1) : Int32(0)
        i = ci + (ax == Int32(1) ? sh : Int32(0))
        j = cj + (ax == Int32(2) ? sh : Int32(0))
        k = ck + (ax == Int32(3) ? sh : Int32(0))
        F, gf = _ctu_face6(D, S1, S2, S3, Tau, Ge, base, i, j, k, nd, λ, γ, ax,
                           dsc, ssc, esc, gsc, vb1, vb2, vb3, kz)
        ok = isfinite(F[1]) & isfinite(F[2]) & isfinite(F[3]) &
             isfinite(F[4]) & isfinite(F[5]) & isfinite(gf)
        # TODO(ctu-species-reflux): species lanes left ZERO on the CTU path —
        # register stride is (6+NS) for layout compatibility (nsp passed via
        # the launch's nspreg).  Production scheme is :rk2, which has them.
        r = (Int32(e) - Int32(1)) * (Int32(6) + nspreg)
        reg[r+1] -= wc * ifelse(ok, F[1], 0.0f0)
        reg[r+2] -= wc * ifelse(ok, F[2], 0.0f0)
        reg[r+3] -= wc * ifelse(ok, F[3], 0.0f0)
        reg[r+4] -= wc * ifelse(ok, F[4], 0.0f0)
        reg[r+5] -= wc * ifelse(ok, F[5], 0.0f0)
        reg[r+6] -= wc * ifelse(ok, gf,   0.0f0)
    end
end

@kernel function _capture_fine_ctu_k!(reg, @Const(ent),
                                      @Const(D), @Const(S1), @Const(S2), @Const(S3),
                                      @Const(Tau), @Const(Ge),
                                      @Const(Dsc), @Const(Ssc), @Const(Esc), @Const(Gsc),
                                      @Const(Vb),
                                      wf::Float32, λ::Float32, γ::Float32,
                                      nd::Int32, stride::Int32, nspreg::Int32, kz::Int32)
    e = @index(Global)
    b = (Int32(e) - Int32(1)) * Int32(16)
    @inbounds begin
        base = (ent[b+7] - Int32(1)) * stride
        dsc = Dsc[ent[b+7]]; ssc = Ssc[ent[b+7]]; esc = Esc[ent[b+7]]; gsc = Gsc[ent[b+7]]
        vq = 3 * (ent[b+7] - Int32(1))
        vb1 = Vb[vq+1]; vb2 = Vb[vq+2]; vb3 = Vb[vq+3]   # block's bulk frame (BAM_BULKV)
        fi = ent[b+8]; fj = ent[b+9]; fk = ent[b+10]
        ax = ent[b+5]; side = ent[b+6]
        sh = side == Int32(1) ? Int32(0) : Int32(-1)
        a1 = 0.0f0; a2 = 0.0f0; a3 = 0.0f0; a4 = 0.0f0; a5 = 0.0f0; a6 = 0.0f0
        t1i = ax == Int32(1) ? Int32(0) : Int32(1)
        t1j = ax == Int32(1) ? Int32(1) : Int32(0)
        t1k = Int32(0)
        t2i = Int32(0)
        t2j = ax == Int32(3) ? Int32(1) : Int32(0)
        t2k = ax == Int32(3) ? Int32(0) : Int32(1)
        for s2 in Int32(0):kz, s1 in Int32(0):Int32(1)   # 2-D: t2 is the flat z
            i = fi + (ax == Int32(1) ? sh : Int32(0)) + s1 * t1i + s2 * t2i
            j = fj + (ax == Int32(2) ? sh : Int32(0)) + s1 * t1j + s2 * t2j
            k = fk + (ax == Int32(3) ? sh : Int32(0)) + s1 * t1k + s2 * t2k
            F, gf = _ctu_face6(D, S1, S2, S3, Tau, Ge, base, i, j, k, nd, λ, γ, ax,
                               dsc, ssc, esc, gsc, vb1, vb2, vb3, kz)
            ok = isfinite(F[1]) & isfinite(F[2]) & isfinite(F[3]) &
                 isfinite(F[4]) & isfinite(F[5]) & isfinite(gf)
            a1 += ifelse(ok, F[1], 0.0f0); a2 += ifelse(ok, F[2], 0.0f0)
            a3 += ifelse(ok, F[3], 0.0f0); a4 += ifelse(ok, F[4], 0.0f0)
            a5 += ifelse(ok, F[5], 0.0f0); a6 += ifelse(ok, gf,   0.0f0)
        end
        # TODO(ctu-species-reflux): see the coarse CTU capture note.
        r = (Int32(e) - Int32(1)) * (Int32(6) + nspreg)
        q = (kz == Int32(1) ? 0.25f0 : 0.5f0) * wf     # mean over the 2² (2-D: 2) sub-faces
        reg[r+1] += q * a1; reg[r+2] += q * a2; reg[r+3] += q * a3
        reg[r+4] += q * a4; reg[r+5] += q * a5; reg[r+6] += q * a6
    end
end

# reg += wf · ¼ Σ_{2×2 subfaces} F_fine  (+ species ρX lanes, as in the coarse capture)
@kernel function _capture_fine_k!(reg, @Const(ent),
                                  @Const(D), @Const(S1), @Const(S2), @Const(S3),
                                  @Const(Tau), @Const(Ge), spin,
                                  @Const(Dsc), @Const(Ssc), @Const(Esc), @Const(Gsc),
                                  wf::Float32, γ::Float32, nd::Int32, stride::Int32,
                                  ::Val{NS}, kz::Int32) where {NS}
    e = @index(Global)
    b = (Int32(e) - Int32(1)) * Int32(16)
    @inbounds begin
        base = (ent[b+7] - Int32(1)) * stride
        dsc = Dsc[ent[b+7]]; ssc = Ssc[ent[b+7]]; esc = Esc[ent[b+7]]; gsc = Gsc[ent[b+7]]
        fi = ent[b+8]; fj = ent[b+9]; fk = ent[b+10]
        ax = ent[b+5]; side = ent[b+6]
        # face of the INSIDE fine cell: hi face for side=+1 (window at the cell),
        # lo face for side=−1 (window at cell−1)
        sh = side == Int32(1) ? Int32(0) : Int32(-1)
        a1 = 0.0f0; a2 = 0.0f0; a3 = 0.0f0; a4 = 0.0f0; a5 = 0.0f0; a6 = 0.0f0
        aXa = 0.0f0; aXb = 0.0f0
        di = ax == Int32(1) ? Int32(1) : Int32(0)
        dj = ax == Int32(2) ? Int32(1) : Int32(0)
        dk = ax == Int32(3) ? Int32(1) : Int32(0)
        t1i = ax == Int32(1) ? Int32(0) : Int32(1)                 # transverse unit 1
        t1j = ax == Int32(1) ? Int32(1) : Int32(0)
        t1k = Int32(0)
        t2i = Int32(0)                                             # transverse unit 2
        t2j = ax == Int32(3) ? Int32(1) : Int32(0)
        t2k = ax == Int32(3) ? Int32(0) : Int32(1)
        for s2 in Int32(0):kz, s1 in Int32(0):Int32(1)   # 2-D: t2 is the flat z
            i = fi + (ax == Int32(1) ? sh : Int32(0)) + s1 * t1i + s2 * t2i
            j = fj + (ax == Int32(2) ? sh : Int32(0)) + s1 * t1j + s2 * t2j
            k = fk + (ax == Int32(3) ? sh : Int32(0)) + s1 * t1k + s2 * t2k
            F, gf = _face6(D, S1, S2, S3, Tau, Ge, base, i, j, k, nd, γ, ax, dsc, ssc, esc, gsc)
            ok = isfinite(F[1]) & isfinite(F[2]) & isfinite(F[3]) &
                 isfinite(F[4]) & isfinite(F[5]) & isfinite(gf)
            a1 += ifelse(ok, F[1], 0.0f0); a2 += ifelse(ok, F[2], 0.0f0)
            a3 += ifelse(ok, F[3], 0.0f0); a4 += ifelse(ok, F[4], 0.0f0)
            a5 += ifelse(ok, F[5], 0.0f0); a6 += ifelse(ok, gf,   0.0f0)
            if NS > 0
                iL = base + _lidx(i, j, k, nd)
                iR = base + _lidx(i + di, j + dj, k + dk, nd)
                for q in 1:NS
                    XL = decode_log2sp(Float32, spin[q][iL])
                    XR = decode_log2sp(Float32, spin[q][iR])
                    fX = F[1] * (F[1] >= 0.0f0 ? XL : XR)
                    okX = ok & isfinite(fX)
                    q == 1 && (aXa += ifelse(okX, fX, 0.0f0))
                    q == 2 && (aXb += ifelse(okX, fX, 0.0f0))
                end
            end
        end
        r = (Int32(e) - Int32(1)) * (Int32(6) + Int32(NS))
        q = (kz == Int32(1) ? 0.25f0 : 0.5f0) * wf     # mean over the 2² (2-D: 2) sub-faces
        reg[r+1] += q * a1; reg[r+2] += q * a2; reg[r+3] += q * a3
        reg[r+4] += q * a4; reg[r+5] += q * a5; reg[r+6] += q * a6
        NS >= 1 && (reg[r+7] += q * aXa)
        NS >= 2 && (reg[r+8] += q * aXb)
    end
end

# U_coarse += λ_c · side · reg, GROUPED by target cell (one thread per unique
# coarse cell sums all its face entries) — atomic-free (f16 has no atomics) and
# scale-aware (physical correction ÷ the target block's class scale on store).
@kernel function _reflux_apply_k!(D, S1, S2, S3, Tau, Ge, sp, res,
                                  @Const(Dsc), @Const(Ssc), @Const(Esc), @Const(Gsc),
                                  @Const(ent), @Const(reg), @Const(perm), @Const(gs),
                                  λc::Float32, γ::Float32, scap::Float32, geref::Float32,
                                  spref::Float32,
                                  taurecon::Float32, nd::Int32, stride::Int32,
                                  ::Val{NS}) where {NS}
    g = @index(Global)
    @inbounds begin
        a1 = 0.0f0; a2 = 0.0f0; a3 = 0.0f0; a4 = 0.0f0; a5 = 0.0f0; a6 = 0.0f0
        aXa = 0.0f0; aXb = 0.0f0
        for ii in gs[g]:gs[Int32(g)+Int32(1)] - Int32(1)
            e = perm[ii]
            b = (e - Int32(1)) * Int32(16)
            w = λc * Float32(ent[b+6])
            r = (e - Int32(1)) * (Int32(6) + Int32(NS))
            a1 += w * reg[r+1]; a2 += w * reg[r+2]; a3 += w * reg[r+3]
            a4 += w * reg[r+4]; a5 += w * reg[r+5]; a6 += w * reg[r+6]
            NS >= 1 && (aXa += w * reg[r+7])
            NS >= 2 && (aXb += w * reg[r+8])
        end
        e0 = perm[gs[g]]
        b = (e0 - Int32(1)) * Int32(16)
        slot = ent[b+1]
        base = (slot - Int32(1)) * stride
        idx  = base + _lidx(ent[b+2], ent[b+3], ent[b+4], nd)
        dsc = Dsc[slot]; ssc = Ssc[slot]; esc = Esc[slot]; gsc = Gsc[slot]
        nD  = Float32(D[idx])   + a1 / dsc
        nS1 = Float32(S1[idx])  + a2 / ssc
        nS2 = Float32(S2[idx])  + a3 / ssc
        nS3 = Float32(S3[idx])  + a4 / ssc
        nT  = Float32(Tau[idx]) + a5 / esc
        nG  = Float32(Ge[idx])  + geref * a6 / gsc   # geref: BAM_NO_GEREFLUX diag (0 skips Ge reflux)
        # taurecon (BAM_TAU_RECON, default on): re-derive Tau = KE(S,D)+Ge from the
        # refluxed momenta/density/internal energy instead of the independent a5
        # total-energy flux correction — same phantom-dispersion cure as the
        # restriction reconstruct, so the sync can't heat this C/F coarse cell.
        ρp = max(nD * dsc, 1.0f-30)
        keR = 0.5f0 * ((nS1*ssc)^2 + (nS2*ssc)^2 + (nS3*ssc)^2) / ρp
        nT  = ifelse(taurecon > 0.5f0, (keR + max(nG,1.0f-30) * gsc) / esc, nT)
        # keep-old-state safety (the hydro-epilogue twin): a nonfinite,
        # positivity-violating, or CFL-inconsistent correction must never
        # overwrite a finite coarse cell (see the CTU epilogue cap).
        ok = isfinite(nD) & isfinite(nS1) & isfinite(nS2) &
             isfinite(nS3) & isfinite(nT) & isfinite(nG) &
             (nD > 0.0f0) & (nT > 0.0f0)
        invD = 1.0f0 / max(nD, 1.0f-30)
        vm = max(abs(nS1), max(abs(nS2), abs(nS3))) * invD
        cn = sqrt(γ * (γ - 1.0f0) * max(nG, 1.0f-30) * invD)
        ok &= (vm + cn) * λc < 1.0f0
        # absolute signal cap (BAM_SMAXCAP) — see the CTU epilogue twin: the
        # λ-relative bound alone ratchets under a hot-cell/shrinking-λ feedback
        ok &= (scap <= 0.0f0) | (vm + cn < scap)
        # species: apply the ρX correction consistently with the ρ correction.
        # The u16 quantum is far larger than a single apply's correction, so we
        # accumulate into the per-target f32 residual `res` and encode the
        # TARGET value; whatever the codec cannot represent stays in `res` for
        # the next apply (first-order sigma-delta — corrections integrate
        # instead of being rounded away every substep).
        ρold = max(Float32(D[idx]) * dsc, 1.0f-30)
        ρnew = max(nD * dsc, 1.0f-30)
        if NS >= 1 && spref > 0.0f0
            Xo = decode_log2sp(Float32, sp[1][idx])
            want = Xo * ρold + aXa + res[(Int32(g)-Int32(1))*Int32(NS)+Int32(1)]
            if ok
                code = encode_log2sp(max(want, 0.0f0) / ρnew)
                got = decode_log2sp(Float32, code) * ρnew
                sp[1][idx] = code
                res[(Int32(g)-Int32(1))*Int32(NS)+Int32(1)] = want - got
            end
        end
        if NS >= 2 && spref > 0.0f0
            Xo = decode_log2sp(Float32, sp[2][idx])
            want = Xo * ρold + aXb + res[(Int32(g)-Int32(1))*Int32(NS)+Int32(2)]
            if ok
                code = encode_log2sp(max(want, 0.0f0) / ρnew)
                got = decode_log2sp(Float32, code) * ρnew
                sp[2][idx] = code
                res[(Int32(g)-Int32(1))*Int32(NS)+Int32(2)] = want - got
            end
        end
        D[idx]   = ok ? _narrow(eltype(D),   nD)  : D[idx]
        S1[idx]  = ok ? _narrow(eltype(S1),  nS1) : S1[idx]
        S2[idx]  = ok ? _narrow(eltype(S2),  nS2) : S2[idx]
        S3[idx]  = ok ? _narrow(eltype(S3),  nS3) : S3[idx]
        Tau[idx] = ok ? _narrow(eltype(Tau), nT)  : Tau[idx]
        Ge[idx]  = ok ? _narrow(eltype(Ge),  nG)  : Ge[idx]
    end
end

"""
    capture_fine!(hier, l, buf, w)

Accumulate `+w·¼ΣF_fine` into the (l−1, l) register from level `l`'s stage-input
buffer `buf`.  Weights: same-dt integration w=½ per RK stage; strict 2:1
subcycling w=¼ per stage (the extra ½ is the dt_f/dt_c time average).
"""
function capture_fine!(hier::AMRHierarchy, l::Int, buf::Symbol, w::Float32;
                       λ::Float32 = hier.λ)
    lev = hier.levels[l + 1]
    n = get(lev.tabs, :cfn, 0)::Int
    n == 0 && return nothing
    ff = buf === :R ? gasfields(lev) : gasfields_o(lev)
    NS = lev.nsp
    sf = buf === :R ? ntuple(q -> lev.sp[q], NS) : ntuple(q -> lev.spo[q], NS)
    if hier.scheme === :ctu
        _capture_fine_ctu_k!(lev.be)(_tabf(lev, :cfreg), _tabi(lev, :cf), ff...,
                                     lev.Dsc, lev.Ssc, lev.Esc, lev.Gsc, _vbtab(lev),
                                     w, λ,
                                     Float32(hier.gamma), Int32(lev.nd),
                                     Int32(lev.stride), Int32(NS), _kz(lev); ndrange = n)
    else
        _capture_fine_k!(lev.be)(_tabf(lev, :cfreg), _tabi(lev, :cf), ff..., sf,
                                 lev.Dsc, lev.Ssc, lev.Esc, lev.Gsc, w,
                                 Float32(hier.gamma), Int32(lev.nd),
                                 Int32(lev.stride), Val(NS), _kz(lev); ndrange = n)
    end
    return nothing
end

"""
    capture_coarse!(hier, l, buf, w)

Accumulate `−w·F_coarse` into the (l−1, l) register from level `l−1`'s stage-input
buffer `buf` (w=½ per coarse RK stage in all integration modes).
"""
function capture_coarse!(hier::AMRHierarchy, l::Int, buf::Symbol, w::Float32;
                         λ::Float32 = hier.λ)
    lev = hier.levels[l + 1]; plev = hier.levels[l]
    n = get(lev.tabs, :cfn, 0)::Int
    n == 0 && return nothing
    fc = buf === :R ? gasfields(plev) : gasfields_o(plev)
    NS = lev.nsp
    sc = buf === :R ? ntuple(q -> plev.sp[q], NS) : ntuple(q -> plev.spo[q], NS)
    if hier.scheme === :ctu
        _capture_coarse_ctu_k!(lev.be)(_tabf(lev, :cfreg), _tabi(lev, :cf), fc...,
                                       plev.Dsc, plev.Ssc, plev.Esc, plev.Gsc,
                                       _vbtab(plev), w, λ,
                                       Float32(hier.gamma), Int32(plev.nd),
                                       Int32(plev.stride), Int32(NS), _kz(plev); ndrange = n)
    else
        _capture_coarse_k!(lev.be)(_tabf(lev, :cfreg), _tabi(lev, :cf), fc..., sc,
                                   plev.Dsc, plev.Ssc, plev.Esc, plev.Gsc, w,
                                   Float32(hier.gamma), Int32(plev.nd),
                                   Int32(plev.stride), Val(NS); ndrange = n)
    end
    return nothing
end

"Same-dt convenience (Phase-1/2 integrator): fine + coarse capture in one call."
function capture_cf!(hier::AMRHierarchy, l::Int; buf::Symbol, wf::Float32, wc::Float32)
    capture_fine!(hier, l, buf, wf)
    capture_coarse!(hier, l, buf, wc)
    return nothing
end

"""
    reflux_apply!(hier, l, λc)

Apply the accumulated (l−1, l) corrections to the coarse R buffers and zero the
register.  `λc` = the coarse level's dt/dx for the completed step.
"""
function reflux_apply!(hier::AMRHierarchy, l::Int, λc::Float32)
    lev = hier.levels[l + 1]; plev = hier.levels[l]
    n = get(lev.tabs, :cfn, 0)::Int
    n == 0 && return nothing
    ng_ = lev.tabs[:cfng]::Int
    ng_ == 0 && return nothing
    scap = Float32(parse(Float64, get(ENV, "BAM_SMAXCAP", "0")))
    geref = get(ENV, "BAM_NO_GEREFLUX", "0") == "1" ? 0.0f0 : 1.0f0   # diag: skip Ge-lane reflux
    spref = get(ENV, "BAM_NO_SPREFLUX", "0") == "1" ? 0.0f0 : 1.0f0   # diag: skip species-lane reflux
    taurecon = get(ENV, "BAM_TAU_RECON", "0") == "1" ? 1.0f0 : 0.0f0  # re-derive Tau=KE+Ge post-reflux (opt-in; breaks shock conservation)
    NS = lev.nsp
    spc = ntuple(q -> plev.sp[q], NS)
    _reflux_apply_k!(lev.be)(gasfields(plev)..., spc, _tabf(lev, :cfres),
                             plev.Dsc, plev.Ssc, plev.Esc, plev.Gsc,
                             _tabi(lev, :cf), _tabf(lev, :cfreg),
                             _tabi(lev, :cfperm), _tabi(lev, :cfgs), λc,
                             Float32(hier.gamma), scap, geref, spref, taurecon,
                             Int32(plev.nd), Int32(plev.stride), Val(NS); ndrange = ng_)
    fill!(_tabf(lev, :cfreg), 0.0f0)
    return nothing
end
