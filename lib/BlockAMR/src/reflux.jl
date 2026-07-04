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
    for s in lev.live
        m = lev.meta[s]
        for ax in 1:3, side in (Int32(-1), Int32(1))
            t1, t2 = ax == 1 ? (2, 3) : ax == 2 ? (1, 3) : (1, 2)
            slab_lo = ntuple(d -> d == ax ?
                (side == 1 ? Int128(m.origin[d]) + B : Int128(m.origin[d]) - 1) :
                Int128(m.origin[d]), 3)
            slab_len = ntuple(d -> d == ax ? 1 : B, 3)
            covered = falses(B, B)                      # transverse (t1, t2), 1-based
            for c in overlapping_blocks(lev, slab_lo, slab_len)
                cm = lev.meta[c]
                isempty(axis_images(slab_lo[ax], 1, Int128(cm.origin[ax]), B,
                                    Int128(lev.P[ax]))) && continue
                for a1 in axis_images(slab_lo[t1], B, Int128(cm.origin[t1]), B, Int128(lev.P[t1])),
                    a2 in axis_images(slab_lo[t2], B, Int128(cm.origin[t2]), B, Int128(lev.P[t2]))
                    r1 = Int(a1[1] - slab_lo[t1]); r2 = Int(a2[1] - slab_lo[t2])
                    covered[r1+1:r1+a1[2], r2+1:r2+a2[2]] .= true
                end
            end
            for f2 in 0:2:B-2, f1 in 0:2:B-2            # 2×2 fine groups (t1=f1, t2=f2)
                cov = covered[f1+1, f2+1]
                # even origins/extents ⇒ coverage is uniform over each 2×2 group
                @assert all(covered[f1+i, f2+j] == cov for i in 1:2, j in 1:2)
                cov && continue
                # outside coarse cell (global coarse coords)
                q = Vector{Int128}(undef, 3)
                q[ax] = side == 1 ? (Int128(m.origin[ax]) + B) >> 1 :
                                    (Int128(m.origin[ax]) >> 1) - 1
                q[t1] = (Int128(m.origin[t1]) + f1) >> 1
                q[t2] = (Int128(m.origin[t2]) + f2) >> 1
                qw = ntuple(d -> wrapc(q[d], plev.P[d]), 3)
                owner = overlapping_blocks(plev, ntuple(d -> Int128(qw[d]), 3), (1, 1, 1))
                length(owner) == 1 ||
                    error("C/F face target not uniquely owned (nesting violation?)")
                po = plev.meta[owner[1]]
                cc = ntuple(d -> Int32(Int(mod(Int128(qw[d]) - Int128(po.origin[d]),
                                               Int128(plev.P[d]))) + plev.ng), 3)
                fc = Vector{Int32}(undef, 3)
                fc[ax] = side == 1 ? Int32(ng + B - 1) : Int32(ng)
                fc[t1] = Int32(ng + f1); fc[t2] = Int32(ng + f2)
                append!(ent, Int32[owner[1], cc..., ax, side, s, fc..., 0, 0, 0, 0, 0, 0])
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
    lev.tabs[:cfreg] = device_zeros(lev.be, Float32, (max(6n, 1),))
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
                        dsc::Float32, ssc::Float32, esc::Float32)
    di = ax == Int32(1) ? Int32(1) : Int32(0)
    dj = ax == Int32(2) ? Int32(1) : Int32(0)
    dk = ax == Int32(3) ? Int32(1) : Int32(0)
    idx(m) = base + _lidx(i + m * di, j + m * dj, k + m * dk, nd)
    U1 = _loadU(D, S1, S2, S3, Tau, Ge, idx(Int32(-1)), dsc, ssc, esc)
    U2 = _loadU(D, S1, S2, S3, Tau, Ge, idx(Int32(0)), dsc, ssc, esc)
    U3 = _loadU(D, S1, S2, S3, Tau, Ge, idx(Int32(1)), dsc, ssc, esc)
    U4 = _loadU(D, S1, S2, S3, Tau, Ge, idx(Int32(2)), dsc, ssc, esc)
    F, _ = _face_flux(_prim_de(U1, γ), _prim_de(U2, γ), _prim_de(U3, γ),
                      _prim_de(U4, γ), γ, ax)
    gf = _geflux(F[1], U2[6], U2[1], U3[6], U3[1])
    return F, gf
end

# reg −= wc · F_coarse   (the flux the coarse sweep applied at the interface)
@kernel function _capture_coarse_k!(reg, @Const(ent),
                                    @Const(D), @Const(S1), @Const(S2), @Const(S3),
                                    @Const(Tau), @Const(Ge),
                                    @Const(Dsc), @Const(Ssc), @Const(Esc),
                                    wc::Float32, γ::Float32, nd::Int32, stride::Int32)
    e = @index(Global)
    b = (Int32(e) - Int32(1)) * Int32(16)
    @inbounds begin
        base = (ent[b+1] - Int32(1)) * stride
        dsc = Dsc[ent[b+1]]; ssc = Ssc[ent[b+1]]; esc = Esc[ent[b+1]]
        ci = ent[b+2]; cj = ent[b+3]; ck = ent[b+4]
        ax = ent[b+5]; side = ent[b+6]
        # side=+1 → the coarse cell's LOW face: face between (q−1, q) → window at q−1
        # side=−1 → HIGH face: between (q, q+1) → window at q
        sh = side == Int32(1) ? Int32(-1) : Int32(0)
        i = ci + (ax == Int32(1) ? sh : Int32(0))
        j = cj + (ax == Int32(2) ? sh : Int32(0))
        k = ck + (ax == Int32(3) ? sh : Int32(0))
        F, gf = _face6(D, S1, S2, S3, Tau, Ge, base, i, j, k, nd, γ, ax, dsc, ssc, esc)
        r = (Int32(e) - Int32(1)) * Int32(6)
        reg[r+1] -= wc * F[1]; reg[r+2] -= wc * F[2]; reg[r+3] -= wc * F[3]
        reg[r+4] -= wc * F[4]; reg[r+5] -= wc * F[5]; reg[r+6] -= wc * gf
    end
end

# reg += wf · ¼ Σ_{2×2 subfaces} F_fine
@kernel function _capture_fine_k!(reg, @Const(ent),
                                  @Const(D), @Const(S1), @Const(S2), @Const(S3),
                                  @Const(Tau), @Const(Ge),
                                  @Const(Dsc), @Const(Ssc), @Const(Esc),
                                  wf::Float32, γ::Float32, nd::Int32, stride::Int32)
    e = @index(Global)
    b = (Int32(e) - Int32(1)) * Int32(16)
    @inbounds begin
        base = (ent[b+7] - Int32(1)) * stride
        dsc = Dsc[ent[b+7]]; ssc = Ssc[ent[b+7]]; esc = Esc[ent[b+7]]
        fi = ent[b+8]; fj = ent[b+9]; fk = ent[b+10]
        ax = ent[b+5]; side = ent[b+6]
        # face of the INSIDE fine cell: hi face for side=+1 (window at the cell),
        # lo face for side=−1 (window at cell−1)
        sh = side == Int32(1) ? Int32(0) : Int32(-1)
        a1 = 0.0f0; a2 = 0.0f0; a3 = 0.0f0; a4 = 0.0f0; a5 = 0.0f0; a6 = 0.0f0
        t1i = ax == Int32(1) ? Int32(0) : Int32(1)                 # transverse unit 1
        t1j = ax == Int32(1) ? Int32(1) : Int32(0)
        t1k = Int32(0)
        t2i = Int32(0)                                             # transverse unit 2
        t2j = ax == Int32(3) ? Int32(1) : Int32(0)
        t2k = ax == Int32(3) ? Int32(0) : Int32(1)
        for s2 in Int32(0):Int32(1), s1 in Int32(0):Int32(1)
            i = fi + (ax == Int32(1) ? sh : Int32(0)) + s1 * t1i + s2 * t2i
            j = fj + (ax == Int32(2) ? sh : Int32(0)) + s1 * t1j + s2 * t2j
            k = fk + (ax == Int32(3) ? sh : Int32(0)) + s1 * t1k + s2 * t2k
            F, gf = _face6(D, S1, S2, S3, Tau, Ge, base, i, j, k, nd, γ, ax, dsc, ssc, esc)
            a1 += F[1]; a2 += F[2]; a3 += F[3]; a4 += F[4]; a5 += F[5]; a6 += gf
        end
        r = (Int32(e) - Int32(1)) * Int32(6)
        q = 0.25f0 * wf
        reg[r+1] += q * a1; reg[r+2] += q * a2; reg[r+3] += q * a3
        reg[r+4] += q * a4; reg[r+5] += q * a5; reg[r+6] += q * a6
    end
end

# U_coarse += λ_c · side · reg, GROUPED by target cell (one thread per unique
# coarse cell sums all its face entries) — atomic-free (f16 has no atomics) and
# scale-aware (physical correction ÷ the target block's class scale on store).
@kernel function _reflux_apply_k!(D, S1, S2, S3, Tau, Ge,
                                  @Const(Dsc), @Const(Ssc), @Const(Esc),
                                  @Const(ent), @Const(reg), @Const(perm), @Const(gs),
                                  λc::Float32, nd::Int32, stride::Int32)
    g = @index(Global)
    @inbounds begin
        a1 = 0.0f0; a2 = 0.0f0; a3 = 0.0f0; a4 = 0.0f0; a5 = 0.0f0; a6 = 0.0f0
        for ii in gs[g]:gs[Int32(g)+Int32(1)] - Int32(1)
            e = perm[ii]
            b = (e - Int32(1)) * Int32(16)
            w = λc * Float32(ent[b+6])
            r = (e - Int32(1)) * Int32(6)
            a1 += w * reg[r+1]; a2 += w * reg[r+2]; a3 += w * reg[r+3]
            a4 += w * reg[r+4]; a5 += w * reg[r+5]; a6 += w * reg[r+6]
        end
        e0 = perm[gs[g]]
        b = (e0 - Int32(1)) * Int32(16)
        slot = ent[b+1]
        base = (slot - Int32(1)) * stride
        idx  = base + _lidx(ent[b+2], ent[b+3], ent[b+4], nd)
        dsc = Dsc[slot]; ssc = Ssc[slot]; esc = Esc[slot]
        D[idx]   = _narrow(eltype(D),   Float32(D[idx])   + a1 / dsc)
        S1[idx]  = _narrow(eltype(S1),  Float32(S1[idx])  + a2 / ssc)
        S2[idx]  = _narrow(eltype(S2),  Float32(S2[idx])  + a3 / ssc)
        S3[idx]  = _narrow(eltype(S3),  Float32(S3[idx])  + a4 / ssc)
        Tau[idx] = _narrow(eltype(Tau), Float32(Tau[idx]) + a5 / esc)
        Ge[idx]  = _narrow(eltype(Ge),  Float32(Ge[idx])  + a6 / esc)
    end
end

"""
    capture_fine!(hier, l, buf, w)

Accumulate `+w·¼ΣF_fine` into the (l−1, l) register from level `l`'s stage-input
buffer `buf`.  Weights: same-dt integration w=½ per RK stage; strict 2:1
subcycling w=¼ per stage (the extra ½ is the dt_f/dt_c time average).
"""
function capture_fine!(hier::AMRHierarchy, l::Int, buf::Symbol, w::Float32)
    lev = hier.levels[l + 1]
    n = get(lev.tabs, :cfn, 0)::Int
    n == 0 && return nothing
    ff = buf === :R ? gasfields(lev) : gasfields_o(lev)
    _capture_fine_k!(lev.be)(lev.tabs[:cfreg], lev.tabs[:cf], ff...,
                             lev.Dsc, lev.Ssc, lev.Esc, w,
                             Float32(hier.gamma), Int32(lev.nd),
                             Int32(lev.stride); ndrange = n)
    return nothing
end

"""
    capture_coarse!(hier, l, buf, w)

Accumulate `−w·F_coarse` into the (l−1, l) register from level `l−1`'s stage-input
buffer `buf` (w=½ per coarse RK stage in all integration modes).
"""
function capture_coarse!(hier::AMRHierarchy, l::Int, buf::Symbol, w::Float32)
    lev = hier.levels[l + 1]; plev = hier.levels[l]
    n = get(lev.tabs, :cfn, 0)::Int
    n == 0 && return nothing
    fc = buf === :R ? gasfields(plev) : gasfields_o(plev)
    _capture_coarse_k!(lev.be)(lev.tabs[:cfreg], lev.tabs[:cf], fc...,
                               plev.Dsc, plev.Ssc, plev.Esc, w,
                               Float32(hier.gamma), Int32(plev.nd),
                               Int32(plev.stride); ndrange = n)
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
    _reflux_apply_k!(lev.be)(gasfields(plev)..., plev.Dsc, plev.Ssc, plev.Esc,
                             lev.tabs[:cf], lev.tabs[:cfreg],
                             lev.tabs[:cfperm], lev.tabs[:cfgs], λc,
                             Int32(plev.nd), Int32(plev.stride); ndrange = ng_)
    fill!(lev.tabs[:cfreg], 0.0f0)
    return nothing
end
