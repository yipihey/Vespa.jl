# boundary.jl — domain boundary conditions on NON-periodic faces.
#
# Periodic axes need nothing here: the sibling tables wrap and the level tiling
# IS the periodic BC.  Along a non-periodic axis the table builders drop every
# wrap image (tables.jl `_wrapped_out`, `_clip_domain`), so the ghost cells of a
# block that sit BEYOND the domain edge are left for this pass, which runs LAST
# in `fill_ghosts!` (after prolongation + sibling copies):
#
#   :outflow     ghost = the edge cell (zero gradient)
#   :reflecting  ghost = mirror image about the face, normal momentum negated
#   :inflow      ghost = a fixed physical state (ρ,u,v,w,P), encoded per block
#
# One rectangle job per (block, face) on the domain edge, covering the ng-deep
# ghost slab along the face normal and the FULL stored extent transversely.
# Axes are filled in order x, y (, z) in separate launches, so a corner ghost
# beyond two faces is built from the already-filled first-axis ghosts — the
# standard composition, correct for every BC pairing.  Every level runs this
# (fine blocks touching the domain edge synthesize their own edge ghosts; their
# C/F prolongation was clipped to the domain).
#
# Job encoding (RectJobTable, tables.jl): dst = src = slot; [3..5] slab low
# corner (stored frame); [9..11] extent; [12] axis; [13] side (0 lo, 1 hi);
# [14] kind (1 outflow, 2 reflecting, 3 inflow); [15] face id 2(axis−1)+side+1.
# Species (u16 log₂ fractions) take the edge value on :inflow faces (zero
# gradient) — the inflow state carries no composition.

const BC_OUTFLOW, BC_REFLECTING, BC_INFLOW = Int32(1), Int32(2), Int32(3)
_bc_code(k::Symbol) = k === :outflow ? BC_OUTFLOW : k === :reflecting ? BC_REFLECTING :
                      k === :inflow ? BC_INFLOW : error("no fill for BC $k")

"""
    build_boundary_tables!(hier, l)

(Re)build level `l`'s per-axis domain-face fill tables `:bc1`…`:bc3` (absent on
periodic axes) and the per-field inflow value arrays `:bcval`.
"""
function build_boundary_tables!(hier::AMRHierarchy, l::Int)
    lev = hier.levels[l + 1]
    for a in 1:3
        delete!(lev.tabs, Symbol(:bc, a))
    end
    all(lev.per) && return nothing
    B = lev.B; ng = lev.ng
    for a in 1:lev.dim
        lev.per[a] && continue
        acc = _JobAcc(2 * length(lev.live))
        for s in lev.live
            m = lev.meta[s]
            for side in 0:1
                edge = side == 0 ? m.origin[a] == 0 :
                                   m.origin[a] + UInt128(B) == lev.P[a]
                edge || continue
                kind = _bc_code(hier.bc[a][side + 1])
                doff = ntuple(d -> d == a ? (side == 0 ? 0 : ng + B) : 0, 3)
                ext  = ntuple(d -> d == a ? ng : _ndax(lev, d), 3)
                _push_job!(acc, s, s, doff, (0, 0, 0), ext,
                           (a, side, Int(kind)))
                # the face id rides in the pad slot [15]; [16] flags a same-level
                # (or beyond-domain) neighbour on the transverse lo (bit 0) / hi
                # (bit 1) side — CT completes the hi-wall faces of the unsupplied
                # transverse ghost rows (ct.jl)
                acc.jobs[(acc.n - 1) * _JOB + 15] = Int32(2 * (a - 1) + side + 1)
                acc.jobs[(acc.n - 1) * _JOB + 16] = _transverse_flags(lev, m.origin, a)
            end
        end
        lev.tabs[Symbol(:bc, a)] = to_device_table(lev.be, _finish(acc))
    end
    # inflow values per field (D,S1,S2,S3,Tau,Ge) per face id (6 faces), physical
    γ = hier.gamma
    VT = _wt(eltype(lev.D))                          # f64 runs keep f64 inflow states
    vals = ntuple(_ -> zeros(VT, 6), 10)             # gas (6) + MHD (Bx,By,Bz,ψ)
    for a in 1:3, side in 1:2
        hier.bc[a][side] === :inflow || continue
        ρ, u, v, w, P, bx, by, bz = hier.inflow[a][side]
        ge = P / (γ - 1)
        em = hier.physics === :mhd ? 0.5 * (bx^2 + by^2 + bz^2) : 0.0  # MHD: Tau is total E
        fid = 2 * (a - 1) + side
        for (fi, x) in enumerate((ρ, ρ * u, ρ * v, ρ * w, ge + 0.5ρ * (u^2 + v^2 + w^2) + em,
                                  ge, bx, by, bz, 0.0))
            vals[fi][fid] = VT(x)
        end
    end
    lev.tabs[:bcval] = map(v -> to_device(lev.be, v, VT), vals)
    return nothing
end

function _transverse_flags(lev::Level, origin, a::Int)
    f = Int32(0)
    for t in 1:lev.dim
        t == a && continue
        for (bit, sh) in ((0, -lev.B), (1, lev.B))
            n = ntuple(d -> d == t ? Int128(origin[d]) + sh : Int128(origin[d]), 3)
            ok = if !lev.per[t] && !(0 <= n[t] < Int128(lev.P[t]))
                true                                 # beyond the domain: the other axis' BC
            else
                haskey(lev.byorigin, ntuple(d -> wrapc(n[d], lev.P[d]), 3))
            end
            ok && (f |= Int32(1) << bit)
        end
    end
    return f
end

# One thread per ghost cell of the job's slab.  `flipax` = the axis whose normal
# momentum this field is (S1→1, S2→2, S3→3; 0 otherwise); `useval` = 1 for gas
# fields (inflow from `val`), 0 for species (inflow = zero gradient).
@kernel function _bc_fill_k!(F, @Const(sc), @Const(val), @Const(jobs), @Const(cellstart),
                             njobs::Int32, nd::Int32, stride::Int32, ng::Int32, B::Int32,
                             flipax::Int32, useval::Int32)
    t = @index(Global)
    t0 = Int32(t) - Int32(1)
    b, ri, rj, rk = _job_decode(jobs, cellstart, njobs, t0)
    @inbounds begin
        slot = jobs[b+1]
        di = jobs[b+3] + ri; dj = jobs[b+4] + rj; dk = jobs[b+5] + rk
        a = jobs[b+12]; side = jobs[b+13]; kind = jobs[b+14]; fid = jobs[b+15]
        g = a == Int32(1) ? di : (a == Int32(2) ? dj : dk)     # normal stored index
        # source index along the normal: edge cell (outflow / species inflow)
        # or mirror image about the face (reflecting)
        sg = if kind == BC_REFLECTING
            side == Int32(0) ? Int32(2) * ng - Int32(1) - g :
                               Int32(2) * (ng + B) - Int32(1) - g
        else
            side == Int32(0) ? ng : ng + B - Int32(1)
        end
        si = a == Int32(1) ? sg : di
        sj = a == Int32(2) ? sg : dj
        sk = a == Int32(3) ? sg : dk
        base = (slot - Int32(1)) * stride
        didx = base + (dk * nd + dj) * nd + di + Int32(1)
        if kind == BC_INFLOW && useval == Int32(1)
            F[didx] = _narrow(eltype(F), val[fid] / eltype(val)(sc[slot]))
        else
            v = F[base + (sk * nd + sj) * nd + si + Int32(1)]
            F[didx] = (kind == BC_REFLECTING && flipax == a) ? -v : v
        end
    end
end

"""
    fill_boundary!(hier, l; buf = :R)

Synthesize level-`l` ghosts beyond the non-periodic domain faces in buffer
`buf` (gas + species).  No-op on a fully periodic hierarchy.
"""
function fill_boundary!(hier::AMRHierarchy, l::Int; buf::Symbol = :R)
    lev = hier.levels[l + 1]
    all(lev.per) && return nothing
    dst = buf === :R ? (gasfields(lev)..., mfields(lev)...) :
                       (gasfields_o(lev)..., mfields_o(lev)...)
    scs = (classes(lev)..., lev.Bsc, lev.Bsc, lev.Bsc, lev.Bsc)
    sps = buf === :R ? lev.sp : lev.spo
    vals = lev.tabs[:bcval]
    # normal momentum AND normal B flip at a reflecting face; ψ is even
    flips = (Int32(0), Int32(1), Int32(2), Int32(3), Int32(0), Int32(0),
             Int32(1), Int32(2), Int32(3), Int32(0))
    for a in 1:lev.dim
        tab = get(lev.tabs, Symbol(:bc, a), nothing)
        (tab === nothing || tab.total == 0) && continue
        for (fi, (F, sc)) in enumerate(zip(dst, scs))
            isempty(F) && continue
            _bc_fill_k!(lev.be)(F, sc, vals[fi], tab.jobs, tab.cellstart,
                                Int32(tab.njobs), Int32(lev.nd), Int32(lev.stride),
                                Int32(lev.ng), Int32(lev.B), flips[fi], Int32(1);
                                ndrange = tab.total)
        end
        for S in sps
            _bc_fill_k!(lev.be)(S, lev.Dsc, vals[1], tab.jobs, tab.cellstart,
                                Int32(tab.njobs), Int32(lev.nd), Int32(lev.stride),
                                Int32(lev.ng), Int32(lev.B), Int32(0), Int32(0);
                                ndrange = tab.total)
        end
    end
    isct(lev) && ct_fill_face_boundary!(hier, l; buf)   # CT face pools (ct.jl)
    return nothing
end
