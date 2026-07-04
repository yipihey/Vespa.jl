# regrid.jl — dynamic refinement: device flagging → host clustering → rebuild.
#
# Flow (Enzo-style rebuild, coarse→fine, flags first):
#   1. per-level criterion flags on device (one fused kernel, per-block counts);
#   2. fine flags projected DOWN (a flagged fine cell forces its parent cell to
#      stay flagged, + margin) so surviving deep structure stays nested;
#   3. flags dilated by nbuf cells (across block boundaries — sets are global);
#   4. per-parent clustering on the Bh = B÷2 lattice: a child footprint is created
#      for every lattice tile containing a flagged cell.  Lattice snapping makes
#      all block origins multiples of B ⇒ any two footprints are IDENTICAL or
#      DISJOINT: persistence is a pure origin match and no overlap copies exist;
#   5. new blocks fill by conservative interior prolongation from the parent;
#      vanished blocks are simply freed (their information already lives in the
#      parent through the per-step restriction);
#   6. every level's data-motion tables + C/F registers are rebuilt.
#
# Criterion (Phase 2): overdensity threshold on D.  The policy carries plain
# numbers (device-evaluable), not closures.

"""
    BlockRefinementPolicy(; dthresh, nbuf = 2, every = 4, lmax = 1)

Refine level-l cells where `D > dthresh` (plus fine-structure projection), with
`nbuf` cells of dilation, regridding every `every` hierarchy steps, refining no
deeper than `lmax`.
"""
Base.@kwdef struct BlockRefinementPolicy
    dthresh :: Float64
    nbuf    :: Int = 2
    every   :: Int = 4
    lmax    :: Int = 1
end

@kernel function _flag_k!(flags, count, @Const(D), @Const(live_d), thresh::Float32,
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
        f = Float32(D[idx]) > thresh
        flags[idx] = UInt8(f)
        f && (KA.@atomic count[slot] += Int32(1))
    end
end

"Global level-l cells where the criterion fires (host Set from one device pass)."
function _criterion_cells(hier::AMRHierarchy, l::Int, pol::BlockRefinementPolicy)
    lev = hier.levels[l + 1]
    out = Set{NTuple{3,Int128}}()
    isempty(lev.live) && return out
    flags = device_zeros(lev.be, UInt8, (lev.cap * lev.stride,))
    count = device_zeros(lev.be, Int32, (lev.cap,))
    n = length(lev.live) * lev.B^3
    _flag_k!(lev.be)(flags, count, lev.D, lev.live_d, Float32(pol.dthresh),
                     Int32(lev.B), Int32(lev.ng), Int32(lev.nd), Int32(lev.stride);
                     ndrange = n)
    hc = Array(count); hf = Array(flags)
    for s in lev.live
        hc[s] == 0 && continue
        m = lev.meta[s]; base = (Int(s) - 1) * lev.stride
        for k in 0:lev.B-1, j in 0:lev.B-1, i in 0:lev.B-1
            idx = base + ((k + lev.ng) * lev.nd + (j + lev.ng)) * lev.nd + (i + lev.ng) + 1
            hf[idx] == 0 && continue
            push!(out, (Int128(m.origin[1]) + i, Int128(m.origin[2]) + j,
                        Int128(m.origin[3]) + k))
        end
    end
    return out
end

function _dilate(cells::Set{NTuple{3,Int128}}, n::Int, P::Origin)
    n == 0 && return cells
    out = Set{NTuple{3,Int128}}()
    for g in cells, dk in -n:n, dj in -n:n, di in -n:n
        push!(out, (Int128(wrapc(g[1] + di, P[1])), Int128(wrapc(g[2] + dj, P[2])),
                    Int128(wrapc(g[3] + dk, P[3]))))
    end
    return out
end

_project(cells::Set{NTuple{3,Int128}}) =
    Set{NTuple{3,Int128}}((g[1] >> 1, g[2] >> 1, g[3] >> 1) for g in cells)

"""
    regrid!(hier, pol) -> nothing

Rebuild every refinable level from fresh criterion flags (+ fine-structure
projection + dilation).  Persistent blocks keep their slots and data untouched;
new blocks are conservatively prolonged from their parents; vanished blocks are
freed.  All tables and C/F registers are rebuilt.
"""
function regrid!(hier::AMRHierarchy, pol::BlockRefinementPolicy)
    Ltarget = min(pol.lmax, hier.Lcap)
    # 1–3: flags per level (empty where the level does not exist yet),
    # fine→coarse projection (with margin), dilation
    fl = [ l + 1 <= length(hier.levels) ? _criterion_cells(hier, l, pol) :
           Set{NTuple{3,Int128}}() for l in 0:Ltarget ]          # fl[l+1] = level-l flags
    for l in Ltarget-1:-1:1
        union!(fl[l + 1], _dilate(_project(fl[l + 2]), 1 + cld(pol.nbuf, 2),
                                  level_period(hier.nbase, l)))
    end
    for l in 0:Ltarget-1
        fl[l + 1] = _dilate(fl[l + 1], pol.nbuf, level_period(hier.nbase, l))
    end
    # 4–6: rebuild children levels coarse→fine
    for lc in 1:Ltarget
        _rebuild_level!(hier, lc, fl[lc])
    end
    for l in 0:length(hier.levels)-1
        build_level_tables!(hier, l)
        l >= 1 && build_cf_register!(hier, l)
    end
    return nothing
end

function _rebuild_level!(hier::AMRHierarchy, lc::Int, flags::Set{NTuple{3,Int128}})
    plev = hier.levels[lc]
    lev  = ensure_level!(hier, lc)
    B = hier.B; Bh = B ÷ 2
    # footprints: per parent, lattice tiles (Bh) containing flagged cells
    want = Dict{Origin,Tuple{Int32,NTuple{3,Int16}}}()
    for g in flags
        for p in overlapping_blocks(plev, ntuple(d -> Int128(g[d]), 3), (1, 1, 1))
            pm = plev.meta[p]
            loc = ntuple(d -> Int(mod(Int128(g[d]) - Int128(pm.origin[d]),
                                      Int128(plev.P[d]))), 3)
            all(loc .< B) || continue
            off = ntuple(d -> Int16((loc[d] ÷ Bh) * Bh), 3)
            org = child_origin(pm.origin, off, lev.P)
            want[org] = (p, off)
        end
    end
    # persistence: free vanished, create missing (data via interior prolongation)
    for s in copy(lev.live)
        haskey(want, lev.meta[s].origin) || remove_block!(hier, lc, s)
    end
    news = Int32[]
    for (org, (p, off)) in want
        haskey(lev.byorigin, org) && continue
        push!(news, add_block!(hier, lc, p, Int.(off)))
    end
    if !isempty(news)
        tab = to_device_table(lev.be, build_prolong_jobs(lev, plev; interior = true,
                                                         only_slots = news))
        for (fd, fR) in zip(gasfields(lev), gasfields(plev))
            _run_prolong!(lev.be, tab, fd, fR, fR, 0.0f0, lev.nd, lev.stride)
        end
        for (sd, sR) in zip(lev.sp, plev.sp)
            _run_prolong!(lev.be, tab, sd, sR, sR, 0.0f0, lev.nd, lev.stride)
        end
        for s in news
            lev.meta[s].flags &= ~FLAG_NEW
        end
    end
    return nothing
end
