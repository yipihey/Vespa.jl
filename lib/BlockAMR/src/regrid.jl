# regrid.jl — dynamic refinement: device flagging → tile cascade → rebuild.
#
# Flow (Enzo-style rebuild, coarse→fine, flags first) — ALL per-cell flag work
# stays on the device as dense per-block UInt8 arrays; the host only ever sees
# per-block quantities (counts, 8 octant bits, footprint tiles).  The previous
# Set{NTuple{3,Int128}}-of-cells pipeline hashed every flagged cell ×125 during
# dilation and stalled 256³ deep runs for minutes per regrid.
#
#   1. per-level criterion flags on device (one fused kernel, per-block counts);
#   2. tile CASCADE fine→coarse (host, O(#blocks)): T[lf] = forced level-lf
#      child footprints = footprints of level-lf blocks holding ≥1 criterion
#      flag ∪ tiles touched by the projection boxes of T[lf+1] — so surviving
#      deep structure stays nested (a box is footprint+rim, [t−R, t+Bh+R));
#   3. criterion flags ghost-exchanged (sibling tables wrap periodically) and
#      dilated ±nbuf on device, then reduced to 8 octant-any bits per block;
#   4. `want` per child level: octant bits name (parent, octant) directly —
#      parents revalidated by origin AFTER coarser rebuilds (persistence keeps
#      surviving slots; vanished parents drop their flags, exactly the old
#      owner-at-rebuild-time semantics) — plus the T[lc] tiles expanded through
#      their (nbuf-grown) boxes onto the parent lattice, again by origin, which
#      correctly reaches parents CREATED earlier in this same pass;
#   5. new blocks fill by conservative interior prolongation from the parent;
#      vanished blocks are simply freed (their information already lives in the
#      parent through the per-step restriction);
#   6. Morton slot compaction of churned levels; every affected level's
#      data-motion tables + C/F registers are rebuilt.
#
# Criterion: overdensity threshold on D (×lfac per level).  The policy carries
# plain numbers (device-evaluable), not closures.

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
    lfac    :: Float64 = 1.0   # threshold scales ×lfac per level (Enzo-style
                               # Lagrangian refinement: lfac=8 ⇒ ~constant
                               # cell mass triggers; 1.0 = flat threshold)
end

@kernel function _flag_k!(flags, count, @Const(D), @Const(live_d), @Const(Dsc),
                          thresh::Float32,
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
        f = Float32(D[idx]) * Dsc[slot] > thresh
        flags[idx] = UInt8(f)
        f && (KA.@atomic count[slot] += Int32(1))
    end
end

# ±nbuf box dilation of the criterion flags; reads the ghost frame (filled by
# the sibling exchange, so spill crosses block boundaries and the periodic
# wrap), writes interior only.  Requires nbuf ≤ ng.
@kernel function _dilate_flags_k!(out, @Const(fin), @Const(live_d), nbuf::Int32,
                                  B::Int32, ng::Int32, nd::Int32, stride::Int32)
    t = @index(Global)
    t0 = Int32(t) - Int32(1)
    B3 = B * B * B
    bi = t0 ÷ B3; c = t0 % B3
    @inbounds begin
        slot = live_d[bi + Int32(1)]
        base = (slot - Int32(1)) * stride
        i = c % B + ng; j = (c ÷ B) % B + ng; k = c ÷ (B * B) + ng
        v = UInt8(0)
        for dk in -nbuf:nbuf, dj in -nbuf:nbuf, di in -nbuf:nbuf
            v |= fin[base + _lidx(i + di, j + dj, k + dk, nd)]
        end
        out[base + _lidx(i, j, k, nd)] = v
    end
end

# one thread per (block, octant): any-flag over the Bh³ interior cells of the
# octant — the entire per-cell → per-child-tile clustering reduction.
@kernel function _octant_any_k!(oct, @Const(flags), @Const(live_d),
                                B::Int32, ng::Int32, nd::Int32, stride::Int32)
    t = @index(Global)
    t0 = Int32(t) - Int32(1)
    bi = t0 ÷ Int32(8); o = t0 % Int32(8)
    Bh = B ÷ Int32(2)
    @inbounds begin
        slot = live_d[bi + Int32(1)]
        base = (slot - Int32(1)) * stride
        oi = (o & Int32(1)) * Bh + ng
        oj = ((o >> 0x01) & Int32(1)) * Bh + ng
        ok = ((o >> 0x02) & Int32(1)) * Bh + ng
        v = UInt8(0)
        for k in Int32(0):Bh-Int32(1), j in Int32(0):Bh-Int32(1), i in Int32(0):Bh-Int32(1)
            v |= flags[base + _lidx(oi + i, oj + j, ok + k, nd)]
        end
        oct[t0 + Int32(1)] = v
    end
end

"Criterion flags for level l: device UInt8 (interior of live blocks) + host counts."
function _criterion_flags(hier::AMRHierarchy, l::Int, pol::BlockRefinementPolicy)
    lev = hier.levels[l + 1]
    flags = device_zeros(lev.be, UInt8, (lev.cap * lev.stride,))
    count = device_zeros(lev.be, Int32, (lev.cap,))
    _flag_k!(lev.be)(flags, count, lev.D, lev.live_d, lev.Dsc,
                     Float32(pol.dthresh * pol.lfac^l),
                     Int32(lev.B), Int32(lev.ng), Int32(lev.nd), Int32(lev.stride);
                     ndrange = length(lev.live) * lev.B^3)
    return flags, Array(count)
end

"""
    compact_level!(hier, l) -> Bool

Physically re-pack level `l`'s live blocks into slots `1:nlive` in Morton order.
This is the form in which locality ordering pays: spatially adjacent blocks get
adjacent pool storage (reordering only the LAUNCH list regressed 10× — see the
note in `build_level_tables!`).  Gas + species permute through the O twins
(stale at a root-step boundary, which regridding already requires); `phi` and
the DM pool chain through `rhs` (rewritten at every solve).  Children's
`parent` references on level `l+1` are remapped; the caller must rebuild the
tables / C/F registers of `l` AND `l+1` afterwards.
"""
function compact_level!(hier::AMRHierarchy, l::Int)
    lev = hier.levels[l + 1]
    n = length(lev.live)
    n == 0 && return false
    order = sort(lev.live; by = s -> _morton(lev.meta[s], lev.B))
    order == Int32.(1:n) && return false           # already dense + Morton
    perm_d = to_device(lev.be, order, Int32)
    for (r, o) in zip(gasfields(lev), gasfields_o(lev))
        _permute_slots!(lev.be, o, r, perm_d, n, lev.stride)
    end
    for (r, o) in zip(lev.sp, lev.spo)
        _permute_slots!(lev.be, o, r, perm_d, n, lev.stride)
    end
    swap_buffers!(lev)                             # permuted O twins become R
    _permute_slots!(lev.be, lev.rhs, lev.phi, perm_d, n, lev.stride)
    lev.phi, lev.rhs = lev.rhs, lev.phi
    if !isempty(lev.dm)                            # dm is lazy (deposit levels only)
        _permute_slots!(lev.be, lev.rhs, lev.dm, perm_d, n, lev.stride)
        lev.dm, lev.rhs = lev.rhs, lev.dm          # rhs ends as scratch garbage
    end
    # per-block scales (host-side; power-of-two, so the move is exact)
    for f in (:Dsc, :Ssc, :Esc)
        h = Array(getfield(lev, f))
        hn = ones(Float32, lev.cap); hn[1:n] .= h[order]
        copyto!(getfield(lev, f), hn)
    end
    # host topology: meta shuffle, dense live list, fresh registries
    old2new = zeros(Int32, lev.cap)
    for (i, s) in enumerate(order)
        old2new[s] = Int32(i)
    end
    lev.meta = vcat(lev.meta[order], [BlockMeta() for _ in 1:lev.cap - n])
    lev.live = Int32.(1:n); empty!(lev.freelist)
    empty!(lev.tilemap); empty!(lev.byorigin)
    for s in Int32(1):Int32(n)
        _register!(lev, s)
    end
    sync_live!(lev)
    if l + 2 <= length(hier.levels)
        clev = hier.levels[l + 2]
        for s in clev.live
            clev.meta[s].parent = old2new[clev.meta[s].parent]
        end
    end
    return true
end

"""
    regrid!(hier, pol; compact = true) -> nothing

Rebuild every refinable level from fresh criterion flags (+ fine-structure
projection + dilation).  Persistent blocks keep their slots and data untouched
(until the Morton compaction pass, pure data movement); new blocks are
conservatively prolonged from their parents; vanished blocks are freed.  All
tables and C/F registers are rebuilt.
"""
function regrid!(hier::AMRHierarchy, pol::BlockRefinementPolicy; compact::Bool = true)
    Ltarget = min(pol.lmax, hier.Lcap)
    nbuf = pol.nbuf
    @assert nbuf <= hier.ng "nbuf=$nbuf > ng=$(hier.ng): device dilation reads the ghost frame"
    R  = 2 + cld(nbuf, 2)                  # projection rim (nesting margin)
    Bh = hier.B ÷ 2
    ltop = min(Ltarget - 1, length(hier.levels) - 1)   # deepest level that clusters
    # 1) criterion flags on device, levels 0..Ltarget−1 (the deepest level's
    #    flags have no children to force and are never consumed)
    Fd  = Dict{Int,Any}()
    cnt = Dict{Int,Vector{Int32}}()
    for l in 0:ltop
        lev = hier.levels[l + 1]
        isempty(lev.live) && continue
        Fd[l], cnt[l] = _criterion_flags(hier, l, pol)
    end
    # 2) tile cascade fine→coarse: T[lf] = forced level-lf child footprints as
    #    level-(lf−1)-cell tile origins (multiples of Bh; every block spans
    #    exactly one parent tile).  Pure tile arithmetic — no per-cell state —
    #    so flags forced OUTSIDE currently-live blocks still cascade down.
    T = Dict{Int,Set{NTuple{3,Int128}}}()
    for lf in ltop:-1:1
        tset = Set{NTuple{3,Int128}}()
        lev = hier.levels[lf + 1]
        if haskey(cnt, lf)
            for s in lev.live
                cnt[lf][s] == 0 && continue
                m = lev.meta[s]
                push!(tset, ntuple(d -> Int128(m.origin[d] >> 1), 3))
            end
        end
        if haskey(T, lf + 1)
            Pp = level_period(hier.nbase, lf - 1)
            for t in T[lf + 1]             # box [t−R, t+Bh+R) in level-lf cells
                rng = ntuple(d -> fld(fld(t[d] - R, 2), Bh):fld(fld(t[d] + Bh + R - 1, 2), Bh), 3)
                for tk in rng[3], tj in rng[2], ti in rng[1]
                    push!(tset, (Int128(wrapc(Int128(ti) * Bh, Pp[1])),
                                 Int128(wrapc(Int128(tj) * Bh, Pp[2])),
                                 Int128(wrapc(Int128(tk) * Bh, Pp[3]))))
                end
            end
        end
        isempty(tset) || (T[lf] = tset)
    end
    # 3) ghost-exchange + dilate the criterion flags, reduce to octant bits
    #    (8 per block) — captured with block ORIGINS so step 4 can revalidate
    #    parents after coarser levels rebuild.
    oct = Dict{Int,Tuple{Vector{Origin},Vector{UInt8}}}()
    for l in 0:ltop
        haskey(cnt, l) || continue
        sum(cnt[l]) == 0 && (delete!(Fd, l); continue)
        lev = hier.levels[l + 1]
        F = Fd[l]
        _run_copy!(lev.be, _tabt(lev, :sib), F, F, lev.nd, lev.stride)
        G = device_zeros(lev.be, UInt8, (lev.cap * lev.stride,))
        _dilate_flags_k!(lev.be)(G, F, lev.live_d, Int32(nbuf), Int32(lev.B),
                                 Int32(lev.ng), Int32(lev.nd), Int32(lev.stride);
                                 ndrange = length(lev.live) * lev.B^3)
        nb = length(lev.live)
        o = device_zeros(lev.be, UInt8, (8 * nb,))
        _octant_any_k!(lev.be)(o, G, lev.live_d, Int32(lev.B), Int32(lev.ng),
                               Int32(lev.nd), Int32(lev.stride); ndrange = 8 * nb)
        ho = Array(o)
        bits = zeros(UInt8, nb)
        for bi in 1:nb, oo in 0:7
            ho[8 * (bi - 1) + oo + 1] != 0 && (bits[bi] |= UInt8(1) << oo)
        end
        oct[l] = (Origin[lev.meta[s].origin for s in lev.live], bits)
        delete!(Fd, l)                     # release the flag pools eagerly
    end
    # 4–6: rebuild children levels coarse→fine from `want` maps, then compact
    # churned levels, then rebuild tables ONLY where topology changed (a level's
    # :pro/:res/:cf also depend on its PARENT, so a change at l dirties l and
    # l+1; level 0 never changes here — its sibling table, the largest,
    # survives every regrid).
    changed = falses(Ltarget + 2)          # sized by TARGET (ensure_level! grows hier.levels)
    for lc in 1:Ltarget
        plev = hier.levels[lc]
        Pc = level_period(hier.nbase, lc)
        want = Dict{Origin,Tuple{Int32,NTuple{3,Int16}}}()
        if haskey(oct, lc - 1)             # criterion octants, parents by origin
            origins, bits = oct[lc - 1]
            for bi in eachindex(origins)
                b8 = bits[bi]; b8 == 0x00 && continue
                s = get(plev.byorigin, origins[bi], Int32(0))
                s == Int32(0) && continue  # parent vanished this pass → dropped
                for oo in 0:7
                    (b8 >> oo) & 0x01 == 0x01 || continue
                    off = ntuple(d -> Int16(((oo >> (d - 1)) & 1) * Bh), 3)
                    want[child_origin(origins[bi], off, Pc)] = (s, off)
                end
            end
        end
        if haskey(T, lc)                   # projection boxes → parent lattice tiles
            g = R + nbuf                   # painted boxes were dilated by nbuf too
            Pp = level_period(hier.nbase, lc - 1)
            for t in T[lc]
                rng = ntuple(d -> fld(t[d] - g, Bh):fld(t[d] + Bh + g - 1, Bh), 3)
                for tk in rng[3], tj in rng[2], ti in rng[1]
                    τ = (wrapc(Int128(ti) * Bh, Pp[1]), wrapc(Int128(tj) * Bh, Pp[2]),
                         wrapc(Int128(tk) * Bh, Pp[3]))
                    porg = ntuple(d -> (τ[d] ÷ UInt128(hier.B)) * UInt128(hier.B), 3)
                    s = get(plev.byorigin, porg, Int32(0))
                    s == Int32(0) && continue          # unowned space → dropped
                    off = ntuple(d -> Int16(Int(τ[d] - porg[d])), 3)
                    want[child_origin(porg, off, Pc)] = (s, off)
                end
            end
        end
        changed[lc + 1] = _rebuild_level!(hier, lc, want) > 0
    end
    # Morton slot compaction of the churned levels, coarse→fine (parent remaps
    # cascade), BEFORE the table rebuild — every table references slot ids, and
    # the changed-flag propagation below already re-dirties both l and l+1.
    if compact
        for lc in 1:Ltarget
            changed[lc + 1] && compact_level!(hier, lc)
        end
    end
    for l in 0:length(hier.levels)-1
        if changed[l + 1] || (l >= 1 && changed[l])
            build_level_tables!(hier, l)
            l >= 1 && build_cf_register!(hier, l)
        end
    end
    # φ ring maintenance, coarse→fine (children prolong from refreshed parent
    # rings) — see refresh_phi_ghosts! for why this cannot wait for the solves.
    for l in 0:length(hier.levels)-1
        (changed[l + 1] || (l >= 1 && changed[l])) && refresh_phi_ghosts!(hier, l)
    end
    return nothing
end

"Rebuild one level's block set from its `want` footprint map; returns adds+frees."
function _rebuild_level!(hier::AMRHierarchy, lc::Int,
                         want::Dict{Origin,Tuple{Int32,NTuple{3,Int16}}})
    plev = hier.levels[lc]
    lev  = ensure_level!(hier, lc)
    # persistence: free vanished, create missing (data via interior prolongation)
    nchg = 0
    for s in copy(lev.live)
        haskey(want, lev.meta[s].origin) || (remove_block!(hier, lc, s); nchg += 1)
    end
    news = Int32[]
    for (org, (p, off)) in want
        haskey(lev.byorigin, org) && continue
        push!(news, add_block!(hier, lc, p, Int.(off)))
        nchg += 1
    end
    if !isempty(news)
        # recycled slots first: kill the dead block's data in EVERY pool —
        # readers before the next ghost fill (the news prolongation stencil
        # one cell into a NEW parent's ghost ring; the children's Dirichlet
        # φ refill) must never see it.
        zero_slots!(lev, news)
        # new blocks inherit their parent's class scales (power-of-two, so the
        # prolongation ratio is exact); update_scales! re-windows them later.
        hDsc = Array(lev.Dsc); hSsc = Array(lev.Ssc); hEsc = Array(lev.Esc)
        pDsc = Array(plev.Dsc); pSsc = Array(plev.Ssc); pEsc = Array(plev.Esc)
        for s in news
            p = lev.meta[s].parent
            hDsc[s] = pDsc[p]; hSsc[s] = pSsc[p]; hEsc[s] = pEsc[p]
        end
        copyto!(lev.Dsc, hDsc); copyto!(lev.Ssc, hSsc); copyto!(lev.Esc, hEsc)
        tab = to_device_table(lev.be, build_prolong_jobs(lev, plev; interior = true,
                                                         only_slots = news))
        for (fd, fR, scd, scs) in zip(gasfields(lev), gasfields(plev),
                                      classes(lev), classes(plev))
            _run_prolong!(lev.be, tab, fd, fR, fR, 0.0f0, lev.nd, lev.stride, scd, scs)
        end
        for (sd, sR) in zip(lev.sp, plev.sp)
            _run_prolong!(lev.be, tab, sd, sR, sR, 0.0f0, lev.nd, lev.stride)
        end
        # φ seed: recycled slots carry a DEAD block's stale potential — a
        # garbage warm start the RB sweeps cannot recover from regionally
        # (the bamr256L10 Tau runaway ~10 steps after heavy de-refinement:
        # wrong φ → runaway kicks → f16 Tau rot, invisible to max_signal).
        # Prolong the parent's converged φ instead; Dirichlet+sweeps refine it.
        _run_prolong!(lev.be, tab, lev.phi, plev.phi, plev.phi, 0.0f0,
                      lev.nd, lev.stride)
        for s in news
            lev.meta[s].flags &= ~FLAG_NEW
        end
    end
    return nchg
end
