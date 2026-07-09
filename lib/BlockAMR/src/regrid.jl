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
    # zoom: for levels ≥ zoom_lmin, keep only blocks whose center lies within
    # zoom_r (box units, periodic) of zoom_center — targeted deep refinement
    # of one object without paying box-wide breadth at every level.
    zoom_center :: NTuple{3,Float64} = (0.0, 0.0, 0.0)
    zoom_r      :: Float64 = 0.0     # 0 = zoom disabled
    zoom_lmin   :: Int = 3
    # dm_criterion: refine on the TOTAL matter density (gas + CIC-deposited DM)
    # instead of gas alone, so DM-dominated minihalos refine on their own
    # overdensity.  The DM path uses a QUASI-LAGRANGIAN per-level threshold
    # (×RefineBy^ndim=8 per level, NOT `lfac`): `dthresh` is the particle-per-
    # base-cell count that triggers (RAMSES m_refine, ~8), robust to CIC shot
    # noise at depth.  Needs `parts` + `mass_code` passed to regrid!.
    dm_criterion :: Bool = false
    # ZOOM STATIC MESH (Enzo MustRefineParticles / RAMSES mass_cut_refine):
    # for levels l < must_refine_lmax, flag cells by PRESENCE of the (fine, zoom-
    # region) particles — density > must_refine_thr·(mean) — NOT by overdensity.
    # This keeps the whole IC/zoom region refined to must_refine_lmax (the IC
    # finest level), following the particles as they collapse and NEVER de-
    # refining where they are.  Adaptive `dthresh·8^l` refinement then applies
    # only at l ≥ must_refine_lmax (in the finest box).  0 = disabled (all-adaptive).
    must_refine_lmax :: Int = 0
    must_refine_thr  :: Float64 = 0.5     # fraction of mean fine-region density
    # JEANS REFINEMENT (Truelove): resolve the local Jeans length by ≥ jeans_ncells
    # cells — flag cell if λ_J < jeans_ncells·dx_l.  In the code's super-comoving
    # units (∇²φ = 1.5Ωm·a·δ ⇒ G_sc = 3Ωm·a/8π), c_s² = γ(γ-1)·Ge·Esc/ρ gives
    # λ_J² = [8π²γ(γ-1)/(3Ωm a)]·Ge·Esc/ρ².  The a-dependent coefficient is passed
    # to regrid! each step as `jeans_coef` (the driver knows a); this holds only the
    # cell target.  OR'd with the density/DM criterion, so halos still form on
    # overdensity while cooling cores go deep on Jeans.  0 = disabled.
    jeans_ncells :: Float64 = 0.0
    # Jeans DENSITY FLOOR (gas code density): apply the Jeans test ONLY where
    # ρ_gas > jeans_rho_floor.  In super-comoving units the smooth high-z IGM is
    # itself Jeans-"under-resolved" at 256 cells (cold gas, small c_s in these
    # units), so an ungated Jeans criterion refines the whole box.  Gating by
    # overdensity confines deep Jeans refinement to the COLLAPSING core (a cooling
    # halo), the physical target.  0 = no floor.
    jeans_rho_floor :: Float64 = 0.0
end

@kernel function _flag_k!(flags, count, @Const(D), @Const(dm), @Const(live_d), @Const(Dsc),
                          @Const(Ge), @Const(Gsc), thresh::Float32, dmmul::Float32,
                          jcoef::Float32, jdx2::Float32, jfloor::Float32,
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
        # dmmul=0 ⇒ gas-only (dm is a dummy array); dmmul=1 ⇒ TOTAL matter
        # (gas + CIC-deposited DM), the physical collapse criterion.
        dens = Float32(D[idx]) * Dsc[slot] + dmmul * Float32(dm[idx])
        f = dens > thresh
        # Jeans (jcoef>0): refine if λ_J < N_J·dx, i.e. λ_J² < (N_J·dx)².
        # λ_J² = jcoef·(Ge·Esc)/ρ²  ⇒  jcoef·Ge·Esc < jdx2·ρ² (ρ = gas density).
        if jcoef > 0.0f0
            ρg = Float32(D[idx]) * Dsc[slot]
            # gate by the density floor: only refine on Jeans in dense/collapsing gas
            f |= (ρg > jfloor) && ((jcoef * Float32(Ge[idx]) * Gsc[slot]) < (jdx2 * ρg * ρg))
        end
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
function _criterion_flags(hier::AMRHierarchy, l::Int, pol::BlockRefinementPolicy;
                          parts = nothing, mass_code = 0.0, jeans_coef = 0.0)
    lev = hier.levels[l + 1]
    flags = device_zeros(lev.be, UInt8, (lev.cap * lev.stride,))
    count = device_zeros(lev.be, Int32, (lev.cap,))
    # DM-driven refinement: deposit DM onto THIS level and flag on the TOTAL
    # matter density (gas + CIC-DM), so DM-dominated minihalos refine on their
    # own overdensity, not only where gas has traced them.
    #
    # QUASI-LAGRANGIAN per-level threshold (RAMSES poisson_flag `nref ≥ m_refine`,
    # Enzo Grid_FlagCellsToBeRefinedByMass): `dm` is the CIC density over the
    # LOCAL level-l cell volume (deposit_particles_level! uses mass_code/dx_l³),
    # so the mean occupancy per cell drops ×RefineBy^ndim=8 each level and a
    # single stray particle already reads ≈8^l × mean.  A FLAT overdensity
    # threshold would therefore flag shot noise everywhere at depth (the z≈59,
    # 7191-L3-block pathology).  Scaling the threshold ×8 per level makes it a
    # CONSTANT-cell-mass / constant-particle-count test — `dthresh` is the
    # particle-per-(base-)cell count that triggers, unchanged across levels,
    # exactly RAMSES's mass criterion relative to the (level) mean density.  The
    # gas-only path keeps its own `lfac` (gas density is smooth, not shot-noisy).
    usedm = pol.dm_criterion && parts !== nothing
    if usedm
        deposit_particles_level!(hier, l, parts; mass_code)          # fills lev.dm
        dmarr = lev.dm; dmmul = 1.0f0
        # Static zoom mesh: below the IC finest level, flag by PRESENCE (fine-
        # region density > must_refine_thr·mean) so the whole zoom region is
        # kept refined to must_refine_lmax and never de-refines (Enzo/RAMSES
        # must-refine).  At/above it, the adaptive overdensity threshold.
        thr = (pol.must_refine_lmax > 0 && l < pol.must_refine_lmax) ?
              Float32(pol.must_refine_thr) : Float32(pol.dthresh * 8.0^l)  # 8 = RefineBy^ndim
    else
        dmarr = lev.D; dmmul = 0.0f0                                 # gas-only (dummy)
        thr = Float32(pol.dthresh * pol.lfac^l)
    end
    # Jeans: enabled only for a gas run (Ge allocated) with a positive coefficient
    # and cell target.  jdx2 = (N_J·dx_l)²; the a-dependent physics is in jeans_coef.
    usejeans = pol.jeans_ncells > 0 && jeans_coef > 0 && !isempty(lev.Ge)
    gearr  = usejeans ? lev.Ge  : lev.D
    escarr = usejeans ? lev.Gsc : lev.Dsc     # Ge decodes on its own scale Gsc
    jcoef  = usejeans ? Float32(jeans_coef) : 0.0f0
    jdx2   = usejeans ? Float32((pol.jeans_ncells * level_dx(hier, l))^2) : 0.0f0
    jfloor = Float32(pol.jeans_rho_floor)
    _flag_k!(lev.be)(flags, count, lev.D, dmarr, lev.live_d, lev.Dsc, gearr, escarr,
                     thr, dmmul, jcoef, jdx2, jfloor,
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
    # gravity_only (no O-twins): skip the gas/species permute entirely — D is 0 in
    # every slot (no hydro), so its slot order is irrelevant; φ/dm/scales below still
    # compact.  Otherwise permute the gas + species R fields through their O-twins.
    if !isempty(lev.Do)
        for (r, o) in zip(gasfields(lev), gasfields_o(lev))
            _permute_slots!(lev.be, o, r, perm_d, n, lev.stride)
        end
        for (r, o) in zip(lev.sp, lev.spo)
            _permute_slots!(lev.be, o, r, perm_d, n, lev.stride)
        end
        swap_buffers!(lev)                         # permuted O twins become R
    end
    _permute_slots!(lev.be, lev.rhs, lev.phi, perm_d, n, lev.stride)
    lev.phi, lev.rhs = lev.rhs, lev.phi
    if !isempty(lev.dm)                            # dm is lazy (deposit levels only)
        _permute_slots!(lev.be, lev.rhs, lev.dm, perm_d, n, lev.stride)
        lev.dm, lev.rhs = lev.rhs, lev.dm          # rhs ends as scratch garbage
    end
    # per-block scales (host-side; power-of-two, so the move is exact)
    for f in (:Dsc, :Ssc, :Esc, :Gsc)
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
function regrid!(hier::AMRHierarchy, pol::BlockRefinementPolicy; compact::Bool = true,
                 parts = nothing, mass_code = 0.0, jeans_coef = 0.0)
    Ltarget = min(pol.lmax, hier.Lcap)
    nbuf = pol.nbuf
    @assert nbuf <= hier.ng "nbuf=$nbuf > ng=$(hier.ng): device dilation reads the ghost frame"
    R  = 2 + cld(nbuf, 2)                  # projection rim (nesting margin)
    Bh = hier.B ÷ 2
    ltop = min(Ltarget - 1, length(hier.levels) - 1)   # deepest level that clusters
    _t0 = time()
    # 1) criterion flags on device, levels 0..Ltarget−1 (the deepest level's
    #    flags have no children to force and are never consumed)
    Fd  = Dict{Int,Any}()
    cnt = Dict{Int,Vector{Int32}}()
    for l in 0:ltop
        lev = hier.levels[l + 1]
        isempty(lev.live) && continue
        Fd[l], cnt[l] = _criterion_flags(hier, l, pol; parts, mass_code, jeans_coef)
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
    tFLAG = time() - _t0; _t1 = time()
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
        if pol.zoom_r > 0 && lc >= pol.zoom_lmin
            # drop wants outside the zoom sphere; their descendants drop
            # automatically next level (parent vanished → byorigin miss).
            # NESTING-SAFE radius: shrink the sphere with DEPTH so each level's
            # blocks + their ghost shell stay strictly inside the parent's sphere.
            # Otherwise a child near the sphere edge needs a parent NEIGHBOR that
            # the filter dropped at lc-1 → "prolongation under-covered".  The margin
            # per level is the parent (B/2 + ng + nbuf) footprint in box units; it
            # is geometric in depth, so the total shrink is bounded (~2× the coarsest).
            # A child at the sphere edge needs a WHOLE parent-block neighbor for its
            # ghost prolongation, so the per-level margin is a full parent block
            # (B) + ghost (ng) + criterion buffer (nbuf), in box units at lc-1.
            margin = 0.0
            for k in (pol.zoom_lmin + 1):lc
                margin += (hier.B + hier.ng + pol.nbuf) / (hier.nbase[1] * 2.0^(k - 1))
            end
            reff = max(pol.zoom_r - margin, 0.0)
            r2 = reff^2
            filter!(want) do kv
                org = kv.first
                d2 = 0.0
                for d in 1:3
                    x = (Float64(org[d]) + hier.B / 2) / Float64(Pc[d])
                    δx = abs(x - pol.zoom_center[d])
                    δx = min(δx, 1.0 - δx)         # periodic
                    d2 += δx * δx
                end
                d2 <= r2
            end
        end
        changed[lc + 1] = _rebuild_level!(hier, lc, want) > 0
    end
    tRB = time() - _t1
    # Morton slot compaction of the churned levels, coarse→fine (parent remaps
    # cascade), BEFORE the table rebuild — every table references slot ids, and
    # the changed-flag propagation below already re-dirties both l and l+1.
    _prof = haskey(ENV, "BAM_RGPROF")
    tCP = @elapsed if compact
        for lc in 1:Ltarget
            changed[lc + 1] && compact_level!(hier, lc)
        end
    end
    tTB = @elapsed for l in 0:length(hier.levels)-1
        if changed[l + 1] || (l >= 1 && changed[l])
            build_level_tables!(hier, l)
            l >= 1 && build_cf_register!(hier, l)
        end
    end
    # φ ring maintenance, coarse→fine (children prolong from refreshed parent
    # rings) — see refresh_phi_ghosts! for why this cannot wait for the solves.
    tPH = @elapsed for l in 0:length(hier.levels)-1
        (changed[l + 1] || (l >= 1 && changed[l])) && refresh_phi_ghosts!(hier, l)
    end
    _prof && (println("  regrid phases: flag=$(round(tFLAG,digits=2)) rebuild=$(round(tRB,digits=2)) compact=$(round(tCP,digits=2)) tables=$(round(tTB,digits=2)) phi=$(round(tPH,digits=2)) s"); flush(stdout))
    return nothing
end

"Rebuild one level's block set from its `want` footprint map; returns adds+frees."
function _rebuild_level!(hier::AMRHierarchy, lc::Int,
                         want::Dict{Origin,Tuple{Int32,NTuple{3,Int16}}})
    plev = hier.levels[lc]
    lev  = ensure_level!(hier, lc)
    # persistence: free vanished, create missing (data via interior prolongation).
    # BATCHED: per-block sync_live! (device upload of `live`) and the remove-scan
    # (findall over `live`) are each O(nlive) — doing them per add/remove made a
    # bulk regrid O(blocks²) (the 15 s L13 build).  Instead defer both: unregister +
    # free vanished, rebuild `live` in ONE filter!, add with sync=false, sync ONCE.
    nchg = 0
    nremoved = 0
    for s in lev.live
        if !haskey(want, lev.meta[s].origin)
            remove_block!(hier, lc, s; sync = false, defer_live = true); nremoved += 1
        end
    end
    if nremoved > 0
        filter!(s -> isalive(lev.meta[s]), lev.live)   # one O(nlive) pass drops the freed slots
        nchg += nremoved
    end
    news = Int32[]
    for (org, (p, off)) in want
        haskey(lev.byorigin, org) && continue
        push!(news, add_block!(hier, lc, p, Int.(off); sync = false))
        nchg += 1
    end
    (nremoved > 0 || !isempty(news)) && sync_live!(lev)   # single device upload
    if !isempty(news)
        # recycled slots first: kill the dead block's data in EVERY pool —
        # readers before the next ghost fill (the news prolongation stencil
        # one cell into a NEW parent's ghost ring; the children's Dirichlet
        # φ refill) must never see it.
        zero_slots!(lev, news)
        # new blocks inherit their parent's class scales (power-of-two, so the
        # prolongation ratio is exact); update_scales! re-windows them later.
        hDsc = Array(lev.Dsc); hSsc = Array(lev.Ssc); hEsc = Array(lev.Esc); hGsc = Array(lev.Gsc)
        pDsc = Array(plev.Dsc); pSsc = Array(plev.Ssc); pEsc = Array(plev.Esc); pGsc = Array(plev.Gsc)
        for s in news
            p = lev.meta[s].parent
            hDsc[s] = pDsc[p]; hSsc[s] = pSsc[p]; hEsc[s] = pEsc[p]; hGsc[s] = pGsc[p]
        end
        copyto!(lev.Dsc, hDsc); copyto!(lev.Ssc, hSsc); copyto!(lev.Esc, hEsc); copyto!(lev.Gsc, hGsc)
        tab = to_device_table(lev.be, build_prolong_jobs(lev, plev; interior = true,
                                                         only_slots = news))
        for (fd, fR, scd, scs) in zip(gasfields(lev), gasfields(plev),
                                      classes(lev), classes(plev))
            isempty(fd) && continue     # gravity_only: unused hydro pool (zero-size) — skip
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
