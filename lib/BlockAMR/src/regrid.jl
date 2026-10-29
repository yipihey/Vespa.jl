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

# One-time warnings when the nesting-safe zoom margin has eaten the effective
# radius below a block (refinement geometrically impossible at that level — a
# SILENT collapse-killer otherwise; cost hours on the BB collapse).  Keyed by
# level so a long run warns once per depth, not every regrid.
const _ZOOM_WARNED = Set{Int}()
# level-birth registry (root step at which each level last became nonempty) —
# drives the newlev_delay cascade rate-limit; per-process (fresh each run)
const _LEV_BORN = Dict{Int,Int}()

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
    # SECOND, TIGHTER sphere (Enzo nested MustRefineRegion): for levels ≥ zoom_lmin2,
    # ALSO require the block center within zoom_r2 of zoom_center — confines DEEP
    # refinement strictly inside the IC high-res region while the (broader) primary
    # sphere handles the coarse levels whose block+ghost margin exceeds that region.
    zoom_r2     :: Float64 = 0.0     # 0 = second sphere disabled
    zoom_lmin2  :: Int = 999
    # dm_criterion: refine on the TOTAL matter density (gas + CIC-deposited DM)
    # instead of gas alone, so DM-dominated minihalos refine on their own
    # overdensity.  The DM path uses a QUASI-LAGRANGIAN per-level threshold
    # (×RefineBy^ndim=8 per level, NOT `lfac`): `dthresh` is the particle-per-
    # base-cell count that triggers (RAMSES m_refine, ~8), robust to CIC shot
    # noise at depth.  Needs `parts` + `mass_code` passed to regrid!.
    dm_criterion :: Bool = false
    # ADDITIVE gas+DM mode (dm_thresh > 0 with dm_criterion=true): the GAS
    # criterion `D > dthresh·lfac^l` stays active AND the DM criterion flags on
    # total matter > dm_thresh·8^l (dm_thresh = particles per base cell, the
    # RAMSES m_refine / Enzo per-level-adjusted overdensity).  dm_thresh = 0
    # keeps the historical REPLACE semantics (dthresh is the DM threshold, gas
    # term off) for compatibility.
    dm_thresh :: Float64 = 0.0
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
    # COLD-T Jeans density floor (code gas density): the fixed-T insurance term
    # applies only above this (the live-T term uses jeans_rho_floor).  Keep it
    # at the runaway threshold (~1e4× gas mean) so the virialization era stays
    # on live-T + ladders (fs14 lesson).
    jeans_rho_floor2 :: Float64 = 0.0
    # Jeans DENSITY FLOOR (gas code density): apply the Jeans test ONLY where
    # ρ_gas > jeans_rho_floor.  In super-comoving units the smooth high-z IGM is
    # itself Jeans-"under-resolved" at 256 cells (cold gas, small c_s in these
    # units), so an ungated Jeans criterion refines the whole box.  Gating by
    # overdensity confines deep Jeans refinement to the COLLAPSING core (a cooling
    # halo), the physical target.  0 = no floor.
    jeans_rho_floor :: Float64 = 0.0
    # cascade rate-limit: a NEW level may only form once its parent has evolved
    # ≥ newlev_delay ROOT steps since the parent's own creation.  Prolonged
    # (formerly PC, now limited-linear) fill data must relax under the hydro
    # before being refined again — at high Jeans resolution the cascade
    # otherwise nests un-evolved prolongation of coarse data level after level
    # (the N_J=256 constant-patch pathology).  0 = off.
    newlev_delay :: Int = 0
end

@kernel function _flag_k!(flags, count, @Const(D), @Const(dm), @Const(live_d), @Const(Dsc),
                          @Const(Ge), @Const(Gsc), thresh::Float32, gthresh::Float32,
                          dmmul::Float32,
                          jcoef::Float32, jdx2::Float32, jfloor::Float32, efix::Float32,
                          jfloor2::Float32,
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
        # ADDITIVE criteria: gas density vs gthresh OR total matter (gas +
        # CIC-deposited DM, dmmul=1) vs thresh.  Either may be Inf32 (off):
        # gas-only runs disable the dm term, historical replace-mode DM runs
        # disable the gas term.
        ρg = Float32(D[idx]) * Dsc[slot]
        dens = ρg + dmmul * Float32(dm[idx])
        f = (ρg > gthresh) | (dens > thresh)
        # THREE-TERM criterion (the fs14 lesson: replacing live-T Jeans with a
        # high-floor cold-T term removed ALL virial-era Jeans refinement — the
        # density ladders top out at L4-L5 on a minihalo, ~65k× overdensity
        # would be needed for L7):
        #   density ladders (above) ⊕ LIVE-T Jeans at jfloor (morphology/
        #   interior resolution — what carried fs9/fs13's approved virial era)
        #   ⊕ COLD-T Jeans (efix at T_J, Enzo JeansRefinementColdTemperature)
        #   at the HIGHER jfloor2 (wall insurance: monotone in density, so
        #   pileup heating can never relax it — the live-T feedback that
        #   rebuilt the walls in fs13).
        # Jeans (jcoef>0): refine if λ_J < N_J·dx, i.e. λ_J² < (N_J·dx)².
        # λ_J² = jcoef·(Ge·Esc)/ρ²  ⇒  jcoef·Ge·Esc < jdx2·ρ² (ρ = gas density).
        if jcoef > 0.0f0
            # live-T term (floor jfloor): interior resolution, tracks the physics
            f |= (ρg > jfloor) &&
                 ((jcoef * Float32(Ge[idx]) * Gsc[slot]) < (jdx2 * ρg * ρg))
            # cold-T term (floor jfloor2): heat-proof deep-collapse insurance
            (efix > 0.0f0 && ρg > jfloor2) &&
                (f |= (jcoef * efix * ρg) < (jdx2 * ρg * ρg))
        end
        flags[idx] = UInt8(f)
        f && (KA.@atomic count[slot] += Int32(1))
    end
end

# ±nbuf box dilation of the criterion flags; reads the ghost frame (filled by
# the sibling exchange, so spill crosses block boundaries and the periodic
# wrap), writes interior only.  Requires nbuf ≤ ng.
@kernel function _dilate_flags_k!(out, @Const(fin), @Const(live_d), nbuf::Int32,
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
        v = UInt8(0)
        kb = nbuf * _kzv(dimv)                       # flat z: no z dilation
        for dk in -kb:kb, dj in -nbuf:nbuf, di in -nbuf:nbuf
            v |= fin[base + _lidx(i + di, j + dj, k + dk, nd)]
        end
        out[base + _lidx(i, j, k, nd)] = v
    end
end

# one thread per (block, octant): any-flag over the Bh³ interior cells of the
# octant — the entire per-cell → per-child-tile clustering reduction.  2-D
# (kz = 0): 4 quadrants of Bh² cells; the octant bit layout is shared (bit 2,
# the z half, is never set).
@kernel function _octant_any_k!(oct, @Const(flags), @Const(live_d),
                                B::Int32, ng::Int32, nd::Int32, stride::Int32,
                                kz::Int32)
    t = @index(Global)
    t0 = Int32(t) - Int32(1)
    no = kz == Int32(1) ? Int32(8) : Int32(4)
    bi = t0 ÷ no; o = t0 % no
    Bh = B ÷ Int32(2)
    @inbounds begin
        slot = live_d[bi + Int32(1)]
        base = (slot - Int32(1)) * stride
        oi = (o & Int32(1)) * Bh + ng
        oj = ((o >> 0x01) & Int32(1)) * Bh + ng
        ok = kz * (((o >> 0x02) & Int32(1)) * Bh + ng)
        v = UInt8(0)
        for k in Int32(0):kz * (Bh - Int32(1)), j in Int32(0):Bh-Int32(1), i in Int32(0):Bh-Int32(1)
            v |= flags[base + _lidx(oi + i, oj + j, ok + k, nd)]
        end
        oct[t0 + Int32(1)] = v
    end
end

"""Tile origin (level cells, multiples of Bh) of tile index `ti` per axis — wrapped
on periodic axes, `nothing` if outside the domain on a non-periodic one; the flat
z of 2-D is 0."""
function _tile_in_domain(ti::NTuple{3,<:Integer}, Bh::Int, P::Origin,
                         per::NTuple{3,Bool}, dim::Int)
    out = (Int128(0), Int128(0), Int128(0))
    for d in 1:dim
        x = Int128(ti[d]) * Bh
        if per[d]
            x = Int128(wrapc(x, P[d]))
        elseif !(0 <= x < Int128(P[d]))
            return nothing
        end
        out = Base.setindex(out, x, d)
    end
    return out
end

"Criterion flags for level l: device UInt8 (interior of live blocks) + host counts."
function _criterion_flags(hier::AMRHierarchy, l::Int, pol::BlockRefinementPolicy;
                          parts = nothing, mass_code = 0.0, jeans_coef = 0.0,
                          jeans_efix = 0.0,
                          static_parts = nothing, static_mass_code = 0.0)
    lev = hier.levels[l + 1]
    flags = device_zeros(lev.be, UInt8, (lev.cap * lev.stride,))
    count = device_zeros(lev.be, Int32, (lev.cap,))
    # Static (must-refine) levels flag from a FROZEN reference particle set so the
    # zoom mesh reproduces the IC (MUSIC) grid structure and does NOT drift/grow
    # with the collapsing particles — L<must_refine_lmax rebuilds identically every
    # regrid.  Adaptive levels (l ≥ must_refine_lmax) use the LIVE positions so new
    # refinement follows the collapse inside the frozen zoom region.
    #
    # A MULTI-MASS zoom nests by particle mass: MUSIC level (base+l+1) is refined
    # wherever particles of that mass tier AND finer live, giving progressively
    # larger rectangles at coarser levels (L10 ⊃ L11 ⊃ L12).  `static_parts` may
    # therefore be a Function `l -> parts` (per-level mass tier) as well as a fixed
    # NamedTuple; likewise `static_mass_code`.  Using only the finest tier for every
    # level collapses L1/L2 toward the L12 footprint — the bug this fixes.
    is_mustref = pol.dm_criterion && pol.must_refine_lmax > 0 && l < pol.must_refine_lmax
    sp = static_parts isa Function ? static_parts(l) : static_parts
    sm = static_mass_code isa Function ? static_mass_code(l) : static_mass_code
    dparts = (is_mustref && sp !== nothing) ? sp : parts
    dmcode = (is_mustref && sp !== nothing) ? sm : mass_code
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
    # DM criterion capped at the particle-machinery depth (21-bit lattice key,
    # l ≤ 16 at 512/B16): deep levels' total-matter flags are meaningless anyway
    # (the particle tier saturates its resolution far shallower; PSPREAD caps
    # the deposit) — and the deposit's block lookup hard-errors past the key
    # (the fs16/fs16r L17-era crash, the first runs to survive that deep).
    usedm = pol.dm_criterion && dparts !== nothing && l <= particle_lmax(hier)
    usedm && hier.dim != 3 && error("BlockAMR 2-D: the DM refinement criterion is 3-D only")
    additive = usedm && pol.dm_thresh > 0
    if usedm
        deposit_particles_level!(hier, l, dparts; mass_code = dmcode)  # fills lev.dm
        dmarr = lev.dm; dmmul = 1.0f0
        # Static zoom mesh: below the IC finest level, flag by PRESENCE (fine-
        # region density > must_refine_thr·mean) so the whole zoom region is
        # kept refined to must_refine_lmax and never de-refines (Enzo/RAMSES
        # must-refine).  At/above it, the adaptive overdensity threshold.
        thr = is_mustref ? Float32(pol.must_refine_thr) :
              Float32((additive ? pol.dm_thresh : pol.dthresh) * 8.0^l)  # 8 = RefineBy^ndim
        # additive mode: the gas criterion runs alongside the DM one; replace
        # mode (dm_thresh=0, historical): gas term disabled.
        gthr = additive ? Float32(pol.dthresh * pol.lfac^l) : Inf32
    else
        dmarr = lev.D; dmmul = 0.0f0                                 # gas-only (dummy)
        thr = Inf32
        gthr = Float32(pol.dthresh * pol.lfac^l)
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
                     thr, gthr, dmmul, jcoef, jdx2, jfloor, Float32(jeans_efix),
                     Float32(pol.jeans_rho_floor2),
                     Int32(lev.B), Int32(lev.ng), Int32(lev.nd), Int32(lev.stride),
                     Val(lev.dim); ndrange = length(lev.live) * _nact(lev))
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
        for (r, o) in zip(mfields(lev), mfields_o(lev))
            isempty(r) || _permute_slots!(lev.be, o, r, perm_d, n, lev.stride)
        end
        swap_buffers!(lev)                         # permuted O twins become R
        # species σΔ residuals follow their cells (permute through the rhs
        # scratch; leaving them behind would inject wrong-cell remainders)
        for q in eachindex(lev.spr)
            isempty(lev.spr[q]) && continue
            _permute_slots!(lev.be, lev.rhs, lev.spr[q], perm_d, n, lev.stride)
            lev.spr[q], lev.rhs = lev.rhs, lev.spr[q]
        end
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
                 parts = nothing, mass_code = 0.0, jeans_coef = 0.0, jeans_efix = 0.0,
                 static_parts = nothing, static_mass_code = 0.0)
    Ltarget = min(pol.lmax, hier.Lcap)
    nbuf = pol.nbuf
    @assert nbuf <= hier.ng "nbuf=$nbuf > ng=$(hier.ng): device dilation reads the ghost frame"
    R  = 2 + cld(nbuf, 2)                  # projection rim (nesting margin)
    Bh = hier.B ÷ 2
    dim = hier.dim; per = periodic_axes(hier)
    nO = dim == 3 ? 8 : 4                  # child octants (2-D: quadrants) per block
    ltop = min(Ltarget - 1, length(hier.levels) - 1)   # deepest level that clusters
    _t0 = time()
    # 1) criterion flags on device, levels 0..Ltarget−1 (the deepest level's
    #    flags have no children to force and are never consumed)
    Fd  = Dict{Int,Any}()
    cnt = Dict{Int,Vector{Int32}}()
    for l in 0:ltop
        lev = hier.levels[l + 1]
        isempty(lev.live) && continue
        Fd[l], cnt[l] = _criterion_flags(hier, l, pol; parts, mass_code, jeans_coef,
                                         jeans_efix, static_parts, static_mass_code)
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
            Pp = level_period(hier.nbase, lf - 1, dim)
            for t in T[lf + 1]             # box [t−R, t+Bh+R) in level-lf cells
                rng = ntuple(d -> d > dim ? (0:0) :
                             fld(fld(t[d] - R, 2), Bh):fld(fld(t[d] + Bh + R - 1, 2), Bh), 3)
                for tk in rng[3], tj in rng[2], ti in rng[1]
                    τ = _tile_in_domain((ti, tj, tk), Bh, Pp, per, dim)
                    τ === nothing || push!(tset, τ)
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
                                 Int32(lev.ng), Int32(lev.nd), Int32(lev.stride),
                                 Val(lev.dim); ndrange = length(lev.live) * _nact(lev))
        nb = length(lev.live)
        o = device_zeros(lev.be, UInt8, (nO * nb,))
        _octant_any_k!(lev.be)(o, G, lev.live_d, Int32(lev.B), Int32(lev.ng),
                               Int32(lev.nd), Int32(lev.stride), _kz(lev); ndrange = nO * nb)
        ho = Array(o)
        bits = zeros(UInt8, nb)
        for bi in 1:nb, oo in 0:nO-1
            ho[nO * (bi - 1) + oo + 1] != 0 && (bits[bi] |= UInt8(1) << oo)
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
        Pc = level_period(hier.nbase, lc, dim)
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
            Pp = level_period(hier.nbase, lc - 1, dim)
            for t in T[lc]
                rng = ntuple(d -> d > dim ? (0:0) :
                             fld(t[d] - g, Bh):fld(t[d] + Bh + g - 1, Bh), 3)
                for tk in rng[3], tj in rng[2], ti in rng[1]
                    τi = _tile_in_domain((ti, tj, tk), Bh, Pp, per, dim)
                    τi === nothing && continue
                    τ = ntuple(d -> UInt128(τi[d]), 3)
                    porg = ntuple(d -> d > dim ? UInt128(0) :
                                       (τ[d] ÷ UInt128(hier.B)) * UInt128(hier.B), 3)
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
            # ghost prolongation.  The filter tests block CENTERS, so the worst case
            # is the 3D DIAGONAL: a kept child's ghost corner reaches
            # √3·(B/2+ng)·dx_l past its center, and the parent block containing that
            # cell has its center up to √3·(B/2)·dx_{l-1} farther — √3·(3B/4+ng/2)
            # parent cells total (= 22.5 for B=16, ng=2), NOT the linear B+ng.  The
            # linear margin under-covered exactly at a dense sphere-edge shell
            # (haloA run 4: ghost at r=0.1245 vs parent keep-radius 0.1207 →
            # "prolongation under-covered").  √3·(B+ng+nbuf) bounds it with room
            # for the nbuf flag dilation; still geometric in depth, total ≲ 3.5×
            # the coarsest term.
            margin = 0.0
            for k in (pol.zoom_lmin + 1):lc
                margin += sqrt(3.0) * (hier.B + hier.ng + pol.nbuf) /
                          (hier.nbase[1] * 2.0^(k - 1))
            end
            reff = max(pol.zoom_r - margin, 0.0)
            r2 = reff^2
            # guard: the depth-shrinking margin can drive reff below one block
            # half-width, so this level is dropped purely by geometry even where
            # the physics flags refinement — warn (once/level) with the zoom_r
            # that would keep this depth alive, so it never silently stalls.
            blk_hw = (hier.B / 2) / (hier.nbase[1] * 2.0^lc)   # box units at lc
            if !isempty(want) && reff < blk_hw && !(lc in _ZOOM_WARNED)
                push!(_ZOOM_WARNED, lc)
                @warn "BlockAMR zoom margin killed refinement at level $lc: " *
                      "effective radius reff=$(round(reff, sigdigits=3)) box < block " *
                      "half-width $(round(blk_hw, sigdigits=3)) (zoom_r=$(pol.zoom_r), " *
                      "margin=$(round(margin, sigdigits=3))). Increase zoom_r to " *
                      "≳ $(round(margin + blk_hw, sigdigits=3)) or set zoom_r=0 and " *
                      "confine by the Jeans/density criterion instead." maxlog = 1
            end
            filter!(want) do kv
                org = kv.first
                d2 = 0.0
                for d in 1:dim
                    x = (Float64(org[d]) + hier.B / 2) / Float64(Pc[d])
                    δx = abs(x - pol.zoom_center[d])
                    δx = min(δx, 1.0 - δx)         # periodic
                    d2 += δx * δx
                end
                d2 <= r2
            end
        end
        # SECOND (tighter, deep) sphere — same depth-shrinking nesting margin,
        # measured from zoom_lmin2, so DEEP refinement stays inside the IC high-res
        # region.  Composes with the primary sphere (a want must pass BOTH).
        if pol.zoom_r2 > 0 && lc >= pol.zoom_lmin2
            margin2 = 0.0
            for k in (pol.zoom_lmin2 + 1):lc
                margin2 += sqrt(3.0) * (hier.B + hier.ng + pol.nbuf) /
                           (hier.nbase[1] * 2.0^(k - 1))
            end
            r2b = max(pol.zoom_r2 - margin2, 0.0)^2
            filter!(want) do kv
                org = kv.first
                d2 = 0.0
                for d in 1:dim
                    x = (Float64(org[d]) + hier.B / 2) / Float64(Pc[d])
                    δx = abs(x - pol.zoom_center[d]); δx = min(δx, 1.0 - δx)
                    d2 += δx * δx
                end
                d2 <= r2b
            end
        end
        # cascade rate-limit (policy newlev_delay): gate the CREATION of a new
        # level on its parent's age; existing levels are never touched.  At
        # restart the birth registry is empty → all pre-existing levels count
        # as old → the first new deeper level forms immediately, as intended.
        # a not-yet-created level (lc+1 > length) is trivially empty → the
        # newlev gate applies (it IS a brand-new level).  From ICs only level 0
        # exists at the first regrid; the from-restart path preloaded L1+ which
        # masked this out-of-bounds access.
        newlev = lc + 1 > length(hier.levels) || isempty(hier.levels[lc + 1].live)
        if pol.newlev_delay > 0 && !isempty(want) && newlev
            born = get(_LEV_BORN, lc - 1, typemin(Int) ÷ 2)
            Int(hier.nstep[1]) - born < pol.newlev_delay && empty!(want)
        end
        changed[lc + 1] = _rebuild_level!(hier, lc, want) > 0
        if !isempty(hier.levels[lc + 1].live)
            haskey(_LEV_BORN, lc) || (_LEV_BORN[lc] = Int(hier.nstep[1]))
        else
            delete!(_LEV_BORN, lc)
        end
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
    # gravity_only (DMO): the base level's O-twins are zero-size and there is no
    # hydro flux, so the coarse–fine REFLUX register (build_cf_register! / build_cf_faces)
    # is never consumed — skip it.  It is an O(N_fine·B²) host build that otherwise
    # dominates the regrid (~90% of the tables phase at deep DM refinement: ~4 s of
    # a ~4.4 s tables phase at 100k blocks; regrid ~5-6× faster with it skipped).
    grav_only = isempty(hier.levels[1].Do)
    tTB = @elapsed for l in 0:length(hier.levels)-1
        if changed[l + 1] || (l >= 1 && changed[l])
            build_level_tables!(hier, l)
            (l >= 1 && !grav_only) && build_cf_register!(hier, l)
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

"""
    grow_level!(hier, l, pol; jeans_coef = 0.0) -> nadded

Enzo/RAMSES-style PER-SUBSTEP refinement (`RebuildHierarchy` in every
`EvolveLevel` step / `refine_fine` in every `amr_step`): flag level `l` on the
gas + Jeans criteria and immediately add any missing level-`l+1` child blocks —
GROW-ONLY, so deeper levels stay nested by construction and no data is ever
destroyed mid-W-cycle.  This closes the refinement-latency gap that builds the
Truelove-wall: with root-cadence regridding the deepest level evolves up to
`REGRIDN` root steps (2^l substeps each) of pileup before the next level can
appear — a free-fall time of over-compression, grid-scale shocking to 10⁵-10⁶ K
and collisional-ionization relics (the super-ceiling xe zones).  Called at the
top of every `advance_level_w!` substep, the latency drops to ONE substep of
the flagging level, and a collapse can cascade several new levels within a
single root step (the recursion grows `l+2` at the child's own entry).

Deliberately NOT here (root-cadence `regrid!` keeps them): derefinement, the
DM criterion (walls are gas-driven; per-substep deposits are wasted work),
cross-parent flag projection, and compaction.  New children are accepted only
where the level-`l` lattice covers their ghost-grown footprint (always true at
the collapse interior where walls form); edge candidates wait for `regrid!`.
"""
function grow_level!(hier::AMRHierarchy, l::Int, pol::BlockRefinementPolicy;
                     jeans_coef::Real = 0.0, jeans_efix::Real = 0.0,
                     on_newlevel = nothing)
    Ltarget = min(pol.lmax, hier.Lcap)
    l + 1 <= Ltarget || return 0
    l >= pol.must_refine_lmax || return 0     # static IC mesh: root cadence only
    lev = hier.levels[l + 1]
    isempty(lev.live) && return 0
    # gas + Jeans flags (parts = nothing ⇒ the DM term is off in _criterion_flags)
    F, cnt = _criterion_flags(hier, l, pol; parts = nothing, mass_code = 0.0,
                              jeans_coef, jeans_efix)
    sum(cnt) == 0 && return 0
    nbuf = pol.nbuf
    _run_copy!(lev.be, _tabt(lev, :sib), F, F, lev.nd, lev.stride)
    G = device_zeros(lev.be, UInt8, (lev.cap * lev.stride,))
    _dilate_flags_k!(lev.be)(G, F, lev.live_d, Int32(nbuf), Int32(lev.B),
                             Int32(lev.ng), Int32(lev.nd), Int32(lev.stride),
                             Val(lev.dim); ndrange = length(lev.live) * _nact(lev))
    nb = length(lev.live)
    nO = lev.dim == 3 ? 8 : 4
    o = device_zeros(lev.be, UInt8, (nO * nb,))
    _octant_any_k!(lev.be)(o, G, lev.live_d, Int32(lev.B), Int32(lev.ng),
                           Int32(lev.nd), Int32(lev.stride), _kz(lev); ndrange = nO * nb)
    ho = Array(o)
    Bh = hier.B ÷ 2
    Pc = level_period(hier.nbase, l + 1, hier.dim)
    clev = l + 2 <= length(hier.levels) ? hier.levels[l + 2] : nothing
    # zoom-sphere radii with the standard depth-shrinking nesting margins
    reff2 = Inf
    if pol.zoom_r > 0 && l + 1 >= pol.zoom_lmin
        margin = sum(sqrt(3.0) * (hier.B + hier.ng + nbuf) /
                     (hier.nbase[1] * 2.0^(k - 1)) for k in (pol.zoom_lmin + 1):(l + 1);
                     init = 0.0)
        reff2 = max(pol.zoom_r - margin, 0.0)^2
    end
    reff2b = Inf
    if pol.zoom_r2 > 0 && l + 1 >= pol.zoom_lmin2
        margin2 = sum(sqrt(3.0) * (hier.B + hier.ng + nbuf) /
                      (hier.nbase[1] * 2.0^(k - 1)) for k in (pol.zoom_lmin2 + 1):(l + 1);
                      init = 0.0)
        reff2b = max(pol.zoom_r2 - margin2, 0.0)^2
    end
    insphere(corg, r2) = begin
        r2 === Inf && return true
        d2 = 0.0
        for d in 1:hier.dim
            x = (Float64(corg[d]) + hier.B / 2) / Float64(Pc[d])
            δx = abs(x - pol.zoom_center[d]); δx = min(δx, 1.0 - δx)
            d2 += δx * δx
        end
        d2 <= r2
    end
    want = Dict{Origin,Tuple{Int32,NTuple{3,Int16}}}()
    nnew = 0
    for (bi, s) in enumerate(lev.live)
        bits = UInt8(0)
        for oo in 0:nO-1
            ho[nO * (bi - 1) + oo + 1] != 0 && (bits |= UInt8(1) << oo)
        end
        bits == 0x00 && continue
        org = lev.meta[s].origin
        for oo in 0:nO-1
            (bits >> oo) & 0x01 == 0x01 || continue
            off = ntuple(d -> Int16(((oo >> (d - 1)) & 1) * Bh), 3)
            corg = child_origin(org, off, Pc)
            clev !== nothing && haskey(clev.byorigin, corg) && continue
            (insphere(corg, reff2) && insphere(corg, reff2b)) || continue
            # margin: the child's ghost-grown footprint (±2 level-l cells around
            # [org+off, org+off+Bh)) must be lattice-covered at level l
            ok = true
            for ck in (lev.dim == 3 ? (-2, Bh + 1) : (0,)), cj in (-2, Bh + 1), ci in (-2, Bh + 1)
                qr = ntuple(d -> Int128(org[d]) + Int128(off[d]) + Int128((ci, cj, ck)[d]), 3)
                # beyond a non-periodic domain face nothing needs covering
                any(d -> !lev.per[d] && !(0 <= qr[d] < Int128(lev.P[d])), 1:3) && continue
                q = ntuple(d -> UInt128(wrapc(qr[d], lev.P[d])), 3)
                if lattice_owner(lev, q, hier.B) == Int32(0)
                    ok = false; break
                end
            end
            ok || continue
            want[corg] = (s, off)
            nnew += 1
        end
    end
    nnew == 0 && return 0
    # Enzo-style INTERPOLATED OUTPUT hook: the deepest, most-informative state a
    # run ever holds is the instant just BEFORE a new level is born (the old
    # deepest level is fully evolved and NOT yet superseded by a wall-adjacent
    # newcomer).  The callback fires exactly once per level birth, before any
    # topology change; levels sit within one coarse substep of each other
    # (fine-first W-cycle), which is the same skew Enzo's interpolated outputs
    # carry.  Restart checkpoints stay at root cadence — this is for outputs.
    if on_newlevel !== nothing && (clev === nothing || isempty(clev.live))
        on_newlevel("preL$(l + 1)")
    end
    if clev !== nothing            # persistence: existing children must survive
        for cs in clev.live
            m = clev.meta[cs]
            want[m.origin] = (m.parent, m.offset)
        end
    end
    _rebuild_level!(hier, l + 1, want)
    haskey(_LEV_BORN, l + 1) || (_LEV_BORN[l + 1] = Int(hier.nstep[1]))
    build_level_tables!(hier, l + 1)
    build_cf_register!(hier, l + 1)
    if l + 3 <= length(hier.levels) && !isempty(hier.levels[l + 3].live)
        build_level_tables!(hier, l + 2)   # child tables reference the grown parent
        build_cf_register!(hier, l + 2)
    end
    refresh_phi_ghosts!(hier, l + 1)
    return nnew
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
        # BAM_EVAR=internal: new blocks get Tau REBUILT from prolonged Ge+KE
        # (RAMSES interpol_var=1), not interpolated as the KE-dominated total.
        evar_int = get(ENV, "BAM_EVAR", "total") == "internal" &&
                   !isempty(gasfields(lev)[5])
        for (fi, (fd, fR, scd, scs)) in enumerate(zip(gasfields(lev), gasfields(plev),
                                                      classes(lev), classes(plev)))
            isempty(fd) && continue     # gravity_only: unused hydro pool (zero-size) — skip
            (evar_int && fi == 5) && continue
            _run_prolong!(lev.be, tab, fd, fR, fR, 0.0f0, lev.nd, lev.stride, scd, scs;
                          kz = _kz(lev))
        end
        if evar_int && tab.total > 0
            _reconstruct_tau_k!(lev.be)(gasfields(lev)..., lev.Dsc, lev.Ssc,
                                        lev.Esc, lev.Gsc, tab.jobs, tab.cellstart,
                                        Int32(tab.njobs), Int32(lev.nd),
                                        Int32(lev.stride); ndrange = tab.total)
        end
        for (sd, sR) in zip(lev.sp, plev.sp)
            _run_prolong!(lev.be, tab, sd, sR, sR, 0.0f0, lev.nd, lev.stride)
        end
        for (fd, fR) in zip(mfields(lev), mfields(plev))  # MHD: B, ψ limited-linear
            isempty(fd) || _run_prolong!(lev.be, tab, fd, fR, fR, 0.0f0, lev.nd, lev.stride,
                                         lev.Bsc, plev.Bsc; kz = _kz(lev))
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
