# stepping.jl — hierarchy time integration.
#
# Phase 1 (this file, no subcycling yet): every level advances the SAME dt per
# hierarchy step, SSP-RK2, with the coarse–fine machinery live: fine ghosts
# prolong from the parent stage state (θ=0 for stage 1, θ=1 for stage 2 —
# exactly the RK boundary state), C/F fluxes are captured per stage (weight ½
# each), then reflux corrects the uncovered coarse cells and restriction
# overwrites the covered ones.  λ_l = dt/dx_l per level here; the strict-2:1
# W-cycle (Phase 3) collapses them into the single global λ.

# ── phase timers (ENV-gated, sync-accurate) ──────────────────────────────────
# Attribute per-root-step wall time to (phase, level).  OFF by default (zero
# overhead — the _tic guard returns before any sync).  enable_timers!() turns it
# on; each _tic/_toc pair synchronizes the backend so async GPU kernels are
# actually attributed to their phase (perturbs overlap, but that's the point of
# a cost breakdown).  Report with timer_report().
const _TIMERS = Dict{Tuple{Symbol,Int},Float64}()
const _TCOUNT = Dict{Tuple{Symbol,Int},Int}()
const _TON    = Ref(false)
enable_timers!(on::Bool = true) = (_TON[] = on; on)
reset_timers!() = (empty!(_TIMERS); empty!(_TCOUNT); nothing)

@inline function _tic(be)
    _TON[] || return 0.0
    KA.synchronize(be)
    return time()
end
@inline function _toc(be, phase::Symbol, l::Int, t0)
    _TON[] || return nothing
    KA.synchronize(be)
    k = (phase, l)
    _TIMERS[k] = get(_TIMERS, k, 0.0) + (time() - t0)
    _TCOUNT[k] = get(_TCOUNT, k, 0) + 1
    return nothing
end

"Formatted per-(phase,level) timing breakdown since the last reset_timers!()."
function timer_report(; io = stdout)
    isempty(_TIMERS) && (println(io, "[timers] (none)"); return nothing)
    tot = sum(values(_TIMERS))
    println(io, "── phase timers ── total ", round(tot, digits = 3), " s")
    # by phase (summed over levels)
    byphase = Dict{Symbol,Float64}()
    for ((p, _), v) in _TIMERS; byphase[p] = get(byphase, p, 0.0) + v; end
    for (p, v) in sort(collect(byphase); by = x -> -x[2])
        @printf(io, "  %-16s %8.3f s  %5.1f%%\n", p, v, 100v / tot)
    end
    println(io, "  ── by (phase, level) ──")
    for ((p, l), v) in sort(collect(_TIMERS); by = x -> -x[2])
        v / tot < 0.01 && continue
        @printf(io, "  %-16s L%-2d %8.3f s  %5.1f%%  (%d calls)\n",
                p, l, v, 100v / tot, _TCOUNT[(p, l)])
    end
    return nothing
end

"""
    hierarchy_rk2_step!(hier, dt) -> nothing

One synchronous SSP-RK2 step of the whole hierarchy over `dt` (same dt on every
level — the Phase-1/2 integrator).  Final state lands in the R buffers.
"""
function hierarchy_rk2_step!(hier::AMRHierarchy, dt::Real; reflux::Bool = true)
    L = length(hier.levels) - 1
    λl(l) = Float32(dt / level_dx(hier, l))
    for l in 0:L                                     # ── stage 1: O = R − λΔF(R)
        isempty(hier.levels[l+1].live) && continue
        fill_ghosts!(hier, l; θ = 0.0f0, buf = :R)
        stage_level!(hier, l, λl(l); w = 0.0f0, IN = :R, OUT = :O)
        reflux && l >= 1 && capture_cf!(hier, l; buf = :R, wf = 0.5f0, wc = 0.5f0)
    end
    for l in 0:L                                     # ── stage 2: R = ½R + ½(O − λΔF(O))
        isempty(hier.levels[l+1].live) && continue
        fill_ghosts!(hier, l; θ = 1.0f0, buf = :O)
        stage_level!(hier, l, λl(l); w = 0.5f0, IN = :O, OUT = :R)
        reflux && l >= 1 && capture_cf!(hier, l; buf = :O, wf = 0.5f0, wc = 0.5f0)
    end
    for l in L:-1:1                                  # ── couple levels downward
        isempty(hier.levels[l+1].live) && continue
        reflux && reflux_apply!(hier, l, λl(l - 1))
        restrict_level!(hier, l)
    end
    hier.nstep[1] += 1
    return nothing
end

# ── strict 2:1 W-cycle (fine-first, frozen coarse ghosts) ─────────────────────
"""
    compute_lambda!(hier) -> λ::Float32

The ONE global fraction λ = dt_l/dx_l for the whole hierarchy: under strict 2:1
binary subcycling every level satisfies its CFL iff λ ≤ cfl/smax_l for all l, so
λ = cfl / max_l(smax_l).  Stored in `hier.λ`.
"""
function compute_lambda!(hier::AMRHierarchy)
    smax = 0.0f0
    for l in 0:length(hier.levels)-1
        isempty(hier.levels[l+1].live) && continue
        smax = max(smax, max_signal(hier.levels[l+1], hier.gamma))
    end
    # BAM_SMAXCAP (absolute signal cap, code units; 0 = off): cells beyond the
    # cap are numerical garbage the epilogue/reflux guards keep frozen — they
    # must not strangle the global dτ (the observed 470× collapse), or the
    # λ-relative guards ratchet ever hotter.  Clamp the CFL denominator too.
    scap = Float32(parse(Float64, get(ENV, "BAM_SMAXCAP", "0")))
    scap > 0.0f0 && (smax = min(smax, scap))
    hier.λ = Float32(hier.cfl) / max(smax, 1.0f-30)
    return hier.λ
end

"""
    advance_level_w!(hier, l, λ)

One level-`l` step of the strict-2:1 W-cycle over dt_l = λ·dx_l, RAMSES
fine-first: children take their two half-steps FIRST (their coarse ghosts frozen
at this level's current time — the parent R buffer is untouched until after they
finish), then this level's SSP-RK2 substep runs, then the children are restricted
in and the (l, l+1) reflux corrects this level's uncovered boundary cells.
Reflux weights: fine ¼ per fine stage (RK ½ × time-average ½), coarse ½ per
coarse stage; the apply multiplies the single global λ.
"""
# NaN-stage hunt (BAM_NANSTAGE=1): reduce the level's decoded signal after each
# W-cycle stage and abort at the FIRST poisoned one.  A post-advance check only
# sees the NaN after restriction has smeared it across every level (it reports
# "level 0" no matter where it started); this pins (level, stage, substep).
# One device reduction per stage — hunt-mode cost only, off by default.
@inline function _nanstage!(hier::AMRHierarchy, l::Int, stage::Symbol)
    get(ENV, "BAM_NANSTAGE", "0") == "1" || return nothing
    lev = hier.levels[l + 1]
    s = max_signal(lev, hier.gamma)
    ok = isfinite(s)
    if ok
        # max_signal derives cs from one energy field — a lone NaN in the OTHER
        # dual-energy field is invisible to it (chem later converts Ge-NaN into
        # Tau-NaN and only THEN the signal trips, misattributing the stage).
        # Hunt mode: check both energy pools directly (pool-wide count; ghosts
        # and dead slots hold finite data in practice).
        ok = count(x -> !isfinite(Float32(x)), lev.Ge) == 0 &&
             count(x -> !isfinite(Float32(x)), lev.Tau) == 0
    end
    if ok && !isempty(lev.phi)
        # φ poisoning is invisible to the gas gates once the kicks are
        # nonfinite-safe (they absorb NaN ∇φ) — the haloA step-99 meltdown
        # poisoned L2–L9 φ pool-wide while every gas gate stayed green.
        nphi = count(x -> !isfinite(Float32(x)), lev.phi)
        if nphi > 0
            println("NANSTAGE   phi: $nphi nonfinite at level $l after :$stage")
            for (nm, arr) in ((:rhs, lev.rhs), (:dm, lev.dm))
                isempty(arr) && continue
                nb = count(x -> !isfinite(Float32(x)), arr)
                mx = mapreduce(x -> isfinite(Float32(x)) ? abs(Float32(x)) : 0.0f0,
                               max, arr; init = 0.0f0)
                println("NANSTAGE   $nm: nonfinite=$nb  max|finite|=$mx")
            end
            ok = false
        end
    end
    if !ok
        # which field carries it? (counts cover the whole pool incl. ghosts and
        # dead slots — a live-interior nonfinite is already established by smax)
        for (nm, arr) in ((:D, lev.D), (:S1, lev.S1), (:S2, lev.S2), (:S3, lev.S3),
                          (:Tau, lev.Tau), (:Ge, lev.Ge))
            nb = count(x -> !isfinite(Float32(x)), arr)
            nb > 0 && println("NANSTAGE   field $nm: $nb nonfinite (pool-wide)")
        end
        # block forensics: which LIVE blocks, what scales, R (output) AND O
        # (input) values — a dead/freed slot's stale garbage must not misdirect
        # the hunt, and R-NaN with O-finite proves the stage CREATED it while
        # R-NaN with O-NaN proves it PREDATES the stage (input side).
        hge = Array(lev.Ge); htau = Array(lev.Tau); hd = Array(lev.D)
        hgeo = Array(lev.Geo); htauo = Array(lev.Tauo); hdo = Array(lev.Do)
        hDsc = Array(lev.Dsc); hSsc = Array(lev.Ssc)
        hEsc = Array(lev.Esc); hGsc = Array(lev.Gsc)
        vbt = get(lev.tabs, :vb, nothing)
        hvb = vbt === nothing ? nothing : Array(vbt)
        liveset = Set(Int.(lev.live))
        seen = Set{Int}(); ndead = 0
        for ii in eachindex(hge)
            (isfinite(Float32(hge[ii])) && isfinite(Float32(htau[ii]))) && continue
            slot = (ii - 1) ÷ lev.stride + 1
            slot in liveset || (ndead += 1; continue)
            slot in seen && continue
            push!(seen, slot)
            length(seen) > 6 && break
            c = (ii - 1) % lev.stride
            i_ = c % lev.nd; j_ = (c ÷ lev.nd) % lev.nd; k_ = c ÷ (lev.nd * lev.nd)
            inint = all(x -> lev.ng <= x < lev.B + lev.ng, (i_, j_, k_))
            m = lev.meta[slot]
            vbs = hvb === nothing ? "n/a" :
                  string((hvb[3(slot-1)+1], hvb[3(slot-1)+2], hvb[3(slot-1)+3]))
            println("NANSTAGE   LIVE slot $slot origin $(Int128.(m.origin)) ",
                    inint ? "INTERIOR" : "GHOST", " cell ($i_,$j_,$k_)  scales: ",
                    "Dsc=$(hDsc[slot]) Ssc=$(hSsc[slot]) Esc=$(hEsc[slot]) Gsc=$(hGsc[slot])",
                    "  vb=$vbs")
            println("NANSTAGE     R(out): D=$(Float32(hd[ii])) Ge=$(Float32(hge[ii])) Tau=$(Float32(htau[ii]))",
                    "   O(in): D=$(Float32(hdo[ii])) Ge=$(Float32(hgeo[ii])) Tau=$(Float32(htauo[ii]))")
            # LAUNCH-COVERAGE forensics: over this block's ACTIVE cells, how many
            # R cells are (a) nonfinite, (b) bitwise-IDENTICAL to the pre-launch
            # output buffer's content?  A kernel-processed block has R ≠ stale-R
            # nearly everywhere; R≡stale on a swath ⇒ those cells were NEVER
            # WRITTEN by the launch (live_d/coverage bug, not arithmetic).
            base_ = (slot - 1) * lev.stride
            nnf = 0; nact = 0
            for kk in 1:lev.B, jj in 1:lev.B, ii2 in 1:lev.B
                q_ = base_ + ((lev.ng+kk-1)*lev.nd + (lev.ng+jj-1))*lev.nd + (lev.ng+ii2-1) + 1
                nact += 1
                isfinite(Float32(hge[q_])) || (nnf += 1)
            end
            println("NANSTAGE     block actives: $nact  nonfinite-Ge: $nnf")
        end
        ndead > 0 && println("NANSTAGE   (+$ndead nonfinite in DEAD/freed slots — stale, ignored)")
        # live vs live_d consistency (stale device list ⇒ launch skipped blocks)
        hlv = Array(lev.live_d)
        println("NANSTAGE   live(host)=$(length(lev.live))  live_d(dev)=$(length(hlv))  ",
                "same=$(length(lev.live) == length(hlv) && all(Int.(lev.live) .== Int.(hlv[1:length(lev.live)])))")
        println("NANSTAGE   tabs keys: ", sort(collect(keys(lev.tabs))))
        # full-state dump for offline CPU replay (BAM_NANDUMP=path prefix)
        dpre = get(ENV, "BAM_NANDUMP", "")
        if dpre != ""
            dump = (l = l, stage = stage, substep = hier.nstep[l + 1],
                    λ = hier.λ, B = lev.B, ng = lev.ng, nd = lev.nd,
                    stride = lev.stride, cap = lev.cap, nsp = lev.nsp,
                    live = copy(lev.live),
                    origins = [lev.meta[s].origin for s in lev.live],
                    D = Array(lev.D), S1 = Array(lev.S1), S2 = Array(lev.S2),
                    S3 = Array(lev.S3), Tau = Array(lev.Tau), Ge = Array(lev.Ge),
                    Do = Array(lev.Do), S1o = Array(lev.S1o), S2o = Array(lev.S2o),
                    S3o = Array(lev.S3o), Tauo = Array(lev.Tauo), Geo = Array(lev.Geo),
                    sp = [Array(s) for s in lev.sp], spo = [Array(s) for s in lev.spo],
                    Dsc = Array(lev.Dsc), Ssc = Array(lev.Ssc),
                    Esc = Array(lev.Esc), Gsc = Array(lev.Gsc),
                    vb = get(lev.tabs, :vb, nothing) === nothing ? nothing :
                         Array(lev.tabs[:vb]))
            fn = dpre * "_L$(l)_$(stage)_ss$(hier.nstep[l+1]).jls"
            open(fn, "w") do io
                Serialization.serialize(io, dump)
            end
            println("NANSTAGE   dumped $fn")
        end
        flush(stdout)
        error("NANSTAGE: level $l nonfinite after :$stage (substep $(hier.nstep[l + 1]))")
    end
    return nothing
end

# BAM_SPWATCH=<level>: print the watched level's max species codes (decoded)
# after every W-cycle operation — the codec is monotonic, so one device
# reduction per species suffices.  The first jump names the guilty operation.
function _spwatch(hier::AMRHierarchy, l::Int, tag::Symbol)
    wl = parse(Int, get(ENV, "BAM_SPWATCH", "-1"))
    wl < 0 && return nothing
    lev = hier.levels[wl + 1]
    isempty(lev.live) && return nothing
    m1 = decode_log2sp(Float64, maximum(lev.sp[1]))
    m2 = decode_log2sp(Float64, maximum(lev.sp[2]))
    o1 = isempty(lev.spo) ? 0.0 : decode_log2sp(Float64, maximum(lev.spo[1]))
    o2 = isempty(lev.spo) ? 0.0 : decode_log2sp(Float64, maximum(lev.spo[2]))
    @printf("SPWATCH L%d after %s(l=%d ss=%d): R %.5e %.5e | O %.5e %.5e\n",
            wl, tag, l, Int(hier.nstep[l + 1]), m1, m2, o1, o2)
    flush(stdout)
    return nothing
end

# Enzo/RAMSES-style per-substep refinement (see grow_level!): the driver stores
# (pol = policy, jc = jeans_coef) here each root step; nothing = disabled.  A
# Ref because advance_level_w! recurses — the policy travels with the hierarchy
# walk without threading another kwarg through every call site.
const GROWREGRID = Ref{Any}(nothing)

function advance_level_w!(hier::AMRHierarchy, l::Int, λ::Float32;
                          φ = nothing, chem = nothing, selfgrav = nothing)
    lev = hier.levels[l + 1]
    isempty(lev.live) && return nothing
    # per-substep GROW-ONLY refinement: refinement latency = one substep of the
    # level that wants it (Enzo RebuildHierarchy / RAMSES refine_fine cadence) —
    # kills the Truelove-wall over-heating built by root-cadence regridding.
    # The recursion grows l+2 at the child's own entry, so a fast collapse can
    # cascade multiple new levels within one root step.
    g = GROWREGRID[]
    if g !== nothing
        _t = _tic(lev.be)
        na = grow_level!(hier, l, g.pol; jeans_coef = g.jc, jeans_efix = g.ef,
                         on_newlevel = get(g, :dump, nothing))
        _toc(lev.be, :growregrid, l, _t)
        na > 0 && get(ENV, "BAM_GROWVERB", "0") == "1" &&
            (println("GROW L$(l + 1) +$na blocks (substep $(Int(hier.nstep[l + 1])) of L$l)"); flush(stdout))
        # Enzo "outputNow" analog: touch the trigger file in the run directory
        # and the NEXT level substep anywhere in the W-cycle writes an output
        # immediately — investigate a long run whenever you want, without
        # waiting hours for a root-step checkpoint.  The file is removed BEFORE
        # dumping (single-threaded walk ⇒ exactly one dump per touch).
        fpath = get(g, :force, nothing)
        if fpath !== nothing && get(g, :dump, nothing) !== nothing && isfile(fpath)
            rm(fpath; force = true)
            g.dump("forced")
        end
    end
    haskids = l + 2 <= length(hier.levels) && !isempty(hier.levels[l + 2].live)
    if haskids                                    # fine first (frozen parent state)
        # out-of-core: keep THIS (parent) level resident while its children run —
        # they read its φ/gas for C/F ghost prolongation + Dirichlet gravity BCs.
        prefetch_level!(lev)
        advance_level_w!(hier, l + 1, λ; φ, chem, selfgrav)
        advance_level_w!(hier, l + 1, λ; φ, chem, selfgrav)
    end
    # out-of-core: pull THIS level's pools onto the GPU right before its kernels
    # (children ran first and may have evicted it).  Cold levels stay host-
    # preferred; the working set is resident at ~1.2× vs fully-resident. No-op
    # unless memory_mode() === :managed.
    prefetch_level!(lev)
    dt_l = Float64(λ) * level_dx(hier, l)
    # KDK K1.  Native self-gravity (selfgrav = (coef=…, nsweep=…, rho_ext=…)):
    # levels ≥ 1 SOLVE their own Poisson (warm-started red-black on the block
    # union, Dirichlet from the parent φ pool) and kick from it; level 0 kicks
    # from the external topgrid φ (whose pool copy seeds the level-1 BCs).
    # gravity solve CADENCE (gravk ≥ 2): at levels ≥ gravk_lmin, run the level
    # Poisson solve only every gravk-th substep and kick from the existing pool
    # φ in between.  With λ_J resolved by N_J cells, φ drifts by ~1/N_J per
    # fine substep — but each solve must TOUCH its dense pyramid (gather +
    # residual) even when the ε-exit skips the cycles, and once the deep
    # levels' pyramids exceed the device the per-substep touch pages the
    # whole thing over PCIe.  Skipping the solve entirely is the only way to
    # skip the traffic.  Staleness = gravk·dt_l, smallest at the finest levels
    # (dt_l ∝ 2^-l) — restrict to l ≥ gravk_lmin so coarse φ stays fresh.
    dosolve = true
    if selfgrav !== nothing
        gravk = get(selfgrav, :gravk, 0)
        # cadence band [gravk_lmin, gravk_lmax]: the original out-of-core use
        # capped traffic at DEEP levels (lmin=6, lmax=∞); the shallow-static-
        # union case (DM-criterion runs: L1-L3 = 90+% of wall time in full-MG
        # every substep) wants the OPPOSITE band — cadence the big shallow
        # unions, keep every-substep converged solves at the collapse levels
        # (stale φ at the free-fall-capped finest level = central-mass pile-up).
        if gravk > 1 && get(selfgrav, :gravk_lmin, 6) <= l <= get(selfgrav, :gravk_lmax, 999)
            dosolve = hier.nstep[l + 1] % gravk == 0
        end
    end
    if selfgrav !== nothing && (l >= 1 || φ === nothing)
        # cosmological convention: the source is coef·(ρ_tot − rho_mean) = coef·δ
        # at EVERY level (matching the mean-subtracted topgrid FFT that feeds the
        # level-1 Dirichlet BCs); use_dm adds the per-level particle deposit.
        # The K1 KICK runs every substep (pool φ); only the SOLVE obeys the cadence.
        _t = _tic(lev.be)
        if !dosolve
            # cadence substep: TRACK with warm-started red-black sweeps on the
            # POOLS (traffic ∝ real cells; no dense pyramid touched).  The
            # collapse-driven φ changes are LOCAL — RB converges them in a few
            # sweeps from the last solve; the long wavelengths (which RB cannot
            # move) evolve on the slow K·dt_l timescale and are corrected by
            # the full MG solve every gravk-th substep.  NEVER skip outright:
            # at the free-fall-capped finest level dt ≈ 0.1·t_ff, so a skipped
            # solve is ~t_ff-stale φ → infall overshoot and central mass
            # pile-up (the j256g 3.5e10 spike).
            solve_gravity_level!(hier, l; source_coef = selfgrav.coef,
                                 nsweep = get(selfgrav, :gravk_nsweep, 8),
                                 rho_ext = get(selfgrav, :rho_ext, nothing),
                                 rho_mean = get(selfgrav, :rho_mean, 0),
                                 use_dm = get(selfgrav, :use_dm, false),
                                 residual = false)
        elseif l >= max(1, get(selfgrav, :mg2_lmin, 3)) && get(selfgrav, :mg2, false)
            # MG2's coarse correction carries a Dirichlet-0 halo — valid only for
            # unions COMPACT relative to the periodic domain.  The shallow static
            # levels (L1/L2 zoom sphere) span the box → the correction's long-
            # wavelength part diverges (measured rend up to 1e9×r0; all 23 guard
            # fires in the m2g bench were l ≤ 2).  They stay on the dense masked
            # path (cheap there); mg2_lmin gates the handoff.
            # block-topology MG (MG2): resident-pool smoothing + one SMALL dense
            # coarse grid (bbox÷cfac) — no per-substep giant-pyramid traffic
            solve_gravity_level_mg2!(hier, l; source_coef = selfgrav.coef,
                                     rho_ext = get(selfgrav, :rho_ext, nothing),
                                     rho_mean = get(selfgrav, :rho_mean, 0),
                                     use_dm = get(selfgrav, :use_dm, false),
                                     ncycle = get(selfgrav, :ncycle, 2),
                                     nsm = get(selfgrav, :nsm, 4),
                                     cfac = get(selfgrav, :cfac, 4),
                                     nub = get(selfgrav, :nub, 60),
                                     mgeps = get(selfgrav, :mgeps, 0))
        elseif l >= 1 && get(selfgrav, :mg, false)
            # masked multigrid (RAMSES multigrid_fine): CONVERGES the level φ —
            # gas (the level D pool) + optional DM deposit source — on any union
            # size; plain RB-GS below only tracks small unions.  The Dirichlet
            # ring uses the frozen parent φ (fine-first W-cycle convention; the
            # second substep's ring is lagged dt_parent/2 — a bc_phi extrapolation
            # from a parent φ_old is the known refinement if it ever matters).
            solve_gravity_level_mg!(hier, l; source_coef = selfgrav.coef,
                                    rho_ext = get(selfgrav, :rho_ext, nothing),
                                    rho_mean = get(selfgrav, :rho_mean, 0),
                                    use_dm = get(selfgrav, :use_dm, false),
                                    ncycle = get(selfgrav, :ncycle, 4),
                                    nu1 = get(selfgrav, :nu1, 3),
                                    nu2 = get(selfgrav, :nu2, 3),
                                    nub = get(selfgrav, :nub, 80),
                                    mgeps = get(selfgrav, :mgeps, 0))
        else
            solve_gravity_level!(hier, l; source_coef = selfgrav.coef,
                                 nsweep = l == 0 ? get(selfgrav, :nsweep0, 120) :
                                                   get(selfgrav, :nsweep, 30),
                                 rho_ext = get(selfgrav, :rho_ext, nothing),
                                 rho_mean = get(selfgrav, :rho_mean, 0),
                                 use_dm = get(selfgrav, :use_dm, false),
                                 residual = false)     # no diagnostic sync per substep
        end
        _toc(lev.be, :grav_solve, l, _t)
        _t = _tic(lev.be); grav_kick_level_pool!(hier, l, 0.5 * dt_l); _toc(lev.be, :grav_kick, l, _t)
    elseif φ !== nothing
        _t = _tic(lev.be); grav_kick_level!(hier, l, φ, 0.5 * dt_l); _toc(lev.be, :grav_kick, l, _t)
    end
    _nanstage!(hier, l, :grav_K1)
    if hier.scheme === :ctu
        # single-pass CTU: one ghost fill, one fused launch (R→O), captures from
        # the intact R, then swap.  Single-stage weights: fine ½/substep (time
        # average over the 2:1 pair), coarse 1 — same per-coarse-step totals as
        # the RK2 path's (¼,½)×2.
        _t = _tic(lev.be); fill_ghosts!(hier, l; θ = 0.0f0, buf = :R); _toc(lev.be, :ghost, l, _t)
        _t = _tic(lev.be); ctu_level!(hier, l, λ); _toc(lev.be, :hydro, l, _t)
        _t = _tic(lev.be)
        l >= 1   && capture_fine!(hier, l, :R, 0.5f0; λ)
        haskids  && capture_coarse!(hier, l + 1, :R, 1.0f0; λ)
        swap_buffers!(lev)
        _toc(lev.be, :capture, l, _t)
    else
        fill_ghosts!(hier, l; θ = 0.0f0, buf = :R)
        _spwatch(hier, l, :ghost1)
        stage_level!(hier, l, λ; w = 0.0f0, IN = :R, OUT = :O)
        l >= 1   && capture_fine!(hier, l, :R, 0.25f0)
        haskids  && capture_coarse!(hier, l + 1, :R, 0.5f0)
        fill_ghosts!(hier, l; θ = 0.0f0, buf = :O)
        _spwatch(hier, l, :ghost2)
        stage_level!(hier, l, λ; w = 0.5f0, IN = :O, OUT = :R)
        _spwatch(hier, l, :stage2)
        l >= 1   && capture_fine!(hier, l, :O, 0.25f0)
        haskids  && capture_coarse!(hier, l + 1, :O, 0.5f0)
    end
    _nanstage!(hier, l, :hydro)
    if selfgrav !== nothing && (l >= 1 || φ === nothing)           # KDK: K2
        _t = _tic(lev.be); grav_kick_level_pool!(hier, l, 0.5 * dt_l); _toc(lev.be, :grav_kick, l, _t)
    elseif φ !== nothing
        _t = _tic(lev.be); grav_kick_level!(hier, l, φ, 0.5 * dt_l); _toc(lev.be, :grav_kick, l, _t)
    end
    _nanstage!(hier, l, :grav_K2)
    if chem !== nothing                                           # analytic H+H₂
        # chem CADENCE (chemk ≥ 2): run every chemk-th substep with the
        # ACCUMULATED dt.  Chem is a per-cell ODE with internal subcycling —
        # non-stiff cells (dt ≪ t_cool by definition) tolerate the coarser
        # cadence, stiff cells do the same total iterations either way; what
        # the cadence removes is the fixed per-cell solver overhead (~19 ns ×
        # cells × 2^l calls — the deepest level's dominant cost at high N_J).
        ck = get(chem, :chemk, 1)
        if ck <= 1 || hier.nstep[l + 1] % ck == ck - 1
            cdt = dt_l * max(ck, 1)
            chemargs = Base.structdiff(chem, NamedTuple{(:chemk,)})
            _t = _tic(lev.be); chem_level!(hier, l, cdt; chemargs...); _toc(lev.be, :chem, l, _t)
            _nanstage!(hier, l, :chem)
        end
    end
    if haskids
        _t = _tic(lev.be)
        restrict_level!(hier, l + 1)
        _spwatch(hier, l, :restrict)
        reflux_apply!(hier, l + 1, λ)
        _spwatch(hier, l, :reflux)
        _toc(lev.be, :restrict_reflux, l, _t)
        _nanstage!(hier, l, :restrict_reflux)
        # out-of-core: the child level is now fully advanced AND consumed by this
        # level's restrict/reflux — MOVE it back to host RAM, freeing the device
        # for the working set.  (No-op unless memory_mode() === :managed.)
        evict_level!(hier.levels[l + 2])
    end
    hier.nstep[l + 1] += 1
    return nothing
end

"""
    advance_hierarchy!(hier) -> dt_root

One root step of the strict-2:1 W-cycle: λ from the global CFL, then the
recursive fine-first advance.  Level l takes exactly 2^l substeps of λ·dx_l.
"""
function advance_hierarchy!(hier::AMRHierarchy; φ = nothing, chem = nothing,
                            selfgrav = nothing)
    λ = compute_lambda!(hier)
    selfgrav !== nothing && φ !== nothing && phi_from_global!(hier, φ)
    advance_level_w!(hier, 0, λ; φ, chem, selfgrav)
    return λ * level_dx(hier, 0)
end

"CFL timestep from the finest occupied level: dt = cfl·dx_finest/max_signal."
function compute_dt(hier::AMRHierarchy)
    smax = 0.0f0; lfin = 0
    for l in 0:length(hier.levels)-1
        isempty(hier.levels[l+1].live) && continue
        smax = max(smax, max_signal(hier.levels[l+1], hier.gamma))
        lfin = l
    end
    return hier.cfl * level_dx(hier, lfin) / max(Float64(smax), 1e-30)
end

# ── diagnostics (host; exact f64 sums for the conservation gates) ─────────────
"""
    total_conserved(hier) -> (mass, px, py, pz, E)

Volume-weighted totals over the composite grid: fine cells count where they
exist; covered coarse cells are excluded (their mass lives on the fine level).
"""
function total_conserved(hier::AMRHierarchy)
    tot = zeros(Float64, 5)
    for l in 0:length(hier.levels)-1
        lev = hier.levels[l+1]
        isempty(lev.live) && continue
        dV = level_dx(hier, l)^3
        flev = l + 1 <= length(hier.levels) - 1 ? hier.levels[l+2] : nothing
        hD  = Array(lev.D); hS1 = Array(lev.S1); hS2 = Array(lev.S2)
        hS3 = Array(lev.S3); hT = Array(lev.Tau)
        hdsc = Array(lev.Dsc); hssc = Array(lev.Ssc); hesc = Array(lev.Esc)
        for s in lev.live
            m = lev.meta[s]; base = (Int(s) - 1) * lev.stride
            dsc = Float64(hdsc[s]); ssc = Float64(hssc[s]); esc = Float64(hesc[s])
            # per-cell covered mask from the fine level (host, test-grade)
            for k in 0:lev.B-1, j in 0:lev.B-1, i in 0:lev.B-1
                if flev !== nothing && !isempty(flev.live)
                    g = (Int128(m.origin[1]) + i, Int128(m.origin[2]) + j,
                         Int128(m.origin[3]) + k)
                    gf = ntuple(d -> Int128(2) * g[d], 3)
                    covered = !isempty(overlapping_blocks(flev, gf, (2, 2, 2)))
                    covered && continue
                end
                idx = base + ((k + lev.ng) * lev.nd + (j + lev.ng)) * lev.nd +
                      (i + lev.ng) + 1
                tot[1] += Float64(hD[idx])  * dsc * dV
                tot[2] += Float64(hS1[idx]) * ssc * dV
                tot[3] += Float64(hS2[idx]) * ssc * dV
                tot[4] += Float64(hS3[idx]) * ssc * dV
                tot[5] += Float64(hT[idx])  * esc * dV
            end
        end
    end
    return (tot[1], tot[2], tot[3], tot[4], tot[5])
end
