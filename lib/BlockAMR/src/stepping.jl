# stepping.jl — hierarchy time integration.
#
# Phase 1 (this file, no subcycling yet): every level advances the SAME dt per
# hierarchy step, SSP-RK2, with the coarse–fine machinery live: fine ghosts
# prolong from the parent stage state (θ=0 for stage 1, θ=1 for stage 2 —
# exactly the RK boundary state), C/F fluxes are captured per stage (weight ½
# each), then reflux corrects the uncovered coarse cells and restriction
# overwrites the covered ones.  λ_l = dt/dx_l per level here; the strict-2:1
# W-cycle (Phase 3) collapses them into the single global λ.

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
function advance_level_w!(hier::AMRHierarchy, l::Int, λ::Float32;
                          φ = nothing, chem = nothing, selfgrav = nothing)
    lev = hier.levels[l + 1]
    isempty(lev.live) && return nothing
    haskids = l + 2 <= length(hier.levels) && !isempty(hier.levels[l + 2].live)
    if haskids                                    # fine first (frozen parent state)
        advance_level_w!(hier, l + 1, λ; φ, chem, selfgrav)
        advance_level_w!(hier, l + 1, λ; φ, chem, selfgrav)
    end
    dt_l = Float64(λ) * level_dx(hier, l)
    # KDK K1.  Native self-gravity (selfgrav = (coef=…, nsweep=…, rho_ext=…)):
    # levels ≥ 1 SOLVE their own Poisson (warm-started red-black on the block
    # union, Dirichlet from the parent φ pool) and kick from it; level 0 kicks
    # from the external topgrid φ (whose pool copy seeds the level-1 BCs).
    if selfgrav !== nothing && l >= 1
        solve_gravity_level!(hier, l; source_coef = selfgrav.coef,
                             nsweep = get(selfgrav, :nsweep, 30),
                             rho_ext = get(selfgrav, :rho_ext, nothing))
        grav_kick_level_pool!(hier, l, 0.5 * dt_l)
    elseif φ !== nothing
        grav_kick_level!(hier, l, φ, 0.5 * dt_l)
    end
    fill_ghosts!(hier, l; θ = 0.0f0, buf = :R)
    stage_level!(hier, l, λ; w = 0.0f0, IN = :R, OUT = :O)
    l >= 1   && capture_fine!(hier, l, :R, 0.25f0)
    haskids  && capture_coarse!(hier, l + 1, :R, 0.5f0)
    fill_ghosts!(hier, l; θ = 0.0f0, buf = :O)
    stage_level!(hier, l, λ; w = 0.5f0, IN = :O, OUT = :R)
    l >= 1   && capture_fine!(hier, l, :O, 0.25f0)
    haskids  && capture_coarse!(hier, l + 1, :O, 0.5f0)
    if selfgrav !== nothing && l >= 1                              # KDK: K2
        grav_kick_level_pool!(hier, l, 0.5 * dt_l)
    elseif φ !== nothing
        grav_kick_level!(hier, l, φ, 0.5 * dt_l)
    end
    chem === nothing || chem_level!(hier, l, dt_l; chem...)        # analytic H+H₂
    if haskids
        restrict_level!(hier, l + 1)
        reflux_apply!(hier, l + 1, λ)
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
