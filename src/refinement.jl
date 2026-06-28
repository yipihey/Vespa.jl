# refinement.jl — physically-motivated AMR refinement indicators, ported from the
# original Enzo CellFlaggingMethods and validated against Enzo's own flagging
# (lib/EnzoLib enzomodules_flag_* primitives) as the oracle.
#
# Each indicator has the `RefinementPolicy` signature `(sim, cell) -> Float64`; a cell
# is flagged when the value exceeds the policy's `refine_above`. By convention here the
# indicators are NORMALISED so the refine threshold is 1.0 (value > 1 ⇒ refine), which
# lets several criteria be combined by `max` (a union, matching Enzo's
# `CellFlaggingMethod = 4 6`).

# ── Jeans-length refinement (Enzo CellFlaggingMethod = 6) ────────────────────────────
"""
    jeans_length_indicator(sim, cell; G, cells_per_jeans=4, cs_min=0.0) -> Float64

Resolve the Jeans length by at least `cells_per_jeans` cells (mirrors Enzo's
`CellFlaggingMethod = 6` with `RefineByJeansLengthSafetyFactor = cells_per_jeans`).
Returns `Δx · cells_per_jeans / λ_J`, where the Jeans length is
`λ_J = c_s · √(π / (G·ρ))`. With `RefinementPolicy(refine_above = 1.0)` a cell flags
when `Δx > λ_J / cells_per_jeans` — i.e. the Jeans length spans fewer than
`cells_per_jeans` cells. `cs_min` floors the sound speed (the analogue of Enzo's
`JeansRefinementColdTemperature`) so cold dense gas does not over-refine.

`G` is the gravitational constant in the simulation's unit system (default: the value
on `sim.grav` when self-gravity is enabled, else 1.0). Pure.
"""
@inline function jeans_length_indicator(sim::Simulation, cell; G::Real = _grav_G(sim),
                                        cells_per_jeans::Real = 4, cs_min::Real = 0.0)
    W  = _W(sim, cell)
    ρ  = W[1]
    cs = max(sound_speed(sim.model, W), cs_min)
    λJ = cs * sqrt(π / (G * ρ))
    dx = maximum(cell_width(sim.backend, cell))
    return dx * cells_per_jeans / λJ
end

# Gravitational constant the Jeans length uses: the self-gravity solver's G if enabled,
# else 1.0 (the indicator's G kwarg should be passed explicitly for non-gravity runs).
@inline _grav_G(sim::Simulation) = sim.grav === nothing ? 1.0 : sim.grav.G

"""
    jeans_refinement_policy(sim; cells_per_jeans=4, max_level, every=8, cs_min=0.0,
                            G=_grav_G(sim)) -> RefinementPolicy

Convenience builder: a `RefinementPolicy` whose indicator is
[`jeans_length_indicator`](@ref) with `refine_above = 1.0`.
"""
jeans_refinement_policy(sim::Simulation; cells_per_jeans::Real = 4,
                        max_level::Integer, every::Integer = 8, cs_min::Real = 0.0,
                        G::Real = _grav_G(sim)) =
    RefinementPolicy(refine_above = 1.0, max_level = max_level, every = every,
                     indicator = (s, c) -> jeans_length_indicator(s, c; G = G,
                                  cells_per_jeans = cells_per_jeans, cs_min = cs_min))

# ── DM-particle-count refinement (Enzo CellFlaggingMethod = 4, particle mass) ─────────
"""
    deposit_particle_counts!(sim, px, py, pz) -> Dict{NTuple{D,Int},Int}

Nearest-grid-point deposit of dark-matter particles onto the BASE-grid resolution: for
each particle, increment the integer base-cell index `floor((x−lo)/Δx_base)` it falls in.
Positions `px,py,pz` are in the simulation's physical domain (`domain(backend)`). Returns
a sparse count map keyed by the per-axis base-cell index tuple.

This mirrors Enzo's `DepositParticleMassFlaggingField` for equal-mass particles: a cell's
mass is `count·m_particle`, so the "refine at ≥N particles" test is `count ≥ N`. NGP at
the base resolution is exact for the level-0 regrid (the dominant cosmology case: refine
the uniform grid where DM clusters); sub-cell deposition at deeper levels needs adaptive
point-location and is a follow-up.
"""
function deposit_particle_counts!(sim::Simulation, px, py, pz)
    b = sim.backend
    D = rank(b)
    dom = domain(b)                                   # ((lo,hi),…) per axis
    nb  = _base_dims(b)                               # base-grid cells per axis
    lo  = ntuple(d -> dom[d][1], D)
    dx  = ntuple(d -> (dom[d][2] - dom[d][1]) / nb[d], D)
    pos = (px, py, pz)
    counts = Dict{NTuple{D,Int},Int}()
    np = length(px)
    @inbounds for p in 1:np
        idx = ntuple(d -> clamp(Int(floor((pos[d][p] - lo[d]) / dx[d])), 0, nb[d] - 1), D)
        counts[idx] = get(counts, idx, 0) + 1
    end
    return counts
end

# Base-grid cells per axis = current cells at relative level 0 (uniform base). Derived
# from the domain and a level-0 cell width (all base cells share it).
@inline function _base_dims(b)
    D = rank(b); dom = domain(b)
    # find any level-0 leaf to read the base cell width; fall back to the first cell.
    w = nothing
    for_each_cell(b) do c
        if w === nothing && level_of(b, c) == 0
            w = cell_width(b, c)
        end
    end
    w === nothing && error("deposit_particle_counts!: no level-0 cell to size the base grid")
    return ntuple(d -> round(Int, (dom[d][2] - dom[d][1]) / w[d]), D)
end

"""
    particle_count_indicator(sim, cell; counts) -> Float64

Per-cell DM particle count (Enzo `CellFlaggingMethod = 4`). `counts` is the map from
[`deposit_particle_counts!`](@ref); returns the count of the base cell containing this
cell's center. With `RefinementPolicy(refine_above = N−1)` a cell flags when it holds
≥ N particles (e.g. `refine_above = 3.0` ⇒ refine at ≥4).
"""
@inline function particle_count_indicator(sim::Simulation, cell; counts)
    b = sim.backend; D = rank(b); dom = domain(b)
    nb = _base_dims(b)
    ctr = cell_center(b, cell)
    idx = ntuple(d -> clamp(Int(floor((ctr[d] - dom[d][1]) /
                ((dom[d][2] - dom[d][1]) / nb[d]))), 0, nb[d] - 1), D)
    return Float64(get(counts, idx, 0))
end

"""
    particle_refinement_policy(sim, px, py, pz; nmin=4, max_level, every=8) -> RefinementPolicy

Convenience builder: deposits the particle counts once (NGP at base resolution) and
returns a `RefinementPolicy` that refines cells holding ≥ `nmin` DM particles. Re-deposit
(rebuild the policy) when particles have moved significantly.
"""
function particle_refinement_policy(sim::Simulation, px, py, pz; nmin::Integer = 4,
                                    max_level::Integer, every::Integer = 8)
    counts = deposit_particle_counts!(sim, px, py, pz)
    RefinementPolicy(refine_above = nmin - 1.0, max_level = max_level, every = every,
                     indicator = (s, c) -> particle_count_indicator(s, c; counts = counts))
end
