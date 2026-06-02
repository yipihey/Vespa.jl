# The finite-volume driver (ADR P1): ordinary, inspectable Julia, written
# entirely against the `MeshInterface` seam. It is **ghost-free** and
# **neighbor-driven** — exactly the model a hierarchical, hanging-node mesh
# exposes — so the same code runs unchanged on the uniform `RefMesh` and on
# HierarchicalGrids.jl.
#
# Scheme (second order in space and time):
#   * per-cell PLM reconstruction with minmod-limited primitive slopes,
#   * HLLC flux at each face between the two reconstructed face states,
#   * conservative flux-divergence accumulation: each interior face adds +F·area
#     to one cell's net-flux-out and −F·area to the other, so Σ over cells is
#     zero to round-off and the update U −= dt·(Σflux)/V is conservative,
#   * SSP-RK2 (Heun) time integration.
#
# Boundaries are resolved per face by `neighbor`: an `Interior` (incl. periodic
# wrap) participates as an ordinary face; a `DomainBoundary` synthesizes a ghost
# face state from the boundary condition (outflow = copy, reflecting = flip the
# normal velocity). No ghost cells exist in storage.

"""
    Simulation(backend, problem; layout=SoA())

Bind a `Problem` to a mesh `backend`, allocate the conserved fields (plus RK
scratch) with the requested `layout`, and set initial conditions. `backend` may
be a bare backend or an `Instrumented` wrapper (P10) — identical code path.
"""
mutable struct Simulation{B,P,S,V}
    backend::B
    problem::P
    bcs::BoundaryConditions
    γ::Float64
    state::S            # conserved U (canonical, layout-tested)
    u0::S               # RK scratch: state at start of step
    acc::S              # net-flux-out accumulator (pre-÷V)
    sv::V               # cached NTuple{5} views into `state`
    u0v::V
    accv::V
    layout::AbstractLayout
    t::Float64
    step::Int
    grav::Any           # nothing, or a GravityField (self-gravity; default off)
end

function Simulation(backend, prob::Problem; layout::AbstractLayout = SoA())
    N = rank(backend)
    spec = FieldSpec(collect(FIELD_NAMES))
    state = allocate_fields(backend, spec; layout = layout)
    u0 = allocate_fields(backend, spec; layout = layout)
    acc = allocate_fields(backend, spec; layout = layout)
    views(store) = ntuple(i -> field_view(backend, store, FIELD_NAMES[i]), NVAR)
    bcs = _as_bcs(prob.bcs, N)
    sim = Simulation(backend, prob, bcs, prob.γ, state, u0, acc,
                     views(state), views(u0), views(acc), layout, 0.0, 0, nothing)
    set_initial_conditions!(sim)
    return sim
end

_as_bcs(bc::AbstractBC, N::Int) = BoundaryConditions(bc, Val(N))
_as_bcs(bcs::BoundaryConditions, ::Int) = bcs
_as_bcs(pairs::Tuple, ::Int) = BoundaryConditions(pairs)

# -- per-cell conserved state access through cached views --
@inline get_U(v, cell) = ntuple(i -> @inbounds(v[i][cell]), NVAR)
@inline function set_U!(v, cell, U)
    @inbounds for i in 1:NVAR
        v[i][cell] = U[i]
    end
    return nothing
end

function set_initial_conditions!(sim::Simulation)
    b, γ, init = sim.backend, sim.γ, sim.problem.init
    N = rank(b)
    for_each_cell(b) do cell
        c = cell_center(b, cell)
        coords = ntuple(d -> d <= N ? c[d] : 0.0, 3)
        W = NTuple{5,Float64}(init(coords...))
        set_U!(sim.sv, cell, prim2cons(W, γ))
    end
    return sim
end

# -- boundary ghost state from a face state (no ghost cells; synthesized) --
@inline function ghost_state(W::NTuple{5,Float64}, ::Outflow, ::Int)
    return W                                   # zero-gradient
end
@inline function ghost_state(W::NTuple{5,Float64}, ::Reflecting, axis::Int)
    # Mirror: flip the velocity component normal to this boundary.
    return ntuple(i -> i == axis + 1 ? -W[i] : W[i], 5)
end
@inline ghost_state(W::NTuple{5,Float64}, ::Periodic, ::Int) = W   # never reached

# Primitive state at a cell.
@inline _W(sim::Simulation, cell) = cons2prim(get_U(sim.sv, cell), sim.γ)

# Neighbor primitive state across (axis, side): interior cell, or BC ghost of the
# current cell's primitive state.
@inline function _neighbor_W(sim::Simulation, cell, Wc::NTuple{5,Float64},
                             axis::Int, side::Symbol)
    nb = neighbor(sim.backend, cell, axis, side; bcs = sim.bcs)
    if nb isa Interior
        return _W(sim, nb.cell)
    else
        return ghost_state(Wc, nb.bc, axis)
    end
end

# minmod-limited PLM slope of every primitive component along `axis` at `cell`.
@inline function _plm_slope(sim::Simulation, cell, Wc::NTuple{5,Float64}, axis::Int)
    WL = _neighbor_W(sim, cell, Wc, axis, :lo)
    WR = _neighbor_W(sim, cell, Wc, axis, :hi)
    return ntuple(i -> limited_slope(WL[i], Wc[i], WR[i]), NVAR)
end

"""
    compute_dt(sim; level=nothing)

Maximum stable timestep from the CFL condition. With `level` set, the min is
taken over only that refinement level's leaves (each level picks its own stable
step — the basis for subcycling); `nothing` takes it over all leaves (the
single-rate global step). Returns `Inf` if the level has no leaves.
"""
function compute_dt(sim::Simulation; level = nothing)
    b, γ, cfl = sim.backend, sim.γ, sim.problem.cfl
    N = rank(b)
    grav = sim.grav
    invdt = 0.0
    grav_invdt = 0.0
    for_each_cell(b; level = level) do cell
        W = _W(sim, cell)
        c = sound_speed(W, γ)
        w = cell_width(b, cell)
        for d in 1:N
            vd = (W[2], W[3], W[4])[d]
            invdt = max(invdt, (abs(vd) + c) / w[d])
        end
        grav === nothing || (grav_invdt = max(grav_invdt, _gravity_invdt(grav, W[1])))
    end
    invdt = max(invdt, grav_invdt)               # free-fall limiter (gravity on)
    return invdt == 0.0 ? Inf : cfl / invdt
end

# Accumulate net-flux-out (pre-÷V) for the current `sim.state` into `sim.acc`.
# Face enumeration is delegated to the backend (`for_each_face`), which emits each
# unique (sub)face once with the cell(s) on each side and the physical face area.
# The per-face kernel dispatches on the `NeighborRef` types so interior,
# lo-boundary, and hi-boundary faces are distinct methods with no branching. This
# is conservative by construction — an interior face adds +F·area to the left
# cell's net-flux-out and −F·area to the right's — and stays conservative across
# coarse↔fine sub-faces once the backend emits them (Phase B).
function accumulate_flux!(sim::Simulation; reflux = nothing)
    b = sim.backend
    av = sim.accv
    for_each_cell(b) do cell
        set_U!(av, cell, ntuple(_ -> 0.0, NVAR))
    end
    for_each_face(b; bcs = sim.bcs) do leftref, rightref, axis, area
        _flux_face!(sim, leftref, rightref, axis, area; reflux = reflux)
    end
    return nothing
end

# Reconstructed primitive face state from cell `i`, extrapolated by half a
# minmod-limited PLM slope toward `side` (:lo subtracts, :hi adds).
@inline function _face_value(sim::Simulation, i, axis::Int, side::Symbol)
    Wi = _W(sim, i)
    s = _plm_slope(sim, i, Wi, axis)
    h = side === :hi ? 0.5 : -0.5
    return ntuple(k -> Wi[k] + h * s[k], NVAR)
end

# interior↔interior face: +axis normal points i→j.
@inline function _flux_face!(sim::Simulation, left::Interior, right::Interior,
                             axis::Int, area::Float64; reflux = nothing)
    i, j = left.cell, right.cell
    WL = _face_value(sim, i, axis, :hi)
    WR = _face_value(sim, j, axis, :lo)
    F = hllc_flux(WL, WR, sim.γ, axis)
    av = sim.accv
    @inbounds for k in 1:NVAR
        fk = F[k] * area
        av[k][i] += fk        # flux leaves i
        av[k][j] -= fk        # flux enters j
    end
    if reflux !== nothing
        for reg in reflux
            _reflux_capture!(sim, reg, i, j, F, area)
        end
    end
    return nothing
end

# hi-side domain boundary: interior cell i on the left, ghost on the right.
@inline function _flux_face!(sim::Simulation, left::Interior, right::DomainBoundary,
                             axis::Int, area::Float64; reflux = nothing)
    i = left.cell
    WL = _face_value(sim, i, axis, :hi)
    Wg = ghost_state(WL, right.bc, axis)
    F = hllc_flux(WL, Wg, sim.γ, axis)
    av = sim.accv
    @inbounds for k in 1:NVAR
        av[k][i] += F[k] * area      # outward normal +axis
    end
    return nothing
end

# lo-side domain boundary: ghost on the left, interior cell j on the right.
@inline function _flux_face!(sim::Simulation, left::DomainBoundary, right::Interior,
                             axis::Int, area::Float64; reflux = nothing)
    j = right.cell
    WR = _face_value(sim, j, axis, :lo)
    Wg = ghost_state(WR, left.bc, axis)
    F = hllc_flux(Wg, WR, sim.γ, axis)
    av = sim.accv
    @inbounds for k in 1:NVAR
        av[k][j] -= F[k] * area      # outward normal −axis
    end
    return nothing
end

# state ← src (per cell). `level` restricts the write to one refinement level
# (the subcycling primitive); `nothing` writes every leaf (the single-rate path).
function _copy_state!(sim::Simulation, dst, src; level = nothing)
    for_each_cell(sim.backend; level = level) do cell
        set_U!(dst, cell, get_U(src, cell))
    end
    return nothing
end

# Forward-Euler-style apply: dst = base − dt·acc/V  (acc = net flux out).
function _euler_apply!(sim::Simulation, dst, base, dt; level = nothing)
    b = sim.backend
    for_each_cell(b; level = level) do cell
        invV = dt / cell_volume(b, cell)
        Ub = get_U(base, cell)
        A = get_U(sim.accv, cell)
        set_U!(dst, cell, ntuple(i -> Ub[i] - invV * A[i], NVAR))
    end
    return nothing
end

# SSP-RK2 combine: state = 0.5·u0 + 0.5·(state − dt·acc/V).
function _rk2_combine!(sim::Simulation, dt; level = nothing)
    b = sim.backend
    for_each_cell(b; level = level) do cell
        invV = dt / cell_volume(b, cell)
        U0 = get_U(sim.u0v, cell)
        U1 = get_U(sim.sv, cell)
        A = get_U(sim.accv, cell)
        set_U!(sim.sv, cell, ntuple(i -> 0.5 * U0[i] + 0.5 * (U1[i] - invV * A[i]), NVAR))
    end
    return nothing
end

"""
    step!(sim, dt; level=nothing)

Advance one full timestep `dt` with SSP-RK2. With `level` set, only the leaves at
that refinement level are updated (the subcycling primitive); flux accumulation
still reads neighbors at every level, so a finer level advances against its
(frozen, within this substep) coarse-neighbor data. `level=nothing` updates all
leaves — the single-rate path, unchanged. `step!` does not advance `sim.t`/`step`
when `level` is given (the subcycle driver owns time bookkeeping).
"""
function step!(sim::Simulation, dt::Float64; level = nothing)
    _copy_state!(sim, sim.u0v, sim.sv; level = level)        # u0 ← Uⁿ
    accumulate_flux!(sim)                                     # acc ← L(Uⁿ)
    sim.grav === nothing || apply_gravity_source!(sim, sim.grav; level = level)
    _euler_apply!(sim, sim.sv, sim.u0v, dt; level = level)   # U1 = Uⁿ − dt·(L−S)(Uⁿ)
    accumulate_flux!(sim)                                     # acc ← L(U1)
    sim.grav === nothing || apply_gravity_source!(sim, sim.grav; level = level)
    _rk2_combine!(sim, dt; level = level)                     # Uⁿ⁺¹ = ½Uⁿ + ½(U1 − dt·(L−S)(U1))
    if level === nothing
        sim.t += dt
        sim.step += 1
    end
    return sim
end

# As `step!(sim, dt; level)`, but also capture coarse↔fine face fluxes into the
# given flux registers (`regs`, a tuple). For each register, `signs[r]` scales the
# capture: +1 when this level is the FINE side of that register's interface (add
# the fine flux·dt), −1 when it is the COARSE side (subtract the coarse flux·dt).
# SSP-RK2's effective flux is ½(stage-1)+½(stage-2), so each stage contributes
# `sign · 0.5 · dt` to the register (set on `reg.scale` before each accumulate).
function step_level!(sim::Simulation, dt::Float64, level::Int, regs, signs)
    _copy_state!(sim, sim.u0v, sim.sv; level = level)
    _set_scales!(regs, signs, 0.5 * dt)
    accumulate_flux!(sim; reflux = regs)                     # stage 1, ½dt weight
    sim.grav === nothing || apply_gravity_source!(sim, sim.grav; level = level)
    _euler_apply!(sim, sim.sv, sim.u0v, dt; level = level)
    _set_scales!(regs, signs, 0.5 * dt)
    accumulate_flux!(sim; reflux = regs)                     # stage 2, ½dt weight
    sim.grav === nothing || apply_gravity_source!(sim, sim.grav; level = level)
    _rk2_combine!(sim, dt; level = level)
    return sim
end

@inline function _set_scales!(regs, signs, base::Float64)
    @inbounds for r in eachindex(regs)
        regs[r].scale = signs[r] * base
    end
    return nothing
end

# ───────────────────────────── AMR (science layer, P1) ──────────────────────
# Refinement *policy* is science: it changes between problems and lives in Julia.
# The mechanics (conservative prolong/restrict on refine, hanging-node face
# enumeration) are the substrate's job, behind the seam. `regrid!` tags leaves by
# a refinement indicator and asks the backend to refine them; the backend
# conservatively remaps every field store registered on the mesh.

"""
    density_gradient_indicator(sim, cell) -> Float64

Max relative density jump `|Δρ|/max(ρ)` to a face neighbor — a cheap shock/
contact detector. Used as the default refinement indicator.
"""
function density_gradient_indicator(sim::Simulation, cell)
    b = sim.backend
    ρc = _W(sim, cell)[1]
    g = 0.0
    for d in 1:rank(b)
        for side in (:lo, :hi)
            nb = neighbor(b, cell, d, side; bcs = sim.bcs)
            nb isa Interior || continue
            ρn = _W(sim, nb.cell)[1]
            g = max(g, abs(ρn - ρc) / max(ρc, ρn, eps()))
        end
    end
    return g
end

"""
    RefinementPolicy(; refine_above, max_level, every=8,
                       indicator=density_gradient_indicator)

A problem's adaptivity, as a plain value (ADR P9). `regrid!` refines every leaf
below `max_level` whose `indicator(sim, cell)` exceeds `refine_above`, scheduled
every `every` steps. Refine-only this milestone (conservative growth of the
refined region); coarsening is a follow-up.
"""
struct RefinementPolicy{F}
    indicator::F
    refine_above::Float64
    max_level::Int
    every::Int
end
RefinementPolicy(; refine_above::Real, max_level::Integer, every::Integer = 8,
                 indicator = density_gradient_indicator) =
    RefinementPolicy(indicator, Float64(refine_above), Int(max_level), Int(every))

"""
    regrid!(sim, policy) -> Int

Tag and refine leaves per `policy`; returns the number refined. The backend
remaps all tracked fields conservatively, so total mass/energy are unchanged by
the regrid itself.
"""
function regrid!(sim::Simulation, policy::RefinementPolicy)
    b = sim.backend
    to_refine = Any[]
    for_each_cell(b) do cell
        if level_of(b, cell) < policy.max_level &&
           policy.indicator(sim, cell) > policy.refine_above
            push!(to_refine, cell)
        end
    end
    isempty(to_refine) || refine!(b, to_refine)
    return length(to_refine)
end

"""
    evolve!(sim; verbose=false, policy=nothing, callback=nothing, callback_every=1)

Integrate to `problem.tfinal`, choosing each `dt` from the CFL condition and
clipping the final step to land exactly on `tfinal`. If a `RefinementPolicy` is
given, regrid every `policy.every` steps (and once before the first step) so the
mesh tracks moving features.

`callback`, if given, is a plain function `callback(sim, stage)` invoked over the
live state with `stage ∈ (:init, :step, :final)`: once before the first step
(`:init`, at `t=0`), after every `callback_every` steps (`:step`), and once at
the end (`:final`). This is the viz/analysis/checkpoint hook; the core stays
agnostic to what the callback does. Default `nothing` leaves behavior unchanged.

`subcycle=true` switches to AMR level time-subcycling (each level advances at its
own CFL dt; finer levels substep to catch up — see [`evolve_level!`](@ref)).
Default `false` keeps the single-rate global-dt path, which is exactly
conservative at coarse↔fine interfaces (subcycled refluxing is a follow-up).
"""
function evolve!(sim::Simulation; verbose::Bool = false,
                 policy::Union{Nothing,RefinementPolicy} = nothing,
                 callback = nothing, callback_every::Int = 1,
                 subcycle::Bool = false)
    if subcycle
        return _evolve_subcycle!(sim; verbose = verbose, policy = policy,
                                 callback = callback, callback_every = callback_every)
    end
    tfinal = sim.problem.tfinal
    callback !== nothing && callback(sim, :init)
    while sim.t < tfinal * (1 - 1e-12)
        if policy !== nothing && policy.every > 0 && sim.step % policy.every == 0
            n = regrid!(sim, policy)
            verbose && n > 0 && @printf("  regrid: +%d cells → %d leaves\n", n, n_cells(sim.backend))
        end
        sim.grav === nothing || solve_poisson!(sim, sim.grav)   # φ(ρⁿ): g held over the step
        dt = min(compute_dt(sim), tfinal - sim.t)
        step!(sim, dt)
        if callback !== nothing && callback_every > 0 && sim.step % callback_every == 0
            callback(sim, :step)
        end
        verbose && @printf("step %5d   t = %.6f   dt = %.3e   (%d cells)\n",
                           sim.step, sim.t, dt, n_cells(sim.backend))
    end
    return sim
end

# ─────────────────────── AMR time subcycling (P1 science) ───────────────────
# Classic-Enzo level subcycling on EnzoNG's composite (leaf-only) mesh: each
# refinement level advances at its own CFL-stable dt, and a finer level takes the
# integer number of substeps needed to catch up to the coarse step
# (`EvolveLevel.C` / `SetLevelTimeStep.C`). This is a PERFORMANCE change (coarse
# levels take large steps instead of being throttled to the finest cell's dt),
# and with the flux register (see reflux.jl) it is also exactly conservative at
# coarse↔fine interfaces. It is opt-in (`evolve!(...; subcycle=true)`); the
# default `evolve!` keeps the single-rate path.

# Active flux registers (and their signs) while stepping `level`: the parent's
# register sees this level as the FINE side (+1, add fine flux·dt); this level's
# own register sees it as the COARSE side (−1, subtract coarse flux·dt).
@inline function _active_regs(parent_reg, own_reg)
    if parent_reg === nothing && own_reg === nothing
        return ((), ())
    elseif parent_reg === nothing
        return ((own_reg,), (-1.0,))
    elseif own_reg === nothing
        return ((parent_reg,), (1.0,))
    else
        return ((parent_reg, own_reg), (1.0, -1.0))
    end
end

"""
    evolve_level!(sim, level, dt_target; verbose=false, parent_reg=nothing) -> Float64

Advance every leaf at refinement `level` by total time `dt_target`, recursing
into `level+1` between substeps (the Berger–Colella subcycling recursion). The
level takes `n = ceil(dt_target / dt_stable(level))` equal substeps of
`dt_target/n`, so it lands exactly on the coarse step boundary; `n` is the
refinement-in-time ratio (≈ the spatial refinement factor at matched wave
speeds). Returns the substep size used. A level with no leaves is a no-op.

Coarse↔fine conservation: this level creates the flux register for its interface
with `level+1` (where it is the coarse side), threads it down so the finer level's
substeps add their time-integrated fine flux, and applies the net correction to
its coarse leaves after each substep's fine subcycles complete. `parent_reg` (the
register for the `level-1 ↔ level` interface) captures this level's fine-side
flux. Pass `nothing` (default) at the root or to advance a level standalone
without refluxing.
"""
function evolve_level!(sim::Simulation, level::Int, dt_target::Float64;
                       verbose::Bool = false, parent_reg = nothing)
    b = sim.backend
    # Does this level have any leaves? (compute_dt returns Inf if empty.)
    dt_stable = compute_dt(sim; level = level)
    isfinite(dt_stable) || return dt_target          # empty level: nothing to do
    n = max(1, ceil(Int, dt_target / dt_stable * (1 - 1e-12)))
    dt_sub = dt_target / n
    has_finer = level < max_level(b)
    own_reg = has_finer ? _flux_register(level) : nothing
    regs, signs = _active_regs(parent_reg, own_reg)
    for _ in 1:n
        own_reg === nothing || empty!(own_reg.delta)   # fresh per coarse substep
        if isempty(regs)
            step!(sim, dt_sub; level = level)          # no interfaces: plain step
        else
            step_level!(sim, dt_sub, level, regs, signs)
        end
        if has_finer
            evolve_level!(sim, level + 1, dt_sub; verbose = verbose, parent_reg = own_reg)
            _reflux_apply!(sim, own_reg)               # correct this level's coarse leaves
        end
    end
    verbose && @printf("    level %d: %d substep(s) × %.3e\n", level, n, dt_sub)
    return dt_sub
end

# Opt-in subcycled integration (driven from evolve!(...; subcycle=true)).
function _evolve_subcycle!(sim::Simulation; verbose, policy, callback, callback_every)
    tfinal = sim.problem.tfinal
    callback !== nothing && callback(sim, :init)
    while sim.t < tfinal * (1 - 1e-12)
        if policy !== nothing && policy.every > 0 && sim.step % policy.every == 0
            n = regrid!(sim, policy)
            verbose && n > 0 && @printf("  regrid: +%d cells → %d leaves\n", n, n_cells(sim.backend))
        end
        # Self-gravity: one composite Poisson solve per root step; g = −∇φ is held
        # across all fine subcycles (1st-order-in-time coupling). To tighten it,
        # move this solve into evolve_level!'s substep loop at level 0.
        sim.grav === nothing || solve_poisson!(sim, sim.grav)
        # Root step size is the level-0 CFL dt, clipped to land on tfinal.
        dt_root = min(compute_dt(sim; level = 0), tfinal - sim.t)
        evolve_level!(sim, 0, dt_root; verbose = verbose)
        sim.t += dt_root
        sim.step += 1
        if callback !== nothing && callback_every > 0 && sim.step % callback_every == 0
            callback(sim, :step)
        end
        verbose && @printf("root step %5d   t = %.6f   dt = %.3e   (%d cells)\n",
                           sim.step, sim.t, dt_root, n_cells(sim.backend))
    end
    return sim
end
