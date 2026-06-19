# Coarse–fine flux refluxing for AMR time subcycling (ADR P5: the solver builds
# refluxing on the substrate's conservative transfer). On Vespa's composite
# (leaf-only) mesh there is no coarse-under-fine cell to "project down"; instead
# we correct the *coarse leaf adjacent to a refinement boundary* so that the flux
# it saw over its big step equals the time-integrated flux the fine leaves on the
# other side actually carried over their substeps. This is the modern
# flux-register form of Enzo's `UpdateFromFinerGrids.C` /
# `Grid_CorrectForRefinedFluxes.C`, and it restores exact conservation across the
# interface that subcycling otherwise breaks (the two sides advance with
# different dt).
#
# Mechanics. A `FluxRegister` accumulates, per (coarse leaf, conserved component),
# the signed flux mismatch in *conserved units already divided by nothing* — i.e.
# the same `flux·area·dt` quantity the state update consumes as `acc·dt/V` (so the
# correction is `register / V_coarse`, applied once after the fine level finishes
# its subcycles). Sign convention matches `_flux_face!`: across a face with +axis
# normal pointing left→right, the flux `F·area` *leaves* the left cell and
# *enters* the right.
#
#   • Coarse step over dt_c: for each coarse↔fine face touching coarse cell `c`,
#     subtract the coarse flux contribution to `c` (we will replace it):
#       register[c] -= sign_c · F_coarse · area_fine · dt_c
#   • Each fine substep over dt_f: add the fine flux contribution that crossed
#     the same physical interface:
#       register[c] += sign_c · F_fine · area_fine · dt_f
#     (sign_c is +1 if `c` is the left/low side of the face, −1 if the right/high
#     side — i.e. whether the +axis normal points out of or into `c`.)
#   • After the fine level's subcycles complete, apply:
#       U[c] += register[c] / V_coarse ;  register[c] = 0
#
# The fine `area_fine` is exactly what `for_each_face` already passes (it emits
# coarse↔fine sub-faces carrying the fine area), and summing the fine sub-faces
# over a coarse face reconstructs the coarse face area, so the registers balance
# to round-off. Because flux capture happens inside the SSP-RK2 stages, we record
# the RK2-averaged face flux (½ stage-1 + ½ stage-2), matching the actual update.

"""
    FluxRegister

Per-coarse-leaf accumulator of the signed coarse↔fine flux mismatch (in
flux·area·time units). Keyed by coarse cell handle; value is an `NTuple{NVAR}`.
Created per coarse–fine level pair for the duration of one coarse step.

`coarse_level` selects which interfaces to capture (the `(coarse_level,
coarse_level+1)` jump). `scale` is set by the driver before each capture pass: it
folds the signed timestep and the RK-stage weight together — `−½·dt_c` on each of
the coarse step's two RK stages, `+½·dt_f` on each fine substep's two stages — so
the register ends a coarse step holding `Σ(fine flux·dt_f) − (coarse flux·dt_c)`
in net-flux-out units, ready to apply as `U_coarse −= register / V_coarse`.
"""
mutable struct FluxRegister{NV,T}
    delta::Dict{Any,NTuple{NV,T}}           # coarse leaf handle → accumulated mismatch
    coarse_level::Int                       # the coarser side's level
    scale::T                                # signed dt × RK-stage weight, per pass
end

# Register sized by the equation set (nvars), typed by the field precision Tf.
function _flux_register(sim::Simulation, coarse_level::Int)
    nv = nvars(sim.model); T = _Tf(sim)
    return FluxRegister{nv,T}(Dict{Any,NTuple{nv,T}}(), coarse_level, zero(T))
end

@inline _reg_add!(reg::FluxRegister{NV,T}, c, v::NTuple{NV,T}) where {NV,T} =
    (reg.delta[c] = get(reg.delta, c, ntuple(_ -> zero(T), Val(NV))) .+ v; nothing)

# Capture one interior face's flux into the register IFF it is a coarse↔fine face
# straddling (coarse_level, coarse_level+1). `F` is the HLLC flux, `area` the fine
# sub-face area; the +axis normal points i→j (i=left, j=right). The coarse cell of
# the pair gets `sign·weight·F·area` with sign set by which side it is on.
@inline function _reflux_capture!(sim::Simulation, reg::FluxRegister{NV,T}, i, j,
                                  F, area::Real) where {NV,T}
    b = sim.backend
    li = level_of(b, i)
    lj = level_of(b, j)
    li == lj && return nothing                      # conforming face: not an interface
    cl = reg.coarse_level
    s = reg.scale; aT = T(area)
    if li == cl && lj == cl + 1
        # coarse is the LEFT cell i: +axis normal leaves i, so flux that leaves
        # the coarse cell carries sign +1 in its net-flux-out accumulator.
        _reg_add!(reg, i, map(f -> s * f * aT, F))
    elseif lj == cl && li == cl + 1
        # coarse is the RIGHT cell j: +axis normal enters j (−1 in net-flux-out).
        _reg_add!(reg, j, map(f -> -s * f * aT, F))
    end
    # any other level combination (deeper jump) is handled by that pair's own
    # register; level gaps across a face are ≤ 1 by the backend's balance rule.
    return nothing
end

# Apply the accumulated mismatch to the coarse leaves and reset. Verified to
# matter: disabling this on a 2-level subcycled Sod run takes mass drift from
# ~7e-13 (round-off) to ~2.5e-3 (0.45%), i.e. it is what restores conservation.
function _reflux_apply!(sim::Simulation, reg::FluxRegister{NV,T}) where {NV,T}
    b = sim.backend
    for (c, d) in reg.delta
        invV = one(T) / T(cell_volume(b, c))
        U = get_U(sim.sv, c)
        set_U!(sim.sv, c, map((u, dd) -> u - invV * dd, U, d))
    end
    empty!(reg.delta)
    return nothing
end

# ── boundary-flux recording (ADR-0003 part A) ─────────────────────────────────
# Records the time-integrated +axis flux `∫ F·area dt` through each DOMAIN-BOUNDARY
# face, keyed by (axis, side, boundary-cell). On the EnzoBackend a grid's outer
# faces ARE the coarse–fine interface, so this is the fine grid's RefinedFluxes (and
# a coarse grid's InitialFluxes at a subgrid) that Enzo's CorrectForRefinedFluxes
# needs. Captured during `accumulate_flux!` (both SSP-RK2 stages, `scale=½dt` each),
# exactly like the coarse↔fine `FluxRegister`, so the value is consistent with the
# gas update the slot performed.
mutable struct BoundaryFluxRegister{NV,T}
    flux::Dict{Tuple{Int,Symbol,Any},NTuple{NV,T}}    # outer (axis, side, cell) → ∫F·area dt
    interior::Dict{Tuple{Int,Any},NTuple{NV,T}}       # interior (axis, lo_cell) → ∫F·area dt
    scale::T                                          # dt × RK-stage weight, per pass
    record_interior::Bool                             # capture interior faces? (AMR coarse InitialFluxes)
end
function _bflux_register(sim::Simulation; record_interior::Bool = false)
    nv = nvars(sim.model); T = _Tf(sim)
    return BoundaryFluxRegister{nv,T}(Dict{Tuple{Int,Symbol,Any},NTuple{nv,T}}(),
                                      Dict{Tuple{Int,Any},NTuple{nv,T}}(), zero(T), record_interior)
end

@inline function _bflux_capture!(reg::BoundaryFluxRegister{NV,T}, axis::Int, side::Symbol,
                                 cell, F, area::Real) where {NV,T}
    k = (axis, side, cell)
    s = reg.scale; aT = T(area)
    add = map(f -> s * f * aT, F)
    reg.flux[k] = get(reg.flux, k, ntuple(_ -> zero(T), Val(NV))) .+ add
    return nothing
end

# Interior face (between two same-grid cells), keyed by the +axis LO cell. The
# coarse grid's flux at a subgrid-boundary face IS an interior face here; the
# AMR reflux bridge looks these up to fill Enzo's InitialFluxes. Only recorded
# when the register opts in (single-grid / part-A paths leave `interior` empty).
@inline function _bflux_capture_interior!(reg::BoundaryFluxRegister{NV,T}, axis::Int,
                                          lo_cell, F, area::Real) where {NV,T}
    reg.record_interior || return nothing
    k = (axis, lo_cell)
    s = reg.scale; aT = T(area)
    add = map(f -> s * f * aT, F)
    reg.interior[k] = get(reg.interior, k, ntuple(_ -> zero(T), Val(NV))) .+ add
    return nothing
end

# ── ND face-plane raster (ADR-0003 follow-up #2) ──────────────────────────────
# Enzo stores a coarse–fine face flux as a flat plane `Left/RightFluxes[field][dim]`
# of `prod(Dim)` cells, where `Dim[d] = EndGlobalIndex[dim][d] − StartGlobalIndex
# [dim][d] + 1` and the flux dim is collapsed (Dim[dim]=1). Its consumer
# (`Grid_CorrectForRefinedFluxes.C:460`) addresses cell at global index `g` by
#   FluxIndex = Σ_d (g[d] − StartGlobalIndex[dim][d]) · Π_{e<d} Dim[e]
# i.e. column-major over the orthogonal dims, dim-0 fastest. In 1D the plane is a
# single cell; in ND it is the (D−1)-plane of the face. `bflux_plane` rasterizes
# one (dim, side) plane: it walks the plane offsets, maps each to Vespa's active
# CartesianIndex (1-based), looks up the recorded flux NTuple, and returns the
# `Vector{Float64}` in Enzo's units (`enzo_value = bflux/Vcell`) for component `comp`.
#
# `start`,`stop` are the length-`rank` global StartGlobalIndex/EndGlobalIndex of
# this face plane; `g0` is the grid's first-active-cell global index (length-rank).
# `flux_off` is the active-index offset along the flux dim itself: the orthogonal
# dims map straight (`a = g − g0 + 1`), but the flux-dim register key follows the
# verified 1D mapping (Left subgrid face → lo cell `g − g0`, Right → `g − g0 + 1`;
# own outer boundary → the boundary cell directly), supplied by the caller.
# `lookup(I)` returns the recorded flux for active cell `I`, or `nothing` (⇒ 0).
@inline function _plane_dims(start::NTuple{R,Int}, stop::NTuple{R,Int}) where {R}
    return ntuple(d -> stop[d] - start[d] + 1, Val(R))
end

function bflux_plane(::Val{R}, dim::Int, start::NTuple{R,Int}, stop::NTuple{R,Int},
                     g0::NTuple{R,Int}, flux_off::Int, comp::Int, Vcell::Real,
                     lookup) where {R}
    Dim = _plane_dims(start, stop)
    n = prod(Dim)
    out = Vector{Float64}(undef, n)
    @inbounds for lin in 0:n-1
        # decode column-major (dim-0 fastest) plane offset → per-dim global index
        rem = lin
        active = ntuple(Val(R)) do d
            od = rem % Dim[d]
            rem ÷= Dim[d]
            g = start[d] + od
            d == dim + 1 ? flux_off : (g - g0[d] + 1)   # flux dim: caller-supplied active key; else direct
        end
        I = CartesianIndex(active)
        F = lookup(I)
        out[lin+1] = (F !== nothing && comp > 0) ? Float64(F[comp]) / Float64(Vcell) : 0.0
    end
    return out
end
