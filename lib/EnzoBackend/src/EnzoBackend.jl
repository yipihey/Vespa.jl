"""
    EnzoBackend

A `MeshInterface` backend over a **live Enzo grid**, so Vespa's *unchanged*
`Simulation`/driver runs through the seam on Enzo-owned state — the full
seam-level integration (vs the E3 slot, which reused Vespa kernels directly).

`EnzoGridMesh` satisfies the seam by delegating the (uniform) geometry/topology to
a `RefMesh.UniformMesh` over the grid's ACTIVE region, and links the live Enzo
handle + field map. Because Vespa stores **conserved** `(ρ, ρv, E)` while Enzo
stores `(Density, Velocity1, TotalEnergy_specific)`, the field state cannot be a
zero-copy alias — `sync_from_enzo!`/`sync_to_enzo!` transform between them around
each Vespa step. ND single-grid (1D/2D/3D); AMR drives it per-grid per-level via
the Julia EvolveLevel (the `:julia` hydro slot iterates the grids on a level).
"""
module EnzoBackend

import MeshInterface as MI
using RefMesh: UniformMesh
import EnzoLib

export EnzoGridMesh, sync_from_enzo!, sync_to_enzo!, enzo_parent_ghost

# The backend stores only Ints — both the Enzo FieldType field indices and the
# CONSERVED-state role indices (which sv component is density / momentum / energy).
# The role indices are supplied by the caller FROM the EquationSet model
# (`density_index`/`momentum_indices`/`energy_index`), so the variable choice is
# the model's, not hardcoded here — and EnzoBackend stays free of any Vespa dep.
# T is the FIELD-STATE precision (Float64 default, Float32 for the precision/perf
# benchmark). GEOMETRY stays Float64 (`geom`) — the hydro kernels take `area::Float64`,
# and widths/areas are O(N) and uniform — while the conserved-state ARRAYS the
# kernels read/write are typed T (allocated from `alloc`, a same-shape T-mesh). So
# a Float32 mesh is an f32-STORAGE solver (the bandwidth-dominant part) with f64
# geometry: exactly the realistic precision knob.
struct EnzoGridMesh{N,T} <: MI.AbstractMeshBackend
    geom::UniformMesh{N,Float64}      # geometry/topology (f64): centers, widths, areas, faces
    alloc::UniformMesh{N,T}           # same-shape T-mesh used ONLY for field allocation/views
    h::Ptr{Cvoid}                     # live Enzo session/problem handle
    grid::Int                         # grid index in the hierarchy
    active::NTuple{N,Int}             # active cells per dim (= geom dims, ghost-free)
    strides::NTuple{N,Int}            # column-major strides into the flat Enzo array (incl ghosts)
    nghost::Int                       # ghost zones per side
    di::Int                           # Enzo 0-based field index: Density
    vi::NTuple{3,Int}                 # Velocity1/2/3 field indices, -1 if absent
    ei::Int                           # TotalEnergy field index
    cdi::Int                          # conserved index of mass density
    cmom::NTuple{3,Int}               # conserved indices of (x,y,z) momentum
    cei::Int                          # conserved index of total energy
end

# 0-based Enzo field index for a FieldType, or -1 if the grid lacks it.
function _field_or(h, ftype::Integer, grid::Integer)
    try
        return EnzoLib.field_index(h, ftype; grid = grid)
    catch
        return -1
    end
end

"""
    EnzoGridMesh(h; grid=0, nghost=3, domain, precision=Float64,
                 cons_density=1, cons_momentum=(2,3,4), cons_energy=5)

Wrap a live Enzo grid (1D/2D/3D) as a seam backend. `domain` is the N-tuple of
per-axis `(lo,hi)` (must match the grid rank). `precision` is the field-state
element type. The `cons_*` role indices come from the `EquationSet` model.
"""
function EnzoGridMesh(h::Ptr{Cvoid}; grid::Integer = 0, nghost::Integer = 3,
                      domain = ((0.0, 1.0),), precision::Type = Float64,
                      cons_density::Integer = 1, cons_momentum = (2, 3, 4),
                      cons_energy::Integer = 5)
    rank = EnzoLib.problem_grid_rank(h, grid)
    rank == length(domain) ||
        throw(ArgumentError("EnzoGridMesh: grid rank $rank ≠ domain length $(length(domain))"))
    gd = EnzoLib.problem_grid_dims(h, grid)                       # full per-dim dims (incl ghosts)
    gdims  = ntuple(d -> gd[d], rank)
    active = ntuple(d -> gdims[d] - 2 * nghost, rank)
    strides = ntuple(d -> d == 1 ? 1 : prod(ntuple(k -> gdims[k], d - 1)), rank)
    geom  = UniformMesh(active, domain)
    alloc = UniformMesh(active, domain; T = precision)
    di = EnzoLib.field_index(h, 0; grid = grid)                  # Density
    vi = ntuple(k -> _field_or(h, 3 + k, grid), 3)               # Velocity1/2/3 (types 4/5/6)
    ei = EnzoLib.field_index(h, 1; grid = grid)                  # TotalEnergy (specific)
    return EnzoGridMesh{rank,precision}(geom, alloc, h, Int(grid), active, strides, Int(nghost),
                                        di, vi, ei, Int(cons_density),
                                        Tuple(Int.(cons_momentum)), Int(cons_energy))
end

# 1-based flat index into the column-major Enzo field array (incl ghosts) for the
# active cell at CartesianIndex `I` (1-based over the active region).
@inline function _enzo_flat(m::EnzoGridMesh{N}, I::CartesianIndex{N}) where {N}
    f = 0
    @inbounds for d in 1:N
        f += (m.nghost + I[d] - 1) * m.strides[d]
    end
    return f + 1
end

# ── seam: geometry/topology → f64 `geom`; field storage → T-precision `alloc` ──
MI.rank(m::EnzoGridMesh)        = MI.rank(m.geom)
MI.domain(m::EnzoGridMesh)      = MI.domain(m.geom)
MI.n_cells(m::EnzoGridMesh)     = MI.n_cells(m.geom)
MI.max_level(m::EnzoGridMesh)   = MI.max_level(m.geom)
MI.level_of(m::EnzoGridMesh, args...)    = MI.level_of(m.geom, args...)
MI.cell_center(m::EnzoGridMesh, args...) = MI.cell_center(m.geom, args...)
MI.cell_width(m::EnzoGridMesh, args...)  = MI.cell_width(m.geom, args...)
MI.cell_volume(m::EnzoGridMesh, args...) = MI.cell_volume(m.geom, args...)
MI.face_area(m::EnzoGridMesh, args...)   = MI.face_area(m.geom, args...)
MI.neighbor(m::EnzoGridMesh, args...; kw...)        = MI.neighbor(m.geom, args...; kw...)
MI.allocate_fields(m::EnzoGridMesh, args...; kw...) = MI.allocate_fields(m.alloc, args...; kw...)
MI.field_eltype(m::EnzoGridMesh)                    = MI.field_eltype(m.alloc)   # field precision T
MI.coord_eltype(m::EnzoGridMesh)                    = MI.coord_eltype(m.geom)    # geometry: Float64
MI.field_view(m::EnzoGridMesh, args...)             = MI.field_view(m.alloc, args...)
MI.for_each_cell(f, m::EnzoGridMesh; kw...) = MI.for_each_cell(f, m.geom; kw...)
MI.for_each_face(f, m::EnzoGridMesh; kw...) = MI.for_each_face(f, m.geom; kw...)

# ── field sync: live Enzo grid ↔ Vespa conserved views (ND) ──────────────────
# Each active cell (CartesianIndex over the active region) maps to a column-major
# flat index of the ghost-zoned Enzo field via `_enzo_flat`. Enzo stores
# (Density, Velocity1..k, TotalEnergy_specific); Vespa stores conserved
# (ρ, ρv, E_density). Velocity components absent on the grid contribute 0 momentum.
"Pull the live Enzo grid state into Vespa's conserved views `sv` (Enzo → conserved)."
function sync_from_enzo!(sv, m::EnzoGridMesh{N,T}) where {N,T}
    d  = EnzoLib.problem_get_field(m.h, m.di, m.grid)
    es = EnzoLib.problem_get_field(m.h, m.ei, m.grid)     # specific total energy
    vf = ntuple(k -> m.vi[k] >= 0 ? EnzoLib.problem_get_field(m.h, m.vi[k], m.grid) : nothing, 3)
    @inbounds for I in CartesianIndices(m.active)
        f = _enzo_flat(m, I)
        ρ = d[f]
        sv[m.cdi][I] = ρ
        for k in 1:3
            vk = vf[k] === nothing ? 0.0 : vf[k][f]
            sv[m.cmom[k]][I] = ρ * vk
        end
        sv[m.cei][I] = ρ * es[f]                          # total energy density = ρ·e_specific
    end
    return nothing
end

"Push Vespa's conserved views `sv` back into the live Enzo grid (conserved → Enzo)."
function sync_to_enzo!(m::EnzoGridMesh{N,T}, sv) where {N,T}
    d  = EnzoLib.problem_get_field(m.h, m.di, m.grid)
    es = EnzoLib.problem_get_field(m.h, m.ei, m.grid)
    vf = ntuple(k -> m.vi[k] >= 0 ? EnzoLib.problem_get_field(m.h, m.vi[k], m.grid) : nothing, 3)
    @inbounds for I in CartesianIndices(m.active)
        f = _enzo_flat(m, I)
        ρ = sv[m.cdi][I]; E = sv[m.cei][I]
        d[f] = ρ
        for k in 1:3
            vf[k] === nothing || (vf[k][f] = sv[m.cmom[k]][I] / ρ)
        end
        es[f] = E / ρ                                     # back to specific total energy
    end
    EnzoLib.problem_set_field(m.h, m.di, d; grid = m.grid)
    EnzoLib.problem_set_field(m.h, m.ei, es; grid = m.grid)
    for k in 1:3
        m.vi[k] >= 0 && EnzoLib.problem_set_field(m.h, m.vi[k], vf[k]; grid = m.grid)
    end
    return nothing
end

# ── parent-ghost coupling (ADR-0003 follow-up #1) ─────────────────────────────
# Enzo fills a subgrid's ghost zones from its parent (InterpolateBoundaryFromParent,
# via session_set_boundary) BEFORE the hydro solve. Vespa's driver is ghost-free
# and otherwise synthesizes an Outflow (zero-gradient) ghost at a subgrid's outer
# faces — wrong when a wave sits ON the coarse–fine interface (rel-error × flux =
# the residual ~1e-3 per-step / ~1e-5 end-to-end drift). This reads Enzo's ALREADY-
# interpolated ghost zone adjacent to a boundary active cell and returns it as a
# CONSERVED tuple in the model's role order (cdi, cmom[1..3], cei), so the driver
# (via a ParentGhost BC) uses the parent value instead of an Outflow copy.
#
# Snapshot the field arrays ONCE (they are valid at hook entry, right after
# session_set_boundary, and Vespa only writes ACTIVE cells back within a step) and
# capture them in the returned closure `(axis, side, cell) -> U_cons::NTuple{NV,T}`.
# The ghost cell adjacent to active boundary cell `I` is one Enzo zone outward of
# `_enzo_flat(m, I)` along `axis`: −strides[axis] for :lo, +strides[axis] for :hi.

"Enzo conserved (role-ordered) ghost tuple `NV`-long; absent velocity ⇒ 0 momentum."
@inline function _enzo_ghost_cons(m::EnzoGridMesh{N,T}, ::Val{NV}, d, es, vf,
                                  axis::Int, side::Symbol, cell::CartesianIndex{N}) where {N,T,NV}
    base = _enzo_flat(m, cell)
    g = side === :lo ? base - m.strides[axis] : base + m.strides[axis]
    @inbounds begin
        ρ  = d[g]
        E  = ρ * es[g]                                  # total energy density = ρ·e_specific
        px = vf[1] === nothing ? zero(T) : T(ρ * vf[1][g])
        py = vf[2] === nothing ? zero(T) : T(ρ * vf[2][g])
        pz = vf[3] === nothing ? zero(T) : T(ρ * vf[3][g])
    end
    # Place into role order. cdi/cmom/cei are 1..NV; build by component lookup.
    return ntuple(Val(NV)) do c
        c == m.cdi    ? T(ρ) :
        c == m.cmom[1] ? px :
        c == m.cmom[2] ? py :
        c == m.cmom[3] ? pz :
        c == m.cei    ? T(E) : zero(T)
    end
end

"""
    enzo_parent_ghost(m::EnzoGridMesh) -> closure(axis, side, cell) -> U_cons::NTuple

Snapshot grid `m`'s live Enzo ghost zones (parent-interpolated by Enzo's
`session_set_boundary` before the solve) and return a closure giving the CONSERVED
role-ordered ghost state at any outer boundary face. Wrap it in a `ParentGhost` BC
(after `cons2prim`) so Vespa's driver consumes Enzo's parent ghosts at coarse–fine
interfaces instead of an Outflow copy. Call ONCE per step, after `sync_from_enzo!`.
"""
function enzo_parent_ghost(m::EnzoGridMesh{N,T}) where {N,T}
    d  = EnzoLib.problem_get_field(m.h, m.di, m.grid)
    es = EnzoLib.problem_get_field(m.h, m.ei, m.grid)
    vf = ntuple(k -> m.vi[k] >= 0 ? EnzoLib.problem_get_field(m.h, m.vi[k], m.grid) : nothing, 3)
    NV = max(m.cdi, m.cei, maximum(m.cmom))                 # conserved variable count (role indices are 1..NV)
    return (axis::Int, side::Symbol, cell::CartesianIndex{N}) ->
        _enzo_ghost_cons(m, Val(NV), d, es, vf, axis, side, cell)
end

end # module
