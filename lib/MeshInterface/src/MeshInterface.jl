"""
    MeshInterface

The load-bearing boundary of EnzoNG (ADR-0001, P2/P10): the single interface
that the driver, solvers, problem specs, and analysis are written against. It
defines **everything** the project requires of an AMR substrate and **nothing**
about any concrete backend.

A backend is any concrete `AbstractMeshBackend` that implements the generic
functions declared here. The first two backends are the in-repo `RefMesh` (a
pure, correct reference) and `HGBackend` (an adapter over HierarchicalGrids.jl).
The reference and the adapter are validated test-for-test against each other.

## Design: handle-based, ghost-free, neighbor-driven

The seam is shaped by the operations a finite-volume solver actually performs on
an AMR substrate — not by any one backend's storage. A cell is an **opaque
handle** (a `CartesianIndex` for the uniform reference; an `Int` cell id for
HierarchicalGrids.jl). There are **no ghost cells** in the interface: boundaries
are resolved per-face by [`neighbor`](@ref), which returns either an interior
neighbor handle (including periodic wrap) or a [`DomainBoundary`](@ref) carrying
the boundary condition. This is exactly the model a hierarchical, hanging-node
mesh exposes, and it degenerates correctly to a uniform grid.

Fields are **cell-average (finite-volume)** scalars (ADR P2), one per named
component per cell, allocated with a per-field memory layout (P3) and read
through zero-copy handle-indexed views. Conservative inter-level transfer (P5)
is `restrict!`/`prolong!`. Measurement (P10) is the `Instrumented{B}` wrapper,
which compiles away when unused.
"""
module MeshInterface

# ───────────────────────────────── Backend ──────────────────────────────────

"""
    AbstractMeshBackend

Supertype of every AMR substrate. The driver/solvers/specs never name a concrete
subtype; the backend is injected at the `Simulation` constructor (a single type
parameter). Implement the generic functions below for a new backend.
"""
abstract type AbstractMeshBackend end

# ───────────────────────────────── Layouts (P3) ─────────────────────────────
# Logical indexing is decoupled from physical memory layout. Switching layout is
# a constructor change, never a kernel change. Layout is assigned by physics.

"Supertype of field memory layouts."
abstract type AbstractLayout end

"Struct-of-arrays: each field is its own contiguous array. Favors directional sweeps."
struct SoA <: AbstractLayout end

"Array-of-structs: all fields of a cell adjacent. Favors per-cell physics (chemistry)."
struct AoS <: AbstractLayout end

"""
    Blocked{B}

Cells grouped into `B`-sized blocks (the Taichi `ti.root.dense(M).dense(B)`
analog). Favors cache-blocked sweeps and coarse Morton-ordered regions.
"""
struct Blocked{B} <: AbstractLayout end

block_size(::Blocked{B}) where {B} = B

# ───────────────────────── Boundary conditions ──────────────────────────────
# Plain value types. They are resolved by the backend in `neighbor`; the solver
# layer interprets the returned `DomainBoundary` (e.g. flipping the normal
# momentum for `Reflecting`). Periodic boundaries are never reported as a
# DomainBoundary — `neighbor` returns the wrapped interior cell instead.

"Supertype of boundary conditions."
abstract type AbstractBC end

"Zero-gradient (the ghost state copies the boundary cell). Used by the Sod tube."
struct Outflow <: AbstractBC end

"Wrap-around boundary. `neighbor` returns the opposite-edge cell as interior."
struct Periodic <: AbstractBC end

"Mirror boundary (the ghost state reflects the boundary cell, normal velocity flipped)."
struct Reflecting <: AbstractBC end

"""
    BoundaryConditions(spec)

Per-axis, per-side boundary conditions. `spec` is one `AbstractBC` (applied to
every side) or a tuple of `(lo, hi)` `AbstractBC` pairs, one pair per axis.
"""
struct BoundaryConditions{N}
    sides::NTuple{N,Tuple{AbstractBC,AbstractBC}}
end

# A single BC applied to every side. (The inner constructor converts the
# concrete pair tuple to the `Tuple{AbstractBC,AbstractBC}` field type.)
BoundaryConditions(bc::AbstractBC, ::Val{N}) where {N} =
    BoundaryConditions{N}(ntuple(_ -> (bc, bc), N))

# Per-axis `(lo, hi)` pairs of (possibly heterogeneous) BC subtypes. Signature is
# deliberately `::Tuple` so it does not collide with the struct's auto-generated
# `NTuple{N,Tuple{AbstractBC,AbstractBC}}` outer constructor.
function BoundaryConditions(pairs::Tuple)
    N = length(pairs)
    sides = ntuple(d -> (pairs[d][1]::AbstractBC, pairs[d][2]::AbstractBC), N)
    return BoundaryConditions{N}(sides)
end

"BC on a given `axis` and `side` (`:lo` or `:hi`)."
@inline bc_on(bcs::BoundaryConditions, axis::Int, side::Symbol) =
    side === :lo ? bcs.sides[axis][1] : bcs.sides[axis][2]

# ───────────────────────── Field specification (P2) ─────────────────────────

"""
    FieldSpec(names)

Declares the named cell-average fields a problem needs. `names` is an iterable of
`Symbol`s. The field model is finite-volume cell averages (one scalar per cell
per field); face/edge-centered storage (for MHD constrained transport) is a
later, additive extension and is intentionally not part of this milestone.
"""
struct FieldSpec
    names::Vector{Symbol}
    # Inner constructor (replaces the default), so there is exactly one
    # `FieldSpec` method and no collision with an auto-generated one.
    FieldSpec(names) = new(Symbol[Symbol(n) for n in names])
end

field_names(s::FieldSpec) = s.names

"Opaque, backend-owned handle to allocated field memory. Obtain views via `field_view`."
abstract type AbstractFieldStore end

# ───────────────────────── Neighbor resolution ──────────────────────────────

"Result of a [`neighbor`](@ref) query: an interior cell, or a domain boundary."
abstract type NeighborRef end

"An interior neighbor (including a periodic wrap). `cell` is a backend cell handle."
struct Interior{H} <: NeighborRef
    cell::H
end

"A domain boundary carrying the boundary condition the solver must apply."
struct DomainBoundary{BC<:AbstractBC} <: NeighborRef
    bc::BC
end

# ──────────────────────────── Interface contract ────────────────────────────
# Generic functions a backend must implement. Declared here with docstrings;
# methods live in backends. Calling an unimplemented one is a MethodError that
# names exactly what the backend is missing.

# -- shape / topology (P2) --
"`rank(backend)` → spatial dimensionality."
function rank end
"`domain(backend)` → physical extent, a tuple of `(lo, hi)` per axis."
function domain end
"`n_cells(backend)` → number of leaf cells."
function n_cells end
"""
    for_each_cell(f, backend; level=nothing)

Apply `f(cell)` to every leaf cell, where `cell` is an opaque backend handle
usable to index field views and to query geometry/neighbors.

With `level` set, iterate only the leaves at that refinement level (root = 0) —
the primitive AMR time-subcycling needs to advance one level while freezing the
others. `level=nothing` (the default) iterates all leaves, identical to the
single-rate path. On a uniform mesh `level=0` visits every cell and any
`level>0` visits none, so subcycling degenerates to one global step.
"""
function for_each_cell end
"`level_of(backend, cell)` → refinement level (root = 0)."
function level_of end
"`max_level(backend)` → deepest refinement level present."
function max_level end
"""
    refine!(backend, cells)

Split every leaf handle in the collection `cells`, conservatively **prolonging**
all fields the backend has been asked to track (injection for cell averages —
each child inherits the parent average). The backend re-derives its leaf set and
neighbor topology; cached handles into field stores remain valid. Returns the
backend.
"""
function refine! end
"""
    coarsen!(backend, parents)

Merge the children of every handle in `parents` back into a leaf, conservatively
**restricting** all tracked fields (volume-weighted mean of children). Returns
the backend.
"""
function coarsen! end

# -- geometry (P4, integer-exact; physical coords derived, never float128) --
"`cell_center(backend, cell)` → physical center coordinates (derived)."
function cell_center end
"`cell_width(backend, cell)` → physical width per axis (derived)."
function cell_width end
"`cell_volume(backend, cell)` → physical cell volume (derived)."
function cell_volume end
"`face_area(backend, cell, axis)` → area of `cell`'s face normal to `axis` (derived)."
function face_area end
"""
    neighbor(backend, cell, axis, side; bcs) -> NeighborRef

The neighbor of `cell` across the face on `side` (`:lo`/`:hi`) of `axis`, with
boundaries resolved per `bcs`. Returns [`Interior`](@ref) for an interior or
periodic-wrapped neighbor, or [`DomainBoundary`](@ref) otherwise. Exact integer
topology — no float epsilon.
"""
function neighbor end

"""
    for_each_face(f, backend; bcs)

Apply `f(left, right, axis, area)` exactly once to every unique (sub)face of the
mesh. `left` and `right` are [`NeighborRef`](@ref)s — `Interior(cell)` for a real
cell or `DomainBoundary(bc)` for a domain edge — and the `+axis` normal points
from `left` to `right`. `area` is the physical area of the (smaller) face.

This is the backend's responsibility precisely because face enumeration is where
AMR topology lives: a conforming face is emitted once; a coarse↔fine face is
emitted as several **sub-faces**, each carrying the fine cell's (smaller) area,
so a flux-divergence solver written against this contract is automatically
conservative across refinement-level jumps and hanging nodes. On a uniform mesh
it degenerates to one interior face per cell-pair plus the domain-boundary faces.
"""
function for_each_face end

# -- fields (P3) --
"""
    allocate_fields(backend, spec::FieldSpec; layout=SoA()) → AbstractFieldStore

Allocate the declared cell-average fields with the requested memory layout.
"""
function allocate_fields end
"""
    field_view(backend, store, name) → view

A handle-indexed view of one named field: `view[cell]` reads/writes that cell's
average. Logically identical across layouts.
"""
function field_view end

# -- inter-level transfer (P5, conservative by construction) --
"`restrict!(backend, coarse_store, fine_store)` → volume-weighted mean of children."
function restrict! end
"`prolong!(backend, fine_store, coarse_store)` → injection (child inherits parent average)."
function prolong! end

# ───────────────────────────── Instrumentation (P10) ────────────────────────
include("instrument.jl")

export AbstractMeshBackend,
    AbstractLayout, SoA, AoS, Blocked, block_size,
    AbstractBC, Outflow, Periodic, Reflecting, BoundaryConditions, bc_on,
    FieldSpec, field_names, AbstractFieldStore,
    NeighborRef, Interior, DomainBoundary,
    rank, domain, n_cells, for_each_cell, for_each_face, level_of, max_level,
    refine!, coarsen!,
    cell_center, cell_width, cell_volume, face_area, neighbor,
    allocate_fields, field_view, restrict!, prolong!,
    Instrumented, span_report, reset_spans!

end # module
