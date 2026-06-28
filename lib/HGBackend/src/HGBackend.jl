"""
    HGBackend

Adapter that implements the `MeshInterface` contract (ADR-0001, P2) on top of
**HierarchicalGrids.jl** (HG). It is the AMR backend behind the seam; the in-repo
`RefMesh` is the uniform convergence oracle it is validated against.

HG is the substrate the architecture targets: integer-exact relative geometry,
a DFS-ordered octree with O(1) navigation, BC-aware face-neighbor queries with
hanging-node enumeration (`face_fine_neighbors`), and an `AdaptiveField` that
conservatively remaps field data on refinement/coarsening. This adapter is thin
— it maps Vespa's handle-based, ghost-free seam onto HG's native operations and
does no physics.

## Field storage

Cell-average state is stored as a `PolynomialFieldSet` with `BernsteinBasis{D,0}`
(degree 0 ⇒ the single coefficient *is* the finite-volume cell mean — the same
choice HG's own Sod example makes). This is the representation `AdaptiveField`
supports, giving HG's tested conservative remap-on-refine (piecewise-constant
prolongation; volume-weighted-mean restriction) for free. A thin `HGScalarView`
exposes it through the seam's scalar `view[cell]` contract.

## Mesh & topology

`HGMesh` is mutable and **2:1 balanced** (`balanced=true`), so refinement keeps
the level gap across any face ≤ 1. Cell handles are HG integer leaf ids. The leaf
list and neighbor graph are re-derived after every regrid (`_resync!`); HG caches
the neighbor graph internally and invalidates it on mutation.
"""
module HGBackend

using MeshInterface
import MeshInterface as MI

using HierarchicalGrids: HierarchicalMesh, EulerianFrame, FrameBoundaries,
                         refine_cells!, coarsen_cells!, enumerate_leaves, n_cells,
                         cell_physical_box, ensure_physical_boxes!,
                         ensure_neighbor_graph!, face_neighbors,
                         face_fine_neighbors, face_neighbors_with_bcs,
                         find_children, is_leaf, level_of, isotropic_mask,
                         allocate_polynomial_fields, BernsteinBasis,
                         AdaptiveField, dispose!,
                         BCKind, PERIODIC, OUTFLOW, REFLECTING
import HierarchicalGrids as HG

export HGMesh

# ─────────────────────────── layout mapping ─────────────────────────────────
_hg_layout(::MI.SoA) = HG.SoA()
_hg_layout(::MI.AoS) = HG.AoS()
_hg_layout(::MI.Blocked{B}) where {B} = HG.Blocked{B}()

# ─────────────────────── BC vocabulary mapping ──────────────────────────────
_hg_bckind(::Outflow)    = OUTFLOW
_hg_bckind(::Reflecting) = REFLECTING
_hg_bckind(::Periodic)   = PERIODIC

_hg_frame_bcs(bcs::BoundaryConditions, D::Int) =
    FrameBoundaries(ntuple(d -> (_hg_bckind(bc_on(bcs, d, :lo)),
                                 _hg_bckind(bc_on(bcs, d, :hi))), D))
# Static-rank form: `ntuple(_, Val(D))` is type-stable ⇒ the FrameBoundaries is
# isbits/stack-allocated. Use this wherever `D` is a type parameter to avoid the
# per-call allocation of the runtime-`Int` method above.
_hg_frame_bcs(bcs::BoundaryConditions, ::Val{D}) where {D} =
    FrameBoundaries(ntuple(d -> (_hg_bckind(bc_on(bcs, d, :lo)),
                                 _hg_bckind(bc_on(bcs, d, :hi))), Val(D)))

# ─────────────────────────────── the mesh ───────────────────────────────────

"""
    HGMesh(dims, domain; T=Float64)

A 2:1-balanced HG-backed mesh, initialized uniform (each entry of `dims` the same
power of two; the root is isotropically refined `log2(dims[1])` times). It refines
dynamically thereafter via [`refine!`](@ref). `domain` is a tuple of `(lo, hi)`
physical bounds per axis.
"""
mutable struct HGMesh{D,T,M} <: AbstractMeshBackend
    mesh::HierarchicalMesh{D,M}      # M (sibling-index type) concretely captured ⇒ `m.mesh`
                                     # access is type-stable (no per-call box in neighbor/
                                     # face_neighbor_handles/for_each_face).
    frame::EulerianFrame{D,T}
    leaves::Vector{Int}
    lo::NTuple{D,T}
    hi::NTuple{D,T}
    base_level::Int        # HG absolute level of the uniform base grid (log2(dims))
end

function HGMesh(dims::NTuple{D,Integer}, domain::NTuple{D,<:Tuple};
               T::Type = Float64) where {D}
    n = dims[1]
    all(==(n), dims) ||
        throw(ArgumentError("HGMesh requires equal dims per axis, got $dims"))
    (n > 0 && ispow2(n)) ||
        throw(ArgumentError("HGMesh requires dims to be a power of two, got $n"))
    k = round(Int, log2(n))

    mesh = HierarchicalMesh{D}(; balanced = true)
    for _ in 1:k
        refine_cells!(mesh, enumerate_leaves(mesh))
    end

    lo = ntuple(d -> T(domain[d][1]), D)
    hi = ntuple(d -> T(domain[d][2]), D)
    frame = EulerianFrame(mesh, lo, hi)

    m = HGMesh(mesh, frame, Int[], lo, hi, k)   # infer {D,T,M} from the field types
    _resync!(m)
    return m
end

# Re-derive the leaf list and warm HG's caches after any mesh mutation.
function _resync!(m::HGMesh)
    ensure_physical_boxes!(m.frame)
    ensure_neighbor_graph!(m.mesh)
    m.leaves = Int.(enumerate_leaves(m.mesh))
    return m
end

@inline _face_index(axis::Int, side::Symbol) = side === :lo ? 2axis - 1 : 2axis

# ─────────────────────────── topology / shape ───────────────────────────────
MI.rank(::HGMesh{D}) where {D} = D
MI.domain(m::HGMesh{D}) where {D} = ntuple(d -> (m.lo[d], m.hi[d]), D)
MI.n_cells(m::HGMesh) = length(m.leaves)
# Level is reported *relative to the uniform base grid*: 0 = a base cell, 1 = a
# once-refined cell, etc. This is the natural AMR semantics for refinement
# policies (`max_level` = levels of adaptive refinement above the base) and keeps
# the meaning independent of how fine the base happens to be.
MI.level_of(m::HGMesh, i::Integer) = Int(level_of(m.mesh, i)) - m.base_level
MI.max_level(m::HGMesh) = maximum(i -> Int(level_of(m.mesh, i)), m.leaves) - m.base_level

function MI.for_each_cell(f, m::HGMesh; level::Union{Nothing,Integer} = nothing)
    if level === nothing
        @inbounds for i in m.leaves
            f(i)
        end
    else
        lev = Int(level)
        @inbounds for i in m.leaves
            MI.level_of(m, i) == lev && f(i)
        end
    end
    return nothing
end

# ───────────────────────── geometry (derived) ───────────────────────────────
@inline function MI.cell_center(m::HGMesh{D}, i::Integer) where {D}
    lo, hi = cell_physical_box(m.frame, i)
    return ntuple(d -> 0.5 * (lo[d] + hi[d]), D)
end

@inline function MI.cell_width(m::HGMesh{D}, i::Integer) where {D}
    lo, hi = cell_physical_box(m.frame, i)
    return ntuple(d -> hi[d] - lo[d], D)
end

@inline function MI.cell_volume(m::HGMesh{D}, i::Integer) where {D}
    lo, hi = cell_physical_box(m.frame, i)
    v = one(eltype(lo))                 # geometry precision (not leaked to Float64)
    @inbounds for d in 1:D
        v *= hi[d] - lo[d]
    end
    return v
end

@inline function MI.face_area(m::HGMesh{D}, i::Integer, axis::Int) where {D}
    lo, hi = cell_physical_box(m.frame, i)
    a = one(eltype(lo))
    @inbounds for d in 1:D
        d == axis || (a *= hi[d] - lo[d])
    end
    return a
end

# ─────────────────────── neighbor resolution (BC-aware) ─────────────────────
# Single representative neighbor (used by the PLM slope stencil). Across a
# coarse↔fine face this returns HG's representative leaf, which is an adequate
# stencil neighbor for the limited slope. Periodic wrap and domain BCs are
# resolved by HG's `face_neighbors_with_bcs`.
function MI.neighbor(m::HGMesh{D}, i::Integer, axis::Int, side::Symbol;
                     bcs::BoundaryConditions) where {D}
    f = _face_index(axis, side)
    nb = face_neighbors_with_bcs(m.mesh, i, _hg_frame_bcs(bcs, Val(D)))[f]
    (nb === nothing || nb == 0) && return DomainBoundary(bc_on(bcs, axis, side))
    return Interior(Int(nb))
end

# Bulk, allocation-free per-face neighbor handles (see MeshInterface). Reads the
# CACHED, BC-resolved representative tuple `face_neighbors_with_bcs` (a 0 entry is
# a domain boundary or non-leaf) and returns plain `Int` handles + a boundary mask
# — no per-face `NeighborRef` box, no runtime-`Int` FrameBoundaries rebuild.
@inline function MI.face_neighbor_handles(m::HGMesh{D}, i::Integer;
                                  bcs::BoundaryConditions) where {D}
    t = face_neighbors_with_bcs(m.mesh, i, _hg_frame_bcs(bcs, Val(D)))
    h = ntuple(f -> (t[f] === nothing || t[f] == 0) ? Int(i) : Int(t[f]), Val(2D))
    b = ntuple(f -> (t[f] === nothing || t[f] == 0), Val(2D))
    return (h, b)
end

# ───────────────────── face enumeration with hanging nodes ──────────────────
# Emit every unique (sub)face exactly once with the physical area of the SMALLER
# (fine) cell, so the flux-divergence driver is conservative across level jumps.
#
# Dedup rule, per leaf `i`, axis `d`:
#   • conforming faces (same level)  → emitted once, from the HI side;
#   • coarse↔fine faces              → emitted once, from the FINE cell's side
#     (the coarse cell skips them — it would otherwise see several fine
#     neighbors). `face_fine_neighbors` returns the per-face neighbor leaves:
#     several fine leaves when `i` is the coarse side (skip), or `[coarse rep]`
#     when `i` is the fine side (emit).
# Periodic axes (no fine neighbor recorded ⇒ empty list) are resolved via the
# BC-aware representative and emitted once from the hi side. Periodic across a
# refinement boundary is a documented follow-up (the AMR tests use Outflow).
# Reads the neighbor graph's per-leaf NTuple `representatives` + the multi-fine `fine`
# Dict DIRECTLY (instead of `face_fine_neighbors`, which allocates a fresh Vector —
# `copy(list)` / `UInt32[rep]` / `UInt32[]` — on EVERY call: 2·D·n_leaf per traversal,
# exploding (~27 allocs/leaf) on fragmented AMR meshes). Emission/dedup is unchanged:
# conforming/i-fine from the hi side, coarse↔fine from the fine side, periodic via the
# BC table, all once. NeighborRef args are concretely typed at each call site (no box).
function MI.for_each_face(f, m::HGMesh{D}; bcs::BoundaryConditions) where {D}
    g = ensure_neighbor_graph!(m.mesh)
    reps = g.representatives
    fined = g.fine
    fbcs = _hg_frame_bcs(bcs, D)
    mesh = m.mesh
    @inbounds for i in m.leaves
        ii = Int(i); li = Int(level_of(mesh, i))
        for d in 1:D
            a = MI.face_area(m, i, d)

            # ---- hi face (left = i, right = hi-neighbour) ----
            fhi = 2d
            fl = get(fined, (UInt32(ii), UInt8(fhi)), nothing)
            if fl === nothing
                rep = reps[ii][fhi]
                if rep == 0
                    per = face_neighbors_with_bcs(mesh, i, fbcs)[fhi]
                    if per === nothing || per == 0
                        f(Interior(i), DomainBoundary(bc_on(bcs, d, :hi)), d, a)
                    else
                        f(Interior(i), Interior(Int(per)), d, a)        # periodic, once
                    end
                elseif Int(level_of(mesh, rep)) <= li                   # conforming / i-fine sub-face
                    f(Interior(i), Interior(Int(rep)), d, a)
                end
            else
                for j in fl                                             # i coarse: neighbours finer → skip
                    Int(level_of(mesh, j)) > li && continue
                    f(Interior(i), Interior(Int(j)), d, a)
                end
            end

            # ---- lo face (emitted only as i-fine sub-faces: coarse → fine) ----
            flo = 2d - 1
            fl = get(fined, (UInt32(ii), UInt8(flo)), nothing)
            if fl === nothing
                rep = reps[ii][flo]
                if rep == 0
                    per = face_neighbors_with_bcs(mesh, i, fbcs)[flo]
                    (per === nothing || per == 0) &&
                        f(DomainBoundary(bc_on(bcs, d, :lo)), Interior(i), d, a)
                    # periodic lo face: emitted from the partner's hi side (skip)
                elseif Int(level_of(mesh, rep)) < li                    # coarse on the lo side → emit
                    f(Interior(Int(rep)), Interior(i), d, a)
                end
            else
                for j in fl
                    Int(level_of(mesh, j)) < li && f(Interior(Int(j)), Interior(i), d, a)
                end
            end
        end
    end
    return nothing
end

# ──────────────────────────────── fields ────────────────────────────────────
# Storage is a PolynomialFieldSet{Bernstein{D,0}} wrapped in an AdaptiveField so
# HG conservatively remaps it on refine/coarsen. The store is registered as a
# refinement listener at allocation; mutating the mesh remaps every store.

"Holds an `AdaptiveField` over a degree-0 polynomial field set (cell means).
Parametrized on the AdaptiveField's CONCRETE type so `parent(store.af)` (the inner
`PolynomialFieldSet`) is concretely typed — otherwise an abstract `af::AdaptiveField`
field erases the type and every `parent`/view access returns `Any` (allocating). The
inner field set keeps its type across regrids (`_rebuild_with_n` preserves the params),
so `A` is stable."
struct HGFieldStore{A<:AdaptiveField} <: AbstractFieldStore
    af::A
    names::Vector{Symbol}
end

"""
    HGScalarView(store, name)

Seam-contract view: `v[cell]` reads/writes the scalar cell mean of field `F`. It
dereferences `parent(store.af)` on each access so it stays valid across regrids
(AdaptiveField rebuilds the inner field set on refinement). The field name is a TYPE
parameter `F`, so `getproperty(parent(af), F)` resolves statically (no runtime-symbol
`getproperty` ⇒ no per-access allocation); combined with the concretely-typed
`HGFieldStore{A}`, the whole `v[cell]` chain inlines to a plain array index.
"""
struct HGScalarView{T,S,F} <: AbstractArray{T,1}
    store::S
end
# Infer the coefficient precision T from the named polynomial field (degree-0 ⇒
# the single coeff IS the cell mean; its element type is the field precision), and
# encode the field name as the type parameter F.
function HGScalarView(store, name::Symbol)
    pf = getproperty(Base.parent(store.af), name)
    T = length(pf) == 0 ? Float64 : typeof(pf[1][1])   # coefficient precision from a live entry
    return HGScalarView{T,typeof(store),name}(store)
end

@inline _pf(v::HGScalarView{T,S,F}) where {T,S,F} = getproperty(Base.parent(v.store.af), F)
Base.size(v::HGScalarView) = (Base.parent(v.store.af).n,)
Base.IndexStyle(::Type{<:HGScalarView}) = IndexLinear()
Base.@propagate_inbounds Base.getindex(v::HGScalarView, i::Integer) = _pf(v)[Int(i)][1]
Base.@propagate_inbounds Base.setindex!(v::HGScalarView{T}, x, i::Integer) where {T} =
    (_pf(v)[Int(i)] = (T(x),); x)

MI.field_eltype(::HGMesh{D,T}) where {D,T} = T   # default field precision = geometry T
MI.coord_eltype(::HGMesh{D,T}) where {D,T} = T

function MI.allocate_fields(m::HGMesh{D,T}, spec::FieldSpec;
                            layout::AbstractLayout = SoA(),
                            eltype::Type = T) where {D,T}
    layout isa SoA ||
        error("HGBackend: HG's adaptive polynomial storage is wired for SoA here " *
              "(got $(layout)). RefMesh exercises SoA/AoS/Blocked.")
    names = field_names(spec)
    basis = BernsteinBasis{D,0}()
    nt = NamedTuple{Tuple(names)}(ntuple(_ -> eltype, length(names)))   # field precision per name
    pfs = allocate_polynomial_fields(_hg_layout(layout), basis, n_cells(m.mesh); nt...)
    z = (zero(eltype),)
    @inbounds for nm in names, i in 1:n_cells(m.mesh)
        getproperty(pfs, nm)[i] = z
    end
    af = AdaptiveField(pfs, m.mesh)         # registers the conservative remap listener
    return HGFieldStore(af, collect(names))
end

MI.field_view(::HGMesh, store::HGFieldStore, name::Symbol) = HGScalarView(store, name)

# ───────────────────────────── AMR (refine / coarsen) ───────────────────────
# refine_cells!/coarsen_cells! mutate the tree and fire the refinement listeners
# of every registered field store, conservatively remapping their data. We then
# re-derive the leaf set and caches.

function MI.refine!(m::HGMesh{D}, cells) where {D}
    isempty(cells) && return m
    ids = collect(Int.(cells))
    # Pass explicit fully-isotropic masks: with the default (`nothing`), HG's
    # batch split does not reliably split every axis at depth (it can fall back
    # to a single-axis split), which would make refinement non-isotropic. An
    # explicit isotropic mask per cell guarantees 2^D children each.
    masks = fill(isotropic_mask(D), length(ids))
    refine_cells!(m.mesh, ids, masks)   # split_masks is positional in HG
    _resync!(m)
    return m
end

function MI.coarsen!(m::HGMesh, parents)
    isempty(parents) && return m
    coarsen_cells!(m.mesh, collect(Int.(parents)))
    _resync!(m)
    return m
end

"Parent id of a leaf's family (for coarsening decisions): the HG parent index."
parent_of(m::HGMesh, i::Integer) = Int(HG.find_parent(m.mesh, i))

"Children leaf ids of a (non-leaf) cell, or empty."
children_of(m::HGMesh, i::Integer) = Int.(find_children(m.mesh, i))

end # module
