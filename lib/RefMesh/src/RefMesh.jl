"""
    RefMesh

A small, deliberately simple, **correct** pure-Julia implementation of the
`MeshInterface` contract (ADR-0001, P2). It is the reference backend and the
correctness oracle that `HGBackend` (and later Rust/GPU backends) are validated
against test-for-test.

Priorities: clarity and exactness over speed. Geometry is integer-exact —
logical cell indices are integers; physical centers/widths/volumes are *derived*
on demand (P4), never stored as absolute floats. There are **no ghost cells**:
boundaries are resolved per-face by `neighbor`, exactly as a hierarchical mesh
does, so the same solver runs unchanged here and on HierarchicalGrids.jl.

Field storage honors the `SoA` / `AoS` / `Blocked{B}` layouts (P3) behind
identical handle indexing, so solver kernels are layout-independent. Cell handles
are `CartesianIndex`. Milestone-1 scope is a single-level uniform structured
mesh with the cell-average finite-volume field model and conservative 2:1
restrict/prolong (P5); hierarchical `refine!`/`coarsen!` are deferred.
"""
module RefMesh

using MeshInterface
import MeshInterface as MI

export UniformMesh, BlockedArray

# ───────────────────────────── Blocked storage (P3) ─────────────────────────
# A layout-flexible array: logical CartesianIndex addressing over a buffer laid
# out in B^rank blocks. Logically identical to a dense array; physically blocked
# (the Taichi dense(M).dense(B) analog). Correctness, not speed, is the point.

"""
    BlockedArray{T}(undef, shape, B)

`N`-dimensional array of element type `T` and logical `shape`, stored in cubic
blocks of side `B`. Indexing is by logical coordinates; the block mapping is an
internal detail. Demonstrates that kernels using only logical indexing are
layout-independent.
"""
struct BlockedArray{T,N} <: AbstractArray{T,N}
    data::Vector{T}
    shape::NTuple{N,Int}
    B::Int
    nblocks::NTuple{N,Int}
end

function BlockedArray{T}(::UndefInitializer, shape::NTuple{N,Int}, B::Int) where {T,N}
    nblocks = ntuple(d -> cld(shape[d], B), N)
    len = prod(nblocks) * B^N
    return BlockedArray{T,N}(Vector{T}(undef, len), shape, B, nblocks)
end

Base.size(A::BlockedArray) = A.shape
Base.IndexStyle(::Type{<:BlockedArray}) = IndexCartesian()

@inline function _linindex(A::BlockedArray{T,N}, I::Vararg{Int,N}) where {T,N}
    Bn = A.B
    blk = 0          # block index, column-major over nblocks
    blkstride = 1
    ino = 0          # in-block index, column-major over B^N
    instride = 1
    @inbounds for d in 1:N
        i = I[d] - 1
        b = i ÷ Bn
        r = i % Bn
        blk += b * blkstride
        blkstride *= A.nblocks[d]
        ino += r * instride
        instride *= Bn
    end
    return blk * (Bn^N) + ino + 1
end

Base.@propagate_inbounds Base.getindex(A::BlockedArray{T,N}, I::Vararg{Int,N}) where {T,N} =
    A.data[_linindex(A, I...)]
Base.@propagate_inbounds Base.setindex!(A::BlockedArray{T,N}, v, I::Vararg{Int,N}) where {T,N} =
    (A.data[_linindex(A, I...)] = v)

# ───────────────────────────────── The mesh ─────────────────────────────────

"""
    UniformMesh(dims, domain; T=Float64)

Single-level uniform mesh. `dims` is the interior cell count per axis; `domain`
is a tuple of `(lo, hi)` physical bounds per axis. No ghost cells. Physical
coordinates appear only here, at the I/O boundary.
"""
struct UniformMesh{N,T} <: AbstractMeshBackend
    dims::NTuple{N,Int}
    lo::NTuple{N,T}
    hi::NTuple{N,T}
end

function UniformMesh(dims::NTuple{N,Integer}, domain::NTuple{N,<:Tuple};
                     T::Type = Float64) where {N}
    lo = ntuple(d -> T(domain[d][1]), N)
    hi = ntuple(d -> T(domain[d][2]), N)
    return UniformMesh{N,T}(Int.(dims), lo, hi)
end

# -- shape / topology --
MI.rank(::UniformMesh{N}) where {N} = N
MI.domain(m::UniformMesh{N}) where {N} = ntuple(d -> (m.lo[d], m.hi[d]), N)
MI.n_cells(m::UniformMesh) = prod(m.dims)
MI.max_level(::UniformMesh) = 0
MI.level_of(::UniformMesh, ::CartesianIndex) = 0

_cells(m::UniformMesh{N}) where {N} = CartesianIndices(ntuple(d -> 1:m.dims[d], N))

function MI.for_each_cell(f, m::UniformMesh; level::Union{Nothing,Integer} = nothing)
    # Uniform mesh is a single level (0): `level=0` (or nothing) visits all cells;
    # any deeper level visits none, so subcycling is a verified no-op here.
    (level === nothing || level == 0) || return nothing
    for I in _cells(m)
        f(I)
    end
    return nothing
end

# -- geometry (derived, integer-exact) --
@inline _width(m::UniformMesh{N}) where {N} =
    ntuple(d -> (m.hi[d] - m.lo[d]) / m.dims[d], N)

MI.cell_width(m::UniformMesh, ::CartesianIndex) = _width(m)
MI.cell_width(m::UniformMesh) = _width(m)
MI.cell_volume(m::UniformMesh, ::CartesianIndex) = prod(_width(m))

function MI.cell_center(m::UniformMesh{N,T}, I::CartesianIndex{N}) where {N,T}
    w = _width(m)
    return ntuple(d -> m.lo[d] + (I[d] - T(0.5)) * w[d], N)
end

"Area of the face normal to `axis` = product of the cell widths on the other axes."
function MI.face_area(m::UniformMesh{N}, ::CartesianIndex{N}, axis::Int) where {N}
    w = _width(m)
    a = one(eltype(w))
    for d in 1:N
        d == axis || (a *= w[d])
    end
    return a
end

# -- neighbor resolution (no ghosts; BC resolved here) --
const _SIDE = (lo = -1, hi = 1)

function MI.neighbor(m::UniformMesh{N}, I::CartesianIndex{N}, axis::Int, side::Symbol;
                     bcs::BoundaryConditions) where {N}
    step = _SIDE[side]
    j = I[axis] + step
    n = m.dims[axis]
    if 1 <= j <= n
        return Interior(_replace(I, axis, j))
    end
    bc = bc_on(bcs, axis, side)
    if bc isa Periodic
        jw = j < 1 ? n : 1                      # wrap to the opposite edge
        return Interior(_replace(I, axis, jw))
    end
    return DomainBoundary(bc)
end

@inline _replace(I::CartesianIndex{N}, axis::Int, val::Int) where {N} =
    CartesianIndex(ntuple(d -> d == axis ? val : I[d], N))

# Face enumeration: emit each unique face once. The hi-side query yields every
# interior face (incl. the periodic wrap, which the hi-edge cell resolves to the
# lo-edge cell) and the hi-domain boundary; the lo-side query contributes only
# true domain boundaries (interior/periodic lo faces are already emitted as some
# other cell's hi face). Uniform mesh ⇒ no hanging nodes, so every face is a full
# face with `face_area(cell, axis)`.
function MI.for_each_face(f, m::UniformMesh{N}; bcs::BoundaryConditions) where {N}
    for I in _cells(m)
        for d in 1:N
            a = MI.face_area(m, I, d)
            f(Interior(I), MI.neighbor(m, I, d, :hi; bcs = bcs), d, a)
            nb_lo = MI.neighbor(m, I, d, :lo; bcs = bcs)
            nb_lo isa Interior || f(nb_lo, Interior(I), d, a)
        end
    end
    return nothing
end

# -- fields (P3) --

"Backend-owned field storage. `view_of[name]` is the handle-indexed array for one field."
struct GridFields{N,L<:AbstractLayout} <: AbstractFieldStore
    layout::L
    shape::NTuple{N,Int}
    names::Vector{Symbol}
    view_of::Dict{Symbol,AbstractArray}
end

MI.field_eltype(::UniformMesh{N,T}) where {N,T} = T   # default field precision = geometry T
MI.coord_eltype(::UniformMesh{N,T}) where {N,T} = T   # geometry/coordinate precision

function MI.allocate_fields(m::UniformMesh{N,T}, spec::FieldSpec;
                            layout::AbstractLayout = SoA(),
                            eltype::Type = T) where {N,T}    # override ⇒ field precision ≠ geometry
    shape = m.dims
    names = field_names(spec)
    return GridFields(layout, shape, names, _build_views(layout, eltype, shape, names))
end

function _build_views(::SoA, ::Type{T}, shape::NTuple{N,Int}, names) where {T,N}
    d = Dict{Symbol,AbstractArray}()
    for nm in names
        d[nm] = zeros(T, shape)
    end
    return d
end

function _build_views(::AoS, ::Type{T}, shape::NTuple{N,Int}, names) where {T,N}
    backing = zeros(T, (length(names), shape...))
    d = Dict{Symbol,AbstractArray}()
    for (i, nm) in enumerate(names)
        d[nm] = view(backing, i, ntuple(_ -> Colon(), N)...)
    end
    return d
end

function _build_views(layout::Blocked, ::Type{T}, shape::NTuple{N,Int}, names) where {T,N}
    B = block_size(layout)
    d = Dict{Symbol,AbstractArray}()
    for nm in names
        A = BlockedArray{T}(undef, shape, B)
        fill!(A.data, zero(T))
        d[nm] = A
    end
    return d
end

MI.field_view(::UniformMesh, store::GridFields, name::Symbol) = store.view_of[name]

# -- conservative inter-level transfer (P5), 2:1 ratio --
# Single-level milestone: implemented as standalone coarse↔fine pair operations
# so the conservation property is testable now. `restrict!` is the volume-
# weighted mean of children (arithmetic mean for equal-volume 2:1 splits);
# `prolong!` is injection (each child inherits the parent average). Both
# preserve Σ value×volume to round-off.

const REFINE_RATIO = 2

"""
    restrict!(coarse_mesh, coarse_store, fine_store)

Average each `2^rank` block of fine cells into one coarse cell, for every field.
`fine` must have `2×` the dims of `coarse`.
"""
function MI.restrict!(cm::UniformMesh{N}, coarse::GridFields, fine::GridFields) where {N}
    r = REFINE_RATIO
    inv = 1 / r^N
    for nm in coarse.names
        C = coarse.view_of[nm]
        F = fine.view_of[nm]
        for Ic in CartesianIndices(ntuple(d -> 1:cm.dims[d], N))
            acc = zero(eltype(C))
            base = ntuple(d -> r * (Ic[d] - 1), N)
            for off in CartesianIndices(ntuple(_ -> 1:r, N))
                acc += F[CartesianIndex(ntuple(d -> base[d] + off[d], N))]
            end
            C[Ic] = acc * inv
        end
    end
    return nothing
end

"""
    prolong!(coarse_mesh, fine_store, coarse_store)

Inject each coarse cell average into its `2^rank` fine children, for every field.
"""
function MI.prolong!(cm::UniformMesh{N}, fine::GridFields, coarse::GridFields) where {N}
    r = REFINE_RATIO
    for nm in coarse.names
        C = coarse.view_of[nm]
        F = fine.view_of[nm]
        for Ic in CartesianIndices(ntuple(d -> 1:cm.dims[d], N))
            val = C[Ic]
            base = ntuple(d -> r * (Ic[d] - 1), N)
            for off in CartesianIndices(ntuple(_ -> 1:r, N))
                F[CartesianIndex(ntuple(d -> base[d] + off[d], N))] = val
            end
        end
    end
    return nothing
end

# -- AMR topology (deferred; RefMesh is the uniform convergence oracle) --
# Per ADR-0001 milestone, hierarchical AMR lives on HGBackend (which has the tree,
# hanging-node queries, and conservative remap); RefMesh stays uniform and is used
# as the fixed-resolution oracle that AMR runs are validated against. A uniform
# RefMesh at the AMR run's finest level is the convergence target.
MI.refine!(::UniformMesh, cells) =
    error("RefMesh is the uniform convergence oracle and does not refine; " *
          "run AMR on HGBackend and validate against a uniform-fine RefMesh.")
MI.coarsen!(::UniformMesh, parents) =
    error("RefMesh is the uniform convergence oracle and does not coarsen.")

end # module
