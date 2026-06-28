"""
    PPMKernels

A KernelAbstractions.jl port of Enzo's Piecewise-Parabolic-Method (PPM)
DirectEuler hydro solver — the `inteuler → twoshock → flux_twoshock → euler`
chain, with the dual-energy formalism, gravity, and colour advection — written
**once** and run on both the CPU (the parity oracle) and the Metal GPU.

Design contract:

  * **One source, two devices.** Every compute kernel is a precision-generic
    `@kernel` parameterised on the element type `T`. The CPU backend runs it in
    `Float64` (to certify the algorithm *is* Enzo's, bit-tight against the Fortran
    golden fixtures) and in `Float32`; the Metal backend runs it only in `Float32`
    (Apple GPUs have no `Float64`). f32 CPU↔Metal agreement is the parity gate.

  * **Backend by name.** `backend(:cpu)` always works; `backend(:metal)` resolves
    only after `using Metal` has loaded the package extension (Apple Silicon).
    Allocation/host-transfer go through [`device_zeros`](@ref) / [`to_device`](@ref)
    / [`to_host`](@ref), which the Metal extension specialises — the kernels and
    tests never name a concrete array type.
"""
module PPMKernels

using KernelAbstractions
const KA = KernelAbstractions

export backend, has_backend, device_zeros, to_device, to_host

# ── backend registry ─────────────────────────────────────────────────────────
# `:cpu` is always present; the Metal extension registers `:metal` on load.
const _BACKENDS = Dict{Symbol,Any}(:cpu => CPU())

"Register a KernelAbstractions backend under `name` (used by the Metal extension)."
register_backend!(name::Symbol, be) = (_BACKENDS[name] = be)

"True when backend `name` is available (`:metal` needs `using Metal` first)."
has_backend(name::Symbol) = haskey(_BACKENDS, name)

"""
    backend(name::Symbol = :cpu)

The KernelAbstractions backend registered under `name`. `:cpu` is always
available; `:metal` requires `using Metal` (Apple Silicon) to have loaded the
`PPMKernelsMetalExt` extension.
"""
function backend(name::Symbol = :cpu)
    return get(_BACKENDS, name) do
        error("PPM backend :$name is not available. " *
              (name === :metal ? "Run `using Metal` first (Apple Silicon only)." :
               "Known backends: $(collect(keys(_BACKENDS)))."))
    end
end

# ── device array helpers (specialised by the Metal extension) ────────────────
"""
    device_zeros(be, T, dims::Dims) -> AbstractArray{T}

A zero-filled array of element type `T` and shape `dims` living on backend `be`.
The CPU default is `zeros(T, dims)`; the Metal extension returns an `MtlArray`.
"""
device_zeros(::CPU, ::Type{T}, dims::Dims) where {T} = zeros(T, dims)

"""
    to_device(be, a, T = eltype(a)) -> AbstractArray{T}

Copy host array `a` onto backend `be`, converting to element type `T`. Used by
tests to stage fixture inputs (always `Float64` on disk) at the working precision.
"""
function to_device(be, a::AbstractArray, ::Type{T} = eltype(a)) where {T}
    d = device_zeros(be, T, size(a))
    copyto!(d, convert(Array{T}, a))
    return d
end

# Host reads are THE synchronization boundary: the compute kernels launch without
# per-kernel syncs (KA orders kernels on the backend queue; the sweep batches a
# single sync before recycling its scratch), so we synchronize here once before
# copying device memory to the host.
"`to_host(a)` — a plain host `Array` copy of a device array; synchronizes first."
function to_host(a::AbstractArray)
    KA.synchronize(KA.get_backend(a))
    return Array(a)
end

# ── scratch pool ──────────────────────────────────────────────────────────────
# Profiling the 3-D sweep showed ALLOCATION (not synchronization) is the GPU
# bottleneck — each step churns ~90 full-grid scratch arrays per sweep. When a
# pool is active (`with_pool`), hot-path scratch is recycled across sweeps/steps
# instead of reallocated; when inactive (the default — all tests), `_scratch`
# falls back to a fresh allocation, so behaviour is identical. The per-kernel
# syncs are kept (they let buffers complete before reuse).
export with_pool, clear_pool!

mutable struct ScratchPool
    free::Dict{Tuple{DataType,Int},Vector{Any}}   # available buffers by (type, length)
    used::Vector{Any}                              # checked out since the last reset
end
ScratchPool() = ScratchPool(Dict{Tuple{DataType,Int},Vector{Any}}(), Any[])
const _POOL = Ref{Union{Nothing,ScratchPool}}(nothing)

"Acquire a length-`len` scratch array shaped like `proto` (pooled when active)."
function _scratch(proto, len::Int; zero::Bool = true)
    pool = _POOL[]
    if pool === nothing
        a = similar(proto, len)
        zero && fill!(a, Base.zero(eltype(proto)))
        return a
    end
    bucket = get(pool.free, (typeof(proto), len), nothing)
    a = (bucket !== nothing && !isempty(bucket)) ? pop!(bucket) : similar(proto, len)
    push!(pool.used, a)
    zero && fill!(a, Base.zero(eltype(proto)))
    return a
end

"Return all checked-out scratch to the free list (call between independent sweeps)."
function _pool_reset!()
    pool = _POOL[]
    pool === nothing && return
    for a in pool.used
        push!(get!(() -> Any[], pool.free, (typeof(a), length(a))), a)
    end
    empty!(pool.used)
    return
end

"""
    with_pool(f)

Run `f` with the scratch pool active so repeated `ppm_step_3d!`/`ppm_sweep_1d!`
calls recycle buffers instead of reallocating. The pool is discarded on exit.
Wrap a benchmark/evolution loop in it for the GPU-allocation win.
"""
function with_pool(f)
    prev = _POOL[]
    _POOL[] = ScratchPool()
    try
        return f()
    finally
        _POOL[] = prev
    end
end

"Drop any cached scratch buffers held by the active pool."
clear_pool!() = (p = _POOL[]; p === nothing || (empty!(p.free); empty!(p.used)); nothing)

# ── compute kernels (added as each component is ported + certified) ──────────
include("eos.jl")              # pgas2d / pgas2d_dual            (Phase 2.1) ✓
include("calcdiss.jl")         # diffusion + flattening          (Phase 2.2) ✓
include("intvar.jl")           # per-variable PPM reconstruction (Phase 2.3a) ✓
include("inteuler.jl")         # PPM parabolic reconstruction     (Phase 2.3) ✓
include("twoshock.jl")         # two-shock Riemann solver         (Phase 2.4) ✓
include("flux_twoshock.jl")    # physical fluxes                  (Phase 2.5) ✓
include("euler.jl")            # conservative update              (Phase 2.6) ✓
include("colour.jl")           # passive species advection (rides the PPM mass flux)
include("sweep.jl")            # composed 1-D directional sweep   (Phase 3) ✓
include("ppm_grid.jl")         # 3-D Strang-split sweeps          (Phase 4) ✓
include("muscl.jl")            # PLM + HLL flux line (Enzo HydroMethod=3, HD_RK)
include("muscl_grid.jl")       # 3-D unsplit SSP-RK2 MUSCL driver (HydroMethod=3)
include("ppml.jl")             # PPML per-cell primitives (Ustyugov+ 2009)
include("ppml_grid.jl")        # 3-D Strang-split stateful PPML driver
include("reflux_capture.jl")   # frec → coarse–fine reflux registers (AMR flux correction)

end # module
