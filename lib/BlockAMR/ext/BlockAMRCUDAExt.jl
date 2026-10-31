"""
    BlockAMRCUDAExt

Package extension lighting up the CUDA backend for `BlockAMR`: registers `:cuda`
and specialises `device_zeros` onto `CuArray`. Loaded automatically when `CUDA`
is present in the environment.
"""
module BlockAMRCUDAExt

using BlockAMR
using CUDA
const Adapt = CUDA.Adapt

function __init__()
    if CUDA.functional()
        BlockAMR.register_backend!(:cuda, CUDABackend())
        BlockAMR.enable_tiling!(:cuda)     # ~42 KB shared mem fits NVIDIA (48+ KB)
    end
end

"""
    BlockAMR.disable_kernel_prefetch!() [CUDA]

Overwrite CUDA.jl's `KernelAdaptor` conversion for dense `CuArray`s to SKIP the
per-launch `cuMemPrefetchAsync` of unified-memory arguments.  CUDA.jl prefetches
the WHOLE array on EVERY kernel launch (unconditional; upstream "TODO: make
this configurable?").  Benign while total UM fits the device; once the pools
exceed it, every launch's multi-GB prefetch triggers real eviction+migration —
a PCIe storm that stalled one root step for >24 h at 80+ GB of pools (Jul 2026
N_J=256 ladder; SIGUSR1 profile: main thread ~100% inside cuMemPrefetchAsync
via adapt_storage).  BlockAMR's managed mode does its OWN deterministic
residency (advise_host! + per-level prefetch_device!/prefetch_host! in the
W-cycle), so the per-launch prefetch is pure overhead.  Deliberate method
overwrite (type piracy) — call once before heavy kernel use; costs a one-time
recompile of launch wrappers.
"""
function BlockAMR.disable_kernel_prefetch!()
    @eval function Adapt.adapt_storage(::CUDA.KernelAdaptor,
                                       xs::CUDA.DenseCuArray{T,N}) where {T,N}
        Base.unsafe_convert(CUDA.CuDeviceArray{T,N,CUDA.AS.Global}, xs)
    end
    return nothing
end

function BlockAMR.device_zeros(::CUDABackend, ::Type{T}, dims::Dims) where {T}
    if BlockAMR.memory_mode() === :managed
        a = CuArray{T,length(dims),CUDA.UnifiedMemory}(undef, dims)
        fill!(a, zero(T))
        return a
    end
    return CUDA.zeros(T, dims)
end

# unwrap a unified CuArray to its UnifiedMemory buffer (nothing otherwise)
_unified(a::CuArray{<:Any,<:Any,CUDA.UnifiedMemory}) = a.data[].mem
_unified(::Any) = nothing

# preferred location = host, but device may access it (paged in on touch)
function BlockAMR.advise_host!(a::CuArray)
    um = _unified(a)
    if um !== nothing && length(a) > 0
        CUDA.advise(um, CUDA.CU_MEM_ADVISE_SET_PREFERRED_LOCATION;
                    device = CUDA.DEVICE_CPU)          # host-resident by default
        CUDA.advise(um, CUDA.CU_MEM_ADVISE_SET_ACCESSED_BY)  # device maps it (no fault)
    end
    return a
end

# pull the whole pool onto the GPU now (hot working set)
function BlockAMR.prefetch_device!(a::CuArray)
    um = _unified(a)
    um !== nothing && length(a) > 0 && CUDA.prefetch(um)
    return a
end

# push the whole pool BACK to host RAM now (explicit eviction) — frees device
# memory for the working level.  Deterministic movement, not a driver-paging hint.
function BlockAMR.prefetch_host!(a::CuArray)
    um = _unified(a)
    um !== nothing && length(a) > 0 && CUDA.prefetch(um; device = CUDA.DEVICE_CPU)
    return a
end


# ── CUDA Graph replay of root steps (BlockAMR.advance_hierarchy_graph!) ───────
mutable struct _GraphEntry
    exec::Any                       # CuGraphExec
    swapped::Vector{Bool}           # levels whose R/O roles flip per root step
    refs::Vector{Any}               # strong refs: no captured buffer is freed/reused
end
const _GRAPHS = WeakKeyDict{Any,Dict{Tuple{Int,UInt},_GraphEntry}}()
const _SEEN = WeakKeyDict{Any,Set{Tuple{Int,UInt}}}()
const _GSTAT = BlockAMR._GSTAT       # counters live in the core (graph_stats)

function BlockAMR._graph_step!(::CUDABackend, hier::BlockAMR.AMRHierarchy, λq, k, cooling)
    refs = BlockAMR._graph_refs(hier)
    key = (k, BlockAMR._graph_fp(refs))
    cache = get!(() -> Dict{Tuple{Int,UInt},_GraphEntry}(), _GRAPHS, hier)
    seen = get!(() -> Set{Tuple{Int,UInt}}(), _SEEN, hier)
    e = get(cache, key, nothing)
    if e === nothing && !(key in seen)
        # first occurrence of this (rung, topology, parity): an ordinary step — it
        # also allocates any lazily-built scratch, which a capture may not; record
        # only keys that recur (a graph used once is pure cost)
        length(seen) >= 16 && empty!(seen)
        push!(seen, key); _GSTAT[3] += 1
        return BlockAMR.advance_level_w!(hier, 0, λq; φ = nothing, chem = nothing,
                                         selfgrav = nothing, cooling)
    end
    if e === nothing
        before = [lev.D for lev in hier.levels]
        nst = hier.nstep[1]
        g = CUDA.capture() do       # records the launches; nothing executes yet
            BlockAMR.advance_level_w!(hier, 0, λq; φ = nothing, chem = nothing,
                                      selfgrav = nothing, cooling)
        end
        swapped = [hier.levels[i].D !== before[i] for i in eachindex(before)]
        # the host-side bookkeeping ran during capture: undo it, replay below
        for (i, sw) in enumerate(swapped); sw && BlockAMR.swap_buffers!(hier.levels[i]); end
        hier.nstep[1] = nst
        length(cache) >= 4 && empty!(cache)          # topology moved on: drop stale graphs
        e = _GraphEntry(CUDA.instantiate(g), swapped, refs); _GSTAT[2] += 1
        cache[key] = e
    end
    CUDA.launch(e.exec); _GSTAT[1] += 1
    for (i, sw) in enumerate(e.swapped); sw && BlockAMR.swap_buffers!(hier.levels[i]); end
    hier.nstep[1] += 1
    return nothing
end

end # module
