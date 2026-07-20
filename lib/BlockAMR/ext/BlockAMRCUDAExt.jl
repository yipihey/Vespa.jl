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

end # module
