"""
    BlockAMRCUDAExt

Package extension lighting up the CUDA backend for `BlockAMR`: registers `:cuda`
and specialises `device_zeros` onto `CuArray`. Loaded automatically when `CUDA`
is present in the environment.
"""
module BlockAMRCUDAExt

using BlockAMR
using CUDA

function __init__()
    if CUDA.functional()
        BlockAMR.register_backend!(:cuda, CUDABackend())
        BlockAMR.enable_tiling!(:cuda)     # ~42 KB shared mem fits NVIDIA (48+ KB)
    end
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

end # module
