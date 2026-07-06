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

BlockAMR.device_zeros(::CUDABackend, ::Type{T}, dims::Dims) where {T} =
    CUDA.zeros(T, dims)

end # module
