module PPMKernelsCUDAExt

using PPMKernels
using CUDA

function __init__()
    if CUDA.functional()
        PPMKernels.register_backend!(:cuda, CUDABackend())
    end
end

PPMKernels.device_zeros(::CUDABackend, ::Type{T}, dims::Dims) where {T} =
    CUDA.zeros(T, dims)

end # module
