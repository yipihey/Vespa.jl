"""
    PoissonKernelsCUDAExt

Package extension that lights up the CUDA (NVIDIA GPU) backend for
`PoissonKernels`. Loaded automatically when `CUDA` is present in the
environment. Registers the `:cuda` backend and specialises the device-array
helpers onto `CuArray`. CUDA supports both Float32 and Float64.
"""
module PoissonKernelsCUDAExt

using PoissonKernels
using CUDA

function __init__()
    if CUDA.functional()
        PoissonKernels.register_backend!(:cuda, CUDABackend())
    end
end

PoissonKernels.device_zeros(::CUDABackend, ::Type{T}, dims::Dims) where {T} =
    CUDA.zeros(T, dims)

end # module
