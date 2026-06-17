"""
    EmissionKernelsCUDAExt

Lights up the CUDA (NVIDIA GPU) backend for `EmissionKernels`. Loaded automatically
when `CUDA` is present; registers `:cuda` and specialises the device-array helpers
onto `CuArray`. CUDA supports Float32 and Float64.
"""
module EmissionKernelsCUDAExt

using EmissionKernels
using CUDA

function __init__()
    if CUDA.functional()
        EmissionKernels.register_backend!(:cuda, CUDABackend())
    end
end

EmissionKernels.device_zeros(::CUDABackend, ::Type{T}, dims::Dims) where {T} =
    CUDA.zeros(T, dims)

end # module
