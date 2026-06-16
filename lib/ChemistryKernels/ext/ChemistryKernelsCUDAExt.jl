"""
    ChemistryKernelsCUDAExt

Package extension that lights up the CUDA (NVIDIA GPU) backend for
`ChemistryKernels`. Loaded automatically when `CUDA` is present in the
environment. Registers the `:cuda` backend and specialises the device-array
helpers onto `CuArray`. CUDA supports both Float32 and Float64.
"""
module ChemistryKernelsCUDAExt

using ChemistryKernels
using CUDA

function __init__()
    if CUDA.functional()
        ChemistryKernels.register_backend!(:cuda, CUDABackend())
    end
end

ChemistryKernels.device_zeros(::CUDABackend, ::Type{T}, dims::Dims) where {T} =
    CUDA.zeros(T, dims)

end # module
