"""
    EmissionKernelsMetalExt

Lights up the Metal (Apple GPU) backend for `EmissionKernels`. Loaded automatically
when `Metal` is present; registers `:metal` and specialises the device-array helpers
onto `MtlArray`. Metal is Float32-only.
"""
module EmissionKernelsMetalExt

using EmissionKernels
using Metal

function __init__()
    if Metal.functional()
        EmissionKernels.register_backend!(:metal, Metal.MetalBackend())
    end
end

EmissionKernels.device_zeros(::Metal.MetalBackend, ::Type{T}, dims::Dims) where {T} =
    Metal.zeros(T, dims)

end # module
