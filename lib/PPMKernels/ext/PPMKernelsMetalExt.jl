"""
    PPMKernelsMetalExt

Package extension that lights up the Metal (Apple GPU) backend for `PPMKernels`.
Loaded automatically when `Metal` is present in the environment. Registers the
`:metal` backend and specialises the device-array helpers onto `MtlArray`.
Metal is Float32-only, so callers must request `Float32` element types.
"""
module PPMKernelsMetalExt

using PPMKernels
using Metal

function __init__()
    # Only register a usable GPU if the system actually has one (CI on Apple
    # hardware without a functional Metal device should degrade gracefully).
    if Metal.functional()
        PPMKernels.register_backend!(:metal, Metal.MetalBackend())
    end
end

# `Metal.zeros(T, dims)` allocates a zero-filled MtlArray on the default device.
PPMKernels.device_zeros(::Metal.MetalBackend, ::Type{T}, dims::Dims) where {T} =
    Metal.zeros(T, dims)

end # module
