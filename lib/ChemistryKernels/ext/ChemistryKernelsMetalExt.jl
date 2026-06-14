"""
    ChemistryKernelsMetalExt

Package extension that lights up the Metal (Apple GPU) backend for
`ChemistryKernels`. Loaded automatically when `Metal` is present. Registers the
`:metal` backend and specialises the device-array helpers onto `MtlArray`.
Metal is Float32-only, so callers must request `Float32` element types.
"""
module ChemistryKernelsMetalExt

using ChemistryKernels
using Metal

function __init__()
    if Metal.functional()
        ChemistryKernels.register_backend!(:metal, Metal.MetalBackend())
    end
end

ChemistryKernels.device_zeros(::Metal.MetalBackend, ::Type{T}, dims::Dims) where {T} =
    Metal.zeros(T, dims)

end # module
