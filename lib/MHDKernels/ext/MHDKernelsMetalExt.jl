module MHDKernelsMetalExt

using MHDKernels
using Metal

function __init__()
    if Metal.functional()
        MHDKernels.register_backend!(:metal, MetalBackend())
    end
end

MHDKernels.device_zeros(::MetalBackend, ::Type{T}, dims::Dims) where {T} = Metal.zeros(T, dims)

end # module
