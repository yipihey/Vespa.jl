module MHDKernelsMetalExt

using MHDKernels
using Metal

function __init__()
    if Metal.functional()
        MHDKernels.register_backend!(:metal, MetalBackend())
    end
end

MHDKernels.device_zeros(::MetalBackend, ::Type{T}, dims::Dims) where {T} = Metal.zeros(T, dims)

function MHDKernels._cube_launch!(::Val, be::MetalBackend, s::MHDKernels.MHDState{T}, dt::Real, ch::Real, decay::Real) where {T}
    N = s.dims[1]; nb = N ÷ MHDKernels.CTB; dtdx = T(dt) / s.dx; cx, cy, cz = MHDKernels.bc_codes(s)
    rec = Val(MHDKernels.recon_code_of(s)); per = Val(MHDKernels.all_periodic(s)); rsol = Val(MHDKernels.riemann_code_of(s))
    MHDKernels.step_cube_kernel!(be, MHDKernels.CUBE_GS)(
        s.scratch..., s.U..., N, nb, cx, cy, cz, Val(Float16), rec, per, rsol, dtdx, s.γ, T(ch), T(decay),
        s.smallr, s.pfl, s.llf_dmin, s.llf_pmin; ndrange = nb * nb * nb * MHDKernels.CUBE_GS)
    MHDKernels.KA.synchronize(be)
end

end # module
