module MHDKernelsMetalExt

using MHDKernels
using Metal

const MG = Metal.MPSGraphs
using Metal.MPSGraphs: MPSGraph, MPSGraphFFTDescriptor, MPSGraphTensor,
    MPSGraphTensorData, placeholderTensor, id, NSDictionary, NSArray, NSNumber,
    NSString

struct _RadiationForwardPlan
    graph::Any
    input::Any
    output::Any
end

struct _RadiationInversePlan
    graph::Any
    input::Any
    output::Any
end

const _RADIATION_FORWARD_CACHE=Dict{NTuple{3,Int},_RadiationForwardPlan}()
const _RADIATION_INVERSE_CACHE=Dict{NTuple{3,Int},_RadiationInversePlan}()
const _RADIATION_FORWARD_BATCH_CACHE=Dict{NTuple{4,Int},_RadiationForwardPlan}()
const _RADIATION_INVERSE_BATCH_CACHE=Dict{NTuple{4,Int},_RadiationInversePlan}()

function _radiation_fft_descriptor(;inverse::Bool,
        scaling=MG.MPSGraphFFTScalingModeNone)
    d=MPSGraphFFTDescriptor(MG.@objc [MPSGraphFFTDescriptor descriptor]::id{MPSGraphFFTDescriptor})
    d.inverse=inverse; d.scalingMode=scaling; d.roundToOddHermitean=false
    d
end

function _radiation_forward_plan(N)
    get!(_RADIATION_FORWARD_CACHE,N) do
        g=MPSGraph(); x=placeholderTensor(g,N,Float32,"radiation_real")
        axes=NSArray(NSNumber.([0,1,2]))
        desc=_radiation_fft_descriptor(inverse=false)
        out=MPSGraphTensor(MG.@objc [g::id{MPSGraph} realToHermiteanFFTWithTensor:x::id{MPSGraphTensor} axes:axes::id{NSArray} descriptor:desc::id{MPSGraphFFTDescriptor} name:"radiation_rfft"::id{NSString}]::id{MPSGraphTensor})
        _RadiationForwardPlan(g,x,out)
    end
end


function _radiation_inverse_plan(H)
    get!(_RADIATION_INVERSE_CACHE,H) do
        g=MPSGraph(); x=placeholderTensor(g,H,ComplexF32,"radiation_hat")
        axes=NSArray(NSNumber.([0,1,2]))
        desc=_radiation_fft_descriptor(inverse=true,scaling=MG.MPSGraphFFTScalingModeSize)
        out=MPSGraphTensor(MG.@objc [g::id{MPSGraph} HermiteanToRealFFTWithTensor:x::id{MPSGraphTensor} axes:axes::id{NSArray} descriptor:desc::id{MPSGraphFFTDescriptor} name:"radiation_irfft"::id{NSString}]::id{MPSGraphTensor})
        _RadiationInversePlan(g,x,out)
    end
end

function _radiation_forward_batch_plan(N)
    get!(_RADIATION_FORWARD_BATCH_CACHE,N) do
        g=MPSGraph(); x=placeholderTensor(g,N,Float32,"radiation_real_batch")
        # MPSGraph tensor dimensions are reversed relative to Julia. Axis 0 is
        # the four-channel batch dimension; axes 1:3 are spatial.
        axes=NSArray(NSNumber.([1,2,3]))
        desc=_radiation_fft_descriptor(inverse=false)
        out=MPSGraphTensor(MG.@objc [g::id{MPSGraph} realToHermiteanFFTWithTensor:x::id{MPSGraphTensor} axes:axes::id{NSArray} descriptor:desc::id{MPSGraphFFTDescriptor} name:"radiation_rfft_batch"::id{NSString}]::id{MPSGraphTensor})
        _RadiationForwardPlan(g,x,out)
    end
end

function _radiation_inverse_batch_plan(H)
    get!(_RADIATION_INVERSE_BATCH_CACHE,H) do
        g=MPSGraph(); x=placeholderTensor(g,H,ComplexF32,"radiation_hat_batch")
        axes=NSArray(NSNumber.([1,2,3]))
        desc=_radiation_fft_descriptor(inverse=true,scaling=MG.MPSGraphFFTScalingModeSize)
        out=MPSGraphTensor(MG.@objc [g::id{MPSGraph} HermiteanToRealFFTWithTensor:x::id{MPSGraphTensor} axes:axes::id{NSArray} descriptor:desc::id{MPSGraphFFTDescriptor} name:"radiation_irfft_batch"::id{NSString}]::id{MPSGraphTensor})
        _RadiationInversePlan(g,x,out)
    end
end

function MHDKernels._radiation_forward_batch!(
        chat::Metal.MtlArray{ComplexF32,4},x::Metal.MtlArray{Float32,4})
    P=_radiation_forward_batch_plan(size(x))
    MG.@autoreleasepool begin
        feeds=Dict{MPSGraphTensor,MPSGraphTensorData}(P.input=>MPSGraphTensorData(x))
        results=Dict{MPSGraphTensor,MPSGraphTensorData}(P.output=>MPSGraphTensorData(chat))
        cb=MG.MPSCommandBuffer(Metal.global_queue(Metal.device()))
        MG.encode!(cb,P.graph,NSDictionary(feeds),NSDictionary(results),MG.nil,
                   MG.MPSGraphExecutionDescriptor())
        MG.commit!(cb); MG.wait_completed(cb)
    end
    chat
end

function MHDKernels._radiation_inverse_batch!(
        x::Metal.MtlArray{Float32,4},chat::Metal.MtlArray{ComplexF32,4})
    P=_radiation_inverse_batch_plan(size(chat))
    MG.@autoreleasepool begin
        feeds=Dict{MPSGraphTensor,MPSGraphTensorData}(P.input=>MPSGraphTensorData(chat))
        results=Dict{MPSGraphTensor,MPSGraphTensorData}(P.output=>MPSGraphTensorData(x))
        cb=MG.MPSCommandBuffer(Metal.global_queue(Metal.device()))
        MG.encode!(cb,P.graph,NSDictionary(feeds),NSDictionary(results),MG.nil,
                   MG.MPSGraphExecutionDescriptor())
        MG.commit!(cb); MG.wait_completed(cb)
    end
    x
end


function MHDKernels._radiation_forward!(chat::Metal.MtlArray{ComplexF32,3},
                                         x::Metal.MtlArray{Float32,3})
    P=_radiation_forward_plan(size(x))
    MG.@autoreleasepool begin
        feeds=Dict{MPSGraphTensor,MPSGraphTensorData}(P.input=>MPSGraphTensorData(x))
        results=Dict{MPSGraphTensor,MPSGraphTensorData}(P.output=>MPSGraphTensorData(chat))
        cb=MG.MPSCommandBuffer(Metal.global_queue(Metal.device()))
        MG.encode!(cb,P.graph,NSDictionary(feeds),NSDictionary(results),MG.nil,
                   MG.MPSGraphExecutionDescriptor())
        MG.commit!(cb); MG.wait_completed(cb)
    end
    chat
end


function MHDKernels._radiation_inverse!(x::Metal.MtlArray{Float32,3},
                                         chat::Metal.MtlArray{ComplexF32,3})
    P=_radiation_inverse_plan(size(chat))
    MG.@autoreleasepool begin
        feeds=Dict{MPSGraphTensor,MPSGraphTensorData}(P.input=>MPSGraphTensorData(chat))
        results=Dict{MPSGraphTensor,MPSGraphTensorData}(P.output=>MPSGraphTensorData(x))
        cb=MG.MPSCommandBuffer(Metal.global_queue(Metal.device()))
        MG.encode!(cb,P.graph,NSDictionary(feeds),NSDictionary(results),MG.nil,
                   MG.MPSGraphExecutionDescriptor())
        MG.commit!(cb); MG.wait_completed(cb)
    end
    x
end

function __init__()
    if Metal.functional()
        MHDKernels.register_backend!(:metal, MetalBackend())
    end
end

MHDKernels.device_zeros(::MetalBackend, ::Type{T}, dims::Dims) where {T} = Metal.zeros(T, dims)

function MHDKernels._cube_launch!(::Val, be::MetalBackend, s::MHDKernels.MHDState{T}, dt::Real, ch::Real, decay::Real) where {T}
    N = s.dims[1]; nb = N ÷ MHDKernels.CTB; dtdx = T(dt) / s.dx; cx, cy, cz = MHDKernels.bc_codes(s)
    rec = Val(MHDKernels.recon_code_of(s)); per = Val(MHDKernels.all_periodic(s)); rsol = Val(MHDKernels.riemann_code_of(s))
    gs = MHDKernels.METAL_CUBE_GS
    if N % MHDKernels.RCTBX == 0
        nbx = N ÷ MHDKernels.RCTBX
        nby = N ÷ MHDKernels.RCTBY
        nbz = N ÷ MHDKernels.RCTBZ
        MHDKernels.step_cube_rect_kernel!(be, gs)(
            s.scratch..., s.U..., N, nbx, nby, cx, cy, cz,
            Val(Float32), rec, per, rsol, dtdx, s.γ, T(ch), T(decay),
            s.smallr, s.pfl, s.llf_dmin, s.llf_pmin, Val(false),
            MHDKernels._no_drag_coefficients(T),Val(false),Val(false),Val(false),
            Val(false);
            ndrange = nbx * nby * nbz * gs)
    else
        MHDKernels.step_cube_kernel!(be, gs)(
            s.scratch..., s.U..., N, nb, cx, cy, cz, Val(Float32), rec, per, rsol,
            dtdx, s.γ, T(ch), T(decay), s.smallr, s.pfl, s.llf_dmin, s.llf_pmin,
            Val(false),MHDKernels._no_drag_coefficients(T),Val(false),Val(false),
            Val(false),Val(false);
            ndrange = nb * nb * nb * gs)
    end
    MHDKernels.KA.synchronize(be)
end

function MHDKernels._cube_drag_launch!(be::MetalBackend,
        s::MHDKernels.MHDState{T}, dt::Real, ch::Real, decay::Real,
        q::T) where {T}
    N = s.dims[1]; nb = N ÷ MHDKernels.CTB; dtdx = T(dt) / s.dx
    cx, cy, cz = MHDKernels.bc_codes(s)
    rec = Val(MHDKernels.recon_code_of(s)); per = Val(MHDKernels.all_periodic(s))
    rsol = Val(MHDKernels.riemann_code_of(s))
    dc = MHDKernels._drag_coefficients(q)
    gs = MHDKernels.METAL_CUBE_GS
    if N % MHDKernels.RCTBX == 0
        nbx = N ÷ MHDKernels.RCTBX
        nby = N ÷ MHDKernels.RCTBY
        nbz = N ÷ MHDKernels.RCTBZ
        MHDKernels.step_cube_rect_kernel!(be, gs)(
            s.scratch..., s.U..., N, nbx, nby, cx, cy, cz,
            Val(Float32), rec, per, rsol, dtdx, s.γ, T(ch), T(decay),
            s.smallr, s.pfl, s.llf_dmin, s.llf_pmin, Val(true),
            dc,Val(false),Val(false),Val(false),Val(false);
            ndrange = nbx * nby * nbz * gs)
    else
        MHDKernels.step_cube_kernel!(be, gs)(
            s.scratch..., s.U..., N, nb, cx, cy, cz, Val(Float32), rec, per, rsol,
            dtdx, s.γ, T(ch), T(decay), s.smallr, s.pfl, s.llf_dmin, s.llf_pmin,
            Val(true),dc,Val(false),Val(false),Val(false),Val(false);
            ndrange = nb * nb * nb * gs)
    end
    MHDKernels.KA.synchronize(be)
end


function MHDKernels._cube_radiation_launch!(be::MetalBackend,
        s::MHDKernels.MHDState{T},dt::Real,ch::Real,decay::Real,q::T,
        correction,track_density::Val,track_magnetic::Val,check::Val,
        fallback::Val{FALLBACK},global_fallback::Val{GLOBAL},
        robust_fallback::Val{ROBUST}) where {T,FALLBACK,GLOBAL,ROBUST}
    N=s.dims[1]; dtdx=T(dt)/s.dx; cx,cy,cz=MHDKernels.bc_codes(s)
    rec=ROBUST ? Val(MHDKernels.RECON_PLM_MINMOD) :
        Val(MHDKernels.recon_code_of(s))
    per=Val(MHDKernels.all_periodic(s))
    rsol=ROBUST ? Val(MHDKernels.RSOLVE_HLL) :
        GLOBAL ? Val(MHDKernels.RSOLVE_LLF) :
        Val(MHDKernels.riemann_code_of(s))
    dc=MHDKernels._drag_coefficients(q)
    kernel_check=GLOBAL ? Val(3) : FALLBACK ? Val(2) : check
    gs=MHDKernels.METAL_CUBE_GS
    if N % MHDKernels.RCTBX == 0
        nbx=N÷MHDKernels.RCTBX; nby=N÷MHDKernels.RCTBY; nbz=N÷MHDKernels.RCTBZ
        MHDKernels.step_cube_rect_kernel!(be,gs)(s.scratch...,s.U[1],s.U[2],
            s.U[3],s.U[4],s.U[5],s.U[6],s.U[7],s.U[8],correction,
            N,nbx,nby,cx,cy,cz,Val(Float32),rec,per,rsol,dtdx,s.γ,T(ch),T(decay),s.smallr,
            s.pfl,s.llf_dmin,s.llf_pmin,Val(true),dc,Val(true),track_density,
            track_magnetic,kernel_check;
            ndrange=nbx*nby*nbz*gs)
    else
        nb=N÷MHDKernels.CTB
        MHDKernels.step_cube_kernel!(be,gs)(s.scratch...,s.U[1],s.U[2],s.U[3],
            s.U[4],s.U[5],s.U[6],s.U[7],s.U[8],correction,N,nb,cx,cy,cz,
            Val(Float32),rec,per,rsol,dtdx,s.γ,T(ch),T(decay),s.smallr,s.pfl,
            s.llf_dmin,s.llf_pmin,Val(true),dc,Val(true),track_density,
            track_magnetic,kernel_check;
            ndrange=nb*nb*nb*gs)
    end
    MHDKernels.KA.synchronize(be)
end

end # module
