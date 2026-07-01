"""
    PoissonKernelsMetalExt

Package extension that lights up the Metal (Apple GPU) backend for `PoissonKernels`.
Loaded automatically when `Metal` is present in the environment. Registers the
`:metal` backend and specialises the device-array helpers onto `MtlArray`.
Metal is Float32-only, so callers must request `Float32` element types.
"""
module PoissonKernelsMetalExt

using PoissonKernels
using Metal

const MG = Metal.MPSGraphs
using Metal.MPSGraphs: MPSGraph, MPSGraphFFTDescriptor, MPSGraphTensor,
    MPSGraphTensorData, placeholderTensor, id, NSDictionary, NSArray, NSNumber,
    NSString

function __init__()
    # Only register a usable GPU if the system actually has one (CI on Apple
    # hardware without a functional Metal device should degrade gracefully).
    if Metal.functional()
        PoissonKernels.register_backend!(:metal, Metal.MetalBackend())
    end
end

# `Metal.zeros(T, dims)` allocates a zero-filled MtlArray on the default device.
PoissonKernels.device_zeros(::Metal.MetalBackend, ::Type{T}, dims::Dims) where {T} =
    Metal.zeros(T, dims)

struct _MPSRFFTPlan
    graph::Any
    rho::Any
    gk::Any
    coef::Any
    out::Any
    Gk::Any
    coefd::Any
    coefh::Vector{Float32}
end

const _MPS_RFFT_CACHE = Dict{Any,_MPSRFFTPlan}()

struct _MPSForwardRFFTPlan
    graph::Any
    rho::Any
    out::Any
    chat::Any
end

const _MPS_FORWARD_RFFT_CACHE = Dict{Any,_MPSForwardRFFTPlan}()

function _fft_descriptor(; inverse::Bool, scaling = MG.MPSGraphFFTScalingModeNone)
    desc = MPSGraphFFTDescriptor(MG.@objc [MPSGraphFFTDescriptor descriptor]::id{MPSGraphFFTDescriptor})
    desc.inverse = inverse
    desc.scalingMode = scaling
    desc.roundToOddHermitean = false
    return desc
end

function _real_to_hermitean(g, x, axes, desc)
    MPSGraphTensor(MG.@objc [g::id{MPSGraph} realToHermiteanFFTWithTensor:x::id{MPSGraphTensor} axes:axes::id{NSArray} descriptor:desc::id{MPSGraphFFTDescriptor} name:"rfft"::id{NSString}]::id{MPSGraphTensor})
end

function _hermitean_to_real(g, x, axes, desc)
    MPSGraphTensor(MG.@objc [g::id{MPSGraph} HermiteanToRealFFTWithTensor:x::id{MPSGraphTensor} axes:axes::id{NSArray} descriptor:desc::id{MPSGraphFFTDescriptor} name:"irfft"::id{NSString}]::id{MPSGraphTensor})
end

function _real_part(g, x)
    MPSGraphTensor(MG.@objc [g::id{MPSGraph} realPartOfTensor:x::id{MPSGraphTensor} name:"real"::id{NSString}]::id{MPSGraphTensor})
end

function _imag_part(g, x)
    MPSGraphTensor(MG.@objc [g::id{MPSGraph} imaginaryPartOfTensor:x::id{MPSGraphTensor} name:"imag"::id{NSString}]::id{MPSGraphTensor})
end

function _complex_tensor(g, r, i)
    MPSGraphTensor(MG.@objc [g::id{MPSGraph} complexTensorWithRealTensor:r::id{MPSGraphTensor} imaginaryTensor:i::id{MPSGraphTensor} name:"complex"::id{NSString}]::id{MPSGraphTensor})
end

function _mps_rfft_plan(::Type{Float32}, N::NTuple{3,Int},
                        L::NTuple{3,Float64}, greens::Symbol)
    get!(_MPS_RFFT_CACHE, (Float32, N, L, greens)) do
        half = PoissonKernels._rfft_dims(N)
        gh = greens === :spectral ? PoissonKernels._greens_periodic(Float32, N, L) :
             greens === :discrete7 ? PoissonKernels._greens_discrete7(Float32, N, L) :
             error("fft_poisson_rfft!: greens must be :spectral or :discrete7")

        graph = MPSGraph()
        rho = placeholderTensor(graph, N, Float32, "rho")
        gk = placeholderTensor(graph, half, Float32, "gk")
        coef = placeholderTensor(graph, (1,), Float32, "coef")

        axes = NSArray(NSNumber.([0, 1, 2]))
        fdesc = _fft_descriptor(inverse=false)
        idesc = _fft_descriptor(inverse=true, scaling=MG.MPSGraphFFTScalingModeSize)

        rhok = _real_to_hermitean(graph, rho, axes, fdesc)
        gcoef = MG.multiplicationWithPrimaryTensor(graph, gk, coef, "gcoef")
        phr = MG.multiplicationWithPrimaryTensor(graph, _real_part(graph, rhok), gcoef, "mulr")
        phi = MG.multiplicationWithPrimaryTensor(graph, _imag_part(graph, rhok), gcoef, "muli")
        phik = _complex_tensor(graph, phr, phi)
        out = _hermitean_to_real(graph, phik, axes, idesc)

        _MPSRFFTPlan(graph, rho, gk, coef, out, Metal.MtlArray(gh),
                     Metal.MtlArray(Float32[1]), Float32[1])
    end
end

function _mps_rfft_plan(::Type{T}, N::NTuple{3,Int},
                        L::NTuple{3,Float64}, greens::Symbol) where {T}
    error("Metal MPSGraph FFT supports Float32 only, got $T for size $N")
end

function _mps_forward_rfft_plan(::Type{Float32}, N::NTuple{3,Int})
    get!(_MPS_FORWARD_RFFT_CACHE, (Float32, N)) do
        half = PoissonKernels._rfft_dims(N)
        graph = MPSGraph()
        rho = placeholderTensor(graph, N, Float32, "rho")
        axes = NSArray(NSNumber.([0, 1, 2]))
        out = _real_to_hermitean(graph, rho, axes, _fft_descriptor(inverse=false))
        _MPSForwardRFFTPlan(graph, rho, out, Metal.zeros(ComplexF32, half))
    end
end

function _mps_forward_rfft!(x::Metal.MtlArray{Float32,3})
    P = _mps_forward_rfft_plan(Float32, size(x))
    feeds = Dict{MPSGraphTensor,MPSGraphTensorData}(P.rho => MPSGraphTensorData(x))
    results = Dict{MPSGraphTensor,MPSGraphTensorData}(P.out => MPSGraphTensorData(P.chat))
    cmdbuf = MG.MPSCommandBuffer(Metal.global_queue(Metal.device()))
    MG.encode!(cmdbuf, P.graph, NSDictionary(feeds), NSDictionary(results),
               MG.nil, MG.MPSGraphExecutionDescriptor())
    MG.commit!(cmdbuf)
    MG.wait_completed(cmdbuf)
    return P.chat
end

function PoissonKernels.fft_poisson_rfft!(φ::Metal.MtlArray{T,3},
                                          ρ::Metal.MtlArray{T,3};
                                          G::Real=1.0, a::Real=1.0,
                                          boxsize=1.0,
                                          greens::Symbol=:spectral) where {T}
    N = size(ρ)
    size(φ) == N || error("fft_poisson_rfft!: φ size $(size(φ)) != ρ size $N")
    L = boxsize isa Number ? ntuple(_ -> Float64(boxsize), 3) :
        ntuple(d -> Float64(boxsize[d]), 3)
    P = _mps_rfft_plan(T, N, L, greens)
    P.coefh[1] = Float32(G) / Float32(a)
    copyto!(P.coefd, P.coefh)

    feeds = Dict{MPSGraphTensor,MPSGraphTensorData}(
        P.rho => MPSGraphTensorData(ρ),
        P.gk => MPSGraphTensorData(P.Gk),
        P.coef => MPSGraphTensorData(P.coefd),
    )
    results = Dict{MPSGraphTensor,MPSGraphTensorData}(
        P.out => MPSGraphTensorData(φ),
    )
    cmdbuf = MG.MPSCommandBuffer(Metal.global_queue(Metal.device()))
    MG.encode!(cmdbuf, P.graph, NSDictionary(feeds), NSDictionary(results),
               MG.nil, MG.MPSGraphExecutionDescriptor())
    MG.commit!(cmdbuf)
    MG.wait_completed(cmdbuf)
    return φ
end

function _power_spectrum_aniso_mps(fields::Tuple; boxsize=1.0, nmu::Integer=4,
                                   nbins::Integer=0, axis::Integer=1)
    isempty(fields) && error("power_spectrum_aniso_gpu needs at least one field")
    f1 = fields[1]
    N = size(f1)
    all(size(f) == N for f in fields) || error("power_spectrum_aniso_gpu field sizes must match")
    all(f -> f isa Metal.MtlArray{Float32,3}, fields) ||
        error("Metal power_spectrum_aniso_gpu supports Float32 MtlArray fields")
    all(==(N[1]), N) || error("power_spectrum_aniso_gpu needs a cubic grid, got $N")

    n = N[1]
    L = Float64(boxsize isa Number ? boxsize : boxsize[1])
    nb = nbins > 0 ? Int(nbins) : n ÷ 2
    nμ = Int(nmu)
    kf = 2π / L
    kny = n / 2
    invN3 = 1.0 / Float64(n)^3
    V = L^3

    Psum = zeros(Float64, nb, nμ)
    ksum = zeros(Float64, nb)
    cnt = zeros(Int, nb, nμ)
    first_field = true

    for f in fields
        hk = Array(_mps_forward_rfft!(f))
        @inbounds for k in 0:n-1
            kz = k <= n ÷ 2 ? k : k - n
            for j in 0:n-1
                ky = j <= n ÷ 2 ? j : j - n
                for i in 0:n÷2
                    kx = i
                    km = sqrt(Float64(kx*kx + ky*ky + kz*kz))
                    (km <= 0 || km > kny) && continue
                    ka = axis == 1 ? kx : axis == 2 ? ky : kz
                    b = clamp(1 + floor(Int, (km/kny)*nb), 1, nb)
                    mb = clamp(1 + floor(Int, (abs(Float64(ka))/km)*nμ), 1, nμ)
                    mult = (i == 0 || i == n ÷ 2) ? 1 : 2
                    z = hk[i+1, j+1, k+1]
                    Psum[b, mb] += mult * V * (Float64(real(z))^2 + Float64(imag(z))^2) * invN3 * invN3
                    if first_field
                        ksum[b] += mult * km * kf
                        cnt[b, mb] += mult
                    end
                end
            end
        end
        first_field = false
    end

    tot = vec(sum(cnt, dims=2))
    kc = [tot[b] > 0 ? ksum[b]/tot[b] : NaN for b in 1:nb]
    P = [cnt[b,m] > 0 ? Psum[b,m]/cnt[b,m] : NaN for b in 1:nb, m in 1:nμ]
    return (k = kc, P = P, Nmodes = cnt)
end

function PoissonKernels.power_spectrum_aniso_gpu(field::Metal.MtlArray{Float32,3};
                                                 boxsize=1.0, nmu::Integer=4,
                                                 nbins::Integer=0, axis::Integer=1)
    _power_spectrum_aniso_mps((field,); boxsize=boxsize, nmu=nmu, nbins=nbins, axis=axis)
end

function PoissonKernels.power_spectrum_aniso_gpu(fields::Tuple{Vararg{Metal.MtlArray{Float32,3}}};
                                                 boxsize=1.0, nmu::Integer=4,
                                                 nbins::Integer=0, axis::Integer=1)
    _power_spectrum_aniso_mps(fields; boxsize=boxsize, nmu=nmu, nbins=nbins, axis=axis)
end

end # module
