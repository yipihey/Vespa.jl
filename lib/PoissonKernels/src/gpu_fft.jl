# GPU-resident radix-2 FFT + periodic Poisson root solve (the last CPU step in the
# gravity path — FFTW — moved onto the device).  Real/imag are kept as SEPARATE
# Float32/Float64 arrays (no Complex bitstype in the Metal kernels, for safety).
#
# 1-D Cooley–Tukey, decimation-in-time: bit-reversal permutation, then log2(N)
# butterfly stages (one kernel launch per stage; butterflies within a stage are
# independent so they're race-free, and KA queue-orders the stages).  3-D = the
# 1-D transform along axis 1, then permutedims to bring axes 2 and 3 to the front.
# Only powers of two (the SB root grids are 128³/256³).  CPU `fft_poisson_root!`
# stays the oracle; this is validated bit-for-(f32)-bit against it.

using KernelAbstractions: @kernel, @index, @Const

_ispow2(n) = n > 0 && (n & (n - 1)) == 0
_log2i(n) = (m = 0; while (1 << m) < n; m += 1; end; m)

# bit-reversal index table (0-based) for length N = 2^bits
function _bitrev_table(N::Int)
    bits = _log2i(N); br = Vector{Int32}(undef, N)
    @inbounds for i in 0:N-1
        r = 0; x = i
        for _ in 1:bits; r = (r << 1) | (x & 1); x >>= 1; end
        br[i+1] = r
    end
    br
end

@kernel function _bitrev_k!(yr, yi, @Const(xr), @Const(xi), @Const(br))
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        s = br[i] + 1
        yr[i, j, k] = xr[s, j, k]; yi[i, j, k] = xi[s, j, k]
    end
end

# one butterfly stage along axis 1; half = m/2, m = 2^s; sgn=-1 forward, +1 inverse
@kernel function _fft_stage_k!(xr, xi, half::Int, m::Int, sgn, twopi)
    p, j, k = @index(Global, NTuple)          # p ∈ 1:N1/2 butterfly index
    @inbounds begin
        pp = p - 1; g = pp ÷ half; kk = pp % half
        i0 = g * m + kk + 1; i1 = i0 + half
        T = eltype(xr)
        th = sgn * twopi * T(kk) / T(m)
        wr = cos(th); wi = sin(th)
        ar = xr[i0, j, k]; ai = xi[i0, j, k]
        br_ = xr[i1, j, k]; bi = xi[i1, j, k]
        tr = wr * br_ - wi * bi; ti = wr * bi + wi * br_
        xr[i0, j, k] = ar + tr; xi[i0, j, k] = ai + ti
        xr[i1, j, k] = ar - tr; xi[i1, j, k] = ai - ti
    end
end

# in-place 1-D FFT along axis 1 of a 3-D (re,im) pair; scratch sr,si same shape.
function _fft_axis1!(xr, xi, sr, si, brdev, sgn::Int)
    be = KA.get_backend(xr); T = eltype(xr); N1, N2, N3 = size(xr)
    _bitrev_k!(be)(sr, si, xr, xi, brdev; ndrange = (N1, N2, N3))
    copyto!(xr, sr); copyto!(xi, si)
    twopi = T(2) * T(pi)
    s = 1
    while (1 << s) <= N1
        m = 1 << s; half = m >> 1
        _fft_stage_k!(be)(xr, xi, half, m, T(sgn), twopi; ndrange = (N1 ÷ 2, N2, N3))
        s += 1
    end
    return xr, xi
end

# 3-D FFT (sgn=-1 fwd, +1 inv) — axis 1, then permute axes 2,3 to the front.
function _fft3d!(xr, xi, brdev, sgn::Int)
    sr = similar(xr); si = similar(xi)
    _fft_axis1!(xr, xi, sr, si, brdev, sgn)
    xr = permutedims(xr, (2, 1, 3)); xi = permutedims(xi, (2, 1, 3))
    sr = similar(xr); si = similar(xi); _fft_axis1!(xr, xi, sr, si, brdev, sgn)
    xr = permutedims(xr, (2, 1, 3)); xi = permutedims(xi, (2, 1, 3))
    xr = permutedims(xr, (3, 2, 1)); xi = permutedims(xi, (3, 2, 1))
    sr = similar(xr); si = similar(xi); _fft_axis1!(xr, xi, sr, si, brdev, sgn)
    xr = permutedims(xr, (3, 2, 1)); xi = permutedims(xi, (3, 2, 1))
    return xr, xi
end

# full (c2c) periodic Green's function G(k) = -1/k², G(0)=0, on the device.
function _greens_full(be, ::Type{T}, N::NTuple{3,Int}, L::NTuple{3,Float64}) where {T}
    Gk = Array{T,3}(undef, N)
    w = ntuple(d -> T(2π) / T(L[d]), 3)
    @inbounds for k in 0:N[3]-1
        kz = w[3] * T(k <= N[3] ÷ 2 ? k : k - N[3])
        for j in 0:N[2]-1
            ky = w[2] * T(j <= N[2] ÷ 2 ? j : j - N[2])
            for i in 0:N[1]-1
                kx = w[1] * T(i <= N[1] ÷ 2 ? i : i - N[1])
                ks = kx*kx + ky*ky + kz*kz
                Gk[i+1, j+1, k+1] = ks == zero(T) ? zero(T) : -one(T) / ks
            end
        end
    end
    to_device(be, Gk, T)
end

const _GPUFFT_CACHE = Dict{Any,Any}()

@kernel function _scale_by_k!(xr, xi, @Const(Gk), coef)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        c = coef * Gk[i, j, k]
        xr[i, j, k] *= c; xi[i, j, k] *= c
    end
end

"""
    fft_poisson_root_gpu!(phi, rho; G=1.0, a=1.0, boxsize=1.0) -> phi

GPU-resident periodic Poisson solve (the device counterpart of `fft_poisson_root!`):
forward radix-2 FFT of `rho` → multiply by `coef·G(k)` → inverse FFT → real part,
all on `rho`'s backend. Dimensions must be powers of two. Bit-identical (to f32) with
the FFTW path. `phi`, `rho` are 3-D device arrays of the same shape.
"""
function fft_poisson_root_gpu!(phi::AbstractArray{T,3}, rho::AbstractArray{T,3};
                               G::Real = 1.0, a::Real = 1.0, boxsize = 1.0) where {T}
    be = KA.get_backend(rho); N = size(rho)
    all(_ispow2, N) || error("fft_poisson_root_gpu! needs power-of-two dims, got $N")
    all(==(N[1]), N) || error("fft_poisson_root_gpu! currently needs a cubic grid (one bit-reversal table), got $N")
    L = boxsize isa Number ? ntuple(_ -> Float64(boxsize), 3) : ntuple(d -> Float64(boxsize[d]), 3)
    ckey = (typeof(be), T, N, L)
    brdev, Gk = get!(_GPUFFT_CACHE, ckey) do
        (to_device(be, _bitrev_table(N[1]), Int32), _greens_full(be, T, N, L))
    end
    xr = copyto!(similar(rho), rho); xi = KA.zeros(be, T, N...)
    xr, xi = _fft3d!(xr, xi, brdev, -1)               # forward
    _scale_by_k!(be)(xr, xi, Gk, T(G) / T(a); ndrange = N)
    xr, xi = _fft3d!(xr, xi, brdev, +1)               # inverse (unnormalized)
    invN = one(T) / T(prod(N))
    @inbounds phi .= xr .* invN                        # real part, normalized
    KA.synchronize(be)
    return phi
end
