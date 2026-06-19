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

# IN-PLACE bit-reversal along axis 1: swap element i with r=bitrev(i), guarded by
# i<r so each pair is touched exactly once (bit-reversal is an involution ⇒ the
# partner workitem's guard is false ⇒ no concurrent write, race-free on CPU & GPU).
# Avoids the out-of-place scratch write + full copy-back (≈3 array passes/axis).
@kernel function _bitrev_inplace_k!(xr, xi, @Const(br))
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        r = br[i] + 1
        if i < r
            t = xr[i, j, k]; xr[i, j, k] = xr[r, j, k]; xr[r, j, k] = t
            t = xi[i, j, k]; xi[i, j, k] = xi[r, j, k]; xi[r, j, k] = t
        end
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

# in-place 1-D FFT along axis 1 of a 3-D (re,im) pair (no scratch — in-place bitrev).
function _fft_axis1!(xr, xi, brdev, sgn::Int)
    be = KA.get_backend(xr); T = eltype(xr); N1, N2, N3 = size(xr)
    _bitrev_inplace_k!(be)(xr, xi, brdev; ndrange = (N1, N2, N3))
    twopi = T(2) * T(pi)
    s = 1
    while (1 << s) <= N1
        m = 1 << s; half = m >> 1
        _fft_stage_k!(be)(xr, xi, half, m, T(sgn), twopi; ndrange = (N1 ÷ 2, N2, N3))
        s += 1
    end
    return xr, xi
end

# Per-(backend,T,N) cached transpose targets tr,ti (the bit-reversal is now in-place,
# so no sr,si scratch).  Reused across all three axes AND across calls, so a 3-D FFT
# allocates NOTHING after the first.  Cubic grid ⇒ every buffer shares one N³ shape.
const _FFT_SCRATCH = Dict{Any,Any}()
_fft_scratch(be, ::Type{T}, N) where {T} =
    get!(_FFT_SCRATCH, (typeof(be), T, N)) do
        (KA.zeros(be, T, N...), KA.zeros(be, T, N...))
    end

# ── axis transposes ───────────────────────────────────────────────────────────
# The 3-D FFT brings each axis to the front (axis 1) before the 1-D transform. On
# the GPU `permutedims!` is already a tuned device kernel; on the CPU it is Base's
# SERIAL transpose — the only non-KA, non-threaded step in the host FFT. So on the
# CPU KA backend we transpose with a threaded `@kernel` (one workitem per element,
# split over `Threads.nthreads()`), keeping the whole host FFT "purely KA + host
# threads"; on every other backend we keep the optimized `permutedims!`.
@kernel function _transpose12_k!(dst, @Const(src))      # perm (2,1,3): swap axes 1,2
    i, j, k = @index(Global, NTuple)
    @inbounds dst[i, j, k] = src[j, i, k]
end
@kernel function _transpose13_k!(dst, @Const(src))      # perm (3,2,1): swap axes 1,3
    i, j, k = @index(Global, NTuple)
    @inbounds dst[i, j, k] = src[k, j, i]
end

# out-of-place transpose dst ← permutedims(src, perm); KA-threaded on CPU, permutedims! else
@inline function _transpose!(dst, src, perm::NTuple{3,Int}, be)
    if be isa KA.CPU
        if perm === (2, 1, 3)
            _transpose12_k!(be)(dst, src; ndrange = size(dst))
        else
            _transpose13_k!(be)(dst, src; ndrange = size(dst))
        end
    else
        permutedims!(dst, src, perm)
    end
    return dst
end

# 3-D FFT (sgn=-1 fwd, +1 inv) — transform axis 1, then bring axes 2,3 to the front via
# out-of-place transposes into cached buffers (no per-call allocation).  On the CPU
# the transposes are threaded KA kernels (see `_transpose!`); on the GPU, permutedims!.
function _fft3d!(xr, xi, brdev, sgn::Int)
    be = KA.get_backend(xr); T = eltype(xr); N = size(xr)
    tr, ti = _fft_scratch(be, T, N)
    _fft_axis1!(xr, xi, brdev, sgn)                               # axis 1
    _transpose!(tr, xr, (2, 1, 3), be); _transpose!(ti, xi, (2, 1, 3), be)
    _fft_axis1!(tr, ti, brdev, sgn)                               # axis 2
    _transpose!(xr, tr, (2, 1, 3), be); _transpose!(xi, ti, (2, 1, 3), be)
    _transpose!(tr, xr, (3, 2, 1), be); _transpose!(ti, xi, (3, 2, 1), be)
    _fft_axis1!(tr, ti, brdev, sgn)                               # axis 3
    _transpose!(xr, tr, (3, 2, 1), be); _transpose!(xi, ti, (3, 2, 1), be)
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

"""
    power_spectrum_gpu(delta; boxsize=1.0, nbins=0) -> (k, P, Nmodes)

Isotropic power spectrum P(k) of a real overdensity field `delta` (a cubic,
power-of-two device array), computed with the GPU radix-2 FFT.  Convention: the
mean-normalized transform δ̂(k) = (1/N³) Σ_x δ(x) e^{-ik·x} and
P(k) = V ⟨|δ̂(k)|²⟩ with V = boxsize³ — so a white-noise field of variance σ²
returns P = σ²·V/N³ (the shot/grid floor).  Modes are radially binned in |k| in
units of the fundamental k_f = 2π/boxsize; `nbins` defaults to N/2 linear bins
out to the Nyquist k_f·N/2.  Returns bin-centre wavenumbers `k` (physical, =
2π/length), the binned power `P`, and the mode count `Nmodes` per bin (bins with
no modes are dropped).  The FFT runs on `delta`'s backend; only the (small)
|δ̂|² radial binning is on the host.
"""
function power_spectrum_gpu(delta::AbstractArray{T,3}; boxsize = 1.0, nbins::Integer = 0) where {T}
    be = KA.get_backend(delta); N = size(delta)
    all(_ispow2, N) || error("power_spectrum_gpu needs power-of-two dims, got $N")
    all(==(N[1]), N) || error("power_spectrum_gpu currently needs a cubic grid, got $N")
    n = N[1]
    L = boxsize isa Number ? Float64(boxsize) : Float64(boxsize[1])
    brdev = get!(_GPUFFT_CACHE, (:brev, typeof(be), T, n)) do
        to_device(be, _bitrev_table(n), Int32)
    end
    xr = copyto!(similar(delta), delta); xi = KA.zeros(be, T, N...)
    xr, xi = _fft3d!(xr, xi, brdev, -1)                # forward DFT (unnormalized)
    KA.synchronize(be)
    ar = Array(xr); ai = Array(xi)                     # bring |δ̂|² home for binning
    invN3 = 1.0 / Float64(n)^3
    V = L^3
    nb = nbins > 0 ? Int(nbins) : n ÷ 2
    kf = 2π / L                                         # fundamental wavenumber
    kny_modes = n / 2                                  # Nyquist in fundamental units
    Psum = zeros(Float64, nb); ksum = zeros(Float64, nb); cnt = zeros(Int, nb)
    @inbounds for k in 0:n-1
        kz = k <= n ÷ 2 ? k : k - n
        for j in 0:n-1
            ky = j <= n ÷ 2 ? j : j - n
            for i in 0:n-1
                kx = i <= n ÷ 2 ? i : i - n
                km = sqrt(Float64(kx*kx + ky*ky + kz*kz))   # |k| in fundamental units
                (km <= 0 || km > kny_modes) && continue
                b = clamp(1 + floor(Int, (km / kny_modes) * nb), 1, nb)
                re = Float64(ar[i+1, j+1, k+1]) * invN3
                im = Float64(ai[i+1, j+1, k+1]) * invN3
                Psum[b] += V * (re*re + im*im)
                ksum[b] += km * kf
                cnt[b]  += 1
            end
        end
    end
    keep = cnt .> 0
    kc = (ksum[keep] ./ cnt[keep])
    Pc = (Psum[keep] ./ cnt[keep])
    return (k = kc, P = Pc, Nmodes = cnt[keep])
end
