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
# Cached imaginary buffer, reused (zeroed) every solve — so the hot gravity FFT allocates NOTHING
# per call (the fresh `KA.zeros` xi otherwise collides with the pooled ρpi transient at the ceiling).
_fft_imag(be, ::Type{T}, N) where {T} =
    get!(() -> KA.zeros(be, T, N...), _FFT_SCRATCH, (:imag, typeof(be), T, N))

"""
    poisson_scratch_i32(be, T, N) -> Int32 view of an N³ FFT scratch buffer (reinterpreted)

The Stockham FFT's real scratch `sr` (Float32, N³) reinterpreted as an Int32 array of the same shape,
for callers that need a large Int32 accumulator BEFORE the FFT runs (e.g. the deterministic CIC
deposit).  Reusing this buffer avoids a separate N³·4 B allocation at the grid ceiling.  The FFT
overwrites `sr` on its first stage, so the borrow must be finished before `fft_poisson_root_gpu!`.
"""
poisson_scratch_i32(be, ::Type{T}, N) where {T} = reinterpret(Int32, _fft_scratch(be, T, N)[1])

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

# ── Mixed-radix (2·3·5) Stockham autosort — non-power-of-two cubic grids ─────────
# The radix-2 path above is in-place (bit-reversal); mixed radix uses a self-sorting
# Stockham (no digit-reversal) that ping-pongs between two buffer pairs.  This is what
# lets the gravity FFT run at 2ᵃ3ᵇ5ᶜ sizes (768/864/900/960/1024) with a SMALL, poolable
# scratch instead of cuFFT's opaque ~18 B/cell work area.  Validated bit-close vs FFTW.
_factorize235(n::Int) = begin
    fs = Int[]; m = n
    for p in (2, 3, 5); while m % p == 0; push!(fs, p); m ÷= p; end; end
    m == 1 ? fs : nothing                              # nothing ⇒ has a prime factor > 5
end

# One Stockham stage along axis 1.  The twiddle is FUSED into the radix-R DFT (combined phase
# r·((jz%Ns)+kk·Ns)/(Ns·R)) so each output reads the R inputs directly — no per-thread local
# array (keeps it backend-agnostic; R ≤ 5 ⇒ ≤25 reads/butterfly).  Self-sorting scatter (idxD).
@kernel function _stockham_stage_k!(outr, outi, @Const(inr), @Const(ini),
                                    R::Int, Ns::Int, NR::Int, sgn, twopi)
    j, a2, a3 = @index(Global, NTuple)
    @inbounds begin
        T = eltype(outr); jz = j - 1
        base = jz % Ns; idxD = (jz ÷ Ns) * Ns * R + base
        for kk in 0:R-1
            ph0 = sgn * twopi * T(base + kk*Ns) / T(Ns*R)
            sr = zero(T); si = zero(T)
            for r in 0:R-1
                xr = inr[jz + r*NR + 1, a2, a3]; xi = ini[jz + r*NR + 1, a2, a3]
                c = cos(T(r)*ph0); s = sin(T(r)*ph0)
                sr += c*xr - s*xi;  si += c*xi + s*xr
            end
            outr[idxD + kk*Ns + 1, a2, a3] = sr;  outi[idxD + kk*Ns + 1, a2, a3] = si
        end
    end
end

# 1-D Stockham along axis 1 (ping-pong cr↔or); result left in (cr,ci), (or,oi) free as scratch.
function _stockham_axis1!(cr, ci, or, oi, factors, sgn::Int)
    be = KA.get_backend(cr); T = eltype(cr); N1, N2, N3 = size(cr); tp = T(2) * T(pi)
    ar, ai, brr, bii = cr, ci, or, oi; Ns = 1
    for R in factors
        _stockham_stage_k!(be)(brr, bii, ar, ai, R, Ns, N1 ÷ R, T(sgn), tp; ndrange = (N1 ÷ R, N2, N3))
        ar, brr = brr, ar; ai, bii = bii, ai; Ns *= R
    end
    ar === cr || (copyto!(cr, ar); copyto!(ci, ai))    # odd #stages ⇒ result in scratch: copy back
    return nothing
end

# 3-D mixed-radix FFT with ONLY two buffer pairs (xr,xi)+(sr,si): each axis Stockham leaves its
# result in the first pair; the out-of-place transpose ping-pongs into the other — no 3rd buffer.
function _fft3d_mixed!(xr, xi, sr, si, factors, sgn::Int)
    be = KA.get_backend(xr)
    _stockham_axis1!(xr, xi, sr, si, factors, sgn)                       # axis 1 → (xr,xi)
    _transpose!(sr, xr, (2,1,3), be); _transpose!(si, xi, (2,1,3), be)   # (sr,si) ← transpose
    _stockham_axis1!(sr, si, xr, xi, factors, sgn)                       # axis 2 → (sr,si)
    _transpose!(xr, sr, (2,1,3), be); _transpose!(xi, si, (2,1,3), be)   # back → (xr,xi)
    _transpose!(sr, xr, (3,2,1), be); _transpose!(si, xi, (3,2,1), be)   # (sr,si) ← axis1↔3
    _stockham_axis1!(sr, si, xr, xi, factors, sgn)                       # axis 3 → (sr,si)
    _transpose!(xr, sr, (3,2,1), be); _transpose!(xi, si, (3,2,1), be)   # back → (xr,xi)
    return xr, xi
end

# ── Real-input FFT (rfft) Poisson — ~½ the FLOPs, memory, and transpose volume of the c2c path ──
# rfft along axis-1: pack the n reals into M=n/2 complex, c2c(M), Hermitian split → the (M+1,n,n)
# half-complex grid; then c2c along axes 2,3 on the half-grid, Green's, and the inverse.  Needs an
# EVEN 2·3·5-smooth n (with M=n/2 also 2·3·5-smooth).  Validated bit-close vs the c2c solver / FFTW.
@kernel function _rpack_k!(zr, zi, @Const(x))                 # (n,n,n) real → (M,n,n) complex
    m, j, k = @index(Global, NTuple)
    @inbounds begin zr[m,j,k] = x[2m-1,j,k]; zi[m,j,k] = x[2m,j,k] end
end
@kernel function _runpack_k!(x, @Const(zr), @Const(zi))       # (M,n,n) complex → (n,n,n) real
    m, j, k = @index(Global, NTuple)
    @inbounds begin x[2m-1,j,k] = zr[m,j,k]; x[2m,j,k] = zi[m,j,k] end
end
# Hermitian split Z (M,n,n) → X (M+1,n,n): E=½(Z[k]+conj Z[M-k]), O=(1/2i)(Z[k]−conj Z[M-k]), X=E+W_N^k O.
@kernel function _rsplit_k!(Xr, Xi, @Const(Zr), @Const(Zi), M::Int, tw)
    kk, j, k = @index(Global, NTuple)                         # kk∈1:M+1, k1=kk-1
    @inbounds begin
        T = eltype(Xr); k1 = kk-1; km = mod(M-k1, M)
        zkr = Zr[k1%M+1,j,k]; zki = Zi[k1%M+1,j,k]; zcr = Zr[km+1,j,k]; zci = -Zi[km+1,j,k]
        Er = T(0.5)*(zkr+zcr); Ei = T(0.5)*(zki+zci); Or = T(0.5)*(zki-zci); Oi = T(-0.5)*(zkr-zcr)
        wv = tw*T(k1); wr = cos(wv); ws = sin(wv)             # W_N^{k1}=exp(sgn·2πk1/n)
        Xr[kk,j,k] = Er + wr*Or - ws*Oi; Xi[kk,j,k] = Ei + wr*Oi + ws*Or
    end
end
# inverse split X (M+1,n,n) → Z (M,n,n): E=½(X[k]+conj X[M-k]), O=½W^{-k}(X[k]−conj X[M-k]), Z=E+iO.
@kernel function _runsplit_k!(Zr, Zi, @Const(Xr), @Const(Xi), M::Int, tw)
    kk, j, k = @index(Global, NTuple)                         # kk∈1:M, k1=kk-1
    @inbounds begin
        T = eltype(Zr); k1 = kk-1; mk = M-k1
        xkr = Xr[kk,j,k]; xki = Xi[kk,j,k]; xcr = Xr[mk+1,j,k]; xci = -Xi[mk+1,j,k]
        Er = T(0.5)*(xkr+xcr); Ei = T(0.5)*(xki+xci); Dr = xkr-xcr; Di = xki-xci
        wv = -tw*T(k1); wr = cos(wv); ws = sin(wv)            # W_N^{-k1}
        Or = T(0.5)*(wr*Dr - ws*Di); Oi = T(0.5)*(wr*Di + ws*Dr)
        Zr[kk,j,k] = Er - Oi; Zi[kk,j,k] = Ei + Or            # Z = E + i·O
    end
end
# Green's −1/k² on the half-grid (M+1,n,n): kx = 0..M (the non-negative rfft half), ky,kz folded.
@kernel function _scale_half_k!(Xr, Xi, coef, n::Int, w1, w2, w3)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        T = eltype(Xr)
        kx = w1*T(i-1); ky = w2*T((j-1)<=n÷2 ? (j-1) : (j-1-n)); kz = w3*T((k-1)<=n÷2 ? (k-1) : (k-1-n))
        ks = kx*kx+ky*ky+kz*kz; c = ks==zero(T) ? zero(T) : coef*(-one(T)/ks)
        Xr[i,j,k]*=c; Xi[i,j,k]*=c
    end
end
# c2c along axis 2 or 3 of a complex (n1,n2,n3): transpose-to-front, Stockham, transpose back.
# Only ONE extra pair (tr,ti): after the transpose Xr,Xi→tr,ti the source Xr,Xi is FREE, so it doubles
# as the Stockham ping-pong (reshaped to the transposed layout) ⇒ the rfft needs just 2 buffer pairs.
function _c2c_axis!(Xr, Xi, tr, ti, axis::Int, factors, sgn::Int, be)
    dims = size(Xr); perm = axis == 2 ? (2,1,3) : (3,2,1); tsh = ntuple(d -> dims[perm[d]], 3); nel = prod(dims)
    trv = reshape(view(vec(tr),1:nel),tsh); tiv = reshape(view(vec(ti),1:nel),tsh)
    _transpose!(trv, Xr, perm, be); _transpose!(tiv, Xi, perm, be)   # Xr,Xi → tr,ti; Xr,Xi now free
    srv = reshape(view(vec(Xr),1:nel),tsh); siv = reshape(view(vec(Xi),1:nel),tsh)   # ping-pong ← freed Xr,Xi
    _stockham_axis1!(trv, tiv, srv, siv, factors, sgn)        # length dims[perm[1]] along axis 1
    _transpose!(Xr, trv, perm, be); _transpose!(Xi, tiv, perm, be)   # perm is its own inverse
    return nothing
end
# cached rfft buffers: 2 pairs of (M+1,n,n) ≈ 8 B/cell (the c2c path is 12).  Allocated once (warm-up).
_rfft_bufs(be, ::Type{T}, n) where {T} = get!(_FFT_SCRATCH, (:rfft, typeof(be), T, n)) do
    M = n÷2
    (Xr=KA.zeros(be,T,M+1,n,n), Xi=KA.zeros(be,T,M+1,n,n), tr=KA.zeros(be,T,M+1,n,n), ti=KA.zeros(be,T,M+1,n,n))
end

"""
    fft_poisson_rfft_ka!(phi, rho; G, a, boxsize) -> phi

Periodic Poisson solve with the in-tree REAL FFT (rfft): ~half the work/memory of `fft_poisson_root_gpu!`
(the half-complex grid is (n/2+1)×n×n).  Even, 2·3·5-smooth cubic `n` with n/2 also 2·3·5-smooth; else
it delegates to the c2c solver.  Bit-close to `fft_poisson_root_gpu!` / FFTW.
"""
function fft_poisson_rfft_ka!(phi::AbstractArray{T,3}, rho::AbstractArray{T,3};
                              G::Real=1.0, a::Real=1.0, boxsize=1.0) where {T}
    be = KA.get_backend(rho); N = size(rho); n = N[1]
    all(==(n), N) || error("fft_poisson_rfft_ka!: cubic grid required, got $N")
    M = n ÷ 2; facM = iseven(n) ? _factorize235(M) : nothing; facN = _factorize235(n)
    (facM === nothing || facN === nothing) &&
        return fft_poisson_root_gpu!(phi, rho; G=G, a=a, boxsize=boxsize)   # fall back to c2c
    L = boxsize isa Number ? ntuple(_->Float64(boxsize),3) : ntuple(d->Float64(boxsize[d]),3)
    w = ntuple(d->T(2π)/T(L[d]),3); coef = T(G)/T(a); tw = -T(2π)/T(n)
    B = _rfft_bufs(be, T, n); Mnn = M*n*n                     # cached (allocated once in the warm-up)
    zr  = reshape(view(vec(B.tr),1:Mnn),M,n,n); zi  = reshape(view(vec(B.ti),1:Mnn),M,n,n)  # packed/axis-1 ← tr,ti
    zsr = reshape(view(vec(B.Xr),1:Mnn),M,n,n); zsi = reshape(view(vec(B.Xi),1:Mnn),M,n,n)  # axis-1 ping-pong ← Xr,Xi
    _rpack_k!(be)(zr, zi, rho; ndrange=(M,n,n))
    _stockham_axis1!(zr, zi, zsr, zsi, facM, -1)              # rfft axis-1: c2c(M) on packed
    _rsplit_k!(be)(B.Xr, B.Xi, zr, zi, M, tw; ndrange=(M+1,n,n))       # → half-complex (Xr,Xi)
    _c2c_axis!(B.Xr, B.Xi, B.tr, B.ti, 2, facN, -1, be)
    _c2c_axis!(B.Xr, B.Xi, B.tr, B.ti, 3, facN, -1, be)
    _scale_half_k!(be)(B.Xr, B.Xi, coef, n, w[1], w[2], w[3]; ndrange=(M+1,n,n))
    _c2c_axis!(B.Xr, B.Xi, B.tr, B.ti, 3, facN, +1, be)
    _c2c_axis!(B.Xr, B.Xi, B.tr, B.ti, 2, facN, +1, be)
    _runsplit_k!(be)(zr, zi, B.Xr, B.Xi, M, tw; ndrange=(M,n,n))       # Xr,Xi → zr,zi (tr,ti)
    _stockham_axis1!(zr, zi, zsr, zsi, facM, +1)              # irfft axis-1
    _runpack_k!(be)(phi, zr, zi; ndrange=(M,n,n))
    phi .*= T(2)/T(n)^3          # axis-1 round-trips ×M=×n/2 (inner c2c is length n/2) ⇒ total n³/2
    KA.synchronize(be)
    return phi
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

# On-the-fly Green's multiply: φ̂ = coef·(−1/k²)·ρ̂ with k² formed from the indices — no cached
# N³ Gk array (−4 B/cell; at the grid ceiling that array is the difference between fitting and OOM).
@kernel function _scale_by_k_onfly!(xr, xi, coef, n1::Int, n2::Int, n3::Int, w1, w2, w3)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        T = eltype(xr)
        kx = w1 * T((i-1) <= n1÷2 ? (i-1) : (i-1-n1))
        ky = w2 * T((j-1) <= n2÷2 ? (j-1) : (j-1-n2))
        kz = w3 * T((k-1) <= n3÷2 ? (k-1) : (k-1-n3))
        ks = kx*kx + ky*ky + kz*kz
        c = ks == zero(T) ? zero(T) : coef * (-one(T) / ks)
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
    all(==(N[1]), N) || error("fft_poisson_root_gpu! needs a cubic grid, got $N")
    factors = _factorize235(N[1])
    factors === nothing && error("fft_poisson_root_gpu! needs a 2·3·5-smooth dim, got $(N[1])")
    L = boxsize isa Number ? ntuple(_ -> Float64(boxsize), 3) : ntuple(d -> Float64(boxsize[d]), 3)
    pow2 = _ispow2(N[1])
    brdev = pow2 ? get!(() -> to_device(be, _bitrev_table(N[1]), Int32),
                        _GPUFFT_CACHE, (:brev, typeof(be), T, N[1])) : nothing
    w = ntuple(d -> T(2π) / T(L[d]), 3); coef = T(G) / T(a)   # G(k)=−1/k² formed on the fly (no Gk array)
    # In-place when φ aliases ρ (the GRAV1BUF shared buffer): use `rho` as the real part directly —
    # no N³ copy (−4 B/cell), and the FFT working set is ρ + xi + one scratch pair, not two copies.
    xr = phi === rho ? rho : copyto!(similar(rho), rho)
    xi = _fft_imag(be, T, N); fill!(xi, zero(T))       # cached (allocated once, in the warmup's clean pool)
    if pow2                                           # radix-2, in-place bit-reversal (SB root grids)
        _fft3d!(xr, xi, brdev, -1)
        _scale_by_k_onfly!(be)(xr, xi, coef, N[1], N[2], N[3], w[1], w[2], w[3]; ndrange = N)
        _fft3d!(xr, xi, brdev, +1)
    else                                              # mixed radix 2·3·5, Stockham (2 buffer pairs)
        sr, si = _fft_scratch(be, T, N)
        _fft3d_mixed!(xr, xi, sr, si, factors, -1)
        _scale_by_k_onfly!(be)(xr, xi, coef, N[1], N[2], N[3], w[1], w[2], w[3]; ndrange = N)
        _fft3d_mixed!(xr, xi, sr, si, factors, +1)
    end
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

"""
    power_spectrum_aniso_gpu(field; boxsize=1.0, nmu=4, nbins=0, axis=1) -> (k, P, Nmodes)

Anisotropic P(k,μ) on the device.  `field` is a real cubic power-of-two device array
(scalar overdensity) OR a 3-tuple of such arrays (a vector field → power Σ_c|f̂_c|², e.g.
velocity).  μ = |k_axis|/|k| (the v_bc stream is along `axis`; default 1).  Returns the
radial-bin centre wavenumbers `k` (physical), the binned power `P` of shape `(nbins, nmu)`,
and the per-(k,μ) mode count `Nmodes`.  Same normalization as [`power_spectrum_gpu`].  The
FFT(s) run on `field`'s backend; only the (k,μ) binning of |f̂|² is on the host.
"""
function power_spectrum_aniso_gpu(field; boxsize = 1.0, nmu::Integer = 4,
                                  nbins::Integer = 0, axis::Integer = 1)
    fields = field isa Tuple ? field : (field,)
    f1 = fields[1]; be = KA.get_backend(f1); N = size(f1); T = eltype(f1)
    all(_ispow2, N) || error("power_spectrum_aniso_gpu needs power-of-two dims, got $N")
    all(==(N[1]), N) || error("power_spectrum_aniso_gpu needs a cubic grid, got $N")
    n = N[1]; L = Float64(boxsize isa Number ? boxsize : boxsize[1])
    brdev = get!(_GPUFFT_CACHE, (:brev, typeof(be), T, n)) do
        to_device(be, _bitrev_table(n), Int32)
    end
    Pacc = nothing                                       # Σ_components |f̂|² (unnormalized), on host
    for f in fields
        xr = copyto!(similar(f), f); xi = KA.zeros(be, T, N...)
        xr, xi = _fft3d!(xr, xi, brdev, -1); KA.synchronize(be)   # forward DFT on DEVICE
        ar = Array(xr); ai = Array(xi)
        p = Float64.(ar).^2 .+ Float64.(ai).^2
        Pacc = Pacc === nothing ? p : (Pacc .+ p)
    end
    invN3 = 1.0/Float64(n)^3; V = L^3; nb = nbins > 0 ? Int(nbins) : n ÷ 2
    kf = 2π/L; kny = n/2
    Psum = zeros(Float64, nb, nmu); ksum = zeros(Float64, nb); cnt = zeros(Int, nb, nmu)
    @inbounds for k in 0:n-1
        kz = k <= n÷2 ? k : k - n
        for j in 0:n-1
            ky = j <= n÷2 ? j : j - n
            for i in 0:n-1
                kx = i <= n÷2 ? i : i - n
                km = sqrt(Float64(kx*kx + ky*ky + kz*kz))
                (km <= 0 || km > kny) && continue
                ka = axis == 1 ? kx : axis == 2 ? ky : kz
                b  = clamp(1 + floor(Int, (km/kny)*nb), 1, nb)
                mb = clamp(1 + floor(Int, (abs(Float64(ka))/km)*nmu), 1, nmu)
                Psum[b,mb] += V * Pacc[i+1,j+1,k+1] * invN3 * invN3
                ksum[b] += km*kf; cnt[b,mb] += 1
            end
        end
    end
    tot = vec(sum(cnt, dims=2))
    kc = [tot[b] > 0 ? ksum[b]/tot[b] : NaN for b in 1:nb]
    P  = [cnt[b,m] > 0 ? Psum[b,m]/cnt[b,m] : NaN for b in 1:nb, m in 1:nmu]
    return (k = kc, P = P, Nmodes = cnt)
end
