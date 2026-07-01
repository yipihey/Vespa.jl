# Root-grid FFT Poisson solve — port of Enzo's ComputePotentialFieldLevelZeroPer
# (src/enzo/ComputePotentialFieldLevelZero.C) + the periodic Green's function
# (Grid_PreparePeriodicGreensFunction.C, the default INFLUENCE1 form):
#
#   φ̂(k) = coef · G(k) · ρ̂(k),   G(k) = -1/k²  (G(0)=0),   coef = GravConst / a
#   k_d = 2π · (wrapped integer mode) / DomainSize_d        (the continuum operator)
#
# The forward/inverse FFT runs on the HOST via FFTW (there is no KernelAbstractions
# FFT; the root is one small uniform grid, so a host round-trip is cheap — this is
# the deliberate "CPU-host FFT" decision). Everything else stays KA-friendly; device
# (Metal) arrays are accepted and staged to the host around the transform. The solve
# is spectral, so it is EXACT for resolved modes — certified against analytic φ.

using LinearAlgebra: mul!
using FFTW: rfft, irfft, plan_rfft, plan_irfft, set_num_threads

"""
    fft_set_num_threads!(n) -> n

Set the number of CPU threads FFTW uses for the (real) transforms in
`fft_poisson_root!` — the "parallel CPU FFT" for a top-grid Poisson solve.  Call
ONCE before the first `fft_poisson_root!` of a given size: plans are built lazily
on first use and cached per `(eltype,dims,boxsize,greens)`, baking in the thread
count at creation.  Returns `n`.
"""
fft_set_num_threads!(n::Integer) = (set_num_threads(Int(n)); Int(n))

# real-to-complex Green's function on the rfft grid (N1÷2+1, N2, N3), built on the
# host (it feeds the host FFT). Frequencies use the standard (physically correct)
# wrap; for -1/k² only |k| matters. `L` = per-axis domain size (code units, =1 for
# cosmology). DC mode set to 0 (the periodic null space — RHS must be zero-mean).
function _greens_periodic(::Type{T}, N::NTuple{3,Int}, L::NTuple{3,Float64}) where {T}
    n1 = N[1] ÷ 2 + 1
    Gk = Array{T}(undef, n1, N[2], N[3])
    w1 = T(2π) / T(L[1]); w2 = T(2π) / T(L[2]); w3 = T(2π) / T(L[3])
    @inbounds for k in 0:N[3]-1
        km = k ≤ N[3] ÷ 2 ? k : k - N[3]; kz = w3 * T(km)
        for j in 0:N[2]-1
            jm = j ≤ N[2] ÷ 2 ? j : j - N[2]; ky = w2 * T(jm)
            for i in 0:n1-1
                kx = w1 * T(i)
                ksqr = kx * kx + ky * ky + kz * kz
                Gk[i+1, j+1, k+1] = ksqr == zero(T) ? zero(T) : -one(T) / ksqr
            end
        end
    end
    return Gk
end

# Green's function of the DISCRETE 7-point Laplacian: eigenvalues
# λ(m) = -Σ_d (4/h_d²)·sin²(π m_d / N_d), h_d = L_d/N_d.  An FFT solve with this
# kernel is the EXACT solution of the same linear system a finite-difference
# relaxation solver (RAMSES's multigrid/CG, Enzo's subgrid MG) iterates on — so
# it certifies those solvers to their convergence tolerance, with no O(h²)
# spectral-vs-discrete gap.  DC mode 0 (RHS must be zero-mean), like :spectral.
function _greens_discrete7(::Type{T}, N::NTuple{3,Int}, L::NTuple{3,Float64}) where {T}
    n1 = N[1] ÷ 2 + 1
    Gk = Array{T}(undef, n1, N[2], N[3])
    c = ntuple(d -> T(4) * (T(N[d]) / T(L[d]))^2, 3)     # 4/h_d²
    @inbounds for k in 0:N[3]-1
        sz = c[3] * sin(T(π) * T(k) / T(N[3]))^2
        for j in 0:N[2]-1
            sy = c[2] * sin(T(π) * T(j) / T(N[2]))^2
            for i in 0:n1-1
                sx = c[1] * sin(T(π) * T(i) / T(N[1]))^2
                lam = sx + sy + sz
                Gk[i+1, j+1, k+1] = lam == zero(T) ? zero(T) : -one(T) / lam
            end
        end
    end
    return Gk
end

_host(a::AbstractArray) = a isa Array ? a : to_host(a)

# ── per-(T,N,L) plan + Green's-function + scratch cache ───────────────────────
# fft_poisson_root! is called once per gravity step on a FIXED root grid, so the
# Green's function (an N1·N2·N3-element build) and the FFTW plans (whose creation
# dominates a single transform) are computed ONCE and reused.  This turns the
# per-call cost into just the two transforms + one elementwise multiply — the
# 126³ rebuild + plan creation that dominated the profile are gone.
_rfft_dims(N::NTuple{3,Int}) = (N[1] ÷ 2 + 1, N[2], N[3])

struct _FFTPlan{T,PF,PI}
    Gk::Array{T,3}
    fwd::PF                       # plan_rfft
    inv::PI                       # plan_irfft
    chat::Array{Complex{T},3}     # cached rfft workspace
end
const _FFT_CACHE = Dict{Tuple{DataType,NTuple{3,Int},NTuple{3,Float64},Symbol},Any}()
const _FFT_REAL_STAGING = Dict{Tuple{DataType,NTuple{3,Int}},Any}()

function _fft_plan(::Type{T}, N::NTuple{3,Int}, L::NTuple{3,Float64}, greens::Symbol) where {T}
    get!(_FFT_CACHE, (T, N, L, greens)) do
        rbuf = zeros(T, N)
        fwd  = plan_rfft(rbuf)
        chat = Array{Complex{T},3}(undef, _rfft_dims(N))
        inv  = plan_irfft(chat, N[1])
        Gk = greens === :spectral ? _greens_periodic(T, N, L) :
             greens === :discrete7 ? _greens_discrete7(T, N, L) :
             error("fft_poisson_root!: greens must be :spectral or :discrete7 (got $greens)")
        _FFTPlan{T,typeof(fwd),typeof(inv)}(Gk, fwd, inv, chat)
    end::_FFTPlan
end

function _fft_real_staging(::Type{T}, N::NTuple{3,Int}) where {T}
    get!(_FFT_REAL_STAGING, (T, N)) do
        Array{T,3}(undef, N)
    end::Array{T,3}
end

"""
    fft_poisson_root!(phi, rho; G=1.0, a=1.0, boxsize=1.0, greens=:spectral) -> phi

Solve the periodic Poisson equation `∇²φ = (G/a)·ρ` on a uniform grid by the
spectral (FFT) method — Enzo's root-grid gravity solver. `phi`, `rho` are 3-D
arrays (CPU or Metal; staged to the host for the FFT). `boxsize` is the per-axis
domain size in code units (scalar for a cubic box). `rho` should be zero-mean
(periodic solvability); the DC mode is dropped. Returns `phi` (filled in place).

`greens` selects the kernel: `:spectral` (default, the continuum −1/k² — Enzo's
root convention, exact for resolved modes) or `:discrete7` (the discrete 7-point
Laplacian's eigenvalues — the EXACT solution of the linear system RAMSES's
multigrid/CG or any finite-difference relaxation solver converges to, the choice
for bit-level cross-certification against those solvers).

The Green's function and FFTW plans are cached per `(eltype, dims, boxsize,
greens)` (see `_fft_plan`), so repeated calls on the same root grid pay only the
two transforms + the spectral multiply.  For the lowest latency (no device↔host
staging) pass a **host** `rho`/`phi`; device arrays are accepted and staged.
"""
function fft_poisson_root!(phi::AbstractArray{T,3}, rho::AbstractArray{T,3};
                           G::Real = 1.0, a::Real = 1.0, boxsize = 1.0,
                           greens::Symbol = :spectral) where {T}
    N = size(rho)
    L = boxsize isa Number ? ntuple(_ -> Float64(boxsize), 3) :
        ntuple(d -> Float64(boxsize[d]), 3)
    P = _fft_plan(T, N, L, greens)
    phi_h = phi isa Array ? phi : _fft_real_staging(T, N)
    copyto!(phi_h, _host(rho))                 # use caller's φ as the real FFT input when host-side
    mul!(P.chat, P.fwd, phi_h)                 # ρ̂ = rfft(ρ)
    coef = T(G) / T(a)
    @inbounds @. P.chat = coef * P.Gk * P.chat # φ̂ = coef·G(k)·ρ̂
    mul!(phi_h, P.inv, P.chat)                 # back to real space (normalized)
    phi === phi_h || copyto!(phi, phi_h)       # host→device only when needed
    return phi
end

# ── array-generic rfft Poisson (FFTW for Array, cuFFT for CuArray) ─────────────
# Solves ∇²φ = (G/a)·ρ on `ρ`'s OWN device via the AbstractFFTs `plan_rfft`/`plan_irfft`
# interface, which dispatches to FFTW (host) or cuFFT (CuArray/MtlArray) automatically.
# Real-to-complex (rfft) ⇒ half the spectral storage of a c2c transform, and cuFFT
# supports ARBITRARY sizes (best for 2^a·3^b·5^c·7^d) — no power-of-two restriction.
# Plans + the −1/k² Green's function (rfft half-grid) are cached per (array-type,T,N,L,greens).
struct _RFFTPlan{PF,PI,A,C}
    fwd::PF; inv::PI; Gk::A; chat::C      # Gk/chat live on ρ's device
end
const _RFFT_CACHE = Dict{Any,Any}()

"""
    fft_poisson_rfft!(φ, ρ; G=1.0, a=1.0, boxsize=1.0, greens=:spectral) -> φ

In-place periodic Poisson solve `∇²φ = (G/a)·ρ` on `ρ`'s native device using the real
FFT (`plan_rfft`).  `φ`,`ρ` are 3-D arrays of the SAME type (host `Array` → FFTW; `CuArray`
→ cuFFT — no host staging).  Arbitrary (non-power-of-two) sizes are allowed.  DC mode
dropped (RHS should be mean-zero; the cosmological source already is).
"""
function fft_poisson_rfft!(φ::AbstractArray{T,3}, ρ::AbstractArray{T,3};
                           G::Real=1.0, a::Real=1.0, boxsize=1.0, greens::Symbol=:spectral) where {T}
    N = size(ρ)
    L = boxsize isa Number ? ntuple(_->Float64(boxsize),3) : ntuple(d->Float64(boxsize[d]),3)
    P = get!(_RFFT_CACHE, (typeof(ρ), T, N, L, greens)) do
        fwd = plan_rfft(ρ)                       # device plan (cuFFT) or host (FFTW)
        chat = similar(ρ, Complex{T}, _rfft_dims(N))
        inv = plan_irfft(chat, N[1])
        gh  = greens === :spectral ? _greens_periodic(T, N, L) :
              greens === :discrete7 ? _greens_discrete7(T, N, L) :
              error("fft_poisson_rfft!: greens must be :spectral or :discrete7")
        gk  = ρ isa Array ? gh : copyto!(similar(ρ, T, size(gh)), gh)   # Green's onto ρ's device
        _RFFTPlan(fwd, inv, gk, chat)
    end::_RFFTPlan
    coef = T(G) / T(a)
    mul!(P.chat, P.fwd, ρ)                       # rfft(ρ) — complex, (N₁÷2+1,N₂,N₃)
    @. P.chat = coef * P.Gk * P.chat             # φ̂ = coef·G(k)·ρ̂
    mul!(φ, P.inv, P.chat)                       # inverse rfft (normalized)
    return φ
end
