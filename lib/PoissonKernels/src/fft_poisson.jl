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

using FFTW: rfft, irfft, plan_rfft, plan_irfft

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

_host(a::AbstractArray) = a isa Array ? a : to_host(a)

# ── per-(T,N,L) plan + Green's-function + scratch cache ───────────────────────
# fft_poisson_root! is called once per gravity step on a FIXED root grid, so the
# Green's function (an N1·N2·N3-element build) and the FFTW plans (whose creation
# dominates a single transform) are computed ONCE and reused.  This turns the
# per-call cost into just the two transforms + one elementwise multiply — the
# 126³ rebuild + plan creation that dominated the profile are gone.
struct _FFTPlan{T,PF,PI}
    Gk::Array{T,3}
    fwd::PF                       # plan_rfft (applied with `*`)
    inv::PI                       # plan_irfft (applied with `*`)
    rbuf::Array{T,3}              # real scratch (decouples caller's ρ from FFTW)
end
const _FFT_CACHE = Dict{Tuple{DataType,NTuple{3,Int},NTuple{3,Float64}},Any}()

function _fft_plan(::Type{T}, N::NTuple{3,Int}, L::NTuple{3,Float64}) where {T}
    get!(_FFT_CACHE, (T, N, L)) do
        rbuf = zeros(T, N)
        fwd  = plan_rfft(rbuf)
        chat = fwd * rbuf                      # probe the rfft shape for the inverse plan
        inv  = plan_irfft(chat, N[1])
        _FFTPlan{T,typeof(fwd),typeof(inv)}(_greens_periodic(T, N, L), fwd, inv, rbuf)
    end::_FFTPlan
end

"""
    fft_poisson_root!(phi, rho; G=1.0, a=1.0, boxsize=1.0) -> phi

Solve the periodic Poisson equation `∇²φ = (G/a)·ρ` on a uniform grid by the
spectral (FFT) method — Enzo's root-grid gravity solver. `phi`, `rho` are 3-D
arrays (CPU or Metal; staged to the host for the FFT). `boxsize` is the per-axis
domain size in code units (scalar for a cubic box). `rho` should be zero-mean
(periodic solvability); the DC mode is dropped. Returns `phi` (filled in place).

The Green's function and FFTW plans are cached per `(eltype, dims, boxsize)`
(see `_fft_plan`), so repeated calls on the same root grid pay only the two
transforms + the spectral multiply.  For the lowest latency (no device↔host
staging) pass a **host** `rho`/`phi`; device arrays are accepted and staged.
"""
function fft_poisson_root!(phi::AbstractArray{T,3}, rho::AbstractArray{T,3};
                           G::Real = 1.0, a::Real = 1.0, boxsize = 1.0) where {T}
    N = size(rho)
    L = boxsize isa Number ? ntuple(_ -> Float64(boxsize), 3) :
        ntuple(d -> Float64(boxsize[d]), 3)
    P = _fft_plan(T, N, L)
    copyto!(P.rbuf, _host(rho))                # host (or staged-from-device) ρ
    chat = P.fwd * P.rbuf                      # ρ̂ = rfft(ρ)
    coef = T(G) / T(a)
    @inbounds @. chat = coef * P.Gk * chat     # φ̂ = coef·G(k)·ρ̂
    phi_h = P.inv * chat                       # back to real space (normalized)
    copyto!(phi, phi_h)                        # host→host or host→device
    return phi
end
