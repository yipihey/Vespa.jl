# twogrid_gpu.jl — a two-grid ("coarse FFT + fine Gauss-Seidel") periodic Poisson solve.
#
# Solves ∇²φ = (G/a)·ρ on a periodic cubic grid, replacing a full-resolution FFT with:
#   restrict ρ (2×) → COARSE cuFFT solve → trilinear prolong → K red-black GS sweeps (7-pt FD).
#
# Why: at the 900³ memory ceiling the full cuFFT OOMs (which is why the hand-rolled mixed-radix KA rFFT
# exists — but it is ~16× slower).  A HALF-resolution cuFFT fits and is cheap, and the streaming scale
# (k≈800/Mpc) sits well BELOW the coarse Nyquist, so the coarse spectral solve resolves it exactly; the
# red-black GS only builds the modes above the coarse Nyquist (near fine-Nyquist, physically unimportant).
# The GS uses the FD 7-point Laplacian: for k below the coarse Nyquist FD≈spectral, so the physics is
# preserved; the operator only differs (spectral→FD) in the top octave.  Also 8× smaller FFT workspace.
#
# Scaling (validated by a single-mode probe): the FFT solves ∇²φ=(G/a)ρ with φ̂=-(G/a)ρ̂/k².  The FD update
# solving the same is  φ[i] = (Σ_neighbours − dx²·(G/a)·ρ[i]) / 6  with dx = boxsize/n.

# ── factor-2 restrict (cell-centred 2×2×2 average): fine (2nc)³ → coarse (nc)³ ──
@kernel function _tg_restrict2!(dst, @Const(src))
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        bi = 2i-1; bj = 2j-1; bk = 2k-1
        dst[i,j,k] = (src[bi,bj,bk]   + src[bi+1,bj,bk]   + src[bi,bj+1,bk]   + src[bi+1,bj+1,bk] +
                      src[bi,bj,bk+1] + src[bi+1,bj,bk+1] + src[bi,bj+1,bk+1] + src[bi+1,bj+1,bk+1]) * eltype(dst)(0.125)
    end
end

# ── factor-2 trilinear prolong (periodic, cell-centred): coarse (nc)³ → fine (2nc)³ ──
# Each fine cell's centre lies at ±¼ of a coarse cell from its nearest coarse centre ⇒ weights (¾,¼) per axis.
@kernel function _tg_prolong2!(dst, @Const(src), nc::Int)
    fi, fj, fk = @index(Global, NTuple)
    @inbounds begin
        T = eltype(dst)
        ci = (fi+1) >> 1; cj = (fj+1) >> 1; ck = (fk+1) >> 1                    # nearest coarse (weight ¾)
        ci1 = mod1(isodd(fi) ? ci-1 : ci+1, nc)                                 # second coarse (weight ¼), periodic
        cj1 = mod1(isodd(fj) ? cj-1 : cj+1, nc)
        ck1 = mod1(isodd(fk) ? ck-1 : ck+1, nc)
        a = T(0.75); b = T(0.25)
        dst[fi,fj,fk] = a*a*a*src[ci,cj,ck]  + b*a*a*src[ci1,cj,ck]  + a*b*a*src[ci,cj1,ck]  + b*b*a*src[ci1,cj1,ck] +
                        a*a*b*src[ci,cj,ck1] + b*a*b*src[ci1,cj,ck1] + a*b*b*src[ci,cj1,ck1] + b*b*b*src[ci1,cj1,ck1]
    end
end

# ── one red-black Gauss-Seidel colour sweep of the periodic 7-point FD Poisson ∇²φ=(G/a)ρ ──
# ss = dx²·(G/a).  For a mean-zero ρ this preserves Σφ (no null-mode drift), so no mean-pin is needed.
@kernel function _tg_rbgs!(phi, @Const(rho), ss, n::Int, color::Int)
    i, j, k = @index(Global, NTuple)
    @inbounds if (i + j + k) & 1 == color
        ip = i==n ? 1 : i+1; im = i==1 ? n : i-1
        jp = j==n ? 1 : j+1; jm = j==1 ? n : j-1
        kp = k==n ? 1 : k+1; km = k==1 ? n : k-1
        phi[i,j,k] = (phi[ip,j,k] + phi[im,j,k] + phi[i,jp,k] + phi[i,jm,k] + phi[i,j,kp] + phi[i,j,km]
                      - ss * rho[i,j,k]) * eltype(phi)(1//6)
    end
end

const _TG_SCRATCH = Dict{Any,Any}()
_tg_bufs(be, ::Type{T}, nc) where {T} = get!(_TG_SCRATCH, (typeof(be), T, nc)) do
    (rc = KA.zeros(be, T, nc, nc, nc), pc = KA.zeros(be, T, nc, nc, nc))
end

"""
    fft_poisson_2grid!(phi, rho; G, a, boxsize, nsweeps=6) -> phi

Periodic Poisson solve `∇²φ = (G/a)·ρ` via a two-grid cycle: half-resolution cuFFT coarse solve +
`nsweeps` red-black Gauss-Seidel sweeps of the 7-point FD operator on the fine grid.  In place in `phi`
(a cubic device array; `rho` matching, even `n` with `n/2` cuFFT-able).  The coarse scratch is cached.
"""
function fft_poisson_2grid!(phi::AbstractArray{T,3}, rho::AbstractArray{T,3};
                            G::Real=1.0, a::Real=1.0, boxsize=1.0, nsweeps::Int=6) where {T}
    be = KA.get_backend(rho); n = size(rho, 1); nc = n ÷ 2
    all(==(n), size(rho)) || error("fft_poisson_2grid!: cubic grid required, got $(size(rho))")
    iseven(n) || error("fft_poisson_2grid!: needs even n, got $n")
    bx = boxsize isa Number ? Float64(boxsize) : Float64(boxsize[1])
    buf = _tg_bufs(be, T, nc)
    _tg_restrict2!(be)(buf.rc, rho; ndrange=(nc,nc,nc))
    copyto!(buf.pc, buf.rc)
    fft_poisson_rfft!(buf.pc, buf.pc; G=G, a=a, boxsize=boxsize)     # coarse cuFFT solve (½ the axes)
    _tg_prolong2!(be)(phi, buf.pc, nc; ndrange=(n,n,n))
    ss = T((bx/n)^2 * (G/a))
    for _ in 1:nsweeps
        _tg_rbgs!(be)(phi, rho, ss, n, 0; ndrange=(n,n,n))
        _tg_rbgs!(be)(phi, rho, ss, n, 1; ndrange=(n,n,n))
    end
    KA.synchronize(be)
    return phi
end
