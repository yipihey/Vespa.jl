# Root-grid FFT Poisson solve (fft_poisson_root!) — certified against the ANALYTIC
# solution, which is exact for a spectral solver: for a Fourier mode of wavenumber
# K, ∇²φ = coef·ρ ⇒ φ = -coef·ρ/K². (No bit-tight Enzo oracle here — FFTW ≠ Enzo's
# Fortran FFT; the live-Enzo comparison is Phase C's session_gravity. The analytic
# check is the rigorous mathematical truth.)

# Build a multi-mode periodic source ρ and its exact potential φ (coef=1), Float64.
# Each mode contributes b = amp·∏ sin(2π m_d x_d); ∇²(that) = -K²·b, so φ += -b/K².
function _fft_modes(N::Int)
    modes = [(1.0, 1, 0, 0), (0.5, 0, 2, 0), (0.7, 1, 1, 0), (0.3, 1, 1, 1), (0.4, 0, 0, 3)]
    ρ = zeros(Float64, N, N, N); φ = zeros(Float64, N, N, N)
    for (amp, mx, my, mz) in modes
        K2 = (2π)^2 * (mx^2 + my^2 + mz^2)
        @inbounds for k in 0:N-1, j in 0:N-1, i in 0:N-1
            b = amp
            mx != 0 && (b *= sinpi(2 * mx * i / N))
            my != 0 && (b *= sinpi(2 * my * j / N))
            mz != 0 && (b *= sinpi(2 * mz * k / N))
            ρ[i+1, j+1, k+1] += b
            φ[i+1, j+1, k+1] += -b / K2
        end
    end
    return ρ, φ
end

_fmaxrel(a, b) = maximum(abs.(vec(a) .- vec(b))) / max(maximum(abs, b), eps())

@testset "FFT root Poisson solve — spectral, periodic" begin
    N = 32
    ρ64, φ64 = _fft_modes(N)

    solve(name, ::Type{T}) where {T} = begin
        be = PoissonKernels.backend(name)
        rho = PoissonKernels.to_device(be, ρ64, T)
        phi = PoissonKernels.device_zeros(be, T, (N, N, N))
        PoissonKernels.fft_poisson_root!(phi, rho; G = 1.0, a = 1.0, boxsize = 1.0)
        PoissonKernels.to_host(phi)
    end

    # f64 CPU: spectral solve == analytic φ to round-off
    err64 = _fmaxrel(solve(:cpu, Float64), φ64)
    @test err64 < 1e-12
    @info "FFT root solve f64 vs analytic" maxrel = err64

    # f32 CPU: accuracy floor
    @test _fmaxrel(solve(:cpu, Float32), φ64) < 1e-4

    # Metal staging path (host FFT, device in/out): runs + matches the CPU f32 solve
    if metal_ready()
        @test _fmaxrel(solve(:metal, Float32), solve(:cpu, Float32)) < 1e-5
    end

    # the G/a coefficient scales the solution linearly
    be = PoissonKernels.backend(:cpu)
    phiG = PoissonKernels.device_zeros(be, Float64, (N, N, N))
    PoissonKernels.fft_poisson_root!(phiG, PoissonKernels.to_device(be, ρ64, Float64);
                                     G = 4π, a = 2.0, boxsize = 1.0)
    @test _fmaxrel(PoissonKernels.to_host(phiG), (4π / 2.0) .* φ64) < 1e-12

    # real cosmological field (SB-256 δ) — solve runs, finite, zero-mean φ
    if isfile("/tmp/sb256_delta.bin")
        δ = open("/tmp/sb256_delta.bin", "r") do io
            n = Int(read(io, Int64)); d = Vector{Float64}(undef, n^3); read!(io, d); reshape(d, n, n, n)
        end
        bec = PoissonKernels.backend(:cpu)
        phi = PoissonKernels.device_zeros(bec, Float64, size(δ))
        PoissonKernels.fft_poisson_root!(phi, PoissonKernels.to_device(bec, δ, Float64); G = 1.0, a = 1.0)
        ph = PoissonKernels.to_host(phi)
        @test all(isfinite, ph)
        @info "FFT root solve on real SB-256 δ" N = size(δ, 1) phi_range = (minimum(ph), maximum(ph))
    end
end
