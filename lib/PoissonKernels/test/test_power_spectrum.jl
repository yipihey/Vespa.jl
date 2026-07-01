@testset "Power spectrum rFFT path" begin
    if metal_ready()
        N = 16
        δ = [sin(2f0*pi*Float32(i-1)/N) +
             0.25f0*cos(4f0*pi*Float32(j-1)/N) +
             0.125f0*sin(6f0*pi*Float32(k-1)/N)
             for i in 1:N, j in 1:N, k in 1:N]
        δ .-= Float32(sum(δ) / length(δ))

        pc = PoissonKernels.power_spectrum_aniso_gpu(δ; boxsize=1.0, nmu=4, nbins=8, axis=1)
        pm = PoissonKernels.power_spectrum_aniso_gpu(Metal.MtlArray(δ);
                                                     boxsize=1.0, nmu=4, nbins=8, axis=1)
        mask = .!isnan.(pc.P)
        @test pm.Nmodes == pc.Nmodes
        @test maximum(abs.(pm.P[mask] .- pc.P[mask])) / maximum(abs.(pc.P[mask])) < 1e-5

        f1 = [sin(2f0*pi*Float32(i-1)/N) for i in 1:N, j in 1:N, k in 1:N]
        f2 = [cos(2f0*pi*Float32(j-1)/N) for i in 1:N, j in 1:N, k in 1:N]
        f3 = [sin(2f0*pi*Float32(k-1)/N) for i in 1:N, j in 1:N, k in 1:N]
        pcv = PoissonKernels.power_spectrum_aniso_gpu((f1, f2, f3);
                                                      boxsize=1.0, nmu=4, nbins=8, axis=2)
        pmv = PoissonKernels.power_spectrum_aniso_gpu((Metal.MtlArray(f1),
                                                       Metal.MtlArray(f2),
                                                       Metal.MtlArray(f3));
                                                      boxsize=1.0, nmu=4, nbins=8, axis=2)
        maskv = .!isnan.(pcv.P)
        @test pmv.Nmodes == pcv.Nmodes
        @test maximum(abs.(pmv.P[maskv] .- pcv.P[maskv])) / maximum(abs.(pcv.P[maskv])) < 1e-5
    else
        @test_skip "Metal not available — rFFT P(k,mu) parity skipped"
    end
end
