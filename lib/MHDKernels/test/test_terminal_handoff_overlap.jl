module TestTerminalHandoffOverlap

using Test
using MHDKernels

const _old_backend = get(ENV, "MHD_BACKEND", nothing)
ENV["MHD_BACKEND"] = "cpu"
include("pmf_dm_lattice_bench.jl")
if _old_backend === nothing
    delete!(ENV, "MHD_BACKEND")
else
    ENV["MHD_BACKEND"] = _old_backend
end

@testset "terminal/full-MHD handoff overlap diagnostic" begin
    N = 16
    be = MHDKernels.backend(:cpu)
    s = MHDKernels.allocate_state(be, Float32, (N, N, N);
                                  dx=1 / N, gamma=5 / 3,
                                  recon=:ppm, riemann=:hlld)
    MHDKernels.init_pmf_batchelor!(s; brms=0.1f0, rho0=1f0, p0=1f0,
                                   seed=42, kcut=2pi, kmax=2pi,
                                   mode_lock_n=N)

    zero_momentum = _terminal_handoff_momenta!(be, s, 32.0, Inf, 0.0;
                                                measure=true)
    @test isapprox(zero_momentum.relative, sqrt(2.0); rtol=2e-6)
    @test zero_momentum.full_v_rms == 0
    @test zero_momentum.terminal_v_rms > 0

    _terminal_handoff_momenta!(be, s, 32.0, Inf, 1.0)
    matched = _terminal_handoff_momenta!(be, s, 32.0, Inf, 0.0;
                                         measure=true)
    @test matched.relative < 2e-5
    @test matched.delta_v_rms < 2e-8
    @test isapprox(matched.full_v_rms, matched.terminal_v_rms; rtol=2e-6)
end

@testset "cosine handoff releases terminal momentum smoothly" begin
    alpha = [_hybrid_handoff_alpha(1.0, remaining, 64) for remaining in 64:-1:1]
    @test alpha[1] == 1.0
    @test alpha[end] == 0.0
    @test all(diff(alpha) .<= 0)
    @test 0.45 < alpha[32] < 0.55
end

end
