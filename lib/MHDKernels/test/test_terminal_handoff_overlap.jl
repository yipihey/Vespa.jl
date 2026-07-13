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

@testset "source cadence limits the outer MHD step" begin
    @test _limit_source_dt(8.0, 1.0, Inf) == (dt=8.0, limited=false)
    @test _limit_source_dt(8.0, 1.0, 4.0) == (dt=4.0, limited=true)
    @test _limit_source_dt(0.5, 1.0, 1.0) == (dt=0.5, limited=false)
end

@testset "hybrid aware cadence retains explicit MHD subcycles" begin
    @test _limit_hybrid_aware_dt(16.0, 1.0, 4.0) ==
          (dt=4.0, nsub=4, limited=true)
    @test _limit_hybrid_aware_dt(4.0, 1.0, 16.0) ==
          (dt=4.0, nsub=4, limited=false)
end

@testset "cosmological and hydro clocks preserve drag impulse" begin
    gamma = 3.2e-11
    dtphys = 4.5e9
    dtau = 0.03
    vunit = 8.0e5
    lbox = 6.0 * _PMF_KPC_CM
    amid = 2.5e-4
    dth = _hydro_code_dt(dtphys, dtau, vunit, lbox, amid)
    gamma_h = _terminal_gamma_code(gamma, dtphys, dtau, vunit, lbox, amid)
    @test gamma_h * dth ≈ gamma * dtphys rtol=2e-15
    @test _hydro_code_dt(dtphys, dtau, vunit, NaN, amid) == dtau
    @test _hydro_subinterval(dth, dtau, 0.25dtau) ≈ 0.25dth rtol=2e-15
end

end
