# Opt-in GPU/CPU-device test suite for the KernelAbstractions extensions (KAFlux,
# KA composite Poisson, KA particle push). Needs the `test/gpu` environment
# (KernelAbstractions + CUDA + HGBackend); the CUDA gates skip cleanly when no
# device is present. Run: `<julia> --project=test/gpu test/gpu/runtests.jl`.
using Test

@testset "Vespa GPU/KA extensions" begin
    include("test_ka_flux_gpu.jl")       # P1: batched-per-face flux (KAFlux), round-off AMR
    include("test_ka_gravity_gpu.jl")    # P2.2: KA composite Poisson, across-level parity
    include("test_ka_particles_gpu.jl")  # P2.3: KA particle push, parity vs CPU
    include("test_mg_poisson_gpu.jl")    # geometric-multigrid-preconditioned Poisson (uniform): ≥3× fewer iters
    include("test_cosmology_native_gpu.jl")  # P4: native DM cosmology on GPU (gravity+expansion+AMR)
end
