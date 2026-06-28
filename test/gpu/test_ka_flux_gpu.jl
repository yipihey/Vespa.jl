# GPU + CPU device test for the KA flux backend (`KAFlux`), the Phase-1 "GPU
# parity" gate of the re-platform plan. Proves the SAME `_accumulate_flux_batched!`
# machinery that conserves to round-off with `HostBatchedFlux` also does so when the
# per-face Riemann solve runs as a KernelAbstractions kernel — on CPU-KA always, and
# on CUDA when a device is present.
#
#   * CPU-KA (`KAFlux(CPU())`) ≡ `HostBatchedFlux`: same `hllc_flux`, same scatter ⇒
#     bit-identical (worst diff 0), and round-off conservation under AMR.
#   * CUDA (`KAFlux(CUDABackend())`): round-off conservation under AMR (the flux is
#     copied to the host and scattered identically, so telescoping is exact); and
#     solution parity with the host path to floating-point tolerance (GPU FMA may
#     differ in the last bits, which does NOT affect conservation).
#
# Run: `<julia> --project=test/gpu test/gpu/test_ka_flux_gpu.jl`

using Test
using MeshInterface
using RefMesh
using Vespa
using HGBackend
using KernelAbstractions
using CUDA

include(joinpath(@__DIR__, "..", "..", "problems", "sedov_blast.jl"))

# Run a problem on HGBackend with an optional flux backend and optional AMR policy.
function run_with(prob; flux = nothing, policy = nothing, subcycle = false)
    mesh = HGMesh(prob.dims, prob.domain)
    sim = Simulation(mesh, prob)
    sim.flux = flux
    t0 = conserved_totals(sim)
    policy === nothing ? evolve!(sim) : evolve!(sim; policy = policy, subcycle = subcycle)
    return sim, t0, conserved_totals(sim)
end

# Worst per-component state difference between two runs over a matched cell order.
function worst_state_diff(a::Simulation, b::Simulation)
    as = collect(cell_samples(a)); bs = collect(cell_samples(b))
    length(as) == length(bs) || return Inf
    w = 0.0
    for (x, y) in zip(as, bs), k in eachindex(x[2])
        w = max(w, abs(x[2][k] - y[2][k]))
    end
    return w
end

@testset "KAFlux on KernelAbstractions (CPU + CUDA)" begin
    sod_policy() = RefinementPolicy(refine_above = 0.1, max_level = 2, every = 4)

    @testset "CPU-KA: KAFlux(CPU()) ≡ HostBatchedFlux, round-off AMR" begin
        prob = sod_problem_defaults(n = 64)
        host, h0, h1 = run_with(prob; flux = HostBatchedFlux(), policy = sod_policy())
        cpu,  c0, c1 = run_with(prob; flux = KAFlux(CPU()),     policy = sod_policy())

        @test c1.mass ≈ c0.mass rtol = 1e-9          # round-off conservation
        @test c1.energy ≈ c0.energy rtol = 1e-9
        @test n_cells(cpu.backend) == n_cells(host.backend)
        @test worst_state_diff(cpu, host) < 1e-12    # same arithmetic ⇒ bit-identical
        @info "CPU-KA vs Host" worst = worst_state_diff(cpu, host) mass_drift = abs(c1.mass - c0.mass)
    end

    if !CUDA.functional()
        @info "CUDA not functional — skipping the CUDA flux gate (CPU-KA path verified)"
    else
        @info "CUDA device" name = CUDA.name(CUDA.device())
        cu() = KAFlux(CUDABackend())

        @testset "CUDA: round-off conservation under AMR (refined Sod)" begin
            prob = sod_problem_defaults(n = 64)
            sim, t0, t1 = run_with(prob; flux = cu(), policy = sod_policy())
            @test max_level(sim.backend) >= 1
            @test t1.mass ≈ t0.mass rtol = 1e-9
            @test t1.energy ≈ t0.energy rtol = 1e-9
            @info "CUDA Sod AMR" leaves = n_cells(sim.backend) mass_drift = abs(t1.mass - t0.mass)
        end

        @testset "CUDA: round-off conservation under AMR (2D Sedov)" begin
            prob = sedov_problem(n = 32, tfinal = 0.02)
            sim, t0, t1 = run_with(prob; flux = cu(),
                                   policy = RefinementPolicy(refine_above = 0.05, max_level = 2, every = 4))
            @test max_level(sim.backend) >= 1
            @test t1.mass ≈ t0.mass rtol = 1e-9
            @test t1.energy ≈ t0.energy rtol = 1e-9
        end

        @testset "CUDA: subcycled refined Sod conserves (FluxRegister reflux)" begin
            prob = sod_problem_defaults(n = 64)
            sim, t0, t1 = run_with(prob; flux = cu(), policy = sod_policy(), subcycle = true)
            @test t1.mass ≈ t0.mass rtol = 1e-9
            @test t1.energy ≈ t0.energy rtol = 1e-9
        end

        @testset "CUDA: solution parity with the host path (uniform mesh, no regrid)" begin
            # No regridding ⇒ identical mesh, so cell-by-cell parity isolates the
            # flux values. GPU f64 may differ from CPU in the last bits (FMA), so
            # the bound is floating-point, not bit-identical.
            prob = sod_problem_defaults(n = 128)
            host, _, _ = run_with(prob; flux = HostBatchedFlux())
            gpu,  _, _ = run_with(prob; flux = cu())
            w = worst_state_diff(gpu, host)
            @info "CUDA vs Host (uniform Sod)" worst_state_diff = w
            @test w < 1e-9
        end
    end
end
