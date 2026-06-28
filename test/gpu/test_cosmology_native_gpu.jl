# Phase 4 GPU gate: native cosmological DM structure formation on the GPU. The whole
# stack runs device-resident — KA composite Poisson (`enable_cosmology!(...; ka=)`)
# and the KA particle push (`evolve_cosmology!(...; ka=)`) — on an HGBackend
# hierarchy that refines on DM overdensity, driven by comoving self-gravity +
# expansion. This is the GPU proof of the CICASS-run backbone (no Enzo), at a scale
# that runs in seconds; the driver scales to 128³ + 3 levels.
#
# Run: `<julia> --project=test/gpu test/gpu/test_cosmology_native_gpu.jl`

using Test
using MeshInterface
using RefMesh
using Vespa
using HGBackend
using KernelAbstractions
using CUDA

@testset "Cosmology P4 (GPU): native DM structure formation" begin
    if !CUDA.functional()
        @info "CUDA not functional — skipping the GPU cosmology gate"
    else
        @info "CUDA device" name = CUDA.name(CUDA.device())
        zi, zc, kx, n = 20.0, 1.0, 2π, 16
        Aof(a) = a * (1 + zc) / (1 + zi)
        vamp(a) = -sqrt(2 / 3) * sqrt(a) * (1 + zc) / ((1 + zi) * kx)
        xmap(ξ, a) = ξ - Aof(a) / kx * sin(kx * ξ)
        dom = ntuple(_ -> (0.0, 1.0), 3)

        prob = Problem(name = "zp", dims = (n, n, n), domain = dom, γ = 5 / 3, bcs = Periodic(),
                       init = (x, y, z) -> (1.0, 0.0, 0.0, 0.0, 1e-6), tfinal = 1.0)
        # f32 FIELDS (production precision), f64 positions/times — the precision policy.
        sim = Simulation(HGMesh((n, n, n), dom), prob; eltype = Float32)
        enable_cosmology!(sim; OmegaMatter = 1.0, OmegaLambda = 0.0, HubbleConstantNow = 0.5,
                          ComovingBoxSize = 64.0, InitialRedshift = zi, FinalRedshift = 4.0,
                          MaxExpansionRate = 0.03, ka = CUDABackend())

        ξx = Float64[]; px = Float64[]; py = Float64[]; pz = Float64[]; vx = Float64[]
        for i in 0:n-1, j in 0:n-1, k in 0:n-1
            ξ = (i + 0.5) / n
            push!(ξx, ξ); push!(px, mod(xmap(ξ, 1.0), 1.0))
            push!(py, (j + 0.5) / n); push!(pz, (k + 0.5) / n)
            push!(vx, vamp(1.0) * sin(kx * ξ))
        end
        np = length(px)
        ps = enable_particles!(sim; px = px, py = py, pz = pz,
                               vx = vx, vy = zeros(np), vz = zeros(np), m = fill(1.0 / np, np))
        sdisp(p) = maximum(abs.(mod.(p .- ξx .+ 0.5, 1.0) .- 0.5))
        disp0 = sdisp(ps.px)

        policy = RefinementPolicy(refine_above = 1.08, max_level = 2, every = 6,
                                  indicator = (s, c) -> particle_density(s, c))
        ns = evolve_cosmology!(sim; a_final = 2.0, policy = policy, regrid_every = 6,
                               ka = CUDABackend())
        af = sim.cosmo.a

        @test eltype(sim.sv[1]) === Float32                # fields are f32
        @test eltype(ps.px) === Float64                    # positions stay f64
        @test af > 1.95                                    # ran to z≈9.5 on the GPU
        @test max_level(sim.backend) >= 1                  # AMR engaged on the sheet
        deposit_particle_density!(sim, ps)
        @test particle_deposited_mass(sim, ps) ≈ sum(ps.m) rtol = 1e-6    # f32 deposit (measured ~7e-9)
        @test isapprox(sdisp(ps.px) / disp0, af; rtol = 0.05)            # D ∝ a (f32: still 2.003 vs 2.0)
        @info "GPU native cosmology (f32 fields)" steps = ns a_final = af max_level = max_level(sim.backend) leaves = n_cells(sim.backend)
    end
end
