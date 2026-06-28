# Phase 4 (P4): end-to-end cosmological structure formation on the NATIVE KA/Julia
# stack — no Enzo. A dark-matter Zel'dovich pancake on an HGBackend hierarchy is
# evolved by `evolve_cosmology!` (comoving self-gravity + Hubble expansion + KDK
# particles + AMR refinement on DM overdensity). This exercises every native
# subsystem the CICASS run needs — HG hierarchy, composite Poisson, particle PM,
# expansion, conservative remap-on-refine — together. Gates:
#   1. AMR engages on the collapsing sheet (max_level ≥ 1) via the DM-overdensity
#      indicator (the cosmological zoom mechanism).
#   2. The CIC deposit conserves DM mass to round-off on the evolved, refined mesh
#      (through every regrid).
#   3. The displacement grows as the EdS growing mode D ∝ a (linear structure
#      growth), to AMR/CIC tolerance.
# The driver is ready to scale to 128³ + 3 levels; gas hydro + chemistry under one
# clock is the remaining production integration. This is a SMALL/FAST CPU smoke test
# (the real, larger gate runs on the GPU: test/gpu/test_cosmology_native_gpu.jl).

using RefMesh
using HGBackend

@testset "Cosmology P4: native DM structure formation with AMR" begin
    zi, zc, kx, n = 20.0, 1.0, 2π, 8
    Aof(a) = a * (1 + zc) / (1 + zi)
    vamp(a) = -sqrt(2 / 3) * sqrt(a) * (1 + zc) / ((1 + zi) * kx)
    xmap(ξ, a) = ξ - Aof(a) / kx * sin(kx * ξ)
    dom = ntuple(_ -> (0.0, 1.0), 3)

    prob = Problem(name = "zp", dims = (n, n, n), domain = dom, γ = 5 / 3, bcs = Periodic(),
                   init = (x, y, z) -> (1.0, 0.0, 0.0, 0.0, 1e-6), tfinal = 1.0)
    sim = Simulation(HGMesh((n, n, n), dom), prob)
    enable_cosmology!(sim; OmegaMatter = 1.0, OmegaLambda = 0.0, HubbleConstantNow = 0.5,
                      ComovingBoxSize = 64.0, InitialRedshift = zi, FinalRedshift = 4.0,
                      MaxExpansionRate = 0.03)

    # DM particles: Lagrangian lattice displaced to the a=1 Zel'dovich state; total
    # mass = box volume ⇒ mean deposited density = 1 (overdensity = ρ_DM).
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

    # refine where the DM is overdense (the collapsing pancake)
    policy = RefinementPolicy(refine_above = 1.06, max_level = 1, every = 4,
                              indicator = (s, c) -> particle_density(s, c))

    nsteps = evolve_cosmology!(sim; a_final = 1.6, policy = policy, regrid_every = 4)
    af = sim.cosmo.a

    @test af > 1.55                                         # ran to the target
    @test max_level(sim.backend) >= 1                       # AMR engaged on the sheet
    # DM mass conserved by the CIC deposit on the evolved, refined mesh
    deposit_particle_density!(sim, ps)
    @test particle_deposited_mass(sim, ps) ≈ sum(ps.m) rtol = 1e-12
    # growing mode D ∝ a (coarse 8³/CIC ⇒ loose)
    @test isapprox(sdisp(ps.px) / disp0, af; rtol = 0.15)
    @info "native cosmology run" steps = nsteps a_final = af max_level = max_level(sim.backend) leaves = n_cells(sim.backend)
end
