# Phase 2 (P2.1): dark-matter particle self-gravity on the composite mesh — CPU
# correctness path. Particles are CIC-deposited to a density field added to the
# existing across-level composite Poisson RHS (src/gravity.jl), and the resulting
# g = −∇φ is interpolated back (same CIC kernel) to push them (KDK leapfrog).
#
# Four gates establish correctness before the AMR point-location (P2.1d) and the
# KA/GPU port (P2.3):
#   1. CIC deposit conserves mass (Σ ρ_p·V == Σ m_p) to round-off.
#   2. A particle's potential equals that of the equivalent gas overdensity (the
#      deposit feeds the SAME Poisson source) — bit-identical φ.
#   3. Momentum conservation under KDK pushes (the PM property: same CIC kernel for
#      deposit + interp, antisymmetric −∇φ ⇒ Σ m_p g_p = 0) to round-off.
#   4. A two-body pair forms a BOUND orbit (attractive gravity; separation orbits,
#      does not escape).

using RefMesh
using HGBackend
using Random

_pbox(dims) = Problem(name = "pbox", dims = dims, domain = ntuple(_ -> (0.0, 1.0), 3),
                      γ = 5 / 3, bcs = Periodic(),
                      init = (x, y, z) -> (1e-8, 0.0, 0.0, 0.0, 1e-8), tfinal = 1.0)

@testset "Particles P2.1: DM self-gravity (CPU, uniform mesh)" begin
    dom = ntuple(_ -> (0.0, 1.0), 3)

    @testset "CIC deposit conserves mass" begin
        dims = (16, 16, 16)
        sim = Simulation(UniformMesh(dims, dom), _pbox(dims))
        np = 500
        m = 0.1 .+ collect(range(0.0, 1.0; length = np))
        ps = enable_particles!(sim; px = collect(range(0.01, 0.99; length = np)),
                               py = collect(range(0.02, 0.97; length = np)),
                               pz = collect(range(0.03, 0.95; length = np)), m = m)
        deposit_particle_density!(sim, ps)
        @test particle_deposited_mass(sim, ps) ≈ sum(m) rtol = 1e-12
    end

    @testset "particle φ == equivalent-gas φ" begin
        dims = (16, 16, 16)
        cellctr = ((4 + 0.5) / dims[1], (7 + 0.5) / dims[2], (9 + 0.5) / dims[3])
        M = 3.0; Vc = prod(1.0 ./ dims)

        simA = Simulation(UniformMesh(dims, dom), _pbox(dims))
        gA = enable_gravity!(simA; G = 1.0)
        di = density_index(simA.model)
        for_each_cell(simA.backend) do c
            ctr = cell_center(simA.backend, c)
            if all(abs.(ctr .- cellctr) .< 1e-9)
                U = Vespa.get_U(simA.sv, c)
                Vespa.set_U!(simA.sv, c, (U[1] + M / Vc, U[2], U[3], U[4], U[5]))
            end
            return nothing
        end
        solve_poisson!(simA, gA)

        simB = Simulation(UniformMesh(dims, dom), _pbox(dims))
        gB = enable_gravity!(simB; G = 1.0)
        enable_particles!(simB; px = [cellctr[1]], py = [cellctr[2]], pz = [cellctr[3]], m = [M])
        solve_poisson!(simB, gB)

        maxd = Ref(0.0)
        for_each_cell(simA.backend) do c
            maxd[] = max(maxd[], abs(gA.phiv[c] - gB.phiv[c]))
            return nothing
        end
        @test maxd[] < 1e-10           # same RHS ⇒ same φ (CG solved to tol)
    end

    @testset "momentum conservation under KDK push" begin
        dims = (16, 16, 16)
        sim = Simulation(UniformMesh(dims, dom), _pbox(dims))
        enable_gravity!(sim; G = 1.0)
        np = 600
        rng(a, b, n) = collect(range(a, b; length = n))
        # Bulk drift (0.05) so the NET momentum is clearly nonzero; internal
        # self-gravity cannot change it, so it must be conserved to round-off.
        ps = enable_particles!(sim; px = rng(0.05, 0.95, np), py = rng(0.07, 0.93, np),
                               pz = rng(0.04, 0.96, np),
                               vx = 0.05 .+ 0.01 .* rng(-1, 1, np), vy = 0.01 .* rng(1, -1, np),
                               vz = 0.01 .* rng(-0.5, 0.5, np), m = fill(1.0 / np, np))
        solve_poisson!(sim, sim.grav)
        p0 = particle_momentum(ps)
        for _ in 1:8
            push_particles!(sim, 0.005)
        end
        p1 = particle_momentum(ps)
        drift = maximum(abs.(p1 .- p0))
        scale = maximum(abs.(p0))      # net px-momentum ≈ 0.05·Σm = 0.05
        @test drift / scale < 1e-6     # PM momentum conservation (measured ~1e-10)
    end

    @testset "two-body bound orbit" begin
        dims = (32, 32, 32)
        sim = Simulation(UniformMesh(dims, dom), _pbox(dims))
        enable_gravity!(sim; G = 1.0)
        ps = enable_particles!(sim; px = [0.42, 0.58], py = [0.5, 0.5], pz = [0.5, 0.5],
                               vx = [0.0, 0.0], vy = [0.25, -0.25], vz = [0.0, 0.0],
                               m = [0.05, 0.05])
        solve_poisson!(sim, sim.grav)
        sep0 = abs(ps.px[1] - ps.px[2])
        seps = Float64[]
        for _ in 1:60
            push_particles!(sim, 0.004)
            push!(seps, abs(ps.px[1] - ps.px[2]))
        end
        @test maximum(seps) <= 1.25 * sep0     # stays bound (does not escape)
        @test minimum(seps) < 0.5 * sep0       # orbits inward (gravity is attractive)
        @test all(isfinite, seps)
    end

    # P2.1d: the cloud-overlap deposit/interp on a GENUINELY REFINED mesh. Mass is
    # conserved EXACTLY (overlap tiling is gap-free across level jumps); momentum is
    # conserved to the AMR hanging-node level (not round-off — summation-by-parts is
    # inexact at coarse↔fine faces), still small.
    @testset "deposit + push on a refined (multi-level) mesh" begin
        mesh = HGMesh((16, 16, 16), dom)
        # central octant → level 1, inner core → level 2
        tr = Any[]
        for_each_cell(mesh) do c
            all(0.3 .< cell_center(mesh, c) .< 0.7) && push!(tr, c)
            return nothing
        end
        refine!(mesh, tr)
        tr2 = Any[]
        for_each_cell(mesh) do c
            (level_of(mesh, c) == 1 && all(0.4 .< cell_center(mesh, c) .< 0.6)) && push!(tr2, c)
            return nothing
        end
        refine!(mesh, tr2)
        @test max_level(mesh) == 2

        sim = Simulation(mesh, _pbox((16, 16, 16)))
        enable_gravity!(sim; G = 1.0)
        np = 800
        rng = MersenneTwister(3)
        # volume-filling random positions + a bulk drift (vx≈0.05) so net momentum
        # is well-defined; the cloud spans level boundaries throughout the core.
        ps = enable_particles!(sim; px = rand(rng, np), py = rand(rng, np), pz = rand(rng, np),
                               vx = 0.05 .+ 0.01 .* randn(rng, np),
                               vy = 0.01 .* randn(rng, np), vz = 0.01 .* randn(rng, np),
                               m = fill(1.0 / np, np))
        deposit_particle_density!(sim, ps)
        @test particle_deposited_mass(sim, ps) ≈ sum(ps.m) rtol = 1e-12   # exact across levels

        solve_poisson!(sim, sim.grav)
        p0 = particle_momentum(ps)
        for _ in 1:5
            push_particles!(sim, 0.004)
        end
        p1 = particle_momentum(ps)
        @test maximum(abs.(p1 .- p0)) / maximum(abs.(p0)) < 5e-3          # AMR PM (measured ~6e-4)
    end
end
