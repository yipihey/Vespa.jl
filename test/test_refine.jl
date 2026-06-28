# AMR refinement-indicator tests: Jeans-length + DM-particle-count (ported from Enzo
# CellFlaggingMethod = 6 and = 4). Unit checks of the indicator formulas + selective
# refinement, and a conservation check that regridding preserves totals.
# The Enzo-oracle cross-check (vs enzomodules_flag_jeans) lives in the EnzoLib suite,
# gated on grid_available().

using Vespa, MeshInterface, HGBackend, Test

# A uniform 3-D box; constant primitive state (ρ, 0,0,0, p).
function _uniform_problem(; n = 8, ρ = 1.0, p = 1.0, γ = 5 / 3)
    init(x, y, z) = (ρ, 0.0, 0.0, 0.0, p)
    Problem(; name = "uniform", dims = (n, n, n),
            domain = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
            γ = γ, bcs = Periodic(), init = init, tfinal = 1.0, cfl = 0.4)
end

_first_cell(b) = (out = Ref{Any}(nothing); for_each_cell(b) do c
                      out[] === nothing && (out[] = c); end; out[])

@testset "AMR refinement indicators (Jeans + particle-count)" begin

    @testset "Jeans-length indicator formula" begin
        γ = 5 / 3; ρ = 1.0; p = 1.0; G = 1.0; cpj = 4
        prob = _uniform_problem(n = 8, ρ = ρ, p = p, γ = γ)
        sim  = Simulation(HGMesh(prob.dims, prob.domain), prob)
        c    = _first_cell(sim.backend)
        cs   = sqrt(γ * p / ρ)
        λJ   = cs * sqrt(π / (G * ρ))
        dx   = 1.0 / 8
        @test jeans_length_indicator(sim, c; G = G, cells_per_jeans = cpj) ≈ dx * cpj / λJ
        # a large cs floor lengthens λ_J ⇒ indicator shrinks (cold-gas suppression)
        @test jeans_length_indicator(sim, c; G = G, cells_per_jeans = cpj, cs_min = 1e3) <
              jeans_length_indicator(sim, c; G = G, cells_per_jeans = cpj)
    end

    @testset "Jeans refinement threshold (selective)" begin
        prob = _uniform_problem(n = 8, ρ = 1.0, p = 1.0, γ = 5 / 3)
        # cells_per_jeans=4 ⇒ indicator ≈ 0.22 < 1 ⇒ no refinement
        sim1 = Simulation(HGMesh(prob.dims, prob.domain), prob)
        @test regrid!(sim1, jeans_refinement_policy(sim1; cells_per_jeans = 4,
                      max_level = 1, every = 1, G = 1.0)) == 0
        # cells_per_jeans=40 ⇒ indicator ≈ 2.2 > 1 ⇒ every base cell refines
        sim2 = Simulation(HGMesh(prob.dims, prob.domain), prob)
        n0 = n_cells(sim2.backend)
        nref = regrid!(sim2, jeans_refinement_policy(sim2; cells_per_jeans = 40,
                       max_level = 1, every = 1, G = 1.0))
        @test nref == n0
        @test max_level(sim2.backend) >= 1
    end

    @testset "DM particle-count deposit + indicator" begin
        prob = _uniform_problem(n = 8)
        sim  = Simulation(HGMesh(prob.dims, prob.domain), prob)
        # 5 particles in base cell (0,0,0); 1 in cell (4,4,4)
        clump = [0.03, 0.05, 0.07, 0.09, 0.11]
        px = vcat(clump, [0.56]); py = vcat(clump, [0.56]); pz = vcat(clump, [0.56])
        counts = deposit_particle_counts!(sim, px, py, pz)
        @test counts[(0, 0, 0)] == 5
        @test counts[(4, 4, 4)] == 1
        # indicator reads the count of the base cell containing each leaf
        nflag = 0
        for_each_cell(sim.backend) do c
            v = particle_count_indicator(sim, c; counts = counts)
            v >= 4 && (nflag += 1)
        end
        @test nflag == 1                      # only the clump cell has ≥4
    end

    @testset "particle refinement refines exactly the clump (selective)" begin
        prob = _uniform_problem(n = 8)
        sim  = Simulation(HGMesh(prob.dims, prob.domain), prob)
        clump = [0.03, 0.05, 0.07, 0.09]
        px = vcat(clump, [0.56, 0.58]); py = vcat(clump, [0.56, 0.20]); pz = vcat(clump, [0.56, 0.80])
        pol = particle_refinement_policy(sim, px, py, pz; nmin = 4, max_level = 1, every = 1)
        @test regrid!(sim, pol) == 1          # only the 4-particle base cell refines
        @test max_level(sim.backend) >= 1
    end

    @testset "regrid conserves totals" begin
        prob = _uniform_problem(n = 8, ρ = 1.0, p = 1.0)
        sim  = Simulation(HGMesh(prob.dims, prob.domain), prob)
        t0 = conserved_totals(sim)
        regrid!(sim, jeans_refinement_policy(sim; cells_per_jeans = 40, max_level = 1,
                every = 1, G = 1.0))
        t1 = conserved_totals(sim)
        for (a, b) in zip(values(t0), values(t1))
            @test isapprox(a, b; rtol = 1e-12, atol = 1e-12)
        end
    end
end
