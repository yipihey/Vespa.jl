# The Sod shock tube: the milestone-1 acceptance test for the finite-volume
# scheme. Compared against the exact Riemann solution and checked for
# conservation. Mirrors Enzo's run/Hydro/Hydro-1D/SodShockTube. Runs on RefMesh
# here; the same problem is run on HGBackend in test_hgbackend.jl.

"L1 density error of a 1D run against the exact Riemann solution at its final time."
function sod_l1_error(sim)
    d = dump_fields(sim)
    γ, t = sim.problem.γ, sim.t
    WL = (1.0, 0.0, 1.0)        # (ρ, u, p) left
    WR = (0.125, 0.0, 0.1)      # right
    l1 = 0.0
    for i in eachindex(d.x)
        ρ_exact = exact_riemann_sample(WL, WR, γ, (d.x[i] - 0.5) / t)[1]
        l1 += abs(d.density[i] - ρ_exact)
    end
    return l1 / length(d.x)
end

@testset "Sod shock tube on RefMesh vs exact Riemann solution" begin
    prob = sod_problem_defaults(n = 256)
    mesh = UniformMesh(prob.dims, prob.domain)
    sim = Simulation(mesh, prob)

    t0 = conserved_totals(sim)
    evolve!(sim)
    t1 = conserved_totals(sim)

    # Waves never reach the (quiescent) outflow boundaries by t=0.2, so mass and
    # energy are conserved to round-off. Total momentum is NOT conserved: the
    # static pressure difference across the ends exerts a net force — physical.
    @test t1.mass ≈ t0.mass rtol = 1e-12
    @test t1.energy ≈ t0.energy rtol = 1e-12

    l1 = sod_l1_error(sim)
    @info "Sod L1 density error (RefMesh)" cells = 256 l1
    @test l1 < 0.01

    d = dump_fields(sim)
    @test all(d.density .> 0)
    @test all(d.pressure .> 0)
    @test maximum(d.density) ≤ 1.0 + 1e-6
    @test minimum(d.density) ≥ 0.125 - 1e-6
end
