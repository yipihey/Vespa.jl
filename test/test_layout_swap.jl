# Layout independence (ADR-0001 P3 + definition-of-done): the same problem spec,
# run only by changing the field layout at the constructor, must produce the same
# numbers. Kernels touch fields exclusively through handle indexing, so SoA and
# Blocked storage are interchangeable.

@testset "SoA ↔ Blocked produce identical results (RefMesh)" begin
    prob = sod_problem_defaults(n = 128)

    run_with(layout) = begin
        mesh = UniformMesh(prob.dims, prob.domain)
        sim = Simulation(mesh, prob; layout = layout)
        evolve!(sim)
        dump_fields(sim)
    end

    a = run_with(SoA())
    b = run_with(Blocked{4}())

    @test a.density ≈ b.density rtol = 1e-13
    @test a.pressure ≈ b.pressure rtol = 1e-13
    @test a.velocity_x ≈ b.velocity_x rtol = 1e-13
end
