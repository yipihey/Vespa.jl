# HGBackend validation (ADR build step 4 preview): the HierarchicalGrids.jl
# adapter must satisfy the same interface contract as RefMesh and produce the
# same physics on the same problem spec — the oracle/drop-in relationship the
# architecture is built on.

using HGBackend

@testset "HGBackend interface conformance" begin
    make(dims, dom; bcs) = HGMesh(dims, dom)
    # HierarchicalGrids' cell-average field model currently implements indexed
    # access for SoA only (its AoS/Blocked cell-average getindex is unimplemented),
    # so the adapter advertises SoA. RefMesh exercises all three layouts.
    run_interface_conformance(make; label = "HGBackend", layouts = (SoA(),))
end

@testset "Sod shock tube on HGBackend vs exact Riemann" begin
    prob = sod_problem_defaults(n = 256)
    mesh = HGMesh(prob.dims, prob.domain)
    sim = Simulation(mesh, prob)

    t0 = conserved_totals(sim)
    evolve!(sim)
    t1 = conserved_totals(sim)

    @test t1.mass ≈ t0.mass rtol = 1e-12
    @test t1.energy ≈ t0.energy rtol = 1e-12

    l1 = sod_l1_error(sim)
    @info "Sod L1 density error (HGBackend)" cells = 256 l1
    @test l1 < 0.01

    d = dump_fields(sim)
    @test all(d.density .> 0)
    @test all(d.pressure .> 0)
end

@testset "Cross-backend agreement: RefMesh ≡ HGBackend (ADR oracle)" begin
    prob = sod_problem_defaults(n = 256)

    run_on(mesh) = begin
        sim = Simulation(mesh, prob)
        evolve!(sim)
        dump_fields(sim)
    end

    ref = run_on(UniformMesh(prob.dims, prob.domain))
    hg  = run_on(HGMesh(prob.dims, prob.domain))

    # Same spec, same scheme, two independent substrates: results must agree to
    # tight tolerance (timestep sequences are identical given identical CFL state,
    # so this is near-bitwise, allowing only FP reassociation differences).
    @test ref.x ≈ hg.x rtol = 1e-12
    @test ref.density ≈ hg.density rtol = 1e-10
    @test ref.pressure ≈ hg.pressure rtol = 1e-10
    @test ref.velocity_x ≈ hg.velocity_x rtol = 1e-10
end

@testset "Instrumented{HGBackend} per-operation report (P10)" begin
    prob = sod_problem_defaults(n = 64)
    imesh = Instrumented(HGMesh(prob.dims, prob.domain))
    sim = Simulation(imesh, prob)
    evolve!(sim)
    rep = span_report(imesh)
    @test !isempty(rep)
    @test :for_each_cell in first.(rep)
    @info "Instrumented{HGBackend} cost report"
    for row in rep
        @info "  span" name = row[1] calls = row[2] total_ms = round(row[3], digits = 3)
    end
end
