# Phase 1 of AMR time subcycling (composite-grid, ADR P1 science layer): each
# refinement level advances at its own CFL dt, finer levels substep to catch up
# (`EvolveLevel.C`/`SetLevelTimeStep.C`). This is opt-in (`evolve!(...; subcycle=
# true)`) and a PERFORMANCE change — it is not yet conservative at coarse↔fine
# interfaces (the Phase-2 flux register closes that), so these tests assert the
# invariants that must hold NOW: a uniform mesh is a verified no-op, the substep
# counts match the refinement-in-time ratio, and global conservation is tracked.

using HGBackend

@testset "Subcycling no-op on a single-level mesh (RefMesh ≡ subcycle path)" begin
    # On a uniform mesh there is only level 0: subcycling collapses to one root
    # step per global step, so the result must match the single-rate path.
    prob = sod_problem_defaults(n = 128)

    flat = begin
        sim = Simulation(UniformMesh(prob.dims, prob.domain), prob)
        evolve!(sim)                       # single-rate (default)
        dump_fields(sim)
    end
    sub = begin
        sim = Simulation(UniformMesh(prob.dims, prob.domain), prob)
        evolve!(sim; subcycle = true)      # subcycled, but only one level exists
        dump_fields(sim)
    end

    @test flat.x ≈ sub.x rtol = 1e-12
    @test flat.density ≈ sub.density rtol = 1e-12
    @test flat.pressure ≈ sub.pressure rtol = 1e-12
    @test flat.velocity_x ≈ sub.velocity_x rtol = 1e-12
end

@testset "Subcycling no-op on a uniform HGMesh" begin
    prob = sod_problem_defaults(n = 128)
    sim = Simulation(HGMesh(prob.dims, prob.domain), prob)
    evolve!(sim; subcycle = true)
    @test max_level(sim.backend) == 0          # never refined → still one level
    # Compare to the single-rate HG run on the same spec.
    ref = Simulation(HGMesh(prob.dims, prob.domain), prob); evolve!(ref)
    a = dump_fields(sim); b = dump_fields(ref)
    @test a.density ≈ b.density rtol = 1e-12
    @test a.pressure ≈ b.pressure rtol = 1e-12
end

@testset "Refinement-in-time ratio: factor-2 level → 2 substeps" begin
    # Build a static 2-level mesh (refine the left half of a 32-cell base grid)
    # and check that level 1 takes exactly the substeps needed to catch the
    # level-0 step. At matched wave speeds, a 2× finer cell has ~2× smaller CFL
    # dt, so evolve_level!(sim, 0, dt0) drives level 1 with n = ceil(dt0/dt1).
    prob = sod_problem_defaults(n = 32)
    mesh = HGMesh(prob.dims, prob.domain)
    sim = Simulation(mesh, prob)

    # Refine every level-0 leaf in the left half once.
    to_refine = Any[]
    for_each_cell(mesh) do c
        cell_center(mesh, c)[1] < 0.5 && push!(to_refine, c)
    end
    refine!(mesh, to_refine)
    @test max_level(mesh) == 1

    dt0 = compute_dt(sim; level = 0)
    dt1 = compute_dt(sim; level = 1)
    @test isfinite(dt0) && isfinite(dt1)
    # finer cells are half-width ⇒ stable dt roughly halved (allow scheme slack).
    @test dt1 < dt0
    @test 1.6 < dt0 / dt1 < 2.4

    # Drive level 1 directly with the coarse target dt0: it must subcycle into
    # n = ceil(dt0/dt1) substeps to catch up. evolve_level! returns the substep
    # size, so dt0/dt_sub is the refinement-in-time ratio (≈2 here). (Calling on
    # level 0 would return dt0 itself with n=1 — the level-1 subcycling happens
    # inside that recursion, not in its return value.)
    expected_n = max(1, ceil(Int, dt0 / dt1 * (1 - 1e-12)))
    dt_sub = evolve_level!(sim, 1, dt0)
    n = round(Int, dt0 / dt_sub)
    @test n == expected_n
    @test n >= 2                                  # factor-2 refinement actually subcycled
end

@testset "Subcycled run advances and tracks conservation (pre-reflux)" begin
    # A dynamically-refined Sod run under subcycling. Mass conservation is NOT
    # yet exact at coarse↔fine interfaces (Phase 2), so we only assert the run
    # advances to tfinal, stays positive, and mass drift is bounded/small — the
    # tight rtol=1e-9 conservation oracle is the Phase-2 gate.
    prob = sod_problem_defaults(n = 64)
    mesh = HGMesh(prob.dims, prob.domain)
    sim = Simulation(mesh, prob)
    policy = RefinementPolicy(refine_above = 0.1, max_level = 2, every = 4)

    m0 = conserved_totals(sim).mass
    evolve!(sim; policy = policy, subcycle = true)
    m1 = conserved_totals(sim).mass

    @test sim.t ≈ prob.tfinal rtol = 1e-10
    @test max_level(mesh) >= 1
    pos = Ref(true)
    for_each_cell(mesh) do c
        W = primitive_at(sim, c)
        (W[1] > 0 && W[5] > 0) || (pos[] = false)
    end
    @test pos[]
    # Bounded drift now (outflow BCs let some mass leave); Phase 2 tightens this.
    @test isfinite(m1)
    @test abs(m1 - m0) / m0 < 0.05
end
