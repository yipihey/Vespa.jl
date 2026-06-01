# Phase 2 of AMR time subcycling: coarse–fine flux refluxing. Subcycling alone
# breaks conservation at refinement boundaries (the two sides advance with
# different dt); the flux register (src/reflux.jl) records the coarse-flux vs
# time-integrated-fine-flux mismatch on each coarse leaf and applies the
# correction after the fine level subcycles, restoring exact conservation — the
# composite-grid form of Enzo's UpdateFromFinerGrids/CorrectForRefinedFluxes.
#
# Oracle: a Sod run whose waves never reach the (outflow) domain edge conserves
# total mass/energy to round-off. The single-rate path already passes this
# (test_amr); here we assert the SUBCYCLED path does too — the Phase-2 gate.

using HGBackend

@testset "Reflux: subcycled AMR conserves mass/energy to round-off" begin
    # Same setup as the single-rate AMR conservation test, but subcycled. The
    # 1e-9 tolerance is the gate: it only holds if the flux register correctly
    # reconciles coarse and fine fluxes at the interface.
    prob = sod_problem_defaults(n = 64)
    mesh = HGMesh(prob.dims, prob.domain)
    sim = Simulation(mesh, prob)
    policy = RefinementPolicy(refine_above = 0.1, max_level = 2, every = 4)

    t0 = conserved_totals(sim)
    evolve!(sim; policy = policy, subcycle = true)
    t1 = conserved_totals(sim)

    @test max_level(mesh) >= 1                       # AMR + subcycling engaged
    @test t1.mass ≈ t0.mass rtol = 1e-9
    @test t1.energy ≈ t0.energy rtol = 1e-9

    pos = Ref(true)
    for_each_cell(mesh) do c
        W = primitive_at(sim, c)
        (W[1] > 0 && W[5] > 0) || (pos[] = false)
    end
    @test pos[]
    @info "Reflux Sod (subcycled)" leaves = n_cells(mesh) max_level = max_level(mesh) mass_drift = abs(t1.mass - t0.mass)
end

@testset "Reflux: 3-level subcycled run conserves (nested registers)" begin
    # Deeper hierarchy exercises nested registers (a level is simultaneously the
    # fine side of its parent's register and the coarse side of its own).
    prob = sod_problem_defaults(n = 64)
    mesh = HGMesh(prob.dims, prob.domain)
    sim = Simulation(mesh, prob)
    policy = RefinementPolicy(refine_above = 0.05, max_level = 3, every = 3)

    t0 = conserved_totals(sim)
    evolve!(sim; policy = policy, subcycle = true)
    t1 = conserved_totals(sim)

    @test max_level(mesh) >= 2
    @test t1.mass ≈ t0.mass rtol = 1e-9
    @test t1.energy ≈ t0.energy rtol = 1e-9
end

@testset "Reflux: subcycled solution matches single-rate to scheme tolerance" begin
    # Refluxed subcycling and single-rate are different time integrations, so
    # they won't agree to round-off, but they solve the same PDE on the same
    # mesh sequence → the bulk solution must agree to discretization tolerance.
    prob = sod_problem_defaults(n = 64)
    policy() = RefinementPolicy(refine_above = 0.1, max_level = 2, every = 4)

    run(sub) = begin
        sim = Simulation(HGMesh(prob.dims, prob.domain), prob)
        evolve!(sim; policy = policy(), subcycle = sub)
        sim
    end
    flat = run(false)
    subd = run(true)

    # Volume-weighted L1 difference in density between the two runs, sampled on a
    # common uniform grid via exact-Riemann-style binning is overkill; instead
    # compare each against the exact Riemann solution and require the subcycled
    # error to be within ~30% of the single-rate error (same order, no blow-up).
    function l1_vs_exact(sim)
        b = sim.backend; γ = sim.problem.γ; t = sim.t
        WL = (1.0, 0.0, 1.0); WR = (0.125, 0.0, 0.1)
        err = 0.0; vol = 0.0
        for_each_cell(b) do c
            x = cell_center(b, c)[1]; v = cell_volume(b, c)
            ρ = primitive_at(sim, c)[1]
            ρe = exact_riemann_sample(WL, WR, γ, (x - 0.5) / t)[1]
            err += abs(ρ - ρe) * v; vol += v
        end
        return err / vol
    end
    e_flat = l1_vs_exact(flat)
    e_sub  = l1_vs_exact(subd)
    @info "Reflux L1 vs exact" single_rate = e_flat subcycled = e_sub
    @test e_sub < 0.02                                # solution is sound
    @test e_sub < 1.5 * e_flat                        # no accuracy blow-up vs single-rate
end
