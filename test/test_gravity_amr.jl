# Phase 3c — self-gravity under AMR + time subcycling. The wiring (solve_poisson!
# once per root step in _evolve_subcycle!, gravity source threaded through
# step_level!, free-fall limiter in per-level compute_dt) is exercised end to end.
# Oracles:
#   G1  global momentum conserved to round-off on a periodic self-grav box (the
#       discrete zero-net-force property: Σρg·V = 0), through AMR + subcycling.
#   G2  the run completes to tfinal under policy + subcycle with gravity on, stays
#       positive, and refinement actually engaged.
#   G3  φ store survives a regrid (auto-remapped) and the solver re-converges.

using HGBackend

# Periodic self-gravitating box with a central overdensity (2D) that collapses
# and triggers refinement on the density gradient.
function _selfgrav_box(n; A = 0.3, p0 = 0.1, γ = 5 / 3)
    L = 1.0
    function init(x, y, z)
        r2 = (x - 0.5)^2 + (y - 0.5)^2
        # Narrow Gaussian (σ ≈ 0.05 ⇒ ~3 cells at n=64): steep enough that the
        # cell-to-cell relative density jump clears the refine_above threshold.
        ρ = 1.0 + A * exp(-r2 / 0.005)
        return (ρ, 0.0, 0.0, 0.0, p0)
    end
    return Problem(; name = "selfgrav2d", dims = (n, n), domain = ((0.0, L), (0.0, L)),
                   γ = γ, bcs = Periodic(), init = init, tfinal = 0.05, cfl = 0.3)
end

@testset "G1: momentum conserved to round-off (zero net self-force), AMR+subcycle" begin
    prob = _selfgrav_box(64)
    sim = Simulation(HGMesh(prob.dims, prob.domain), prob)
    enable_gravity!(sim; G = 1.0, bcs = Periodic())
    policy = RefinementPolicy(refine_above = 0.03, max_level = 2, every = 4)

    p0 = conserved_totals(sim)
    evolve!(sim; policy = policy, subcycle = true)
    p1 = conserved_totals(sim)

    @test max_level(sim.backend) >= 1                  # AMR engaged
    # Periodic box: gravity exerts no NET force, hydro fluxes are periodic ⇒
    # total momentum stays at its (zero) initial value to round-off. Use absolute
    # tolerance since the totals are ~0.
    @test abs(p1.momentum_x - p0.momentum_x) < 1e-9
    @test abs(p1.momentum_y - p0.momentum_y) < 1e-9
    # Mass is conserved to round-off (reflux + periodic).
    @test p1.mass ≈ p0.mass rtol = 1e-9
    @info "self-grav momentum drift" dpx = abs(p1.momentum_x - p0.momentum_x) dpy = abs(p1.momentum_y - p0.momentum_y)
end

@testset "G2: self-gravitating subcycled run completes, stays positive" begin
    prob = _selfgrav_box(64)
    sim = Simulation(HGMesh(prob.dims, prob.domain), prob)
    enable_gravity!(sim; G = 1.0, bcs = Periodic())
    evolve!(sim; policy = RefinementPolicy(refine_above = 0.03, max_level = 2, every = 4),
            subcycle = true)
    @test sim.t ≈ prob.tfinal rtol = 1e-10
    pos = Ref(true)
    for_each_cell(sim.backend) do c
        W = primitive_at(sim, c)
        (W[1] > 0 && W[5] > 0) || (pos[] = false)
    end
    @test pos[]
end

@testset "G3: φ store survives regrid; solver re-converges" begin
    prob = _selfgrav_box(64)
    sim = Simulation(HGMesh(prob.dims, prob.domain), prob)
    grav = enable_gravity!(sim; G = 1.0, bcs = Periodic(), tol = 1e-10, maxiter = 5000)
    _, rr0 = solve_poisson!(sim, grav)
    @test rr0 < 1e-8
    # Force a regrid (φ is auto-remapped by the backend's AdaptiveField), then
    # re-solve: the solver must still converge on the new (refined) leaf set.
    regrid!(sim, RefinementPolicy(refine_above = 0.03, max_level = 2, every = 1))
    @test max_level(sim.backend) >= 1
    _, rr1 = solve_poisson!(sim, grav)
    @test rr1 < 1e-8
    @info "phi resolve after regrid" relres_before = rr0 relres_after = rr1 leaves = n_cells(sim.backend)
end
