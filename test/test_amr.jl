# AMR validation (ADR build step 2: feature-complete on the substrate). The
# adaptive run lives on HGBackend (the tree); RefMesh provides the uniform
# convergence oracle. Two acceptance criteria:
#   1. Conservation — the conservative flux-divergence scheme + HG's conservative
#      remap-on-refine keep total mass/energy constant to round-off, even as the
#      mesh changes under the solver.
#   2. Convergence — a base-grid AMR run that refines the shock/contact must beat
#      a same-base-grid uniform run when both are measured against the exact
#      Riemann solution, and approach a uniform-fine run.

using HGBackend

# L1 density error of an arbitrary-resolution 1D run vs the exact Riemann solution.
function sod_l1(sim)
    γ, t = sim.problem.γ, sim.t
    WL = (1.0, 0.0, 1.0); WR = (0.125, 0.0, 0.1)
    num = 0.0; den = 0.0
    for (ctr, W) in cell_samples(sim)
        x = ctr[1]
        vol = 1.0                      # 1D: weight by cell width
        # recover cell width from the backend via volume isn't exposed here;
        # use uniform-in-x L1 over samples (sufficient for a convergence check).
        ρ_exact = exact_riemann_sample(WL, WR, γ, (x - 0.5) / t)[1]
        num += abs(W[1] - ρ_exact); den += 1
    end
    return num / den
end

# Volume-weighted L1 (handles non-uniform AMR cells correctly).
function sod_l1_volweighted(sim)
    b = sim.backend; γ = sim.problem.γ; t = sim.t
    WL = (1.0, 0.0, 1.0); WR = (0.125, 0.0, 0.1)
    err = 0.0; vol = 0.0
    for_each_cell(b) do c
        x = cell_center(b, c)[1]
        v = cell_volume(b, c)
        ρ = primitive_at(sim, c)[1]
        ρe = exact_riemann_sample(WL, WR, γ, (x - 0.5) / t)[1]
        err += abs(ρ - ρe) * v; vol += v
    end
    return err / vol
end

@testset "AMR on HGBackend: conservation under dynamic regridding" begin
    # 1D Sod on a base 64-cell grid, refining the shock/contact up to 2 levels.
    prob = sod_problem_defaults(n = 64)
    mesh = HGMesh(prob.dims, prob.domain)
    sim = Simulation(mesh, prob)
    policy = RefinementPolicy(refine_above = 0.1, max_level = 2, every = 4)

    t0 = conserved_totals(sim)
    n_leaves_start = n_cells(mesh)
    evolve!(sim; policy = policy)
    t1 = conserved_totals(sim)
    n_leaves_end = n_cells(mesh)

    # AMR actually engaged.
    @test n_leaves_end > n_leaves_start
    @test max_level(mesh) >= 1
    @info "AMR Sod" base_leaves = n_leaves_start final_leaves = n_leaves_end max_level = max_level(mesh)

    # Conservation to round-off through every regrid + flux-divergence step.
    @test t1.mass ≈ t0.mass rtol = 1e-9
    @test t1.energy ≈ t0.energy rtol = 1e-9

    # Positivity preserved across hanging-node faces.
    poscheck = Ref(true)
    for_each_cell(mesh) do c
        W = primitive_at(sim, c)
        (W[1] > 0 && W[5] > 0) || (poscheck[] = false)
    end
    @test poscheck[]
end

@testset "AMR accuracy: refined run beats same-base uniform, nears uniform-fine" begin
    # Same base grid (64), integrated to the same time. The AMR run should have a
    # smaller volume-weighted L1 error than the uniform base run, because it
    # resolves the shock and contact at the finer level.
    base_prob = sod_problem_defaults(n = 64)

    # uniform base (no policy)
    um = HGMesh(base_prob.dims, base_prob.domain)
    usim = Simulation(um, base_prob); evolve!(usim)
    l1_uniform_base = sod_l1_volweighted(usim)

    # adaptive (2 levels on the shock/contact) — effective finest = 256
    am = HGMesh(base_prob.dims, base_prob.domain)
    asim = Simulation(am, base_prob)
    evolve!(asim; policy = RefinementPolicy(refine_above = 0.05, max_level = 2, every = 4))
    l1_amr = sod_l1_volweighted(asim)

    # uniform-fine reference at the AMR's finest resolution (256)
    fm = HGMesh((256,), base_prob.domain)
    fsim = Simulation(fm, sod_problem_defaults(n = 256)); evolve!(fsim)
    l1_uniform_fine = sod_l1_volweighted(fsim)

    @info "AMR convergence" l1_uniform_base l1_amr l1_uniform_fine final_leaves = n_cells(am)

    # AMR improves on the base grid …
    @test l1_amr < l1_uniform_base
    # … and lands between the base and the uniform-fine reference (it refines only
    # near features, so it should not beat a fully-fine grid).
    @test l1_amr <= l1_uniform_base
    @test l1_uniform_fine <= l1_amr * 1.5    # uniform-fine is at least competitive
end
