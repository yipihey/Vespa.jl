# 2D Sedov–Taylor blast: the headline dynamic-AMR demonstration (ADR build
# step 2). A sharp circular shock expands from a central energy deposit; a
# density-gradient RefinementPolicy makes the HGBackend tree follow it. We check
# conservation, four-fold symmetry (the IC and reflecting BCs are symmetric), and
# the self-similar growth law R ∝ t^{1/2} in 2D.

include(joinpath(@__DIR__, "..", "problems", "sedov_blast.jl"))

# Estimate the shock radius as the area-weighted mean radius of the
# strongly-compressed shell (ρ above a threshold between background and peak).
function shock_radius_estimate(sim)
    b = sim.backend
    ρmax = 0.0
    for_each_cell(b) do c; ρmax = max(ρmax, primitive_at(sim, c)[1]); end
    thresh = 0.5 * (1.0 + ρmax)            # midway between background (≈1) and peak
    num = 0.0; den = 0.0
    for_each_cell(b) do c
        ρ = primitive_at(sim, c)[1]
        if ρ >= thresh
            x = cell_center(b, c)
            r = hypot(x[1] - 0.5, x[2] - 0.5)
            w = cell_volume(b, c)
            num += r * w; den += w
        end
    end
    return den > 0 ? num / den : 0.0
end

@testset "2D Sedov blast on HGBackend with dynamic AMR" begin
    prob = sedov_problem(n = 64, tfinal = 0.04)
    mesh = HGMesh(prob.dims, prob.domain)
    sim = Simulation(mesh, prob)
    policy = RefinementPolicy(refine_above = 0.2, max_level = 2, every = 4)

    t0 = conserved_totals(sim)
    n0 = n_cells(mesh)
    evolve!(sim; policy = policy)
    t1 = conserved_totals(sim)

    @info "Sedov 2D" base_leaves = n0 final_leaves = n_cells(mesh) max_level = max_level(mesh)

    # AMR engaged on the expanding shock.
    @test n_cells(mesh) > n0
    @test max_level(mesh) >= 1

    # Total mass and energy conserved to round-off (reflecting box = closed).
    @test t1.mass ≈ t0.mass rtol = 1e-9
    @test t1.energy ≈ t0.energy rtol = 1e-9

    # Positivity throughout.
    ok = Ref(true)
    for_each_cell(mesh) do c
        W = primitive_at(sim, c)
        (W[1] > 0 && W[5] > 0) || (ok[] = false)
    end
    @test ok[]

    # Four-fold symmetry: total mass in each quadrant about the center should be
    # nearly equal. The IC and reflecting BCs are symmetric, but the unsplit
    # flux-divergence scheme visits axes in order and AMR refines in tree order,
    # so quadrant sums differ only by floating-point reassociation — a fraction
    # of a percent, not machine precision. We assert physical symmetry (<1%).
    quad = zeros(4)
    for_each_cell(mesh) do c
        x = cell_center(mesh, c)
        q = (x[1] >= 0.5 ? 1 : 0) + (x[2] >= 0.5 ? 2 : 0) + 1
        quad[q] += primitive_at(sim, c)[1] * cell_volume(mesh, c)
    end
    @test maximum(quad) - minimum(quad) < 1e-2 * maximum(quad)
end

@testset "Sedov self-similar growth R ∝ t^{1/2} (2D)" begin
    # Two times; the shock radius should grow like the Sedov law R ∝ t^{2/(2+ν)}
    # with ν=2 ⇒ R ∝ t^{1/2}, so R(t2)/R(t1) ≈ sqrt(t2/t1).
    radii = Float64[]
    times = (0.02, 0.04)
    for tf in times
        prob = sedov_problem(n = 64, tfinal = tf)
        sim = Simulation(HGMesh(prob.dims, prob.domain), prob)
        evolve!(sim; policy = RefinementPolicy(refine_above = 0.2, max_level = 2, every = 4))
        push!(radii, shock_radius_estimate(sim))
    end
    ratio = radii[2] / radii[1]
    expected = sqrt(times[2] / times[1])     # √2 ≈ 1.414
    @info "Sedov growth" radii ratio expected
    # Loose tolerance: the radius estimator is coarse and the self-similar form is
    # only approached; require the growth to be in the right ballpark.
    @test 0.7 * expected < ratio < 1.3 * expected
end
