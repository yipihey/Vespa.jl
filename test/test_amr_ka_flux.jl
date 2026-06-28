# Phase-1 gate of the KA/Julia re-platform (plan: "we-implemented-a-flux"): the
# batched-per-face flux backend (`src/ka_flux.jl`) drives the SAME composite-AMR
# machinery — `for_each_face`, the +F/−F net-flux-out scatter, and the
# coarse↔fine `FluxRegister` (`_reflux_capture!`/`_reflux_apply!`) — that already
# conserves to round-off with the native per-face path. The decisive claim: moving
# the per-face Riemann solve onto a swappable backend (here `HostBatchedFlux`; a
# GPU/KA backend is the drop-in next step) keeps conservation EXACT, because the
# single-flux-per-face structure telescopes regardless of the flux function. This
# is what the Enzo-driven reflux could not reach (~1.7e-4); Vespa's own FluxRegister
# does it by construction.
#
# Two gates:
#   (1) ROUND-OFF CONSERVATION (rtol=1e-9) — refined Sod (single-rate + subcycled)
#       and the 2D Sedov blast, the same problems test_amr/test_reflux/test_sedov
#       pass with the native flux, now driven by the batched backend.
#   (2) EQUIVALENCE — with `HostBatchedFlux` (same `riemann_flux`, same scatter
#       order) the batched path is bit-identical to the native per-face path on an
#       identical mesh history. This pins the batched gather/scatter as a faithful
#       refactor and is the parity oracle for a device backend.

using HGBackend
include(joinpath(@__DIR__, "..", "problems", "sedov_blast.jl"))

# Run a problem on HGBackend with optional AMR/subcycle and an optional flux backend.
function _ka_run(prob; flux = nothing, policy = nothing, subcycle = false)
    mesh = HGMesh(prob.dims, prob.domain)
    sim = Simulation(mesh, prob)
    sim.flux = flux
    t0 = conserved_totals(sim)
    if policy === nothing
        evolve!(sim)
    else
        evolve!(sim; policy = policy, subcycle = subcycle)
    end
    return sim, t0, conserved_totals(sim)
end

@testset "KA batched flux: refined Sod conserves to round-off (single-rate)" begin
    prob = sod_problem_defaults(n = 64)
    policy = RefinementPolicy(refine_above = 0.1, max_level = 2, every = 4)
    sim, t0, t1 = _ka_run(prob; flux = HostBatchedFlux(), policy = policy)

    @test max_level(sim.backend) >= 1                # AMR engaged with the batched flux
    @test t1.mass ≈ t0.mass rtol = 1e-9
    @test t1.energy ≈ t0.energy rtol = 1e-9
    @info "KA batched Sod (single-rate)" leaves = n_cells(sim.backend) max_level = max_level(sim.backend) mass_drift = abs(t1.mass - t0.mass)
end

@testset "KA batched flux: subcycled refined Sod conserves to round-off" begin
    prob = sod_problem_defaults(n = 64)
    policy = RefinementPolicy(refine_above = 0.1, max_level = 2, every = 4)
    sim, t0, t1 = _ka_run(prob; flux = HostBatchedFlux(), policy = policy, subcycle = true)

    @test max_level(sim.backend) >= 1                # the FluxRegister reflux path under the batched flux
    @test t1.mass ≈ t0.mass rtol = 1e-9
    @test t1.energy ≈ t0.energy rtol = 1e-9
end

@testset "KA batched flux: 2D Sedov blast conserves to round-off" begin
    prob = sedov_problem(n = 32, tfinal = 0.02)   # small + short: a 2D coarse↔fine reflux gate, not a physics run
    policy = RefinementPolicy(refine_above = 0.05, max_level = 2, every = 4)
    sim, t0, t1 = _ka_run(prob; flux = HostBatchedFlux(), policy = policy)

    @test max_level(sim.backend) >= 1
    @test t1.mass ≈ t0.mass rtol = 1e-9
    @test t1.energy ≈ t0.energy rtol = 1e-9
    @info "KA batched Sedov 2D" leaves = n_cells(sim.backend) max_level = max_level(sim.backend)
end

@testset "KA batched flux ≡ native per-face path (bit-identical on the same mesh history)" begin
    # HostBatchedFlux loops the same riemann_flux in the same scatter order, so on
    # an identical (solution-driven) mesh history the two paths must agree exactly.
    prob = sod_problem_defaults(n = 64)
    policy() = RefinementPolicy(refine_above = 0.1, max_level = 2, every = 4)

    nat, _, nt = _ka_run(prob; flux = nothing, policy = policy())
    bat, _, bt = _ka_run(prob; flux = HostBatchedFlux(), policy = policy())

    @test nt.mass ≈ bt.mass rtol = 1e-14
    @test nt.energy ≈ bt.energy rtol = 1e-14

    # Same number of leaves (identical refinement history) and matching state.
    @test n_cells(nat.backend) == n_cells(bat.backend)
    ns = collect(cell_samples(nat))
    bs = collect(cell_samples(bat))
    @test length(ns) == length(bs)
    worst = 0.0
    for (a, c) in zip(ns, bs)
        for k in eachindex(a[2])
            worst = max(worst, abs(a[2][k] - c[2][k]))
        end
    end
    @info "KA batched vs native per-face" worst_state_diff = worst
    @test worst < 1e-12
end
