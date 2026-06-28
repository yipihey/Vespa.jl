# Geometric-multigrid-preconditioned Poisson (opt-in `enable_gravity!(...; precond=:mg)`).
# A cell-centered V-cycle on the uniform base grid preconditions the composite CG: it
# converges to the SAME φ as plain CG (the outer operator A = −∇²·V is unchanged) but in
# a number of iterations nearly INDEPENDENT of the condition number — the multigrid win.
# On a broadband (CG-hard) source plain CG needs ~hundreds of iterations while MG-PCG
# needs ~8. Gate: MG-PCG ≡ plain CG to solver tolerance AND ≥3× fewer iterations, on
# CPU-KA always and CUDA when present. (Refined meshes fall back to plain CG until the
# AMR V-cycle is hardened — see CONTINUE_HERE.md.)
#
# Run: `<julia> --project=test/gpu test/gpu/test_mg_poisson_gpu.jl`

using Test
using MeshInterface
using RefMesh
using Vespa
using HGBackend
using KernelAbstractions
using CUDA
using Random

const MGDOM = ntuple(_ -> (0.0, 1.0), 3)
_mgprob(n) = Problem(name = "mg", dims = (n, n, n), domain = MGDOM, γ = 5/3,
                     bcs = Periodic(), init = (x,y,z) -> (1.0, 0.0, 0.0, 0.0, 1.0), tfinal = 1.0)

# Solve with a deterministic BROADBAND density (seeded white noise) injected into the gas
# state — this excites all eigenmodes, so unpreconditioned CG converges only at ~√κ rate.
function _mg_solve(n; ka, precond)
    sim = Simulation(HGMesh((n, n, n), MGDOM), _mgprob(n))
    g = enable_gravity!(sim; G = 1.0, tol = 1e-8, ka = ka, precond = precond)
    rng = MersenneTwister(7); dv = sim.sv[density_index(sim.model)]
    vals = Float64[]; for_each_cell(sim.backend) do c; push!(vals, 1.0 + 0.5*randn(rng)); return nothing; end
    i = 0; for_each_cell(sim.backend) do c; i += 1; dv[c] = eltype(dv)(vals[i]); return nothing; end
    it, rr = solve_poisson!(sim, g)
    phi = Float64[]
    for_each_cell(sim.backend) do c; push!(phi, Float64(g.phiv[c])); return nothing; end
    return phi, it, rr
end

@testset "MG-preconditioned Poisson (CPU + CUDA)" begin
    backends = CUDA.functional() ? [("KA-CPU", CPU()), ("CUDA", CUDABackend())] : [("KA-CPU", CPU())]
    CUDA.functional() && @info "CUDA device" name = CUDA.name(CUDA.device())
    n = 32
    for (bname, be) in backends
        pc, itc, rrc = _mg_solve(n; ka = be, precond = :none)
        pm, itm, rrm = _mg_solve(n; ka = be, precond = :mg)
        @test rrm <= 1e-8                                  # MG-PCG converged to tolerance
        @test maximum(abs.(pc .- pm)) < 1e-6               # same φ as plain CG (same operator)
        @test 3 * itm < itc                                # multigrid: ≥3× fewer iterations
        @info "MG vs CG" backend = bname iters_cg = itc iters_mg = itm maxdiff = maximum(abs.(pc .- pm))
    end
end

# A REFINED mesh: the composite-smoothed cycle (damped-Jacobi on the real CSR operator +
# base V-cycle) must converge robustly under AMR and match plain CG — the AMR-hardening gate.
function _mg_solve_amr(; ka, precond)
    m = HGMesh((16, 16, 16), MGDOM)
    tr = Any[]
    for_each_cell(m) do c
        all(0.3 .< cell_center(m, c) .< 0.7) && push!(tr, c); return nothing
    end
    refine!(m, tr)
    sim = Simulation(m, _mgprob(16))
    g = enable_gravity!(sim; G = 1.0, tol = 1e-7, ka = ka, precond = precond)
    rng = MersenneTwister(3); dv = sim.sv[density_index(sim.model)]
    vals = Float64[]; for_each_cell(sim.backend) do c; push!(vals, 1.0 + 0.5*randn(rng)); return nothing; end
    i = 0; for_each_cell(sim.backend) do c; i += 1; dv[c] = eltype(dv)(vals[i]); return nothing; end
    it, rr = solve_poisson!(sim, g)
    phi = Float64[]; for_each_cell(sim.backend) do c; push!(phi, Float64(g.phiv[c])); return nothing; end
    return phi, it, rr, n_cells(sim.backend)
end

@testset "MG under AMR: robust + matches CG (CPU + CUDA)" begin
    backends = CUDA.functional() ? [("KA-CPU", CPU()), ("CUDA", CUDABackend())] : [("KA-CPU", CPU())]
    for (bname, be) in backends
        pc, itc, _, ncell = _mg_solve_amr(; ka = be, precond = :none)
        pm, itm, rrm, _   = _mg_solve_amr(; ka = be, precond = :mg)
        @test ncell > 16^3                                 # genuinely refined
        @test rrm <= 1e-7                                  # converged (NOT maxiter) — the robustness gate
        @test itm < 40                                     # well short of maxiter; far fewer than CG
        @test maximum(abs.(pc .- pm)) < 1e-5               # same φ as plain CG under AMR
        @info "AMR MG vs CG" backend = bname leaves = ncell iters_cg = itc iters_mg = itm maxdiff = maximum(abs.(pc .- pm))
    end
end
