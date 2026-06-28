# GPU + CPU device test for the KA composite Poisson solver (P2.2) — `enable_gravity!
# (...; ka=backend)`. The matrix-free CG of src/gravity.jl runs on a KA backend via a
# CSR-gather matvec (the −∇²·V operator, coarse↔fine sub-faces included), so it stays
# across-level. Gate: the device CG matches the CPU CG to round-off on uniform AND
# refined meshes, on CPU-KA always and CUDA when present — including a DM-particle
# source (tying P2.1's deposit into the KA RHS path).
#
# Run: `<julia> --project=test/gpu test/gpu/test_ka_gravity_gpu.jl`

using Test
using MeshInterface
using RefMesh
using Vespa
using HGBackend
using KernelAbstractions
using CUDA

const DOM = ntuple(_ -> (0.0, 1.0), 3)
# multi-mode density ⇒ the CG actually iterates (not a single-eigenmode 1-step solve)
_init = (x, y, z) -> (1.0 + 0.2 * sin(2π * x) + 0.15 * cos(4π * y) * sin(2π * z), 0.0, 0.0, 0.0, 1.0)
_prob(dims) = Problem(name = "pb", dims = dims, domain = DOM, γ = 5 / 3, bcs = Periodic(),
                      init = _init, tfinal = 1.0)

_uniform() = UniformMesh((24, 24, 24), DOM)
function _refined()
    m = HGMesh((16, 16, 16), DOM)
    tr = Any[]
    for_each_cell(m) do c
        all(0.3 .< cell_center(m, c) .< 0.7) && push!(tr, c)
        return nothing
    end
    refine!(m, tr)
    return m
end

# Solve on `mesh` with the given KA backend (nothing = CPU CG); return φ as a vector
# in leaf order, plus iteration count. `setup!` optionally adds particles.
function _solve_phi(mesh; ka = nothing, setup! = nothing)
    sim = Simulation(mesh, _prob((24, 24, 24)))
    g = enable_gravity!(sim; G = 1.0, tol = 1e-10, ka = ka)
    setup! === nothing || setup!(sim)
    it, rr = solve_poisson!(sim, g)
    phi = Float64[]
    for_each_cell(sim.backend) do c
        push!(phi, Float64(g.phiv[c]))
        return nothing
    end
    return phi, it
end

@testset "KA composite Poisson (CPU + CUDA)" begin
    @testset "uniform: KA-CPU ≡ CPU CG (round-off)" begin
        pc, itc = _solve_phi(_uniform())
        pk, itk = _solve_phi(_uniform(); ka = CPU())
        @test itc > 1                              # multi-mode ⇒ real CG iterations
        @test maximum(abs.(pc .- pk)) < 1e-10
    end

    @testset "refined: KA-CPU ≡ CPU CG (round-off, across-level)" begin
        pc, _ = _solve_phi(_refined())
        pk, _ = _solve_phi(_refined(); ka = CPU())
        @test length(pc) > 16^3                    # genuinely refined
        @test maximum(abs.(pc .- pk)) < 1e-10
    end

    if !CUDA.functional()
        @info "CUDA not functional — CUDA Poisson gates skipped (CPU-KA verified)"
    else
        @info "CUDA device" name = CUDA.name(CUDA.device())
        @testset "uniform: CUDA ≡ CPU CG" begin
            pc, _ = _solve_phi(_uniform())
            pg, _ = _solve_phi(_uniform(); ka = CUDABackend())
            @test maximum(abs.(pc .- pg)) < 1e-9
        end
        @testset "refined: CUDA ≡ CPU CG (across-level)" begin
            pc, _ = _solve_phi(_refined())
            pg, _ = _solve_phi(_refined(); ka = CUDABackend())
            @test maximum(abs.(pc .- pg)) < 1e-9
        end
        @testset "CUDA Poisson with a DM-particle source (P2.1 ⊕ P2.2)" begin
            # Same particle cloud into the CPU CG and the CUDA CG: the deposited DM
            # density flows through _fill_poisson_rhs! into both, so φ must agree.
            addp(sim) = enable_particles!(sim;
                px = collect(range(0.05, 0.95; length = 400)),
                py = collect(range(0.95, 0.05; length = 400)),
                pz = collect(range(0.05, 0.95; length = 400)), m = fill(0.5 / 400, 400))
            pc, _ = _solve_phi(_uniform(); setup! = addp)
            pg, _ = _solve_phi(_uniform(); ka = CUDABackend(), setup! = addp)
            @test maximum(abs.(pc .- pg)) < 1e-9
        end
    end
end
