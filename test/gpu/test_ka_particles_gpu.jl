# GPU + CPU device test for the KA particle push (P2.3) —
# `push_particles!(sim, dt; ka=backend)`. The KDK kinematics (interp-gather + kicks +
# drift) run on a KA backend; with the Poisson re-solve kept on the CPU CG the device
# push is parity-exact with the CPU push (same f64, same gather order). A second case
# runs the FULLY device step (KA Poisson ⊕ device push) and checks it tracks closely.
#
# Run: `<julia> --project=test/gpu test/gpu/test_ka_particles_gpu.jl`

using Test
using MeshInterface
using RefMesh
using Vespa
using HGBackend
using KernelAbstractions
using CUDA
using Random

const DOM = ntuple(_ -> (0.0, 1.0), 3)
_pbox(dims) = Problem(name = "pbox", dims = dims, domain = DOM, γ = 5 / 3, bcs = Periodic(),
                      init = (x, y, z) -> (1e-8, 0.0, 0.0, 0.0, 1e-8), tfinal = 1.0)

# Build an identical particle sim each call (so CPU and device runs start equal).
function _make(mesh; grav_ka = nothing)
    sim = Simulation(mesh, _pbox((16, 16, 16)))
    enable_gravity!(sim; G = 1.0, ka = grav_ka)
    rng = MersenneTwister(11)
    enable_particles!(sim; px = rand(rng, 600), py = rand(rng, 600), pz = rand(rng, 600),
                      vx = 0.05 .+ 0.01 .* randn(rng, 600), vy = 0.01 .* randn(rng, 600),
                      vz = 0.01 .* randn(rng, 600), m = fill(1.0 / 600, 600))
    solve_poisson!(sim, sim.grav)
    return sim
end

_state(ps) = (copy(ps.px), copy(ps.vx))
_maxdiff(a, b) = max(maximum(abs.(a[1] .- b[1])), maximum(abs.(a[2] .- b[2])))

_umesh() = UniformMesh((16, 16, 16), DOM)
function _amesh()
    m = HGMesh((16, 16, 16), DOM)
    tr = Any[]
    for_each_cell(m) do c
        all(0.3 .< cell_center(m, c) .< 0.7) && push!(tr, c)
        return nothing
    end
    refine!(m, tr)
    return m
end

@testset "KA particle push (CPU + CUDA)" begin
    backends = CUDA.functional() ? [("KA-CPU", CPU()), ("CUDA", CUDABackend())] : [("KA-CPU", CPU())]
    CUDA.functional() && @info "CUDA device" name = CUDA.name(CUDA.device())

    for (mname, meshf) in [("uniform", _umesh), ("refined", _amesh)]
        # CPU reference push (CPU Poisson)
        cpu = _make(meshf())
        for _ in 1:6
            push_particles!(cpu, 0.005)
        end
        ref = _state(cpu.particles)

        for (bname, be) in backends
            # device push, CPU Poisson re-solve ⇒ parity-exact with the CPU push
            dev = _make(meshf())
            for _ in 1:6
                push_particles!(dev, 0.005; ka = be)
            end
            d = _maxdiff(ref, _state(dev.particles))
            @info "device push parity" mesh = mname backend = bname maxdiff = d
            @test d < 1e-11
        end

        if CUDA.functional()
            # fully device: KA Poisson ⊕ device push. φ differs from CPU CG by
            # round-off each solve, so positions track closely but not bit-exactly.
            full = _make(meshf(); grav_ka = CUDABackend())
            for _ in 1:6
                push_particles!(full, 0.005; ka = CUDABackend())
            end
            @test _maxdiff(ref, _state(full.particles)) < 1e-6
        end
    end
end
