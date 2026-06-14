using PowerFoam
using Printf

const GAMMA = 1.4
const N = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 8
const NSTEPS = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 8
const DT = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 0.005

function maybe_backend()
    lowercase(get(ENV, "POWERFOAM_BACKEND", "cpu")) == "metal" || return nothing
    try
        @eval using Metal
        return MetalBackend()
    catch err
        @warn "Metal backend requested but unavailable; using CPU" err
        return nothing
    end
end

function grid_points(n)
    pts = Matrix{Float64}(undef, n * n, 2)
    q = 1
    for j in 1:n, i in 1:n
        pts[q, 1] = (i - 0.5) / n
        pts[q, 2] = (j - 0.5) / n
        q += 1
    end
    return pts
end

function main()
    pts = grid_points(N)
    mesh = power_diagram(PowerSites2D(pts))
    rho = [pts[i, 1] < 0.5 ? 1.0 : 2.0 for i in axes(pts, 1)]
    vx = fill(0.15, length(rho))
    vy = zeros(length(rho))
    state = euler_state_2d(mesh; rho, vx, vy, pressure = 1.0, gamma = GAMMA)
    geom = arepo_mesh_arrays(mesh)
    be = maybe_backend()
    if be !== nothing
        state = to_backend(be, state; T = Float32)
        geom = to_backend(be, geom; T = Float32)
    end

    total0 = total_conserved_2d(state, geom)
    @printf("moving contact: %d cells, steps=%d dt=%.4g backend=%s\n",
            length(rho), NSTEPS, DT, be === nothing ? "cpu" : "metal")
    @printf("%-5s %-10s %-12s %-12s %-12s\n", "step", "cells", "mass_err", "energy_err", "rho_range")

    for step in 1:NSTEPS
        prim = conserved_to_primitive_2d(state; gamma = GAMMA)
        vmesh = hcat(prim.vx, prim.vy)
        moved = moving_mesh_step_2d!(state, mesh; dt = DT, gamma = GAMMA,
                                     mesh_velocity = vmesh, riemann = :hll,
                                     backend = be === nothing ? nothing : be)
        mesh = moved.mesh
        geom = moved.geom
        t = total_conserved_2d(state, geom)
        p = conserved_to_primitive_2d(state; gamma = GAMMA)
        @printf("%-5d %-10d %-12.3e %-12.3e %.5f..%.5f\n",
                step, length(mesh.cells),
                abs(t.mass - total0.mass) / abs(total0.mass),
                abs(t.energy - total0.energy) / abs(total0.energy),
                minimum(p.rho), maximum(p.rho))
    end
end

main()
