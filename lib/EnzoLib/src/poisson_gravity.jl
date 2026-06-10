# ── Phase C: the production gravity=:julia hook ───────────────────────────────
#
# The slot swaps ONLY the Poisson solve, exactly like the RAMSES gravity slot:
# Enzo keeps the deposit, the root FFT, the parent-potential BC interpolation
# (all inside PrepareDensityField) and the force differencing
# (ComputeAccelerations, via session_gravity_post); the guest solves each
# subgrid's Dirichlet problem with PoissonKernels' W-cycle on the very system
# Enzo's SolveForPotential iterates — certified to 1.6e-15 against it
# (test_gravity_subgrid_slot.jl).

"""
    poisson_gravity_hook(; gravconst = 4π, rtol = 1e-10) -> hook

The `gravity = :julia` EvolveLevel hook: level 0 runs Enzo's own chain (the
root solve is the already-certified FFT); level > 0 deposits + interpolates
the BC through Enzo (`session_prepare_density`), solves every subgrid with
`vcycle_solve!(dirichlet = true)` on `(Σφ−6φ) = gravconst·dx²·GMF`, writes the
potential back, and lets Enzo difference it (`session_gravity_post`).
"""
function poisson_gravity_hook(; gravconst::Real = 4π, rtol::Real = 1e-10)
    return function (h::Handle, level::Integer, dt)
        if level == 0
            session_gravity(h, 0)
            return nothing
        end
        session_prepare_density(h, level)
        n = session_num_grids_on_level(h, level)
        for i in 0:n-1
            g = problem_grid_index_on_level(h, level, i)
            dims = problem_gmf_dims(h, g)
            gmf = problem_get_gravitating_mass(h, g)
            sol = problem_get_potential(h, g)            # parent-interpolated BC + guess
            gd = problem_grid_dims(h, g)
            l, r = problem_grid_edge(h, g)
            dx = (r[1] - l[1]) / (gd[1] - 6)             # 3 ghost zones per side
            hfac = Float64(dims[1] - 1) * Float64(dims[2] - 1) * Float64(dims[3] - 1)
            rhs = zeros(Float64, dims)
            @inbounds for k in 2:dims[3]-1, j in 2:dims[2]-1, ii in 2:dims[1]-1
                rhs[ii, j, k] = hfac * gravconst * dx^2 * gmf[ii, j, k]
            end
            PoissonKernels.vcycle_solve!(sol, rhs; rtol = rtol, maxcycles = 200,
                                         cycle = :W, dirichlet = true)
            problem_set_potential(h, sol, g)
        end
        session_gravity_post(h, level)
        return nothing
    end
end
