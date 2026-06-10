# ── Phase C: the SUBGRID (level>0) gravity solve, certified vs live Enzo ─────
#
# The recorded "wire the level>0 hook" step: Enzo's own subgrid gravity chain
# (PrepareDensityField interpolates the parent potential into the subgrid's
# PotentialField = the Dirichlet BC + initial guess; SolveForPotential then
# iterates MG on  L·φ = α·GMF) is replicated with PoissonKernels'
# `vcycle_solve!(dirichlet = true)` on the SAME inputs read through the bridge.
#
# Two independent gates, the framework's standard idiom:
#   1. fit the scalar α in  (Σφ−6φ) = α·GMF  from Enzo's OWN converged solution
#      — the fit residual certifies the system replication without our solver;
#   2. the KA Dirichlet V-cycle from the same BC + rhs lands on Enzo's φ to
#      Enzo's own MG residual level.
#
# Fixture: GravityTest (ProblemType 23) — 32³ root + a nested subgrid + 5000
# particles, SelfGravity on; PotentialIterations raised so the oracle is
# deeply converged.

using Test
using EnzoLib
import PoissonKernels

const GRAV_PF_SRC = normpath(joinpath(@__DIR__, "..", "..", "..", "..",
                                      "run", "GravitySolver", "GravityTest", "GravityTest.enzo"))

@testset "Phase C: subgrid gravity slot vs live Enzo SolveForPotential" begin
    if !(EnzoLib.grid_available() && isfile(GRAV_PF_SRC))
        @test_skip false
    else
        dir = mktempdir()
        par = read(GRAV_PF_SRC, String)
        par = replace(par, r"StopTime\s*=\s*\S+" => "StopTime = 1.0")  # let particles MOVE
        par *= "\nPotentialIterations = 30\n"        # converge the ORACLE deeply
        pf = joinpath(dir, "GravityTest.enzo")
        write(pf, par)
        cd(dir) do
            h = EnzoLib.session_init(pf)
            h == C_NULL && error("session_init failed for GravityTest")
            try
                n1 = EnzoLib.session_num_grids_on_level(h, 1)
                @test n1 >= 1                          # the nested subgrid exists
                # the EvolveLevel order: root gravity first (the parent potential
                # the subgrid BC interpolates from), then the level-1 chain
                EnzoLib.session_gravity(h, 0)
                EnzoLib.session_prepare_density(h, 1)  # deposit + parent-interpolated φ (the BC)
                g = EnzoLib.problem_grid_index_on_level(h, 1, 0)
                dims = EnzoLib.problem_gmf_dims(h, g)
                gmf = EnzoLib.problem_get_gravitating_mass(h, g)
                phi_pre = EnzoLib.problem_get_potential(h, g)
                # ── ORACLE: Enzo's own subgrid multigrid ──────────────────────
                EnzoLib.session_gravity(h, 1)
                phi_post = EnzoLib.problem_get_potential(h, g)
                # ── gate 1: fit α in (Σφ − 6φ) = α·GMF on Enzo's solution ─────
                nx, ny, nz = dims
                num = 0.0; den = 0.0
                lap = zeros(dims)
                for k in 2:nz-1, j in 2:ny-1, i in 2:nx-1
                    lap[i, j, k] = (phi_post[i-1, j, k] + phi_post[i+1, j, k] +
                                    phi_post[i, j-1, k] + phi_post[i, j+1, k] +
                                    phi_post[i, j, k-1] + phi_post[i, j, k+1] -
                                    6 * phi_post[i, j, k])
                    num += lap[i, j, k] * gmf[i, j, k]
                    den += gmf[i, j, k]^2
                end
                alpha = num / den
                resid = 0.0; scale = 0.0
                for k in 2:nz-1, j in 2:ny-1, i in 2:nx-1
                    resid = max(resid, abs(lap[i, j, k] - alpha * gmf[i, j, k]))
                    scale = max(scale, abs(alpha * gmf[i, j, k]))
                end
                rel_fit = resid / scale
                @test rel_fit < 1e-3                  # Enzo's φ satisfies OUR operator
                # ── gate 2: the KA Dirichlet V-cycle on the same BC + rhs ─────
                hfac = Float64(nx - 1) * Float64(ny - 1) * Float64(nz - 1)
                rhs = zeros(dims)
                for k in 2:nz-1, j in 2:ny-1, i in 2:nx-1
                    rhs[i, j, k] = hfac * alpha * gmf[i, j, k]
                end
                sol = copy(phi_pre)                   # faces = the parent-interpolated BC
                PoissonKernels.vcycle_solve!(sol, rhs; rtol = 1e-12, maxcycles = 200,
                                             cycle = :W, dirichlet = true)
                pscale = maximum(abs, phi_post .- sum(phi_post) / length(phi_post))
                dphi = maximum(abs.(sol[2:end-1, 2:end-1, 2:end-1] .-
                                    phi_post[2:end-1, 2:end-1, 2:end-1])) / pscale
                @test dphi < 10 * rel_fit + 1e-8      # lands within the oracle's residual
                @test maximum(abs, gmf) > 0           # non-vacuous source
                @info "Phase C subgrid gravity" dims = dims alpha = alpha rel_fit = rel_fit dphi = dphi n_subgrids = n1
            finally
                EnzoLib.free_problem(h)
            end
        end

        # ── the PRODUCTION hook: a full AMR evolve, :julia vs :enzo ───────────
        # Same fixture, two sequential sessions in this (already isolated)
        # process; the only difference is who solves the subgrid Poisson
        # problem.  ProblemType 23 deliberately FREEZES particle positions
        # (UpdateParticlePositions.C: "don't move the particles") — but the
        # velocities integrate the force field every step, so THEY are the
        # observable: after N steps the kicked velocities must agree.
        function evolve_particles(gravity_impl)
            cd(dir) do
                h = EnzoLib.session_init(pf)
                h == C_NULL && error("session_init failed")
                try
                    eng = gravity_impl === :enzo ?
                        EnzoLib.EngineConfig(; hydro = :enzo, gravity = :enzo) :
                        EnzoLib.EngineConfig(; hydro = :enzo, gravity = :julia,
                            hooks = Dict{Symbol,Function}(:gravity =>
                                EnzoLib.poisson_gravity_hook(; rtol = 1e-11)))
                    for _ in 1:5
                        EnzoLib.evolve_level!(h, 0, 0.0; engine = eng, regrid = false)
                    end
                    ng = EnzoLib.problem_num_grids(h)
                    return reduce(vcat, [reduce(hcat,
                        [EnzoLib.problem_get_particle_vel(h, d, g) for d in 0:2])
                        for g in 0:ng-1])
                finally
                    EnzoLib.free_problem(h)
                end
            end
        end
        v_e = evolve_particles(:enzo)
        v_j = evolve_particles(:julia)
        @test size(v_j) == size(v_e)
        vmax = maximum(abs, v_e)
        @test vmax > 1e-3                       # the force field genuinely kicked them
        dv = maximum(abs.(v_j .- v_e))
        @test dv < 1e-8 * vmax                  # identical integrated forces
        @info "Phase C full evolve: gravity=:julia vs :enzo" n_particles = size(v_e, 1) vmax = vmax max_dv = dv rel = dv / vmax
    end
end
