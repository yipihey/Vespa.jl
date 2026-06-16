# Live differential cert for the GPU-resident particle push vs Enzo's
# session_update_particles.
#
# The CIC interpolation and the leapfrog kick/drift formulas are already certified
# bit-tight against the Fortran (cic_interp.F + the comoving update) in
# PoissonKernels' test_particle_push.jl. What needs a LIVE oracle is that the
# comoving coefficients — built from `EnzoLib.session_expansion_factor` at the exact
# sub-step times — and the leapfrog ORDER reproduce Enzo's update to round-off.
#
# We use a UNIFORM particle lattice at cell centres with a constant velocity: its
# CIC density is exactly uniform, so self-gravity is ~0 (to FFT round-off) and
# Enzo's update reduces to pure drift  x ← x + (dt/a(t+½dt))·v . Feeding the GPU
# push zeroed acceleration grids must reproduce Enzo's particles bit-for-bit —
# isolating the drift coefficient + leapfrog plumbing on the live f64 hierarchy.
# (The nonzero-force interpolation is exercised end-to-end by the cicass A/B run.)

using Test, Printf
using MultiCode, EnzoLib, PoissonKernels

@testset "resident particle push ≡ Enzo session_update_particles (drift, live f64)" begin
    if !(EnzoLib.grid_available() && isdir(MultiCode.ENZO_DMONLY_DIR))
        @warn "resident-particle cert skipped" grid = EnzoLib.grid_available() dmonly = isdir(MultiCode.ENZO_DMONLY_DIR)
        @test_skip false
    else
        spec = ZeldovichSpec()
        N = spec.n
        z_init = spec.z_init
        # boot the EdS-patched dm_only CosmologySimulation (same recipe as run_enzo_zeldovich)
        dir = mktempdir()
        for f in readdir(MultiCode.ENZO_DMONLY_DIR)
            p = joinpath(MultiCode.ENZO_DMONLY_DIR, f)
            isfile(p) && filesize(p) < 10^7 && !endswith(f, ".enzo") && cp(p, joinpath(dir, f))
        end
        par = read(joinpath(MultiCode.ENZO_DMONLY_DIR, "dm_only.enzo"), String)
        for (pat, rep) in (r"CosmologyOmegaMatterNow\s*=\s*\S+" => "CosmologyOmegaMatterNow    = 1.0",
                           r"CosmologyOmegaLambdaNow\s*=\s*\S+" => "CosmologyOmegaLambdaNow    = 0.0",
                           r"CosmologySimulationOmegaCDMNow\s*=\s*\S+" => "CosmologySimulationOmegaCDMNow = 1.0",
                           r"CosmologyInitialRedshift\s*=\s*\S+" => "CosmologyInitialRedshift   = $(z_init)",
                           r"CosmologyComovingBoxSize\s*=\s*\S+" => "CosmologyComovingBoxSize   = $(spec.box_mpch)")
            par = replace(par, pat => rep)
        end
        par *= "\nStaticHierarchy = 1\nMaximumRefinementLevel = 0\n"
        pf = joinpath(dir, "resident_drift.enzo"); write(pf, par)

        cd(dir) do
            h = EnzoLib.session_init(pf)
            h == C_NULL && error("session_init failed")
            try
                np = EnzoLib.problem_num_particles(h, 0)
                @test np == N^3
                # uniform cell-centred lattice + constant velocity (distinct per axis)
                q = Float64[ ((c == 1 ? mod(p-1, N) : c == 2 ? mod(div(p-1, N), N) : div(p-1, N^2)) + 0.5) / N
                             for p in 1:np, c in 1:3 ]
                v0 = (0.013, -0.0071, 0.0055)
                for d in 0:2
                    EnzoLib.problem_set_particle_pos(h, d, q[:, d+1])
                    EnzoLib.problem_set_particle_vel(h, d, fill(v0[d+1], np))
                end

                # drive one cycle by hand so we control t_start and dt
                t0 = EnzoLib.session_time(h)
                EnzoLib.session_set_boundary(h, 0)
                dt = EnzoLib.session_compute_dt(h, 0)
                EnzoLib.session_set_dt(h, dt, 0)
                EnzoLib.session_gravity(h, 0)                  # accel ≈ 0 for the uniform lattice

                # the force must be ~0 (this makes it a pure-drift differential)
                amax = maximum(abs.(EnzoLib.problem_get_acceleration(h, 0, 0)))
                @info "uniform-lattice force" amax
                @test amax < 1e-8

                pos0 = [EnzoLib.problem_get_particle_pos(h, d, 0) for d in 0:2]
                vel0 = [EnzoLib.problem_get_particle_vel(h, d, 0) for d in 0:2]

                # ENZO reference update
                EnzoLib.session_update_particles(h, 0)
                pos_ref = [EnzoLib.problem_get_particle_pos(h, d, 0) for d in 0:2]
                vel_ref = [EnzoLib.problem_get_particle_vel(h, d, 0) for d in 0:2]

                # reset and run the GPU (CPU-f64) resident push with ZEROED accel grids
                for d in 0:2
                    EnzoLib.problem_set_particle_pos(h, d, pos0[d+1])
                    EnzoLib.problem_set_particle_vel(h, d, vel0[d+1])
                end
                st = MultiCode.resident_particles_init(h, PoissonKernels.backend(:cpu), Float64; grid=0, wrap=0.0)
                fill!(st.gx, 0.0); fill!(st.gy, 0.0); fill!(st.gz, 0.0)
                MultiCode.particle_push_gpu!(st, h, 0, dt; refresh=false, sync=true)
                pos_gpu = [EnzoLib.problem_get_particle_pos(h, d, 0) for d in 0:2]
                vel_gpu = [EnzoLib.problem_get_particle_vel(h, d, 0) for d in 0:2]

                # independent bridge sanity: a(t) ≡ session_cosmology's a at current time
                a_now, _ = EnzoLib.session_cosmology(h)
                a_ef, dadt_ef = EnzoLib.session_expansion_factor(h, EnzoLib.session_time(h))
                @test a_ef ≈ a_now rtol=1e-12
                @test dadt_ef > 0                              # expanding EdS

                perr = maximum(maximum(abs.(pos_gpu[d] .- pos_ref[d])) for d in 1:3)
                verr = maximum(maximum(abs.(vel_gpu[d] .- vel_ref[d])) for d in 1:3)
                @info "resident push vs Enzo (drift)" pos_maxabs=perr vel_maxabs=verr dt=dt a=a_now
                @test perr < 1e-12                             # bit-tight drift coefficient + leapfrog
                @test verr < 1e-12                             # zero-force kick is identity, both sides
                # and the drift actually moved the particles (non-trivial test)
                @test maximum(abs.(pos_ref[1] .- pos0[1])) > 1e-6
            finally
                EnzoLib.free_problem(h)
            end
        end
    end
end
