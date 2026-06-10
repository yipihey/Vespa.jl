# ── the gravity guest slot (ADR-0006 "Next"): KA Poisson inside RAMSES ───────
#
# The first slice of the mini-ramses-kernel KA re-expression: RAMSES deposits
# the density (`rho_fine`) and differences the force (`force_fine`); the guest
# replaces `multigrid` with PoissonKernels' FFT solve on the DISCRETE 7-point
# Green's function — the exact solution of the very linear system RAMSES's MG
# iterates on.  Gates:
#   - the deposit carried the injected mode (the gate is not vacuous);
#   - φ(KA) ≡ φ(RAMSES MG @ ε=1e-12) to solver tolerance, mean-removed (the
#     periodic gauge) — at a loose ε=1e-3 the same diff is ~1e-3, so this IS
#     tracking the oracle's convergence, not a fixed-point coincidence;
#   - force_fine on each φ agrees to the same tolerance;
#   - the injected single-mode density is a discrete eigenfunction, so φ(KA)
#     must ALSO match the closed-form discrete solution at machine precision —
#     an analytic anchor independent of both solvers.

using Test
using MultiCode
using RamsesLib
import PoissonKernels
using Metal                       # enables the :metal backend (Apple Silicon)

haskey(ENV, "RAMSES_LIB") || (ENV["RAMSES_LIB"] =
    normpath(joinpath(@__DIR__, "..", "..", "..", "..", "..",
                      "mini-ramses", "bin64h", "libramses3d.dylib")))

@testset "the gravity guest slot (KA Poisson vs RAMSES multigrid)" begin
    if !RamsesLib.available()
        @test_skip false
    else
        level = 5; amp = 0.05
        r = run_ramses_gravity_compare(level = level, amp = amp, eps = 1e-12)
        try
            @test r.rho_dev > 0.9 * amp                  # the deposit carried the mode
            @test r.dphi < 1e-11                         # φ ≡ MG to solver tolerance
            @test r.dforce < 1e-11                       # force_fine sees the same φ
            @info "gravity slot vs RAMSES MG" dphi = r.dphi dforce = r.dforce phi_scale = r.phi_scale n = r.n

            # the analytic anchor: the injected mode is a discrete eigenfunction
            # ∇²_7pt [sin(2πx)sin(4πy)cos(2πz)] = −λ·(mode), λ = Σ_d 4n²sin²(π m_d/n)
            n = r.n
            ck, phi = MultiCode.ramses_grid_field(r.handle, :phi, level)
            lam = 4 * n^2 * (sin(π * 1 / n)^2 + sin(π * 2 / n)^2 + sin(π * 1 / n)^2)
            err = 0.0
            for k in 1:n, j in 1:n, i in 1:n
                x = (i - 0.5) / n; y = (j - 0.5) / n; z = (k - 0.5) / n
                mode = sin(2π * x) * sin(4π * y) * cos(2π * z)
                phi_exact = -4π * amp * mode / lam
                err = max(err, abs(phi[i, j, k] - phi_exact))
            end
            @test err / r.phi_scale < 1e-11              # machine-precision analytic anchor
            @info "gravity slot analytic anchor" rel_err = err / r.phi_scale
        finally
            r.free()
        end

        # ── the REFINED-level Dirichlet solve (Next-4 slice 2) ────────────────
        # A cuboid fine region (flag1 through the bridge → the host's own
        # refine), host deposits + coarse solve; the fine system is RAMSES's
        # ghost-interpolated Dirichlet problem.  Two independent gates:
        # (1) the ORACLE's converged solution satisfies OUR assembled system
        #     (interpol_phi ghosts + Enzo-MG rhs scaling) at machine precision
        #     — certifies the replication without involving the KA solver;
        # (2) the KA vcycle_solve!(dirichlet=true) from scratch lands on the
        #     oracle to solver tolerance.
        ra = run_ramses_gravity_amr_compare(levc = 5, half = 4, eps = 1e-12)
        try
            @test ra.n_fine_octs == 512                  # the 16³ cuboid exists
            @test ra.nf == (16, 16, 16)
            @test ra.resid_oracle < 1e-13                # system replication ≡ ε
            @test ra.dphi < 1e-10                        # KA Dirichlet ≡ RAMSES CG
            @info "refined-level Dirichlet solve" dphi = ra.dphi resid_oracle = ra.resid_oracle nf = ra.nf
        finally
            ra.free()
        end

        # ── the IRREGULAR refined region (Next-6): masked CG on a blob ────────
        # A spherical blob of refined cells (genuinely non-cuboid — asserted):
        # same two independent gates as the cuboid case, on the masked system.
        rb = run_ramses_gravity_blob_compare(levc = 5, radius = 0.18, eps = 1e-12)
        try
            @test !rb.is_cuboid                          # the region is irregular
            @test rb.n_fine_octs > 0
            @test rb.resid_oracle < 1e-13                # masked-system replication ≡ ε
            @test rb.dphi < 1e-10                        # masked CG ≡ RAMSES CG
            @info "irregular-region (blob) solve" dphi = rb.dphi resid_oracle = rb.resid_oracle octs = rb.n_fine_octs bbox = rb.nloc cg_iters = rb.cg_iters
        finally
            rb.free()
        end

        # ── the KA masked CG on Metal (Next-7): same source, f32, GPU ─────────
        if PoissonKernels.has_backend(:metal)
            rm_ = run_ramses_gravity_blob_compare(levc = 5, radius = 0.18,
                                                  eps = 1e-12, device = :metal)
            try
                @test !rm_.is_cuboid
                @test rm_.dphi < 1e-4                # the f32 residual floor
                @info "irregular-region solve on Metal (f32)" dphi = rm_.dphi cg_iters = rm_.cg_iters relres = rm_.cg_relres
            finally
                rm_.free()
            end
        else
            @test_skip false
        end
    end
end
