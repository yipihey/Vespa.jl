# ── the dfmm engine in the harness (ADR-0006 Phase 5: library convergence) ───
#
# The dual-frame moment method — the first solver born inside the ecosystem —
# runs the harness's Sod spec (γ = 5/3, its closure's index) on its own
# Lagrangian segments through the `MultiCodeDfmmExt` package extension, and is
# gated against the SAME exact-Riemann oracle as every legacy engine:
#   - mass bit-exact and total momentum at round-off (the variational
#     scheme's exactness claims, verified in the harness's own terms);
#   - L1(ρ) within dfmm's documented Tier-A.1 band vs the exact solution;
#   - a report row lands next to the cross-code Sod comparison.

using Test
using Printf
using MultiCode
using dfmm                       # activates MultiCodeDfmmExt

@testset "the dfmm engine (dual-frame moment method) in the Sod harness" begin
    spec = SodSpec(gamma = 5 / 3, t = 0.2)
    r = run_dfmm_sod(spec; N = 100)
    l1 = MultiCode.sod_l1(r.profile, spec, r.t)
    @test abs(r.mass - r.mass0) == 0.0                # Lagrangian mass: bit-exact
    @test abs(r.momentum) < 1e-12                     # variational momentum exactness
    @test l1.rho < 0.08                               # Tier-A.1 band (N=100)
    @test l1.u < 0.2
    @info "dfmm Sod vs exact" l1_rho = l1.rho l1_u = l1.u momentum = r.momentum steps = r.steps seconds = round(r.seconds; digits = 2)

    dir = normpath(joinpath(@__DIR__, "..", "..", "..", "reports", "multicode"))
    mkpath(dir)
    md = joinpath(dir, "dfmm_sod.md")
    open(md, "w") do io
        println(io, "# dfmm in the Sod harness (ADR-0006 Phase 5)\n")
        println(io, "The dual-frame moment method (variational, symplectic, Lagrangian ",
                "segments) on the harness Sod spec (γ = 5/3, t̂ = $(spec.t)), via the ",
                "`MultiCodeDfmmExt` package extension — certified against the same ",
                "exact-Riemann oracle as the legacy engines.\n")
        println(io, "| engine | cells | steps | wall-clock [s] | L1(ρ) | L1(u) | mass drift | total momentum |")
        println(io, "|--------|-------|-------|----------------|-------|-------|------------|----------------|")
        @printf(io, "| dfmm (τ=%.0e) | %d | %d | %.2f | %.4f | %.4f | %.1e | %.1e |\n",
                r.diag.tau, r.diag.N, r.steps, r.seconds, l1.rho, l1.u,
                abs(r.mass - r.mass0), abs(r.momentum))
        println(io, "\nMass is conserved bit-exactly (the Lagrangian masses are labels, ",
                "not state) and the total momentum stays at round-off — the variational ",
                "integrator's exactness claims, reproduced inside the harness.")
    end
    @test isfile(md)
    @info "dfmm engine report" path = md
end
