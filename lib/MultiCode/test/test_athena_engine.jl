# ── Athena++ in the Sod harness (the wrapper-registry on-ramp) ────────────────
#
# The fourth legacy engine: the stock athinput.sod run in-process through
# AthenaLib (the MultiCodeAthenaExt package extension), gated against the SAME
# exact-Riemann oracle as Enzo/RAMSES/Arepo/dfmm.  Skips cleanly when the
# Athena++ capi library is not built.

using Test
using Printf
using MultiCode
using AthenaLib                  # activates MultiCodeAthenaExt

@testset "the Athena++ engine in the Sod harness" begin
    if !AthenaLib.available()
        @warn "libathena_capi not found — skipping" lib = AthenaLib.libpath()
        @test_skip false
    else
        spec = SodSpec()                                   # γ=1.4, t̂=0.1, x0=0.5
        r = run_athena_sod(spec)
        l1 = MultiCode.sod_l1(r.profile, spec, r.t)
        @test r.t_end ≈ spec.t atol = 1e-6                 # reached the epoch
        @test r.mass_drift < 1e-12                         # exact conservation (.hst)
        @test l1.rho < 0.05                                # the PPM-class L1 band
        @test l1.u < 0.05
        @info "Athena++ Sod vs exact" l1_rho = l1.rho l1_u = l1.u mass_drift = r.mass_drift seconds = round(r.seconds; digits = 2)

        dir = normpath(joinpath(@__DIR__, "..", "..", "..", "reports", "multicode"))
        mkpath(dir)
        md = joinpath(dir, "athena_sod.md")
        open(md, "w") do io
            println(io, "# Athena++ in the Sod harness (wrapper-registry on-ramp)\n")
            println(io, "The stock `athinput.sod` (γ = 1.4, interface recentred to x = 0.5) ",
                    "run IN-PROCESS through AthenaLib via the `MultiCodeAthenaExt` package ",
                    "extension, against the same exact-Riemann oracle as the other engines.\n")
            println(io, "| engine | cells | wall-clock [s] | L1(ρ) | L1(u) | mass drift |")
            println(io, "|--------|-------|----------------|-------|-------|------------|")
            @printf(io, "| athena++ (:hydro) | %d | %.2f | %.4f | %.4f | %.1e |\n",
                    r.diag.nx1, r.seconds, l1.rho, l1.u, r.mass_drift)
        end
        @test isfile(md)
        @info "Athena engine report" path = md
    end
end
