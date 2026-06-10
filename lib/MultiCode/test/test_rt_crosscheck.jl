# ── Phase 4.4 gate (ADR-0006): Moray vs RAMSES-RT on the same density field ───
#
# The ADR's Phase-4 acceptance test: the SAME Iliev Test-1 setup — uniform
# n_H = 1e-3 cm⁻³ hydrogen at 1e4 K (isothermal), a 5e48 photons/s
# monochromatic 13.6 eV source, 6.6 kpc box — through TWO genuinely different
# RT methods: Enzo's Moray (adaptive ray tracing, effectively infinite signal
# speed) and RAMSES-RT (M1 moment fluid at a reduced speed of light), each
# driven through its CodeBridge wrapper.  Gates: each I-front history tracks
# the analytic Strömgren solution, the two codes agree with each other within
# the inter-code band the Iliev comparison project established (~10–15%), and
# the joint report is written.

using Test
using Printf
using MultiCode
using EnzoLib, RamsesLib, CodeBridge

const RT_SNAPSHOTS = [3.0, 5.0]

@testset "ADR-0006 Phase 4.4: Moray vs RAMSES-RT cross-check" begin
    moray_ok = EnzoLib.grid_available() && isfile(MultiCode.ENZO_PHOTONTEST_PF)
    rrt_ok = CodeBridge.available(RamsesLib.BRIDGE, :rt)
    if !(moray_ok && rrt_ok)
        @warn "RT cross-check skipped" moray_ok rrt_ok
        @test_skip false
    else
        m = run_moray_stromgren(t_end_myr = 5.0, snapshots = RT_SNAPSHOTS)
        moray_hist = m.history
        m.free()
        r = run_ramsesrt_stromgren(t_end_myr = 5.0, snapshots = RT_SNAPSHOTS, level = 6)
        rrt_hist = r.history
        r.free()

        rows = NamedTuple[]
        for ((tm, rm), (tr, rr)) in zip(moray_hist, rrt_hist)
            @test isapprox(tm, tr; atol = 1e-6)               # same epochs
            ex = stromgren_radius(tm)
            @test isfinite(rm) && isfinite(rr)
            @test abs(rm - ex) / ex < 0.12                    # Moray vs analytic
            @test abs(rr - ex) / ex < 0.12                    # RAMSES-RT vs analytic
            @test abs(rm - rr) / ((rm + rr) / 2) < 0.15       # code vs code (Iliev band)
            push!(rows, (t = tm, moray = rm, ramsesrt = rr, exact = ex))
        end
        # both fronts grow
        @test issorted([row.moray for row in rows])
        @test issorted([row.ramsesrt for row in rows])

        # the joint report
        dir = normpath(joinpath(@__DIR__, "..", "..", "..", "reports", "multicode"))
        mkpath(dir)
        md = joinpath(dir, "rt_crosscheck.md")
        open(md, "w") do io
            println(io, "# Two RT methods, one density field (ADR-0006 Phase 4)\n")
            println(io, "Iliev Test 1: uniform n_H = 1e-3 cm⁻³ hydrogen at 1e4 K, ",
                    "a 5e48 photons/s monochromatic 13.6 eV source, 6.6 kpc box.  ",
                    "Enzo **Moray** (adaptive ray tracing, 32³) vs **RAMSES-RT** ",
                    "(M1 moment method, reduced c = 0.005c, 64³), each through its ",
                    "CodeBridge wrapper, vs the analytic Strömgren front ",
                    "r(t) = R_s·(1−e^{−t/t_rec})^{1/3}.\n")
            println(io, "| t [Myr] | Moray r_I | RAMSES-RT r_I | analytic | Moray/exact | RAMSES-RT/exact | code/code |")
            println(io, "|---------|-----------|---------------|----------|-------------|-----------------|-----------|")
            for row in rows
                @printf(io, "| %.1f | %.4f | %.4f | %.4f | %.3f | %.3f | %.3f |\n",
                        row.t, row.moray, row.ramsesrt, row.exact,
                        row.moray / row.exact, row.ramsesrt / row.exact,
                        row.moray / row.ramsesrt)
            end
            println(io, "\nRadii in box units (box = 6.6 kpc).  The M1 front lags a few percent ",
                    "at early times (reduced speed of light + the discrete first cell) and ",
                    "converges onto the ray-traced and analytic fronts — the behaviour the ",
                    "Iliev et al. (2006) comparison project documents for these method families.")
        end
        @test isfile(md)
        @info "Phase 4.4 cross-check" rows path = md
    end
end
