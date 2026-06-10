# ── the cosmology gate (ADR-0006 flagship 1, cosmological): one particle set,
# Enzo and RAMSES, against the exact trajectory ────────────────────────────────
#
# A Zel'dovich plane wave with ZERO initial velocities (no velocity-unit
# convention enters), the SAME Julia-generated lattice+displacement injected
# through both codes' particle bridges (Enzo: the new bridge setters on the
# EdS-patched dm_only CosmologySimulation; RAMSES: init_particles on the
# UNITS=COSMO flavor with Julia-written grafic headers), evolved a_i → 4·a_i.
# Oracle: the closed-form mixed-mode growth b(a/a_i) = 3/5·x + 2/5·x^{-3/2}.

using Test
using Printf
using MultiCode
using EnzoLib, RamsesLib, CodeBridge

@testset "Zel'dovich: one particle set, Enzo + RAMSES vs the exact trajectory" begin
    ramses_ok = CodeBridge.available(RamsesLib.BRIDGE, :cosmo)
    enzo_ok = EnzoLib.grid_available() && isdir(MultiCode.ENZO_DMONLY_DIR)
    if !(ramses_ok && enzo_ok)
        @warn "Zel'dovich gate skipped" ramses_ok enzo_ok
        @test_skip false
    else
        spec = ZeldovichSpec()
        rows = NamedTuple[]
        for (label, runner) in (("enzo-pm", run_enzo_zeldovich),
                                ("ramses-pm", run_ramses_zeldovich))
            r = runner(spec)
            try
                m = zeldovich_measure(r.xp, spec)        # also gates y/z immobility
                bA = zeldovich_growth(r.a) * spec.A
                @test r.a >= spec.a_ratio                 # reached the epoch
                @test abs(m.bA / bA - 1) < 0.03           # the exact growth factor
                @test m.rms_resid / spec.A < 0.06         # the displacement SHAPE
                push!(rows, (label = label, a = r.a, steps = r.steps, seconds = r.seconds,
                             bA = m.bA, bA_exact = bA, resid = m.rms_resid,
                             psi = m.bA / zeldovich_growth(r.a)))
                @info "zeldovich engine" label a = round(r.a; digits = 3) ratio = round(m.bA / bA; digits = 4) resid_over_A = round(m.rms_resid / spec.A; digits = 4) steps = r.steps
            finally
                r.free()
            end
        end
        # cross-code: the growth-normalized displacement amplitude ψ̂ = bA/b(a)
        # must agree between the codes (each measured at its own final a)
        @test abs(rows[1].psi - rows[2].psi) / spec.A < 0.02

        dir = normpath(joinpath(@__DIR__, "..", "..", "..", "reports", "multicode"))
        mkpath(dir)
        md = joinpath(dir, "zeldovich_comparison.md")
        open(md, "w") do io
            println(io, "# One particle set, two cosmology codes (ADR-0006)\n")
            println(io, "Zel'dovich plane wave, EdS, ZERO initial velocities: the same ",
                    "$(spec.n)³ lattice with ψ = $(spec.A)·sin(2πx) injected through both ",
                    "codes' particle bridges and evolved a_i → $(spec.a_ratio)·a_i ",
                    "(z = $(spec.z_init) start).  Oracle: the closed-form mixed-mode growth ",
                    "b(x) = (3/5)x + (2/5)x^{-3/2}.\n")
            println(io, "| engine | steps | wall-clock [s] | a/a_i | bA measured | bA exact | ratio | rms shape resid / A |")
            println(io, "|--------|-------|----------------|-------|-------------|----------|-------|---------------------|")
            for r in rows
                @printf(io, "| %s | %d | %.2f | %.3f | %.5f | %.5f | %.4f | %.4f |\n",
                        r.label, r.steps, r.seconds, r.a, r.bA, r.bA_exact,
                        r.bA / r.bA_exact, r.resid / spec.A)
            end
            println(io, "\nEnzo runs its CosmologySimulation machinery (PM gravity + comoving ",
                    "expansion via the certified EvolveLevel slots); RAMSES runs its ",
                    "supercomoving `amr_step` production loop (UNITS=COSMO build, grafic ",
                    "headers written from Julia).  Identical particles, zero shared code, ",
                    "one analytic answer.")
        end
        @test isfile(md)
        @info "Zel'dovich comparison report" path = md
    end
end
