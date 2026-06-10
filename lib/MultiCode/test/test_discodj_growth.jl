# ── DISCO-DJ → Enzo + RAMSES: the differentiable-IC growth gate ──────────────
#
# DISCO-DJ's JAX 1LPT (≡ Zel'dovich) displacement field, turned into
# ZERO-velocity particles (the Next-2 trick — no velocity-unit convention can
# enter) and evolved through BOTH legacy codes in EdS.  The whole linear field
# follows b(a/aᵢ) = ⅗x + ⅖x^{−3/2} mode-independently, so the oracle gates the
# full random field.  Skips without the DISCO-DJ venv, the Enzo bridge, or the
# RAMSES cosmo build.

# PythonCall binds its interpreter at load — set the env BEFORE DiscoDJLib.
let py = get(ENV, "DISCODJ_PYTHON",
             "/Users/tabel/Projects/disco-dj-fem/.venv/bin/python")
    if isfile(py)
        get!(ENV, "JULIA_PYTHONCALL_EXE", py)
        ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
        get!(ENV, "JAX_PLATFORMS", "cpu")
    end
end

using Test
using Printf
using MultiCode
using DiscoDJLib                  # activates MultiCodeDiscoDJExt
using EnzoLib, RamsesLib, CodeBridge

@testset "DISCO-DJ ICs through Enzo + RAMSES vs the exact growth" begin
    ok = DiscoDJLib.available() && EnzoLib.grid_available() &&
         CodeBridge.available(RamsesLib.BRIDGE, :cosmo) &&
         isdir(MultiCode.ENZO_DMONLY_DIR)
    if !ok
        @warn "DISCO-DJ growth gate skipped" discodj = DiscoDJLib.available()
        @test_skip false
    else
        r = run_discodj_growth(res = 32)
        @test r.sigma_ic > 0.01                       # non-vacuous fluctuations
        for e in r.engines
            @test e.a >= 4.0 * (1 - 0.06)             # reached the epoch
            @test e.corr_ic > 0.9                     # the field keeps its shape
            # large scales track the closed-form mixed-mode growth at the
            # MEASURED epoch (the few-% deficit is the shared PM resolution)
            @test 0.94 < e.growth_coarse / e.growth_exact < 1.02
            @info "DISCO-DJ growth engine" e.label a = round(e.a; digits = 3) growth_coarse = round(e.growth_coarse; digits = 4) b_exact = round(e.growth_exact; digits = 4) corr_ic = round(e.corr_ic; digits = 4)
        end
        @test r.cross_corr > 0.995                    # Enzo ≡ RAMSES on the same field
        @info "DISCO-DJ cross-code" cross_corr = r.cross_corr

        dir = normpath(joinpath(@__DIR__, "..", "..", "..", "reports", "multicode"))
        mkpath(dir)
        md = joinpath(dir, "discodj_growth.md")
        open(md, "w") do io
            println(io, "# DISCO-DJ ICs through Enzo + RAMSES (ADR-0006 wrapper on-ramp)\n")
            println(io, "DISCO-DJ's differentiable JAX 1LPT field as ZERO-velocity particles ",
                    "(no velocity-unit convention), evolved aᵢ → 4aᵢ in EdS through both ",
                    "codes; the whole linear field follows b(x) = ⅗x + ⅖x^{−3/2}.\n")
            println(io, "| engine | steps | a/aᵢ | large-scale growth | b(a) exact | ratio | corr vs ICs |")
            println(io, "|--------|-------|------|--------------------|------------|-------|-------------|")
            for e in r.engines
                @printf(io, "| %s | %d | %.3f | %.4f | %.4f | %.4f | %.4f |\n",
                        e.label, e.steps, e.a, e.growth_coarse, e.growth_exact,
                        e.growth_coarse / e.growth_exact, e.corr_ic)
            end
            @printf(io, "\nEnzo ↔ RAMSES final-field correlation: **%.4f** — differentiable ",
                    r.cross_corr)
            println(io, "ICs, two legacy codes, one analytic answer.")
        end
        @test isfile(md)
        @info "DISCO-DJ growth report" path = md
    end
end
