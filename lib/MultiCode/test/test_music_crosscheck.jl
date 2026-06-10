# ── the MUSIC injector validation (the wrapper-registry on-ramp) ──────────────
#
# ONE MusicSpec realization, TWO live codes booted on MUSIC's own outputs
# (Enzo on parameter_file.txt + particle ICs; RAMSES UNITS=COSMO on the
# grafic2 level dir), the INITIAL CIC density fields correlated.  The expected
# floor is the float32 precision of the grafic planes (~1e-7 relative) — the
# Enzo path carries float64 displacements.  Skips cleanly without the MUSIC
# library, the Enzo bridge, or the RAMSES cosmo build.

using Test
using Printf
using MultiCode
using MusicLib                    # activates MultiCodeMusicExt
using EnzoLib, RamsesLib, CodeBridge

@testset "MUSIC injector: one realization, Enzo + RAMSES initial fields" begin
    ok = MusicLib.available() && EnzoLib.grid_available() &&
         CodeBridge.available(RamsesLib.BRIDGE, :cosmo)
    if !ok
        @warn "MUSIC cross-check skipped" music = MusicLib.available() enzo = EnzoLib.grid_available()
        @test_skip false
    else
        r = run_music_crosscheck(level = 5)
        @test r.corr > 0.999999                      # the same realization
        @test r.rms < 1e-5                           # the f32 grafic floor
        @test abs(r.sigma_enzo / r.sigma_ramses - 1) < 1e-6
        @test r.sigma_enzo > 0.01                    # non-vacuous fluctuations
        @info "MUSIC injector cross-check" corr = r.corr rms = r.rms sigma = r.sigma_enzo n = r.n

        dir = normpath(joinpath(@__DIR__, "..", "..", "..", "reports", "multicode"))
        mkpath(dir)
        md = joinpath(dir, "music_crosscheck.md")
        open(md, "w") do io
            println(io, "# MUSIC injector validation (ADR-0006 wrapper on-ramp)\n")
            println(io, "ONE `MusicSpec` realization (identical seeds), generated in-process ",
                    "in two formats; Enzo booted on the generated `parameter_file.txt` + ",
                    "particle ICs, RAMSES (UNITS=COSMO) on the grafic2 level directory; the ",
                    "initial CIC density contrasts compared with no evolution.\n")
            println(io, "| n | corr(δ_E, δ_R) | rms(δ_E−δ_R)/σ | σ_E | σ_R |")
            println(io, "|---|----------------|----------------|-----|-----|")
            @printf(io, "| %d³ | %.15f | %.2e | %.6f | %.6f |\n",
                    r.n, r.corr, r.rms, r.sigma_enzo, r.sigma_ramses)
            println(io, "\nThe residual is the float32 precision of the grafic planes — the ",
                    "two injection chains are otherwise identical.")
        end
        @test isfile(md)
        @info "MUSIC cross-check report" path = md
    end
end
