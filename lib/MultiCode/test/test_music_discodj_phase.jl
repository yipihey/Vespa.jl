using Test

let py = get(ENV, "DISCODJ_PYTHON",
             normpath(joinpath(@__DIR__, "..", "..", "..", "..", "..",
                               "disco-dj-fem", ".venv", "bin", "python")))
    if isfile(py)
        get!(ENV, "JULIA_PYTHONCALL_EXE", py)
        ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
        get!(ENV, "JAX_PLATFORMS", "cpu")
    end
end

using MultiCode
using MusicLib
using DiscoDJLib

@testset "MUSIC ↔ DISCO-DJ phase audit" begin
    if !DiscoDJLib.available()
        @warn "DISCO-DJ phase audit skipped" python = DiscoDJLib.pypath()
        @test_skip false
    else
        md = normpath(joinpath(@__DIR__, "..", "..", "..", "reports", "multicode",
                               "shared_phases_and_zoom_poisson.md"))
        r = run_music_discodj_phase_report(res = 16, seed = 42, report_path = md)
        @test r.music_readback_corr > 1 - 1e-12
        @test r.music_mirror_corr < -1 + 1e-12
        @test abs(r.discodj_seed_cross_corr) < 0.2
        @test isfile(md)
        @info "MUSIC/DISCO-DJ phase audit" corr = r.music_discodj_corr seed_cross = r.discodj_seed_cross_corr report = md
    end
end
