# SUITE — confirm the new Julia driver reproduces Enzo's own test problems. A
# curated, fast, reliably-passing subset of Enzo's quicksuite (a single-grid tube,
# an AMR tube, and an MHD tube) is run through BOTH Enzo's EvolveHierarchy and the
# Julia EvolveLevel (full replication) and the final fields compared. The full
# 63-problem sweep lives in examples/run_enzo_suite.jl (categorized table).
#
# Guarded on grid_available() (needs the heavy Session bridge library).

include(joinpath(@__DIR__, "enzo_suite_common.jl"))

# (problem .enzo, tolerance, note). Tolerances: single-grid is bit-for-bit;
# AMR/MHD agree to the cross-level-ordering / scheme level.
const SUITE_CASES = [
    ("Hydro/Hydro-1D/SodShockTube/SodShockTube.enzo",        1e-12, "single-grid, bit-for-bit"),
    ("Hydro/Hydro-1D/Toro-2-ShockTubeAMR/Toro-2-ShockTubeAMR.enzo", 1e-3, "4-level AMR"),
    ("MHD/1D/BrioWu-MHD-1D/BrioWu-MHD-1D.enzo",              1e-2, "MHD (Dedner), 9 fields"),
]

if !EnzoLib.grid_available()
    @info "Session bridge not built — skipping Enzo test-suite confirmation"
else
    @testset "Enzo test problems: Julia EvolveLevel ≡ EvolveHierarchy" begin
        for (rel, tol, note) in SUITE_CASES
            pf = joinpath(RUN_DIR, rel)
            r = compare_problem(pf)
            @info "suite problem" problem = basename(dirname(pf)) max_field_error = r.err nfields = r.nfields note
            @test r.nfields >= 1
            @test r.err < tol
        end
    end
end
