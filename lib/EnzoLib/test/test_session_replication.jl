# E2 — full-replication, the EvolveHierarchy-level confirmation. Runs ONLY when
# the heavy Session bridge library (libenzomodules_grid, linking the full Enzo
# .so) is built; skipped otherwise so the pilot suite stays buildable without it.
#
# The new Julia-driven EvolveLevel must reproduce Enzo's own EvolveHierarchy
# (the old code) BIT-FOR-BIT on the Toro-1 / Sod shock tube.

const PROB = normpath(joinpath(@__DIR__, "..", "..", "..", "..",
                               "run", "Hydro", "Hydro-1D", "Toro-1-ShockTube",
                               "Toro-1-ShockTube.enzo"))

if !EnzoLib.grid_available()
    @info "Session bridge not built — skipping EvolveHierarchy-level replication test" lib = EnzoLib.grid_libpath()
else
    @testset "Julia EvolveLevel ≡ Enzo EvolveHierarchy (bit-for-bit)" begin
        dj = EnzoLib.session_replicate_density(PROB)    # new Julia driver, old Enzo steps
        de = EnzoLib.reference_density(PROB)             # Enzo's own EvolveHierarchy
        @test length(dj) == length(de)
        @test all(isfinite, dj) && all(>(0), dj)
        linf = maximum(abs.(dj .- de))
        @info "full replication" cells = length(dj) Linf = linf bit_identical = count(dj .== de)
        @test linf == 0.0                                # bit-for-bit identical
    end
end
