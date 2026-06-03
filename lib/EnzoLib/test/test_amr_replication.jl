# E4 — AMR via the Julia recursive EvolveLevel. The full AMR time integrator —
# level subcycling, conservative flux correction, projection from finer levels,
# and dynamic regridding — is driven FROM JULIA on the certified Session steps
# (EnzoLib.evolve_level!/run_amr_density), and must reproduce Enzo's own
# EvolveHierarchy (with AMR) on the root grid.
#
# It is NOT bit-for-bit (AMR has subtle cross-level ordering), but matches to the
# same L1 ~ few×1e-5 the EnzoModules README reports for its Python run_amr.
# Guarded on grid_available() (needs the heavy Session bridge library).

const AMR_PROB = normpath(joinpath(@__DIR__, "..", "..", "..", "..",
                                   "run", "Hydro", "Hydro-1D", "SodShockTube",
                                   "SodShockTubeAMR.enzo"))   # 1D, 4 levels, RefineBy 2

if !EnzoLib.grid_available()
    @info "Session bridge not built — skipping AMR replication test"
else
    @testset "E4: Julia recursive EvolveLevel ≈ Enzo EvolveHierarchy (AMR)" begin
        dj = EnzoLib.run_amr_density(AMR_PROB)       # Julia-driven AMR
        de = EnzoLib.reference_density(AMR_PROB)      # Enzo's own AMR EvolveHierarchy
        @test length(dj) == length(de)
        @test all(isfinite, dj) && all(>(0), dj)
        l1 = sum(abs.(dj .- de)) / length(dj)
        @info "AMR replication (root grid)" cells = length(dj) L1 = l1 Linf = maximum(abs.(dj .- de))
        @test l1 < 2e-4                              # matches Enzo's AMR to ~5e-5
    end

    # grid→level enumeration (ADR-0003 prerequisite): a :julia AMR slot needs to
    # iterate the grids on a level. Verify every grid reports a consistent level
    # and the per-level index lists match session_num_grids_on_level.
    @testset "grid_level / grids_on_level (AMR slot prerequisite)" begin
        cd(EnzoLib._workdir(AMR_PROB)) do
            h = EnzoLib.session_init(AMR_PROB)
            try
                EnzoLib.session_rebuild(h, 0)
                ng = EnzoLib.problem_num_grids(h)
                levels = [EnzoLib.problem_grid_level(h, g) for g in 0:ng-1]
                @test all(>=(0), levels)                  # every grid placed in the hierarchy
                @test 0 in levels                         # a root grid exists
                for L in 0:maximum(levels)
                    idx = EnzoLib.grids_on_level(h, L)
                    @test length(idx) == EnzoLib.session_num_grids_on_level(h, L)
                    @test all(g -> EnzoLib.problem_grid_level(h, g) == L, idx)
                end
            finally
                EnzoLib.free_problem(h)
            end
        end
    end
end
