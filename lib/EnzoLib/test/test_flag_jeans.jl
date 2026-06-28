# Enzo Jeans-flagging bridge (enzomodules_flag_jeans, CellFlaggingMethod = 6) wired into
# EnzoLib as the oracle for the native Vespa jeans_length_indicator. Gated on
# grid_available() (needs libenzomodules_grid).
#
# This checks the C↔Julia wiring and the Jeans PHYSICS DIRECTION: at fixed gas state, a
# larger cell resolves the Jeans length with fewer cells, so the flagged-cell count is
# monotonically non-decreasing in Δx and spans none→all. (Cell-exact agreement with the
# native physical-cgs formula additionally needs an Enzo internal-units calibration — its
# TemperatureUnits/Mu normalization makes Enzo's λ_J ~3× ours; tracked as a follow-up.)

using EnzoLib, Test

if !EnzoLib.grid_available()
    @info "grid bridge not built — skipping Enzo flag_jeans oracle" lib = EnzoLib.grid_libpath()
else
    @testset "Enzo flag_jeans bridge (CellFlaggingMethod=6)" begin
        n = 8
        ρ = fill(1.0e-24, n, n, n); e = fill(1.0e10, n, n, n)   # uniform gas state
        counts = Int[]
        for dx in (1.0e19, 1.0e20, 3.0e20, 1.0e21, 1.0e23)
            flag, cnt = EnzoLib.flag_jeans_ref(ρ, e; dx = dx, safety = 4.0,
                            density_units = 1.0, length_units = 1.0, time_units = 1.0)
            @test count(!iszero, flag) == cnt            # count matches the flag field
            @test 0 <= cnt <= n^3
            push!(counts, cnt)
        end
        @info "flag_jeans vs Δx" counts
        @test issorted(counts)                            # bigger cells ⇒ ≥ as many flagged
        @test counts[1] == 0 && counts[end] == n^3        # spans none → all
    end
end
