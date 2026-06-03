# The set_acceleration_field bridge — the enabling primitive for a :julia gravity
# slot to feed an :enzo hydro. Validates: (1) write→read round-trips exactly, and
# (2) Enzo's own gravity (session_gravity) populates AccelerationField with a
# nontrivial field, so it is the array SolveHydroEquations reads as the source.
#
# Guarded on grid_available() (needs the Session bridge library).

include(joinpath(@__DIR__, "enzo_suite_common.jl"))

if !EnzoLib.grid_available()
    @info "Session bridge not built — skipping set_acceleration test"
else
    @testset "set_acceleration_field bridge primitive" begin
        pf = paramfile_for(joinpath(RUN_DIR, "Cosmology", "ZeldovichPancake", "ZeldovichPancake.enzo"))
        cd(EnzoLib._workdir(pf)) do
            h = EnzoLib.session_init(pf)
            try
                EnzoLib.session_set_boundary(h, 0)
                n = EnzoLib.problem_grid_size(h, 0)

                # (1) round-trip: a written acceleration reads back (to the Enzo
                # AccelerationField storage precision).
                want = collect(range(-1.0, 1.0; length = n))
                EnzoLib.problem_set_acceleration(h, 0, want; grid = 0)
                got = EnzoLib.problem_get_acceleration(h, 0, 0)
                @test maximum(abs, got .- want) < 1e-6   # exact up to the stored float precision

                # (2) Enzo's gravity solve fills AccelerationField with structure.
                EnzoLib.session_gravity(h, 0)
                gx = EnzoLib.problem_get_acceleration(h, 0, 0)
                @test length(gx) == n
                @test any(!iszero, gx)                  # a real, nonzero acceleration field
                @test all(isfinite, gx)
                @info "Enzo AccelerationField[0]" cells = n max_abs = maximum(abs, gx)
            finally
                EnzoLib.free_problem(h)
            end
        end
    end
end
