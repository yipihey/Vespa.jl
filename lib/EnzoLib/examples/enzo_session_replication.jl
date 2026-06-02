# THE confirmation: the new Julia driver, reusing all the OLD Enzo routines via
# the live Session, reproduces Enzo's own monolithic EvolveHierarchy **bit-for-bit**.
#
# `session_replicate_density` runs a Julia-reimplemented EvolveLevel (set_boundary
# → compute_dt → set_dt → solve_hydro → advance_time) on the live Enzo hierarchy;
# `reference_density` runs Enzo's `EvolveHierarchy` (the old code). Both call the
# identical certified legacy kernels, so the result is bit-for-bit identical
# (Linf = 0) — the README's guarantee, reached from Julia.
#
# Prereqs (heavy native build; see EnzoModules/deps):
#   1. libenzo_p8_b8.dylib  — full Enzo serial lib (gcc-15, HDF5 with v16 API)
#   2. libenzomodules_grid.dylib — the Session bridge (bash build_grid_darwin.sh)
# Run:
#   ENZOMODULES_GRID_LIB=<repo>/EnzoModules/deps/libenzomodules_grid.dylib \
#   julia --project=lib/EnzoLib/test lib/EnzoLib/examples/enzo_session_replication.jl [path/to/problem.enzo]

using EnzoLib

const PROB = length(ARGS) >= 1 ? ARGS[1] :
    normpath(joinpath(@__DIR__, "..", "..", "..", "..",
                      "run", "Hydro", "Hydro-1D", "Toro-1-ShockTube", "Toro-1-ShockTube.enzo"))

EnzoLib.grid_available() ||
    error("grid/session library not built — see EnzoModules/deps/build_grid_darwin.sh")

dj = EnzoLib.session_replicate_density(PROB)   # new Julia EvolveLevel
de = EnzoLib.reference_density(PROB)            # old Enzo EvolveHierarchy

linf = maximum(abs.(dj .- de))
nexact = count(dj .== de)
println("Problem: ", basename(PROB))
println("  Julia-driven EvolveLevel  vs  Enzo EvolveHierarchy")
println("    ncells          = ", length(dj))
println("    Linf            = ", linf)
println("    bit-identical   = ", nexact, "/", length(dj), " cells")
println("    ⇒ ", linf == 0.0 ? "BIT-FOR-BIT IDENTICAL (Linf = 0) ✓" :
                  linf < 1e-12 ? "identical to machine precision" : "DIVERGENT ✗")
