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
    ("Cosmology/ZeldovichPancake/ZeldovichPancake.enzo",     1e-3, "cosmology: gravity + comoving expansion"),
    # AMR+cosmology, StopCycle=5: density matches to ~7e-4; residual ~1.7% is the
    # cold-gas Hubble-drag/gravity coupling at 16-cell resolution (operator split,
    # 2e-5 at 256 cells). Gated at 2.5e-2 to catch regression to the old blow-up.
    ("Cosmology/AMRZeldovichPancake/AMRZeldovichPancake.enzo", 2.5e-2, "AMR + cosmology, StopCycle-limited"),
]

# Particle-only gravity problems (no baryon fields): compare final particle
# positions instead. Bit-for-bit (perr=0) — the Julia EvolveLevel moves the same
# particles with the same Enzo routines.
const PARTICLE_CASES = [
    ("GravitySolver/TestOrbit/TestOrbit.enzo",       1e-12, "2-body orbit, particle gravity"),
    ("GravitySolver/GravityTest/GravityTest.enzo",   1e-12, "5000 particles, AMR self-gravity"),
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

    # Particle (and gravity) problems leak Enzo process globals across sessions,
    # so — like the discovery harness — run each in its OWN subprocess (cmp_one.jl)
    # and parse the RESULT line, rather than calling compare_problem in-process.
    @testset "Enzo particle problems: positions match EvolveHierarchy" begin
        worker = joinpath(@__DIR__, "cmp_one.jl")
        for (rel, tol, note) in PARTICLE_CASES
            pf = joinpath(RUN_DIR, rel)
            out = read(setenv(`$(Base.julia_cmd()) --project=$(@__DIR__) $worker $pf`, ENV), String)
            ri = findfirst(l -> startswith(l, "RESULT|"), split(out, '\n'))
            @test ri !== nothing
            line = split(out, '\n')[ri]
            getf(k) = (m = match(Regex("\\|$k=([^|]+)"), line); m === nothing ? "" : m.captures[1])
            perr = tryparse(Float64, getf("perr")); npart = tryparse(Int, getf("nparticles"))
            @info "particle problem" problem = basename(dirname(pf)) particle_error = perr nparticles = npart note
            @test npart !== nothing && npart >= 1
            @test perr !== nothing && perr < tol
        end
    end
end
