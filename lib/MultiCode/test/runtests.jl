# ── Phase 2 gate (ADR-0006 D3): canonical state + cross-code comparison ───────
#
#  1. exact-solver self-checks (limits, jump conditions)
#  2. per code: run the ONE Sod spec natively → canonicalize → conservation
#     ledger vs the analytic reference → extract/inject!/extract ROUND-TRIP
#     bit-identical → L1 against the exact solution
#  3. the cross-code report builds and names every code that ran

using Test
using MultiCode
using CodeBridge, EnzoLib, RamsesLib, ArepoLib

# the RAMSES hydro build (the gravity-only bin64s default has no godunov)
const RAMSES_HYDRO_LIB = get(ENV, "RAMSES_HYDRO_LIB",
    normpath(joinpath(@__DIR__, "..", "..", "..", "..", "..",
                      "mini-ramses", "bin64h", "libramses3d.dylib")))
haskey(ENV, "RAMSES_LIB") || (ENV["RAMSES_LIB"] = RAMSES_HYDRO_LIB)

# Next-9 (MUSIC injector): generation runs in MUSIC's own CodeBridge worker
# (worker = true) — immune to the OpenMP/FFTW runtime pollution that made
# in-process generation segfault in a many-code host process (the D2 fix).
include("test_music_crosscheck.jl")
include("test_discodj_growth.jl")   # Next-10: DISCO-DJ ICs vs the exact growth (also boot-sensitive)

const spec = SodSpec()
const ref = MultiCode.sod_reference_ledger(spec)

@testset "MultiCode (ADR-0006 Phase 2)" begin

    @testset "exact Sod solver" begin
        # far-field limits
        @test MultiCode.exact_sod(spec, -10.0).rho ≈ spec.rhoL
        @test MultiCode.exact_sod(spec, 10.0).rho ≈ spec.rhoR
        # the classic Sod star state (Toro): p* ≈ 0.30313, u* ≈ 0.92745
        mid = MultiCode.exact_sod(spec, 0.5)      # between contact and shock
        @test mid.p ≈ 0.30313 atol = 2e-5
        @test mid.u ≈ 0.92745 atol = 2e-5
        # contact: density jumps, pressure/velocity continuous
        l = MultiCode.exact_sod(spec, 0.92)
        r = MultiCode.exact_sod(spec, 0.94)
        @test l.p ≈ r.p atol = 1e-10
        @test l.rho > r.rho
    end

    results = NamedTuple[]

    @testset "Enzo: native Sod → canonical" begin
        if !(EnzoLib.grid_available() && isfile(MultiCode.ENZO_SOD_PF))
            @test_skip false
        else
            r = run_enzo_sod(spec)
            try
                cs = r.cs
                lg = ledger(cs)
                # ledger vs analytic reference AND vs Enzo's own integral (geometry check)
                @test abs(lg.mass - ref.mass) / ref.mass < 1e-10
                @test abs(lg.mass - r.diag.mass_bridge) / ref.mass < 1e-10
                @test abs(lg.energy - ref.energy) / ref.energy < 1e-6
                # round-trip: extract → inject! → extract, bit-identical
                MultiCode.enzo_inject!(r.handle, cs)
                cs2 = enzo_extract(r.handle)
                rt = cs2.rho == cs.rho && cs2.mom == cs.mom && cs2.etot == cs.etot
                @test rt
                @test MultiCode.ledger_drift(ledger(cs2), lg) == 0.0
                # accuracy vs exact
                prof = r.profile
                l1 = sod_l1(prof, spec, r.t)
                @test l1.rho < 0.05
                push!(results, (code = :enzo, cs = cs, t = r.t, profile = prof, l1 = l1,
                                roundtrip = rt, notes = "PPM DirectEuler, 100 cells, 1-D; " *
                                "bridge mass integral matches the adapter to round-off."))
            finally
                r.free()
            end
        end
    end

    @testset "RAMSES: native Sod → canonical" begin
        if !RamsesLib.available()
            @test_skip false
        else
            r = run_ramses_sod(spec; level = 7)
            try
                cs = r.cs
                @test ncells(cs) == (2^7)^3
                lg = ledger(cs)
                # 2M-cell Float64 accumulation: round-off floor is O(n·eps) ≈ 1e-10
                @test abs(lg.mass - ref.mass) / ref.mass < 1e-10
                @test abs(lg.energy - ref.energy) / ref.energy < 1e-10
                # round-trip via set_hydro!/get_hydro, bit-identical
                MultiCode.ramses_inject!(r.handle, cs)
                cs2 = ramses_extract(r.handle; lev = r.diag.level, boxlen = 2.0)
                rt = cs2.rho == cs.rho && cs2.mom == cs.mom && cs2.etot == cs.etot &&
                     cs2.pos == cs.pos
                @test rt
                prof = r.profile
                @test length(prof.x) == 2^6                  # the window half of 128 x-points
                @test prof.scatter < 1e-10                   # transverse symmetry
                l1 = sod_l1(prof, spec, r.t)
                @test l1.rho < 0.06
                push!(results, (code = :ramses, cs = cs, t = r.t, profile = prof, l1 = l1,
                                roundtrip = rt, notes = "unsplit MUSCL + HLLC, 128³ uniform, " *
                                "3-D double-length tube (periodic seam outside the window); " *
                                "transverse density scatter $(round(prof.scatter; sigdigits=2))."))
            finally
                r.free()
            end
        end
    end

    @testset "Arepo: native Sod → canonical" begin
        arepo_dir = ArepoLib.available() ? normpath(dirname(ArepoLib.libpath())) : ""
        py_ok = ArepoLib.available() && MultiCode._arepo_python(arepo_dir) !== nothing
        if !py_ok
            @test_skip false
        else
            r = run_arepo_sod(spec)
            try
                cs = r.cs
                lg = ledger(cs)
                # moving-mesh + example tolerances (mass exact, energy to solver tolerance)
                @test abs(lg.mass - ref.mass) / ref.mass < 1e-6
                @test abs(lg.energy - ref.energy) / ref.energy < 1e-3
                rt = MultiCode.arepo_roundtrip_conserved(r.handle)
                @test rt
                prof = r.profile
                l1 = sod_l1(prof, spec, r.t)
                @test l1.rho < 0.06
                push!(results, (code = :arepo, cs = cs, t = r.t, profile = prof, l1 = l1,
                                roundtrip = rt, notes = "moving-mesh finite volume, 128 cells; " *
                                "box 20, TimeMax = 20·t̂ ⇒ the identical normalized problem; " *
                                "profile windowed to the periodic-seam-clean region $(r.diag.window)."))
            finally
                r.free()
            end
        end
    end

    @testset "the cross-code report" begin
        @test !isempty(results)
        dir = normpath(joinpath(@__DIR__, "..", "..", "..", "reports", "multicode"))
        md = sod_report(results, spec; dir = dir)
        @test isfile(md)
        @test isfile(joinpath(dir, "sod_profiles.svg"))
        txt = read(md, String)
        for r in results
            @test occursin(String(r.code), txt)
        end
        @info "Phase 2 report" path = md codes = [r.code for r in results] l1 = [r.l1.rho for r in results]
    end
end

include("test_ramses_slot.jl")     # Phase 3: the first cross-code slot (+ Metal, Phase 5)
include("test_moray_exchange.jl")  # Phase 4.1/4.2: Moray + exchange + flagship 3
include("test_rt_crosscheck.jl")   # Phase 4.4: Moray vs RAMSES-RT (the ADR gate)
include("test_rt_guest.jl")        # Phase 5: flagship 2 — RAMSES-RT inside Enzo
include("test_sedov_compare.jl")   # the science-grade run: one Sedov IC, four engines
include("test_amr_guest.jl")       # the guest slot under AMR (composite raster)
include("test_amr_fastpath.jl")    # Next-3: the per-level fast path (ghosts + flux registers)
include("test_gravity_slot.jl")    # Next-4: the gravity guest slot (KA Poisson vs RAMSES MG)
include("test_dfmm_engine.jl")     # Next-5: dfmm via the MultiCodeDfmmExt package extension
include("test_athena_engine.jl")   # Next-8: Athena++ via MultiCodeAthenaExt (registry on-ramp)
# Next-11: the GADGET-4 halo service — in its OWN PROCESS: HDF5.jl (via
# Gadget4Lib) and Enzo's grid dylib each carry a libhdf5, and the interposed
# symbols abort whichever loads second.  Process isolation, the D2 way.
@testset "GADGET-4 halo service (isolated process)" begin
    tf = joinpath(@__DIR__, "test_gadget4_halos.jl")
    cmd = `$(Base.julia_cmd()) --project=$(Base.active_project()) $tf`
    @test success(pipeline(cmd; stdout = stdout, stderr = stderr))
end

include("test_zeldovich.jl")       # Next-2: cosmology — one particle set, Enzo + RAMSES vs exact
include("test_chem_engines.jl")    # ChemistryKernels: :kernels engine ≡ Grackle reduced lib (sub-%)
