# ── Phase 1 gate (ADR-0006 D2): N legacy codes alive in ONE Julia session ─────
#
# Enzo (Sod shock tube, AMR) and RAMSES (Sedov blast, unsplit MUSCL hydro) each
# run in their OWN worker process — Enzo through the generated C++ worker (or
# the Julia reference worker), RAMSES through the RamsesLib Julia worker — and
# are stepped INTERLEAVED, cycle by cycle, from one Julia driver.  Arepo joins
# as a third worker for its contract surface (its init is gated separately).
#
# The gate is differential: each code's multi-worker, interleaved result must be
# BIT-IDENTICAL to its single-code, in-process (`:local`) reference run.  Any
# cross-talk between bridges (shared state, wrong shm, contract mix-up) or any
# transport divergence fails the comparison.

using Test
using CodeBridge
using EnzoLib
using RamsesLib
using ArepoLib

const R = RamsesLib

# ── problem setup ─────────────────────────────────────────────────────────────
const ENZO_PF = abspath(joinpath(@__DIR__, "..", "..", "..",
                                 "run", "Hydro", "Hydro-1D", "SodShockTube", "SodShockTubeAMR.enzo"))
const ENZO_WORKER_CPP = abspath(joinpath(@__DIR__, "..", "..", "..",
                                         "EnzoModules", "deps", "enzomodules_worker"))
const ENZO_WORKER_JL  = abspath(joinpath(@__DIR__, "..", "..", "lib", "EnzoLib", "test", "rpc_worker.jl"))
const ENZO_TESTPROJ   = abspath(joinpath(@__DIR__, "..", "..", "lib", "EnzoLib", "test"))

const RAMSES_HYDRO_LIB = get(ENV, "RAMSES_HYDRO_LIB",
    normpath(joinpath(@__DIR__, "..", "..", "..", "..",
                      "mini-ramses", "bin64h", "libramses3d.dylib")))
const RAMSES_NML = get(ENV, "RAMSES_NML",
    normpath(joinpath(@__DIR__, "..", "..", "..", "..",
                      "mini-ramses", "namelist", "sedov3d.nml")))
const RAMSES_PROJ = normpath(joinpath(@__DIR__, "..", "..", "..", "..",
                                      "RamsesNG.jl", "lib", "RamsesLib"))
const RAMSES_WORKER = joinpath(RAMSES_PROJ, "test", "rpc_worker.jl")

const NSTEPS = 8           # interleaved cycles (each: one Enzo cycle + one RAMSES step)

# ── per-code step drivers (IDENTICAL code for local and remote runs) ──────────

# One Julia-driven Enzo evolution: session_init → NSTEPS × (set_boundary →
# compute_dt → set_dt → solve_hydro → advance_time) → final density.
# `oneach(k)` is the interleave hook, called after each Enzo cycle.
function enzo_run(pf; oneach = k -> nothing)
    h = EnzoLib.session_init(pf)
    @assert h != C_NULL "session_init failed for $pf"
    try
        for k in 1:NSTEPS
            EnzoLib.session_set_boundary(h, 0)
            dt = EnzoLib.session_compute_dt(h, 0)
            EnzoLib.session_set_dt(h, dt, 0)
            EnzoLib.session_solve_hydro(h, 0)
            EnzoLib.session_advance_time(h, 0)
            oneach(k)
        end
        return EnzoLib.read_density(h)
    finally
        EnzoLib.free_problem(h)
    end
end

# RAMSES Sedov: init from the namelist (read by the OWNING process relative to
# its cwd), then return a per-step closure + a finisher reading the final state.
function ramses_open(nml)
    h = R.init(nml)
    lev = R.info(h).levelmin
    step = function (k)
        R.newdt_fine!(h, lev)
        R.hydro_step!(h, lev; dt = R.get_dt(h, lev).dtnew)
        return nothing
    end
    finish = function ()
        ck, d = R.get_hydro(h, :uold, 1, lev)    # density + the oct join key
        _, e = R.get_hydro(h, :uold, 5, lev)     # total energy
        return (ckey = ck, rho = d, etot = e)
    end
    return (step = step, finish = finish)
end

# ── the gate ──────────────────────────────────────────────────────────────────
enzo_ok   = EnzoLib.grid_available() && isfile(ENZO_PF)
ramses_ok = isfile(RAMSES_HYDRO_LIB) && isfile(RAMSES_NML)

if !(enzo_ok && ramses_ok)
    @warn "multicode gate skipped" enzo_ok ramses_ok ENZO_PF RAMSES_HYDRO_LIB
    @testset "multicode workers (skipped)" begin
        @test_skip enzo_ok && ramses_ok
    end
else
    ENV["RAMSES_LIB"] = RAMSES_HYDRO_LIB         # hydro flavor, local AND worker

    # a scratch run directory for RAMSES (namelist read relative to cwd)
    ramses_dir = mktempdir()
    cp(RAMSES_NML, joinpath(ramses_dir, "sedov3d.nml"))

    @testset "ADR-0006 D2: multi-worker sessions" begin
        # (a) single-code LOCAL references, one code at a time.
        @assert EnzoLib.backend() === :local && CodeBridge.backend(R.BRIDGE) === :local
        enzo_wd = EnzoLib._workdir(ENZO_PF)
        rho_enzo_ref = cd(() -> enzo_run(ENZO_PF), enzo_wd)
        ramses_ref = cd(ramses_dir) do
            s = ramses_open("sedov3d.nml")
            foreach(s.step, 1:NSTEPS)
            s.finish()
        end
        m0 = sum(ramses_ref.rho)

        # (b) connect BOTH workers — two legacy codes live at once.
        jl = Base.julia_cmd()
        enzo_shm = tempname(); ramses_shm = tempname()
        enzo_cmd = isfile(ENZO_WORKER_CPP) ?
            setenv(`$ENZO_WORKER_CPP $enzo_shm $(EnzoLib.grid_libpath())`; dir = enzo_wd) :
            setenv(`$jl --project=$ENZO_TESTPROJ $ENZO_WORKER_JL $enzo_shm`; dir = enzo_wd)
        EnzoLib.connect_worker!(enzo_cmd; shm = enzo_shm)
        CodeBridge.connect_worker!(R.BRIDGE,
            setenv(`$jl --project=$RAMSES_PROJ $RAMSES_WORKER $ramses_shm`; dir = ramses_dir);
            shm = ramses_shm)
        try
            @test EnzoLib.backend() === :remote
            @test CodeBridge.backend(R.BRIDGE) === :remote          # SIMULTANEOUSLY live

            # (c) INTERLEAVED stepping: each Enzo cycle is followed by a RAMSES
            # step, both via RPC to their respective workers, one driver loop.
            s = ramses_open("sedov3d.nml")
            rho_enzo_multi = enzo_run(ENZO_PF; oneach = s.step)
            ramses_multi = s.finish()

            # (d) the differential gate: bit-identical to the single-code runs.
            @test rho_enzo_multi == rho_enzo_ref                     # Enzo, bit-for-bit
            @test ramses_multi.ckey == ramses_ref.ckey               # same mesh
            @test ramses_multi.rho == ramses_ref.rho                 # RAMSES, bit-for-bit
            @test ramses_multi.etot == ramses_ref.etot
            # physics sanity: the blast developed and conserved mass
            @test maximum(ramses_multi.rho) > 1.000001 * minimum(ramses_multi.rho)
            @test abs(sum(ramses_multi.rho) - m0) / m0 < 1e-12

            # (e) a THIRD code joins: Arepo's contract surface over its own worker.
            if ArepoLib.available()
                p_local = ArepoLib.precision_bytes()
                arepo_shm = tempname()
                arepo_worker = tempname() * ".jl"
                write(arepo_worker, "using ArepoLib; ArepoLib.serve(; shm = ARGS[1])\n")
                CodeBridge.connect_worker!(ArepoLib.BRIDGE,
                    `$jl --project=$(pkgdir(ArepoLib)) $arepo_worker $arepo_shm`;
                    shm = arepo_shm)
                try
                    @test CodeBridge.backend(ArepoLib.BRIDGE) === :remote
                    @test EnzoLib.backend() === :remote              # all three live
                    @test CodeBridge.backend(R.BRIDGE) === :remote
                    @test ArepoLib.precision_bytes() == p_local      # remote ≡ local
                    @info "three codes live in one session" enzo = :remote ramses = :remote arepo = :remote
                finally
                    CodeBridge.disconnect_worker!(ArepoLib.BRIDGE)
                    rm(arepo_shm; force = true); rm(arepo_worker; force = true)
                end
            else
                @info "Arepo library not built — two-code gate only"
            end

            @info "ADR-0006 D2 gate" enzo_cells = length(rho_enzo_multi) ramses_octs = size(ramses_multi.ckey, 1) steps = NSTEPS
        finally
            EnzoLib.disconnect_worker!()
            CodeBridge.disconnect_worker!(R.BRIDGE)
            rm(enzo_shm; force = true); rm(ramses_shm; force = true)
        end
        @test EnzoLib.backend() === :local
        @test CodeBridge.backend(R.BRIDGE) === :local
    end
end
