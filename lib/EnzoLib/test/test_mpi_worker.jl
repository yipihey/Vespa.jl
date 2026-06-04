# ADR-0005 #3b — multi-rank gate over the SUBPROCESS boundary.
#
# Supersedes the in-process MPI attempt (test_mpi_session.jl), which loaded the
# gcc/libstdc++ MPI bridge into the Julia process and aborted in std::locale static
# init (the ADR-0005 collision).  Here the Julia CLIENT carries NO MPI runtime: it
# spawns `mpiexec -n N enzomodules_worker_mpi` (a Julia-free C++ process) and drives
# the live hierarchy over the rpc.jl control channel — rank 0 owns the channel and
# broadcasts each command so collective bridge calls run in lockstep across ranks.
#
# This is launched as a plain script (NOT part of runtests.jl — it needs mpiexec +
# the MPItrampoline/MPIwrapper toolchain), and is a no-op (exits 0) when any piece
# is absent, so it is safe in default CI.  Run it directly:
#
#   MPITRAMPOLINE_LIB=$HOME/opt/mpiwrapper/lib/libmpiwrapper.so \
#   julia --project=lib/EnzoLib/test lib/EnzoLib/test/test_mpi_worker.jl
#
# (ENZONG_ENZO_MPI is NOT needed: the client speaks only RPC and never loads the
# bridge in-process; the worker is handed the MPI bridge path explicitly.)

using Test
using EnzoLib, EnzoFixtures, EnzoNG, MeshInterface, RefMesh, EnzoBackend

const MPIEXEC     = get(ENV, "ENZONG_MPIEXEC", "/opt/homebrew/bin/mpiexec")
const NRANKS      = parse(Int, get(ENV, "ENZONG_MPI_NRANKS", "2"))
const REPO        = abspath(joinpath(@__DIR__, "..", "..", "..", ".."))
const WORKER_MPI  = joinpath(REPO, "EnzoModules", "deps", "enzomodules_worker_mpi")
const MPI_BRIDGE  = joinpath(REPO, "EnzoModules", "deps", "libenzomodules_grid_mpi.dylib")
const MPI_PF      = joinpath(REPO, "run", "Hydro", "Hydro-1D", "SodShockTube", "SodShockTubeAMR.enzo")
const WRAPPER_LIB = get(ENV, "MPITRAMPOLINE_LIB", joinpath(homedir(), "opt", "mpiwrapper", "lib", "libmpiwrapper.so"))

# Skip cleanly unless the whole multi-rank toolchain is present.
function _mpi_ready()
    for (what, path) in (("mpiexec", MPIEXEC), ("worker_mpi", WORKER_MPI),
                         ("mpi bridge", MPI_BRIDGE), ("MPIwrapper", WRAPPER_LIB),
                         ("problem file", MPI_PF))
        isfile(path) || (@info "MPI worker gate skipped — missing $what" path; return false)
    end
    return true
end

if !_mpi_ready()
    @info "ADR-0005 #3b multi-rank gate skipped (toolchain not built)"
else
    @testset "ADR-0005 #3b: $(NRANKS)-rank Enzo substrate over the subprocess boundary" begin
        wd  = EnzoLib._workdir(MPI_PF)
        shm = tempname()
        env = copy(ENV); env["MPITRAMPOLINE_LIB"] = WRAPPER_LIB
        cmd = setenv(`$MPIEXEC -n $NRANKS $WORKER_MPI $shm $MPI_BRIDGE`, env; dir = wd)

        EnzoLib.connect_worker!(cmd; shm = shm)        # handshake verifies the contract hash
        @test EnzoLib.backend() === :remote
        try
            h = EnzoLib.session_init(MPI_PF)            # collective: CommunicationPartitionGrid
            @test h != C_NULL

            # (a) the bridge sees the real rank count, driven from rank 0.
            @test EnzoLib.session_num_ranks(h) == NRANKS
            @test EnzoLib.session_my_rank(h)   == 0

            # (b) MPI actually DISTRIBUTED the hierarchy: grid owners span >1 rank
            # (ReturnProcessorNumber is globally consistent, so rank 0 can report it).
            ng    = EnzoLib.problem_num_grids(h)
            @test ng > 0
            owners = sort(unique(EnzoLib.problem_grid_processor(h, g) for g in 0:ng-1))
            @info "MPI partition" nranks = NRANKS ngrids = ng grid_owners = owners
            @test length(owners) == NRANKS             # every rank owns ≥1 grid
            @test maximum(owners) == NRANKS - 1

            EnzoLib.free_problem(h)                     # collective free, per-rank handle
        finally
            EnzoLib.disconnect_worker!()               # QUIT → all ranks MPI_Finalize
            rm(shm; force = true)
        end
        @test EnzoLib.backend() === :local
    end
end
