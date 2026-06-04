# ADR-0004 multi-rank smoke gate — the Enzo substrate under MPI (MPItrampoline).
# This is NOT part of the in-process runtests.jl; it is launched across ranks.
# MPITRAMPOLINE_LIB selects the MPIwrapper (real MPI); ENZONG_MPITRAMPOLINE points
# at the host libmpitrampoline so EnzoLib can promote it to global scope for the
# bridge's flat MPI_* resolution (see EnzoLib `_promote_mpitrampoline`):
#
#   T=$(julia --project=lib/EnzoLib/test -e \
#       'import MPItrampoline_jll as M; print(realpath(M.libmpitrampoline))')
#   MPITRAMPOLINE_LIB=$HOME/opt/mpiwrapper/lib/libmpiwrapper.so \
#   ENZONG_MPITRAMPOLINE=$T \
#   /opt/homebrew/bin/mpiexec -n 2 <julia> --project=lib/EnzoLib/test \
#       lib/EnzoLib/test/test_mpi_session.jl
#
# It checks that (a) the bridge sees the real rank count, (b) MPI actually
# partitioned the root grid across ranks, (c) original Enzo (`:enzo` hydro) is
# globally conservative under MPI, and (d) the conservative `:julia` hydro slot —
# iterating only local grids (ADR-0004) while Enzo's UpdateFromFinerGrids moves
# flux corrections across ranks — is globally conservative too.
#
# Global totals are an MPI_Allreduce over each rank's LOCAL level-0 grids (the
# root grid is split into one tile per rank by CommunicationPartitionGrid).

ENV["ENZONG_ENZO_MPI"] = "1"          # select the MPI bridge dylib (before any session ccall)
using MPI
using Test
using EnzoLib, EnzoFixtures
using EnzoNG, MeshInterface, RefMesh, EnzoBackend

MPI.Init()
const COMM = MPI.COMM_WORLD
const RANK = MPI.Comm_rank(COMM)
const NRANKS = MPI.Comm_size(COMM)

# Reuse the reflux harness helpers (the :julia hook, _build_grid_sim, …) without
# running its in-process testset.
ENV["REFLUX_NOTEST"] = "1"
include("test_julia_reflux.jl")
delete!(ENV, "REFLUX_NOTEST")

# Active mass+energy of ONE grid (per-grid generalization of read_root_totals).
function grid_totals(h, gi; nghost = 3)
    dims = EnzoLib.problem_grid_dims(h, gi)
    l, r = EnzoLib.problem_grid_edge(h, gi)
    rank = EnzoLib.problem_grid_rank(h, gi)
    active = ntuple(d -> dims[d] - 2nghost, rank)
    cw = ntuple(d -> (r[d] - l[d]) / active[d], rank)
    Vcell = prod(cw)
    strides = ntuple(d -> d == 1 ? 1 : prod(ntuple(k -> dims[k], d - 1)), rank)
    di = EnzoLib.field_index(h, 0; grid = gi)
    ei = EnzoLib.field_index(h, 1; grid = gi)
    dens = EnzoLib.problem_get_field(h, di, gi)
    espec = EnzoLib.problem_get_field(h, ei, gi)
    m = 0.0; e = 0.0
    for I in CartesianIndices(active)
        f = 1 + sum((nghost + I[d] - 1) * strides[d] for d in 1:rank)
        ρ = dens[f]; m += ρ * Vcell; e += ρ * espec[f] * Vcell
    end
    return (mass = m, energy = e)
end

# Global composite total = Allreduce over this rank's LOCAL level-0 grids.
function global_totals(h)
    m = 0.0; e = 0.0
    for gi in EnzoLib.local_grids_on_level(h, 0)
        t = grid_totals(h, gi); m += t.mass; e += t.energy
    end
    return (mass = MPI.Allreduce(m, +, COMM), energy = MPI.Allreduce(e, +, COMM))
end

function run_mpi(pf; hydro::Symbol, nsteps::Int)
    hooks = hydro === :julia ?
        Dict{Symbol,Function}(:hydro =>
            conservative_julia_hydro_hook(; conservative = true, parent_ghost = false)) :
        Dict{Symbol,Function}()
    eng = EnzoLib.EngineConfig(; hydro = hydro, reflux = (hydro === :julia), hooks = hooks)
    cd(EnzoLib._workdir(pf)) do
        h = EnzoLib.session_init(pf); h == C_NULL && error("session_init failed")
        try
            EnzoLib.session_rebuild(h, 0)
            t0 = global_totals(h)
            n = 0
            while EnzoLib.session_time(h) < EnzoLib.session_stop_time(h) && n < nsteps
                EnzoLib.evolve_level!(h, 0, 0.0; engine = eng, regrid = false)
                n += 1
            end
            return (t0 = t0, t1 = global_totals(h),
                    nranks = EnzoLib.session_num_ranks(h),
                    nl0 = length(EnzoLib.grids_on_level(h, 0)), cycles = n)
        finally
            EnzoLib.free_problem(h)
        end
    end
end

drift(r) = (mass = abs(r.t1.mass - r.t0.mass) / r.t0.mass,
            energy = abs(r.t1.energy - r.t0.energy) / r.t0.energy)

re = run_mpi(REFLUX_PF; hydro = :enzo,  nsteps = 25)
rj = run_mpi(REFLUX_PF; hydro = :julia, nsteps = 25)

if RANK == 0
    @info "ADR-0004 MPI smoke" nranks = re.nranks grids_on_level0 = re.nl0 cycles = re.cycles
    @info "drift" enzo = drift(re) julia = drift(rj)
    @testset "ADR-0004: Enzo substrate under MPI (n=$NRANKS)" begin
        @test NRANKS > 1                  # launched multi-rank
        @test re.nranks == NRANKS         # the bridge reports the real rank count
        @test re.nl0 >= NRANKS            # the root grid was partitioned across ranks
        @test drift(re).mass   < 1e-10    # original Enzo: globally conservative under MPI
        @test drift(re).energy < 1e-10
        @test drift(rj).mass   < 1e-3     # :julia slot (local grids): globally conservative
    end
end

MPI.Barrier(COMM)
MPI.Finalize()
