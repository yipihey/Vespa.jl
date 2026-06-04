# ADR-0005 #2 — the differential-oracle gate (full bridge surface).
#
# The SAME high-level EnzoLib wrappers, run through (a) the in-process `:local`
# ccall backend and (b) a separate worker process (`:remote`) over the control
# channel + shared file, must agree BIT-IDENTICALLY.  Because the wrappers are
# unchanged between backends, this exercises the whole `@xcall` surface at once:
# scalar returns (Cint/Cdouble/Handle), every OUT-buffer shape (Float64, Int32,
# Int64, and multi-buffer calls like grid_edge), and an IN-buffer round-trip
# (set_field → get_field).  An RPC path that diverges from the proven local path —
# a marshalling bug, a wrong buffer length, a contract drift — fails here.
#
# Uses the serial bridge; the boundary is transport-agnostic (MPI/C++ worker
# hosting is #3, orthogonal to this protocol).

using Test
using EnzoLib, EnzoFixtures, EnzoNG, MeshInterface, RefMesh, EnzoBackend

const RPC_PF = abspath(joinpath(@__DIR__, "..", "..", "..", "..",
                                "run", "Hydro", "Hydro-1D", "SodShockTube", "SodShockTubeAMR.enzo"))
const RPC_WORKER = joinpath(@__DIR__, "rpc_worker.jl")

# A battery of bridge reads (+ one write round-trip) over the whole surface.  Runs
# against whatever backend is active; returns an ordered name=>value list to diff.
function _gather(h)
    out = Pair{String,Any}[]
    rec(k, v) = push!(out, k => v)

    rec("my_rank",   EnzoLib.session_my_rank(h))         # Cint scalar
    rec("num_ranks", EnzoLib.session_num_ranks(h))
    rec("num_grids", EnzoLib.problem_num_grids(h))
    rec("time",      EnzoLib.session_time(h))            # Cdouble scalar
    rec("stop_time", EnzoLib.session_stop_time(h))
    rec("cycle",     EnzoLib.session_cycle(h))
    rec("dt0",       EnzoLib.session_compute_dt(h, 0))   # Cdouble, global-min under MPI

    ng = EnzoLib.problem_num_grids(h)
    rec("grids_on_level0",       EnzoLib.grids_on_level(h, 0))
    rec("local_grids_on_level0", EnzoLib.local_grids_on_level(h, 0))
    for g in 0:ng-1
        rec("grid_size[$g]",   EnzoLib.problem_grid_size(h, g))
        rec("num_fields[$g]",  EnzoLib.problem_num_fields(h, g))
        rec("grid_rank[$g]",   EnzoLib.problem_grid_rank(h, g))
        rec("grid_level[$g]",  EnzoLib.problem_grid_level(h, g))
        rec("grid_proc[$g]",   EnzoLib.problem_grid_processor(h, g))
        rec("field_types[$g]", EnzoLib.problem_field_types(h, g))      # OUT Int32 buffer
        rec("grid_dims[$g]",   EnzoLib.problem_grid_dims(h, g))        # OUT Int32 buffer
        rec("global_start[$g]", EnzoLib.problem_grid_global_start(h, g)) # OUT Int64 buffer
        l, r = EnzoLib.problem_grid_edge(h, g)                          # 2× OUT Float64 buffers
        rec("grid_edge_l[$g]", l); rec("grid_edge_r[$g]", r)
    end

    di = EnzoLib.field_index(h, 0; grid = 0)             # Density field index (root grid)
    rec("density_index", di)
    rho = EnzoLib.problem_get_field(h, di, 0)            # OUT Float64 buffer (the big one)
    rec("density", rho)

    # IN-buffer round-trip: scale Density by 2, write it back, read it again.  This
    # exercises set_field (IN Float64 buffer) and proves the live grid mutated
    # identically across backends.
    scaled = rho .* 2.0
    EnzoLib.problem_set_field(h, di, scaled; grid = 0)
    rec("density_after_set", EnzoLib.problem_get_field(h, di, 0))

    return out
end

if !EnzoLib.grid_available()
    @info "Session bridge not built — skipping ADR-0005 RPC parity oracle"
else
    @testset "ADR-0005 #2: local ≡ remote parity (full bridge surface)" begin
        # (a) LOCAL reference — in-process ccall.
        EnzoLib.set_backend!(:local)
        hL = cd(EnzoLib._workdir(RPC_PF)) do
            EnzoLib.session_init(RPC_PF)
        end
        @test hL != C_NULL
        local_results = _gather(hL)
        EnzoLib.free_problem(hL)

        # (b) REMOTE — a worker process drives the SAME wrappers via RPC + shm.  The
        # worker is launched in the fixture's workdir (Enzo reads files relative to
        # cwd); session_init itself is an RPC, so the worker pre-inits nothing.
        shm = tempname()
        jl  = Base.julia_cmd()
        cmd = setenv(`$jl --project=$(@__DIR__) $RPC_WORKER $shm`;
                     dir = EnzoLib._workdir(RPC_PF))
        EnzoLib.connect_worker!(cmd; shm = shm)
        @test EnzoLib.backend() === :remote
        try
            hR = EnzoLib.session_init(RPC_PF)
            @test hR != C_NULL
            remote_results = _gather(hR)
            EnzoLib.free_problem(hR)

            @test length(local_results) == length(remote_results)
            for ((lk, lv), (rk, rv)) in zip(local_results, remote_results)
                @test lk == rk                          # same call order/surface
                @test lv == rv                          # BIT-IDENTICAL local vs remote
            end
            @info "ADR-0005 #2 parity" calls = length(local_results) backend = EnzoLib.backend()
        finally
            EnzoLib.disconnect_worker!()
            rm(shm; force = true)
        end
        @test EnzoLib.backend() === :local              # disconnect restored local
    end
end
