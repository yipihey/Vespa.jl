# ADR-0005 prototype — the differential-oracle gate.
#
# Proves the subprocess boundary on its smallest surface: the SAME bridge calls
# routed (a) in-process via ccall and (b) to a separate worker process via the
# control channel + zero-copy shared memory must agree BIT-IDENTICALLY.  This is
# the mechanism that makes the design bug-resistant: an RPC path that diverges
# from the proven local path fails here.  Uses the serial bridge (the boundary is
# transport-agnostic; MPI worker hosting is orthogonal).

using Test
using EnzoLib, EnzoFixtures, EnzoNG, MeshInterface, RefMesh, EnzoBackend
using Mmap

const RPC_PF = abspath(joinpath(@__DIR__, "..", "..", "..", "..",
                                "run", "Hydro", "Hydro-1D", "SodShockTube", "SodShockTubeAMR.enzo"))
const RPC_WORKER = joinpath(@__DIR__, "rpc_worker.jl")

# A typed remote call over the control channel.  Bulk arrays come back through the
# shm region; the descriptor `OK <dtype> <len>` is validated before we read it.
struct Remote; inp::Pipe; outp::Pipe; shm::String; proc::Base.Process; end

function rpc_scalar(r::Remote, cmd::AbstractString)
    println(r.inp, cmd); flush(r.inp)
    return strip(readline(r.outp))
end

function rpc_field(r::Remote, field::Int, grid::Int, expect_len::Int)
    println(r.inp, "get_field $field $grid"); flush(r.inp)
    desc = split(readline(r.outp))                       # "OK f64 N"
    @assert desc[1] == "OK"     "worker error: $(join(desc,' '))"
    @assert desc[2] == "f64"    "unexpected dtype: $(desc[2])"   # descriptor validation
    n = parse(Int, desc[3])
    @assert n == expect_len     "length mismatch: got $n want $expect_len"  # no silent corruption
    io = open(r.shm); v = copy(Mmap.mmap(io, Vector{Float64}, n)); close(io)
    return v
end

if !EnzoLib.grid_available()
    @info "Session bridge not built — skipping ADR-0005 RPC parity prototype"
else
    @testset "ADR-0005: local ≡ remote parity (subprocess boundary prototype)" begin
        # (a) LOCAL reference, via in-process ccall.
        hL = cd(EnzoLib._workdir(RPC_PF)) do
            EnzoLib.session_init(RPC_PF)
        end
        @test hL != C_NULL
        loc_rank  = EnzoLib.session_my_rank(hL)
        loc_ngrid = EnzoLib.problem_num_grids(hL)
        di = EnzoLib.field_index(hL, 0; grid = 0)        # Density field index on root grid
        loc_field = EnzoLib.problem_get_field(hL, di, 0)::Vector{Float64}

        # (b) REMOTE, via a worker process + shm.
        shm = tempname()
        write(shm, zeros(UInt8, sizeof(Float64) * length(loc_field)))
        jl = Base.julia_cmd()
        inp = Pipe(); outp = Pipe()
        proc = run(pipeline(`$jl --project=$(@__DIR__) $RPC_WORKER $shm $RPC_PF`;
                            stdin = inp, stdout = outp, stderr = stderr); wait = false)
        close(inp.out); close(outp.in)
        r = Remote(inp, outp, shm, proc)
        try
            @test strip(readline(r.outp)) == "ready"     # handshake

            rem_rank  = parse(Int, rpc_scalar(r, "my_rank"))
            rem_ngrid = parse(Int, rpc_scalar(r, "num_grids"))
            rem_field = rpc_field(r, di, 0, length(loc_field))

            @test rem_rank  == loc_rank                  # control-channel scalar round-trip
            @test rem_ngrid == loc_ngrid
            @test rem_field == loc_field                 # BIT-IDENTICAL shm array parity
            @info "ADR-0005 parity" rank = rem_rank ngrids = rem_ngrid field_len = length(rem_field)
        finally
            try; println(r.inp, "quit"); flush(r.inp); catch; end
            close(r.inp); wait(r.proc)
            EnzoLib.free_problem(hL)
            rm(shm; force = true)
        end
    end
end
