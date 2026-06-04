# ADR-0005 prototype — the worker side of the subprocess boundary.
#
# A separate process that loads the Enzo bridge, session_init's a fixture, and
# serves typed requests over stdin/stdout (the control channel).  Bulk field
# arrays are returned through an mmap'd shared file (zero-copy); only a typed
# descriptor `OK <dtype> <len>` crosses the control channel.  This is the
# hand-wired decisive slice; the full surface will be generated from the
# session.jl ccall manifest.
#
# Usage:  julia --project=<test> rpc_worker.jl <shm_path> <fixture.enzo>
using EnzoLib, EnzoFixtures, EnzoNG, MeshInterface, RefMesh, EnzoBackend
using Mmap

const SHM = ARGS[1]
const PF  = ARGS[2]

h = Ref{Any}(C_NULL)
cd(EnzoLib._workdir(PF)) do
    h[] = EnzoLib.session_init(PF)
end
h[] == C_NULL && (println(stdout, "FAIL session_init"); flush(stdout); exit(1))
println(stdout, "ready"); flush(stdout)   # handshake: session is up

# Control loop: one request per line, one response line.
while !eof(stdin)
    line = readline(stdin); isempty(line) && continue
    p = split(line)
    cmd = p[1]
    if cmd == "my_rank"
        println(stdout, EnzoLib.session_my_rank(h[]))
    elseif cmd == "num_ranks"
        println(stdout, EnzoLib.session_num_ranks(h[]))
    elseif cmd == "num_grids"
        println(stdout, EnzoLib.problem_num_grids(h[]))
    elseif cmd == "get_field"          # get_field <field> <grid>
        fi = parse(Int, p[2]); gi = parse(Int, p[3])
        v = EnzoLib.problem_get_field(h[], fi, gi)::Vector{Float64}
        io = open(SHM, "r+"); m = Mmap.mmap(io, Vector{Float64}, length(v))
        m .= v; Mmap.sync!(m); close(io)
        println(stdout, "OK f64 $(length(v))")   # self-describing descriptor
    elseif cmd == "quit"
        break
    else
        println(stdout, "ERR unknown:$cmd")
    end
    flush(stdout)
end
EnzoLib.free_problem(h[])
