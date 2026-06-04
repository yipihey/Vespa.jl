# ADR-0005 #2 — the bridge worker process (generic).
#
# Loads the Enzo bridge in-process (`:local`) and serves the FULL bridge surface
# over the control channel + shared file: every C symbol is dispatched by a thunk
# generated from the manifest parsed out of session.jl's `@xcall` sites.  Unlike
# the #1 prototype, this worker pre-inits NOTHING — `session_init` (and every other
# call) is itself an RPC, so the client drives the whole hierarchy lifecycle.  The
# worker's cwd is set by the launcher (Enzo reads problem files relative to cwd).
#
# Usage:  julia --project=<test> rpc_worker.jl <shm_path>
using EnzoLib, EnzoFixtures, EnzoNG, MeshInterface, RefMesh, EnzoBackend

EnzoLib.serve(; shm = ARGS[1])
