# Worker entry point for the RPC parity test: loads the same fixture bindings
# (same file ⇒ same manifest ⇒ same contract hash) and serves the bridge over
# stdin/stdout + the shared file in ARGS[1].
include(joinpath(@__DIR__, "fixture.jl"))
CBFixture.serve(; shm = ARGS[1])
