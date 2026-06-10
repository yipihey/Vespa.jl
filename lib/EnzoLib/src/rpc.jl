# ── ADR-0005 #2: remote bridge transport — now provided by CodeBridge ─────────
#
# The subprocess-worker transport (control channel + shared-file bulk arrays),
# the manifest parsed from session.jl's @xcall sites, the canonical contract
# serialization and its FNV-1a hash, and the Julia reference worker all moved
# verbatim into lib/CodeBridge (ADR-0006 D1), parameterized by a `Bridge`.
# EnzoLib keeps its public names as thin delegators onto `BRIDGE`, so every
# caller (tests, the worker generator, EnzoBackend) is unchanged.
#
# The wire protocol and the contract canonicalization are byte-identical to the
# pre-extraction rpc.jl: `contract_hash()` is the same value, so the already-
# built C++ workers (serial/MPI/f32, with baked WORKER_CONTRACT_HASH) remain
# valid without a rebuild.

"The bridge manifest (symbol => (ret, argtypes) ASTs) parsed from session.jl's @xcall sites."
manifest() = CodeBridge.manifest(BRIDGE)

"""
    contract_canonical() -> String

The canonical, language-independent serialization of the bridge surface (see
`CodeBridge.contract_canonical`). Both the Julia client and the generated C++
worker hash THIS exact string, so their handshake values agree.
"""
contract_canonical() = CodeBridge.contract_canonical(BRIDGE)

"""
    contract_hash() -> UInt64

A stable hash over the bridge surface, exchanged at the worker handshake; a
mismatch refuses the connection rather than silently corrupting data.
"""
contract_hash() = CodeBridge.contract_hash(BRIDGE)

"""
    connect_worker!(cmd::Cmd; shm=tempname()) -> Nothing

Spawn a bridge worker process (`cmd` runs `EnzoLib.serve` or the C++ worker),
verify the contract-hash handshake, and switch the backend to `:remote`.
"""
connect_worker!(cmd::Base.AbstractCmd; shm::AbstractString = tempname()) =
    CodeBridge.connect_worker!(BRIDGE, cmd; shm = shm)

"Tear down the remote worker and return the backend to `:local`."
disconnect_worker!() = CodeBridge.disconnect_worker!(BRIDGE)

"""
    serve(; cin=stdin, cout=stdout, shm=ARGS[1])

The Julia reference worker loop for the Enzo bridge (see `CodeBridge.serve`).
"""
serve(; cin::IO = stdin, cout::IO = stdout, shm::AbstractString = ARGS[1]) =
    CodeBridge.serve(BRIDGE; cin = cin, cout = cout, shm = shm)
