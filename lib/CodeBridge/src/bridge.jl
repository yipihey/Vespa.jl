# ── Bridge: one wrapped legacy code ───────────────────────────────────────────
#
# A Bridge bundles everything `@xcall` needs about one legacy code: its library
# flavors (CPU/Metal, serial/MPI — same symbols, different builds), the active
# transport backend (:local ccall | :remote worker RPC), the live worker
# connection, and the wire-contract configuration (which source files carry the
# @xcall sites, and the contract seed string).
#
# `home` is the module the wrapper lives in: thunk generation (rpc.jl) evaluates
# the manifest's literal type expressions there, so wrapper-local type aliases
# (e.g. EnzoLib's `Handle = Ptr{Cvoid}`) resolve exactly as they do at the
# original call sites.

mutable struct WorkerConn
    proc::Base.Process
    cin::IO          # control channel: client → worker (requests)
    cout::IO         # control channel: worker → client (replies)
    shm::String      # shared-file path for bulk array args
end

mutable struct Bridge
    name::Symbol
    home::Module
    libs::Dict{Symbol,LazyLib}
    flavor::Base.RefValue{Symbol}
    backend::Base.RefValue{Symbol}                       # :local | :remote
    conn::Union{Nothing,WorkerConn}
    manifest_files::Vector{String}
    contract_seed::String
    _manifest::Union{Nothing,Dict{Symbol,Tuple{Any,Any}}}  # lazy cache
end

"""
    Bridge(name, home::Module; libs, flavor=nothing, manifest_files=String[],
           contract_seed="codebridge-contract-v1")

Declare a wrapped legacy code. `libs` maps flavor symbols to `LazyLib`s; with a
single entry `flavor` may be omitted. `manifest_files` are the wrapper source
files whose `@xcall` sites define the wire contract (only needed for the worker
transport); `contract_seed` versions that contract.
"""
function Bridge(name::Symbol, home::Module; libs::Dict{Symbol,LazyLib},
                flavor::Union{Nothing,Symbol} = nothing,
                manifest_files::Vector{String} = String[],
                contract_seed::AbstractString = "codebridge-contract-v1")
    isempty(libs) && error("Bridge $name: at least one LazyLib is required")
    fl = flavor === nothing ?
         (length(libs) == 1 ? first(keys(libs)) :
          error("Bridge $name has $(length(libs)) lib flavors; pass flavor=…")) : flavor
    haskey(libs, fl) || error("Bridge $name: unknown flavor :$fl")
    return Bridge(name, home, libs, Ref(fl), Ref(:local), nothing,
                  manifest_files, String(contract_seed), nothing)
end

"The active-flavor `LazyLib` (or a specific flavor's)."
lib(b::Bridge) = b.libs[b.flavor[]]
lib(b::Bridge, flavor::Symbol) = haskey(b.libs, flavor) ? b.libs[flavor] :
    error("bridge $(b.name): unknown lib flavor :$flavor (have $(sort!(collect(keys(b.libs))))) ")

"Resolve a C symbol in the active (or a specific) flavor's library."
sym(b::Bridge, name::Symbol) = sym(lib(b), name)
sym(b::Bridge, name::Symbol, flavor::Symbol) = sym(lib(b, flavor), name)

"Select the active lib flavor (e.g. `:cpu` vs `:metal`) for subsequent local calls."
flavor!(b::Bridge, f::Symbol) = (lib(b, f); b.flavor[] = f; nothing)
flavor(b::Bridge) = b.flavor[]

libpath(b::Bridge) = libpath(lib(b))
libpath(b::Bridge, f::Symbol) = libpath(lib(b, f))
available(b::Bridge) = available(lib(b))
available(b::Bridge, f::Symbol) = available(lib(b, f))

"The active transport: `:local` (in-process ccall) or `:remote` (worker RPC)."
backend(b::Bridge) = b.backend[]
set_backend!(b::Bridge, s::Symbol) =
    (s in (:local, :remote) || error("backend must be :local or :remote"); b.backend[] = s)

# ── @xcall: one call macro, two transports (ADR-0005) ────────────────────────
"""
    @xcall(:c_symbol, RetType, (ArgTypes...), args...)

Backend-dispatching bridge call against the calling module's `const BRIDGE`.
Local → `ccall(CodeBridge.sym(BRIDGE, :c_symbol), RetType, (ArgTypes...),
args...)` (literal types preserved, byte-identical to a hand-written binding);
remote → `CodeBridge._rpc(...)` to the connected worker.

The 3-argument head (symbol, return type, argtypes tuple) is also the wire
contract: `manifest` parses these sites out of the wrapper's source, so write
the symbol as a literal `:quoted` Symbol and the types as literals.
"""
macro xcall(csym, ret, argtypes, args...)
    a = map(esc, args)
    B = esc(:BRIDGE)
    symf = GlobalRef(@__MODULE__, :sym)
    rpcf = GlobalRef(@__MODULE__, :_rpc)
    quote
        if $B.backend[] === :local
            ccall($symf($B, $(esc(csym))), $(esc(ret)), $(esc(argtypes)), $(a...))
        else
            $rpcf($B, $(esc(csym)), $(esc(ret)), $(esc(argtypes)), ($(a...),))
        end
    end
end

# ── manifest: symbol => (ret_expr, argtypes_expr), parsed from the @xcall sites ──
# Walk each manifest file's AST for `@xcall(:sym, RET, (ARGTYPES...), args...)`
# and record the C symbol with its return-type and argtypes EXPRESSIONS (the
# literal types the worker needs to build a type-stable ccall).  Using Julia's
# own parser — robust to nested parens, no regex, and genuinely derived from
# the bindings source.
function collect_manifest(files::AbstractVector{<:AbstractString})
    man = Dict{Symbol,Tuple{Any,Any}}()
    for f in files
        src = read(f, String)
        top = Meta.parseall(src)
        function walk(x)
            if x isa Expr
                if x.head === :macrocall && !isempty(x.args) && x.args[1] === Symbol("@xcall")
                    a = filter(e -> !(e isa LineNumberNode), x.args[2:end])
                    length(a) >= 3 || error("malformed @xcall in $f: $x")
                    s = a[1] isa QuoteNode ? a[1].value : a[1]
                    s isa Symbol ||
                        error("@xcall symbol must be a literal :Symbol (got $(a[1])) in $f")
                    man[s] = (a[2], a[3])
                end
                foreach(walk, x.args)
            end
            return nothing
        end
        walk(top)
    end
    isempty(man) &&
        error("no @xcall sites found in $(join(files, ", ")) — manifest empty")
    return man
end

"The bridge manifest (symbol => (ret, argtypes) ASTs) parsed from the wrapper's @xcall sites."
function manifest(b::Bridge)
    b._manifest === nothing || return b._manifest
    isempty(b.manifest_files) &&
        error("bridge $(b.name) has no manifest_files; the worker transport needs them")
    b._manifest = collect_manifest(b.manifest_files)
    return b._manifest
end

"""
    contract_canonical(b) -> String

The canonical, language-independent serialization of the bridge surface: the
seed tag followed by one `\\nsym|ret|argtypes` line per symbol in sorted order.
Both the Julia client and any generated foreign worker hash THIS exact string,
so their handshake values agree without the worker reimplementing Julia
internals.  (Byte-identical to EnzoLib's original — baked worker hashes stay
valid.)
"""
function contract_canonical(b::Bridge)
    m = manifest(b)
    io = IOBuffer()
    print(io, b.contract_seed)
    for k in sort!(collect(keys(m)); by = string)
        ret, at = m[k]
        print(io, '\n', k, '|', ret, '|', at)
    end
    return String(take!(io))
end

"FNV-1a 64-bit (the one hash foreign worker generators also compute — reproducible, version-stable)."
function fnv1a64(s::AbstractString)
    h = 0xcbf29ce484222325
    for b in codeunits(s)
        h = (h ⊻ UInt64(b)) * 0x100000001b3      # UInt64 arithmetic wraps mod 2^64
    end
    return h
end

"""
    contract_hash(b) -> UInt64

A stable hash over the bridge surface (`contract_canonical`).  Exchanged at the
worker handshake; a mismatch means the worker and client were built from
different bindings (a rebuilt-one-side bug, incl. a stale generated dispatch)
and the connection is refused rather than silently corrupting data.
"""
contract_hash(b::Bridge) = fnv1a64(contract_canonical(b))
