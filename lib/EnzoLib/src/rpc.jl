# ── ADR-0005 #2: remote bridge transport (subprocess worker + shared memory) ──
#
# The `:remote` backend routes every `@xcall` to a worker PROCESS over a
# line-based control channel; bulk array arguments travel through a shared file
# (a stand-in for POSIX shm — the C++ worker of #3 mmaps it for true zero-copy;
# the protocol is identical).  This is what lets a foreign-C++-runtime worker
# (MPI `libenzo`, later Legion/Rust) drive the live hierarchy WITHOUT its C++
# stdlib ever entering the Julia process (the libstdc++/libc++ collision that
# blocked in-process MPI — see ADR-0005).
#
# Two properties make it bug-resistant by construction:
#
#  1. ONE contract.  The worker dispatches raw C symbols via `ccall` thunks
#     GENERATED from a manifest PARSED OUT OF `session.jl` itself (every `@xcall`
#     site).  There is no second hand-maintained interface to drift from the
#     bindings, and a `contract_hash` handshake rejects a rebuilt-one-side.
#
#  2. No in/out direction table.  Every array argument is round-tripped through
#     the shared file BIDIRECTIONALLY: the client ships the arg's current bytes
#     (real data for an IN buffer, zeros for an OUT buffer) and reads the bytes
#     back after the call (the C kernel's writes for an OUT buffer, unchanged for
#     an IN buffer).  The off-direction transfer is a few KB of zeros/no-ops, so
#     correctness needs only the element type + length — both carried in the
#     request — not knowledge of which Ptr args are written.  The buffer wrappers
#     in session.jl therefore need NO remote-specific code.
#
# The in-process `:local` path (ccall) remains the default and the differential
# ORACLE: `test_rpc_parity.jl` runs the SAME wrappers through both backends and
# asserts bit-identical results.

using Base64

# ── manifest: symbol => (ret_expr, argtypes_expr), parsed from the @xcall sites ──
const _MANIFEST = Ref{Union{Nothing,Dict{Symbol,Tuple{Any,Any}}}}(nothing)

# Walk session.jl's AST for `@xcall(:sym, RET, (ARGTYPES...), args...)` and record
# the C symbol with its return-type and argtypes EXPRESSIONS (the literal types the
# worker needs to build a type-stable ccall).  Using Julia's own parser — robust to
# nested parens, no regex, and genuinely derived from the bindings source.
function _collect_xcall_manifest()
    src = read(joinpath(@__DIR__, "session.jl"), String)
    top = Meta.parseall(src)
    man = Dict{Symbol,Tuple{Any,Any}}()
    function walk(x)
        if x isa Expr
            if x.head === :macrocall && !isempty(x.args) && x.args[1] === Symbol("@xcall")
                a = filter(e -> !(e isa LineNumberNode), x.args[2:end])
                length(a) >= 3 || error("malformed @xcall in session.jl: $x")
                sym = a[1] isa QuoteNode ? a[1].value : a[1]
                man[sym] = (a[2], a[3])
            end
            foreach(walk, x.args)
        end
        return nothing
    end
    walk(top)
    isempty(man) && error("no @xcall sites found in session.jl — manifest empty")
    return man
end

"The bridge manifest (symbol => (ret, argtypes) ASTs) parsed from session.jl's @xcall sites."
manifest() = (_MANIFEST[] === nothing && (_MANIFEST[] = _collect_xcall_manifest()); _MANIFEST[])

"""
    contract_hash() -> UInt

A stable hash over the bridge surface (symbol set + each call's return/argtype
source text).  Exchanged at the worker handshake; a mismatch means the worker and
client were built from different bindings (a rebuilt-one-side bug) and the
connection is refused rather than silently corrupting data.
"""
function contract_hash()
    m = manifest()
    h = hash("enzong-bridge-contract-v1")
    for k in sort!(collect(keys(m)); by = string)
        ret, at = m[k]
        h = hash((string(k), string(ret), string(at)), h)
    end
    return h
end

# ── shared-file buffer I/O (byte-packed; offsets dictated by the client) ──────
# A plain temp file standing in for POSIX shm: seek+write / seek+read! of an
# argument's raw bytes.  The C++ worker (#3) will mmap the same region for true
# zero-copy; the (offset,len,eltype) descriptors on the wire are unchanged.
function _shm_ensure(path::AbstractString, nbytes::Integer)
    sz = filesize(path)
    if sz < nbytes
        open(path, "a") do io
            write(io, zeros(UInt8, nbytes - sz))
        end
    end
    return nothing
end
function _shm_put(path::AbstractString, off::Integer, arr::AbstractVector)
    open(path, "r+") do io
        seek(io, off); write(io, arr)
    end
    return nothing
end
function _shm_get!(path::AbstractString, off::Integer, arr::AbstractVector)
    open(path, "r") do io
        seek(io, off); read!(io, arr)
    end
    return arr
end

# buffer element tag <-> Julia type (the only buffer eltypes the bridge uses)
_buf_tag(::Type{Float64}) = 'd'
_buf_tag(::Type{Int32})   = 'i'
_buf_tag(::Type{Int64})   = 'l'
_buf_tag(T::Type) = error("unsupported bridge buffer eltype $T")
_buf_type(tag::AbstractChar) = tag == 'd' ? Float64 : tag == 'i' ? Int32 :
                               tag == 'l' ? Int64 : error("bad buffer tag $tag")

# ── client side: the remote dispatch the @xcall else-branch calls ─────────────
mutable struct _Worker
    proc::Base.Process
    cin::IO          # control channel: client → worker (requests)
    cout::IO         # control channel: worker → client (replies)
    shm::String      # shared-file path for bulk array args
end
const _CONN = Ref{Union{Nothing,_Worker}}(nothing)

_conn() = _CONN[] === nothing ?
    error("no remote worker connected; EnzoLib.connect_worker!(cmd; shm=…) first") : _CONN[]

"""
    connect_worker!(cmd::Cmd; shm=tempname()) -> Nothing

Spawn a bridge worker process (`cmd` runs `EnzoLib.serve`), establish the control
channel + shared file, verify the contract-hash handshake, and switch the backend
to `:remote`.  All subsequent `@xcall`s route to the worker until
`disconnect_worker!()`.
"""
function connect_worker!(cmd::Base.AbstractCmd; shm::AbstractString = tempname())
    isfile(shm) || write(shm, UInt8[])
    cin = Pipe(); cout = Pipe()
    proc = run(pipeline(cmd; stdin = cin, stdout = cout, stderr = stderr); wait = false)
    close(cin.out); close(cout.in)
    line = readline(cout)
    startswith(line, "READY ") || error("worker handshake failed: $line")
    theirs = parse(UInt, split(line)[2])
    theirs == contract_hash() ||
        error("bridge contract mismatch: worker $(theirs) ≠ client $(contract_hash()) " *
              "(worker and client built from different bindings?)")
    _CONN[] = _Worker(proc, cin, cout, String(shm))
    set_backend!(:remote)
    return nothing
end

"Tear down the remote worker and return the backend to `:local`."
function disconnect_worker!()
    c = _CONN[]
    c === nothing && return nothing
    try; println(c.cin, "QUIT"); flush(c.cin); catch; end
    try; close(c.cin); catch; end
    try; wait(c.proc); catch; end
    _CONN[] = nothing
    set_backend!(:local)
    return nothing
end

# Encode one scalar argument to a whitespace-free token.  Floats are encoded by
# their exact bit pattern so the round-trip is bit-identical (not decimal-printed).
function _enc_scalar(a)
    a isa Bool            ? "i$(Int(a))"                                   :
    a isa Integer         ? "i$(a)"                                        :
    a isa AbstractFloat   ? "f$(string(reinterpret(UInt64, Float64(a)); base=16))" :
    a isa Ptr             ? "p$(UInt(a))"                                  :
    a isa AbstractString  ? "s$(base64encode(String(a)))"                 :
    error("unsupported scalar arg of type $(typeof(a))")
end

_decode_ret(::Type{Cvoid}, tok) = (tok == "void" || error("expected void, got $tok"); nothing)
function _decode_ret(::Type{T}, tok) where {T}
    kind = tok[1]; rest = @view tok[2:end]
    if kind == 'i'
        return T(parse(Int64, rest))
    elseif kind == 'f'
        return T(reinterpret(Float64, parse(UInt64, rest; base = 16)))
    elseif kind == 'p'
        return T === Ptr{Cvoid} ? Ptr{Cvoid}(parse(UInt, rest)) : T(parse(UInt, rest))
    else
        error("undecodable return token $tok for $T")
    end
end

# The remote backend for @xcall.  `ret`/`argtypes` arrive already EVALUATED (the
# macro splices `$(esc(ret))`), so `ret` is a Type and array args are detectable
# by value — no manifest lookup needed on the client.
function _rpc(sym::Symbol, ret, argtypes, args)
    c = _conn()
    toks = String[]
    bufs = Tuple{Int,Any}[]            # (offset, array) to read back after the call
    total = 0
    for a in args                      # first pass: lay out buffer offsets, size the file
        a isa AbstractArray && (total += sizeof(a))
    end
    total > 0 && _shm_ensure(c.shm, total)
    off = 0
    for a in args
        if a isa AbstractArray
            push!(toks, "b$off,$(length(a)),$(_buf_tag(eltype(a)))")
            _shm_put(c.shm, off, a)    # ship IN bytes (zeros for an OUT buffer)
            push!(bufs, (off, a))
            off += sizeof(a)
        else
            push!(toks, _enc_scalar(a))
        end
    end
    println(c.cin, "CALL $(sym) $(join(toks, ' '))"); flush(c.cin)
    reply = readline(c.cout)
    for (o, arr) in bufs               # OUT bytes the kernel wrote (no-op for IN)
        _shm_get!(c.shm, o, arr)
    end
    startswith(reply, "ERR") && error("remote bridge error for $sym: $(reply[5:end])")
    startswith(reply, "RET ") || error("malformed worker reply: $reply")
    return _decode_ret(ret, reply[5:end])
end

# ── worker side: generic dispatcher ───────────────────────────────────────────
# Build a type-stable ccall thunk per symbol from the manifest's literal type
# ASTs.  `@eval` lifts the literal `ret`/`argtypes` into the ccall (which requires
# them literal); the thunks are called via `invokelatest` to dodge world-age.
function _build_thunks(man = manifest())
    thunks = Dict{Symbol,Function}()
    for (sym, (ret, at)) in man
        n = at isa Expr && at.head === :tuple ? length(at.args) :
            (at === :(()) ? 0 : error("argtypes not a tuple for $sym: $at"))
        ps = [Symbol(:a, i) for i in 1:n]
        body = Expr(:call, :ccall, Expr(:call, :_gsym, QuoteNode(sym)), ret, at, ps...)
        thunks[sym] = @eval $(Expr(:->, Expr(:tuple, ps...), body))
    end
    return thunks
end

function _encode_ret(r)
    r === nothing       ? "void"                                          :
    r isa Integer       ? "i$(r)"                                         :
    r isa AbstractFloat ? "f$(string(reinterpret(UInt64, Float64(r)); base=16))" :
    r isa Ptr           ? "p$(UInt(r))"                                   :
    error("unencodable return value $(typeof(r))")
end

# Decode the request's argument tokens, materializing buffer args as arrays backed
# by the IN bytes from the shared file.  Returns (argvalues, outbufs) where outbufs
# = (offset, array) to flush back after the ccall.
function _decode_args(toks, shm)
    vals = Any[]; outbufs = Tuple{Int,Any}[]
    for t in toks
        kind = t[1]; rest = @view t[2:end]
        if kind == 'i'
            push!(vals, parse(Int64, rest))
        elseif kind == 'f'
            push!(vals, reinterpret(Float64, parse(UInt64, rest; base = 16)))
        elseif kind == 'p'
            push!(vals, Ptr{Cvoid}(parse(UInt, rest)))
        elseif kind == 's'
            push!(vals, String(base64decode(String(rest))))
        elseif kind == 'b'
            o, len, tag = split(rest, ',')
            off = parse(Int, o); n = parse(Int, len)
            arr = Vector{_buf_type(tag[1])}(undef, n)
            _shm_get!(shm, off, arr)   # IN bytes (zeros for an OUT buffer)
            push!(vals, arr)
            push!(outbufs, (off, arr))
        else
            error("bad arg token $t")
        end
    end
    return vals, outbufs
end

"""
    serve(; cin=stdin, cout=stdout, shm=ARGS[1])

The worker loop: load the bridge in-process (`:local`), announce the contract
hash, then dispatch each `CALL <sym> <tokens…>` by `ccall`ing the raw C symbol via
a generated thunk.  Array results are flushed back to the shared file before the
reply line is sent.  Runs until `QUIT`/EOF.  This is the Julia reference worker
(and the differential oracle's remote side); the #3 C++ worker speaks the same
wire protocol without a Julia runtime.
"""
function serve(; cin::IO = stdin, cout::IO = stdout, shm::AbstractString = ARGS[1])
    man = manifest()
    thunks = _build_thunks(man)
    println(cout, "READY $(contract_hash())"); flush(cout)
    while !eof(cin)
        line = readline(cin); isempty(line) && continue
        parts = split(line, ' ')
        cmd = parts[1]
        cmd == "QUIT" && break
        if cmd != "CALL"
            println(cout, "ERR bad-command $cmd"); flush(cout); continue
        end
        sym = Symbol(parts[2])
        try
            f = get(thunks, sym, nothing)
            f === nothing && error("unknown bridge symbol $sym")
            vals, outbufs = _decode_args(parts[3:end], shm)
            r = Base.invokelatest(f, vals...)
            for (off, arr) in outbufs
                _shm_put(shm, off, arr)            # flush OUT bytes before replying
            end
            println(cout, "RET $(_encode_ret(r))"); flush(cout)
        catch e
            println(cout, "ERR $(sym): $(sprint(showerror, e))"); flush(cout)
        end
    end
    return nothing
end
