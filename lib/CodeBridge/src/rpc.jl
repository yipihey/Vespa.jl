# ── ADR-0005: remote bridge transport (subprocess worker + shared memory) ─────
#
# The `:remote` backend routes every `@xcall` to a worker PROCESS over a
# line-based control channel; bulk array arguments travel through a shared file
# (a stand-in for POSIX shm — a C++ worker mmaps it for true zero-copy; the
# protocol is identical).  This is what lets a foreign-runtime worker (MPI
# `libenzo`, a second legacy code, N instances of a global-state singleton like
# Arepo) run WITHOUT its C/C++/Fortran runtime ever entering the Julia process.
#
# Two properties make it bug-resistant by construction:
#
#  1. ONE contract.  The worker dispatches raw C symbols via `ccall` thunks
#     GENERATED from the manifest PARSED OUT OF the wrapper source itself
#     (every `@xcall` site).  There is no second hand-maintained interface to
#     drift from the bindings, and the `contract_hash` handshake rejects a
#     rebuilt-one-side.
#
#  2. No in/out direction table.  Every array argument is round-tripped through
#     the shared file BIDIRECTIONALLY: the client ships the arg's current bytes
#     (real data for an IN buffer, zeros for an OUT buffer) and reads the bytes
#     back after the call (the C kernel's writes for an OUT buffer, unchanged
#     for an IN buffer).  The off-direction transfer is a few KB of zeros/
#     no-ops, so correctness needs only the element type + length — both
#     carried in the request — not knowledge of which Ptr args are written.
#     Wrapper code therefore needs NO remote-specific paths.  `Ref` arguments
#     ride the same mechanism as one-element buffers.
#
# The in-process `:local` path (ccall) remains the default and the differential
# ORACLE: a parity test runs the SAME wrappers through both backends and
# asserts bit-identical results.
#
# Wire protocol (byte-identical to EnzoLib's original):
#   worker → client   READY <contract_hash>
#   client → worker   CALL <sym> <tokens…>   |   QUIT
#   worker → client   RET <token>   |   ERR <message>

# ── shared-file buffer I/O (byte-packed; offsets dictated by the client) ──────
function _shm_ensure(path::AbstractString, nbytes::Integer)
    sz = filesize(path)
    if sz < nbytes
        open(path, "a") do io
            write(io, zeros(UInt8, nbytes - sz))
        end
    end
    return nothing
end
function _shm_put(path::AbstractString, off::Integer, arr::AbstractArray)
    open(path, "r+") do io
        seek(io, off); write(io, arr)
    end
    return nothing
end
function _shm_get!(path::AbstractString, off::Integer, arr::AbstractArray)
    open(path, "r") do io
        seek(io, off); read!(io, arr)
    end
    return arr
end

# buffer element tag <-> Julia type (the only buffer eltypes the bridges use)
_buf_tag(::Type{Float64}) = 'd'
_buf_tag(::Type{Int32})   = 'i'
_buf_tag(::Type{Int64})   = 'l'
_buf_tag(T::Type) = error("unsupported bridge buffer eltype $T")
_buf_type(tag::AbstractChar) = tag == 'd' ? Float64 : tag == 'i' ? Int32 :
                               tag == 'l' ? Int64 : error("bad buffer tag $tag")

# ── client side ───────────────────────────────────────────────────────────────
_conn(b::Bridge) = b.conn === nothing ?
    error("no remote worker connected for bridge $(b.name); connect_worker!(bridge, cmd; shm=…) first") :
    b.conn

"""
    connect_worker!(b::Bridge, cmd::Cmd; shm=tempname()) -> Nothing

Spawn a bridge worker process (`cmd` runs `serve` for the same bridge surface),
establish the control channel + shared file, verify the contract-hash
handshake, and switch the bridge's backend to `:remote`.  All subsequent
`@xcall`s on this bridge route to the worker until `disconnect_worker!(b)`.
Each bridge holds its own connection, so several legacy codes can be live in
one session, each in its own worker process (ADR-0006 D2).
"""
function connect_worker!(b::Bridge, cmd::Base.AbstractCmd; shm::AbstractString = tempname())
    b.conn === nothing ||
        error("bridge $(b.name) already has a connected worker; disconnect_worker! first")
    isfile(shm) || write(shm, UInt8[])
    cin = Pipe(); cout = Pipe()
    proc = run(pipeline(cmd; stdin = cin, stdout = cout, stderr = stderr); wait = false)
    close(cin.out); close(cout.in)
    line = readline(cout)
    startswith(line, "READY ") || error("worker handshake failed for $(b.name): $line")
    theirs = parse(UInt, split(line)[2])
    theirs == contract_hash(b) ||
        error("bridge contract mismatch for $(b.name): worker $(theirs) ≠ client " *
              "$(contract_hash(b)) (worker and client built from different bindings?)")
    b.conn = WorkerConn(proc, cin, cout, String(shm))
    set_backend!(b, :remote)
    return nothing
end

"Tear down the bridge's remote worker and return its backend to `:local`."
function disconnect_worker!(b::Bridge)
    c = b.conn
    c === nothing && return nothing
    try; println(c.cin, "QUIT"); flush(c.cin); catch; end
    try; close(c.cin); catch; end
    try; wait(c.proc); catch; end
    b.conn = nothing
    set_backend!(b, :local)
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
# by value — no manifest lookup needed on the client.  `Ref` args are shipped as
# one-element buffers and read back into the Ref after the call.
function _rpc(b::Bridge, sym::Symbol, ret, argtypes, args)
    c = _conn(b)
    toks = String[]
    bufs = Tuple{Int,Any}[]            # (offset, array-or-Ref) to read back after the call
    total = 0
    for a in args                      # first pass: size the shared file
        a isa AbstractArray && (total += sizeof(a))
        a isa Base.RefValue && (total += sizeof(a[]))
    end
    total > 0 && _shm_ensure(c.shm, total)
    off = 0
    for a in args
        if a isa AbstractArray
            push!(toks, "b$off,$(length(a)),$(_buf_tag(eltype(a)))")
            _shm_put(c.shm, off, a)    # ship IN bytes (zeros for an OUT buffer)
            push!(bufs, (off, a))
            off += sizeof(a)
        elseif a isa Base.RefValue
            tmp = [a[]]
            push!(toks, "b$off,1,$(_buf_tag(eltype(tmp)))")
            _shm_put(c.shm, off, tmp)
            push!(bufs, (off, a))
            off += sizeof(tmp)
        else
            push!(toks, _enc_scalar(a))
        end
    end
    # no trailing space on zero-arg calls — the worker splits on ' '
    println(c.cin, isempty(toks) ? "CALL $(sym)" : "CALL $(sym) $(join(toks, ' '))")
    flush(c.cin)
    reply = readline(c.cout)
    for (o, arr) in bufs               # OUT bytes the kernel wrote (no-op for IN)
        if arr isa Base.RefValue
            tmp = [arr[]]
            _shm_get!(c.shm, o, tmp)
            arr[] = tmp[1]
        else
            _shm_get!(c.shm, o, arr)
        end
    end
    startswith(reply, "ERR") && error("remote bridge error for $sym: $(reply[5:end])")
    startswith(reply, "RET ") || error("malformed worker reply: $reply")
    return _decode_ret(ret, reply[5:end])
end

# ── worker side: generic dispatcher ───────────────────────────────────────────
# Build a type-stable ccall thunk per symbol from the manifest's literal type
# ASTs.  The thunks are evaluated in the bridge's HOME module, so wrapper-local
# type aliases (e.g. `Handle = Ptr{Cvoid}`) resolve exactly as at the original
# call sites; they are called via `invokelatest` to dodge world-age.
function _build_thunks(b::Bridge, man = manifest(b))
    thunks = Dict{Symbol,Function}()
    for (s, (ret, at)) in man
        n = at isa Expr && at.head === :tuple ? length(at.args) :
            (at === :(()) ? 0 : error("argtypes not a tuple for $s: $at"))
        ps = [Symbol(:a, i) for i in 1:n]
        lookup = Expr(:call, sym, b, QuoteNode(s))   # `sym`/`b` embedded as values
        body = Expr(:call, :ccall, lookup, ret, at, ps...)
        thunks[s] = Core.eval(b.home, Expr(:->, Expr(:tuple, ps...), body))
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
    serve(b::Bridge; cin=stdin, cout=stdout, shm, isolate_stdio=true)

The worker loop: load the bridge's library in-process (`:local`), announce the
contract hash, then dispatch each `CALL <sym> <tokens…>` by `ccall`ing the raw
C symbol via a generated thunk.  Array results are flushed back to the shared
file before the reply line is sent.  Runs until `QUIT`/EOF.  This is the Julia
reference worker (and the differential oracle's remote side); a foreign worker
(e.g. EnzoModules' C++ MPI worker) speaks the same wire protocol without a
Julia runtime.

`isolate_stdio` (default on, when `cout === stdout`): legacy libraries print
banners to the process's fd 1 (C stdio, Fortran unit 6) — RAMSES's
"Serial execution (no MPI)." mid-protocol, for instance — which would interleave
with the reply lines and corrupt the control channel.  The worker therefore
dup's the original fd 1 as its private control channel and repoints fd 1 at
stderr, so native prints stream harmlessly to the client's stderr.
"""
function serve(b::Bridge; cin::IO = stdin, cout::IO = stdout, shm::AbstractString,
               isolate_stdio::Bool = true)
    if isolate_stdio && cout === stdout
        ctl = ccall(:dup, Cint, (Cint,), 1)          # the control channel keeps fd 1's pipe
        ctl == -1 && error("serve: dup(1) failed")
        rc = ccall(:dup2, Cint, (Cint, Cint), 2, 1)  # fd 1 → stderr for native prints
        rc == -1 && error("serve: dup2(2,1) failed")
        cout = Base.fdio(ctl, true)
    end
    man = manifest(b)
    thunks = _build_thunks(b, man)
    println(cout, "READY $(contract_hash(b))"); flush(cout)
    while !eof(cin)
        line = readline(cin); isempty(line) && continue
        parts = split(line, ' '; keepempty = false)
        cmd = parts[1]
        cmd == "QUIT" && break
        if cmd != "CALL"
            println(cout, "ERR bad-command $cmd"); flush(cout); continue
        end
        s = Symbol(parts[2])
        try
            f = get(thunks, s, nothing)
            f === nothing && error("unknown bridge symbol $s")
            vals, outbufs = _decode_args(parts[3:end], shm)
            r = Base.invokelatest(f, vals...)
            for (off, arr) in outbufs
                _shm_put(shm, off, arr)            # flush OUT bytes before replying
            end
            println(cout, "RET $(_encode_ret(r))"); flush(cout)
        catch e
            println(cout, "ERR $(s): $(sprint(showerror, e))"); flush(cout)
        end
    end
    return nothing
end
