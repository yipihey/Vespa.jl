# A miniature wrapper module over a tiny compiled C library, written exactly the
# way EnzoLib/RamsesLib/ArepoLib write theirs: a `const BRIDGE` + `@xcall`-style
# bindings.  Exercises the full CodeBridge surface (loading, local calls,
# manifest/contract, worker RPC) with no legacy-code build required.
#
# The library path comes from ENV["CB_FIXTURE_LIB"] (set by runtests.jl after
# compiling cbfix.c, and inherited by the worker subprocess).

module CBFixture

using CodeBridge

const BRIDGE = CodeBridge.Bridge(:cbfix, @__MODULE__;
    libs = Dict(:main => CodeBridge.LazyLib(env = "CB_FIXTURE_LIB", default = "/nonexistent/libcbfix",
                                            hint = "runtests.jl compiles it and sets CB_FIXTURE_LIB.")),
    manifest_files = [@__FILE__],
    contract_seed = "cbfixture-contract-v1")

add_d(x, y) = @xcall(:cb_add_d, Cdouble, (Cdouble, Cdouble), x, y)
add_i(x, y) = @xcall(:cb_add_i, Cint, (Cint, Cint), x, y)

"In-place scale of a Float64 vector (an OUT/IN buffer round-trip)."
function scale!(v::Vector{Float64}, s::Real)
    @xcall(:cb_scale, Cvoid, (Ptr{Cdouble}, Cint, Cdouble), v, length(v), Float64(s))
    return v
end

"Sum into a Ref (the Ref-as-one-element-buffer path)."
function sumto(v::Vector{Float64})
    out = Ref{Cdouble}(0.0)
    @xcall(:cb_sum, Cvoid, (Ptr{Cdouble}, Cint, Ptr{Cdouble}), v, length(v), out)
    return out[]
end

"String argument (Cstring marshalling)."
strlen(s::AbstractString) = @xcall(:cb_strlen, Cint, (Cstring,), s)

"Int32 buffer round-trip (the 'i' buffer tag)."
function iota!(v::Vector{Int32})
    @xcall(:cb_iota, Cvoid, (Ptr{Cint}, Cint), v, length(v))
    return v
end

"Zero-argument call (the trailing-space wire case — Arepo's run!/run_step)."
answer() = @xcall(:cb_answer, Cint, ())

serve(; kwargs...) = CodeBridge.serve(BRIDGE; kwargs...)

end # module
