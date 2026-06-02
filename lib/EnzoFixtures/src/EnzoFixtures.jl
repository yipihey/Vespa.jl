"""
    EnzoFixtures

Julia reader for EnzoModules' **golden fixtures** and a mirror of its
tolerance/diff policy (`EnzoModules/enzomodules/{fixtures,diff}.py`). The
`.fixture` text format and the relative+absolute tolerance are language-neutral
*by design* — the README states a Julia rewrite can read the committed fixtures
directly to certify its own port "without depending on Python … it only needs the
same tolerance policy." This package IS that policy in Julia, and the per-method
certification gate of the incremental-rewrite framework.
"""
module EnzoFixtures

export Tolerance, BITWISE, isclose, compare, CompareResult,
       Fixture, load_fixture, load_dir, scalar, array

# ── tolerance policy (mirror of diff.py) ─────────────────────────────────────
"""
    Tolerance(rtol=0.0, atol=0.0)

`rtol == atol == 0` means bitwise equality (integer/index logic, or replay
against the same library). A small positive `rtol` allows for FMA/reduction-order
differences across compilers/precisions or against an analytic reference.
"""
struct Tolerance
    rtol::Float64
    atol::Float64
end
Tolerance(; rtol::Real = 0.0, atol::Real = 0.0) = Tolerance(Float64(rtol), Float64(atol))

"Bitwise-equality tolerance."
const BITWISE = Tolerance(0.0, 0.0)

"""
    isclose(a, b, tol) -> Bool

True when `abs(a-b) <= atol + rtol*abs(b)`. NaN-aware: two NaNs compare equal
(so reference outputs that legitimately contain NaN round-trip).
"""
function isclose(a::Real, b::Real, tol::Tolerance)
    (isnan(a) || isnan(b)) && return isnan(a) && isnan(b)
    (tol.rtol == 0.0 && tol.atol == 0.0) && return a == b
    return abs(a - b) <= tol.atol + tol.rtol * abs(b)
end

"Element-wise comparison result; `ok` is the all-pass flag (use `Bool(r)`)."
struct CompareResult
    ok::Bool
    n::Int
    nfail::Int
    maxabs::Float64
    maxrel::Float64
    worst::Int          # 1-based index of the worst (relative) element, 0 if none
end
Base.Bool(r::CompareResult) = r.ok

"""
    compare(actual, expected, tol) -> CompareResult

Element-wise comparison of two equal-length numeric sequences, reporting the
largest absolute/relative deviation and the worst index — mirrors
`diff.compare`.
"""
function compare(actual, expected, tol::Tolerance)
    length(actual) == length(expected) ||
        throw(DimensionMismatch("actual $(length(actual)) vs expected $(length(expected))"))
    nfail = 0; maxabs = 0.0; maxrel = 0.0; worst = 0
    @inbounds for i in eachindex(actual, expected)
        av = Float64(actual[i]); bv = Float64(expected[i])
        isclose(av, bv, tol) || (nfail += 1)
        da = (isnan(av) || isnan(bv)) ? (isnan(av) && isnan(bv) ? 0.0 : Inf) : abs(av - bv)
        dr = bv == 0.0 ? (av == 0.0 ? 0.0 : Inf) : da / abs(bv)
        da > maxabs && (maxabs = da)
        dr > maxrel && (maxrel = dr; worst = i)
    end
    return CompareResult(nfail == 0, length(actual), nfail, maxabs, maxrel, worst)
end

# ── fixture format (mirror of fixtures.py) ───────────────────────────────────
# Line-oriented text: `# comment`, `key = value` scalars (int/float/string),
# `@name v…` float arrays.
"""
    Fixture(scalars, arrays)

A parsed fixture: `scalars[:key]` (Int/Float64/String) and `arrays[:name]`
(`Vector{Float64}`). Access via [`scalar`](@ref) / [`array`](@ref).
"""
struct Fixture
    scalars::Dict{Symbol,Any}
    arrays::Dict{Symbol,Vector{Float64}}
    path::String
end

"Typed scalar lookup (`scalar(fx, :idim, Int)`); converts to `T`."
scalar(fx::Fixture, k::Symbol) = fx.scalars[k]
scalar(fx::Fixture, k::Symbol, ::Type{T}) where {T} = convert(T, fx.scalars[k])
"Float array lookup (`array(fx, :dslice_out)`)."
array(fx::Fixture, k::Symbol) = fx.arrays[k]

function _parse_scalar(s::AbstractString)
    v = tryparse(Int, s)
    v === nothing || return v
    f = tryparse(Float64, s)
    f === nothing || return f
    return String(s)
end

"""
    load_fixture(path) -> Fixture

Parse one `.fixture` file (the format `tools/capture_*.py` writes via
`fixtures.save_fixture`).
"""
function load_fixture(path::AbstractString)
    scalars = Dict{Symbol,Any}()
    arrays = Dict{Symbol,Vector{Float64}}()
    for raw in eachline(path)
        line = strip(raw)
        (isempty(line) || startswith(line, "#")) && continue
        if startswith(line, "@")
            toks = split(line)
            name = Symbol(toks[1][2:end])
            arrays[name] = Float64[parse(Float64, t) for t in toks[2:end]]
        else
            kv = split(line, "="; limit = 2)
            length(kv) == 2 || continue
            scalars[Symbol(strip(kv[1]))] = _parse_scalar(strip(kv[2]))
        end
    end
    return Fixture(scalars, arrays, String(path))
end

"Load every `*.fixture` in `directory` (sorted by filename)."
function load_dir(directory::AbstractString)
    files = sort!(filter(f -> endswith(f, ".fixture"), readdir(directory; join = true)))
    return [load_fixture(f) for f in files]
end

end # module
