# A problem is source code, not a parameter file (ADR P9): a typed value whose
# initial condition is a plain Julia function `(x...) -> (ρ, vx, vy, vz, p)`,
# JIT-compiled to native code. Cross-parameter checks happen here, at construction.

"""
    Problem(; name, dims, domain, γ, bcs, init, tfinal, cfl=0.4)

Self-contained specification of a hydro problem.

  * `dims`    — interior cell count per axis (a tuple; its length is the rank).
  * `domain`  — `(lo, hi)` physical bounds per axis (a tuple of 2-tuples).
  * `γ`       — adiabatic index.
  * `bcs`     — one `AbstractBC` for all sides, a `BoundaryConditions`, or a
                tuple of `(lo, hi)` `AbstractBC` pairs (one per axis).
  * `init`    — `init(x, y, z)` (coordinates padded with zeros to rank 3)
                returning primitive `(ρ, vx, vy, vz, p)`.
  * `tfinal`  — stop time.
  * `cfl`     — Courant number.
"""
struct Problem{N,F}
    name::String
    dims::NTuple{N,Int}
    domain::NTuple{N,Tuple{Float64,Float64}}
    γ::Float64
    bcs::Any
    init::F
    tfinal::Float64
    cfl::Float64
end

function Problem(; name::AbstractString,
                 dims::NTuple{N,<:Integer},
                 domain::NTuple{N,<:Tuple},
                 γ::Real,
                 bcs,
                 init,
                 tfinal::Real,
                 cfl::Real = 0.4) where {N}
    γ > 1 || throw(ArgumentError("γ must exceed 1 (got $γ)"))
    tfinal > 0 || throw(ArgumentError("tfinal must be positive (got $tfinal)"))
    0 < cfl <= 1 || throw(ArgumentError("cfl must be in (0, 1] (got $cfl)"))
    all(d -> d > 0, dims) || throw(ArgumentError("all dims must be positive"))
    for d in 1:N
        domain[d][2] > domain[d][1] ||
            throw(ArgumentError("domain[$d] must have hi > lo (got $(domain[d]))"))
    end
    dom = ntuple(d -> (Float64(domain[d][1]), Float64(domain[d][2])), N)
    return Problem{N,typeof(init)}(String(name), Int.(dims), dom, Float64(γ),
                                   bcs, init, Float64(tfinal), Float64(cfl))
end

Base.ndims(::Problem{N}) where {N} = N

"""
    sod_problem_defaults(; n=128)

The classic Sod shock tube as a `Problem` (left ρ=1, p=1; right ρ=0.125, p=0.1;
u=0; γ=1.4; interface at x=0.5; outflow BCs; t_final=0.2). Matches the setup of
Enzo's `run/Hydro/Hydro-1D/SodShockTube` test.
"""
function sod_problem_defaults(; n::Integer = 128)
    init(x, y, z) = x < 0.5 ? (1.0, 0.0, 0.0, 0.0, 1.0) : (0.125, 0.0, 0.0, 0.0, 0.1)
    return Problem(; name = "SodShockTube",
                   dims = (Int(n),),
                   domain = ((0.0, 1.0),),
                   γ = 1.4,
                   bcs = MeshInterface.Outflow(),
                   init = init,
                   tfinal = 0.2,
                   cfl = 0.4)
end
