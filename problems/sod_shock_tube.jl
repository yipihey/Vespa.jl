# Problem specification as source code (ADR-0001, P9).
#
# A problem is a Julia file: domain, physics, BCs, and initial conditions as
# typed values and a plain function — no parameter file, no `ProblemType` integer
# dispatch, no separately-drifting documentation. The spec *is* the document, and
# it runs unchanged on any backend behind the substrate seam.
#
# Usage:
#   include("problems/sod_shock_tube.jl")
#   using RefMesh
#   prob = sod_problem(n = 256)
#   mesh = UniformMesh(prob.dims, prob.domain; nghost = prob.nghost)
#   sim  = Simulation(mesh, prob)
#   evolve!(sim; verbose = true)

using Vespa

"""
    sod_problem(; n = 128) -> Problem

The classic Sod shock tube (Sod 1978), matching Enzo's
`run/Hydro/Hydro-1D/SodShockTube`:

  * domain `x ∈ [0, 1]`, `n` cells, outflow boundaries;
  * left state  ρ = 1.000, p = 1.0, u = 0   (x < 0.5);
  * right state ρ = 0.125, p = 0.1, u = 0   (x ≥ 0.5);
  * γ = 1.4, integrated to t = 0.2.

The initial condition is a function `(x, y, z) -> (ρ, vx, vy, vz, p)`, JIT-
compiled to native code at first use.
"""
function sod_problem(; n::Integer = 128)
    init(x, y, z) = x < 0.5 ? (1.0, 0.0, 0.0, 0.0, 1.0) :
                              (0.125, 0.0, 0.0, 0.0, 0.1)
    return Problem(; name   = "SodShockTube",
                     dims   = (Int(n),),
                     domain = ((0.0, 1.0),),
                     γ      = 1.4,
                     bcs    = Outflow(),
                     init   = init,
                     tfinal = 0.2,
                     cfl    = 0.4)
end
