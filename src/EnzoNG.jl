"""
    EnzoNG

The science / orchestration layer of next-generation Enzo (ADR-0001). All code
here is written **only** against `MeshInterface`; it never names a concrete
backend. The backend (RefMesh today; HGBackend, Rust, GPU later) is injected at
the `Simulation` constructor. This package therefore does *not* depend on
RefMesh — that dependency lives only in tests and example drivers, which is what
enforces the substrate seam at the package level (ADR DoD).

Contents:
  * physics kernels — stateless functions over typed memory (P1): an HLLC
    Riemann solver, PLM reconstruction, the ideal-gas EOS;
  * a CFL-driven, ghost-free conservative flux-divergence driver with PLM +
    SSP-RK2 (Julia, P1) that runs unchanged on every backend;
  * the `Problem` type — a problem is a typed value with an initial-condition
    *function*, not a parameter file (P9);
  * conservation diagnostics and a small field dump;
  * an exact Riemann solver used to verify the Sod shock tube.
"""
module EnzoNG

using MeshInterface
using Printf

# Conserved-variable layout used throughout the hydro kernels.
const NVAR = 5
const FIELD_NAMES = (:density, :momentum_x, :momentum_y, :momentum_z, :total_energy)
const MOM_INDEX = (2, 3, 4)   # momentum_x, _y, _z in conserved/primitive vectors

include("eos.jl")
include("riemann.jl")
include("reconstruct.jl")
include("problem.jl")
include("driver.jl")
include("reflux.jl")        # coarse–fine flux registers (used by evolve_level!)
include("gravity.jl")       # self-gravity: composite Poisson (CG) + g source
include("diagnostics.jl")
include("exact_riemann.jl")

export Problem, Simulation, evolve!, evolve_level!, step!, compute_dt,
    enable_gravity!, GravityField, solve_poisson!, apply_laplacian!,
    conserved_totals, dump_fields, cell_samples, primitive_at,
    exact_riemann_sample, sod_problem_defaults,
    RefinementPolicy, regrid!, density_gradient_indicator,
    # re-exported from MeshInterface for spec/driver ergonomics
    SoA, AoS, Blocked, Outflow, Periodic, Reflecting, BoundaryConditions,
    refine!, coarsen!, level_of, max_level,
    Instrumented, span_report, reset_spans!

end # module
