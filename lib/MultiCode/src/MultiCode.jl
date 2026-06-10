"""
    MultiCode

The cross-code layer of ADR-0006 (D3): a **canonical state** no code owns, the
per-code `extract`/`inject!` adapters that convert at the boundary, the
conservation ledger that gates every conversion, and the comparison harness
that runs ONE problem spec through Enzo, RAMSES, and Arepo and writes one
report.

The canonical representation is deliberately minimal: cell centers, volumes,
and the conserved gas state (ρ, ρu, E) as plain `Float64` arrays in **common
normalized units** (box length = 1, with each adapter recording the scales it
divided out).  Nothing downstream of an adapter ever sees a code's internal
units or memory layout — the convert-at-the-boundary rule.

Phase-2 scope: the adapters address each code's *native* uniform-resolution
state (Enzo unigrid root, one RAMSES level, Arepo's Voronoi cell set).  AMR
composites and the R3D Voronoi↔AMR remap operators are Phase 4 (ADR-0006).

This package depends on all three wrappers directly; converting those to
package extensions (weak deps) is deferred polish — the seams are already
per-code modules.
"""
module MultiCode

using Printf
using LinearAlgebra: dot, cross
using CodeBridge
using EnzoLib
using RamsesLib
using ArepoLib
using PPMKernels
import PoissonKernels
import R3D

export CellSet, ledger, ledger_drift, ncells, exact_sod, SodSpec
export enzo_extract, ramses_extract, arepo_extract
export run_enzo_sod, run_ramses_sod, run_arepo_sod
export profile_x, sod_l1, sod_report
export ramses_ppmk_hydro_step!, run_ramses_sod_guest
export ramses_composite_raster, ramses_composite_deraster!, ramses_ppmk_hydro_step_amr!
export ramses_ppmk_hydro_step_amr_fast!
export run_moray_stromgren, moray_ifront_radius, stromgren_radius, stromgren_scales
export deposit_to_grid, deposit_exact, sample_at_points
export run_ramsesrt_stromgren, ramsesrt_ifront_radius
export ramsesrt_set_density!, ramsesrt_xhii_grid, run_enzo_host_ramsesrt
export SedovCompareSpec, sedov_bomb, sedov_radius, sedov_profile
export run_enzo_sedov, run_ramses_sedov, sedov_report
export ZeldovichSpec, zeldovich_particles, zeldovich_growth, zeldovich_measure
export run_enzo_zeldovich, run_ramses_zeldovich
export ramses_grid_field, ramses_set_grid_field!, ramses_ka_poisson!
export run_ramses_gravity_compare

include("canonical.jl")
include("exact_sod.jl")
include("adapters.jl")
include("sod.jl")
include("report.jl")
include("ramses_slot.jl")
include("moray.jl")
include("exchange.jl")
include("ramsesrt.jl")
include("enzo_rt_guest.jl")
include("sedov_compare.jl")
include("zeldovich.jl")
include("gravity_slot.jl")

end # module
