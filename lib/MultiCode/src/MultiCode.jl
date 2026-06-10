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
export run_ramses_gravity_compare, run_ramses_gravity_amr_compare
export ramses_ka_poisson_fine!, run_ramses_gravity_blob_compare
export run_dfmm_sod, run_athena_sod, run_music_crosscheck, run_discodj_growth

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

"""
    run_dfmm_sod(spec = SodSpec(gamma = 5/3, t = 0.2); N = 200, tau = 1e-3,
                 cfl = 0.3, sigma_x0 = 0.02) -> (; profile, t, steps, seconds, …)

The dfmm engine in the Sod harness (ADR-0006 Phase 5, library convergence):
the dual-frame moment method advances the SAME Riemann problem the legacy
engines run, on its own Lagrangian segments, and reports in the harness's
shapes (`profile` → `sod_l1` vs the same exact solution).  Implemented in the
`MultiCodeDfmmExt` package extension — `using dfmm` activates it; this stub
errors otherwise.  γ = 5/3 (the dfmm closure's adiabatic index).
"""
function run_dfmm_sod end

"""
    run_athena_sod(spec = SodSpec(); nx1 = 256) -> (; profile, t, seconds, …)

Athena++ as the fourth legacy engine in the Sod harness: the stock
`athinput.sod` run IN-PROCESS, profile from the final `.tab`, conservation
from the `.hst` history — gated against the same exact-Riemann oracle.
Implemented in `MultiCodeAthenaExt` — `using AthenaLib` activates it.
"""
function run_athena_sod end

"""
    run_music_crosscheck(; boxlength=20.0, zstart=50.0, level=5) -> (; corr, rms, …)

The MUSIC injector validation: ONE MusicSpec realization, Enzo booted on the
generated `parameter_file.txt` + particle ICs and RAMSES (UNITS=COSMO) on the
grafic2 level directory, the two codes' INITIAL CIC density fields correlated.
Implemented in `MultiCodeMusicExt` — `using MusicLib` activates it.
"""
function run_music_crosscheck end

"""
    run_discodj_growth(; res=32, z_init=49.0, a_ratio=4.0, box_mpch=32.0, seed=42)

DISCO-DJ's differentiable 1LPT (≡ Zel'dovich) displacement field, turned into
ZERO-velocity particles (no velocity-unit convention enters) and evolved
through BOTH Enzo and RAMSES in EdS — the whole linear field follows the
closed-form mixed-mode growth b(x) = ⅗x + ⅖x^{−3/2}, gating shape, amplitude,
and Enzo↔RAMSES agreement.  Implemented in `MultiCodeDiscoDJExt` — `using
DiscoDJLib` activates it (set JULIA_PYTHONCALL_EXE before loading).
"""
function run_discodj_growth end

end # module
