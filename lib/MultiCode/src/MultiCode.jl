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
using Random
using CodeBridge
using EnzoLib
using RamsesLib
using ArepoLib
using PPMKernels
using KernelAbstractions: @kernel, @index
import PoissonKernels
import ChemistryKernels
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
export run_dfmm_sod, run_athena_sod, run_athena_sod3d, athena_stage_cellset
export run_music_crosscheck, run_music_discodj_phase_report, run_discodj_growth
export run_gadget4_halos
export run_cicass_streaming, write_grafic_streaming, run_cicass_enzo, run_cicass_ramses

include("runout.jl")              # where a run writes its outputs (scratch, never the repo)
include("fieldio.jl")             # byte-identical field-dump writer + fields.toml side-car
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
include("grackle_service.jl")     # code-neutral reduced primordial chemistry (HII,H2I)
include("grackle_slot.jl")        # wire the service onto RAMSES / Arepo hosts
include("enzo_resident.jl")        # GPU-resident particle push (replaces session_update_particles)
include("patchgrid.jl")            # in-process topgrid decomposition into sibling patches (GPU hydro+chem)
include("global_gravity.jl")       # global CPU-FFT top-grid gravity coupling the patches
include("patch_cosmo.jl")          # super-comoving cosmology (units, Friedmann, drag, particle push)

export PatchGrid, Patch, build_patchgrid, scatter_global!, gather_global
export exchange_ghosts!, exchange_ghosts_axis!, patch_step!, total_mass
export assemble_global_density!, solve_global_poisson!, patch_accel, patch_accel_gpu, global_gravity_accel
export particle_accel_field, assemble_global_density_gpu!, particle_accel_field_gpu, global_gravity_gpu
export Cosmo, cosmo_units, dadtau, dtau_for_dlna, Hofa, growth_D, compton_drag_over_H
export compton_drag_patches!, push_particles!, z_to_a, a_to_z

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

"`run_athena_sod3d(spec; n=32)` — 3-D Athena++ Sod → `CellSet` (VTK readback); see `MultiCodeAthenaExt`."
function run_athena_sod3d end

"""
    athena_stage_cellset(cs; dims, dt, gamma=5/3, workdir=mktempdir()) -> (; cs, ...)

Raster a uniform Cartesian `CellSet` into Athena++'s staged-binary pgen,
advance one hydro step, then read the conserved state back as a `CellSet`.
This is the Athena adapter surface used by hierarchy owners: they rasterize
their blocks to the canonical layout at the boundary, and this extension owns
the Athena-specific file/runtime/readback details.  Implemented in
`MultiCodeAthenaExt` — `using AthenaLib` activates it.
"""
function athena_stage_cellset end

"""
    run_music_crosscheck(; boxlength=20.0, zstart=50.0, level=5) -> (; corr, rms, …)

The MUSIC injector validation: ONE MusicSpec realization, Enzo booted on the
generated `parameter_file.txt` + particle ICs and RAMSES (UNITS=COSMO) on the
grafic2 level directory, the two codes' INITIAL CIC density fields correlated.
Implemented in `MultiCodeMusicExt` — `using MusicLib` activates it.
"""
function run_music_crosscheck end

"""
    write_grafic_streaming(dir, snap; h0=71.0) -> dir

Write a RAMSES grafic IC directory carrying the baryon–dark-matter **streaming
velocity** from a CICASS snapshot: `ic_velbx/y/z` (gas velocity grid),
`ic_velcx/y/z` (CDM velocity grid, CIC-deposited from the DM particles) and
`ic_deltab` (baryon overdensity).  The streaming offset is, by construction,
`mean(ic_velb) - mean(ic_velc)` — the thing a single-phase generator cannot
express.  Implemented in `MultiCodeCICASSExt` — `using CICASSLib` activates it.
"""
function write_grafic_streaming end

"""
    run_cicass_streaming(; vbc=30.0, boxlength=0.2, zstart=100.0, ngrid=128) -> (; ...)

CICASS streaming-IC validation: generate ONE realization with relative velocity
`vbc`, write the RAMSES-native grafic streaming set, and read it back to confirm
the coherent gas–DM bulk offset survives into each code's IC format
(≈ `vbc·(1+z)/1001` km/s along one axis, zero for `vbc=0`).  Implemented in
`MultiCodeCICASSExt` — `using CICASSLib` activates it.
"""
function run_cicass_streaming end

"""
    run_cicass_enzo(; vbc=30.0, boxlength=0.2, zstart=100.0) -> (; ...)

Boot a LIVE Enzo hydro cosmology grid directly from NATIVE CICASS HDF5 ICs (a
fully self-contained `CosmologySimulation`, ProblemType 30 — no external host
template): the realization's gas density/velocity and DM particle position/velocity
are written to Enzo's IC datasets in code units, with the cosmology (Ωb, Ωcdm, Ωr,
ΩΛ flat) set from the realization's own constants.  Reads the coherent gas–DM bulk
offset back out and confirms it survives into Enzo's data structures
(≈ vbc·(1+z)/1001 km/s).  Implemented in `MultiCodeCICASSExt` — `using CICASSLib`
activates it.
"""
function run_cicass_enzo end

"""
    run_cicass_ramses(; vbc=30.0, boxlength=0.2, zstart=100.0, ngrid=128) -> (; ...)

Boot a LIVE RAMSES (UNITS=COSMO, hydro on) purely on the CICASS grafic streaming
set — gas from `ic_deltab`/`ic_velb*`, DM particles from `ic_velc*` — then read
the mass-weighted gas bulk velocity and the DM particle bulk velocity back and
confirm the streaming offset survives into RAMSES.  Implemented in
`MultiCodeCICASSExt` — `using CICASSLib` activates it.
"""
function run_cicass_ramses end

"""
    run_music_discodj_phase_report(; res=32, seed=42, report_path=nothing)

Cross-generator phase audit for the IC roadmap: MUSIC is driven through an
explicit direct white-noise file and its Angulo-Pontzen mirror, while DISCO-DJ
is evaluated from its seed-driven NGenIC-compatible phase generator.  The
report records the fixed/mirror MUSIC control and the same-seed MUSIC↔DISCO-DJ
phase-proxy correlation, making any remaining white-noise inlet mismatch
measurable.  Implemented in `MultiCodeMusicDiscoDJExt` — load both `MusicLib`
and `DiscoDJLib` to activate it.
"""
function run_music_discodj_phase_report end

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

"""
    run_gadget4_halos(xp; box_mpch=50.0, omega_m=0.308, redshift=0.0) -> (; ngroups, …)

GADGET-4's FOF+SUBFIND as a harness SERVICE: particles in MultiCode's
conventions (N×3 rows, [0,1)³) become a halo catalogue.  The particle mass is
cosmologically consistent (m = Ωm·ρ_crit·box³/N) so the linking length is
0.2× the mean spacing.  Implemented in `MultiCodeGadget4Ext` — `using
Gadget4Lib` activates it (G4 runs in a child process; the D2 transport).
"""
function run_gadget4_halos end

end # module
