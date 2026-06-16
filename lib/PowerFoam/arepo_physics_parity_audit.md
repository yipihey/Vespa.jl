# AREPO Physics Parity Audit

Date: 2026-06-14

This audit summarizes what is still missing before PowerFoam can claim physics
parity with AREPO for the 3-D decaying subsonic turbulence target.  Performance
claims should remain secondary until the CPU Float64 physics path closes these
gaps.

## Current Certified Pieces

- Initial-state parity is certified at N4/N8 for live AREPO primitive fields
  and PowerFoam conserved/primitive round trips.
- AREPO-exported geometry conversion is certified at N4/N8 for face pairs,
  volumes, areas, normals, and CSR counts.
- Gradient parity is certified at N4/N8 against AREPO C gradients.
- Mesh generator velocity reconstruction is certified at N4 against
  `SphP[].VelVertex` to roundoff for the current regularization options.
- HLL and LLF trace replay are certified for the full N4 traced two-pass hydro
  update: AREPO scalar trace replay and PowerFoam KA face-flux/cell-update
  replay both reproduce AREPO final conserved fields at about `5.4e-13`.
- HLL and LLF all-pass direct face-trace parity are certified for every active
  row in both N4 traced hydro passes (`694` rows): PowerFoam recomputes
  predictor states and moving-face fluxes from AREPO pass-local snapshots,
  gradients, and face geometry with roundoff-level differences.
- HLL and LLF full predictor replay are certified on AREPO pass geometry:
  PowerFoam recomputes face states, fluxes, and conservative updates without
  consuming AREPO traced face states or traced fluxes, still reproducing AREPO
  final conserved fields at about `5.4e-13`.
- Native 3-D local periodic rebuild geometry is certified for the traced N4
  post-drift/pre-flux passes when supplied AREPO pre-flux generator positions:
  the native table applies AREPO's tiny-face cutoff
  (`1e-5 * max(surface area)`) and an area-weighted polygon face centroid, and
  HLL/LLF gates match both traced passes with `0` missing and `0` extra faces.
- HLL and LLF full predictor replay are also certified on native reconstructed
  pass geometry when using AREPO's traced update targets and native
  face-velocity reconstruction:
  native-geometry replay reproduces AREPO final conserved fields at about
  `5.4e-13`.
- Native face-velocity reconstruction is certified for the traced N4
  post-drift/pre-flux passes after exporting the exact AREPO face endpoint
  positions and per-side `VelVertex` values used by `VF.p1/VF.p2`: the HLL/LLF
  native replay diagnostics show max face-velocity differences
  `2.44943e-15` in pass 1 and `3.67337e-14` in pass 2, and native
  geometry/native face-velocity predictor replay still reproduces AREPO final
  conserved fields at about `5.4e-13`.
- Native update-target ownership is certified for the traced N4 HLL/LLF pass
  sequence in diagnostic replay mode: `POWERFOAM_REPLAY_UPDATE_TARGETS=native`
  derives side ownership from exact endpoint/image identity, and
  `POWERFOAM_REPLAY_UPDATE_TARGETS=native_mesh` derives side ownership from the
  native mesh row orientation after normal alignment.  HLL/LLF both pass with
  native rows, native geometry, native face velocity, and native-mesh update
  targets, with `0` update-target mismatches and final-field replay gaps about
  `5.4e-13`.
- Native row generation no longer depends on AREPO trace row ordering for the
  N4 HLL/LLF predictor replay.  The AREPO bridge now exports each pre-flux
  snapshot's `All.Time`, `SphP[].TimeLastPrimUpdate`, and `P[].TimeBinHydro`,
  and `POWERFOAM_REPLAY_NATIVE_DT_SOURCE=snapshot_time` reconstructs
  AREPO's per-cell predictor extrapolation dt without reading face-trace
  `state_dt_l/state_dt_r`.  HLL/LLF both remain at about `5.4e-13` with
  `0` update-target mismatches.  The diagnostic still uses AREPO pass-local
  snapshots and gradients while the native pass sequence is being brought up.
- Hydro timebin quantization and active-list parity are certified for the
  default N4/N8 synchronized decay states and for an opt-in N8 multirung
  scheduler fixture.  The multirung gate distinguishes raw hydro CFL bins from
  the effective AREPO hydro bins after gravity-bin limiting, and passes for
  three native AREPO steps with `0` effective-bin and active-list mismatches.
- Unit coverage is green at `253/253`.

## Blocking Gaps

### 1. Native update-target and pass-sequence ownership

The largest remaining gap is not the face-flux formula or conservative update
arithmetic, and it is no longer the predictor state construction when AREPO
pass-local gradients and face geometry are supplied.  It is also no longer the
local periodic face table itself, the moving-face velocity reconstruction, or
the update-target ownership for the traced N4 passes.  The remaining gap is
that the native replay still consumes AREPO trace metadata for pass-local cell
state reconstruction and the pass sequence:

- the same sequence of pass-local snapshots and internal gas-cell reorder
  boundaries.

The full native `run_step!` analogue also still needs to construct the same
sequence of pre-flux ordered states, post-drift/pre-flux generator positions,
old volumes for gradients/fluxes, new volumes for conservative update, and
internal gas-cell reorder boundaries without asking AREPO for trace rows.

Closed gate:

- Replaced `POWERFOAM_REPLAY_NATIVE_DT_SOURCE=trace_cells` with
  `snapshot_time`, derived from exported pre-flux snapshot timing, while keeping
  the native-row HLL/LLF replay at the current `5.4e-13` level.

Next gate to close it:

- Reproduce the same pass-local snapshot sequence and internal gas-cell reorder
  boundaries from the native scheduler, rather than receiving them from AREPO.

### 2. Native update-target face table

PowerFoam now has the primitive pieces: geometric endpoints can remain separate
from update targets, side-2-only rows are supported, and update-target face
activity exists.  The native local periodic rebuild now also carries per-face
periodic image-shift metadata and the native-row replay uses the native face
table directly.  What is still missing is promoting that diagnostic row table
  into the production pass sequence with native pre-flux snapshot/reorder
  ownership.

First gate to close it:

- Promote the native-row table into the production `run_step!` analogue and
  compare counts, signs, one-sided geometric rows, one-sided update rows, and
  no-update rows against AREPO trace pass tables without reading
  `trace.update_c1/update_c2`.

### 3. Direct predictor/flux parity beyond pass-1 local-local rows

Closed for AREPO pass geometry.  `arepo_face_trace_gate_3d.jl` now operates on
each pre-flux snapshot and each pass geometry/update table, and HLL/LLF pass on
all active rows.  This item remains relevant only as a regression gate while the
native rebuild catches up.

### 4. Multi-rung hierarchical timestepping

The scheduler-only multirung gate is now closed for the controlled N8 fixture:
`POWERFOAM_HIERARCHY_FIXTURE=multirung` reaches two occupied hydro bins and
passes for three AREPO steps after applying AREPO's gravity-bin limiter to the
raw hydro CFL bins.  This certifies bin quantization, active masks, active
lists, and next synchronization steps for the bridge-facing scheduler helper.

What remains is the production coupling: partial drift, partial local rebuild
halo, active-face update ownership, and final conserved-field parity while only
a subset of cells is active.

Next gate to close it:

- Run the native pass-sequence driver through the same N8 multirung fixture and
  compare active faces, partial rebuild geometry, and final conserved fields at
  every synchronization point.

### 5. Exact native tessellation/rebuild parity after repeated drift

The current native periodic/local rebuild is now topology-equivalent to AREPO
for the traced N4 first-step post-drift passes, including face pairs, volumes,
areas, normals, and face centers.  It is still a near-lattice local halfspace
clipper and GPU production rung, not yet a proven topology-equivalent
replacement for AREPO's Delaunay-backed tessellation on larger grids or after
many mesh-motion cycles.

First gate to close it:

- Extend native post-drift rebuild comparison to N8/N12 and multiple sync
  points: face pair set, duplicate/periodic images, volumes, face areas,
  normals, centers, and update-target ownership.

### 6. Long-time positivity and physical decay diagnostics

The one-step direct gap remains finite and prior long fixed-step native runs can
hit negative pressure.  Until the pass sequence and hierarchy gates are closed,
the TimeMax physical decay comparison is not yet meaningful as a physics-parity
claim.

First gate to close it:

- After the one-step pass-sequence gate passes, run N12/N16 to fixed physical
  output times and compare mass, momentum, energy, vrms, Mach rms, density rms,
  spectra, and positivity margins against AREPO.

## Non-Blocking But Needed Before Final Claims

- GPU/Metal parity must be re-established for the final CPU Float64 physics
  path once the native pass sequence is correct.
- 2-D periodic compact parity remains behind the 3-D target and should not
  block the 3-D decaying turbulence claim unless the final claim includes 2-D.
- HLLC/exact/PPM-style solver comparisons should wait until HLL/LLF geometry
  and hierarchy parity are stable.

## Recommended Order

1. Reproduce AREPO's pass-local pre-flux snapshot sequence and internal
   gas-cell reorder boundaries in the native scheduler.
2. Run native geometry/update-target/face-velocity predictor replay without
   consuming AREPO face geometry, trace row ordering, or trace update metadata.
3. Promote the native-row pass table into the production `run_step!` analogue.
4. Extend native rebuild topology gates to N8/N12 and multiple sync points.
5. Couple the production `run_step!` analogue to the certified multirung
   scheduler and verify partial drift/rebuild/update parity.
6. Run N12/N16 physical-time decay parity.
7. Only then re-open GPU speedups and larger-grid performance claims.
