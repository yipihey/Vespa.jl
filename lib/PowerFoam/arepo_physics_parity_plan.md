# AREPO Physics Parity Plan

This plan freezes performance work and shifts PowerFoam toward physics parity
with AREPO for subsonic periodic turbulence.  The goal is to compare algorithms
only after both codes are provably starting from the same state and using the
same mesh, predictor, flux, timestep, and mesh-motion semantics.

## Acceptance Policy

- Prefer component parity over end-to-end agreement until each component is
  certified.
- Use AREPO single-rank, double-precision runs as the first external reference.
- Keep GPU/Metal comparisons same-precision and secondary until CPU Float64
  PowerFoam agrees with AREPO or has a documented tolerance.
- Keep `POWERFOAM_REBUILD=gpu_compact` as a performance path, but do not use it
  to hide geometry-parity failures.
- Record all gates as durable artifacts under `lib/PowerFoam/examples/out`.

## Phase 1: Fixture And Initial-State Parity

Goal: make PowerFoam and AREPO start from exactly the same cells.

Tasks:

- Add an AREPO initial-state gate that stages the stock 3-D turbulence IC,
  initializes AREPO through `ArepoLib`, exports rho, velocity, pressure, volume,
  centers, and live Voronoi geometry, then builds a PowerFoam `EulerState3D`.
- Compare PowerFoam conserved-to-primitive round trips against AREPO primitive
  fields on CPU Float64.
- Compare KA CPU Float32 and Metal Float32 round trips against the same Float32
  reference.
- Report mass, momentum, energy, vrms, Mach rms, density rms, and pressure
  extrema using AREPO volumes.

Gate:

- N4 and N8 initial-state max absolute differences for rho, velocity, and
  pressure are within Float64 roundoff on CPU.
- Float32 CPU/Metal fields are exactly equal or differ only by documented
  same-precision tolerance.

Status: started with `examples/arepo_initial_state_gate_3d.jl`.

Current evidence:

- N4 CPU Float64 passed against live AREPO primitives: rho/vx/vy/vz maxdiff
  `0`, pressure maxdiff `1.11022302463e-16`.
- N4 KA CPU Float32 round trip passed against `Float32(AREPO)`: rho/vx/vy/vz
  maxdiff `0`, pressure maxdiff `5.96046447754e-08`.
- N8 CPU Float64 passed against live AREPO primitives at the same pressure
  roundoff level.
- Metal is attempted by default, but the gate skips it cleanly when Metal.jl
  cannot expose a device in the current process.
- Artifact: `examples/out/arepo_initial_state_gate_3d/N4/README.md`.

## Phase 2: Geometry Parity

Goal: certify the mesh before comparing hydro updates.

Tasks:

- Extend the current geometry gate to compare AREPO live Voronoi face count,
  neighbor pairs, volumes, face areas, normals, face centers, and CSR ownership.
- Add a small-N artifact that lists mismatched cells/faces rather than only
  aggregate norms.
- Add a scan-backed compact canonical face CSR helper and tiny parity gate
  that checks the scan counts and offsets against the current CPU-reference
  compact row filter on small cases.
- Keep local compact rebuild comparisons separate from exact AREPO topology
  comparisons.

Gate:

- N4 and N8 exact topology/volume/face-field parity when using AREPO-exported
  geometry as input.
- Compact periodic rebuild differences are explicitly classified as an
  approximation until proven topology-equivalent.

Current evidence:

- N4 and N8 AREPO-exported geometry conversion preserve c1/c2 face pairs,
  volumes, face areas, normals, and CSR cell-face counts exactly.
- The external AREPO checkout was rebuilt with `HYDRO_RUNTIME_OPTIONS`, and
  staged parity cases now write `HydroRiemannSolver` explicitly.  Earlier
  solver-labelled artifacts from before this rebuild should not be treated as
  proof that AREPO used that runtime solver.
- Artifacts:
  - `examples/out/arepo_geometry_gate_3d/N4_dt0p001_hll/README.md`
  - `examples/out/arepo_geometry_gate_3d/N8_dt0p001_hll/README.md`

## Phase 3: Gradient, Limiter, Predictor, And Flux Parity

Goal: compare one hydro update as a sum of certified pieces.

Tasks:

- Keep the existing AREPO gradient parity gate as the gradient reference.
- Add predictor-state export/comparison for AREPO face extrapolation:
  spatial extrapolation, time extrapolation, limiter activation, moving-face
  frame transform, and lab-frame flux conversion.
- Compare HLL first, because the current decay build uses `RIEMANN_HLL`.
- Add LLF/HLLC/exact/PPM-style solver gates only after HLL passes.

Gate:

- For N4 and N8, face states and face fluxes match AREPO on the same geometry
  before the cell update is applied.
- One-step conserved update matches AREPO to Float64 tolerance on CPU.

Current evidence:

- N4 gradient parity against AREPO C gradients passed:
  density/pressure gradient maxdiff about `3.15544e-30`, velocity gradient
  maxdiff `2.22045e-16`.
- N8 gradient parity passed:
  density gradient maxdiff `1.26218e-29`, pressure gradient maxdiff
  `6.31089e-30`, velocity gradient maxdiff `6.66134e-16`.
- The AREPO face-trace bridge now exports per-face topology, pass index,
  per-side predictor timestep, moving-face geometry, pre-rotation predicted
  face states, local solver flux, and lab-frame flux.
- N4 HLL reconstructed face-trace parity passed for the pass-1 local-local
  face subset after runtime solver staging was made explicit:
  matched faces `144`, missing faces `0`, left-state maxdiff
  `(1.5987e-14, 2.0552e-8, 2.4680e-8, 3.0429e-8, 3.5763e-8)`,
  right-state maxdiff
  `(1.5987e-14, 2.5519e-8, 3.1972e-8, 3.3075e-8, 3.5763e-8)`,
  and lab-flux-times-area maxdiff
  `(1.1805e-9, 9.0037e-9, 8.5543e-9, 9.1071e-9, 3.6904e-9)`.
- N4 LLF reconstructed face-trace parity also passed on the same subset:
  matched faces `144`, missing faces `0`, with lab-flux-times-area maxdiff
  `(9.7055e-10, 7.8099e-9, 6.5505e-9, 5.4586e-9, 3.6904e-9)`.
- The local solver-flux diagnostic is solver-aware: HLL and LLF local fluxes
  independently match AREPO's pre-lab-frame trace to roundoff.
- A one-step gap diagnostic now aligns cells by particle ID and compares AREPO's
  native step against PowerFoam reconstructed and first-order updates on the
  same initial state.  At N4, PowerFoam predicts the AREPO first sync dt exactly
  (`0.03125`) but reconstructed primitive gaps remain finite after the
  reconstructed flux fix: rho `0.04533`, vx `0.1102`, vy `0.1527`,
  vz `0.1346`, pressure `0.05671`.
- AREPO records two hydro trace passes during the current `run_step!` diagnostic:
  pass 1 has `192` active faces (`144` local-local and `48` one-sided rows),
  while pass 2 has `502` active faces (`314` local-local and `188` one-sided
  rows).  The current face gate certifies only pass-1 local-local rows.
- AREPO also reorders gas cells during `run_step!`; in the N4 gate, `0/64`
  particle IDs remain in the same slot after the native step.  Exact replay of
  AREPO traced fluxes therefore needs the pre-flux ordered state/ID bridge for
  the same internal ordering used by the trace rows.
- Added and smoke-tested the external pre-flux snapshot bridge.  On the N4 HLL
  run it exports two snapshots matching the two trace passes; both have `64`
  cells, conserved mass sum `1.0`, and volume sums within roundoff of `1.0`.
  Pass 1 begins with IDs `[21, 22, 38, 37, 41, ...]`, while pass 2 begins with
  `[1, 2, 18, 17, 21, ...]`, confirming that the bridge captures the internal
  reorder boundary needed for exact replay.
- Added AREPO face-trace update-target indices (`update_c1/update_c2`) because
  one-sided geometric rows can still update local periodic ghost targets after
  AREPO maps `p >= NumGas` back with `p -= NumGas`.
- PowerFoam now keeps geometric face endpoints separate from local update
  targets.  `with_update_targets_3d` rebuilds the update CSR, `_cell_face_csr`
  handles side-2-only local rows (`c1 == 0, c2 > 0`), and
  `face_update_activity_3d` supplies update-aware activity for predicted-state
  flux replay without rewriting geometric `c1/c2`.
- Added `examples/arepo_trace_replay_gate_3d.jl`.  With pre-flux snapshots and
  update-target indices, replaying AREPO's own traced lab-frame fluxes
  reproduces AREPO's conserved updates to roundoff:
  - N4 HLL: trace rows `694`, snapshots `2`, max conserved gap
    `5.42219e-13`.
  - N4 LLF: trace rows `694`, snapshots `2`, max conserved gap
    `5.42064e-13`.
  The same gate now also runs PowerFoam's KA face-flux kernel and cell-update
  kernel from AREPO's traced face states and update-target CSR.  The PowerFoam
  kernel replay matches the scalar AREPO replay to the same tolerance:
  HLL max conserved gap `5.42219e-13`, LLF max conserved gap `5.42065e-13`.
  The bridge now also stores pass-local gradients in each pre-flux snapshot.
  `examples/arepo_face_trace_gate_3d.jl` uses those gradients and
  update-target owners to compare every active row in every traced pass:
  N4 HLL and LLF both pass across `694` rows, with max face-state differences
  `2.22045e-16` and max flux-area differences at roundoff.
  `examples/arepo_trace_replay_gate_3d.jl` now also runs a full PowerFoam
  predictor replay from AREPO pre-flux snapshots, pass-local gradients, traced
  face geometry, and update-target CSR.  It does not consume AREPO's traced
  face states or traced fluxes, and still reproduces AREPO's conserved update:
  HLL predictor replay max conserved gap `5.42219e-13`, LLF `5.42065e-13`.
  The same replay now has a native-geometry mode.  Rebuilding the local
  periodic Voronoi table from AREPO pre-flux generator positions, filtering
  faces with AREPO's `1e-5 * max(surface area)` rule, and using an area-weighted
  polygon face centroid reproduces AREPO pass geometry to roundoff:
  HLL/LLF native trace gates have `0` missing faces and `0` extra faces, with
  max face-center differences about `6.4e-14` in pass 2.  Feeding that native
  geometry into the PowerFoam predictor replay also passes: HLL and LLF native
  predictor replay max conserved gaps are both about `5.42e-13`.
  The AREPO bridge now exports the exact face endpoint positions and per-side
  `VelVertex` values used for `VF.p1/VF.p2`, so the same replay can compute
  AREPO's moving-face velocity correction natively.  With
  `POWERFOAM_REPLAY_GEOMETRY=native` and
  `POWERFOAM_REPLAY_FACE_VELOCITY=native`, HLL and LLF both pass; the native
  face-velocity diagnostic reports max differences `2.44943e-15` in pass 1
  and `3.67337e-14` in pass 2.
  The replay now also has independent update-target modes:
  `POWERFOAM_REPLAY_UPDATE_TARGETS=native` derives side ownership from exact
  endpoint/image identity, while `POWERFOAM_REPLAY_UPDATE_TARGETS=native_mesh`
  derives side ownership from the native mesh row orientation after normal
  alignment.  HLL and LLF both pass with native rows, native geometry, native
  face velocity, native-mesh update targets, and `0` update-target mismatches:
  HLL predictor replay gap `5.42201e-13`, LLF `5.42047e-13`.
  The native local periodic mesh builder now returns `face_image_shift`, which
  carries the periodic image offset needed to move this ownership logic into
  the production row builder instead of relying on traced endpoint positions.
  The bridge also exports each pre-flux snapshot's `All.Time`,
  `SphP[].TimeLastPrimUpdate`, and `P[].TimeBinHydro`; with
  `POWERFOAM_REPLAY_NATIVE_DT_SOURCE=snapshot_time`, the native-row replay now
  reconstructs AREPO's per-cell predictor extrapolation dt without consuming
  face-trace `state_dt_l/state_dt_r`.  HLL and LLF remain at the same
  roundoff-level replay gaps: HLL `5.42201e-13`, LLF `5.42047e-13`, with
  `0` update-target mismatches.
  This proves the remaining PowerFoam one-step gap is not hidden AREPO source
  ordering, predictor math, solver flux arithmetic, KA update arithmetic, or
  native face geometry, face-velocity reconstruction, update-target ownership,
  row ordering from the pass-local generator positions, or per-cell predictor
  extrapolation dt.  The remaining native pass-sequence work is reproducing the
  same drift/rebuild/reorder sequence without consuming AREPO pre-flux
  snapshots.
- Added `examples/arepo_face_trace_gate_3d.jl` as the direct predictor/flux
  parity gate, now active when the sibling AREPO and ArepoLib bridge patches
  are present.
- Added the bridge contract artifact
  `external_patches/arepo_bridge_face_trace_contract.md`, specifying the C ABI
  and Julia binding shape for face traces and hydro timebins.

## Phase 4: Mesh Motion And Regularization

Goal: reproduce AREPO generator motion, not only fluid advection.

AREPO decay configs enable:

- `REGULARIZE_MESH_CM_DRIFT`
- `REGULARIZE_MESH_CM_DRIFT_USE_SOUNDSPEED`
- `REGULARIZE_MESH_FACE_ANGLE`
- `CellShapingSpeed = 0.5`
- `CellMaxAngleFactor = 2.25`

Tasks:

- Port the AREPO mesh-velocity correction policy into PowerFoam.
- Add a gate comparing generator velocity corrections and post-drift centers.
- Keep simple fluid-velocity advection available as a diagnostic mode.

Gate:

- N4/N8 generator velocities and one-step displaced centers match AREPO within
  Float64 tolerance under the same timestep.

Current evidence:

- Added `arepo_mesh_velocity_3d`, reconstructing fluid velocity,
  pressure-gradient half-step, face-angle CM-drift fraction, and sound-speed
  regularization speed from exported AREPO fields.
- N4 live `SphP[].VelVertex` parity passed: max component diff
  `2.77556e-17`, rms component diff `1.65122e-17`.
- Artifact: `examples/out/arepo_mesh_velocity_gate_3d/N4/README.md`.

## Phase 5: Hierarchical Timestepping

Goal: match AREPO's active-cell semantics.

Tasks:

- Promote existing `arepo_hydro_dt_3d` and `arepo_timebin_3d` helpers into a
  scheduler with `TimeBinsHydro`-style active lists and synchronization.
- Apply partial hydro updates, partial drift, and partial mesh rebuilds only to
  active cells plus required halo regions.
- Keep full-step mode as a reference/debug path.

Gate:

- N4/N8 active-cell sets, timestep bins, and synchronized update times match
  AREPO for the decay setup.
- An opt-in N8 multirung fixture reaches multiple occupied bins and matches
  AREPO effective hydro bins, active-cell sets, active lists, and next
  synchronization steps for multiple native AREPO steps.

Current evidence:

- Added `arepo_system_step_3d`, `arepo_next_sync_step_3d`,
  `arepo_active_cells_3d`, `arepo_hydro_timebins_3d`, and
  `active_face_table_3d`.
- Added `finite_volume_reconstructed_hierarchy_step_3d!`, which packs active
  cells and advances with the existing KA active-cell reconstructed update.
- N4 one-step gate predicts AREPO's first synchronization dt exactly:
  PowerFoam predicted `0.03125`, AREPO observed `0.03125`.
- Unit suite covers active-cell selection, next-sync interval, packed active
  face tables, and uniform-flow preservation through the hierarchy step.
- Added `examples/arepo_hierarchy_gate_3d.jl` as the bridge-facing scheduler
  gate.  It skips cleanly until `ArepoLib.get_hydro_timebins` exists, then
  compares AREPO hydro bins, active masks, and next synchronization interval
  against PowerFoam's scheduler helpers.
- Added and rebuilt the external AREPO/ArepoLib hydro-timebin bridge.  The
  hierarchy gate now exports AREPO bins, synchronized masks, and
  `TimeBinsHydro.ActiveParticleList`.  It also exports `P[i].TimeBinGrav` so
  the gate can compare both raw hydro CFL bins and AREPO's effective hydro bins
  after `update_timesteps_from_gravity` clamps hydro to the gravity bin.
- N4 and N8 direct scheduler gates pass after one native AREPO step, i.e. after
  AREPO has assigned next-step bins:
  - N4 post-step: bin mismatches `0`, active mask mismatches `0`, active-list
    mismatches `0`, next sync step `16777216 / 16777216`.
  - N8 post-step: bin mismatches `0`, active mask mismatches `0`, active-list
    mismatches `0`, next sync step `8388608 / 8388608`.
- The N4 hierarchy gate was rerun after explicit `HydroRiemannSolver HLL`
  staging and still passed: bin mismatches `0`, active mask mismatches `0`,
  next sync step `16777216 / 16777216`.
- The same reports intentionally show the bootstrap `run_setup()` state as
  AREPO bin `0` for every gas cell; this is AREPO's pre-`find_timesteps` all
  active synchronized state, not the next-step scheduler assignment.
- Added `POWERFOAM_HIERARCHY_FIXTURE=multirung`, which perturbs the staged N8
  IC internal energy to force multiple timestep bins while leaving the default
  decay fixture unchanged.  The N8 HLL multirung gate passes for one and three
  AREPO steps: `occupied_bins=2`, effective bin mismatches `0`, active mask
  mismatches `0`, active-list mismatches `0`, and next sync
  `1048576 / 1048576`.  The report also preserves raw hydro-bin mismatches so
  the gravity limiter remains visible rather than hidden.

## Phase 6: Physical-Time Decay Parity

Goal: compare the actual decaying subsonic turbulence problem.

Tasks:

- Run AREPO and PowerFoam from the same IC to `TimeMax = 1.0`.
- Compare snapshots at 0.25, 0.5, 0.75, and 1.0.
- Track mass, momentum, energy, vrms, Mach rms, density rms, spectra, and
  positivity margins.
- Use N12/N16 first; keep N24/N32 for later once positivity and timestep
  hierarchy pass.

Gate:

- PowerFoam CPU Float64 matches AREPO within a documented tolerance for the
  canonical HLL decay run.
- GPU Float32 then demonstrates same-precision CPU/Metal parity and a separate
  documented Float32-vs-Float64 tolerance.

## Current Known Gaps

| Gap | Impact | First gate |
| --- | --- | --- |
| Compact local rebuild topology is certified only on the N4 traced post-drift passes, not yet over larger N or longer drift histories | Mesh differences can dominate hydro differences | N8/N12 post-drift geometry parity |
| Native PowerFoam rebuild geometry, moving-face velocity, row ordering, update-target ownership, and predictor extrapolation dt now match AREPO traced pass data, but native-row replay still uses AREPO pre-flux snapshots and gradients | Full one-step update can diverge when running without trace assistance | Native pass-sequence/reorder gate |
| Scheduler-only hierarchical timestep parity is certified for default N4/N8 decay and the controlled N8 multirung fixture, but native partial drift/rebuild/update is not yet coupled to it | Long-time evolution can diverge once only a subset of cells is active | Native multirung pass-sequence gate |
| Compact/local native rebuild exactness after repeated drifts is not yet certified | Moving mesh can diverge after later rebuilds | Multi-step post-drift geometry gate |
| Long fixed-step run can hit negative pressure | Physical-time decay run is not yet robust | Scheduler/positivity gate |
| 2-D periodic compact path incomplete | 2-D turbulence parity remains behind 3-D | Separate 2-D compact plan |

## AREPO Bridge Exports Used By The Parity Gates

Full one-step parity needs bridge visibility below the field-update level while
the native PowerFoam pass sequence is being brought up.  The current gates use:

- Active hydro timebin per gas cell and current active-cell list.
- Pre-flux ordered gas-cell IDs, conserved/primitive state, volumes, centers,
  generator velocities, snapshot time, per-cell `TimeLastPrimUpdate`, and
  hydro timebin for each hydro trace pass.  Implemented in the sibling
  AREPO/ArepoLib checkout as
  `arepo_get_hydro_preflux_state_*_3d` /
  `arepo_get_hydro_preflux_timing_3d` /
  `ArepoLib.get_hydro_preflux_states_3d`.
- Per-face left/right states after spatial extrapolation, time extrapolation,
  velocity-frame rotation, and limiter application.
- Per-face HLL flux after advection and lab-frame conversion, with face area and
  responsibility metadata.
- Per-face update-target indices for the local cells that receive each side's
  flux contribution.  Implemented in the sibling AREPO/ArepoLib checkout as
  `arepo_get_hydro_face_trace_update_indices_3d` and returned by
  `ArepoLib.get_hydro_face_traces_3d` as `update_c1/update_c2`.
- Per-face exact endpoint positions and per-side endpoint `VelVertex` values
  used by AREPO's moving-face velocity correction.  Implemented in the sibling
  AREPO/ArepoLib checkout as `arepo_get_hydro_face_trace_points_3d` and
  returned by `ArepoLib.get_hydro_face_traces_3d` as
  `point_l/point_r/velvertex_l/velvertex_r`.
- Post-drift/pre-flux generator positions and post-rebuild face topology for the
  same synchronization step.  Implemented for pre-flux generator positions in
  the sibling AREPO/ArepoLib checkout and consumed by the native rebuild trace
  and native replay gates.

These exports now prove exact predictor/flux/update parity against AREPO
internals when the trace owns update-target metadata.  The remaining native
work is to reproduce the update-target ownership and pass sequencing directly
inside PowerFoam so the production path no longer needs trace metadata.

Concrete bridge interface:

- `arepo_get_hydro_timebins(...)` exports `P[i].TimeBinHydro`,
  `TimeBinSynchronized`, `TimeBinsHydro.ActiveParticleList`,
  `All.Ti_Current`, and `All.Timebase_interval`.
- `arepo_get_hydro_face_traces_3d(...)` exports face topology, active flags,
  face timestep, area, normal, face center, face velocity, center states,
  predicted face-frame states, and lab-frame fluxes.
- `arepo_get_hydro_preflux_state_3d(...)` exports one snapshot per hydro pass
  with the gas-cell ordering and IDs used by that pass's face rows.
- `ArepoLib.get_hydro_timebins(h)` and
  `ArepoLib.get_hydro_face_traces_3d(h)` wrap those C calls.
- `ArepoLib.get_hydro_preflux_states_3d(h)` wraps the new pre-flux snapshot
  calls.
- PowerFoam consumer gate:
  `examples/arepo_face_trace_gate_3d.jl`.
- The scheduler subset is implemented in the sibling AREPO and ArepoLib
  checkouts and mirrored as `external_patches/arepo_bridge_timebins.patch`.
  It unlocks `examples/arepo_hierarchy_gate_3d.jl`.

## Latest Verification

- Focused LLF face gate:
  `julia --project=lib/PowerFoam -e 'push!(LOAD_PATH, "/Users/tabel/Projects/Arepo.jl/lib/ArepoLib"); ARGS=["4","0.001","llf","1"]; include("lib/PowerFoam/examples/arepo_face_trace_gate_3d.jl")'`
  passed.
- Focused HLL one-step gap:
  `julia --project=lib/PowerFoam -e 'push!(LOAD_PATH, "/Users/tabel/Projects/Arepo.jl/lib/ArepoLib"); ARGS=["4","0.001","hll","1"]; include("lib/PowerFoam/examples/arepo_one_step_gap_3d.jl")'`
  still reports the finite reconstructed primitive gaps listed above.
- Focused HLL hierarchy gate:
  `julia --project=lib/PowerFoam -e 'push!(LOAD_PATH, "/Users/tabel/Projects/Arepo.jl/lib/ArepoLib"); ARGS=["4","0.001","hll","1"]; include("lib/PowerFoam/examples/arepo_hierarchy_gate_3d.jl")'`
  passed.
- Focused HLL multirung hierarchy gate:
  `POWERFOAM_HIERARCHY_FIXTURE=multirung julia --project=lib/PowerFoam -e 'push!(LOAD_PATH, "/Users/tabel/Projects/Arepo.jl/lib/ArepoLib"); ARGS=["8","0.001","hll","3"]; include("lib/PowerFoam/examples/arepo_hierarchy_gate_3d.jl")'`
  passed with two occupied bins and `0` effective-bin / active-list
  mismatches.
- PowerFoam unit suite passes with the bridge-facing additions: `253/253`.
- Pre-flux bridge smoke check passed after rebuilding
  `/Users/tabel/Projects/arepo/libarepo.dylib`: trace rows `694`, snapshots
  `2`, snapshot pass indices `1` and `2`, and mass sums `1.0` for both.
- Trace replay gates passed:
  - HLL: `examples/out/arepo_trace_replay_gate_3d/N4_dt0p001_hll/README.md`
    reports max conserved gap `5.42219e-13`, including PowerFoam KA kernel
    replay and full PowerFoam predictor replay.  With
    `POWERFOAM_REPLAY_ROWS=native`,
    `POWERFOAM_REPLAY_GEOMETRY=native`,
    `POWERFOAM_REPLAY_FACE_VELOCITY=native`,
    `POWERFOAM_REPLAY_UPDATE_TARGETS=native_mesh`, and
    `POWERFOAM_REPLAY_NATIVE_DT_SOURCE=snapshot_time`, the artifact reports
    native-row predictor replay gap `5.42201e-13` and `0` update-target
    mismatches.
  - LLF: `examples/out/arepo_trace_replay_gate_3d/N4_dt0p001_llf/README.md`
    reports max conserved gap `5.42064e-13` and PowerFoam KA kernel replay
    and predictor replay gaps `5.42065e-13`.  With
    the same native-row/native-geometry/native-face-velocity/native-mesh-target
    mode, the predictor replay gap is `5.42047e-13` with `0` update-target
    mismatches.
- Native rebuild trace gates passed:
  - HLL:
    `examples/out/arepo_native_rebuild_trace_gate_3d/N4_dt0p001_hll/README.md`
    reports both passes matched, `0` missing faces, `0` extra faces,
    max area diff `1.07553e-15`, max normal diff `2.71948e-16`,
    max center diff `6.38164e-14`, and max volume diff `2.23779e-16`.
  - LLF:
    `examples/out/arepo_native_rebuild_trace_gate_3d/N4_dt0p001_llf/README.md`
    reports the same native geometry pass after the solver switch.
- All-pass direct face-trace gates passed:
  - HLL:
    `examples/out/arepo_face_trace_gate_3d/N4_dt0p001_hll_allpasses/README.md`
    reports two passes, `694` matched rows, no missing rows, max state
    difference `2.22045e-16`, and max flux-area difference `2.77556e-17`.
  - LLF:
    `examples/out/arepo_face_trace_gate_3d/N4_dt0p001_llf_allpasses/README.md`
    reports two passes, `694` matched rows, no missing rows, max state
    difference `2.22045e-16`, and max flux-area difference `1.38778e-17`.
- Optional pre-flux smoke gate passed:
  `examples/out/arepo_preflux_smoke_gate_3d/N4_dt0p001_hll/README.md`
  reports two snapshots, trace pass indices `1, 2`, unique IDs, mass sums `1`,
  and volume sums `1`.
- PowerFoam unit suite remained green after the new bridge-facing gates,
  update-target CSR unit coverage, side-2-only local row coverage, and
  update-target face activity helper.  It also remained green after adding
  AREPO-style local 3-D face filtering and area-weighted face centroids:
  `248/248`.  After adding native row replay controls, native
  `face_image_shift` metadata, and moving-mesh timestep helper coverage, the
  suite is green at `253/253`.
