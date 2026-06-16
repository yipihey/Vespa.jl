# Arepo.jl Full KA Rewrite Master Plan

## Goal

Build a complete Julia/KernelAbstractions rewrite of the production-relevant
AREPO stack that can run hydro and cosmology test problems at close to the same
accuracy as the original C AREPO, while remaining portable across CPU, Metal,
CUDA/AMDGPU-capable backends, and other KA targets.

The rewrite must not be judged by "it runs" alone.  Promotion requires
component parity, bridge parity against the original C code, final-field
agreement on standard problems, and backend parity between KA CPU and GPU.

## Definitions

- `Arepo.jl`: user-facing Julia package and compatibility layer.  It should be
  able to run either the current C bridge or the pure-Julia KA implementation.
- `PowerFoam`: current location of the AREPO-style mesh/hydro rewrite.  It is
  the rewrite engine until package boundaries are split.
- `AREPO bridge`: live C AREPO/ArepoLib comparison path used as the oracle.
- `KA CPU`: KernelAbstractions CPU backend, used as the single-source reference
  for portable kernels.
- `GPU backend`: Metal first on this laptop; CUDA/AMDGPU should remain possible
  by keeping kernels and memory layouts backend-neutral.

## Architecture

```text
Arepo.jl
  |
  |-- Runtime/API layer
  |     |-- parameter/config parser
  |     |-- problem registry
  |     |-- run loop and output policy
  |     |-- bridge/rewrite backend selector
  |
  |-- KA rewrite engine
  |     |-- mesh/tessellator
  |     |-- hydro
  |     |-- gravity/cosmology
  |     |-- time integration/hierarchy
  |     |-- IO and diagnostics
  |
  |-- Verification harness
        |-- C AREPO bridge gates
        |-- CPU vs GPU parity gates
        |-- standard hydro/cosmology problem matrix
        |-- performance/scaling reports
```

## Non-Negotiable Engineering Rules

- The KA CPU path is the source of truth for portable kernels.
- GPU kernels must be structure-of-arrays, fixed-shape or explicitly compacted,
  and avoid host staging on production paths.
- Every GPU kernel gets a CPU backend parity gate before Metal/CUDA claims.
- Every physical feature gets a bridge or analytic gate before being considered
  complete.
- C AREPO remains the oracle until the rewrite passes the relevant gates.
- Diagnostic knobs stay opt-in.  Default kernels must stay lean.
- Degenerate mesh cases are correctness work, not performance work.

## Current State Snapshot

### Implemented Or Partially Implemented

- 2-D hydro face-table solver, reconstruction, moving mesh, and periodic
  turbulence-oriented paths.
- 3-D hydro face-table solver, gradients, predictor, reconstructed update,
  active-cell/hierarchy-facing paths, and moving mesh scaffolding.
- CPU Delaunay-derived 3-D tessellator reference for perturbed periodic cases.
- Tessellator SoA payloads and KA kernels for periodic image generation,
  dense candidate/halo rows, and circumcenter recomputation.
- AREPO bridge gates for initial state, pre-flux state, face traces, mesh
  velocity, gradients, hierarchy, native rebuild, and selected standard
  problems.
- Existing C bridge can still run original AREPO as the oracle.

### Known Major Gaps

- Production Delaunay/Voronoi semantics are not fully KA/GPU-resident yet.
- Delaunay degeneracy and AREPO tie-breaking for exact lattices are unresolved.
- Candidate compaction, predicate kernels, edge-ring face extraction, compact
  face table, and CSR rebuild are still incomplete on device.
- Gravity/cosmology rewrite is not integrated as an AREPO-equivalent runtime.
- Snapshot/IC/parameter compatibility is incomplete.
- MPI/domain decomposition is not rewritten; single-node shared-memory/GPU is
  the first target.
- Full standard problem matrix is not yet automated as blocking CI.

## Workstreams

### A. Runtime And Package Boundary

Purpose: make `Arepo.jl` a real Julia runtime, not only a bridge wrapper.

Deliverables:

- `Arepo.run(problem_or_param; backend=:bridge|:ka, device=KA.CPU(), options...)`
- Problem registry for hydro and cosmology examples.
- Parameter mapping from AREPO-style `param.txt`/compile flags to Julia runtime
  options where feasible.
- Runtime state object with particles, gas cells, mesh, hydro variables,
  gravity variables, timestep hierarchy, diagnostics, and output policy.
- Backend selector that can run the C bridge, KA CPU, and GPU backends through
  the same high-level harness.

Acceptance:

- Same driver can run at least one bridge problem and one pure-KA problem.
- Outputs common JSON/CSV/HDF5 diagnostics for comparison.
- No source-module dependency on C bridge inside core KA kernels.

### B. Mesh And Tessellator

Purpose: reproduce AREPO moving Voronoi mesh semantics on KA backends.

Phases:

1. CPU semantic reference
   - Status: started.
   - Complete Delaunay insertion/face extraction for perturbed periodic cases.
   - Add exact/tie-breaking policies for co-spherical lattice cases.
   - Compare against bridge N4/N8/N12 topology gates.

2. KA SoA layout
   - Status: underway.
   - Keep points, periodic images, candidates, tetrahedra, circumcenters, faces,
     and CSR as backend-resident SoA buffers.
   - Continue converting dense candidate rows to compact active buffers.

3. Device kernels
   - Periodic image generation: done.
   - Dense candidate/halo rows: done.
   - Circumcenter recomputation: done.
   - Remaining: candidate compaction, predicates, tetra insertion/repair,
     edge-ring face extraction, face compaction, volume/center update, CSR.

4. Incremental/hierarchical rebuild
   - Dirty cell detection.
   - Halo expansion.
   - Dirty face rebuild.
   - Fallback to full rebuild if volume/topology checks fail.

Acceptance:

- N4/N8/N12 bridge topology gates pass for full rebuild.
- KA CPU vs GPU compact arrays match exactly or with documented same-precision
  tolerance.
- Moving-mesh hydro can rebuild without host staging in production mode.
- Incremental rebuild passes active-rung final-field gates before becoming
  default.

### C. Hydro

Purpose: reproduce AREPO hydro behavior for 2-D and 3-D moving Voronoi meshes.

Required features:

- Primitive/conserved conversion and equation-of-state handling.
- LLF, HLL, HLLC where applicable; runtime solver selection.
- Gradient construction, slope limiting, predictor, face states.
- Moving-face ALE fluxes and mesh velocity semantics.
- First-order and reconstructed paths.
- Active-cell/hierarchical timestep update.
- Positivity/stability controls consistent with AREPO where possible.
- Passive scalars/tracers.
- Optional local PPM/PPM-like reconstruction only after baseline parity.

Test problems:

- 1-D: acoustic wave, shock tube, interacting blastwaves.
- 2-D: KH, Noh, Gresho, Yee, contact/blob, odd-even shock, shearing sinusoid.
- 3-D: Noh, decaying subsonic turbulence, Sedov/blast proxy, smooth wave.

Acceptance:

- Uniform-flow preservation on static and moving meshes.
- Conservation of mass/momentum/energy to roundoff in closed periodic boxes.
- CPU bridge vs KA agreement on one-step face traces for selected N.
- Final-field agreement with C AREPO within problem-specific tolerances.
- GPU backend matches KA CPU before any speed claims.

### D. Gravity And Cosmology

Purpose: run gravity-only and hydro+gravity cosmology problems with AREPO-like
accuracy.

Phases:

1. Gravity MVP
   - Direct summation for small N.
   - Periodic PM gravity using existing Poisson/FFT infrastructure where
     available.
   - Kick-drift-kick integration.

2. Cosmological variables
   - Comoving coordinates.
   - Scale factor/time integration.
   - Cosmological source terms for hydro.
   - Unit conversion and Hubble parameters.

3. Tree/PM hierarchy
   - PM long-range force.
   - Tree or approximate short-range force if needed for AREPO parity.
   - Adaptive softenings where required.

4. Cosmology examples
   - Gravity-only box.
   - Zoom gravity-only.
   - Small hydro box without star formation.
   - Star formation only after gravity/hydro baseline passes.

Acceptance:

- Direct gravity matches analytic pairwise checks.
- PM force agrees with controlled reference spectra.
- Gravity-only cosmology mass/power statistics track C AREPO examples.
- Hydro cosmology conserves expected quantities in controlled small boxes.

### E. IO, ICs, And Diagnostics

Purpose: run the same problems and compare outputs without bespoke scripts each
time.

Required features:

- Read/write HDF5 ICs and snapshots compatible with AREPO examples.
- Support coordinates, velocities, masses, internal energy, density,
  center-of-mass, passive scalars, particle IDs, timebins.
- Lightweight diagnostic CSV/JSON summaries for every gate.
- Stable report directories with run tags encoding backend, solver,
  reconstruction, mesh algorithm, and precision.

Acceptance:

- Pure-Julia runtime can ingest at least the key hydro and cosmology ICs used by
  the gate matrix.
- Bridge and rewrite outputs can be compared by one common analysis path.

### F. Verification And CI

Purpose: make correctness promotion mechanical.

Gate tiers:

- Tier 0: unit/component gates, always fast.
- Tier 1: KA CPU parity and small standard problems.
- Tier 2: bridge parity at N4/N8/N12.
- Tier 3: GPU backend parity and larger standard problems.
- Tier 4: performance/scaling and production-style examples.

Promotion rule:

- A feature is diagnostic when it has an example or report.
- A feature is experimental when it passes component tests but not bridge gates.
- A feature is production-candidate when it passes bridge gates and KA CPU/GPU
  parity.
- A feature becomes default only after final-field problem gates pass.

## Milestones

### M0: Planning And Ownership

- Produce master plan.
- Produce per-workstream breakdowns.
- Create gate matrix.
- Assign agent ownership.

Exit criteria:

- Plan artifacts exist under `lib/PowerFoam/planning/`.
- Each workstream has next actions and acceptance gates.

### M1: Pure-KA Hydro Runtime MVP

Scope:

- Package-level API for running a small pure-KA hydro problem.
- 2-D and 3-D static/moving mesh hydro examples through one driver.
- CPU backend first, GPU parity second.

Exit criteria:

- KH 2-D and Noh 3-D smoke run through `Arepo.run(...; backend=:ka)`.
- Existing PowerFoam tests remain green.

### M2: Mesh/Tessellator Production Candidate

Scope:

- Complete compact candidate/face/CSR device path.
- N4/N8 bridge topology gates.
- Degeneracy/tie policy.

Exit criteria:

- Delaunay reference and KA compact mesh agree.
- N4/N8 bridge topology gates pass or have documented geometry-equivalent
  acceptance for degenerate cases.

### M3: Hydro Physics Parity

Scope:

- Solver/reconstruction/predictor/hierarchy parity.
- Standard hydro matrix.

Exit criteria:

- Noh, KH, Gresho/contact/blob, turbulence, and wave gates pass within defined
  tolerances on KA CPU and selected GPU backend.

### M4: Gravity/Cosmology MVP

Scope:

- Direct gravity and periodic PM.
- Gravity-only cosmology box.
- Common IC/snapshot path.

Exit criteria:

- Cosmology gravity-only N32-style gate tracks AREPO mass/power diagnostics.

### M5: Integrated Hydro+Cosmology

Scope:

- Hydrodynamics plus cosmological integration.
- Small hydro cosmology example.

Exit criteria:

- Small hydro cosmology problem runs end-to-end with C AREPO comparison metrics.

### M6: Production Readiness

Scope:

- Device-resident hot paths.
- Stable reports.
- Performance/scaling.
- Documentation.

Exit criteria:

- GPU path beats KA CPU and original-C bridge for at least selected supported
  problem sizes on this machine when idle.
- All production-candidate features have documented gates and limitations.

## Parallel Agent Strategy

Agents should work on bounded, non-overlapping slices:

- Hydro parity breakdown and first implementation target.
- Mesh/tessellator KA kernel ladder.
- Gravity/cosmology rewrite plan and first direct/PM gate.
- Verification/CI gate matrix and report conventions.
- IO/runtime API boundary.

The main orchestrator owns:

- Master plan.
- Cross-workstream dependencies.
- Source integration.
- Final gate execution.
- Promotion decisions.

## Planning Artifacts

The first parallel planning pass has produced concrete work breakdowns that now
serve as the execution queue:

- `hydro_parity_workbreakdown.md`: hydro solver, reconstruction, predictor,
  moving-face, hierarchy, and standard-problem parity work.
- `tessellator_mesh_workbreakdown.md`: production Delaunay/Voronoi semantics,
  KA SoA conversion, degeneracy policy, compact face/CSR rebuild, and topology
  bridge gates.
- `cosmology_gravity_workbreakdown.md`: direct-force oracle, PM gravity MVP,
  comoving integration, IC/snapshot handling, and cosmology gate ladder.
- `verification_gate_matrix.md`: blocking/non-blocking gate taxonomy and
  promotion lanes for bridge, KA CPU, GPU, and standard problems.
- `runtime_api_scaffold.md`: API boundary for the user-facing AREPO-style
  runtime objects and future `Arepo.run` entrypoint.
- `gravity_component_gate.md`: tiny-`N` direct-force oracle gate that will be
  used to certify PM/tree gravity components.
- `hydro_problem_registry.md`: standard hydro problem registry and promotion
  criteria for KH, Noh, Gresho, waves, and turbulence.
- `io_parameter_compatibility.md`: parameter, IC, snapshot, restart, and output
  compatibility audit for AREPO-like runtime parity.

The current integrated code artifact from that pass is
`src/arepo_runtime_scaffold.jl`, which is included and exported by `PowerFoam`
as the seed API for `ArepoRunOptions`, `ArepoProblemSpec`,
`ArepoRuntimeState3D`, and `arepo_run_scaffold`.

## Execution Pass: 2026-06-15

Integrated since the first master-plan pass:

- Runtime scaffold now includes `ArepoHydroSmokeAssessment` and
  `classify_ka_hydro_smoke`, with KH2D/Noh3D-like smoke diagnostics in
  `examples/arepo_runtime_scaffold_smoke.jl`.
- Tiny direct-gravity oracle is available through
  `src/arepo_gravity_scaffold.jl` and package exports:
  `arepo_direct_gravity_accel!`, `arepo_direct_gravity_accel`,
  `arepo_direct_gravity_potential_energy`, and
  `arepo_direct_gravity_oracle`.
- KA tessellator SoA ladder advanced beyond candidate stencils:
  candidate-vs-tetra in-sphere predicate buffers and fixed-shape conflict
  tetra-face row emission are now CPU-backend tested.  A fixed-shape
  boundary-face deduplication mask is also present with a small synthetic unit
  gate; large-cavity production compaction remains next.
- Runtime param-spec smoke now exercises the exported AREPO-style parameter and
  config parser, infers an `ArepoProblemSpec`, and records scaffold/runtime
  diagnostics in a timestamped artifact.
- Verification helper now distinguishes runnable versus planned gates by file
  existence and prints grouped summaries.
- Hydro problem registry smoke and IO/parameter audit scripts now produce
  lightweight artifacts without running heavyweight physics gates.
- Runtime hydro smoke, direct-gravity smoke, and Noh2D proxy-readiness gates
  now write timestamped artifacts under `examples/out/`.
- Fixed-stride per-candidate boundary-face packing is now exported as
  `CompactBoundaryFaces3D` and `pack_boundary_faces_soa_3d`.
- Fixed-stride source-neighbor compact face candidates and CSR-facing offsets
  are now exported as `CompactFaceCandidates3D`,
  `CompactFaceCandidateCSR3D`, `compact_face_candidates_soa_3d`, and
  `compact_face_candidate_csr_soa_3d`.
- Dependency-free AREPO parameter/config parsing is now package-wired through
  `src/arepo_io_parameters.jl` and covered by unit tests.
- Source-owned fixed-stride face incidence rows and update signs are now
  exported as `SourceOwnedFaceCSR3D` and `source_owned_face_csr_soa_3d`.
- The compact-face tessellator export is still topology-only: it can emit a
  debug/prototype `ArepoMeshArrays3D` view, but scan-backed global compact face
  tables and the production hydro CSR rebuild are not done yet.
- Backend-resident reciprocal face-row pairing is now present for source-owned
  compact candidates, including matched row ids, canonical row ids, and owner
  flags.  This is the immediate input to the next scan-backed global dedup
  pass, not yet a production compact face table.
- A fixed-stride canonical face CSR/mesh prototype now lets reciprocal
  source-owned rows gather from one canonical flux row with opposite signs.
  This validates the hydro connectivity contract but is still padded and not
  the scan-backed global compact face table.
- A CPU-reference compact canonical face table now scans owner rows into an
  owner-only face table and rebuilds a normal two-sided hydro CSR.  The next
  tessellator step is replacing that host scan with KA-native prefix
  sum/scatter and real face geometry.
- Tessellator backend probe exercises the CPU KA primitive ladder; Metal is
  currently skipped because `Metal` is not available in the active
  `lib/PowerFoam` project.
- Snapshot IO now has a package-level typed in-memory gas snapshot schema,
  validation, path locator, derived volume/pressure/center fallbacks, and
  a real HDF5 write/read smoke when the local cached `HDF5.jl` dependency is
  available.
- PM gravity now has a package-level numeric periodic tiny-fixture preflight:
  raw image-sum direct diagnostics, a finite symmetric background-subtracted
  image oracle with zero net-force projection, and the repo-local
  `PoissonKernels` PM deposit/FFT/ghost/interp chain run through the
  PowerFoam project via a path source dependency.
- Noh2D now has a package-level executable bounded PowerFoam standard-problem
  rung at `N=24`, `t_final=0.2`, `HLL`, with conservation and radial-bin
  diagnostics.  It is labeled `calibration-PENDING`, not physics parity.
- Sound-wave 2D now has a package-level executable periodic smooth acoustic
  gate at `Nx=32`, `Ny=8`, `t_final=0.05`, with mass/energy conservation,
  L2, amplitude-ratio, and phase diagnostics.  It is also
  `calibration-PENDING`.
- Runtime moving-mesh smoke runs through exported PowerFoam APIs and preserves
  uniform flow to roundoff.
- PM/direct gravity convention is now explicitly force-only,
  background-subtracted periodic force; finite image sums remain diagnostic
  only.
- PM gravity numeric preflight now writes numeric fixture/direct rows, finite
  background-subtracted direct-oracle rows, and PM chain rows through the
  package-owned `run_arepo_pm_gravity_preflight`; the remaining blocker is
  certifying that finite oracle against the production periodic convention.
- Noh2D executable gate now advances a bounded 2-D Noh IC through existing
  PowerFoam hydro and writes metrics/radial-bin/log artifacts.
- Runtime scaffold now recognizes `ArepoProblemSpec(:noh2d)` and
  `ArepoProblemSpec(:soundwave2d)` and executes the package-owned bounded
  standard-problem paths instead of returning only a planning stub.
- Parameter parser smoke now exercises the package-exported parser and writes a
  timestamped artifact.

Latest fast evidence from this pass:

- `lib/PowerFoam/test/runtests.jl`: `692` passing, `1` broken optional PM
  backend check, `693` total, plus snapshot IO `30/30`, including
  source-owned CSR padding, reciprocal compact-face pairing, compact
  owner-only hydro CSR, HDF5 snapshot IO, numeric PM chain preflight,
  PM self-force control, runtime-executed Noh2D, runtime-executed sound-wave
  2D, runtime-executed Gresho 2D, split/direct snapshot preflight,
  snapshot-to-hydro payload conversion, compact canonical face CSR scan
  scaffolding, and sound-wave/Gresho helper tests.
- `examples/arepo_runtime_hydro_smoke.jl`: prebuilt 3-D uniform hydro smoke
  reports zero conserved and primitive drift.
- `examples/arepo_gravity_direct_smoke.jl`: two-body and three-body direct
  gravity checks pass with zero acceleration error and zero momentum residual.
- `examples/arepo_noh2d_proxy_gate.jl`: proxy readiness passes with required
  sources and archived/generated artifacts present.
- `examples/tessellator_backend_parity_probe.jl`: CPU probe passes with
  `faces=37`, `tetras=660`, `boundary_rows=2706`; Metal skipped.
- `examples/arepo_io_runtime_surface_smoke.jl`: exported IO surface audit now
  reports `7 / 14` planned types and `11 / 20` planned functions present, with
  parameter, snapshot, and runtime-feature source files available.
- `examples/arepo_runtime_moving_mesh_smoke.jl`: 27-cell moving-mesh smoke has
  zero conserved drift and roundoff primitive drift.
- `examples/arepo_pm_direct_convention_smoke.jl`: symmetric finite image-sum
  diagnostics have near-zero net-force residual but image-depth-dependent force
  magnitude, supporting the background-subtracted PM convention.
- `examples/arepo_parameter_parser_smoke.jl`: package-facing parser smoke
  passes and writes README/CSV artifacts.
- `examples/arepo_runtime_param_spec_smoke.jl`: exported param/config parser,
  normalized runtime view, inferred `ArepoProblemSpec`, and scaffold smoke all
  write a timestamped artifact.
- `examples/arepo_pm_gravity_gate_skeleton.jl`: numeric periodic preflight
  passes and writes `preflight_rows.csv`; PM mass sum is `4.0`, RHS sum is
  `0.0`, PM net force is roundoff-scale, the finite direct oracle reports
  roundoff net force after projection, PM-vs-direct-oracle diagnostic rows are
  emitted, and the one-particle PM self-force control is roundoff-scale.
- `examples/arepo_noh2d_gate.jl`: executable bounded Noh2D rung passes with
  `calibration-PENDING`, mass drift `1.97e-16`, energy drift `9.87e-16`,
  `rho_max=2.54`, and shock-radius proxy `0.25`.
- `examples/arepo_soundwave2d_gate.jl`: executable periodic sound-wave rung
  passes with `calibration-PENDING`, mass drift `0`, energy drift `0`,
  `rho L2=1.44e-5`, amplitude ratio `0.9797`, and phase error `7.05e-5`.
- `examples/arepo_gresho2d_gate.jl`: executable periodic Gresho vortex rung
  passes with `calibration-PENDING` and writes metrics/profile/log/report
  artifacts for the default `32 x 32`, `t=0.02`, HLL run.
- `examples/arepo_snapshot_io_smoke.jl`: typed snapshot runtime smoke passes
  in-memory validation, locator preflight, and a real HDF5 write/read
  round-trip.
- `examples/arepo_runtime_scaffold_smoke.jl`: package-level runtime smoke now
  executes the sound-wave 2D standard-problem dispatch and reports
  `calibration_pending` with final time `0.01`.
- `examples/arepo_rewrite_gate_matrix.jl --summary-only --observed-pass ...`:
  40 total gates, 35 runnable by file existence, 5 observed artifact-backed
  passes in the latest targeted run, 30 runnable-without-observed-pass rows,
  and 5 planned rows.
- `examples/arepo_hydro_problem_registry_smoke.jl`: KH2D, Noh3D, and
  turbulence rows have local runnable drivers; Noh2D and sound-wave 2D now
  have executable PowerFoam rungs; Gresho now has an executable calibration gate but still lacks an original-AREPO profile comparison.
- `examples/arepo_io_parameter_audit.jl`: package-level parameter parsing and
  snapshot runtime preflight now exist; restart compatibility and output policy
  remain missing runtime surfaces.

## Immediate Execution Targets

1. Convert fixed-stride `CompactFaceCandidates3D` into source-owned update CSR
   arrays, then replace the fixed-stride scaffold with scan-backed global
   compact face table and hydro CSR rebuild.
   - Source-owned fixed-stride CSR, reciprocal row pairing, padded canonical
     CSR/mesh emission, and CPU-reference compact owner-only CSR emission are
     done; next is KA-native compact scan and production geometry measures.
2. Add a first runtime-harness smoke example that constructs an
   `ArepoProblemSpec`, selects KA CPU, and calls an existing PowerFoam hydro
   path through a thin orchestrator.
   - Done for prebuilt static 3-D uniform hydro smoke; next is moving-mesh
     geometry adapter hookup.
3. Certify the finite PM gravity direct oracle against a production periodic
   zero-mode convention; the PM FFT chain and a zero-net-force finite image
   oracle are now loadable and executable from PowerFoam.
4. Calibrate the executable Noh2D, sound-wave 2D, and Gresho 2D rungs against
   original AREPO and analytic thresholds, then promote from
   `calibration-PENDING` to physics gates.
5. Extend snapshot IO from the tiny HDF5 gas round-trip to full AREPO
   IC/snapshot/restart compatibility and output policy.
6. Keep running full PowerFoam tests after each integrated slice.
