# Hydro Parity Work Breakdown

Purpose: define the remaining hydro work to turn the current `lib/PowerFoam`
prototype into an AREPO-shaped hydro path with executable parity gates. This
artifact is scoped to existing PowerFoam sources and examples, not a greenfield
design.

## Current Implemented Hydro Capabilities

- 2-D Euler finite-volume face-table hydro exists on AREPO-shaped mesh arrays:
  `lib/PowerFoam/src/hydro2d.jl`
  - conserved/primitive conversion
  - first-order ALE flux/update split
  - limited gradient reconstruction
  - predicted face states
  - moving-mesh rebuild/update path
  - supported solvers: `:hll`, `:llf`/`:rusanov`
- 3-D Euler finite-volume face-table hydro exists on AREPO-shaped mesh arrays:
  `lib/PowerFoam/src/hydro3d.jl`
  - Cartesian periodic and AREPO-exported Voronoi mesh ingestion
  - first-order face flux + CSR cell gather kernels
  - limited 3-D gradient reconstruction
  - AREPO-style predictor scaffold via `predict_face_states_3d!`
  - moving-face ALE fluxes
  - native moving-mesh step scaffold via `moving_mesh_step_3d!`
  - supported solvers: `:hll`, `:llf`
- AREPO-specific semantics already scaffolded in 3-D:
  - explicit geometric rows separated from update targets:
    `with_update_targets_3d`, `face_update_activity_3d`
  - AREPO-style hydro timestep/bin helpers:
    `arepo_hydro_dt_3d`, `arepo_timebin_3d`, `arepo_hydro_timebins_3d`,
    `arepo_active_cells_3d`, `active_face_table_3d`
  - mesh generator velocity reconstruction:
    `arepo_mesh_velocity_3d`
- Existing certified parity pieces already have executable gates:
  - initial state: `examples/arepo_initial_state_gate_3d.jl`
  - geometry conversion: `examples/arepo_geometry_gate_3d.jl`
  - gradients: `examples/arepo_gradient_parity_3d.jl`
  - predictor/flux trace parity: `examples/arepo_face_trace_gate_3d.jl`
  - replay of traced updates: `examples/arepo_trace_replay_gate_3d.jl`
  - scheduler/timebin parity: `examples/arepo_hierarchy_gate_3d.jl`
  - mesh velocity parity: `examples/arepo_mesh_velocity_gate_3d.jl`
  - native rebuild topology parity: `examples/arepo_native_rebuild_trace_gate_3d.jl`
- Existing problem coverage beyond turbulence:
  - 3-D Noh smoke/final-field diagnostic:
    `examples/arepo_noh3d_smoke_gate.jl`
  - 2-D KH AREPO reference export:
    `examples/arepo_kh2d_original_gate.jl`
  - 2-D KH PowerFoam compare gate:
    `examples/powerfoam_kh2d_compare_gate.jl`
  - 2-D Noh/Sedov proxy table + profiling path:
    `examples/arepo_noh_proxy/*`, `examples/arepo_sedov_proxy/*`
  - matrix/summary driver:
    `examples/arepo_standard_problem_matrix.jl`

## Missing AREPO Hydro Semantics

- Native pass-sequence ownership is not closed yet.
  - PowerFoam can replay AREPO traced passes, but the production
    `run_step!` analogue still does not natively reproduce the same sequence of
    pre-flux states, internal reorders, drift/rebuild boundaries, and pass-local
    update ownership.
- Native update-target production path is incomplete.
  - Diagnostic replay supports one-sided geometric rows and native update
    targets, but the production stepper still needs to promote that row/update
    logic without consuming AREPO trace metadata.
- Hierarchical timestepping is only scheduler-certified, not hydro-certified.
  - bin quantization and active lists are gated, but partial drift, partial
    rebuild, active-face selection, and partial conserved updates are not yet
    proven in the production hydro path.
- Native rebuild parity is only demonstrated on small traced 3-D passes.
  - the local periodic rebuild works as a contract gate, but repeated-drift
    topology equivalence at larger `N` and across multiple sync points is still
    open.
- 2-D hydro semantics are materially behind the 3-D parity path.
  - no executable 2-D final-field parity gate yet for Noh or Sedov
  - no AREPO-certified 2-D predictor/flux parity gate analogous to the 3-D
    face-trace path
  - KH has useful comparison gates, but not the full AREPO hydro-semantics
    closure used by the 3-D turbulence target
- Solver surface is still narrow relative to AREPO.
  - current hydro kernels only expose HLL/LLF
  - HLLC / exact / PPM-style reconstruction choices are not yet implemented as
    production PowerFoam hydro options
- Strong-shock durability remains unproven.
  - Noh/Sedov proxy tooling exists, but positivity, shell shape, radial shock
    metrics, and long-time stability are not yet certified in native PowerFoam
    moving-mesh runs

## Work Packages

### WP1. Native 3-D Pass Sequence Closure

- Goal: make `moving_mesh_step_3d!`/production stepping reproduce AREPO’s
  pass-local state sequence without consuming traced face states or trace-row
  ownership.
- Reuse:
  - `src/hydro3d.jl`
  - `examples/arepo_trace_replay_gate_3d.jl`
  - `examples/arepo_face_trace_gate_3d.jl`
  - `examples/arepo_one_step_gap_3d.jl`
  - `examples/arepo_preflux_smoke_gate_3d.jl`
- Tasks:
  - promote native pre-flux snapshot construction into the production step path
  - reproduce AREPO’s internal gas-cell reorder boundaries in native code
  - preserve old-vs-new volume semantics across drift, reconstruction, and
    conservative update
  - make native row generation, native face velocity, native dt source, and
    native update targets the default production path rather than replay-only
- Acceptance:
  - `arepo_one_step_gap_3d.jl` reaches roundoff-level final conserved gaps on
    CPU Float64 at `N4`, then `N8`
  - no dependence on trace-supplied face states/fluxes/update rows in the final
    passing configuration

### WP2. Native 3-D Update-Target / Active-Face Productionization

- Goal: carry AREPO’s one-sided geometric rows, update ownership, and active
  face semantics into the production hydro kernels.
- Reuse:
  - `src/hydro3d.jl`: `with_update_targets_3d`, `face_update_activity_3d`,
    `active_face_table_3d`,
    `finite_volume_reconstructed_hierarchy_step_3d!`
  - `examples/arepo_face_trace_gate_3d.jl`
  - `examples/arepo_trace_replay_gate_3d.jl`
- Tasks:
  - build production active-face tables from native row ownership
  - verify sign conventions for `c1 == 0` / `c2 > 0` and no-update rows
  - ensure predictor, face-flux, and cell-gather kernels all consume the same
    ownership model
- Acceptance:
  - per-pass row counts, one-sided-row counts, and update-target mismatches are
    zero relative to AREPO on `N4`, then `N8`
  - replay and production paths agree on final conserved updates for the same
    native rows

### WP3. Hierarchical Timestepping Coupled To Hydro

- Goal: move from scheduler parity to active-hydro parity.
- Reuse:
  - `src/hydro3d.jl`: timebin helpers and hierarchy step kernel
  - `examples/arepo_hierarchy_gate_3d.jl`
  - `examples/arepo_standard_problem_matrix.jl`
- Tasks:
  - couple partial drift, partial rebuild, and partial hydro update to the
    certified timebin logic
  - compare active cells, active faces, and final conserved fields at each sync
    point in the multirung fixture
  - verify gravity-limited effective bins remain aligned once hydro updates are
    no longer scheduler-only
- Acceptance:
  - `examples/arepo_hierarchy_gate_3d.jl` passes not only bin/list checks but
    also active-face and final-field checks on the `multirung` fixture
  - no active-mask or next-sync mismatches through at least 3 native AREPO
    steps at `N8`

### WP4. Native Rebuild Topology Extension

- Goal: prove the local periodic native rebuild is robust beyond the traced
  first-step small-`N` contract gate.
- Reuse:
  - `src/tessellation3d.jl`
  - `src/tessellation3d_semantics.jl`
  - `examples/arepo_native_rebuild_trace_gate_3d.jl`
  - `examples/arepo_tessellator_rebuild_gate_matrix.jl`
  - `examples/native_rebuild_gate_3d.jl`
- Tasks:
  - extend parity checks to `N8`/`N12` and repeated sync points
  - keep duplicate/periodic-image handling and tiny-face filtering matched to
    the current AREPO bridge contract
  - compare face pairs, volumes, areas, normals, centers, and update ownership
    after repeated drift
- Acceptance:
  - rebuild gate matrix passes at multiple `N` and repeat counts with explicit
    zero-mismatch topology summaries
  - no geometry-only regressions reintroduced into the certified predictor
    replay path

### WP5. 3-D Solver Surface After Semantics Closure

- Goal: broaden solver/reconstruction choices only after the native hydro path
  matches AREPO for HLL/LLF.
- Reuse:
  - `examples/arepo_solver_matrix_3d.jl`
  - `examples/native_moving_solver_matrix_3d.jl`
  - `examples/arepo_geometry_gate_3d.jl`
- Tasks:
  - keep HLL as the first production target
  - retain LLF as a low-complexity cross-check
  - add HLLC only after native pass sequence and hierarchy close
  - defer exact/PPM-style solver claims until the simpler solver surface is
    stable
- Acceptance:
  - HLL/LLF continue to pass all existing 3-D parity gates after production
    stepper integration
  - any new solver is added with a dedicated geometry/predictor/final-field
    matrix row rather than inferred from HLL success

### WP6. 2-D Hydro Parity Upgrade

- Goal: turn the current 2-D proxies into executable AREPO parity gates.
- Reuse:
  - `src/hydro2d.jl`
  - `examples/arepo_kh2d_original_gate.jl`
  - `examples/powerfoam_kh2d_compare_gate.jl`
  - `examples/arepo_noh_proxy/*`
  - `examples/arepo_sedov_proxy/*`
  - `examples/moving_mesh_contact.jl`
- Tasks:
  - promote Noh proxy into a 2-D executable final-field/profile parity gate
  - promote Sedov proxy into a 2-D executable radial shock/profile gate
  - add a 2-D predictor/flux parity gate for at least one smooth or KH-style
    case before claiming 2-D AREPO semantics
  - keep KH as the first 2-D moving-mesh reconstructed comparison gate
- Acceptance:
  - Noh: radial density profile, shock radius, shell width, and positivity pass
    explicit tolerances
  - Sedov: radial density/pressure/velocity profile and blast symmetry pass
    explicit tolerances
  - KH: final-field and integral metrics stay within declared tolerances against
    the AREPO reference export

### WP7. Strong-Shock Durability And Physical-Time Validation

- Goal: validate that the native hydro path survives beyond one-step parity.
- Reuse:
  - `examples/arepo_noh3d_smoke_gate.jl`
  - `examples/arepo_noh_proxy/profile_snapshots.py`
  - `examples/arepo_sedov_proxy/profile_snapshots.py`
  - `examples/arepo_standard_problem_matrix.jl`
- Tasks:
  - after WP1-WP4 close, run native PowerFoam to fixed physical times
  - compare AREPO vs PowerFoam on strong shocks and smooth turbulence
  - treat negativity or shell breakup as blocking, not cosmetic
- Acceptance:
  - no negative pressure/density on the certified production path for the test
    window
  - turbulence and strong-shock metrics remain within stated tolerances at
    `N12`/`N16` scale gates before larger performance claims resume

## Test Problems To Carry In The Parity Suite

- 3-D decaying subsonic turbulence
  - primary parity target
  - gates: initial state, geometry, gradients, predictor/flux trace, replay,
    hierarchy, solver matrix
- 3-D multirung turbulence fixture
  - active-cell/hierarchy stress case
  - gate: `examples/arepo_hierarchy_gate_3d.jl`
- 3-D Noh
  - strong-shock/positivity stress
  - gate base: `examples/arepo_noh3d_smoke_gate.jl`
- 2-D Noh
  - cylindrical-shock profile/regression
  - current reusable path: `examples/arepo_noh_proxy/*`
- 2-D Sedov
  - blast symmetry/energy/shock profile
  - current reusable path: `examples/arepo_sedov_proxy/*`
- 2-D Kelvin-Helmholtz
  - contact handling + moving-mesh reconstructed comparison
  - gates: `examples/arepo_kh2d_original_gate.jl`,
    `examples/powerfoam_kh2d_compare_gate.jl`
- 2-D contact/blob advection
  - contact preservation and moving-vs-static diffusion
  - current reference path noted by `examples/arepo_standard_problem_matrix.jl`
- 2-D Gresho / Yee / shearing sinusoid
  - smooth-vortex and low-dissipation follow-on suite
  - currently planned, not yet executable

## Acceptance Metrics

- Component parity metrics
  - primitive max-abs differences
  - conserved max-abs differences
  - gradient component max-abs differences
  - face-state max-abs differences
  - flux-times-area max-abs differences
  - update-target mismatch counts
  - row-count / missing-face / extra-face counts
- Geometry metrics
  - face-pair set equality
  - cell volume equality or declared tolerance
  - face area / normal / center differences
  - CSR ownership count differences
- Scheduler/hierarchy metrics
  - raw hydro bin mismatch count
  - effective bin mismatch count
  - active-mask mismatch count
  - active-list mismatch count
  - next-sync-step mismatch
- Physical metrics
  - mass, momentum, energy drift
  - `vrms`, Mach rms, density rms
  - `rho_min`, `rho_max`, `pmin`
  - strong-shock radial profile errors
  - shell width / shock radius / symmetry error
  - positivity failures treated as hard failures

## Existing Files And Gates To Reuse

- Core source surfaces
  - `lib/PowerFoam/src/hydro2d.jl`
  - `lib/PowerFoam/src/hydro3d.jl`
  - `lib/PowerFoam/src/tessellation3d.jl`
  - `lib/PowerFoam/src/tessellation3d_semantics.jl`
- Live parity plans/audits
  - `lib/PowerFoam/arepo_physics_parity_plan.md`
  - `lib/PowerFoam/arepo_physics_parity_audit.md`
- 3-D executable gates
  - `lib/PowerFoam/examples/arepo_initial_state_gate_3d.jl`
  - `lib/PowerFoam/examples/arepo_geometry_gate_3d.jl`
  - `lib/PowerFoam/examples/arepo_gradient_parity_3d.jl`
  - `lib/PowerFoam/examples/arepo_face_trace_gate_3d.jl`
  - `lib/PowerFoam/examples/arepo_trace_replay_gate_3d.jl`
  - `lib/PowerFoam/examples/arepo_one_step_gap_3d.jl`
  - `lib/PowerFoam/examples/arepo_preflux_smoke_gate_3d.jl`
  - `lib/PowerFoam/examples/arepo_mesh_velocity_gate_3d.jl`
  - `lib/PowerFoam/examples/arepo_hierarchy_gate_3d.jl`
  - `lib/PowerFoam/examples/arepo_native_rebuild_trace_gate_3d.jl`
  - `lib/PowerFoam/examples/arepo_tessellator_rebuild_gate_matrix.jl`
  - `lib/PowerFoam/examples/arepo_solver_matrix_3d.jl`
  - `lib/PowerFoam/examples/native_moving_solver_matrix_3d.jl`
- 2-D executable or proxy gates
  - `lib/PowerFoam/examples/arepo_kh2d_original_gate.jl`
  - `lib/PowerFoam/examples/powerfoam_kh2d_compare_gate.jl`
  - `lib/PowerFoam/examples/arepo_noh_proxy/generate_tables.jl`
  - `lib/PowerFoam/examples/arepo_noh_proxy/write_arepo_cases.py`
  - `lib/PowerFoam/examples/arepo_noh_proxy/profile_snapshots.py`
  - `lib/PowerFoam/examples/arepo_sedov_proxy/generate_tables.jl`
  - `lib/PowerFoam/examples/arepo_sedov_proxy/write_arepo_cases.py`
  - `lib/PowerFoam/examples/arepo_sedov_proxy/profile_snapshots.py`
  - `lib/PowerFoam/examples/moving_mesh_contact.jl`
- Suite summary driver
  - `lib/PowerFoam/examples/arepo_standard_problem_matrix.jl`

## Recommended Order

- First close native 3-D pass sequence and update-target production parity.
- Then couple hierarchy to the real hydro/update path.
- Then extend native rebuild parity across larger `N` and repeated sync points.
- Only after that broaden solver surface beyond HLL/LLF.
- In parallel, upgrade 2-D Noh/Sedov from proxy tooling to executable parity
  gates, using KH as the 2-D moving-mesh comparison scaffold.
- Keep physical-time turbulence and strong-shock claims blocked on positivity
  and final-field parity, not on one-step replay success alone.
