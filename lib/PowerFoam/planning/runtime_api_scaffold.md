# Runtime API Scaffold

## Purpose

This note describes the narrow `arepo_runtime_scaffold.jl` shim that will sit
between a future `Arepo.jl` user-facing API and the existing PowerFoam rewrite
pieces.  The current file is intentionally lightweight:

- pure Julia only
- no new dependencies
- no exports yet
- no edits to existing `PowerFoam.jl`, tests, or examples

The goal is to lock in the top-level runtime shapes now so other workstreams can
target a stable include-only interface while hydro, tessellation, and gravity
continue evolving independently.

## What The Scaffold Defines

- `ArepoRunOptions`
  - minimal run-loop controls such as time span, CFL, step limit, and a small
    policy flag for unsupported features
- `ArepoProblemSpec`
  - immutable problem description carrying AREPO-like semantic inputs:
    dimensionality, domain, periodicity, gas/particle counts, physics flags,
    and IC metadata
- `ArepoRuntimeState3D`
  - mutable top-level state object that will eventually hold live mesh, hydro,
    gravity, diagnostics, and output substates
- `ArepoHydroSmokeAssessment`
  - small advisory record describing whether a spec matches the narrow shape of
    a future pure-KA hydro smoke path
- constructor helpers
  - `arepo_problem_spec`
  - `arepo_runtime_state_3d`
  - `classify_ka_hydro_smoke`
- `arepo_run_scaffold(...)`
  - a stub entrypoint that returns a populated state plus explicit diagnostics
    about what is still missing, plus a smoke-path classification in the
    returned payload

## Why A 3-D Runtime Envelope

The current PowerFoam rewrite already has significant 3-D surface area:

- unstructured hydro kernels in `hydro3d.jl`
- 3-D tessellation semantics and SoA builders
- active-cell and hierarchy-facing hydro stepping hooks

Even when a problem is logically 1-D or 2-D, the future runtime can normalize
problem metadata into a 3-D envelope and then choose a lower-dimensional path
inside the orchestrator.  That keeps the runtime API aligned with AREPO’s
general moving-mesh semantics instead of splitting the driver surface too early.

## Planned Integration Path

### 1. Runtime Front Door

The eventual orchestrator should be able to do something close to:

```julia
spec = arepo_problem_spec(:kh2d; dimensionality=2, physics=(hydro=true, tessellation=true, gravity=false))
state = arepo_run(spec; backend=:ka, device=KA.CPU(), options=ArepoRunOptions(...))
```

At that stage, the current scaffold file should still remain the place where the
data shapes live, while the orchestrator owns exports and backend dispatch.

### 1a. Hydro Smoke Classification

Before a full orchestrator exists, callers can already ask whether a spec is a
reasonable candidate for a lightweight hydro-only smoke path:

```julia
spec = arepo_problem_spec(:kh2d; dimensionality=2, gas_cell_count=4096,
                          physics=(hydro=true, tessellation=true, gravity=false))
smoke = classify_ka_hydro_smoke(spec; backend=:ka)
```

The classifier is intentionally modest. It currently checks only top-level
runtime semantics:

- `backend == :ka`
- hydro enabled
- dimensionality in `2:3`
- positive gas cell count
- no gravity / particle coupling
- whether a tessellation adapter would still be required

This is enough for the main rewrite thread to sort problems into:

- good candidates for a future pure-KA hydro smoke harness
- cases that should stay in the broader bridge/native orchestration path
- cases that need mixed-physics runtime support before any smoke harness makes
  sense

The classifier is advisory only; it does not claim that any tessellation or
hydro kernel already runs.

### 2. Tessellation Hookup

`ArepoProblemSpec` already carries the semantic inputs needed by a tessellation
stage:

- dimensionality
- domain bounds
- periodic axes
- gas cell counts
- IC metadata that can later point to generator positions/weights

The next layer should map those fields onto PowerFoam tessellation builders such
as the 3-D Voronoi/Delaunay reference and eventual KA-resident rebuild path.  A
future runtime payload will likely replace `mesh = nothing` with a mesh bundle
containing:

- generator coordinates
- weights if needed
- face-table / CSR arrays
- rebuild counters and predicate diagnostics

### 3. Hydro Hookup

The hydro phase should eventually attach the existing PowerFoam ingredients:

- primitive/conserved state arrays
- gradient work buffers
- face prediction buffers
- ALE flux work buffers
- active-cell or hierarchy metadata

`arepo_run_scaffold` currently records hydro as unsupported on purpose.  Later
versions should turn that diagnostic into a real sequence:

1. construct or rebuild mesh
2. derive face/cell geometry
3. compute gradients and predicted face states
4. evaluate ALE fluxes
5. update conserved variables
6. advance runtime time/step counters

## 4. Gravity And Cosmology Hookup

`ArepoProblemSpec` also carries `particle_count` and generic `physics` flags so
the scaffold can already represent mixed gas/particle runs.  Future integration
should route those semantics into:

- direct-force or PM gravity for small controlled gates first
- comoving/cosmology metadata in `metadata` or a richer physics config
- kick-drift-kick style scheduling that matches AREPO problem semantics

The important point is that the runtime object should speak in AREPO concepts
even when the underlying implementation is pure Julia and PowerFoam-backed.

## Matching AREPO Problem Semantics

This scaffold is not trying to mirror AREPO’s internal C structs one-for-one.
Instead, it preserves the user-visible semantics that matter for a Julia runtime:

- named problem specifications
- periodic vs nonperiodic domains
- gas and collisionless-particle populations
- backend/device selection
- run options and timestep intent
- diagnostics that explain what the runtime did or could not do

That should make it possible to support both:

- standard-problem registry calls
- future parameter-file or snapshot-driven setups

without forcing the kernel files themselves to know about high-level runtime
policy.

## Why Keep It Include-Only For Now

Other branches are actively touching `PowerFoam.jl`, hydro sources, tests, and
examples.  Keeping this scaffold in its own file avoids merge pressure while
still giving the main orchestrator something concrete to include later.

This also preserves a clean separation of responsibilities:

- scaffold file: runtime data shapes and stub API
- orchestrator: exports, backend wiring, package boundary decisions
- hydro/tessellation/gravity files: implementation detail and parity work

## Immediate Next Steps

1. Include the scaffold from the future orchestrator once the export surface is
   ready.
2. Use `classify_ka_hydro_smoke` to decide which standard problems should get a
   direct include-only smoke harness first.
3. Replace `mesh = nothing` with a typed tessellation payload wrapper.
4. Replace hydro/gravity diagnostics with real substate constructors.
5. Add a bridge-compatible parameter/problem adapter without pulling that logic
   into the kernel files.

## Direct-Include Smoke Example

`lib/PowerFoam/examples/arepo_runtime_scaffold_smoke.jl` is a tiny direct
include harness that constructs:

- a KH2D-like periodic hydro spec
- a Noh3D-like strong-shock hydro spec

and prints concise smoke classification plus scaffold diagnostics. This keeps
the slice runnable without waiting for package exports or test integration.

`lib/PowerFoam/examples/arepo_runtime_hydro_smoke.jl` is the package-level
follow-on slice. It uses `using PowerFoam`, builds an `ArepoProblemSpec` for a
prebuilt 3-D Cartesian hydro smoke with `tessellation=false`, verifies that
`classify_ka_hydro_smoke` marks the case as eligible, then runs a one-step
uniform-flow `cartesian_periodic_mesh_arrays_3d` plus `euler_state_3d`
`finite_volume_step_3d!` update and writes a tiny artifact under
`lib/PowerFoam/examples/out/arepo_runtime_hydro_smoke/<timestamp>/`.

## Runtime Param-Spec Smoke Slice

`lib/PowerFoam/examples/arepo_runtime_param_spec_smoke.jl` is the matching
lightweight adoption smoke for the exported runtime parser surface.

It stays entirely on `using PowerFoam` and exercises the API path that a future
parameter-driven runtime entrypoint will need:

- parse representative `param.txt` text with `parse_arepo_param_text`
- parse representative `Config.sh` text with `parse_arepo_config_text`
- normalize and validate the runtime slice with
  `normalize_arepo_parameters` and `validate_arepo_parameters`
- construct an `ArepoProblemSpec` from normalized fields where the mapping is
  already explicit
- classify the resulting problem with `classify_ka_hydro_smoke`
- record a tiny README/CSV artifact under
  `lib/PowerFoam/examples/out/arepo_runtime_param_spec_smoke/<timestamp>/`

The value of this slice is narrow but useful: it proves that the current
exported parser and runtime-scaffold APIs can already support a realistic
"parameter text to problem spec" smoke without reaching into non-exported
helpers or editing `src`/tests.

## Runtime Moving-Mesh Smoke Slice

`lib/PowerFoam/examples/arepo_runtime_moving_mesh_smoke.jl` is the matching
ALE/moving-mesh smoke that stays within the current exported `PowerFoam` API
surface and does not edit core source.

The slice is intentionally tiny:

- builds a `3 x 3 x 3` periodic Voronoi mesh from Cartesian generator points
- uses `local_periodic_voronoi_mesh_arrays_3d`
- prescribes a uniform mesh velocity through exported arguments only
- advances one `moving_mesh_step_3d!` update with `riemann=:hll`
- checks that generator advection matches `advect_generators_3d`
- checks that conserved totals and primitive variables stay unchanged for a
  comoving uniform-flow state

The practical runtime question here is narrow: can a user represent a tiny
moving-mesh hydro smoke with exported APIs alone, before any full orchestrator
exists?

As of the current scaffold, the answer is yes for a controlled periodic ALE
smoke because the needed surface is already exported:

- `local_periodic_voronoi_mesh_arrays_3d`
- `euler_state_3d`
- `moving_mesh_step_3d!`
- `advect_generators_3d`
- `total_conserved_3d`
- `conserved_to_primitive_3d`

If that export surface regresses, the example is written to stop early and
print the missing exported call names instead of silently reaching into
non-exported implementation details.
