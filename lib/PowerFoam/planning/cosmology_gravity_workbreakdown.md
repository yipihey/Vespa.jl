# Cosmology And Gravity Work Breakdown

Date: 2026-06-15

This note scopes the cosmology and self-gravity slice of the KA-based
`Arepo.jl` rewrite.  It is written to fit the current repo reality:

- `lib/PoissonKernels` already has the KA backbone for PM gravity:
  periodic CIC deposit, root FFT Poisson solve, multigrid/Dirichlet solves,
  acceleration differencing, and particle kick/drift.
- `lib/MultiCode` already has useful cosmology fixtures and IO patterns:
  Zel'dovich growth gates, MUSIC/grafic ingestion, Gadget-format IC writing,
  resident particle push logic, and code-neutral report surfaces.
- `lib/PowerFoam` already uses staged parity plans and durable example artifacts
  for AREPO hydrodynamics and standard-problem tracking.

The near-term goal is not full hydro+cosmology parity.  It is a gravity-first,
cosmology-capable MVP that can run the same class of early AREPO problems with
clear acceptance gates and without baking in a future dead-end.

## Acceptance Policy

- Certify gravity components before claiming end-to-end cosmology agreement.
- Use PM-only periodic cosmology as the first production target.
- Treat tree and direct gravity as separate backends with different jobs:
  direct for tiny-N oracle checks, tree for later AREPO-like production parity.
- Reuse existing KA kernels and cross-code fixtures before inventing new
  infrastructure.
- Keep all parity claims same-precision and problem-specific.
- Record every gate as a durable artifact, following the current
  `lib/PowerFoam/examples/out/...` pattern.

## Existing Surfaces To Reuse

### PoissonKernels

- `src/deposit.jl`: periodic CIC deposit with optional drift term and Enzo-style
  `shift=-0.5` registration.
- `src/fft_poisson.jl` and `src/gpu_fft.jl`: root periodic PM solve.
- `src/masked_cg.jl`, `src/vcycle.jl`, `src/mg_batched.jl`: irregular and
  Dirichlet solve infrastructure that can later support tree-PM corrections or
  isolated-region subproblems.
- `src/comp_accel.jl`: acceleration from potential.
- `src/particle_push.jl`: comoving-aware interpolation, kick, and drift kernels.
- `examples/dm_only_gravity.jl`, `examples/enzo_gravity_steps.jl`,
  `examples/cpu_gpu_parity.jl`: good templates for gravity-only certification.

### MultiCode

- `src/zeldovich.jl`: exact pre-shell-crossing cosmology gate and grafic writer.
- `ext/MultiCodeMusicExt.jl`: MUSIC ingestion cross-check pattern.
- `deps/gadget_ic.jl`: minimal Gadget-format IC writer suitable for AREPO IC
  staging.
- `src/enzo_resident.jl`: explicit comoving coefficient handling for a
  production particle push loop.
- `src/gravity_slot.jl`: host-owned mesh plus guest-owned Poisson solve pattern.

### PowerFoam / AREPO planning precedent

- `arepo_physics_parity_plan.md`: stage work by certified pieces, not by broad
  promises.
- `arepo_physics_parity_audit.md`: keep blocking gaps explicit.
- `examples/arepo_standard_problem_matrix.jl`: use original AREPO example names
  and track current gate / next gate / status.

## Scope Split

### Gravity-only cosmology MVP

The MVP should support:

- periodic box;
- collisionless DM particles only at first;
- comoving expansion;
- PM gravity using KA deposit + FFT + accel + particle push;
- Gadget-format IC ingest and snapshot export sufficient to compare against
  stock AREPO cosmology examples;
- single global timestep first, with multirung time bins staged later.

This is the smallest slice that exercises the cosmology-specific gravity path
without waiting for mesh hydrodynamics, moving Voronoi gravity coupling, or
full TreePM.

### Later extensions

- TreePM or tree-only force splitting.
- Gas self-gravity on the moving mesh.
- hierarchical/multirung active timesteps;
- AREPO-style long-range/short-range splitting and opening criteria;
- restart/snapshot compatibility for multi-output cosmology runs.

## Gravity Architecture Options

### 1. PM gravity

Role:

- first production backend;
- default MVP path;
- periodic cosmology workhorse.

Why it fits the repo now:

- `PoissonKernels` already supplies deposit, FFT solve, accel, and push;
- `MultiCode` already has exact cosmology fixtures that naturally target PM;
- PM is enough for gravity-only Zel'dovich and small-box cosmology smoke tests.

Required MVP pieces:

- particle container in AREPO/PowerFoam-native layout;
- periodic CIC deposit from particle positions/masses;
- density mean subtraction / zero-mode handling;
- FFT solve in comoving variables;
- cell-centered acceleration field;
- particle interpolation + kick/drift using comoving coefficients;
- snapshot writer for particles and grid diagnostics.

Primary parity targets:

- `dm_only`-style periodic cosmology runs;
- Zel'dovich plane wave before shell crossing;
- small MUSIC/Gadget IC cross-checks.

### 2. Direct gravity

Role:

- tiny-N oracle only;
- not a production cosmology backend.

Use it for:

- 8-1024 particle unit tests;
- force-law, softening, periodic-image, and comoving-update debugging;
- acceptance tests for PM and future tree implementations.

Required capabilities:

- exact or high-accuracy pairwise force with the same softening convention as
  the production backend;
- optional periodic-image support for miniature boxes;
- deterministic reference outputs for CI-scale fixtures.

Acceptance use:

- direct-vs-PM force comparison on frozen particle distributions;
- direct-vs-tree force comparison once tree exists.

### 3. Tree gravity

Role:

- later production backend;
- eventual AREPO-like short-range or pure-tree path.

Why it should not be MVP:

- no existing KA tree infrastructure is visible in this repo;
- PM already covers the gravity-only cosmology slice;
- tree implementation details can dominate schedule before the cosmology loop
  itself is certified.

Stage it after PM MVP, with one of two directions:

- tree-only backend for non-periodic/small periodic tests;
- TreePM split where PM handles long range and tree handles short range.

Acceptance use:

- force accuracy against direct gravity at tiny N;
- force-spectrum and trajectory agreement against PM on scales where PM should
  dominate;
- later comparison against original AREPO TreePM runs on matched ICs.

## Comoving Variables And Units

The rewrite should make these variables explicit and first-class:

- scale factor `a`;
- time derivative `adot` or equivalent expansion-rate term;
- comoving positions `x`;
- peculiar velocities / AREPO-compatible momentum variable;
- comoving density source for Poisson;
- box size in code units and physical units;
- gravitational softening in comoving and, later if needed, physical capped form.

Minimum design rule:

- every gravity update path must state which quantity lives in comoving units,
  which in peculiar units, and where factors of `a`, `1/a`, `a^2`, or `adot/a`
  enter.

Concrete repo guidance:

- `lib/MultiCode/src/enzo_resident.jl` is the best current local reference for
  an explicit comoving kick/drift update.
- `lib/PoissonKernels/src/particle_push.jl` already encodes the split between
  interpolation half-drift, main drift, and semi-implicit kick.  The AREPO
  rewrite should reuse this clarity even if the exact coefficients differ.

Required MVP documentation/gates:

- one note or report per gate that lists the exact update equations used;
- one unit test or artifact that shows the comoving coefficients at each
  substep for a known cosmology step;
- one cross-check that a frozen homogeneous particle load produces zero net
  peculiar acceleration after mean subtraction.

## Time Integration Work Breakdown

### Phase T1: single global timestep

Goal:

- one synchronized cosmology step loop that advances all particles together.

Required pieces:

- compute gravitational source at step start;
- solve PM potential/acceleration;
- apply cosmology-aware kick/drift/kick;
- update current time and scale factor;
- write snapshot/statistics on synchronized steps.

Acceptance:

- exact trajectory gate passes for Zel'dovich before shell crossing;
- repeated steps preserve periodic wrap and finite energies;
- same-precision CPU backend results are deterministic.

### Phase T2: AREPO-style timebin scaffolding

Goal:

- add timestep quantization and active-bin bookkeeping without yet changing the
  gravity backend.

Required pieces:

- timebin assignment data model;
- active-particle masks/lists;
- synchronization-point accounting;
- snapshot-time landing logic.

Acceptance:

- scheduler fixture reproduces expected active lists and sync times on a small
  controlled case;
- global-step mode remains available as a debug path.

### Phase T3: partial-step gravity coupling

Goal:

- only active particles drift/kick at partial steps while preserving consistent
  long-range force semantics.

Required pieces:

- policy for when the PM mesh is rebuilt and solved on partial steps;
- inactive-particle prediction policy for deposit/interpolation;
- force-age bookkeeping in snapshot/restart state.

Acceptance:

- multirung tiny-box fixture agrees with a reference run that uses much smaller
  global steps;
- no silent regression of the global-step path.

## IC And Snapshot IO Work Breakdown

### IC ingestion

Priority order:

1. Gadget-format particle ICs for direct AREPO interoperability.
2. Julia-generated analytic ICs for unit and exact gates.
3. MUSIC/grafic-derived ICs through the existing `MultiCode` patterns.

Concrete reuse:

- use `lib/MultiCode/deps/gadget_ic.jl` as the format contract for the first
  writer/reader surface;
- use `lib/MultiCode/src/zeldovich.jl` and `ext/MultiCodeMusicExt.jl` patterns
  for exact and MUSIC-derived fixtures.

MVP IC requirements:

- DM particle positions, velocities, IDs, masses;
- cosmology metadata: `a_init`, `OmegaM`, `OmegaLambda`, `HubbleParam`,
  `BoxSize`;
- deterministic ordering and round-trip diagnostics.

Acceptance:

- write/read round trip preserves particle count, IDs, masses, and coordinate
  ranges;
- AREPO stock cosmology example can boot from the produced ICs;
- an analytic Zel'dovich IC produced in Julia yields the expected initial-mode
  amplitude after readback.

### Snapshot output

MVP requirements:

- particle positions, velocities, IDs, masses at each output;
- current `a`, time, timestep, and box metadata;
- optional PM diagnostics: deposited density slices, potential norms, force
  norms, power-spectrum summary.

Later additions:

- timebin state;
- force split diagnostics;
- gas cell state once self-gravitating hydro is added.

Acceptance:

- snapshot cadence can land on requested output times;
- restart from snapshot reproduces the next-step state within same-precision
  tolerance;
- output is sufficient to drive external comparison scripts and artifact pages.

## Required Physics Parity Gates

The rewrite should land these gates in order.

### Gate G1: kernel-level gravity parity

Purpose:

- certify the KA building blocks independently of AREPO.

Checks:

- CPU Float64 vs CPU Float32 sanity;
- CPU Float32 vs Metal Float32 parity where relevant;
- direct small-N reference vs deposit/force/interp outputs on frozen particle
  sets;
- homogeneous periodic load gives zero net force.

Suggested artifact shape:

- follow `lib/PoissonKernels/examples/cpu_gpu_parity.jl` and
  `dm_only_gravity.jl`.

### Gate G2: PM source and force-step certification

Purpose:

- prove the PM pipeline is self-consistent on real cosmology-like inputs.

Checks:

- CIC deposit mass conservation;
- zero-mode subtraction;
- force symmetry on mirrored particle sets;
- potential and acceleration norms stable across repeated solves;
- trajectory update on one step matches the documented comoving formulas.

### Gate G3: exact cosmology growth gate

Purpose:

- certify the cosmology loop before shell crossing.

Problem:

- Zel'dovich plane wave, matching the existing `MultiCode` exact fixture.

Checks:

- growth amplitude matches analytic expectation;
- residual shape error stays bounded;
- y/z immobility in the 1-D plane-wave setup;
- cross-backend results remain within same-precision tolerance.

### Gate G4: IC ingest parity gate

Purpose:

- prove the rewrite starts from the same particle set as the external producer.

Problems:

- Julia-written Gadget IC;
- MUSIC/grafic-derived realization;
- optionally a small stock AREPO cosmology IC.

Checks:

- particle positions/velocities/IDs/masses survive ingest;
- CIC density field correlation against the source realization;
- initial power spectrum agrees within tolerance.

### Gate G5: gravity-only cosmology evolution gate

Purpose:

- certify the MVP end-to-end.

Problems:

- `dm_only`-style periodic cosmology box;
- exact Zel'dovich evolution to a target `a/a_init`;
- optionally a tiny Santa Barbara DM-only subset later.

Checks:

- trajectory/growth agreement;
- energy and momentum diagnostics remain finite and explainable;
- output-time landing works;
- snapshot restart reproduces the next synchronized state.

### Gate G6: cross-code parity gate

Purpose:

- compare against original AREPO, not just internal consistency.

Problems:

- stock AREPO cosmology example with PM-compatible settings first;
- later TreePM example once tree exists.

Checks:

- same ICs, same cosmology metadata, same output epochs;
- particle displacement statistics, power spectra, and selected trajectory
  traces agree to problem-appropriate tolerance;
- differences are attributed to backend choice, timestep hierarchy, or force
  model, not to undocumented IO/unit mismatches.

## Staged Acceptance Criteria

### Stage 0: planning-complete

- this work breakdown exists;
- original AREPO example targets are named;
- gravity backends are split into PM, tree, and direct roles.

### Stage 1: gravity kernel base certified

- PM deposit/solve/accel/push surfaces run through durable KA parity artifacts;
- direct tiny-N oracle exists for frozen-particle checks;
- comoving update equations are documented in one place.

### Stage 2: gravity-only cosmology MVP

- single-global-step PM cosmology loop runs;
- Gadget IC ingest and snapshot output work;
- Zel'dovich exact gate passes;
- homogeneous and tiny-random-box sanity checks pass;
- one external AREPO-compatible DM-only run can be initialized and compared.

This is the first "usable" milestone.

### Stage 3: restart and output maturity

- snapshot restart reproduces the next step within tolerance;
- output-time landing is stable;
- durable comparison reports are emitted automatically.

### Stage 4: timebin-capable cosmology

- timebin scaffolding exists;
- active-list/sync-point fixtures pass;
- global-step debug path still passes Stage 2 gates.

### Stage 5: partial-step production parity

- partial-step PM policy is implemented;
- multirung gravity-only fixture passes;
- cross-code comparisons stay within documented tolerances.

### Stage 6: tree or TreePM extension

- direct-vs-tree tiny-N gate passes;
- PM-vs-TreePM large-scale agreement is documented;
- original AREPO TreePM examples become valid parity targets.

## Suggested Original AREPO Example Targets

Use original names where possible so reports stay recognizable:

- `cosmo_box_gravity_only` or equivalent DM-only periodic cosmology example for
  PM MVP bring-up;
- `Zeldovich`-style plane-wave fixture as the exact gate, even if generated
  locally rather than shipped by AREPO;
- later `SantaBarbara`-class cosmology cases only after PM MVP is stable;
- hydrodynamic examples such as `bauer_springel_turbulence_3d`, `noh_3d`,
  `kh_2d_lecoanet`, and `gresho_2d` remain useful precedent for artifact style,
  but they should not block the gravity-only cosmology MVP.

## Recommended Implementation Order

1. Build the PM-only gravity driver around existing `PoissonKernels` surfaces.
2. Add direct tiny-N oracle checks before adding tree work.
3. Reuse `MultiCode` Zel'dovich and Gadget/MUSIC patterns for IC and exact
   gates instead of inventing new fixtures.
4. Land a gravity-only synchronized cosmology loop with snapshot output.
5. Add restart fidelity and output-time landing.
6. Add timebin scaffolding only after the synchronized PM path is certified.
7. Add tree or TreePM only after the PM cosmology MVP is already useful.

## Key Recommendations

- Make PM gravity the only required backend for the first cosmology milestone.
- Keep direct gravity as a tiny-N oracle from the start.
- Delay tree work until PM + cosmology loop + IO + exact gates are already
  closed.
- Treat comoving coefficient bookkeeping as a parity surface, not as hidden
  implementation detail.
- Reuse `MultiCode` exact-growth and IC tooling aggressively; it already solves
  the hardest "same IC, same cosmology" comparison problems in this repo.
- Require restart fidelity before claiming a credible production cosmology path.
