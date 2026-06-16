# PM Gravity Audit Path Against The Direct Tiny-N Oracle

Date: 2026-06-15

This note pins down the first direct-vs-PM gravity comparison gate for the
AREPO/PowerFoam rewrite. The repo now has an executable tiny periodic
preflight, but it is still intentionally a pre-certification gate: it runs the
PM chain numerically and pairs it with deterministic periodic image-sum
diagnostics plus a finite zero-net-force direct oracle while keeping the
production periodic convention caveat visible.

## Executable Preflight

The repo now owns an executable preflight:

```bash
julia --project=lib/PowerFoam lib/PowerFoam/examples/arepo_pm_gravity_gate_skeleton.jl
```

That script now:

- overlays the repo-local `lib/PoissonKernels/test` environment while probing
  `PoissonKernels`, so the gate can resolve sibling package deps from the
  monorepo without mutating the user's global Julia project state;
- runs the root PM chain numerically when `PoissonKernels` is locally
  loadable: deposit -> mean subtraction -> FFT solve -> periodic ghost fill ->
  gradient -> particle interpolation;
- emits machine-readable numeric rows for the PM fixture, direct periodic
  image-sum diagnostics, the finite background-subtracted direct oracle, and PM
  self-consistency checks;
- writes a timestamped `README.md` plus CSV under
  `lib/PowerFoam/examples/out/arepo_pm_gravity_gate_skeleton/<timestamp>/`;
- reports concrete blockers only when the PM chain itself cannot load.

It is meant to keep the gate honest: the PM side now runs for real, and the
finite direct oracle gives the gate an executable zero-net-force comparison
target, but the artifacts still make it obvious that this is not yet an
Ewald-style or bridge-certified production periodic comparator.

### Machine-readable preflight contract

The preflight CSV now carries these explicit row categories:

- `fixture`: exact numeric setup values such as particle count, `Npm`, ghost
  depth, and cell-center registration residuals;
- `direct_diag`: finite-image-sum acceleration diagnostics for small `Nimg`;
- `direct_oracle`: finite symmetric image sum with the net force projected out
  plus one-shell convergence diagnostics;
- `pm`: numeric PM deposit / solve / interpolation diagnostics;
- `pm_vs_direct_oracle`: explicit PM vs finite zero-net-force direct-oracle
  deltas;
- `blocker`: exact reasons the run could not execute the PM chain, if any.

When the PM side is unavailable, the `blocker` rows now also record the exact
scoped `LOAD_PATH` entries plus the `find_package` results for
`PoissonKernels`, `FFTW`, and `KernelAbstractions`, along with the concrete
`using PoissonKernels` error string.

The production PM harness sequence remains exact:

1. `cic_deposit!`
2. `fft_poisson_root!`
3. `fill_periodic_ghosts!`
4. `interp_accel_to_particles!`

That sequence is now exercised numerically by the example whenever
`PoissonKernels` is available on `LOAD_PATH`.

## Surfaces Inspected

PowerFoam:

- `lib/PowerFoam/src/arepo_gravity_scaffold.jl`
- `lib/PowerFoam/examples/arepo_gravity_direct_smoke.jl`
- `lib/PowerFoam/planning/gravity_component_gate.md`
- `lib/PowerFoam/planning/cosmology_gravity_workbreakdown.md`

PoissonKernels:

- `lib/PoissonKernels/src/deposit.jl`
- `lib/PoissonKernels/src/fft_poisson.jl`
- `lib/PoissonKernels/src/particle_push.jl`
- `lib/PoissonKernels/src/field_ops.jl`
- `lib/PoissonKernels/test/test_deposit.jl`
- `lib/PoissonKernels/test/test_particle_push.jl`

MultiCode:

- `lib/MultiCode/src/gravity_slot.jl`
- `lib/MultiCode/src/zeldovich.jl`
- `lib/MultiCode/src/enzo_resident.jl`
- `lib/MultiCode/examples/cicass_gravity_check.jl`
- `lib/MultiCode/examples/cicass_highz_pk.jl`

## Current Reality

The repo already has the two halves needed for the preflight:

1. A tiny-`N` direct oracle in `PowerFoam`:
   - `arepo_direct_gravity_accel!`
   - `arepo_direct_gravity_accel`
   - `arepo_direct_gravity_potential_energy`
   - `arepo_direct_gravity_oracle`

2. A periodic PM chain in `PoissonKernels`:
   - `cic_deposit!`
   - `fft_poisson_root!`
   - `fill_periodic_ghosts!`
   - `interp_accel_to_particles!`
   - optional `particle_kick!` / `particle_drift!` for later time integration

The old open-box direct oracle remains open-box only (`periodic=true` errors),
while the PM path is explicitly periodic and mean-free. The PM audit therefore
uses a separate helper, `periodic_background_subtracted_image_oracle`, for the
periodic force-only comparison scaffold. The current gate runs as:

- a real PM solve on the periodic frozen-particle fixture; and
- raw periodic image-sum diagnostics plus a finite zero-net-force direct oracle
  on the same fixture.

It still cannot claim certified PM-vs-direct agreement until the finite direct
side is calibrated against the exact periodic convention we want to match.

The current executable limitation is exact and should stay visible in every
preflight artifact: `arepo_direct_gravity_accel!` and
`arepo_direct_gravity_potential_energy` still error with
`periodic=true is not implemented` in
`lib/PowerFoam/src/arepo_gravity_scaffold.jl`.

## Direct-Side Convention Decision

The immediate blocker is conceptual, not implementation detail: the direct side
needs one explicit periodic-force convention before any PM comparison can mean
anything.

Two candidate conventions are reasonable enough to write down:

1. finite periodic image sum:
   - evaluate the Newtonian pair force from a symmetric cube of image offsets
     `n = (nx, ny, nz)`, `nx, ny, nz in [-Nimg, Nimg]`;
   - omit the exact self copy `i == j && n == (0,0,0)`;
   - treat the result as a truncated diagnostic that approaches a periodic
     limit only as `Nimg` grows.

2. force-only background-subtracted periodic convention:
   - define the direct target to be the acceleration generated by the particle
     copies after removing the uniform background / DC mode, matching the
     periodic PM solve's mean-subtracted source;
   - treat force as the only first-class observable for the gate;
   - do not make periodic potential energy part of gate 1.

Recommended convention for the first PM comparison:

- adopt the force-only background-subtracted convention as the production gate
  definition;
- keep finite image sums only as a tiny diagnostic scaffold.

Reason:

- the planned PM path already solves the mean-free periodic problem;
- a truncated image sum is not unique because its answer changes with `Nimg`
  and summation shape/order;
- the first gate is about acceleration at particle locations, where the PM
  convention is clear and the potential gauge is irrelevant;
- this keeps the direct reference aligned with the quantity we actually intend
  to certify later: periodic PM force, not an arbitrary truncated periodic
  energy.

## Exact First Gate

The first gate should compare particle accelerations, not trajectories and not
energies.

### Problem definition

- Domain: periodic unit cube `[0,1)^3`
- Time integration: none; frozen particles only
- Cosmology factors: disabled for the gate
  - `a = 1`
  - no kick/drift
  - deposit `disp = 0`
- Softening: `0` on both sides for the first pass
- Mesh: `Npm = 16`
- Ghost depth for PM interpolation: `ng = 3`

### Particle fixture

Use four particles at exact PM cell centers so the first gate does not mix
registration mistakes with force-law mistakes:

- `x = [ 3.5, 10.5,  6.5, 12.5 ] / 16`
- `y = [ 5.5,  5.5, 11.5,  9.5 ] / 16`
- `z = [ 7.5,  7.5,  4.5, 13.5 ] / 16`
- `m = [ 1.0,  1.0,  1.0,  1.0 ]`
- `vx = vy = vz = 0`

Why this layout:

- exact cell centers match the `shift = -0.5` CIC registration already tested in
  `PoissonKernels`;
- the fixture is asymmetric, so all acceleration components need to be correct;
- equal masses make net-force and pair-symmetry diagnostics easier to interpret.

### PM-side construction

The first PM comparison should use the repo's existing periodic root-mesh chain,
with no extra cosmology or hierarchy logic:

1. Deposit particle mass with
   `cic_deposit!(ρ, px, py, pz, vx, vy, vz, m; N = 16, disp = 0, shift = -0.5)`.
2. Reshape the flat `ρ` vector to `(16, 16, 16)`.
3. Subtract the unweighted mean to form the periodic zero-mode-free source.
4. Solve with `fft_poisson_root!` on the unit box.
5. Embed the active `16^3` acceleration field in a ghost-padded
   `(16 + 2ng)^3 = 22^3` array.
6. Fill periodic ghosts with `fill_periodic_ghosts!(...; ng = 3)`.
7. Interpolate accelerations back to the original particle positions with
   `interp_accel_to_particles!`, using:
   - `dcoef = 0`
   - `cellsize = 1 / 16`
   - `leftedge = (-3/16, -3/16, -3/16)`

### Green's function choice

Use `greens = :spectral` for the first gate.

Reason:

- that is the repo's root-periodic PM convention;
- it matches the stated PM-MVP path in
  `lib/PowerFoam/planning/cosmology_gravity_workbreakdown.md`;
- `:discrete7` remains a useful secondary diagnostic, but it should not define
  the first production-facing PM comparison if the intended backend is the
  spectral root solver.

## Direct-side comparison target

The direct side must evaluate the same frozen periodic configuration and return
per-particle accelerations in the same SoA order as the PM result.

For the gate to be exact, the direct oracle needs all of the following:

- periodic-image gravity on `[0,1)^3`;
- the same effective force law being claimed by the PM gate;
- a clearly documented background/zero-mode convention.

For this audit, "the same effective force law" should mean:

- direct acceleration under the periodic force-only background-subtracted
  convention;
- same particle masses and positions as the PM fixture;
- no softening in gate 1 unless PM and direct both add it explicitly.

`arepo_direct_gravity_oracle` remains useful as the non-periodic force-sign
scaffold. `periodic_background_subtracted_image_oracle` is the current
executable periodic PM comparison scaffold, but it is still a finite-image
oracle rather than a certified production periodic/Ewald comparator.

## Expected Diagnostics

The first gate should emit these diagnostics in this order.

### Registration and conservation

- `sum(ρ_pm) == sum(m)` to floating-point tolerance
- `sum(rhs_pm) ≈ 0`
- one-line confirmation that every particle lies on a PM cell center under the
  `shift = -0.5` convention

These tell us the mesh source is the intended one before we look at forces.

### PM self-consistency

- `sum(m .* ax_pm)`, `sum(m .* ay_pm)`, `sum(m .* az_pm)`
- max absolute self-force on a one-particle cell-centered control case

The PM chain should not create a large net force on the closed particle system.

### Direct-vs-PM force comparison

Per component and per particle:

- `maxabs(ax_pm - ax_dir)`, same for `y`, `z`
- `relLinf` against `max(abs.(a_dir))`
- vector-angle or cosine agreement for each particle acceleration

Aggregate:

- `maxabs(norm(a_pm[i]) - norm(a_dir[i]))`
- center-of-mass force residual on both sides

### Deliberately not first-class in gate 1

- PM vs direct potential energy
- time integration / kicks / drifts
- comoving coefficients
- mixed gas+particle source terms

Those add convention choices that are not needed to certify the first force gate.

## What Is Already Fixed By Existing Repo Surfaces

These assumptions do not need to be rediscovered:

- CIC deposit registration is cell-centered with `shift = -0.5`.
- PM positions are normalized to `[0,1)`.
- PM deposit can fuse a drift term, but the first gate should keep `disp = 0`.
- Particle interpolation expects ghost-padded acceleration grids and explicit
  `leftedge`.
- The root periodic Poisson solve drops the DC mode and therefore expects a
  mean-subtracted source.

## Remaining Certification Gaps

### 1. The open-box direct gravity API is still intentionally non-periodic

`arepo_direct_gravity_accel(...; periodic = true)` intentionally errors today.
The PM gate uses `periodic_background_subtracted_image_oracle` instead of
changing that open-box API.

### 2. The finite oracle still needs production-convention certification

The PM solve compares forces after mean subtraction on a periodic box. A direct
periodic oracle must say explicitly whether it represents:

- particle copies only;
- particle copies plus the uniform compensating background;
- or an Ewald-style convention equivalent to the PM zero-mode choice.

Without this, a direct-vs-PM force mismatch can still be physically ambiguous.

Resolution for this planning note:

- the current finite direct oracle targets the force-only
  background-subtracted convention by projecting the net force to zero;
- it should remain labeled as a finite-image comparison scaffold until we
  certify it against an Ewald-style or bridge-equivalent periodic convention.

### 3. PowerFoam still lacks a production periodic direct comparator

`lib/PowerFoam/examples/arepo_pm_gravity_gate_skeleton.jl` and
`lib/PowerFoam/src/arepo_pm_gravity.jl` now give `PowerFoam` an executable
frozen-particle PM preflight plus finite direct oracle. The remaining gap is
not the PM harness itself; it is the lack of a production direct-side periodic
comparator with the same zero-mode convention.

### 4. Root PM force comparison should lock to acceleration before potential

Potential offsets are gauge-dependent even before periodic/direct conventions
are reconciled. Force is the cleaner first observable.

## Recommended Next Implementation Slice

The current executable slice is:

1. build the four-particle cell-centered fixture;
2. run the frozen PM chain with `greens = :spectral` when `PoissonKernels` is
   loadable;
3. run finite periodic image sums for small `Nimg` as direct diagnostics;
4. run the finite zero-net-force direct oracle and emit PM-vs-oracle deltas.

The next safe slice after that is:

1. add a one-particle self-force control to the PM preflight;
2. add an Ewald-style or bridge-equivalent production periodic convention note;
3. calibrate the finite zero-net-force oracle against that convention;
4. only after that, consider `:discrete7`, softening, or comoving kicks.

That keeps the first gate about one thing: whether the periodic PM force
recovered at the particle locations matches the intended direct tiny-`N`
reference on the same frozen box.
