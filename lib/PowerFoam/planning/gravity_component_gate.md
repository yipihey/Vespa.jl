# Gravity Component Gate

Date: 2026-06-15

This note introduces the first gravity/cosmology rewrite scaffold for the
PowerFoam-side AREPO plan. It is intentionally much smaller than a PM or TreePM
backend: the goal is a tiny-`N` direct-force oracle that other workstreams can
use immediately as a package-exported executable smoke gate.

## Scope

Current gate-owned files:

- `lib/PowerFoam/examples/arepo_gravity_direct_smoke.jl`
- `lib/PowerFoam/src/arepo_gravity_scaffold.jl`

Still no edits here to:

- tests
- existing hydro or tessellation code

## What The Scaffold Provides

The scaffold defines four direct-gravity helpers:

- `arepo_direct_gravity_accel!`
  - in-place direct acceleration evaluation on SoA arrays
- `arepo_direct_gravity_accel`
  - allocating wrapper for the same force law
- `arepo_direct_gravity_potential_energy`
  - total pairwise gravitational potential energy
- `arepo_direct_gravity_oracle`
  - convenience wrapper returning accelerations plus total potential energy

All APIs are plain Julia and dependency-light. They accept particle data in
structure-of-arrays form:

- `x`, `y`, `z`
- `m`
- optional preallocated `ax`, `ay`, `az`

## Force Law And Current Limits

The current direct oracle uses

- Newtonian pairwise gravity;
- scalar Plummer-style softening via `r^2 + eps^2`;
- `periodic=false` only;
- a symmetric `O(N^2)` pair loop.

This is meant for:

- tiny oracle checks;
- force-sign and energy-sign sanity gates;
- PM/backend comparison on frozen miniature particle sets;
- early cosmology/plumbing work where exact tiny-`N` behavior matters more than
  scalability.

This is explicitly not yet:

- periodic-image gravity;
- comoving integration;
- timestep scheduling;
- PM deposition/solve/interpolation;
- tree or TreePM production gravity.

## Why This Gate Exists First

The rewrite needs a force oracle before it needs a production backend. A direct
tiny-`N` reference helps with:

- validating force directions and action-reaction symmetry;
- checking softening conventions before PM/tree code is introduced;
- giving future PM and tree slices a deterministic comparison surface.

That lines up with the broader rewrite rule: certify the component before
claiming the full cosmology loop.

## Executable Smoke Command

The scaffold is now smoke-tested through the package boundary:

```bash
julia --project=lib/PowerFoam lib/PowerFoam/examples/arepo_gravity_direct_smoke.jl
```

The smoke gate:

- uses `using PowerFoam`;
- evaluates a two-body direct-gravity oracle with expected accelerations
  `(3, 0, 0)` and `(-2, 0, 0)` and potential energy `-6`;
- evaluates a three-body triangle oracle on points `(0,0,0)`, `(1,0,0)`,
  `(0,1,0)` with masses `[2,3,4]`;
- verifies direct-force and potential-energy agreement against an explicit
  pair-sum oracle;
- verifies action-reaction by checking the momentum-weighted residual
  `sum(m .* a)` in each component;
- writes a timestamped `README.md` and `results.csv` under
  `lib/PowerFoam/examples/out/arepo_gravity_direct_smoke/<timestamp>/`.

## Immediate Integration Steps

1. Reuse the same SoA surface when the PM gravity slice grows a tiny direct-vs-PM
   comparison gate.
2. Decide whether the production backend should match this scaffold's softening
   convention exactly or document a different one with an adapter gate.
3. Add periodic-image and comoving variants only when a concrete gate needs
   them; do not broaden this file preemptively.
