# Hydro Problem Registry Plan

Date: 2026-06-15

Purpose: define the lightweight registry layer for hydro standard problems in
`lib/PowerFoam/examples/arepo_hydro_problem_registry_smoke.jl` and set the
promotion rule for turning registry rows into executable gates.

## Scope

- Keep this artifact planning-only. It should not add new hydro kernels or edit
  `PowerFoam.jl`, `hydro2d.jl`, `hydro3d.jl`, or `test/runtests.jl`.
- Treat the registry smoke script as a manifest of parity-critical problems,
  not as evidence that the underlying physics is solved.
- Enumerate the current parity set explicitly:
  - `KH2D`
  - `Noh2D`
  - `Noh3D`
  - `SoundWave2D`
  - `Gresho`
  - `wave`
  - `turbulence`

## Registry Status Meanings

| status | meaning |
| --- | --- |
| `runnable` | a local example driver file exists and can be named by path |
| `proxy-only` | local material exists, but it is not yet an executable parity gate |
| `planned` | only the upstream AREPO reference and the desired diagnostic are known |

The registry smoke artifact is acceptable if it can be rerun quickly and emits a
stable `README.md` plus CSV under `lib/PowerFoam/examples/out/`.

## Promotion Criteria

A registry row can move from planning to an executable gate only when all of
the following are true:

1. The row has one canonical Julia command, with any required environment
   variables documented inline.
2. The gate writes a stable artifact directory under
   `lib/PowerFoam/examples/out/<gate>/...`.
3. The gate has at least one explicit pass/fail metric derived from the
   corresponding upstream AREPO example check, not from an ad hoc local metric.
4. The default problem size is smoke-scale enough for routine reruns.
5. The gate exits nonzero on a real failure, so it can later be promoted into a
   CI lane without redesign.

Additional rule: no hydro standard problem should become blocking before the
bridge/component lane beneath it is already green for the same solver/backend
surface.

## Current Row Assessment

| problem | current registry status | local driver/proxy | comment |
| --- | --- | --- | --- |
| `KH2D` | `runnable` | `examples/powerfoam_kh2d_compare_gate.jl` | already executable, but still a small compare gate rather than a frozen CI row |
| `Noh2D` | `proxy-only` | `examples/arepo_noh_proxy/` plus `examples/arepo_noh2d_proxy_gate.jl` | executable readiness shell exists; exact conversion checklist now lives in `planning/noh2d_executable_gate_plan.md`, with `standard_ic.csv` + HLL + final-field parity as the first real rung |
| `Noh3D` | `runnable` | `examples/arepo_noh3d_smoke_gate.jl` | executable now; needs repeated-run tolerance freeze |
| `SoundWave2D` | `runnable` | `examples/arepo_soundwave2d_gate.jl` | executable periodic smooth-wave rung with exact-solution and Fourier diagnostics; still `calibration-PENDING` until its tolerance surface is frozen |
| `Gresho` | `runnable` | `examples/arepo_gresho2d_gate.jl` | periodic vortex gate now exists, but the row stays calibration-PENDING until an original-AREPO profile comparison is added |
| `wave` | `planned` | none | likely needs a thin-3D or dedicated 1-D shim |
| `turbulence` | `runnable` | `examples/arepo_standard_problem_matrix.jl` | registry/bridge rollup exists, but native production closure is still open |

## Next Executable Gate To Convert

Convert `Noh2D` from proxy-only material into a true parity gate next.

Why this row first:

- it already has durable local proxy material in
  `lib/PowerFoam/examples/arepo_noh_proxy/`
- it now has a lightweight executable readiness shell in
  `lib/PowerFoam/examples/arepo_noh2d_proxy_gate.jl`, so the remaining work is
  about physics diagnostics rather than repo plumbing
- it closes a real 2-D strong-shock gap in the parity ladder
- it is narrower and better constrained than starting a fresh `wave` gate
  from scratch
- it gives a clean promotion target: final-field parity plus radial density
  profile and shock-radius metrics against the upstream `noh_2d` reference

Recommended first executable version:

- keep the existing proxy table generation as the staging path
- freeze on `standard_ic.csv`, `N=64`, `t_final=2.0`, and `HLL`
- stage one original AREPO reference run plus one PowerFoam run from the same
  IC
- emit `README.md`, per-code radial metrics, per-code final fields, and
  `field_compare.csv`
- make AREPO self-check success plus final analytic/radial tolerances the first
  blocking `Q0` rule; defer blocking per-field tolerances until repeated reruns

## Immediate Follow-On After `Noh2D`

1. freeze `Noh3D` tolerances over repeated small runs and promote it from smoke
   to a true `Q1` gate
2. freeze `SoundWave2D` tolerances over repeated runs or tie it to an upstream
   AREPO-style reference surface
3. decide whether `contact/blob` or `wave` is the better next fresh-from-zero
  2-D gate once the smooth-wave rung is stable
