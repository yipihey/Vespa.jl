# Sound-Wave 2D Executable Gate Plan

Date: 2026-06-15

Purpose: define the first small periodic smooth-flow calibration rung after the
bounded `Noh2D` gate, using the existing PowerFoam 2-D hydro API and a simple
exact acoustic-wave solution.

## Current Implementation Snapshot

As of 2026-06-15, the first executable version of this rung exists:

- `examples/arepo_soundwave2d_gate.jl`
- package-owned helpers in `src/arepo_standard_problems.jl`
- default artifact root:
  `examples/out/arepo_soundwave2d_gate/Nx32_Ny8_t0p05_hll/`

The current gate:

- builds a periodic Cartesian 2-D mesh in Julia
- seeds a low-amplitude right-moving acoustic mode
- advances it with `finite_volume_step_2d!`
- records mass/energy drift, exact-solution error, and first-mode Fourier
  retention diagnostics
- exits nonzero if the run becomes non-finite, non-positive, or fails its
  minimal numerical sanity checks

The gate is intentionally labeled `calibration-PENDING`, not physics-pass,
because its thresholds are still local sanity thresholds rather than a frozen
AREPO-backed success surface.

## Why This Problem Next

- it exercises the same 2-D hydro core as the shock rows, but in a smooth,
  low-Mach regime
- it is small and fast enough for routine reruns
- it has an exact closed-form reference state, so the first executable rung can
  report meaningful error numbers without staging an external code
- it is lower setup overhead than starting fresh with a rotating-vortex gate

## Frozen Default Rung

- grid: `Nx = 32`, `Ny = 8`
- final time: `t_final = 0.05`
- solver: `HLL`
- gamma: `5/3`
- background state: `rho0 = 1`, `p0 = 1`
- perturbation amplitude: `1e-3`
- mode number: `1`
- domain: `[0, 1] x [0, 1]`, periodic

These values are chosen to keep the run light while still showing measurable
wave propagation and dissipation.

## Current Diagnostics

The gate writes:

- `README.md`
- `metrics_powerfoam.csv`
- `profile_powerfoam.csv`
- `powerfoam.log`

Key metrics:

- `mass_rel_drift`
- `energy_rel_drift`
- `rho_l1`, `rho_l2`
- `vx_l1`, `vx_l2`
- `pressure_l1`, `pressure_l2`
- `rho_mode_amp_ratio`
- `rho_mode_phase_error`

These are enough for a calibration rung, but they are not yet justified as
blocking thresholds.

## Promotion Path

Promote this row only after one of these surfaces exists:

1. repeated-run local tolerance freeze across the default rung and one modest
   refinement rung, or
2. a direct AREPO-backed reference run with matched initial data and a stable
   amplitude/phase comparison contract

Until then:

- keep the rung executable
- keep the label `calibration-PENDING`
- do not advertise it as a certified parity row

## Likely Follow-On

Once this smooth-wave rung is stable, the next fresh 2-D hydro problem should
probably be `Gresho` or `contact/blob`, depending on whether the immediate need
is low-dissipation vortex balance or contact preservation on a moving mesh.
