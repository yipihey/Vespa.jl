# Radiation-overdamped magnetic pre-evolver

`radiation_terminal_ct_induction!` advances only the magnetic field while gas
density and momentum remain fixed. It is intended for an early interval where
radiation drag relaxes velocity much faster than B evolves and the implied
compressive density response is negligible.

The method computes the Maxwell-stress acceleration, transforms it with the
MPS RFFT path on Metal, and applies the mode response from `RadiationClosure`.
The transverse response is `I F_T / gamma_T`. Tight-coupling radiation pressure
balances longitudinal force; the closure's streaming weight restores
`I F_L / gamma_L` as photon pressure vanishes. SSPRK2 constrained transport
recomputes force and velocity at the predictor stage. Optional compensated
increments retain Float32 changes smaller than one B-field ulp.

This is a slow-manifold method. A matched full-MHD continuation must initialize
momentum from the same terminal velocity. A zero-velocity full-MHD state has a
drag startup transient and is not an equivalent calibration reference.

With `diagnostics=true`, the return value includes `divergence_rms`,
`density_change_bound`, `relaxation_ratio`, and `streaming_weight`. Accumulate
`density_change_bound` conservatively and hand off to
`step_radiation_godunov!` before it reaches the calibrated density tolerance.
The current random-field gate uses 1e-5 and requires a large relaxation ratio.
The campaign runner enforces defaults of 1e-5 and 100 respectively, then writes
a handoff checkpoint instead of continuing outside the calibrated envelope.

On the M5 Metal backend, an N256 timing window measured 0.05425 s per step
(309.2 Mcell/s median over cycles 6-12) with diagnostics active. The cold first
call was 2.72 s because it includes MPS plan and kernel setup.

Validation artifacts and reproducible commands are in
`vespa-runs/analysis/20260718-pmf-bonly-overlap/README.md`. Certification covers
a transverse linear mode and a matched low-k random Batchelor field through
N256. It does not certify arbitrary spectra, long frozen-density intervals, or
regimes in which the density bound is exhausted.
