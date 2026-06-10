# Contact-preservation test plan for local PPM

The immediate question is whether the stateless one-ghost local PPM gives up any
of CW84's contact-specific behavior.  Treat this as a contact-focused benchmark,
not a general hydro-code zoo.

## Phase 1: solver-local diagnostics

Use `lib/PPMKernels/bench/compare_contacts.jl` for fast, controlled sweeps across
the same solver set used in the turbulence/sound-wave studies:

- `Hancock-PLM`
- `Hancock-PPM-tr-2shk`
- `Local-PPM-tr-2shk`
- `PPM-DirectEuler`
- `PPML-trace`

Run periodic contact advection and top-hat advection first:

```bash
cd lib/PPMKernels
julia --project=test bench/compare_contacts.jl 256 4 both
```

The useful metrics are contact-specific: density L1 error, 10-90% transition
width in cells, density overshoot/undershoot, pressure wiggle across the contact,
and wall time.  For paper runs, repeat at `nx = 64, 128, 256, 512` and at 1, 4,
and 10 crossings.  These runs write `bench/contact_out/contact_metrics.csv` and
per-case profile CSVs.

## Phase 2: shock/contact interactions

Add the next cases in this order:

1. Sod and Lax tubes with separate contact-region diagnostics, not just global
   L1 error.
2. Shu-Osher shock-density-wave interaction, measuring post-shock entropy-wave
   amplitude and phase.
3. Woodward-Colella blast as a guardrail against removing too much flattening.

These can stay in PPMKernels while we tune the local PPM contact behavior.  The
first Phase-1 smoke runs showed the failure mode to watch: local PPM kept the
contact compact but permitted about 5% density ringing on a pure entropy jump,
while PPML was sharper and non-ringing in that narrow case.  A pure
contact-triggered PLM fallback was robust but broadened under repeated
supersonic advection.  The current local-PPM variant keeps the all-primitive PLM
fallback for shocks, but handles pressure-smooth entropy contacts through a
density-only fallback plus a mild one-ghost density steepener.  At `nx=128`,
`u0=5` (`Mach_adv≈4.2`), this keeps the contact width to about 12-13 cells over
4-10 crossings, versus 16-22 cells for the pure fallback, while retaining the
sound-wave/turbulence advantages.

A density-only THINC/BVD branch is also promising but not yet default-ready. A
one-ghost local BVD proxy selecting a bounded THINC density profile can keep the
same Mach-4 contact to 4-10 cells over repeated crossings, but long-run density
ringing grows unless the THINC blend is heavily damped. The current experiment
uses a conservative partial THINC blend; next work is an anti-oscillation gate or
post-update damping for cells adjacent to THINC-selected contacts.

## Phase 1b: carried-label contact memory

The first material-coordinate prototypes are now in
`Local-PPM-exact-label-THINC` and `Local-PPM-carried-label-THINC`.  Both are
intentionally opt-in and benchmark-local.  The exact-label variant supplies the
uniform-advection label `a(x,t) = x - u0*t mod 1`, so it is an upper-bound test
of whether material-coordinate information can help.  The carried-label variant
evolves `rho*a` and `rho*a^2` inside the local-PPM split sweep: after the hydro
mass flux is computed, a small local-PPM scalar-flux kernel reconstructs the
carried primitive labels and updates the two conserved moments with the same
flux-divergence machinery.  The local variance `<a^2> - <a>^2` is used as a soft
"clean material sheet" factor.  Default `Local-PPM-tr-2shk` behavior is
unchanged.

At `nx=128`, `u0=5`, contact advection:

| crossings | solver | L1(rho) | width | overshoot | undershoot |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | Local-PPM-tr-2shk | 2.1549e-2 | 8 | 6.241e-3 | 4.053e-3 |
| 1 | Local-PPM-exact-label-THINC | 1.8055e-2 | 8 | 6.241e-3 | 3.853e-3 |
| 4 | Local-PPM-tr-2shk | 2.5351e-2 | 10 | 1.663e-2 | 8.786e-3 |
| 4 | Local-PPM-exact-label-THINC | 1.9473e-2 | 8 | 1.656e-2 | 8.733e-3 |
| 10 | Local-PPM-tr-2shk | 2.7832e-2 | 10 | 3.731e-2 | 1.895e-2 |
| 10 | Local-PPM-exact-label-THINC | 2.1279e-2 | 8 | 3.471e-2 | 1.745e-2 |
| 10 | Local-PPM-carried-label-THINC | 2.6815e-2 | 8 | 3.881e-2 | 1.976e-2 |
| 10 | PPML-trace | 9.9047e-3 | 4 | 0 | 0 |

The exact-label top-hat case shows the same positive trend: at 10 crossings it
improves L1 from `2.7753e-2` to `2.0188e-2` and width from 10 to 8 cells, with
slightly lower density ringing.  The in-solver carried-label top-hat result at
10 crossings is `2.6792e-2`, width 8, overshoot `3.913e-2`, undershoot
`1.975e-2`.  That is a real sharpness/L1 improvement over baseline, but with a
small ringing penalty.  Stronger carried-label THINC gains recover the width
more aggressively but produce unacceptable overshoot/undershoot; weaker gains
collapse back to baseline.

Important next step: improve the carried-label robustness rather than simply
raising THINC gain.  Candidates are a seam-aware/non-sawtooth label
representation, bounded color fractions for known material interfaces, or a
post-THINC anti-oscillation gate using the carried variance plus density extrema.

## Phase 3: paper figures through EnzoViz

Once the solver-local suite identifies the decisive cases, promote only those to
EnzoViz pages.  Use EnzoViz for consistent figure styling, frame cadence, axis
ranges, and 2D AMR overlays:

- 1D contact/top-hat/Sod: stacked `density`, `pressure`, `speed` line plots with
  fixed y-ranges across solvers.
- 2D Kelvin-Helmholtz and cloud-crushing: density plus pressure/speed panels,
  with shared colormap ranges and final-frame PNG export from the same
  `VizSession` path.

The EnzoViz layer should not be the first debugging surface for solver behavior:
it is the final visualization path after the profile CSVs and metrics show which
comparisons are worth turning into manuscript figures.
