# Radiation-coupled full MHD

`step_radiation_godunov!` keeps the nonlinear GLM-MHD Godunov update active at
all redshifts. It replaces neither the induction equation nor the Riemann
solver with a terminal-velocity evolution. Instead, FFTs supply a nonlocal
photon-baryon response correction to the local Compton-drag midpoint states.
There is therefore no terminal-to-MHD handoff.

## Model

### Evolved photon-moment closure

Early-PMF production uses `response_model=:moments`. It evolves four persistent
photon fields in the RFFT half spectrum:

```text
delta_gamma, v_gamma,x, v_gamma,y, v_gamma,z.
```

For each mode the gas density and velocity, photon monopole, and photon velocity
obey

```text
delta_b'     = -i k . v_b
v_b'         = -i k cs^2 delta_b + Gamma (v_gamma-v_b) + F_MHD
delta_gamma' = -(4/3) i k . v_gamma
v_gamma'     = -(c^2/4) i k delta_gamma
               + R Gamma (v_b-v_gamma) - S_gamma(k) v_gamma.
```

The scattering terms conserve baryon-plus-photon momentum exactly. In the
tight limit, adiabatic `delta_gamma=4 delta_b/3` gives Eq. 57,
`cs,radiation^2=c^2/[3(1+R)]`. Every mode is advanced with the two-stage,
second-order, L-stable SDIRK method; `Gamma dt` therefore does not impose an
explicit timestep restriction.

The photon quadrupole is eliminated rather than stored. Its transverse stress
rate is

```text
S_T = R Gamma (1/A_T - 1),
```

using the exact static angular integral below. Thus it approaches
`c lambda k^2/5` without cancellation in Float32 and approaches `O(c k)` phase
mixing in free streaming. The longitudinal rate has the standard `4/3`
diffusion enhancement and approaches the same free-streaming limit. This
algebraic variable-Eddington closure retains photon pressure and velocity
memory while avoiding five persistent quadrupole fields. It is a closed P1
model with a nonlocal shear closure, not a full polarized Boltzmann hierarchy.

`initialize_photon_moments!` starts from adiabatic photon density and comoving
gas/photon velocity by default. The runner exposes
`MHD_RADIATION_PHOTON_ADIABATIC` and `MHD_RADIATION_PHOTON_COMOVING` as explicit
initial-condition controls.

### Static angular response

The prior memoryless path uses `response_model=:angular`. For
`q=k lambda_gamma`, the
static angular integrals are

```text
A0 = atan(q)/q
AL = 3 (1-A0)/q^2
AT = (3 A0-AL)/2.
```

Small-q series are evaluated explicitly in Float32. The transverse response
then uses

```text
stream  = 1-AT
I       = R/(R+AT)
gamma_T = I Gamma_Compton stream.
```

With the physical identity `Gamma_Compton R lambda_gamma = c`, this gives the
exact tight-coupling damping `c lambda k^2/[5(1+R)]`; as `q` grows it approaches
local Compton drag. The longitudinal viscosity retains the diffusion-limit
`4/3` ratio and approaches local drag. Radiation pressure uses `AL^2`, which
recovers Eq. 57 in tight coupling and makes the restoring frequency vanish in
free streaming. That last longitudinal construction is a controlled closure,
not a complete frequency-dependent Boltzmann hierarchy.

### Legacy compact bridge

The homogeneous background defines

```text
R = 3 rho_b c^2 / (4 rho_gamma)
q = k lambda_gamma
t       = clamp(q^2 / bridge_q2, 0, 1)
stream  = t^2 (3 - 2 t)
tight = 1 - stream
```

where `lambda_gamma` is the physical Thomson mean free path. Lengths are
converted to fractions of the physical box before constructing the closure.
The prior `response_model=:bridge` remains for reproducibility. Its compact C1
bridge matters physically: Eq. 57 of Jedamzik et al. is a
diffusion-limit EOS, while photon pressure is absent from their free-streaming
fluid. A rational pressure tail multiplied by `c^2` gives a large, fictitious
sound speed long after the photons are optically thin. The default
`bridge_q2=5` remains an exposed calibration parameter, not a fitted Boltzmann
transfer function.

For each Fourier mode the transverse response is integrated exactly as a
damped, forced mode. The longitudinal density and velocity response is the
exact matrix exponential of

```text
d delta / dt = -i k u_L
d u_L / dt   = -gamma_L u_L - i k (cs^2 + crad^2) delta + I F_L.
```

The matched coefficients are

```text
I       = R/(1+R) + (1 - R/(1+R)) stream
gdiff   = c lambda_gamma k^2 / (5 (1+R))
gamma_T = tight^2 gdiff + stream Gamma_Compton
gamma_L = tight^2 longitudinal_viscosity gdiff + stream Gamma_Compton
crad^2  = tight c^2 / (3 (1+R)).
```

The second tight-coupling factor on `gdiff` is required because diffusion is an
optically thick expansion. Without it, `gdiff proportional to k^2` leaves a
spurious damping tail near the transition.

The spectral result is evaluated as a correction to the already implemented
local-drag Godunov predictor. Consequently the free-streaming limit recovers
`step_drag_godunov!` rather than adding a second drag update.

## Coupling sequence

Each full MHD step performs:

1. Four real FFTs of the old velocity and density fields.
2. A homogeneous half-step mode response relative to local drag.
3. A device-side centered Maxwell-stress force followed by three real FFTs and
   a forced half-step mode response.
4. Injection of the resulting velocity correction into the Hancock face
   states before the HLL/GLM flux solve.
5. The normal nonlinear, conservative full-MHD cube update.
6. Full-step homogeneous and Lorentz-force corrections to the trial state.
   The spectral density residual removes the continuity update already
   approximated by the midpoint Godunov flux. Momentum follows the corrected
   velocity, gas internal energy changes adiabatically with density, and
   magnetic energy is retained.

The density residual is essential above the radiation acoustic CFL. Correcting
velocity alone produced a spurious `delta_b_rms=3.58e-3` over the short
high-redshift gate. The density-consistent corrector reduces that to the
`1e-5` level even with only five large steps and converges further under
timestep and spatial refinement.

Metal uses MPSGraph `rfft`/`irfft`. A packed real workspace carries the original
GLM psi field and the three midpoint corrections through the cube kernel,
avoiding Metal's indirect-buffer resource limit. MPS transforms are staged
through the zero-offset segment because nonzero-offset `MtlArray` views are not
honored reliably by MPSGraph. Density and the three vector components are
transformed as one four-channel MPS batch. This reduces MPS command submissions
without increasing the workspace.

`MHD_RADIATION_NYQUIST_GUARD` can reserve an experimental high-frequency guard
in each direction; the default `1` leaves the finite-volume MHD state and the
spectral correction unfiltered. A value of `15/16` delayed the observed
grid-axis instability but introduced a measurable early density floor, so it
is a diagnostic control rather than a production default.

For the legacy bridge only, if the fundamental mode is already above the compact free-streaming boundary,
all resolved modes reduce exactly to local Godunov drag. The implementation
bypasses every radiation FFT and does not allocate a radiation workspace in
that case.

### Bounded free-streaming fast path

The evolved-moment model cannot use the legacy zero-work bypass because its
photon monopole and velocity remain dynamical. It can, however, omit the
spectral correction to the gas when all resolved modes are sufficiently close
to local drag. `radiation_free_streaming_bound` takes the maximum, at the
fundamental mode, of:

- the fractional transverse-drag correction;
- the fractional longitudinal-drag correction;
- radiation pressure relative to gas pressure.

All three decrease monotonically into free streaming. When
`MHD_RADIATION_FREE_STREAM_TOL` is positive and this bound is below the
requested tolerance, the step uses the normal full-MHD local-drag Godunov
update. The persistent photon state is still advanced from the old gas
velocity and density and from the time-centered Lorentz force. This requires
two forward transform batches, but no inverse transforms or spectral gas-state
corrections. The PMF runner defaults to `5e-3`; a zero tolerance disables the
optimization, and the library API remains zero by default.

The runner reports the instantaneous bound, fast and fully spectral step
counts, and separate bare-MHD and radiation timings. The tolerance is part of
checkpoint compatibility and the final TSV record. This is an accuracy
control, not merely a performance switch; each production configuration must
be checked against a zero-tolerance history and photon-spectrum reference.

## Memory

The current workspace owns four real arrays in one packed allocation and four
half-complex spectra:

```text
4 * N^3 * sizeof(Float32)
+ 4 * (N/2+1) * N^2 * sizeof(ComplexF32)
```

This approaches 32 bytes per cell. The evolved photon state adds

```text
4 * (N/2+1) * N^2 * sizeof(ComplexF32)
```

or approximately 16 bytes per cell. Both are allocated once and reused, so there
are no per-step full-grid Julia allocations, but this first implementation is
not yet memory-lean enough for the largest production grids. The next memory
optimization is to alias spectra and real staging against lifetime-compatible
gravity/diagnostic scratch.

## Runner

Enable the path in `test/pmf_dm_lattice_bench.jl` with:

```sh
MHD_BACKEND=metal \
MHD_DRAG_DT_MODE=radiation \
MHD_COMPTON_DRAG=1 \
MHD_TERMINAL_LBOX_CKPC=2 \
MHD_RADIATION_RESPONSE_MODEL=moments \
julia --project=lib/MHDKernels/test \
  lib/MHDKernels/test/pmf_dm_lattice_bench.jl
```

`MHD_RADIATION_BRIDGE_Q2` and `MHD_RADIATION_LONG_VISCOSITY` expose the two
legacy calibration surfaces. `MHD_RADIATION_RESPONSE_MODEL=moments` is the
early-PMF runner default. Per-cycle logs report `R`, `lambda/L`, the
fundamental and Nyquist `q`, streaming weights, and radiation-to-gas sound-speed
ratios.

Radiation mode also changes two runner defaults:

- `MHD_RADIATION_MAX_DLNA=2.5e-4` caps the expansion per step while any resolved
  mode is in the tight/bridge regime. `MHD_RADIATION_HIGHZ_MAX_DLNA` can impose
  a smaller ceiling above `MHD_RADIATION_HIGHZ_ZMIN`; the matched high-start
  campaign uses `5e-5` above `z=1e6`. The cap is released in the exact
  free-streaming bypass.
- `MHD_GLM_PROJECT_EVERY=1` projects the magnetic field each step. This reduced
  `divB*dx/Brms` from order unity to about `2e-7` in the Metal convergence gate.
- `MHD_RADIATION_NONLINEAR_CFL=0.4` limits the displacement represented by the
  spectral correction inside the nonlinear Godunov update. An unlimited value
  made the angular closure non-finite after 16 high-redshift N64 steps.

`MHD_RADIATION_ENERGY_LEDGER_EVERY=N` samples kinetic, internal, magnetic, and
total gas-state changes using the existing old-state and FFT workspaces. With
`:moments` it also reports the quadratic photon perturbation energy and the
resolved gas-plus-photon change. Angular phase mixing represents transfer into
unresolved radiation multipoles/background heat, so this diagnostic is a
resolved perturbation-energy ledger, not global thermodynamic conservation.

For resolution studies, set `MHD_PK_KMAX` together with `MHD_PK_NBINS` so all
grids use the same physical `P(k,mu)` edges. Binning and accumulation remain on
the device; only the small tables return to the host. Gas, DM, photon-density,
and magnetic spectra use one mu bin and are isotropic shell averages.
Photon-velocity and photon-baryon-slip spectra retain the configured angular
bins.

Magnetic diagnostics include `B_vector` and the conservative Maxwell-stress
split:

```text
B_pressure_force = -grad(B^2/2)
B_tension_force  = div(B B)
B_lorentz_force  = B_pressure_force + B_tension_force.
```

The split uses the same centered stencil as the force entering the radiation
coupling. With the measured small magnetic divergence, `div(B B)` is the field
curvature term `(B dot grad)B` up to truncation error. The three auto-spectra
also recover the pressure-tension cross spectrum through
`2 P_PT = P_lorentz - P_pressure - P_tension`; this distinguishes separate
damping from changing cancellation between the force terms.

Run the reproducible Metal production gate with:

```sh
julia --project=lib/MHDKernels/test \
  lib/MHDKernels/test/radiation_production_gate.jl
```

Set `MHD_GATE_TEMPORAL_REFERENCE=1` to add a refined `N=128` temporal reference
that stops at the coarse run's exact final redshift. `MHD_GATE_OUT` selects a durable artifact directory; otherwise the
gate uses a timestamped temporary directory and refuses to overwrite files.

## Validation boundary

Covered automatically:

- tight/free asymptotic coefficient limits;
- exact transverse forced-mode response;
- longitudinal radiation-pressure response;
- exact density and velocity response for a super-acoustic longitudinal mode;
- exact density and velocity response for a Lorentz-forced mode;
- mass conservation and positive density in both single-mode gates;
- recovery of local Godunov drag in the free-streaming limit;
- CPU/Metal full-step parity.
- photon-baryon momentum conservation in the scattering solve;
- the Eq. 57 tight-coupling sound speed;
- persistent photon-state and resolved photon-energy advancement;

Covered by the reproducible production gate:

- fixed-physical-mode 64/128/256 spatial convergence;
- identical initial magnetic amplitude and populated `P(k,mu)` mode counts;
- normalized magnetic divergence below `1e-5`;
- an optional 500-step temporal reference with a 1% magnetic-amplitude gate.

An earlier static-angular gate reported a `6.13e-6` final-`Brms` difference,
but it predates the same-epoch harness correction and is retained only as
historical performance evidence. For `:moments`, the corrected gate compares 22 coarse cycles with
550 refined cycles at `z=994515.090703`; final `Brms` differs by `5.24e-8`.

Required campaign evidence before interpreting a physical UV band:

- single-mode comparison against linear photon-baryon theory across `k lambda`;
- start-redshift convergence for matched comoving Batchelor modes;
- start-redshift, timestep, spatial, physical-cutoff, and overlapping-band
  convergence using identical low-k phases and amplitudes;
- stable recombination-era xHII and thermal histories with symmetric,
  expansion-aware chemistry coupling;
- convergence of the algebraically eliminated quadrupole against a higher
  photon hierarchy before claiming full Boltzmann accuracy.

The moment closure evolves spatial photon energy and momentum and captures
leading-order nonlocal radiation inertia, pressure, diffusion, and
free-streaming phase mixing. It does not evolve polarization or independent
anisotropic radiation moments and must not be described as a full Boltzmann
calculation.

## Metal measurements

Measured on the Apple Metal host with `N=128`, `MHD_CHEM_VUNIT=1e8`, a 2 ckpc
comoving box, 1 pG comoving field normalization, chemistry and gravity off:

| Regime | Redshift | FFT workspace | Steady throughput |
| --- | ---: | ---: | ---: |
| all resolved modes free-streaming | 10000 | 0 | 167.5 Mcell/s |
| radiation diffusion FFT active | 1000000 | 64.5 MiB | 90.5 Mcell/s |

The evolved-moment fixed-mode gate at `z=10^6` measured 6.7, 37.0, and
88.1 Mcell/s at `N=64,128,256`, respectively. At `N=128` the matched static
angular control measured 57.8 Mcell/s, so retaining photon memory currently
costs 36% throughput at that size. The `N=64/128/256` moment runs preserved the
same initial `Brms=4.1681495e-3`, ended at `Brms=4.1681113e-3`,
`4.1681455e-3`, and `4.1681490e-3`, and kept normalized magnetic divergence
below `2.24e-7`. Their density response is at the Float32 diagnostic floor
(`0` to `1.65e-9`) over this short 20-step gate.

A four-cycle `N=64` smoke with full mixing chemistry, gas+DM gravity, MPS FFTs,
and the moment closure sustained 11.3 macro-Mcell/s after warmup. MHD plus the
radiation correction was 65% of measured time and chemistry 17%; the run kept
`xHII=1`, `delta_b_rms=7.73e-9`, and `divB dx/Brms=1.53e-7`.

The active result uses five measured steps after two warmups. Four-channel MPS
batching improved the same short benchmark from 63.0 to 90.5 Mcell/s. The full
Metal regression reports 395.8 Mcell/s for the bare 128-cubed MHD cube, so the
spectral predictor remains the dominant coupled-path overhead.

For the full-network `z=2600` to `500` production interval, the bounded
evolved-moment path (`MHD_RADIATION_FREE_STREAM_TOL=5e-3`) measured 63.8
macro-Mcell/s at `N=128`, compared with 47.4 Mcell/s for the matched
zero-tolerance reference. The split sustained rates were 9.7 ms/cycle for the
PPM/HLL MHD cube and 6.1 ms/cycle for radiation; the previous aggregate
28.5 ms/cycle MHD label included the photon predictor/corrector. Across 21
common outputs the bounded run differed by at most `1.14e-4` in gas-density
RMS, `7.18e-5` in DM-density RMS, and `1.52e-5` in mean HII. A separate
`z=2600` to `2000` photon-spectrum gate kept weighted photon-density,
photon-velocity, and slip differences below `3.46e-3`.

The historical compact-bridge fixed-mode gate used a dimensionless unit box representing 2 ckpc,
1 pG comoving normalization, `z=1e6` to `z=994515`, 20 measured steps after two
warmups, `kmax=20 pi`, and the production `max_dlna=2.5e-4`:

| N | Cells / shortest wavelength | Brms final | delta_b_rms | divB dx / Brms | Throughput |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 6.4 | 4.166443e-3 | 1.3064e-7 | 1.676e-7 | 23.9 Mcell/s |
| 128 | 12.8 | 4.167969e-3 | 4.4627e-9 | 2.234e-7 | 77.8 Mcell/s |
| 256 | 25.6 | 4.168129e-3 | 6.9225e-10 | 1.955e-7 | 106.9 Mcell/s |

The 128/256 final `Brms` mismatch is `3.83e-5`; 64 cells is visibly
under-resolved. Production initial fields should therefore retain at least 12
cells per shortest initialized wavelength. These timings exclude Julia/Metal
startup and diagnostic FFTs. They certify the short leading-order closure
gate, not a complete run from high redshift through recombination to `z=50`.
