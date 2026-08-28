# Early PMF production protocol

## Physical spectrum

A grid Nyquist mode always has two cells per wavelength, independent of grid
size. It cannot be made a converged MHD initial condition by increasing `N`
while continuing to refill to the new Nyquist frequency.

Production therefore defines a resolution-independent `PMFSpectrumSpec` with a
physical `kcut`, a hard physical `kmax`, one seed, a locked low-mode grid, and a
declared normalization. `normalization_kind=:total_band_rms` fixes the total
RMS of the generated band. `:reference_band_rms` fixes the RMS below
`reference_kmax` while allowing added ultraviolet modes to raise the total RMS.
The generated statistics always report `reference_band_brms_measured` and
`total_brms_measured` separately. `MHD_PMF_MIN_CELLS_PER_WAVELENGTH=16`
rejects any run whose hard cutoff is not resolved. `pmf_multiband_manifest`
computes the minimum even grid for each physical UV level.

Finer cumulative levels preserve the exact low-k vector-potential coefficients
and their amplitude while adding modes above the prior cutoff. Their total
`Brms` is allowed to grow; renormalizing every wider spectrum back to one total
`Brms` would change the shared large-scale realization.

Every physical amplitude must be written as

```text
Bcom_rms(z_reference; kmin <= k <= kmax) = value
```

with normalization kind, cutoff window, and seed. A bare "3 pG PMF" is not a
complete Batchelor-spectrum specification. The current direct initializer has
no pre-start magnetic transfer model, so `MHD_PMF_NORMALIZATION_Z` must equal
`MHD_ZSTART`. A field normalized at an earlier primordial epoch must first be
transferred to the handoff redshift, including consistent gas velocity,
density, and photon moments.

## Evolution

- Full GLM MHD, PPM reconstruction, and HLLD remain active at every redshift.
- Terminal and hybrid modes are excluded from production.
- Compton drag uses `MHD_DRAG_DT_MODE=radiation` and the evolved photon-moment
  closure in `radiation_coupling.jl`.
- The nonlinear spectral correction uses a 0.4-cell CFL limit.
- Recombination uses full mixing chemistry, rate tables, expansion-aware
  integration, and symmetric half steps around MHD.
- The fused full mixing network is used throughout; its conservative implicit
  atomic update avoids the former high-redshift limiter stall.
- Lyman-alpha density smoothing is a GPU MPS RFFT with a fixed physical width.
- Gas plus lattice-DM gravity uses the MPS RFFT and midpoint density source.

## Required convergence matrix

1. Fixed physical mode: halve timestep at N64 and N128.
2. Fixed physical cutoff: compare resolutions with at least 32 cells per
   shortest B wavelength and at least 16 per shortest quadratic-force
   wavelength for the first science gate.
3. Cumulative UV levels: retain identical low modes while extending the cutoff;
   compare spectra in every overlap interval.
4. Start redshift: compare progressively earlier starts at common output
   redshifts. Certify only when B, gas/DM, photon, and chemistry observables
   meet the declared tolerance. Otherwise the initialized handoff is still
   missing pre-start evolution.
5. Chemistry: compare half-step size, network cap, and smoothing width through
   recombination; gate xHII, temperature, and density spectra.
6. Energy: report sampled gas kinetic/internal/magnetic/total changes and
   magnetic divergence. Track resolved gas-plus-photon perturbation energy;
   angular phase-mixing losses remain unresolved background heat.

The fixed-spectrum launcher and checker live in `vespa-runs`:

```sh
configs/run_pmf_fixed_spectrum_certification_m5.sh run
python3 tools/pmf_certify/certify_fixed_spectrum.py \
  analysis/20260731-pmf-fixed-spectrum-certification
```

The short built-in Metal gate is:

```sh
julia --project=lib/MHDKernels/test \
  lib/MHDKernels/test/radiation_production_gate.jl
```

It validates compilation, fixed ICs, positivity, divergence, and short spatial
consistency. It does not replace the long start-redshift and UV-band campaign.
