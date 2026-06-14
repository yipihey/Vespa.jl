# Enzo vs RAMSES — detailed comparison (CICASS z=1000→20, z=20 snapshot)

Identical CICASS ICs (seed 113334, verified bit-identical), matched CFL (Courant 0.8,
da/a 0.1), f32, GPU both, same ChemistryKernels Julia-grackle (`engine=:kernels`).
Per-cell physical dumps: `enzo_phase_z20.bin` / `ramses_phase_z20.bin`
(`ρ/ρ̄, n_H[cm⁻³], T[K], f_H2=2n_H2/n_H, x_HII`); curves in `enzo_vs_ramses_phase_z20.csv`.

## 1. Power spectra — AGREE on large scales, diverge on small
DM:   ratio Enzo/RAMSES = 1.13 (k=70) → 1.04 (k≈950) → 0.68 (k≈2700).
Baryon: 1.13 (k=70) → 0.70 (k≈510) → 0.06 (k≈2700).
Large-scale (gravity-driven) growth matches to ~13%. Small-scale baryon power is
strongly damped in Enzo — the one-ghost HLL **LocalPPM is more diffusive** than
RAMSES's MUSCL (expected solver tradeoff; DM, which is PM-gravity only, agrees far better).

## 2. Gas density PDF — EXCELLENT agreement
PDF of log10(ρ_gas/ρ̄): both peak at −0.075, L1 distance 0.018; δ_gas rms 0.390 (Enzo)
vs 0.409 (RAMSES). The density field statistics are essentially identical.

## 3. Temperature–density — DISAGREE (thermal coupling differs)
mean T: Enzo 0.38 K, RAMSES 0.26 K (both far below the z=20 CMB, 57 K — both strongly
over-cool the mean gas). But the **T–ρ relation is qualitatively different**:
| ρ/ρ̄ | T_Enzo [K] | T_RAMSES [K] |
|---|---|---|
| ~1 | ~0 (floored) | ~0.3 |
| ~5 | 30 | 0.5 |
| ~10 | 43 | 0.4 |
Enzo **adiabatically heats collapsing gas** (T rises with ρ), RAMSES's dense gas stays
cold (~0.5 K). Also Enzo's low-density gas energy is floored to ~0 (median T≈0). So the
two codes do NOT agree on the thermal state, despite identical ICs/CFL/chem-solver.

## 4. H₂ fraction–density — DISAGREE by ~75×
mean f_H2: Enzo 3.2e-6, RAMSES 2.4e-4 (RAMSES ~75× higher), at all densities. Since
BOTH use the same ChemistryKernels grackle, this is driven by the divergent **T and x_HII
histories**, not the chem solver: x_HII Enzo 3.4e-4 vs RAMSES 1.1e-4 (3× higher in Enzo).
H₂ forms via the H⁻ channel (T- and x_e-sensitive); the different thermal/ionization
evolution feeds the chem kernel different inputs → very different H₂.

## Bottom line
- **Gravity / density structure: consistent** (P(k) large-scale 13%, PDF excellent).
- **Hydro small-scale: LocalPPM more diffusive** (known tradeoff; use full PPM for
  small-scale baryon power).
- **Thermodynamics + chemistry: NOT consistent** — the temperature evolution diverges
  (Enzo heats dense gas + floors low-density T≈0; both over-cool the mean), and that
  propagates to a ~75× H₂ discrepancy through the shared chem kernel. The cooling/heating
  COUPLING into the chem step (how each code passes/updates internal energy, and the
  energy floor) is the suspect and needs investigation before the chemistry can be
  trusted to agree.
