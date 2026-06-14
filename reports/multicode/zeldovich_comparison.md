# One particle set, two cosmology codes (ADR-0006)

Zel'dovich plane wave, EdS, ZERO initial velocities: the same 32³ lattice with ψ = 0.0325·sin(2πx) injected through both codes' particle bridges and evolved a_i → 4.0·a_i (z = 49.0 start).  Oracle: the closed-form mixed-mode growth b(x) = (3/5)x + (2/5)x^{-3/2}.

| engine | steps | wall-clock [s] | a/a_i | bA measured | bA exact | ratio | rms shape resid / A |
|--------|-------|----------------|-------|-------------|----------|-------|---------------------|
| enzo-pm | 71 | 0.79 | 4.052 | 0.07974 | 0.08060 | 0.9893 | 0.0241 |
| ramses-pm | 14 | 0.29 | 4.205 | 0.08302 | 0.08350 | 0.9942 | 0.0304 |

Enzo runs its CosmologySimulation machinery (PM gravity + comoving expansion via the certified EvolveLevel slots); RAMSES runs its supercomoving `amr_step` production loop (UNITS=COSMO build, grafic headers written from Julia).  Identical particles, zero shared code, one analytic answer.
