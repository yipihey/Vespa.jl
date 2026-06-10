# Two RT methods, one density field (ADR-0006 Phase 4)

Iliev Test 1: uniform n_H = 1e-3 cm⁻³ hydrogen at 1e4 K, a 5e48 photons/s monochromatic 13.6 eV source, 6.6 kpc box.  Enzo **Moray** (adaptive ray tracing, 32³) vs **RAMSES-RT** (M1 moment method, reduced c = 0.005c, 64³), each through its CodeBridge wrapper, vs the analytic Strömgren front r(t) = R_s·(1−e^{−t/t_rec})^{1/3}.

| t [Myr] | Moray r_I | RAMSES-RT r_I | analytic | Moray/exact | RAMSES-RT/exact | code/code |
|---------|-----------|---------------|----------|-------------|-----------------|-----------|
| 3.0 | 0.2322 | 0.2185 | 0.2364 | 0.982 | 0.924 | 1.063 |
| 5.0 | 0.2795 | 0.2654 | 0.2795 | 1.000 | 0.949 | 1.053 |

Radii in box units (box = 6.6 kpc).  The M1 front lags a few percent at early times (reduced speed of light + the discrete first cell) and converges onto the ray-traced and analytic fronts — the behaviour the Iliev et al. (2006) comparison project documents for these method families.
