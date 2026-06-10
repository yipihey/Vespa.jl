# MUSIC injector validation (ADR-0006 wrapper on-ramp)

ONE `MusicSpec` realization (identical seeds), generated in-process in two formats; Enzo booted on the generated `parameter_file.txt` + particle ICs, RAMSES (UNITS=COSMO) on the grafic2 level directory; the initial CIC density contrasts compared with no evolution.

| n | corr(δ_E, δ_R) | rms(δ_E−δ_R)/σ | σ_E | σ_R |
|---|----------------|----------------|-----|-----|
| 32³ | 0.999999999999996 | 1.03e-07 | 0.080386 | 0.080386 |

The residual is the float32 precision of the grafic planes — the two injection chains are otherwise identical.
