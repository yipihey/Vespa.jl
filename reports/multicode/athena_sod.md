# Athena++ in the Sod harness (wrapper-registry on-ramp)

The stock `athinput.sod` (γ = 1.4, interface recentred to x = 0.5) run IN-PROCESS through AthenaLib via the `MultiCodeAthenaExt` package extension, against the same exact-Riemann oracle as the other engines.

| engine | cells | wall-clock [s] | L1(ρ) | L1(u) | mass drift |
|--------|-------|----------------|-------|-------|------------|
| athena++ (:hydro) | 256 | 0.00 | 0.0019 | 0.0034 | 0.0e+00 |
