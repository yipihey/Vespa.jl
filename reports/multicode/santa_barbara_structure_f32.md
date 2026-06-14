# Santa Barbara f32 structure-formation gate

Command:

```sh
ENZOMODULES_GRID_LIB=/Users/tabel/Projects/enzo-dev/EnzoModules/deps/libenzomodules_grid_f32.dylib \
BACKEND=metal \
~/.julia/juliaup/julia-1.12.6+0.aarch64.apple.darwin14/bin/julia \
  --project=lib/PPMKernels/test lib/EnzoLib/examples/sb_compare.jl 6
```

Both configurations start from the same Santa Barbara ICs through the f32 Enzo
bridge.  The reference uses native Enzo PPM + FFT gravity; the candidate uses
EnzoNG PPMKernels/PoissonKernels on Metal-f32.

| quantity | value |
|---|---:|
| cycles | 6 |
| final time, reference | 0.933129 |
| final time, EnzoNG-Metal | 0.933129 |
| relative time difference | 0.00e+00 |
| reference median cycle | 4139.7 ms |
| EnzoNG-Metal median cycle | 500.2 ms |
| total speedup | 8.28 |
| gravity speedup | 4.10 |
| hydro speedup | 15.18 |

| field | relL2 | Linf | reference max norm |
|---|---:|---:|---:|
| rho | 3.766e-04 | 2.263e-04 | 1.002e-01 |
| v1 | 1.331e-01 | 1.071e-04 | 6.222e-04 |
| v2 | 1.319e-01 | 1.250e-04 | 6.017e-04 |
| v3 | 1.381e-01 | 1.224e-04 | 6.235e-04 |
| TE | 2.675e-04 | 2.753e-07 | 1.821e-04 |
| GE | 2.512e-04 | 2.724e-07 | 1.820e-04 |

This is the short structure-formation science gate: EnzoNG-Metal and the native
enzo-f32 reference reach the same epoch from the same cosmological ICs, with
rho/energy parity at the expected f32 SB floor and velocity relative errors
dominated by the very small absolute velocity scale.
