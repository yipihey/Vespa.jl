# AREPO Direct Performance Study

This is a first direct local performance matrix comparing the original AREPO
decaying-turbulence HLL binary against the PowerFoam KernelAbstractions rewrite.

The goal is timing attribution, not a physics-equivalence claim. AREPO is running
its real N12 HDF5 IC and adaptive sync loop to `TimeMax=0.0625`; PowerFoam is
running the N12 synthetic periodic decaying-turbulence compact moving-mesh gate
for 8 fixed steps. The comparison is still useful because both are 3-D moving
mesh HLL turbulence-like workloads at the same cell count.

## Fresh N12 Results

| Case | Backend / ranks | Elapsed s | Per interval s | Relative to AREPO serial |
| --- | --- | ---: | ---: | ---: |
| AREPO HLL | 1 MPI rank | 0.795056 | 0.099382 | 1.00x |
| AREPO HLL | 4 MPI ranks | 0.346238 | 0.043280 | 2.30x |
| PowerFoam KA HLL | CPU Float32 | 0.0930996 | 0.0116375 | 8.54x |
| PowerFoam KA HLL | Metal Float32 | 0.0779346 | 0.00974182 | 10.20x |

At N12, the GPU is only modestly faster than KA CPU (`1.19x`) because the grid is
too small to amortize GPU launch and transfer overheads. The stronger result is
that the KA compact local rebuild is already much faster than original AREPO's
full Voronoi rebuild path on this laptop-sized gate.

## Commands

AREPO serial:

```bash
cd lib/PowerFoam/examples/out/arepo_direct_perf_study/arepo_n12_serial_hll
./Arepo param.txt
```

AREPO MPI:

```bash
cd lib/PowerFoam/examples/out/arepo_direct_perf_study/arepo_n12_mpi4_hll
mpiexec -np 4 ./Arepo param.txt
```

PowerFoam KA CPU/Metal:

```bash
env JULIA_NUM_THREADS=4 \
JULIA_DEPOT_PATH=/private/tmp/enzo_powerfoam_depot:/Users/tabel/.julia \
JULIA_LOAD_PATH=@:lib/PowerFoam:@stdlib \
POWERFOAM_PERF_WARMUP=true \
POWERFOAM_REBUILD=gpu_compact \
/Users/tabel/.julia/juliaup/julia-1.12.6+0.aarch64.apple.darwin14/bin/julia \
--project=lib/MultiCode/test \
lib/PowerFoam/examples/native_moving_solver_matrix_3d.jl \
12 0.001 8 hll 1 reconstruct
```

## Evidence

- AREPO serial: `Code run for 0.795056 seconds!`
- AREPO MPI4: `Code run for 0.346238 seconds!`
- PowerFoam:
  `lib/PowerFoam/examples/out/native_moving_solver_matrix_3d/N12_dt0p001_n8_r1_reconstruct_gpu_compact_hll/solver_summary.csv`

AREPO CPU timers at final step:

| Case | Total timer s | Voronoi cumulative | Hydro cumulative |
| --- | ---: | ---: | ---: |
| 1 MPI rank | 0.80 | 0.72 | 0.03 |
| 4 MPI ranks | 0.35 | 0.30 | 0.01 |

The AREPO timing is dominated by Voronoi construction: about 90% of the serial
runtime and 86% of the 4-rank runtime in `output/cpu.txt`.

## Caveats

- AREPO is double precision, original C/MPI, and writes a final snapshot/restart.
  PowerFoam is Float32 and the benchmark driver does not write snapshots.
- AREPO uses its real N12 IC and adaptive sync loop. PowerFoam uses a synthetic
  periodic turbulence initialization and fixed steps.
- A 128-step PowerFoam N12 run with the current fixed-step driver hit a negative
  pressure stability failure, so long physical-time parity needs a smaller `dt`
  or a positivity/stability gate before it is a fair headline benchmark.
- Larger grids are where the Metal path matters. Existing N24 PowerFoam clean
  artifacts show Metal around 2x faster than KA CPU; this N12 direct matrix is
  intentionally small enough to run AREPO serial and MPI quickly.
