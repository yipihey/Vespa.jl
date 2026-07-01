# CICASS Metal hero performance note

Date: 2026-07-01

Host:
- Apple Silicon Mac17,6, 18 CPU threads, 128 GiB unified memory
- macOS 26.5.1 build 25F80
- Julia 1.12.6 via juliaup

Code state:
- Vespa.jl branch `chem-interior-only`
- FiniteVolumeGodunovKA.jl `main` with Metal DE16 validation fixes
- Driver: `lib/MultiCode/examples/patch_cicass.jl`
- Launch script: `lib/MultiCode/examples/run_cicass_hero_metal1024.sh`

## Configuration

Production-shaped 1024^3 Metal/FVGK f16 packed run:

```bash
CIC_TAG=hero_metal1024_box0p8_1cyc_20260701 \
CIC_MAXCYC=1 \
CIC_PRINT_EVERY=1 \
CIC_MEMPROBE=1 \
CIC_MEMPROBE_CYC=0 \
CIC_PHASE_TIMING=1 \
/usr/bin/time -l lib/MultiCode/examples/run_cicass_hero_metal1024.sh
```

The launch script pins:

```bash
BACKEND=metal
CIC_SOLVER=fvgk
CIC_FVGK_F16=1
CIC_CHEM=analytic
CIC_PACKED=1
CIC_PSORT=0
CIC_PK=1
CIC_NGRID=1024
CIC_NP=1
CIC_BOX=0.8
CIC_STREAM_LOAD=1
```

Staged IC:

```text
reports/multicode/hero_ics/metal_1024_box0p8/cic_stream_1024_box0p8.cicass
```

Size: 94,489,280,608 bytes, reported by `ls -lh` as 88G. The 0.8 Mpc/h box was used because the earlier 1024^3, 0.4 Mpc/h IC generation emitted high-k out-of-range warnings and produced invalid initial statistics.

Raw benchmark artifacts are under ignored `reports/` paths:

```text
reports/multicode/perf/hero_metal1024_box0p8_1cyc_20260701.log
reports/multicode/perf/hero_metal1024_box0p8_1cyc_20260701_pkmu.h5
```

## 1024^3 one-cycle result

The run completed cleanly for one full cycle and wrote the z=1000 `P(k,mu)` table.

Initial z=1000 summary:

```text
delta_b_rms = 2.623e-04
D^2/D0^2    = 1.000e+00
<x_HII>     = 6.243e-02
<T>         = 2577.4 K
```

After cycle 0:

```text
cycle       = 0
a           = 0.00102
z           = 978.225
delta_b_rms = 4.259e-05
rho_max     = 0.171
cycle wall  = 395.28 s
mass drift  = 9.481e-05
```

`/usr/bin/time -l`:

```text
real time               = 449.68 s
user time               = 489.00 s
sys time                = 121.21 s
max resident set size   = 62,154,162,176 bytes = 57.89 GiB
peak memory footprint   = 223,279,568,496 bytes = 207.96 GiB (macOS footprint accounting)
```

Metal allocator probe at the end of cycle 0:

```text
live Metal = 90.61 GiB at 1024^3 = 90.61 B/cell
alloc      = 438.05 GiB cumulative
freed      = 347.44 GiB cumulative
```

The one-cycle process wall includes streamed setup, initial z=1000 output, on-device `P(k,mu)`, and one complete step. The logged 395.28 s cycle wall includes initial output/PK work before the first step.

GPU-synchronized phase timing for the step body:

```text
gravity   = 74.4074 s
hydro     = 49.7977 s
chem      = 0.9001 s
particles = 67.4630 s
total     = 192.5682 s/cycle
```

Phase shares:

```text
gravity   = 39%
hydro     = 26%
chem      = 0%
particles = 35%
```

The raw log was produced before the final reporting-label fix and prints `PER-PHASE (CUDA-synced)` plus an aggregate `top-grid gravity = 192.5682 s/solve`. Treat that aggregate as mislabeled for this run. The corrected CPU top-grid gravity estimate is the component sum:

```text
assemble    = 9.3338 s
FFT         = 57.5147 s
patch_accel = 1.0500 s
part_field  = 6.1333 s
total       = 74.0318 s/solve
```

## Earlier scale probes

Full f16 stack memory probes before the streamed 1024^3 run:

| grid | FFT path | live Metal | cells | bytes/cell |
|---:|---|---:|---:|---:|
| 384^3 | FFTW | 4.41 GiB | 56.6M | 83.6 |
| 512^3 | KA radix-2 | 10.40 GiB | 134.2M | 83.2 |
| 1024^3 | KA radix-2 | 90.61 GiB | 1073.7M | 90.61 |

The 1024^3 live Metal footprint is higher than the 512^3 projection, but it still fits in 128 GiB unified memory for the one-cycle benchmark. Host and Metal memory are not additive in a simple way on unified memory, but the benchmark did complete at this size without swaps.

## Validation context

Metal kernel gates completed before this benchmark:

```text
metal_selfcheck_de16()         passed
metal_selfcheck_de16_colors()  passed
metal_selfcheck_3d()           max delta 0.0
metal_selfcheck_3d_colors()    max delta 0.0
```

End-to-end validation runs:

```text
128^3 Metal PPM f32, analytic chem, z=1000->20:
  delta_b_rms(z=20) = 4.058e-01
  D^2/D0^2          = 2.274e+03
  <x_HII>           = 2.161e-04
  mass drift        = 3.280e-08

256^3 Metal FVGK DE16, analytic chem, z=1000->20:
  delta_b_rms(z=20) = 4.461e-01
  D^2/D0^2          = 2.272e+03
  <x_HII>           = 2.161e-04
  mass drift        = 1.834e-04
```

The 256^3 FVGK result is below the CUDA reference quoted for 256^3 f16-DE (`delta_b_rms` about 4.615e-01, `<x_HII>` about 2.155e-04). The chemistry agrees at the quoted precision; the density growth is the remaining mismatch to keep tracking.

## Caveats and next run

This is a completed 1024^3 startup plus one-cycle performance benchmark, not a completed z=20 hero science run.

An earlier unbounded 1024^3 launch reached the same valid streamed load, z=1000 `P(k,mu)`, and cycle 0 (`372.59 s`) before the interactive tool run was interrupted. No Julia or wrapper process remained afterward. There was no Julia crash report for that launch.

Recommended next command for a longer unattended run:

```bash
CIC_TAG=hero_metal1024_box0p8_z20_20260701 \
CIC_PRINT_EVERY=1 \
CIC_MEMPROBE=1 \
CIC_MEMPROBE_CYC=0 \
/usr/bin/time -l lib/MultiCode/examples/run_cicass_hero_metal1024.sh \
  2>&1 | tee reports/multicode/perf/hero_metal1024_box0p8_z20_20260701.log
```

Use `CIC_PHASE_TIMING=1` only for a breakdown run. It adds GPU synchronization barriers and should not be used as the quoted production throughput.
