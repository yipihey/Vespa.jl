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
CIC_FVGK_STORE=f16
CIC_FVGK_DEDUP=1
CIC_CHEM=analytic
CIC_PACKED=1
CIC_PSORT=0
CIC_PIDS=0
CIC_VEL16=1
CIC_GRAVITY=cpu
CIC_GRAV_HOST32=1
CIC_FFT=ka
CIC_PK=1
CIC_CELL_DUMP=0
CIC_NODUMP=1
CIC_NGRID=1024
CIC_NP=1
CIC_BOX=0.8
CIC_STREAM_LOAD=1
CICASS_REAL_BYTES=4
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

## Lean-memory follow-up

Follow-up changes on 2026-07-01 removed four per-call transients:

- `fft_poisson_root!` and `fft_poisson_rfft!` now cache the complex rFFT buffer and use `mul!` for both transforms. On the root FFTW path, the inverse writes directly into the caller's `phi` host array when possible.
- CPU-gravity particle deposition now reuses a device density scratch plus host staging vector instead of allocating a full device density and `Float64.(to_host(...))` conversion every solve.
- Compton drag is now one fused per-cell KA kernel, eliminating the full-grid `ke0` and `ke1` broadcast temporaries.
- CFL signal-speed reduction now uses a fused block-max KA kernel, eliminating the full-grid `cs` and `sig` broadcast temporaries in `max_signal`. On the CPU-gravity path it borrows the prefix of the particle-density device/host scratch, so it does not allocate a separate reduction buffer.

Allocation probe after warmup at 128^3:

```text
fft_poisson_root! warmed allocated bytes = 224
fft_poisson_rfft! warmed allocated bytes = 80
```

The default hero script uses `CIC_FFT=ka`, so the FFTW `mul!` change does not explain the default 1024^3 cycle timing. It applies to `CIC_FFT=fftw` and to generic `fft_poisson_rfft!` users.

Rerunning the production-shaped 1024^3 one-cycle Metal/FVGK case after the scratch reuse, with the default KA FFT path:

```text
log                     = reports/multicode/perf/hero_metal1024_box0p8_lean_1cyc_20260701.log
real time               = 469.28 s
max resident set size   = 57,716,277,248 bytes = 53.76 GiB
peak memory footprint   = 231,871,636,056 bytes
live Metal              = 90.61 GiB
cycle wall              = 420.16 s
top-grid gravity        = 89.1096 s/solve
```

Compared with the prior one-cycle log's `62,154,162,176` byte max RSS, host resident memory dropped by `4,437,884,928` bytes, matching the removed host conversion transient. Live Metal memory stayed at `90.61 GiB` because the scratch is now persistent resident storage rather than per-solve allocation churn. No monotonic growth was observed in this one-cycle probe; a longer run is still the right gate for slow drift.

The IC realizer now links `libfftw3_threads` and reads `CICASS_FFT_THREADS`, then `FFTW_NUM_THREADS`, then `OMP_NUM_THREADS`. A 128^3 smoke with `CICASS_FFT_THREADS=4` printed:

```text
CICASS FFTW threads: 4
```

The IC realizer can also write f32 field snapshots with `CICASS_REAL_BYTES=4`.
The f32 magic is `CICASS02`; the legacy f64 format remains `CICASS01`. Header
metadata is still f64, while the 11 large field arrays are f32. A 128^3 f32 smoke
wrote:

```text
magic        = CICASS02
field eltype = Float32
file size    = 92,274,784 bytes
```

The corresponding f64 size at 128^3 is `184,549,472` bytes, so the f32 snapshot
is exactly half-size apart from the fixed 96-byte header. Extrapolated to 1024^3,
the `.cicass` IC file drops from about `88.0 GiB` to about `44.0 GiB`.

Post-merge `CICASS02` Metal smokes from
`reports/multicode/hero_ics/f32_smoke128_cicass02/f32_smoke128_cicass02.cicass`,
both with CPU FFTW gravity and no dumps:

```text
128^3 Metal FVGK f16 dedup, analytic chem, np=1, one cycle:
  mass drift      = 1.132e-06
  top-grid gravity = 0.6217 s/solve
  cycle wall       = 4.02 s

128^3 Metal PPM packed, analytic chem, np=2, one cycle:
  mass drift      = 0.000e+00
  top-grid gravity = 0.9166 s/solve
  cycle wall       = 7.40 s
```

## FVGK/particle allocation order fix

The 1024^3 transition peak was still too high because the stream-load path uploaded
the DM particle SoA before the FVGK pre-build/dedup. That made the peak:

```text
f32 patch gas + DM particles + FVGK g.R/g.O
```

The driver now pre-builds and dedups FVGK immediately after gas scatter and before
DM upload in the stream-load, restart, and in-memory snapshot paths. The real 1024^3
log order is now:

```text
streaming CICASS snapshot: .../cic_stream_1024_box0p8.cicass
gas IC ...
FVGK dedup ON: patch gas -> g.R views, ng=0, gesc=1e+07, f32 patch copy freed
DM IC: 1073741824 particles, mass_per=0.8430 (1-f_b), v->code=8.2245e-03
```

The synthetic 1024^3 memory probe for the current lean Metal configuration:

```text
BACKEND=metal CIC_NPROBE=1024 CIC_NP=1 CIC_PACKED=1 CIC_FVGK_F16=1
CIC_FVGK_STORE=f16 CIC_FVGK_DEDUP=1 CIC_VEL16=1 CIC_GRAV1BUF=1 CIC_PIDS=0

1024^3 (1073.7 Mcell) fields=54.00 GiB (live≈32.00)
  patch=0.00 GiB (0 B/c)
  fvgk =32.00 GiB (32 B/c)
  part =18.00 GiB (18 B/c)
  grav = 4.00 GiB (4 B/c)
  sum  =54 B/c

real time             = 20.05 s
max resident set size = 1,339,457,536 bytes
peak footprint        = 76,898,748,312 bytes
```

The component sum is the reliable persistent-array budget. Metal's cumulative
allocator stats can under-report or over-report the live set after large frees; use
the per-array sum for the byte/cell model.

The peak reduction in this probe comes from constructing the Metal DE16 FVGK grid
empty, then gathering the patch IC into `g.U` directly. The old constructor first
materialized a full host `Array{NTuple}` and a full host Float16 staging cube before
uploading to Metal.

Production-shaped 1024^3 one-cycle validation after the reorder, with `CIC_PK=0`
and full dumps disabled so the timing isolates startup plus one step:

```text
CIC_TAG=metal1024_peakfix_host32_1cyc_20260701
cycle 0 wall          = 91.66 s
phase total           = 87.1681 s/cycle
mass drift            = 8.746e-08
live Metal            = 50.06 GiB at 1024^3
real time             = 138.80 s
max resident set size = 39,630,602,240 bytes
peak footprint        = 114,024,757,560 bytes
```

GPU-synchronized phase breakdown for that validation run:

```text
gravity   = 53.4501 s
hydro     =  2.1641 s
chem      =  0.5452 s
particles = 31.0087 s

top-grid gravity = 52.9425 s/solve
  assemble    =  3.8523 s
  FFT         = 49.0083 s
  patch_accel =  0.0819 s
  part_field  =  0.0000 s
```

That run confirms the lifecycle fix on hardware: the FVGK dedup happens before the
13+ GiB particle upload, so the transition peak is patch gas plus FVGK buffers,
then particles arrive after the patch gas copy is gone. A fused output-summary
reduction was also added so `CIC_CELL_DUMP=0` summaries no longer materialize
decoded species, ionization, molecular-weight, and temperature arrays at each
output.

Additional peak cuts in the current run:

- CPU-gravity host density and potential arrays use `CIC_GRAV_HOST32=1`, saving
  one Float32 cube each relative to the prior Float64 host path.
- Dedup CPU gravity reuses the global device potential for particle force
  interpolation (`nc` periodic-wrap path), eliminating the padded particle
  potential copy; `part_field` is now zero.
- The f32 CICASS02 DM stream aliases the raw f32 field as the position conversion
  buffer and only allocates a separate Float16 velocity buffer when `CIC_VEL16=1`.
- Packed gas species are filled directly on device for the uniform initial HII
  fraction instead of staging a full host UInt16 cube.

Post-fusion 128^3 Metal compile smokes:

```text
FVGK f16 dedup, np=1, CIC_CELL_DUMP=0:
  cycle wall      = 3.50 s
  mass drift      = 1.132e-06
  top-grid gravity = 0.9519 s/solve

PPM packed UInt16 species, np=2, CIC_CELL_DUMP=0:
  cycle wall      = 7.51 s
  mass drift      = 0.000e+00
  top-grid gravity = 1.3498 s/solve
```

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
