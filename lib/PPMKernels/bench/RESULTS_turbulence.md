# DecayingTurbulence solver throughput — RK2 / Hancock / PPM, CPU + GPU

Recorded on Apple Silicon (Metal GPU present), `bench/bench_turbulence.jl 32 64 128`.
IC: solenoidal (divergence-free) random-mode velocity at RMS Mach 0.3, uniform ρ/P
(c_s=1); NG=4 ghosts. Metric = active cells updated per second-per-step (Mcell/s);
the scratch pool recycles per-sweep buffers across steps (warm-up step primes it).

Four solvers timed:
- **PPM** = `ppm_step_3d!` — the full PPM DirectEuler pipeline (Enzo HydroMethod=0 port).
- **RK2** = `muscl_step_3d!` — unsplit RK2 MUSCL (Enzo HydroMethod=3 / HD_RK class).
- **Hancock** = `muscl_hancock_step_3d!` (recon=:plm) — dim-split, 3 sweeps/step, PLM.
- **Hancock-PPM** = `muscl_hancock_step_3d!` (recon=:ppm) — same predictor+HLL, parabolic recon.

## Throughput (Mcell/s — higher is better)

| Solver        | 32³ f64 | 32³ f32 | 32³ Metal | 64³ f64 | 64³ f32 | 64³ Metal | 128³ f64 | 128³ f32 | 128³ Metal |
|---------------|--------:|--------:|----------:|--------:|--------:|----------:|---------:|---------:|-----------:|
| PPM (DirectEuler) | 0.85 | 1.27 |  2.73 | 1.43 | 1.78 | 19.79 | 1.62 | 1.78 |  69.52 |
| RK2 (unsplit)     | 3.02 | 3.22 |  2.15 | 3.75 | 4.13 | 10.95 | 4.36 | 4.60 |  42.41 |
| Hancock (PLM)     | 5.00 | 5.14 | 10.61 | 6.06 | 6.75 | 63.93 | 7.12 | 7.50 | 189.34 |
| Hancock-PPM       | 2.11 | 2.14 |  9.12 | 2.72 | 2.81 | 78.75 | 3.05 | 3.16 | 174.83 |

(For reference the raw legacy Fortran 1-D PPM pencil kernel ran at 7.25 / 9.21 /
10.05 Mcell/s single-thread at the three sizes.)

## sec/step (lower is better) — the raw measurement

| Solver        | 32³ f64 | 32³ f32 | 32³ Metal | 64³ f64 | 64³ f32 | 64³ Metal | 128³ f64 | 128³ f32 | 128³ Metal |
|---------------|--------:|--------:|----------:|--------:|--------:|----------:|---------:|---------:|-----------:|
| PPM           | 0.03867 | 0.02571 | 0.01202 | 0.1831 | 0.1473 | 0.01325 | 1.294  | 1.177  | 0.03017 |
| RK2           | 0.01084 | 0.01019 | 0.01525 | 0.06988| 0.0635 | 0.02395 | 0.4811 | 0.4558 | 0.04945 |
| Hancock (PLM) | 0.006549| 0.006372| 0.003087| 0.04326| 0.03885| 0.004101| 0.2946 | 0.2798 | 0.01108 |
| Hancock-PPM   | 0.01554 | 0.01534 | 0.003593| 0.09655| 0.09343| 0.003329| 0.6866 | 0.6646 | 0.012   |

## Findings

- **Hancock (PLM) is the throughput champion everywhere** — 7.5 Mcell/s CPU-f32 and
  **189 Mcell/s Metal** at 128³ (25× GPU speedup over CPU-f32). Its dim-split
  3-sweeps/step structure is the leanest on kernel launches.
- **PPM reconstruction is nearly free on the GPU, expensive on the CPU.** Hancock-PPM
  vs Hancock-PLM is ~2.4× slower on CPU (the per-cell parabola recompute is
  compute-bound) but within ~8% on Metal at 128³ (174 vs 189) and even faster at 64³
  (78.8 vs 63.9) — the extra arithmetic hides under memory bandwidth.
- **RK2 (unsplit) is the weakest on GPU** — 42 Mcell/s at 128³ vs Hancock's 189 (more
  global sweeps/launches per step); at 32³ it is even slower than PPM-DirectEuler on Metal.
- **PPM-DirectEuler scales hardest with size on GPU** (39× metal/cpu-f32 at 128³, the
  most arithmetically intense pipeline) but stays slowest in absolute terms except RK2.
- **The GPU needs to be fed**: Metal speedup grows with grid size (small grids
  underutilize it) — Hancock-PLM metal/cpu-f32 = 2.1× (32³) → 9.5× (64³) → 25× (128³).
