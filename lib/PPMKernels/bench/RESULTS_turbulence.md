# DecayingTurbulence solver throughput — RK2 / Hancock / PPM / PPML, CPU + GPU

Recorded on Apple Silicon (Metal GPU present), `bench/bench_turbulence.jl 32 64 128 256`.
IC: solenoidal (divergence-free) random-mode velocity at RMS Mach 0.3, uniform ρ/P
(c_s=1); NG=4 ghosts. Metric = active cells updated per second-per-step (Mcell/s);
the scratch pool recycles per-sweep buffers across steps (warm-up step primes it).

Five solvers timed:
- **PPM** = `ppm_step_3d!` — the full PPM DirectEuler pipeline (Enzo HydroMethod=0 port).
- **RK2** = `muscl_step_3d!` — unsplit RK2 MUSCL (Enzo HydroMethod=3 / HD_RK class).
- **Hancock** = `muscl_hancock_step_3d!` (recon=:plm) — dim-split, 3 sweeps/step, PLM.
- **Hancock-PPM** = `muscl_hancock_step_3d!` (recon=:ppm) — same predictor+HLL, parabolic recon.
- **PPML** = `ppml_step_3d!` — PPM-on-a-Local-stencil (Ustyugov+ 2009): a STATEFUL
  characteristic-traced solver (persistent face pair + RGK limiter + CW84 monotonize +
  shock flatten + characteristic-trace predictor + HLL-with-star corrector). 9 kernels/
  sweep + 30 grid-arrays of persistent state — by far the heaviest per-cell scheme.

## Throughput (Mcell/s — higher is better)

| Solver        | 32³ f64 | 32³ f32 | 32³ Metal | 64³ f64 | 64³ f32 | 64³ Metal | 128³ f64 | 128³ f32 | 128³ Metal | 256³ f32 | 256³ Metal |
|---------------|--------:|--------:|----------:|--------:|--------:|----------:|---------:|---------:|-----------:|---------:|-----------:|
| PPM (DirectEuler) | 0.85 | 1.27 |  2.73 | 1.43 | 1.78 | 19.79 | 1.62 | 1.78 |  69.52 | 2.26 |  72.20 |
| RK2 (unsplit)     | 3.02 | 3.22 |  2.15 | 3.75 | 4.13 | 10.95 | 4.36 | 4.60 |  42.41 | 4.54 |  96.51 |
| Hancock (PLM)     | 5.00 | 5.14 | 10.61 | 6.06 | 6.75 | 63.93 | 7.12 | 7.50 | 189.34 | 7.40 | 233.12 |
| Hancock-PPM       | 2.11 | 2.14 |  9.12 | 2.72 | 2.81 | 78.75 | 3.05 | 3.16 | 174.83 | 3.04 | 197.16 |
| PPML (Ustyugov)   | 1.40 | 1.36 |  4.74 | 1.76 | 1.87 | 23.48 | 2.10 | 2.14 |  69.20 | 2.25 |  76.93 |

256³ CPU-f64 is omitted: 16.8M cells × the ~100-buffer scratch pool exceeds the f64
memory cap (`CPU_F64_MAX = 150³`), so the bench auto-skips it. (Raw legacy Fortran
1-D PPM pencil kernel, single-thread: 7.25 / 9.21 / 10.05 / 9.08 Mcell/s at 32/64/128/256.)

## sec/step (lower is better) — the raw measurement

| Solver        | 32³ f64 | 32³ f32 | 32³ Metal | 64³ f64 | 64³ f32 | 64³ Metal | 128³ f64 | 128³ f32 | 128³ Metal | 256³ f32 | 256³ Metal |
|---------------|--------:|--------:|----------:|--------:|--------:|----------:|---------:|---------:|-----------:|---------:|-----------:|
| PPM           | 0.03867 | 0.02571 | 0.01202 | 0.1831 | 0.1473 | 0.01325 | 1.294  | 1.177  | 0.03017 | 7.425 | 0.2324  |
| RK2           | 0.01084 | 0.01019 | 0.01525 | 0.06988| 0.0635 | 0.02395 | 0.4811 | 0.4558 | 0.04945 | 3.693 | 0.1738  |
| Hancock (PLM) | 0.006549| 0.006372| 0.003087| 0.04326| 0.03885| 0.004101| 0.2946 | 0.2798 | 0.01108 | 2.268 | 0.07197 |
| Hancock-PPM   | 0.01554 | 0.01534 | 0.003593| 0.09655| 0.09343| 0.003329| 0.6866 | 0.6646 | 0.012   | 5.518 | 0.0851  |
| PPML          | 0.02346 | 0.02415 | 0.006911| 0.1492 | 0.1403 | 0.01116 | 0.9969 | 0.9807 | 0.0303  | 7.471 | 0.2181  |

## Accuracy — DecayingTurbulence, 64³, Mach₀ = 1.0, γ = 1.4, evolved to t = 0.4 (Metal f32)

`bench/run_decaying_turbulence.jl 64 <solver> 1.0 0.4 [recon]`. Decaying (no forcing),
triply-periodic; KE→IE via numerical + shock dissipation. All conserve mass and total
energy to the f32 floor; the discriminator is how much KE they dissipate (less = sharper,
less numerically diffusive) over the same physical time.

| Solver       | KE dissipated | Δmass/M | Δenergy/E |
|--------------|--------------:|--------:|----------:|
| Hancock-PPM  | **34.2 %**    | 1.7e-9  | 2.5e-9    |
| Hancock-PLM  | 40.7 %        | 6.6e-10 | 4.8e-10   |
| PPML         | 55.0 %        | 3.4e-10 | 2.1e-9    |

**Reading**: Hancock-PPM is the least diffusive (parabolic reconstruction preserves the
most small-scale KE); PLM is the middle; **PPML is the MOST dissipative at moderate Mach**.
This is a genuine, expected signature of our *faithful* PPML port: the RGK two-stage
characteristic limiter + CW84 monotonize + the 5-point shock flattener form an aggressive
limiter stack, and we pair it with the diffusive **HLL** Riemann solver (the reference Rust
uses contact-resolving **HLLC**). The Rust itself notes the flattener costs ~10–11 % extra
KE dissipation in M~2 turbulence. So PPML's value is its method (stateful characteristic
tracing + the local face-pair), not raw low-Mach sharpness; the clearest accuracy upgrade
would be swapping HLL→HLLC in the flux (a documented faithful refinement).

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
  underutilize it) — Hancock-PLM metal/cpu-f32 = 2.1× (32³) → 9.5× (64³) → 25× (128³)
  → **31× (256³, 233 Mcell/s peak)**.
- **At 256³ the GPU ordering shifts as occupancy saturates**: RK2 on Metal (96.5
  Mcell/s) overtakes PPM-DirectEuler (72.2) — the reverse of the 32³ result — since
  RK2's larger per-launch work amortizes the launch overhead once the grid is huge.
  Hancock-PPM's parabola recompute also starts to cost (~15% under PLM: 197 vs 233),
  no longer free as it was at 64³–128³.
- **PPML is the heaviest solver everywhere** — ~2.1 Mcell/s CPU-f32, 69 / 77 Mcell/s
  Metal at 128³/256³ (~2.8× slower than Hancock-PLM, on par with PPM-DirectEuler at
  scale). The cost is structural: 9 kernels/sweep (predictor → HLL flux+star → conserved
  update → corrector), the RGK + CW84 + flatten limiter stack, and a persistent face-pair
  state read/written each sweep. The point of PPML is the *method* (stateful
  characteristic tracing), not throughput — for raw speed Hancock-PLM wins; for low-Mach
  sharpness Hancock-PPM wins (see accuracy table above).
- **Optimization — per-axis transposed face-pair storage (+42–46% on Metal at scale).**
  The face pair (30 grid-arrays) is only ever touched in its own axis's sweep, so it is
  stored *permanently in that axis's transposed frame* (velocities pre-rotated to
  normal/transverse roles) instead of lab-frame. The y/z sweeps then read/write it
  directly — eliminating 40 full-grid gather passes per step (10 transpose-in + 10
  transpose-out × 2 non-x sweeps). Bit-identical results; Metal 128³ 48.6→69.2, 256³
  52.7→76.9; CPU +14–21%. (64³ is flat — small grids are launch/compute-bound, not
  transpose-bandwidth-bound.) Further fusion of predictor+Riemann into one kernel is
  blocked by Metal's 31-buffer cap (the fused kernel needs 26 arrays + scalars > 31).
