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
- **PPML** = `ppml_step_3d!` — PPM-on-a-Local-stencil (Ustyugov+ 2009): the FULL method —
  a STATEFUL characteristic-traced solver (persistent face pair + RGK characteristic
  limiter + CW84 monotonize + shock flatten + §6 WENO5 smooth-extremum fallback +
  characteristic-trace predictor + HLLC). 9 kernels/sweep + 30 grid-arrays of persistent
  state — by far the heaviest per-cell scheme.

## Throughput (Mcell/s — higher is better)

| Solver        | 32³ f64 | 32³ f32 | 32³ Metal | 64³ f64 | 64³ f32 | 64³ Metal | 128³ f64 | 128³ f32 | 128³ Metal | 256³ f32 | 256³ Metal |
|---------------|--------:|--------:|----------:|--------:|--------:|----------:|---------:|---------:|-----------:|---------:|-----------:|
| PPM (DirectEuler) | 0.85 | 1.27 |  2.73 | 1.43 | 1.78 | 19.79 | 1.62 | 1.78 |  69.52 | 2.26 |  72.20 |
| RK2 (unsplit)     | 3.02 | 3.22 |  2.15 | 3.75 | 4.13 | 10.95 | 4.36 | 4.60 |  42.41 | 4.54 |  96.51 |
| Hancock (PLM)     | 5.00 | 5.14 | 10.61 | 6.06 | 6.75 | 63.93 | 7.12 | 7.50 | 189.34 | 7.40 | 233.12 |
| Hancock-PPM       | 2.11 | 2.14 |  9.12 | 2.72 | 2.81 | 78.75 | 3.05 | 3.16 | 174.83 | 3.04 | 197.16 |
| PPML (Ustyugov)   | 1.19 | 1.24 |  4.47 | 1.50 | 1.63 | 34.34 | 1.78 | 1.85 |  65.50 | 1.93 |  74.87 |

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
| PPML          | 0.02755 | 0.02647 | 0.007333| 0.1745 | 0.1607 | 0.007634| 1.179  | 1.132  | 0.03202 | 8.701 | 0.2241  |

## Accuracy — DecayingTurbulence, 64³, Mach₀ = 1.0, γ = 1.4, evolved to t = 0.4 (Metal f32)

`bench/run_decaying_turbulence.jl 64 <solver> 1.0 0.4 [recon]`. Decaying (no forcing),
triply-periodic; KE→IE via numerical + shock dissipation. All conserve mass and total
energy to the f32 floor; the discriminator is how much KE they dissipate (less = sharper,
less numerically diffusive) over the same physical time.

| Solver       | KE dissipated | Δmass/M | Δenergy/E |
|--------------|--------------:|--------:|----------:|
| Hancock-PPM  | **34.2 %**    | 1.7e-9  | 2.5e-9    |
| Hancock-PLM  | 40.7 %        | 6.6e-10 | 4.8e-10   |
| PPML (HLL)            | 55.0 % | 3.4e-10 | 2.1e-9 |
| PPML (HLLC)           | 53.2 % | 2.0e-10 | 6.6e-10 |
| PPML (HLLC + WENO5)   | 52.9 % | 7.2e-10 | 8.2e-10 |

**Reading**: Hancock-PPM is the least diffusive (parabolic reconstruction preserves the
most small-scale KE); PLM is the middle; **PPML is the most dissipative at moderate Mach**.
This is a genuine signature of the *full* Ustyugov+ 2009 method: the RGK characteristic
limiter + CW84 monotonize + the shock flattener form an aggressive limiter stack. Upgrading
the Riemann solver HLL→HLLC (contact-resolving) shaves ~1.8 pts (55.0→53.2 %), and the
WENO5 smooth-extremum fallback another ~0.3 pt (53.2→52.9 %) — both small *here* because
M~1 compressible turbulence is shock-dominated (HLLC matters at strong contacts; the WENO5
smoothness test correctly does NOT fire at shocks). WENO5's real payoff is on SMOOTH flow:
a smooth entropy wave advected ~1 period retains ~0.5 % more amplitude with WENO5 than
without (the median limiter slowly erodes smooth extrema; WENO5 does not) — see
`test_ppml.jl`. PPML's value is its method (stateful characteristic tracing + local face
pair + extremum-preserving reconstruction), not raw low-Mach sharpness.

## Supersonic decaying turbulence — all solvers, GPU, half a crossing time

`bench/compare_turb_dissipation.jl 128 5 0.5` — solenoidal Mach₀=5 IC, evolved to
0.5·t_cross (t_cross = L/v_rms), 128³, Metal f32, dual energy on, a SINGLE fixed dt
shared by all solvers. Metric = FINAL state; a less-dissipative solver keeps a higher
final RMS Mach / v_rms. Now sweeps **9 solvers** (the two Riemann variants HLLC/two-shock
and the trace predictor added), ~280 s of GPU stepping for ~900 steps each.

| Solver | Mach_f | v_rms_f | KE diss % | Δmass/M | Mcell/s |
|---|--:|--:|--:|--:|--:|
| **Hancock-PPM-tr-2shk** | **1.452** | **3.244** | **62.4** | 4.6e-10 | 69.1 |
| Hancock-PPM-HLLC    | 1.443 | 3.234 | 62.6 | 1.7e-9  | 79.1 |
| Hancock-PPM-trace   | 1.430 | 3.200 | 63.5 | 7.4e-10 | 76.6 |
| Hancock-PPM         | 1.427 | 3.198 | 63.5 | 1.4e-9  | 79.5 |
| PPM-DirectEuler     | 1.420 | 3.195 | 63.5 | 2.7e-4† | 40.5 |
| Hancock-PLM         | 1.380 | 3.124 | 65.3 | 5.3e-10 | 90.4 |
| RK2 (PLM)           | 1.378 | 3.121 | 65.3 | 3.2e-9  | 67.4 |
| PPML-trace          | 1.254 | 2.937 | 69.0 | 4.6e-11 | 43.5 |
| PPML-Hancock        | 1.242 | 2.918 | 69.4 | 5.7e-10 | 47.8 |

†DirectEuler now uses the `bc!` inter-sweep periodic refill (added to `ppm_step_3d!`) and
the diagnostic reads internal energy from the conserved TOTAL energy (etot−½v²) — the same
footing as the others — so its Mach_f (1.420) now matches Hancock-PPM (1.427) exactly. A
small ~2.7e-4 mass drift remains (its wide PPM stencil + flattener are not perfectly
periodic-consistent at the seam, unlike the simple flux-form solvers' round-off
conservation); it does not affect the dissipation conclusion.

- **In shocks the Riemann solver is the lever, NOT the predictor — the exact mirror of the
  smooth wave.** Holding reconstruction (`:ppm`) fixed: swapping the predictor
  `:hancock`→`:trace` does *nothing* here (1.427/63.5 % → 1.430/63.5 %, identical), because
  the limiter clips the parabola so hard at Mach 5 that the trace's curvature term never
  survives. But upgrading the Riemann solver moves the needle: HLLC 1.443/62.6 % and
  two-shock 1.452/62.4 % are the two LEAST-dissipative solvers in the table — the contact/
  shock resolution is what retains turbulent KE. This is the precise inverse of the advected
  sound wave (predictor = decisive lever there, Riemann inert); the two levers cleanly
  separate by regime.
- **HLLC ≈ two-shock at a fraction of the cost.** HLLC (62.6 %, 79.1 Mcell/s) captures ~90 %
  of two-shock's accuracy gain (62.4 %) at near-zero overhead vs plain HLL (79.5), whereas
  the iterative two-shock star-state solve costs ~13 % throughput (69.1 Mcell/s). For
  shock-dominated flow HLLC is the better throughput/accuracy point; two-shock is the
  accuracy ceiling of the Hancock-PPM branch.

- **The ranking is robust at developed Mach-5 turbulence** (and matches the earlier,
  doubted Mach-1 result — it is NOT a startup artifact): **PPM-reconstruction (Hancock-PPM,
  DirectEuler) least dissipative → PLM → PPML most dissipative.** Resolution matters in the
  absolute (PLM loses 65 % of KE at 128³ vs 72 % at 48³ — finer grids dissipate less), but
  the order is unchanged.
- **Two independent PPM implementations agree**: Hancock-PPM (parabola + Hancock predictor)
  and the certified Enzo-port PPM-DirectEuler land on the SAME numbers (Mach_f 1.427/1.420,
  KE diss 63.5 % both), the least dissipative of all — a strong cross-check.
- **PPM reconstruction DOES help where the limiter is light**: the two PPM-reconstruction
  solvers (no RGK/CW84/flatten/WENO5 stack) beat both PLM and PPML. So "PPM beats PLM" holds.
- **PPML is the most dissipative** because its full Ustyugov limiter stack (RGK + CW84 +
  flattener + HLLC) is aggressive, and Mach-5 turbulence is SHOCK-dominated — the
  high-order machinery (parabola, WENO5) pays off in smooth flow, not at the shocks that
  dominate here. PPML trades supersonic sharpness for robustness; the predictor choice
  (trace vs Hancock) barely moves it (69.0 vs 69.4 %).
- **Affordable on the GPU**: ~21–47 s per solver at 128³ for ~900 steps (40–90 Mcell/s
  incl. the periodic BC fills + dual energy + host diagnostics; Hancock-PLM fastest at 90,
  the full PPM-DirectEuler/PPML pipelines slowest at ~40–48).

## Advected sound wave — smooth-flow accuracy + asymmetry (GPU)

`bench/compare_soundwave.jl 128 4 10 1e-3` — a small-amplitude (A=1e-3) RIGHT-going acoustic
eigenmode on a background translated at +1 sound speed (u₀=cₛ), so the wave travels at 2cₛ
and returns to its start every box crossing — the exact solution is a pure translation, so
any change is NUMERICAL. 128×8² (32 cells/λ), Metal f32, 10 box-translations (~4300 steps).
Measured from the discrete Fourier mode at the fundamental wavenumber: amplitude retention
|c_k|_f/|c_k|_0 (dissipation), phase error in wavelengths (dispersion), and harmonic
distortion √(Σ_{m≥2}|c_mk|²)/|c_k| (waveform ASYMMETRY — a pure sine has zero).

Amplitude is the PEAK-TO-PEAK ratio (the honest amplitude). NOTE: the Fourier-fundamental
|c_k| ratio is MISLEADING here — a sine that distorts into a square wave has a fundamental
4/π≈1.27× a sine of the same peak-to-peak, so PPML's squaring read as a spurious "amp 1.16–
1.21 > 1" / "anti-dissipation" in an earlier draft; it never grew (see `bench/plot_soundwave.jl`).

| Solver | amp (p2p) | phase err (λ) | distortion (→□) | wall (s) |
|---|--:|--:|--:|--:|
| RK2 (PLM)            | 0.41 | −0.075 | 0.13 | — |
| Hancock-PLM         | 0.60 | −0.021 | 0.12 | — |
| Hancock-PPM         | 0.674 | +0.024 | 0.095 | 23 |
| Hancock-PPM-2shk    | 0.676 | +0.023 | 0.090 | 24 |
| **Hancock-PPM-trace** | **0.821** | **−0.014** | **0.172** | **21** |
| Hancock-PPM-tr-2shk | 0.859 | −0.013 | 0.133 | 25 |
| PPM-DirectEuler     | 0.790 | −0.017 | 0.206 | 93 |
| PPML-trace          | 0.945 | −0.010 | 0.270 | 42 |

- **The amplitude ranking INVERTS vs supersonic turbulence.** On a SMOOTH wave the
  high-order reconstructions keep peak-to-peak far better: PLM is most dissipative
  (amp 0.41–0.60), PPM/PPML least (0.79–0.95). Expected — PPM/PPML are *built* for smooth
  accuracy; their aggressive limiter only dominates in shock-dominated flow.
- **The characteristic-trace predictor is the decisive lever — and it is TIME INTEGRATION,
  not reconstruction.** Holding reconstruction (`:ppm`) AND Riemann fixed and swapping ONLY
  the half-step predictor (`:hancock`→`:trace`) lifts amplitude retention 0.674→0.821 and
  flips/shrinks the phase error (+0.024→−0.014). The trace integrates the parabola's
  curvature over each wave's departure region; the Hancock half-step uses only the two
  endpoint faces and *drops* that curvature term — which is exactly the dissipation seen in
  the over-damped Hancock panels. With the trace, the entire Hancock-vs-DirectEuler gap
  closes (and overshoots).
- **Trace BEATS the full DirectEuler pipeline here, at ~4.5× lower cost**: 0.821 amp vs
  0.790, *lower* distortion (0.172 vs 0.206 — a cleaner, less flat-topped sine; see the
  panels in `plot_soundwave.jl`), and 21 s vs 93 s wall. The full PPM characteristic
  time-averaging buys nothing over the parabola-trace half-step on this smooth wave.
- **DON'T be misled by the Hancock panel's low distortion (0.095).** It is low only because
  the wave is so damped (amp 0.67) that it cannot steepen — a tiny wave can't square. At
  *matched* amplitude retention the trace is the cleaner waveform. The fair read is
  trace (0.821 / 0.172) vs DirectEuler (0.790 / 0.206), not trace vs the over-damped Hancock.
- **Two-shock Riemann barely moves the smooth wave** (0.674→0.676 on Hancock; 0.821→0.859
  with trace) — expected, an isentropic acoustic mode has no contact/strong shock for the
  exact Riemann solver to resolve better than HLL. Its small amplitude gain comes with a
  matching distortion change; not worth the cost here (worth it in shocks, see turbulence).
- **PPML still pays its retention by SQUARING the wave** — highest amplitude (0.945) but the
  MOST distortion (0.270), flat-topped peaks (clearest in the bottom-right panel). This is a
  limiter pathology (WENO5 smooth-extremum fallback not fully preventing extremum clipping on
  a long advected wave), a real defect to fix, not a virtue.
- **Phase error (dispersion) shrinks with order**: PLM −0.075 → Hancock-PLM −0.021 →
  trace −0.014 → PPML −0.010 λ.
- **Verdict on smooth flow**: `Hancock-PPM-trace` is the new sweet spot — DirectEuler-class
  (better) accuracy with the cheap eigen-free Hancock skeleton plus only the parabola-trace
  curvature term. `recon=:ppm, predictor=:trace` is the recommended faithful-accuracy default.

## Predictor study — characteristic trace vs Hancock half-step (PPML)

PPML factors into a stateful **reconstruction** (carried face pair + RGK + CW84 + flatten +
WENO5) and a **predictor** that time-centres the limited face pair. `predictor=:trace`
(default, full Ustyugov) integrates the whole parabola over each wave's departure region;
`predictor=:hancock` uses the MUSCL-Hancock half-step on the same limited pair (only the two
endpoint faces → drops the parabola-curvature term; eigen-free, system-agnostic). A
three-way comparison (holding one factor fixed at a time) isolates what each piece buys:

| Variant | recon | predictor | 64³ M~1 turb (KE diss) | smooth wave (amp kept) |
|---|---|---|--:|--:|
| Hancock-PPM   | stencil parabola | Hancock | 36.0 % | 97.09 % |
| PPML-Hancock  | carried pair + RGK/WENO5 | Hancock | 55.9 % | 98.24 % |
| PPML-trace    | carried pair + RGK/WENO5 | trace | 54.2 % | 98.98 % |

- **The reconstruction drives most of the behaviour.** Swapping Hancock-PPM's stencil
  reconstruction for PPML's carried-pair+RGK+WENO5 (same Hancock predictor) flips the
  character: far more turbulence dissipation (36→56 %, the aggressive limiter stack) but
  better smooth-extremum preservation (97.1→98.2 %, the carried pair + WENO5). PPML *is*
  its reconstruction.
- **The characteristic trace adds a small, consistent edge over Hancock** (same recon):
  ~1.7 pts less turbulence dissipation (55.9→54.2 %) and +0.74 pt smooth-wave retention
  (98.24→98.98 %) — precisely the parabola-curvature term the trace keeps in the time
  update and the Hancock half-step drops.
- **On Metal the two predictors are identical in throughput** at scale (128³ 72/73, 256³
  78/78 Mcell/s) — both bandwidth-bound, so the trace's extra algebra hides under memory
  traffic. So `:hancock` buys eigen-free / system-agnostic simplicity (useful for MHD) at
  ~no GPU cost and ~1–2 % accuracy; `:trace` is the accurate, complete default.

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
- **PPML (full method) is the heaviest solver everywhere** — ~1.9 Mcell/s CPU-f32, 65 / 75
  Mcell/s Metal at 128³/256³ (~3× slower than Hancock-PLM, on par with PPM-DirectEuler at
  scale). The cost is structural: 9 kernels/sweep (predictor → HLLC flux+star → conserved
  update → corrector), the RGK + CW84 + flatten + WENO5 limiter/reconstruction stack, and a
  persistent face-pair state read/written each sweep. The WENO5 smooth-extremum fallback
  adds only ~3–5 % (it branches in only at extrema). The point of PPML is the *method*
  (stateful characteristic tracing + extremum-preserving reconstruction), not throughput —
  for raw speed Hancock-PLM wins; for low-Mach sharpness Hancock-PPM wins.
- **Optimization — per-axis transposed face-pair storage (+42–46% on Metal at scale).**
  The face pair (30 grid-arrays) is only ever touched in its own axis's sweep, so it is
  stored *permanently in that axis's transposed frame* (velocities pre-rotated to
  normal/transverse roles) instead of lab-frame. The y/z sweeps then read/write it
  directly — eliminating 40 full-grid gather passes per step (10 transpose-in + 10
  transpose-out × 2 non-x sweeps). Bit-identical results; Metal 128³ 48.6→69.2, 256³
  52.7→76.9; CPU +14–21%. (64³ is flat — small grids are launch/compute-bound, not
  transpose-bandwidth-bound.) Further fusion of predictor+Riemann into one kernel is
  blocked by Metal's 31-buffer cap (the fused kernel needs 26 arrays + scalars > 31).
