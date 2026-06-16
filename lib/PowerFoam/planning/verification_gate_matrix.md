# PowerFoam / AREPO Rewrite Verification And CI Gate Matrix

Date: 2026-06-15

This matrix defines the verification ladder for the KA-based `Arepo.jl` rewrite
surface that currently lives in `lib/PowerFoam`. It is intentionally biased
toward certified component parity before long-run physics or performance claims.

Scope rules for this artifact:

- Use executable gates that already exist in `lib/PowerFoam/examples/` where
  possible.
- Treat original AREPO example `check.py` scripts as the reference definition
  of standard-problem success, not as optional flavor text.
- Keep CPU `Float64` parity with live AREPO as the first promotion path.
- Treat CPU `Float32` and Metal/GPU gates as backend-parity gates on top of a
  certified CPU physics path, not as substitutes for it.
- Keep cosmology gates visible in the matrix even where the PowerFoam surface is
  still pre-production, so CI does not drift into a hydro-only local optimum.
- Distinguish driver-path existence from passing evidence; artifact presence
  alone is not a pass claim.

## Evidence Status Semantics

The lightweight helper `lib/PowerFoam/examples/arepo_rewrite_gate_matrix.jl`
reports a separate evidence state alongside runnable/planned status:

| evidence status | meaning |
| --- | --- |
| `last_observed_pass` | we ran that helper in the current thread and its expected artifact anchor exists |
| `exists` | the driver path exists and is runnable, but this report did not observe a pass in the current thread |
| `not_run` | planned entry, directory-only proxy, or missing driver path |

This keeps the summary honest: existing scripts and old artifact directories can
show that a gate is wired up, but they do not by themselves count as freshly
observed passing evidence.

Observed-pass rule for this document:

- only add `--observed-pass <driver>` for a gate you actually ran in the
  current thread,
- only add it after the gate wrote the artifact anchor named in this matrix,
- do not use it to "upgrade" old artifacts, directory-only proxies, `Pkg.test`
  entries, or `stdout only` rows.

## Promotion Levels

| level | meaning | CI effect |
| --- | --- | --- |
| `D0` | diagnostic prototype, unstable command or tolerance | manual only |
| `D1` | executable diagnostic with durable artifact path | non-blocking CI report |
| `Q0` | smoke gate with explicit pass/fail status | optional presubmit / blocking on dedicated branch |
| `Q1` | certified parity gate at small N with documented tolerance | blocking for the owning subsystem |
| `R0` | regression gate covering a promoted physics path | blocking in default CI |
| `R1` | release gate covering promoted physics plus backend parity/perf envelope | blocking for release / promotion branch |

Promotion policy:

1. A gate starts at `D0` until it has a single documented command, durable
   output location, and explicit pass/fail semantics.
2. A gate can move to `D1` once it writes a stable `README.md` or CSV artifact
   under `lib/PowerFoam/examples/out/`.
3. A gate can move to `Q0` once it exits nonzero on failure and has a bounded
   walltime suitable for CI.
4. A gate can move to `Q1` only after the tolerance is derived from repeated
   runs, not a one-off best case.
5. A gate can move to `R0` only when the upstream dependency chain below it is
   already at `Q1`.
6. No end-to-end hydro, cosmology, GPU, or performance gate may become
   blocking if a prerequisite bridge/component gate is still diagnostic.

## CI Lanes

| lane | target | purpose | current blocking floor |
| --- | --- | --- | --- |
| `unit` | `lib/PowerFoam/test/runtests.jl` | local invariants, conservation, backend staging | `R0` |
| `bridge` | live AREPO via `ArepoLib` | certify state/geometry/predictor/update equivalence | `Q1` on certified pieces |
| `hydro-standard` | stock AREPO hydro examples | problem-level physics regression | `D1` to `Q0`, selective |
| `cosmology` | stock AREPO cosmology examples | gravity/subgrid reference surface for later rewrite stages | `D0` today |
| `backend` | KA CPU vs Metal/GPU | same-precision backend parity after CPU physics closes | `D1` today |
| `perf` | runtime + memory envelope | catch regressions after correctness is stable | `D0` today |

## Unit / Component Gates

These are the first line of defense and should remain the fastest blocking lane.

| gate | source | checks | expected artifact | current level | promote when |
| --- | --- | --- | --- | --- | --- |
| 2-D Voronoi and import invariants | `lib/PowerFoam/test/runtests.jl` | face areas, centroids, imported AREPO polygons, refine-patch metrics | `Pkg.test` summary only | `R0` | keep blocking |
| 2-D weight relaxation and mesh quality | `lib/PowerFoam/test/runtests.jl` | target-area convergence, tiny-face control, compactness metrics | `Pkg.test` summary only | `R0` | keep blocking |
| 2-D hydro conservation | `lib/PowerFoam/test/runtests.jl` | uniform-flow conservation, positivity, LLF/HLL step sanity | `Pkg.test` summary only | `R0` | keep blocking |
| 2-D KA CPU staging | `lib/PowerFoam/test/runtests.jl` | backend transfer, work buffer sizing, positive state after step | `Pkg.test` summary only | `R0` | keep blocking |
| 2-D periodic reconstructed transport | `lib/PowerFoam/test/runtests.jl` | exact preservation of uniform flow and linear data on periodic mesh | `Pkg.test` summary only | `R0` | keep blocking |
| 3-D gradient primitive pieces | `lib/PowerFoam/examples/arepo_gradient_parity_3d.jl` | gradient operator vs AREPO C gradients | `examples/out/arepo_gradient_parity_3d/.../README.md` | `Q1` | already eligible to block bridge lane |

Recommendation: keep `lib/PowerFoam/test/runtests.jl` blocking on every PR and
never dilute it with long external AREPO jobs.

## Bridge Parity Gates

These gates define whether the rewrite can honestly claim to be AREPO-like.

| gate | driver | prerequisite | pass condition | expected artifact | current level | CI recommendation |
| --- | --- | --- | --- | --- | --- | --- |
| Initial-state parity | `lib/PowerFoam/examples/arepo_initial_state_gate_3d.jl` | working `ArepoLib` + stock turbulence IC | CPU `Float64` primitive diffs at roundoff; CPU/Metal `Float32` within same-precision tolerance | `examples/out/arepo_initial_state_gate_3d/N4/README.md`, `N8/README.md` | `Q1` | blocking in bridge lane |
| Geometry parity | `lib/PowerFoam/examples/arepo_geometry_gate_3d.jl` | initial-state parity | face pairs, volumes, areas, normals, centers, CSR counts match exported AREPO geometry | `examples/out/arepo_geometry_gate_3d/.../README.md` | `Q1` | blocking in bridge lane |
| Gradient parity | `lib/PowerFoam/examples/arepo_gradient_parity_3d.jl` | geometry parity | gradient arrays match live AREPO on same geometry | `examples/out/arepo_gradient_parity_3d/.../README.md` | `Q1` | blocking in bridge lane |
| Mesh-velocity parity | `lib/PowerFoam/examples/arepo_mesh_velocity_gate_3d.jl` | geometry parity | `VelVertex` reconstruction matches AREPO to roundoff | `examples/out/arepo_mesh_velocity_gate_3d/.../README.md` | `D1` | non-blocking until artifact/tolerance are stabilized |
| Scheduler / timebin parity | `lib/PowerFoam/examples/arepo_hierarchy_gate_3d.jl` | initial-state parity | hydro bins, effective bins, active masks/lists, next sync step match AREPO | `examples/out/arepo_hierarchy_gate_3d/.../README.md` | `Q1` for controlled fixtures | blocking for scheduler helpers only |
| Face-trace parity | `lib/PowerFoam/examples/arepo_face_trace_gate_3d.jl` | geometry + gradient + trace bridge | every active traced row matches AREPO face states and flux-area values | `examples/out/arepo_face_trace_gate_3d/.../README.md` | `Q1` at N4 HLL/LLF | blocking in bridge lane |
| Trace replay parity | `lib/PowerFoam/examples/arepo_trace_replay_gate_3d.jl` | face-trace parity | replayed conserved update matches AREPO after all traced passes | `examples/out/arepo_trace_replay_gate_3d/.../README.md` | `Q1` at N4 HLL/LLF | blocking in bridge lane |
| Native rebuild parity | `lib/PowerFoam/examples/arepo_native_rebuild_trace_gate_3d.jl` | geometry + trace bridge | native local periodic rebuild reproduces traced pass geometry | `examples/out/arepo_native_rebuild_trace_gate_3d/.../README.md` | `D1` | non-blocking diagnostic |
| Tessellator matrix | `lib/PowerFoam/examples/arepo_tessellator_rebuild_gate_matrix.jl` | native rebuild parity | planning/report wrapper over native rebuild runs | stdout markdown table only | `D1` | planning helper only |
| One-step gap diagnostic | `lib/PowerFoam/examples/arepo_one_step_gap_3d.jl` | trace replay parity | bounded final primitive gap after replacing traced pieces with native pieces | `examples/out/arepo_one_step_gap_3d/.../README.md` | `D1` | non-blocking until pass-sequence closes |
| Preflux smoke | `lib/PowerFoam/examples/arepo_preflux_smoke_gate_3d.jl` | trace bridge | basic availability of pre-flux snapshots and bridge fields | `examples/out/arepo_preflux_smoke_gate_3d/.../README.md` | `D1` | keep non-blocking |

Minimum blocking bridge set for the rewrite:

- initial-state parity
- geometry parity
- gradient parity
- face-trace parity
- trace replay parity
- scheduler parity for the controlled fixture

Do not promote native full-step claims above these gates.

## Hydro Standard-Problem Gates

The original AREPO example checks define the success surface:

- `examples/noh_2d/check.py`: weighted L1 density tolerance vs analytic shock.
- `examples/noh_3d/check.py`: weighted L1 density tolerance vs analytic shock.
- `examples/kh_2d_lecoanet/check.py`: mixed area, vertical KE, enstrophy,
  pressure wiggle, symmetry, plus plots.
- `examples/wave_1d/check.py` and `examples/acoustic_wave_1d/check.py`: linear
  wave amplitude/phase behavior, useful as the reference semantics for a small
  2-D periodic smooth-wave rung before a dedicated AREPO-backed 2-D wave gate.
- `examples/gresho_2d/check.py`: vortex profile retention and pressure balance.
- `examples/yee_2d/check.py`: smooth vortex advection.
- `examples/shearing_sinusoid_2d/check.py`: smooth shear transport.

Matrix:

| problem | AREPO reference | PowerFoam driver | status today | expected artifact | target promotion |
| --- | --- | --- | --- | --- | --- |
| 3-D decaying turbulence | `examples/bauer_springel_turbulence_3d/check.py` | bridge stack + `lib/PowerFoam/examples/arepo_standard_problem_matrix.jl` | strongest certified path, but still component-first | `examples/out/arepo_standard_problem_matrix/.../README.md` plus bridge artifacts | `R0` after native production pass-sequence closes at N8/N12 |
| Noh 3-D | `examples/noh_3d/check.py` | `lib/PowerFoam/examples/arepo_noh3d_smoke_gate.jl` | executable diagnostic with field/radial outputs | `examples/out/arepo_noh3d_smoke_gate/.../README.md`, `radial_bins.csv`, `field_compare.csv` | `Q0` now, `Q1` after tolerances are frozen over repeated runs |
| KH 2-D original AREPO refs | `examples/kh_2d_lecoanet/check.py` | `lib/PowerFoam/examples/arepo_kh2d_original_gate.jl` | executable AREPO reference builder | `examples/out/arepo_kh2d_original_gate/.../README.md`, `kh_metrics_combined.csv`, `analysis/final_fields.csv` | keep `D1`; this is a reference producer, not a blocking rewrite gate |
| KH 2-D PowerFoam compare | same as above | `lib/PowerFoam/examples/powerfoam_kh2d_compare_gate.jl` | executable field-comparison gate, moving rung still smoke-scale | `examples/out/powerfoam_kh2d_compare_gate/.../README.md`, `powerfoam_kh_metrics.csv`, `powerfoam_final_fields.csv`, `field_compare.csv` | `Q0` now, `Q1` once moving rung is no longer smoke-only |
| Noh 2-D | `examples/noh_2d/check.py` | proxy data under `lib/PowerFoam/examples/arepo_noh_proxy/` | proxy only; no executable final-field parity gate | `results/README.md`, CSV metrics, radial-density PNGs | build an executable gate before any blocking use |
| Sound wave 2-D | `examples/acoustic_wave_1d/check.py`, `examples/wave_1d/check.py` semantics reused on a 2-D periodic grid | `lib/PowerFoam/examples/arepo_soundwave2d_gate.jl` | executable calibration gate with exact-solution and Fourier diagnostics; no frozen thresholds yet | `examples/out/arepo_soundwave2d_gate/.../README.md`, `metrics_powerfoam.csv`, `profile_powerfoam.csv` | `D0` until tolerances are frozen or an AREPO reference rung is added |
| Sedov 2-D | local AREPO proxy flow under `lib/PowerFoam/examples/arepo_sedov_proxy/` | proxy generator only | proxy only | proxy tables/plots under `examples/arepo_sedov_proxy/` | build executable radial-profile parity gate |
| Contact/blob 2-D | `examples/contact_blob_2d/check.py` | none yet in PowerFoam; only AREPO-side regression exists | AREPO-only regression | eventual `README.md` + final-field CSV + contact-width metric | `D0` |
| Gresho 2-D | `examples/gresho_2d/check.py` | `lib/PowerFoam/examples/arepo_gresho2d_gate.jl` | executable calibration gate with periodic profile diagnostics | `examples/out/arepo_gresho2d_gate/.../README.md`, `metrics_powerfoam.csv`, `profile_powerfoam.csv` | `D0` until an upstream AREPO profile comparison is added |
| Shearing sinusoid 2-D | `examples/shearing_sinusoid_2d/check.py` | none yet | planned | eventual harmonic-distortion CSV + plot | `D0` |
| Yee 2-D | `examples/yee_2d/check.py` | none yet | planned | eventual phase-error/profile artifact | `D0` |
| Acoustic / wave 1-D | `examples/acoustic_wave_1d/check.py`, `examples/wave_1d/check.py` | none yet | planned; likely thin-3D or dedicated 1-D shim first | amplitude/phase CSV + plot | `D0` |

Blocking guidance for hydro:

- Only `Noh 3-D` and `KH 2-D compare` are close to CI-worthy executable
  problem gates today.
- `Noh 2-D`, `Sedov 2-D`, `Contact/blob`, `Gresho`, `Shearing sinusoid`, and
  `Yee` should stay non-blocking until each has a single command and stable
  artifact path.
- No hydro standard test should block before the bridge lane beneath it is
  green for the same solver/backend.

## Cosmology Gates

The original AREPO cosmology examples already encode useful regression
standards, even if the PowerFoam rewrite is not ready to claim them yet.

| problem | AREPO reference check | reference observable | rewrite relevance | status today | promotion rule |
| --- | --- | --- | --- | --- | --- |
| Gravity-only box | `examples/cosmo_box_gravity_only_3d/check.py` | halo mass function deltas at `z=1,0` | gravity + mesh coupling surface | no PowerFoam gate | keep `D0` until gravity path is integrated |
| Gravity-only zoom | `examples/cosmo_zoom_gravity_only_3d/check.py` | contaminant intrusion and subhalo mass ranking | zoom-region gravity fidelity | no PowerFoam gate | keep `D0` |
| Cosmological star formation box | `examples/cosmo_box_star_formation_3d/check.py` | stellar mass density history + morphology plots | future hydro+gravity+subgrid target | no PowerFoam gate | never block hydro rewrite until subgrid scope exists |

Recommendation: represent cosmology in CI as a planning lane now, not a failing
lane. The right short-term use is a manifest of reference scripts and expected
observables, not dummy red jobs.

## CPU Backend Vs Metal/GPU Parity Gates

| gate | driver | comparison | expected artifact | current level | promotion rule |
| --- | --- | --- | --- | --- | --- |
| Initial-state CPU vs Metal `Float32` | `lib/PowerFoam/examples/arepo_initial_state_gate_3d.jl` | `Float32(AREPO)` vs KA CPU vs Metal | `examples/out/arepo_initial_state_gate_3d/.../README.md` | `Q1` where Metal is available | keep as backend gating only |
| 3-D turbulence CPU vs Metal | `lib/PowerFoam/examples/turbulence_gpu_parity_3d.jl` | KA CPU `Float32` vs Metal `Float32` time histories | `examples/out/turbulence_gpu_parity_3d/.../README.md`, `metrics.csv` | `D1` | promote after native CPU physics path is `R0` |
| 2-D turbulence CPU vs Metal | `lib/PowerFoam/examples/turbulence_gpu_parity_2d.jl` | KA CPU `Float32` vs Metal `Float32` on 2-D moving mesh | `examples/out/turbulence_gpu_parity_2d/.../README.md`, `metrics.csv` | `D1` | promote after 2-D hydro path has a blocking CPU standard test |

Policy:

- Backend parity should compare like-with-like precision.
- A GPU gate may be green while the CPU physics path is still wrong; that does
  not justify promotion.
- Metal-unavailable runs should report `skipped`, not `passed`.

## Performance Gates

Performance gates belong after correctness. They should use fixed commands,
explicit hardware labels, and compare against stored envelopes, not anecdotes.

| gate | source | metric | expected artifact | current level | blocking rule |
| --- | --- | --- | --- | --- | --- |
| Native rebuild scaling | `lib/PowerFoam/examples/arepo_tessellator_rebuild_gate_matrix.jl` plus native rebuild outputs | walltime and topology drift across N/repeats | markdown table + referenced `README.md` artifacts | `D1` | non-blocking |
| 3-D CPU vs Metal turbulence throughput | `lib/PowerFoam/examples/turbulence_gpu_parity_3d.jl` | steps/s, walltime, memory if added | `metrics.csv`, `README.md` | `D1` | non-blocking until CPU physics is `R0` |
| 2-D CPU vs Metal turbulence throughput | `lib/PowerFoam/examples/turbulence_gpu_parity_2d.jl` | steps/s, walltime | `metrics.csv`, `README.md` | `D1` | non-blocking |
| Standard-problem matrix rollup | `lib/PowerFoam/examples/arepo_standard_problem_matrix.jl` | availability of promoted gates and reports | `examples/out/arepo_standard_problem_matrix/.../README.md`, CSVs | `D1` | use as summary dashboard, not blocker |

Performance promotion rule:

1. freeze the solver/mesh path,
2. certify correctness at `R0`,
3. collect at least three repeated measurements per hardware/backend,
4. then enforce a generous regression envelope such as `<= 10%` slowdown.

## Expected Outputs And Artifact Contract

Every promoted executable gate should emit:

- `README.md`: human-readable summary, command context, pass/fail status.
- one machine-readable table: CSV preferred.
- stable output directory under `lib/PowerFoam/examples/out/<gate>/<run-tag>/`.
- explicit solver/backend/problem size in the run tag.

Preferred artifact shape by gate type:

| gate type | required artifacts |
| --- | --- |
| unit/component | test status in `Pkg.test`; optional CSV only if debugging |
| bridge parity | `README.md` + per-field maxdiff table or CSV |
| standard hydro | `README.md` + final-field CSV + profile/metric CSV + plots if the AREPO reference produces them |
| backend parity | `README.md` + synchronized metrics CSV with CPU and GPU rows |
| performance | `README.md` + metrics CSV with hardware/backend labels |

## Recommended Default CI Stack

Short presubmit:

1. `lib/PowerFoam/test/runtests.jl`
2. `arepo_initial_state_gate_3d.jl` at `N4`
3. `arepo_geometry_gate_3d.jl` at `N4`
4. `arepo_face_trace_gate_3d.jl` at `N4`, HLL
5. `arepo_trace_replay_gate_3d.jl` at `N4`, HLL

Nightly bridge:

1. short presubmit stack
2. `arepo_face_trace_gate_3d.jl` at `N4`, LLF
3. `arepo_trace_replay_gate_3d.jl` at `N4`, LLF
4. `arepo_hierarchy_gate_3d.jl` multirung fixture at `N8`
5. `arepo_native_rebuild_trace_gate_3d.jl` at `N8`

Nightly hydro:

1. `arepo_noh3d_smoke_gate.jl`
2. `arepo_kh2d_original_gate.jl`
3. `powerfoam_kh2d_compare_gate.jl`
4. `arepo_standard_problem_matrix.jl` in report mode

Backend nightly:

1. `turbulence_gpu_parity_3d.jl`
2. `turbulence_gpu_parity_2d.jl`

## Key Recommendations

1. Make the bridge lane the authoritative blocker for the rewrite. It is the
   only part already close to a complete correctness story.
2. Promote `arepo_noh3d_smoke_gate.jl` to `Q0` first among the hydro standard
   problems; it already emits the right artifacts.
3. Treat `powerfoam_kh2d_compare_gate.jl` as the second hydro promotion target,
   but do not call it blocking until the moving rung is no longer smoke-scale.
4. Convert the existing `noh_2d` and `sedov_2d` proxy material into executable
   gates before spending CI budget on additional new problems.
5. Keep cosmology visible as a planned lane with named AREPO observables, but
   do not make it red-by-default while the rewrite is still hydro-first.
6. Delay performance blocking until the native pass-sequence gap is closed and
   the CPU `Float64` path has at least one `R0` standard-problem gate.

## Helper Script

`lib/PowerFoam/examples/arepo_rewrite_gate_matrix.jl` is the lightweight
reporting helper for this plan. It does not execute any AREPO or PowerFoam
gates; it only inspects whether the listed driver paths and artifact anchors
exist.

Default behavior:

- prints a compact markdown summary grouped by promotion `level` and CI `lane`
- prints the full markdown gate table
- marks gates as `runnable` when the driver path is an existing `.jl` file
- marks directory-backed proxies or missing paths as `planned`
- marks artifact anchors as `present` or `pending`

Simple flags:

- `--summary-only`: emit just the grouped status summary
- `--table-only`: emit just the full gate table
- `--root <path>` or `--root=<path>`: inspect a different checkout root
- `--observed-pass <driver>` or `--observed-pass=<driver>`: mark a runnable
  file-backed gate as `last_observed_pass` for this report, but only if its
  artifact anchor exists

Observed-pass workflow after lightweight gates in this thread:

```bash
julia --project=. lib/PowerFoam/examples/arepo_rewrite_gate_matrix.jl \
  --summary-only \
  --observed-pass lib/PowerFoam/examples/arepo_runtime_hydro_smoke.jl \
  --observed-pass lib/PowerFoam/examples/arepo_runtime_moving_mesh_smoke.jl \
  --observed-pass lib/PowerFoam/examples/arepo_gravity_direct_smoke.jl
```

Use one `--observed-pass` flag per driver you just ran. Relative paths are
resolved from the checkout root, so the repo-root form shown above is the
intended invocation for notes, CI logs, and follow-up summaries.

If you want the full table instead of only the grouped summary, drop
`--summary-only` and keep the same `--observed-pass` arguments.

Safe to mark `--observed-pass` when the artifact anchor exists:

| lane | eligible drivers |
| --- | --- |
| `runtime` | `lib/PowerFoam/examples/arepo_runtime_hydro_smoke.jl`, `lib/PowerFoam/examples/arepo_runtime_moving_mesh_smoke.jl` |
| `bridge` | `lib/PowerFoam/examples/arepo_initial_state_gate_3d.jl`, `lib/PowerFoam/examples/arepo_geometry_gate_3d.jl`, `lib/PowerFoam/examples/arepo_gradient_parity_3d.jl`, `lib/PowerFoam/examples/arepo_mesh_velocity_gate_3d.jl`, `lib/PowerFoam/examples/arepo_hierarchy_gate_3d.jl`, `lib/PowerFoam/examples/arepo_face_trace_gate_3d.jl`, `lib/PowerFoam/examples/arepo_trace_replay_gate_3d.jl`, `lib/PowerFoam/examples/arepo_native_rebuild_trace_gate_3d.jl`, `lib/PowerFoam/examples/arepo_one_step_gap_3d.jl`, `lib/PowerFoam/examples/arepo_preflux_smoke_gate_3d.jl` |
| `hydro-standard` | `lib/PowerFoam/examples/arepo_noh3d_smoke_gate.jl`, `lib/PowerFoam/examples/arepo_kh2d_original_gate.jl`, `lib/PowerFoam/examples/powerfoam_kh2d_compare_gate.jl`, `lib/PowerFoam/examples/arepo_noh2d_proxy_gate.jl` |
| `backend` | `lib/PowerFoam/examples/turbulence_gpu_parity_3d.jl`, `lib/PowerFoam/examples/turbulence_gpu_parity_2d.jl`, `lib/PowerFoam/examples/tessellator_backend_parity_probe.jl` |
| `gravity` | `lib/PowerFoam/examples/arepo_gravity_direct_smoke.jl` |
| `io` | `lib/PowerFoam/examples/arepo_io_runtime_surface_smoke.jl` |
| `perf` | `lib/PowerFoam/examples/arepo_standard_problem_matrix.jl` |

Never mark `--observed-pass` from this helper for:

- `lib/PowerFoam/test/runtests.jl` or any other `Pkg.test summary` row
- rows whose artifact is `stdout only`
- directory-backed proxy rows such as `lib/PowerFoam/examples/arepo_noh_proxy/`
- AREPO-only `check.py` reference rows or any `n/a` artifact entry
