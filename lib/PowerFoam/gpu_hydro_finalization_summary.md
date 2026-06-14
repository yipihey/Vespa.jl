# GPU Hydro Finalization Summary

This file tracks the production path for AREPO-style 2-D and 3-D hydro in
PowerFoam. Timing claims should be treated as secondary while the local GPU is
busy; correctness, operation counts, synchronized phase timing, and scaling are
the acceptance evidence.

## Production Policy

- Default kernels stay lean: no mesh stats writes, no host staging, and no debug
  counters unless the corresponding environment flag is enabled.
- Diagnostic knobs remain opt-in:
  - `POWERFOAM_MESH_WORK_STATS`
  - `POWERFOAM_MESH_PROFILE`
  - `POWERFOAM_SYNC_TIMING`
- `POWERFOAM_REBUILD=gpu_compact` remains the 3-D moving-mesh production target.
- `POWERFOAM_CANDIDATE_TIER=full` remains the conservative default until reduced
  tiers pass parity and work-count gates.
- Dirty/incremental rebuild remains opt-in until it proves final-field parity or
  a documented same-precision tolerance.

## Correctness Gates

| Path | Gate | Status | Evidence |
| --- | --- | --- | --- |
| 2-D first-order bounded mesh | Conservative HLL/LLF step | Passing | `lib/PowerFoam/test/runtests.jl` |
| 2-D backend primitives | Conserved-to-primitive on KA CPU matches host arrays | Passing | `PowerFoam 2D prototype`, 173/173 |
| 2-D reconstructed gradients | Mesh least-squares gradient recovers a linear field | Passing | `PowerFoam 2D prototype`, 173/173 |
| 2-D face predictor | Predicted face state matches linear reconstruction with zero extrapolation | Passing | `PowerFoam 2D prototype`, 173/173 |
| 2-D reconstructed update | Uniform flow preserves conserved integrals | Passing | `PowerFoam 2D prototype`, 173/173 |
| 2-D reconstructed Metal compile | Tiny reconstructed HLL step on `MetalBackend()` | Passing | rho min `0.999986f0`, pressure max `1.0000140462259832` |
| 2-D reconstructed moving mesh | Bounded host rebuild with backend-resident reconstructed hydro | Passing | N4 CPU/Metal final-field maxdiff `0` |
| 3-D compact inactive-face skip | Predictor and Riemann kernels skip zero-area compact tail faces | Passing | N8 active/capacity `3947/6656`; CPU/Metal maxdiff `0` |
| 3-D compact block scan | Auto serial/parallel KA prefix scan for compact face/CSR block offsets | Passing | Forced parallel N4 CPU/Metal maxdiff `0` |
| 3-D compact moving mesh | CPU/Metal compact path and workstats smoke | Passing | N24 one-step CPU/Metal maxdiff `0`; matched work counts |
| 3-D plane cull | `POWERFOAM_PLANE_CULL=true` vs `false` final-field equality | Passing at N4; N16/N24 still needed | `examples/plane_cull_gate_3d.jl`, maxdiff `0` |

## Implemented This Slice

- Added backend-resident `PrimitiveState2D`, `HydroGradients2D`, and
  `FaceStates2D`.
- Added `conserved_to_primitive_2d!`, `primitive_work_2d`,
  `primitive_to_arrays_2d`, `hydro_gradient_work_2d`, and
  `face_prediction_work_2d`.
- Added packed-buffer 2-D gradient and predictor kernels to keep Metal argument
  counts small.
- Added `calculate_gradients_from_mesh_2d!`,
  `predict_face_states_2d!`, and `finite_volume_reconstructed_step_2d!`.
- Added `moving_mesh_reconstructed_step_2d!` for bounded moving meshes that
  still rebuild on the host but keep reconstructed hydro work on the backend.
- Added unit coverage for 2-D primitive parity, linear gradients, face
  prediction, and reconstructed uniform-flow preservation.

## Remaining Optimization Targets

| Target | Goal | Gate |
| --- | --- | --- |
| 2-D periodic local rebuild | Periodic turbulence boxes can avoid bounded host `power_diagram` rebuilds | N8/N16 CPU vs Metal parity, HLL/LLF, first-order and reconstructed |
| 2-D compact rebuild | Mirror 3-D compact face table, compact CSR, backend area update, backend hydro update | Exact final-field parity on small periodic grids |
| 3-D plane-cull matrix | Promote culling confidence from smoke to canonical artifact | N4/N16/N24 `true` vs `false` equality on full candidate tier |
| Dirty-only rebuild | Rebuild dirty faces plus one-cell halo and fall back on checks | Active-face and volume validation, exact or documented tolerance |
| Scaling artifacts | Separate canonical from diagnostic run tags | Operation counts, synchronized phase timing, async timing when GPU is idle |

## Current Test Command

```bash
env JULIA_DEPOT_PATH=/private/tmp/enzo_powerfoam_depot:/Users/tabel/.julia \
/Users/tabel/.julia/juliaup/julia-1.12.6+0.aarch64.apple.darwin14/bin/julia \
--project=lib/PowerFoam lib/PowerFoam/test/runtests.jl
```

Latest result: `PowerFoam 2D prototype | 176 pass | 176 total`.

## Current 3-D Decaying-Turbulence Optimization

- Compact buffers retain fixed candidate capacity, but only the active prefix is
  used by the compact CSR. The reconstructed predictor and Riemann face-flux
  kernels now skip zero-area compact tail faces before doing reconstruction or
  primitive-to-conserved/Riemann work.
- Compact face and CSR block offsets now use
  `POWERFOAM_COMPACT_BLOCK_SCAN_MODE=auto|serial|parallel`. `auto` keeps the
  tiny serial device scan for small runs and switches to a parallel KA prefix
  scan once the block count reaches
  `POWERFOAM_COMPACT_BLOCK_SCAN_PARALLEL_THRESHOLD`.
- N8 two-step diagnostic: final active faces `3947` out of `6656` compact face
  slots, so about 41% of face lanes are now cheap zero-area exits in the
  predictor and flux kernels.
- CPU/Metal final-field differences stayed exactly zero in the N8 diagnostic.
- Forced parallel block-scan N4 compact smoke also kept CPU/Metal final-field
  differences exactly zero.
- The regression suite now includes a zero-area face predictor check.
- After removing the Metal-specific allocation shim from the library, the N4
  compact Metal smoke still produced CPU/Metal final-field maxdiff `0`.

### Latest N24 Diagnostic

Run command:

```bash
env JULIA_NUM_THREADS=4 \
JULIA_DEPOT_PATH=/private/tmp/enzo_powerfoam_depot:/Users/tabel/.julia \
JULIA_LOAD_PATH=@:lib/PowerFoam:@stdlib \
POWERFOAM_PERF_WARMUP=false \
POWERFOAM_REBUILD=gpu_compact \
POWERFOAM_MESH_WORK_STATS=true \
POWERFOAM_MESH_PROFILE=true \
POWERFOAM_SYNC_TIMING=true \
POWERFOAM_COMPACT_BLOCK_SCAN_MODE=auto \
/Users/tabel/.julia/juliaup/julia-1.12.6+0.aarch64.apple.darwin14/bin/julia \
--project=lib/MultiCode/test \
lib/PowerFoam/examples/native_moving_solver_matrix_3d.jl \
24 0.001 1 hll 1 reconstruct
```

Artifact:
`lib/PowerFoam/examples/out/native_moving_solver_matrix_3d/N24_dt0p001_n1_r1_reconstruct_gpu_compact_hll_sync-timing_mesh-profile_workstats`.

| Metric | CPU Float32 | Metal Float32 |
| --- | ---: | ---: |
| elapsed s | 4.22098646 | 11.6869915 |
| active faces / candidates | 133387 / 179712 | 133387 / 179712 |
| volume sum | 1.00350773 | 1.00350773 |
| density rms | 0.00489340303 | 0.00489340303 |
| pmin | 0.503615504 | 0.503615504 |
| mass drift | 4.58955765e-05 | 4.58955765e-05 |
| energy drift | 4.28557396e-05 | 4.28557396e-05 |
| final-field maxdiff D/Mx/My/Mz/E | 0 / 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 / 0 |

Compact rebuild work counts matched exactly: `179712` candidate faces, `133387`
active faces, `31.6555` planes per face, `0.301611` inside fraction,
`0.00229817` empty fraction, and `0.696091` clipped fraction.

Synchronized Metal new-mesh subphase timing was:

| face_clip | volumes | face_scan | face_pack | cell_scan | csr_fill |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0102778 | 0.000570167 | 0.00107608 | 0.000746208 | 0.000711459 | 0.000415875 |

Interpretation: this is a correctness and attribution pass, not a headline
speedup claim. It confirms the compact path is KA-portable across CPU and Metal
for this N24 decaying-turbulence gate, with exact same-precision final fields and
identical mesh operation counts. The largest compact-rebuild subphase is still
face clipping; the scan/pack/CSR rebuild launches are individually small but
remain plausible fusion targets. Library allocations now go through
`KernelAbstractions.zeros` instead of a Metal-specific storage shim, so backend
selection remains a driver/example concern.

Latest result: `PowerFoam 2D prototype | 176 pass | 176 total`.

## Current Metal Smoke

```bash
env JULIA_NUM_THREADS=4 \
JULIA_DEPOT_PATH=/private/tmp/enzo_powerfoam_depot:/Users/tabel/.julia \
JULIA_LOAD_PATH=@:lib/PowerFoam:@stdlib \
/Users/tabel/.julia/juliaup/julia-1.12.6+0.aarch64.apple.darwin14/bin/julia \
--project=lib/MultiCode/test -e 'using PowerFoam, KernelAbstractions, Metal; ...'
```

Latest result: `(0.999986f0, 1.0000140462259832)`.

## Current 2-D Reconstructed Moving-Mesh Smoke

```bash
env JULIA_NUM_THREADS=4 \
JULIA_DEPOT_PATH=/private/tmp/enzo_powerfoam_depot:/Users/tabel/.julia \
JULIA_LOAD_PATH=@:lib/PowerFoam:@stdlib \
/Users/tabel/.julia/juliaup/julia-1.12.6+0.aarch64.apple.darwin14/bin/julia \
--project=lib/MultiCode/test lib/PowerFoam/examples/turbulence_gpu_parity_2d.jl \
4 0.3 0.001 0.18 hll clamp reconstruct
```

Latest result: `field max abs diffs: D=0 Mx=0 My=0 E=0`.

## Current 3-D Plane-Cull Gate

```bash
env JULIA_NUM_THREADS=4 \
JULIA_DEPOT_PATH=/private/tmp/enzo_powerfoam_depot:/Users/tabel/.julia \
JULIA_LOAD_PATH=@:lib/PowerFoam:@stdlib \
POWERFOAM_PERF_WARMUP=false \
/Users/tabel/.julia/juliaup/julia-1.12.6+0.aarch64.apple.darwin14/bin/julia \
--project=lib/MultiCode/test lib/PowerFoam/examples/plane_cull_gate_3d.jl \
4 0.001 1 hll 1 reconstruct
```

Latest result: `plane-cull gate maxdiff 0`.
