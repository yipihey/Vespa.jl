# AREPO Tessellator Port Gate Design

This matrix is staged from certified N4/N8 checks up to the larger N16/N24
work-count gates. It keeps the tessellator port honest on topology first, then
on replay parity, then on scaling and backend consistency.

## Gate Matrix

| Stage | Gate | Command shape | Compared fields | Pass criteria | Artifact paths |
| --- | --- | --- | --- | --- | --- |
| 1 | N4/N8 geometry parity | `julia --project=lib/PowerFoam/examples lib/PowerFoam/examples/arepo_geometry_gate_3d.jl <N> <dt> <riemann> <steps>` | `c1`, `c2`, face count, vertices, cell-face CSR counts, volume, face area, normals, `center`, `face_center`, `vrms`, `mach_rms`, `density_rms`, `rho_min`, `rho_max`, `pmin`, conserved-field drift, CPU/Metal final fields | Exact row/CSR counts; `c1/c2` mismatches = 0; volume/area/normal diffs at roundoff; CPU/Metal field diffs at same-precision tolerance; final conserved fields stable under 1-step and multi-step runs | `lib/PowerFoam/examples/out/arepo_geometry_gate_3d/<RUN_TAG>/README.md`; `.../metrics.csv`; optional `.../final_state_*.csv` when enabled |
| 2 | N4 native rebuild trace | `julia --project=lib/PowerFoam/examples lib/PowerFoam/examples/arepo_native_rebuild_trace_gate_3d.jl <N> <dt> <riemann> <steps>` with `POWERFOAM_NATIVE_TRACE_RADIUS=1` and `POWERFOAM_NATIVE_TRACE_MIN_FACE_SURFACE_FRACTION=1e-5` | pass index, active trace rows, native rows, matched/missing/extra pair counts, duplicate rows, max area diff, max normal diff, max center diff, max volume diff, extra-row area sum, extra-row area max, extra-row cut ratio, extra-row centers | Missing = 0 and extra = 0; extra faces below AREPO cutoff; max area and volume diffs <= `1e-10`; normal and center diffs at roundoff | `lib/PowerFoam/examples/out/arepo_native_rebuild_trace_gate_3d/<RUN_TAG>/README.md` |
| 3 | N4/N8 hydro replay | `julia --project=lib/PowerFoam/examples lib/PowerFoam/examples/native_moving_solver_matrix_3d.jl <N> <dt> <step_counts> <solvers> <search_radius> <order>` | end-to-end timing rows, backend rows, final `D`, `Mx`, `My`, `Mz`, `E`, `vrms`, `mach_rms`, `density_rms`, `rho_min`, `rho_max`, `pmin`, mass and energy drift, CPU/Metal deltas | CPU and Metal final fields agree to same-precision tolerance; one-step predictor/rebuild replay remains roundoff-level; solver ordering does not change the accepted final state beyond documented tolerance | `lib/PowerFoam/examples/out/native_moving_solver_matrix_3d/<RUN_TAG>/README.md`; `.../solver_summary.csv`; optional `.../final_state_steps*_*.csv` when `POWERFOAM_WRITE_FINAL_FIELDS=1` |
| 4 | N16/N24 work-count scaling | `POWERFOAM_MESH_WORK_STATS=1 POWERFOAM_MESH_PROFILE=1 POWERFOAM_REBUILD=gpu_compact POWERFOAM_ACTIVE_CELLS=gradients julia --project=lib/PowerFoam/examples lib/PowerFoam/examples/native_moving_solver_matrix_3d.jl <N> <dt> <step_counts> <solvers> <search_radius> reconstruct` | `refreshes`, `candidate_faces`, `dirty_cells`, `dirty_faces`, `active_faces`, `clip_planes`, `clip_inside`, `clip_empty`, `clip_clipped`, `tier_rejected`, `face_clip`, `volumes`, `active_cells`, `face_scan`, `face_pack`, `cell_scan`, `csr_fill`, plus final-state parity | Counts are finite and nonnegative; work tallies scale monotonically with problem size; no backend-specific divergence in final fields beyond tolerance; compact rebuild does not regress replay parity | `lib/PowerFoam/examples/out/native_moving_solver_matrix_3d/<RUN_TAG>/README.md`; `.../solver_summary.csv`; `.../final_state_steps*_*.csv` when enabled |

## Staging Notes

1. Run Stage 1 first; it proves the exported AREPO geometry conversion and the
   replay path can still hold roundoff-level physics fields.
2. Run Stage 2 next; it checks the native local periodic rebuild against traced
   AREPO pass tables before trusting the production pass sequence.
3. Run Stage 3 as the narrow hydro regression gate for N4 and N8.
4. Run Stage 4 only after Stage 3 is stable; it is the scaling gate that
   exposes compaction, scan, and candidate-work costs at N16/N24.

## Suggested Default Sweep

- N4: `dt=0.001`, `riemann=hll`, `steps=1`
- N8: `dt=0.001`, `riemann=hll,llf`, `steps=1,8`
- N16/N24: same solver set, `search_radius=1`, `order=reconstruct`, with
  work stats and mesh profile enabled

## Acceptance Summary

- Topology gates must fail closed on any missing, extra, or duplicate face
  ownership that is not already explained by the documented AREPO cutoff.
- Field gates must keep CPU and Metal final arrays aligned after canonical
  sorting or backend-normalized row order.
- Scaling gates must preserve the same replay semantics while making the work
  counters visible enough to compare N16 against N24.
