# AREPO Tessellator Port Rebuild Gate Matrix

This note expands the production tessellator port gates beyond the current
N4 native trace bridge. The next step is to certify the same rebuild semantics
after drift, then after repeated drift/rebuild cycles, at N4, N8, and N12.

## Minimum AREPO Bridge Fields

The smallest bridge payload that supports these gates is:

- generator positions: `pos`
- active mask or active index list for the drifting subset
- periodic domain or box size
- cell volumes: `volume`
- cell centers: `center`
- face endpoints / ownership: `c1`, `c2`
- face geometry: `face_area`, `normal`, `face_center`
- periodic image metadata: `face_image_shift` or equivalent `image_flags`
- update-target ownership: `update_c1`, `update_c2`
- drift bookkeeping: pass index, sync index, and snapshot time

Optional but useful for diagnostics:

- face endpoint coordinates: `point_l`, `point_r`
- per-face drift metadata: `face_dt`, `pass_index`, `active`
- bridge-side reference payload from `build_arepo_tessellation_3d`

## Gate Matrix

| Gate | Command | Pass criteria | Artifact paths |
| --- | --- | --- | --- |
| N4 post-drift | `julia --project=lib/PowerFoam -e 'push!(LOAD_PATH, "/Users/tabel/Projects/Arepo.jl/lib/ArepoLib"); ARGS=["4","0.001","hll","1"]; include("lib/PowerFoam/examples/arepo_native_rebuild_trace_gate_3d.jl")'` | `missing=0`, `extra=0`, `trace_duplicates=0`, `native_duplicates=0`, `max_area_diff <= 1e-10`, `max_normal_diff <= 1e-10`, `max_center_diff <= 1e-10`, `max_volume_diff <= 1e-10`, and `update_target_mismatches=0` | `lib/PowerFoam/examples/out/arepo_native_rebuild_trace_gate_3d/N4_dt0p001_hll/README.md`; `.../post_drift.csv`; `.../bridge_fields.csv` |
| N8 post-drift | `julia --project=lib/PowerFoam -e 'push!(LOAD_PATH, "/Users/tabel/Projects/Arepo.jl/lib/ArepoLib"); ARGS=["8","0.001","hll","1"]; include("lib/PowerFoam/examples/arepo_native_rebuild_trace_gate_3d.jl")'` | same thresholds as N4, with the same fail-closed behavior on missing or extra faces | `lib/PowerFoam/examples/out/arepo_native_rebuild_trace_gate_3d/N8_dt0p001_hll/README.md`; `.../post_drift.csv`; `.../bridge_fields.csv` |
| N12 post-drift | `julia --project=lib/PowerFoam -e 'push!(LOAD_PATH, "/Users/tabel/Projects/Arepo.jl/lib/ArepoLib"); ARGS=["12","0.001","hll","1"]; include("lib/PowerFoam/examples/arepo_native_rebuild_trace_gate_3d.jl")'` | same thresholds as N4, with the same fail-closed behavior on missing or extra faces | `lib/PowerFoam/examples/out/arepo_native_rebuild_trace_gate_3d/N12_dt0p001_hll/README.md`; `.../post_drift.csv`; `.../bridge_fields.csv` |
| N4 repeated-drift | `POWERFOAM_NATIVE_TRACE_REPEAT=3 julia --project=lib/PowerFoam -e 'push!(LOAD_PATH, "/Users/tabel/Projects/Arepo.jl/lib/ArepoLib"); ARGS=["4","0.001","hll","3"]; include("lib/PowerFoam/examples/arepo_native_rebuild_trace_gate_3d.jl")'` | all per-sync passes meet the post-drift thresholds; additionally `sync_pass_mismatch=0`, `face_pair_set_drift=0`, and `volume_closure_drift <= 1e-10` across the repeated cycle | `lib/PowerFoam/examples/out/arepo_native_rebuild_trace_gate_3d/N4_dt0p001_hll_repeat3/README.md`; `.../sync_drift.csv`; `.../bridge_fields.csv` |
| N8 repeated-drift | `POWERFOAM_NATIVE_TRACE_REPEAT=3 julia --project=lib/PowerFoam -e 'push!(LOAD_PATH, "/Users/tabel/Projects/Arepo.jl/lib/ArepoLib"); ARGS=["8","0.001","hll","3"]; include("lib/PowerFoam/examples/arepo_native_rebuild_trace_gate_3d.jl")'` | same repeated-drift criteria as N4 | `lib/PowerFoam/examples/out/arepo_native_rebuild_trace_gate_3d/N8_dt0p001_hll_repeat3/README.md`; `.../sync_drift.csv`; `.../bridge_fields.csv` |
| N12 repeated-drift | `POWERFOAM_NATIVE_TRACE_REPEAT=3 julia --project=lib/PowerFoam -e 'push!(LOAD_PATH, "/Users/tabel/Projects/Arepo.jl/lib/ArepoLib"); ARGS=["12","0.001","hll","3"]; include("lib/PowerFoam/examples/arepo_native_rebuild_trace_gate_3d.jl")'` | same repeated-drift criteria as N4 | `lib/PowerFoam/examples/out/arepo_native_rebuild_trace_gate_3d/N12_dt0p001_hll_repeat3/README.md`; `.../sync_drift.csv`; `.../bridge_fields.csv` |

## What To Measure

For every gate, record:

- face-pair set equality
- duplicate face count
- missing face count
- extra face count
- image-shift or image-flag mismatches
- update-target mismatches
- maximum face-area difference
- maximum face-normal difference
- maximum face-center difference
- maximum cell-volume difference
- per-sync repeated-drift deltas for the repeated gate

The pass/fail line should stay strict on topology: any missing or extra face
is a fail unless the bridge explicitly marks it as a documented cutoff case.
The geometry comparisons should stay at roundoff-level tolerance because the
production tessellator is expected to preserve AREPO semantics, not merely
approximate them.

## Degenerate Lattice Policy

Regular N4/N8/N12 lattice starts are co-spherical and can have multiple valid
Delaunay triangulations.  The gate therefore tracks two algorithm modes:

- `POWERFOAM_TESSELLATOR_ALGORITHM=local_periodic_halfspace`
  - Current production-compatible topology gate.
  - Must remain strict: `missing=0`, `extra=0`, and roundoff-level geometry
    differences.
- `POWERFOAM_TESSELLATOR_ALGORITHM=arepo_delaunay_reference`
  - CPU Delaunay-derived diagnostic gate.
  - Must emit the same hydro contract and record Delaunay payload/counters.
  - On exact lattice degeneracy, topology differences are not promoted until
    the predicate tie-breaking policy reproduces AREPO's production choice or
    the comparison is lifted to canonical Voronoi geometry equivalence.

Example diagnostic command:

```sh
POWERFOAM_TESSELLATOR_ALGORITHM=arepo_delaunay_reference julia --project=lib/PowerFoam -e 'push!(LOAD_PATH, "/Users/tabel/Projects/Arepo.jl/lib/ArepoLib"); ARGS=["4","0.001","hll","1"]; include("lib/PowerFoam/examples/arepo_native_rebuild_trace_gate_3d.jl")'
```

The diagnostic artifact path appends the algorithm name, for example:

```text
lib/PowerFoam/examples/out/arepo_native_rebuild_trace_gate_3d/N4_dt0p001_hll_arepo_delaunay_reference/README.md
```

## Notes

- This is a gate-expansion artifact, not a new test harness.
- The repeated-drift gate should be run only after the single post-drift gate
  is green at the same resolution.
- Keep the report paths stable so the next promotion step can compare N4, N8,
  and N12 without changing the lookup pattern.
