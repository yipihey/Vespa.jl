# Tessellator Backend Parity Probe

Scope: lightweight availability note for the exported KA tessellator primitives in `lib/PowerFoam/src/tessellation3d.jl`, without changing source or tests.

## Purpose

The new tessellator SoA helpers already expose a narrow backend-facing surface:

- `periodic_point_images_soa_3d`
- `dense_candidate_pairs_soa_3d`
- `pack_candidate_stencil_soa_3d`
- `candidate_tetra_predicates_soa_3d`
- `candidate_conflict_face_rows_soa_3d`
- `candidate_boundary_face_rows_soa_3d`
- `recompute_circumcenters_soa_3d`
- `delaunay_soa_3d`
- `tessellation_soa_3d`

This note adds a tiny example probe that:

1. runs those primitives on `KernelAbstractions.CPU()` with a fixed 5-point input;
2. checks whether `Metal` is loadable from the active `lib/PowerFoam` project environment;
3. runs the same sequence on `Metal.MetalBackend()` only when that package is actually loadable.

The script intentionally does not assume `Metal` is part of the project. It treats missing `Metal` as a skip, not a failure.

## Probe script

File: `lib/PowerFoam/examples/tessellator_backend_parity_probe.jl`

Command:

```bash
julia --project=lib/PowerFoam lib/PowerFoam/examples/tessellator_backend_parity_probe.jl
```

## Observed result on 2026-06-15

Ran in the `Vespa.jl` checkout with the `lib/PowerFoam` project environment.

- CPU probe: passed.
- Metal probe: skipped.
- Skip reason: `Base.find_package("Metal")` returned `nothing`, so `Metal` is not available in the active project environment.

CPU counts from the probe run:

| Quantity | Value |
| --- | ---: |
| Faces in reference tessellation | 37 |
| Delaunay tetrahedra | 660 |
| Periodic image rows | 135 |
| Active dense candidate rows | 76 |
| Max packed candidates per source | 20 |
| Valid recomputed circumcenters | 660 |
| Valid predicate rows | 50160 |
| Active conflict rows | 5792 |
| Boundary rows | 2706 |
| Faces in tessellation SoA | 37 |

## Interpretation

- The exported CPU path is live enough for a cheap backend smoke check.
- On this project environment, there is currently no Metal package to import, so GPU execution should be treated as unavailable rather than broken.
- If `Metal` is later added to the environment, the same script should automatically switch from "skipped" to an actual `MetalBackend()` probe without any source changes.
