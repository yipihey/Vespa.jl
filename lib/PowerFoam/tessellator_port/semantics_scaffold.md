# Semantics Scaffold

This note tracks the first lightweight Julia-side semantic layer for the
AREPO 3-D Delaunay/Voronoi port.  The aim is not to replace the tessellator
yet, only to keep the AREPO bookkeeping visible while the CPU reference path
is being wired in.

## Field Map

| Julia scaffold | AREPO meaning |
|---|---|
| `TessellationPredicatePolicy3D` | Predicate mode selection for the rebuild path: adaptive fast path, float64 debug path, exact CPU oracle, or explicit CPU fallback after a fast-path miss. |
| `TessellationPointIdentity3D` | Generator identity plus image/owner metadata that must survive compaction and ghost handling. |
| `TessellationFaceProvenance3D` | Original face row, endpoint cells, ownership, periodic image shift, and orientation/duplicate status for canonical sorting. |
| `TessellationFallbackCounters3D` | Gate-facing counters for `InSphere`, `Orient3d`, `InTetra`, convex-edge checks, exact fallbacks, retries, and degenerate or skipped topology. |

## Semantics Notes

- `image_shift` is the nearest-image offset, not a generic displacement.
- `owner_task` and `owner_index` preserve the flux/update owner even when the
  face row is geometrically mirrored or de-duplicated.
- The counter names intentionally mirror the AREPO gate vocabulary:
  `CountInSphereTests`, `CountInSphereTestsExact`,
  `CountConvexEdgeTest`, `CountConvexEdgeTestExact`,
  `Count_InTetra`, and `Count_InTetraExact`.
- The scaffold stays small on purpose so the later CPU reference path can swap
  in real predicates and topology edits without changing the debug contract.
