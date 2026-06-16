# AREPO Tessellator Semantics Map

Scope: 3-D production tessellation path only, using the packet files in
`voronoi.c`, `voronoi_3d.c`, `voronoi.h`, and `mesh.h`.

| Stage | Source lines | Inputs | Outputs | Invariants | KA port implication |
|---|---|---|---|---|---|
| 1. Build context and seed local points | `voronoi.c:104-276`, `voronoi.c:460-545`, `voronoi_3d.c:133-293` | active hydro cells, current drifted particle state, mesh alloc factors, periodic box size | seeded `Mesh.DP/DT/VF`, `List_InMesh`, `ListExports`, `DPinfinity`, enclosing outer tetra | active cells are drifted to `All.Ti_Current`; swallowed cells are skipped; outer tetra encloses the box; `image_flags` and `Hsml` are initialized for later topology work | split as a host/device setup pass plus a backend-resident point buffer; keep the seed tetra and image metadata deterministic across CPU/GPU |
| 2. Insert points and restore Delaunay topology | `voronoi_3d.c:1476-1862`, `voronoi_3d.c:1874-2079`, `voronoi_3d.c:2886-3293`, `voronoi_3d.c:3913-4605` | new point `pp`, tetra start guess, current `DT` adjacency, exact/adaptive predicates | updated tetra mesh, flip stack, deleted/replaced tetras, final containing tetra | tetra orientation must stay positive; recursive flips must restore Delaunayhood; `InSphere` / `Orient3d` fallbacks must agree with exact arithmetic; infinity tetras are rejected for real inserts | this stage is branchy and recursive, so keep a CPU fallback path; GPU can batch classify candidates, but topology edits need explicit retry/fallback counters |
| 3. Refresh circumcenters and max Delaunay radius | `voronoi_3d.c:3302-3328`, `voronoi_3d.c:3614-3716`, `voronoi.c:672-718` | finalized `DT`, `DTC`, active synchronized cells | per-tetra circumcenters in `DTC`, per-cell `MaxDelaunayRadius` | deleted/infinity tetras are skipped; degenerate linear solves fall back to exact circumcenters; radius is only accumulated for synchronized local cells | store circumcenters as backend-resident SoA; use a separate reduction kernel for radii, not an in-kernel scalar loop |
| 4. Walk edges to build Voronoi faces, volumes, and centers | `voronoi.c:729-793`, `voronoi_3d.c:425-731`, `voronoi.c:818-872`, `voronoi.c:921-963` | `DT`, `DTC`, `DP`, periodic box geometry, active cells and timebin sync state | `VF` face list, per-cell `Volume`, `SurfaceArea`, `Center`, remote `ActiveArea` updates | each edge is visited once via `Edge_visited`; face normals follow the `c1 -> c2` orientation; periodic nearest-image deltas use `nearest_x/y/z`; face area and pyramid volume contributions must close per cell | make this a scan/compact pipeline: edge enumeration, face emission, then owner-grouped accumulation; preserve image-shift/update-target ownership explicitly |
| 5. Project face geometry to hydro-facing arrays | `voronoi.c:1031-1085`, `mesh.h:149-170`, `mesh.h:248-255` | `VF` plus per-cell area/center state | `geometry` fields (`nn`, `nx/ny/nz`, `mx/my/mz`, `px/py/pz`, `cx/cy/cz`) or `ArepoMeshArrays3D` equivalents | negligible faces can be skipped; only synchronized active cells contribute; the left/right cell choice controls which side owns the geometry update | keep a compact face table separate from per-cell geometry; this is the direct seam for the KA hydro path and for AREPO-parity gates |

## Data contracts worth preserving

| Type | Source lines | Contract |
|---|---|---|
| `point` | `voronoi.h:115-135` | stores position, ID, owner task/index, original index, timebin, `image_flags`, and optional integer-mapped coordinates / stencil state |
| `tetra` | `voronoi.h:137-145` | oriented tetra points `p[4]`, adjacent tetrahedra `t[4]`, and opposite-vertex slots `s[4]` define navigation and flips |
| `face` | `mesh.h:149-170` | face endpoints `p1/p2`, optional provenance, area, and face centroid are the compact Voronoi face payload |
| `geometry` | `mesh.h:248-255` | hydro-facing face geometry pack: scalar, normal, midpoint, and centroid components |

## Predicate and periodicity notes

| Mechanism | Source lines | Semantics | Port implication |
|---|---|---|---|
| Integer-mapped coordinates | `voronoi.h:59-85`, `voronoi_3d.c:3879-3899` | doubles are masked into a stable integer interval for exact arithmetic helpers | keep the mapping as a shared utility or CPU fallback, not as a silent GPU approximation |
| `InSphere` path | `voronoi_3d.c:3913-4350` | quick/adaptive/errorbound test first, then exact GMP path when the sign is uncertain | expose a fallback counter and never allow silent topology divergence |
| `Orient3d` path | `voronoi_3d.c:4363-4605` | quick and exact orientation tests with the same sign convention | the GPU path needs the same sign contract for flips and face orientation |
| Periodic nearest image | `voronoi.c:921-963` | wrap deltas into the nearest image unless the axis is reflective | periodic image choice must be identical for face construction and moving-face velocity reconstruction |

## What the port should keep visible

- `CountInSphereTests`, `CountInSphereTestsExact`, `CountConvexEdgeTest`, `CountConvexEdgeTestExact`,
  `Count_InTetra`, and `Count_InTetraExact` are the gate-facing counters already present in AREPO.
- Duplicate faces and image-shift ownership must survive compaction unchanged.
- CPU and GPU should agree on the final compact arrays after canonical sorting; if they do not,
  the difference needs an explicit, documented tolerance or fallback path.
