# AREPO Production Tessellator Port Brief

## Goal

Port the production 3-D Delaunay/Voronoi tessellation semantics needed by
AREPO-style moving-mesh hydro into Julia with KernelAbstractions-compatible
data layouts, so the same implementation can run on CPU and GPU.  The first
physics target remains periodic 3-D subsonic turbulence; 2-D and bounded
special cases are secondary.

This is not a request to rewrite all of AREPO.  The scope is the mesh
construction/rebuild surface needed by PowerFoam's existing hydro path:
generator positions in, compact face/CSR geometry out, with semantics close
enough to pass AREPO topology and final-field gates.

## Source Files In Scope

Primary AREPO sources:

- `/Users/tabel/Projects/arepo/src/mesh/voronoi/voronoi.h`
- `/Users/tabel/Projects/arepo/src/mesh/voronoi/voronoi.c`
- `/Users/tabel/Projects/arepo/src/mesh/voronoi/voronoi_3d.c`
- `/Users/tabel/Projects/arepo/src/mesh/voronoi/voronoi_utils.c`
- `/Users/tabel/Projects/arepo/src/mesh/voronoi/voronoi_ghost_search.c`
- `/Users/tabel/Projects/arepo/src/mesh/voronoi/voronoi_dynamic_update.c`
- `/Users/tabel/Projects/arepo/src/mesh/voronoi/voronoi_exchange.c`
- `/Users/tabel/Projects/arepo/src/mesh/set_vertex_velocities.c`
- `/Users/tabel/Projects/arepo/src/mesh/mesh.h`
- `/Users/tabel/Projects/arepo/src/main/allvars.h`
- `/Users/tabel/Projects/arepo/src/main/proto.h`

PowerFoam integration points:

- `lib/PowerFoam/src/hydro3d.jl`
- `lib/PowerFoam/src/PowerFoam.jl`
- `lib/PowerFoam/examples/arepo_geometry_gate_3d.jl`
- `lib/PowerFoam/examples/arepo_native_rebuild_trace_gate_3d.jl`
- `lib/PowerFoam/examples/arepo_trace_replay_gate_3d.jl`
- `lib/PowerFoam/examples/native_moving_solver_matrix_3d.jl`
- `lib/PowerFoam/arepo_physics_parity_audit.md`

## Current State

PowerFoam already has a near-lattice local periodic halfspace clipper and
compact GPU-oriented rebuild variants.  The parity audit says this is certified
against traced N4 post-drift passes for face pairs, volumes, areas, normals,
centers, and update-target ownership, but it is not yet a proven replacement
for AREPO's Delaunay-backed production tessellator on larger grids or repeated
mesh-motion cycles.

Existing certified pieces to preserve:

- AREPO-exported geometry conversion to `ArepoMeshArrays3D`.
- Gradient, predictor, moving-face flux, and update replay on AREPO/native
  diagnostic face tables.
- Hydro timebin and active-list scheduler parity.
- Compact face/CSR geometry expected by the GPU hydro path.

## Target Julia API

The first production-facing API should be narrow:

```julia
build_arepo_tessellation_3d(points; domain, periodic=true, active=nothing,
                            previous=nothing, backend=CPU(),
                            predicates=:adaptive,
                            return_delaunay=false)
```

Required output:

- `geom::ArepoMeshArrays3D`
  - `c1`, `c2`
  - `cell_face_offsets`, `cell_faces`, `cell_face_signs`
  - `volume`
  - `face_area`
  - `normal_x`, `normal_y`, `normal_z`
- `center::Matrix` or backend-resident SoA equivalent.
- `face_center::Matrix` or backend-resident SoA equivalent.
- periodic/image metadata needed to recover AREPO update-target ownership.
- optional Delaunay adjacency/debug payload for gates.

The GPU production layout should be structure-of-arrays and scan/compact
friendly.  Do not design around nested Julia objects in kernels.

## Semantic Invariants To Preserve

- Periodic nearest-image behavior must match AREPO for face construction and
  moving-face velocity reconstruction.
- Face normals must point consistently from `c1` to `c2`.
- Voronoi face area and face centroid must match AREPO's effective hydro
  geometry, including the tiny-face cutoff behavior already discovered in
  prior gates.
- Cell volume closure must hold per cell and globally.
- Duplicate faces and image-shift rows must preserve update-target ownership.
- Active/local rebuild must include the correct one-cell or larger halo for
  hierarchical timestepping.
- CPU and GPU paths must produce the same final compact arrays, modulo
  documented sort-order normalization.
- Any robust predicate fallback must be explicit; silent GPU-only topology
  divergence is not acceptable.

## What Not To Do

- Do not port MPI/domain decomposition in this phase.
- Do not port gravity, cooling, star formation, FoF/Subfind, or full HDF5 I/O.
- Do not optimize before topology/field gates exist.
- Do not change the hydro state/update API unless the tessellator output
  contract proves insufficient.
- Do not replace all existing diagnostic gates; extend them.

## Acceptance Gates

Initial gates:

- N4/N8/N12 periodic turbulence initial and post-drift generator sets:
  compare AREPO vs Julia face-pair set, image shifts, volumes, face areas,
  normals, face centers, cell centers, and CSR counts.
- Repeated drift gate: same comparisons over multiple sync points.
- CPU backend vs Metal backend: exact compact arrays after canonical sorting,
  or documented same-precision tolerance for floating fields.
- Hydro replay gate: existing native predictor/flux replay remains at
  roundoff-level final-field agreement after swapping in the new tessellator.

Later gates:

- Dirty/incremental rebuild with active cell halo for multirung timestep
  hierarchy.
- N16/N24 scaling with operation counts: candidate points, tetras, flips,
  predicate fallback counts, faces, active faces, and compaction work.

## Agent Token Rules

Agents should not summarize the entire AREPO codebase.  Each task packet below
has a bounded output file.  Keep outputs compact, line-referenced, and directly
actionable.  Prefer tables, invariants, and pseudocode over long prose.

