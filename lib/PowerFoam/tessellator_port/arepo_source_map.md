# AREPO 3-D Tessellator Source Map

This map anchors the Julia/KA port to the production AREPO implementation.
The first port target is semantic parity, not GPU speed.

## Production Files

- `/Users/tabel/Projects/arepo/src/mesh/voronoi/voronoi.h`
  - Defines the core `point`, `tetra`, `tetra_center`, `connection`, and
    `tessellation` layouts.
  - Exposes global counters for in-sphere tests, exact-predicate fallbacks,
    flip counts, edge/face splits, and point-location checks.
- `/Users/tabel/Projects/arepo/src/mesh/voronoi/voronoi.c`
  - Owns high-level mesh construction, mesh updates, ghost search coupling, and
    timebin reconstruction around the tessellator.
- `/Users/tabel/Projects/arepo/src/mesh/voronoi/voronoi_3d.c`
  - Owns the 3-D Delaunay insertion, flips, circumcenters, point-location
    predicates, and Voronoi face/volume extraction.
- `/Users/tabel/Projects/arepo/src/mesh/voronoi/voronoi_ghost_search.c`
  - Owns imported point/image discovery for parallel and periodic boundaries.

## Required Julia Semantics

The Julia reference path should mirror these AREPO concepts before the KA/GPU
layout is optimized:

- `point`
  - physical coordinates
  - cell ID / task / hydro index / original index / timebin
  - periodic `image_flags`
  - optional integer coordinate representation for exact predicate policy
- `tetra`
  - four oriented point indices
  - four adjacent tetra indices, each opposite the matching point
  - neighbor-side index `s`
  - deleted marker equivalent to `t[0] == -1`
- `tetra_center`
  - circumcenter per tetra, with exact fallback for near-degenerate cases
- `face`
  - Delaunay edge endpoints `p1`, `p2`
  - area, center, and optional generating tetra/edge provenance
- `connection`
  - neighbor task/index/image flags
  - face index back-pointer
  - original point identity for MPI/periodic comparisons

## Algorithmic Stages

1. Build an enclosing tetrahedron.
2. Insert points one at a time with point location via orientation predicates.
3. Split tetra/face/edge depending on whether the new point lies inside, on a
   face, or on an edge.
4. Restore Delaunay validity with in-sphere tests and 2-to-3, 3-to-2, and
   4-to-4 flips.
5. Compute circumcenters.
6. Walk Delaunay edge rings to extract Voronoi faces, face centers, areas,
   volumes, and cell-center contributions.
7. Normalize face order with the Julia canonical key before comparing to AREPO
   or backend-specific layouts.

## First Port Boundary

`build_arepo_tessellation_3d` currently advertises
`:local_periodic_halfspace` for the production-compatible fallback path and
`:arepo_delaunay_reference` for the CPU Delaunay-derived reference path.  The
reference path now constructs tetrahedra incrementally, computes circumcenters,
extracts Voronoi faces from Delaunay edge rings, and emits the same
`ArepoMeshArrays3D` contract as the hydro solvers.  It is still a CPU reference
rung; the production GPU target is the same semantic contract converted to SoA
buffers and KA kernels.
