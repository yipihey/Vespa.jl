# Task Packet: KA Data Layout

Output file: `lib/PowerFoam/tessellator_port/data_layout.md`

Read only:

- `arepo_tessellator_port_brief.md`
- `lib/PowerFoam/src/hydro3d.jl`
- `lib/PowerFoam/src/PowerFoam.jl`
- `/Users/tabel/Projects/arepo/src/mesh/voronoi/voronoi.h`
- `/Users/tabel/Projects/arepo/src/mesh/mesh.h`

Deliverable:

- Proposed SoA buffers for Delaunay points, tetra adjacency, candidate/halo
  points, Voronoi faces, compact face table, and CSR.
- Mark each buffer as CPU-only debug, backend-resident production, or
  scan/compact transient.
- Include canonical sort keys for CPU/GPU parity.
- Max 140 lines.

Do not edit code.

