# Halo / Incremental Rebuild Notes

Bounded to:
- `arepo_tessellator_port_brief.md`
- `voronoi_ghost_search.c`
- `voronoi_dynamic_update.c`
- `voronoi_exchange.c`
- `allvars.h`

## 1) What AREPO actually does in ghost / halo search

- The ghost search is driven from the active hydro list and only keeps cells that are still meaningful for the current hydro bin (`TimeBinSynchronized[...]` checks in both local and imported paths).
- For each candidate face/tetra, AREPO picks a local point `q`, uses its `Hsml` as the maximum search radius, and searches around the tetra center vs. the reference point. If the search radius is clipped (`maxdist < 2*h`), it clears the "fully decided" bit on that tetra so the mesh knows it may need more work.
- In periodic / extended search mode, the same physical neighbor can appear multiple times through different image shifts. AREPO tracks this with `image_flags` and `image_bits` so one neighbor-image pair is only inserted once.
- Ghost-point insertion zeroes `SphP[p].ActiveArea` and adds a replicated `point` record with the original particle index plus image metadata.
- Local and imported ghost work follow the same semantic shape: collect requests, exchange the needed points, then append the imported points to `T->DP` and bump `T->Ndp`.

## 2) Connectivity update semantics

- `voronoi_update_connectivity()` first clears the connection lists for all active cells and returns those connection nodes to the free list.
- It then scans every face twice, once from each endpoint, so each incident cell gets a forward connection record.
- A connection is only attached when the owning endpoint is local, gas, still alive, and synchronized for hydro.
- Each connection stores:
  - target task
  - image flags
  - particle ID
  - target index, with `q_index - NumGas` used for local replicated points
  - `dp_index`
  - face index, and optionally `dt_index`
- If the free list runs out, AREPO grows `DC` geometrically and rebuilds the freelist tail.
- `voronoi_get_connected_particles()` and `voronoi_exchange.c` show the same ownership model at a higher level: build `List_InMesh`, dedupe exports by origin and image flags, exchange what the owner needs, then map imported data back onto the local `DP` ordering.

## 3) Single-rank periodic GPU-first scope

- First target: one rank, periodic box, GPU-friendly local rebuild.
- Keep the periodic image bookkeeping and duplicate suppression, but do not require MPI export/import, task sorting, or distributed ownership transfer yet.
- `List_InMesh`, `ListExports`, `PrimExch`, and `GradExch` should remain shape-compatible, but the first implementation can keep all data local.
- What can wait:
  - `MPI_Alltoall` / `MPI_Sendrecv` plumbing
  - foreign-connection import/export
  - multi-task remapping of `DP` indices
  - any domain-decomposition / Peano-Hilbert work tied to task boundaries
- What should not wait:
  - periodic nearest-image semantics
  - stable face ownership and image identity
  - exact per-cell connection list shape
  - active-cell clearing / rebuild ordering

## 4) Dirty-cell / dirty-face halo rules for hierarchy

Proposed incremental rules, aligned with the AREPO semantics above:

- Dirty cell:
  - any active cell in the current hydro bin
  - any cell whose position drifted enough to change its Voronoi neighborhood
  - any cell whose replicated image membership changed
  - any split / merge / swallow / revive event
- Dirty face:
  - any face with a dirty endpoint
  - any face whose image flags changed
  - any face whose owner / remote endpoint changes rank or local-image identity
  - any face whose area / normal / centroid could change because a neighbor moved
- Halo closure:
  - start with all dirty cells
  - add every one-hop neighbor across a dirty face
  - keep adding until every dirty cell has enough neighboring geometry to rebuild its local connections and face data
  - for hierarchy, include the next coarser or finer rung neighbor whenever a face crosses rung boundaries
- Practical rule of thumb:
  - the minimum halo is one cell thick
  - grow it when a dirty cell sits on a periodic wrap, a replicated image boundary, or a rung transition
  - only skip a cell if none of its incident faces are dirty and all its active-bin neighbors are already present locally

## 5) Minimal implementation note

- For the GPU-first single-rank path, represent ghost images as local replicated points with explicit image flags.
- Rebuild connectivity from the active list plus the one-hop halo, then compact the final arrays into the same face / connection order the hydro path expects.
- That gives a clean path to later multi-rank exchange without changing the local data model.
