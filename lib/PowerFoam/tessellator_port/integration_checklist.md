# Tessellator Port Integration Checklist

This checklist is owned by the main integrator.  Subagents write only their
assigned packet outputs; implementation begins after their artifacts are
reviewed and folded into this checklist.

## Phase 0: Contracts

- [x] Create shared port brief.
- [x] Create bounded task packets.
- [x] Review `semantics_map.md`.
- [x] Review `data_layout.md`.
- [x] Review `predicate_plan.md`.
- [x] Review `halo_incremental.md`.
- [x] Review `gate_design.md`.
- [x] Freeze the first Julia API and output contract.
- [x] Add direct AREPO source map for the production 3-D tessellator.

## Active Workstreams

- Main integrator:
  - Owns `build_arepo_tessellation_3d`, reference-schema wiring, gate
    integration, and final include/export decisions.
- Kierkegaard:
  - Owns the first non-invasive semantic scaffold for predicate policy,
    point/image identity, face provenance, and fallback counters.
  - Done and integrated into the PowerFoam module/test suite.
- Archimedes:
  - Owns the rebuild gate matrix for N4/N8/N12 post-drift and repeated-drift
    comparisons.
  - Done, corrected to the local runnable `ArepoLib` launch form, and paired
    with a standalone matrix printer.

## Immediate Implementation Slice

Start with a reference/export gate before touching the production hydro path:

1. Define a `TessellationReference3D` debug schema in a new gate or helper.
   - Done: `src/tessellation3d.jl` exposes `TessellationReference3D`,
     `build_arepo_tessellation_3d`, `tessellation_reference_3d`, and canonical
     face-key helpers while preserving the existing `ArepoMeshArrays3D` hydro
     geometry contract.
2. Extend the AREPO bridge/export side only as needed to expose production
   tessellator comparison fields:
   - particle id / original index / image flags
   - face endpoints and update-target endpoints
   - face area, normal, face center, and cell-center contribution
   - cell volumes and CSR incidence counts
   - optional tetra/debug ids if compiled with the relevant flags
3. Add a canonical sort key:
   `(min(c1,c2), max(c1,c2), image_flags, owner_task, owner_index)`.
4. Compare existing Julia local rebuild output against the AREPO production
   export at N4 first, then N8.
   - N4 native rebuild trace gate now runs through `build_arepo_tessellation_3d`
     and passes with `missing=0`, `extra=0`, and `maxvol=2.23779e-16`.
5. Only after that comparison artifact is stable, introduce the new
   `build_arepo_tessellation_3d` API behind an opt-in flag.
   - Done for the compatibility wrapper.  The wrapper is deliberately labeled
     `:local_periodic_halfspace` until the production Delaunay/Voronoi kernel
     replaces the current local clipper.

## Phase 1: Reference Export And Gates

- [ ] Extend AREPO bridge/export to dump production tessellator details needed
      for topology comparison: face pairs, image shifts, face endpoints,
      tetra/debug ids when available, volumes, areas, normals, centers, and
      CSR/update-target ownership.
- [ ] Add N4/N8/N12 post-drift rebuild comparison gate.
- [ ] Add repeated-drift comparison gate over multiple sync points.
- [x] Add canonical sorting/normalization for CPU/GPU array comparisons.
- [x] Add runnable N4/N8/N12 post-drift and repeated-drift gate matrix.
- [x] Re-run N4 post-drift gate through `build_arepo_tessellation_3d`:
      `missing=0`, `extra=0`, `max_volume_diff=2.23779e-16`.

## Phase 2: CPU Julia Reference

- [x] Implement a CPU reference path that mirrors production Delaunay/Voronoi
      semantics before GPU optimization.
- [x] Add predicate counters and fallback counters.
- [x] Preserve existing `ArepoMeshArrays3D` hydro output contract on the
      Delaunay-derived reference path.
- [x] Pass focused Delaunay reference hydro smoke gate on a perturbed periodic
      3-D point set: 61 faces, total volume `1.0000000000000002`.
- [ ] Pass N4/N8 topology and volume gates against the AREPO production bridge.

## Phase 3: KA Production Layout

- [x] Convert reference point, tetra, circumcenter, face, center, image-shift,
      and CSR structures to SoA buffers.
- [x] Add backend transfer for tessellation SoA buffers.
- [x] Add first KA tessellator kernel: backend-resident circumcenter
      recomputation from point/tetra SoA arrays.
- [x] Add backend-resident periodic image generation for point/image SoA
      buffers.
- [x] Implement dense candidate/halo binning as backend-resident kernels with
      active masks and bin offsets.
- [x] Compact dense candidate/halo rows into fixed per-source active candidate
      stencils.
- [x] Add backend-resident candidate-vs-tetra in-sphere predicate buffers for
      conflict-region construction.
- [x] Emit fixed-shape backend-resident conflict tetra-face rows from predicate
      buffers.
- [x] Add fixed-shape conflict-face boundary deduplication masks.
- [x] Pack boundary faces into fixed-stride per-candidate compact rows.
- [x] Join compact boundary rows with candidate stencil into fixed-stride
      source-neighbor face candidate rows.
- [x] Add fixed-stride CSR-facing counts/offset scaffold for compact face
      candidates.
- [x] Add source-owned fixed-stride face incidence rows and update signs for
      compact face candidates.
- [x] Emit a topology-only `ArepoMeshArrays3D` prototype from compact face
      candidates with fixed-stride padded source-owned rows.
      This is a mesh-topology/debug export only, not the production hydro CSR
      rebuild or the final moving-mesh integration surface.
- [x] Add backend-resident reciprocal source-owned face-row pairing metadata
      (`pair_row`, `canonical_row`, and owner flag) for the next global dedup
      pass.
- [x] Add fixed-stride canonical face CSR and topology-only mesh emission where
      reciprocal source rows gather from one canonical flux row with opposite
      signs.
- [x] Implement CPU-reference scan-backed global compact face table and hydro
      CSR rebuild from canonical owner rows.
- [ ] Replace CPU-reference compact scan with KA-native prefix-sum/scatter.
- [ ] Pass CPU backend vs Metal compact-array parity.

## Phase 4: Incremental Rebuild

- [ ] Implement dirty-cell/dirty-face rebuild with halo expansion.
- [ ] Couple active-cell rebuild to hierarchical timestep masks.
- [ ] Fall back to full rebuild when active-face or volume checks fail.
- [ ] Pass N8 multirung partial rebuild geometry and final-field gates.

## Phase 5: Hydro Integration

- [ ] Replace local halfspace clipper in the production 3-D moving-mesh hydro
      path with the new tessellator behind an opt-in environment flag.
- [ ] Preserve existing native replay parity.
- [ ] Promote the new tessellator to default only after N12/N16 physical-time
      turbulence gates pass.
