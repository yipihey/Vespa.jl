# GPU/SoA Tessellator Implementation Plan

The production target is an AREPO-semantics tessellator whose data stays in
KernelAbstractions-compatible SoA buffers and feeds the existing hydro
`ArepoMeshArrays3D` contract.

## Acceptance Ladder

1. CPU semantic reference
   - Status: done for perturbed periodic 3-D point sets.
   - Builder: `build_arepo_tessellation_3d(...; algorithm=:arepo_delaunay_reference)`.
   - Evidence: focused hydro smoke gate with total volume `1.0000000000000002`.

2. SoA data contract
   - Status: done for reference points/images/tetras/circumcenters and hydro
     face/CSR/center/image-shift arrays.
   - Types: `DelaunaySoA3D` and `TessellationSoA3D`.
   - Evidence: main test suite checks host SoA and KA CPU backend copies.

3. Backend-resident kernels
   - Status: first kernels done.
   - Done: periodic point/image generation and circumcenter recomputation from
     point/tetra SoA arrays.
   - Done: dense candidate/halo binning with active masks and bin offsets.
   - Done: fixed per-source active candidate stencil compaction.
   - Done: candidate-vs-tetra in-sphere predicate masks and margins for
     conflict-region construction.
   - Done: fixed-shape conflict tetra-face row emission from predicate masks.
   - Done: fixed-shape boundary-face deduplication masks.
   - Done: fixed-stride per-candidate boundary-face packing.
   - Done: fixed-stride source-neighbor face candidate rows and CSR-facing
     counts/offset scaffold.
   - Done: source-owned fixed-stride face incidence rows and update signs.
   - Done: topology-only `ArepoMeshArrays3D` prototype with fixed-stride
     padded source-owned rows.
   - Done: reciprocal source-owned face-row pairing metadata with canonical
     owner flags for the global dedup pass.
   - Done: fixed-stride canonical face CSR and topology-only
     `ArepoMeshArrays3D` emission where reciprocal source rows gather from one
     canonical flux row with opposite signs.
   - Done: CPU-reference scan-backed global compact face table and hydro CSR
     rebuild from canonical owner rows.
   - Next: KA-native prefix-sum/scatter for the compact scan, backend parity,
     and real geometric face measures.

4. AREPO bridge topology gates
   - Status: default halfspace gate passes at N4; Delaunay reference diagnostic
     mode is wired behind `POWERFOAM_TESSELLATOR_ALGORITHM`.
   - Next: run N4/N8 Delaunay diagnostics, then implement tie-breaking or
     canonical Voronoi-geometry comparison for co-spherical lattice starts.

5. CPU backend vs Metal backend parity
   - Status: pending.
   - Required before performance claims: exact or documented same-precision
     tolerance for SoA arrays, compact face/CSR arrays, volumes, centers, and
     final hydro fields.

## Current Non-Negotiables

- Keep `local_periodic_halfspace` as the passing production-compatible fallback
  until Delaunay reference topology matches the AREPO bridge on N4/N8.
- Keep the Delaunay reference label honest; do not promote it to default merely
  because perturbed smoke tests pass.
- Treat degenerate lattice cases as predicate/tie-breaking work, not as
  performance work.
- Each new KA kernel must have CPU backend parity before Metal is trusted.
