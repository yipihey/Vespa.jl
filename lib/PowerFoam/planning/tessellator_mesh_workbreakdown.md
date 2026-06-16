# Tessellator and Mesh Work Breakdown

Scope: `lib/PowerFoam/src/tessellation3d.jl`, `lib/PowerFoam/tessellator_port/`, and the current `lib/PowerFoam/examples/*` gate set.

This artifact is the planning seam for the KA-based Arepo.jl mesh rewrite. It is intentionally narrow: tessellator semantics, mesh data contracts, backend parity, and the gates required before promotion into the production moving-mesh path.

## 1. Current baseline

### CPU reference status

| Item | Status | Evidence | Notes |
| --- | --- | --- | --- |
| Production-facing API `build_arepo_tessellation_3d` | done | `tessellation3d.jl` | Explicitly labeled `:local_periodic_halfspace` or `:arepo_delaunay_reference` so gates cannot confuse the current rung with the finished port. |
| Existing production-compatible fallback | done | `build_arepo_tessellation_3d(...; algorithm=:local_periodic_halfspace)` | This is the current passing path for periodic 3-D rebuild gates. It wraps the local periodic halfspace rebuild, not a Delaunay port. |
| CPU Delaunay/Voronoi semantic reference | partial but real | `_delaunay_voronoi_mesh_arrays_3d`, `_bowyer_watson_delaunay3` | Produces `ArepoMeshArrays3D`, face centers, cell centers, image shifts, and optional Delaunay payload. Good enough for smoke/reference work, not yet certified against production AREPO topology on lattice starts. |
| Reference debug schema | done | `TessellationReference3D`, `DelaunayTetrahedra3D` | Canonical face keys/order, face image shifts, backend residency tags, metadata, and optional Delaunay payload already exist. |
| Predicate policy seam | scaffolded | `predicates` kwarg and `TessellationFallbackCounters3D` | API surface is present; production adaptive GPU/CPU exact replay path is not yet implemented. |
| N4 native rebuild trace gate through the API | done | `tessellator_port/integration_checklist.md` | Recorded as passing with `missing=0`, `extra=0`, `max_volume_diff=2.23779e-16` for the fallback path. |
| N4/N8 topology parity for `:arepo_delaunay_reference` vs production AREPO | not done | `gpu_soa_implementation_plan.md`, `integration_checklist.md` | This is the main missing promotion gate for the reference path. |

### SoA / KA kernel status

| Slice | Status | Current implementation |
| --- | --- | --- |
| Point/image SoA layout | done | `PeriodicPointImages3D`, `DelaunaySoA3D`, `TessellationSoA3D` |
| Backend transfer helpers | done | `to_backend` for Delaunay and tessellation SoA payloads |
| Periodic point-image generation kernel | done | `_periodic_point_images_soa_kernel!` |
| Dense candidate/halo pair generation kernel | done | `_dense_candidate_pairs_soa_kernel!` |
| Circumcenter recomputation from tetra SoA | done | `_circumcenters_from_tetra_soa_kernel!` |
| Compact candidate scan/pack | not done | still host/scaffold level |
| Tetra predicate classification on backend | not done | no backend `Orient3d` / `InSphere` decision path yet |
| Face ring extraction from incident tetrahedra | not done | current Delaunay face extraction is CPU host logic |
| Compact face-table build on backend | not done | no KA face dedup/emit pass yet |
| CSR rebuild on backend for Delaunay path | not done | current CSR is built on host logic |
| GPU exact fallback queue / CPU replay | not done | only planned in predicate docs |

## 2. Mesh semantics that must not drift

### 2-D vs 3-D semantics

The repo currently has two different semantic rungs and they should not be blurred:

| Dimension | Current semantic role | Current gate shape | Rewrite implication |
| --- | --- | --- | --- |
| 2-D | Bounded and periodic prototype / comparison environment | `powerfoam_kh2d_compare_gate.jl`, 2-D turbulence parity, moving contact examples | Useful for conservative ALE and moving/static comparisons, but not the authoritative template for the 3-D AREPO tessellator rewrite. |
| 3-D | Primary AREPO rewrite target | `arepo_geometry_gate_3d.jl`, `arepo_native_rebuild_trace_gate_3d.jl`, `native_moving_solver_matrix_3d.jl`, `arepo_hierarchy_gate_3d.jl` | The rewrite must preserve periodic nearest-image semantics, compact face/CSR geometry, scheduler compatibility, and moving-face hydro ownership in 3-D. |

Practical reading:

- 2-D is valuable for smoke, ALE correctness, and periodic moving-mesh behavior.
- 3-D periodic turbulence and hierarchy gates are the actual promotion path for the AREPO-style tessellator.
- No claim of "mesh parity" should be made from 2-D gates alone.

### 3-D hydro contract

The stable 3-D output contract already exists and should be treated as fixed unless a gate proves it insufficient:

- `geom::ArepoMeshArrays3D`
  - `c1`, `c2`
  - `cell_face_offsets`, `cell_faces`, `cell_face_signs`
  - `volume`
  - `face_area`
  - `normal_x`, `normal_y`, `normal_z`
  - `face_vx`, `face_vy`, `face_vz`
- `center`
- `face_center`
- `face_image_shift`
- canonical face sort keys and order
- optional Delaunay/debug payload

Semantic invariants from the current source/docs:

1. Face normals point from `c1` to `c2`.
2. Periodic nearest-image choice must be identical between face construction, mesh-motion reconstruction, and hydro replay.
3. Duplicate faces and image-shift rows must preserve update ownership.
4. Cell volume closure must hold locally and globally.
5. CPU and GPU outputs must compare after canonical sorting, not raw row order.

## 3. Degeneracy and tie-breaking policy

This is the highest-risk semantic gap between the current CPU reference and a promotable production tessellator.

### Required policy

1. Keep `:local_periodic_halfspace` as the passing fallback until the Delaunay path matches the bridge on the relevant gates.
2. Treat regular lattice starts (`N4`, `N8`, `N12`) as genuinely degenerate Delaunay cases, not as a minor numerical nuisance.
3. Do not promote topology mismatches on exact lattice starts unless one of these is true:
   - the tie-break reproduces AREPO's production choice, or
   - the gate is explicitly lifted to canonical Voronoi geometry equivalence.
4. Never allow silent GPU-only topology decisions in the ambiguous predicate regime.

### Required predicate behavior

Per `predicate_plan.md`, the ported stack should behave as follows:

- GPU fast path:
  - evaluate the same quick/error-bounded determinant structure as AREPO
  - accept only when the sign is safely outside the error bound
- Ambiguous path:
  - emit a CPU fallback ticket
  - preserve the same point/image/ownership metadata needed to replay the decision
- CPU exact path:
  - authoritative for every ambiguous topology-changing decision
  - authoritative whenever GPU and CPU disagree on a supposedly safe decision

### Gate-facing counters that must exist

- Existing AREPO-style counters:
  - `CountInSphereTests`
  - `CountInSphereTestsExact`
  - `CountConvexEdgeTest`
  - `CountConvexEdgeTestExact`
  - `Count_InTetra`
  - `Count_InTetraExact`
  - flip/split counters
- New port counters:
  - `predicate_orient_fast_accept`
  - `predicate_orient_ambiguous`
  - `predicate_orient_exact_cpu`
  - `predicate_insphere_fast_accept`
  - `predicate_insphere_ambiguous`
  - `predicate_insphere_exact_cpu`
  - `predicate_gpu_fallback_tickets`
  - `predicate_gpu_cpu_mismatches`
  - `predicate_cpu_exact_replays`
  - topology retry / degenerate-face counters

## 4. Hierarchical timestep and incremental rebuild needs

This is not optional follow-on work. The 3-D rewrite is supposed to feed the active-list hydro path, not only full-sync rebuilds.

### Semantic requirements

The port must support:

- active-cell subsets from AREPO-style hydro timebins
- one-hop neighbor closure at minimum
- larger halo closure when periodic wrap, image transitions, or rung boundaries require it
- stable connectivity rebuild order for active cells
- fallback to full rebuild when active-only rebuild fails closure or consistency checks

### Dirty-set rules to implement

Dirty cell:

- any active cell in the current hydro bin
- any cell whose drift changes Voronoi neighbors
- any cell whose periodic image membership changes
- any split / merge / swallow / revive event

Dirty face:

- any face touching a dirty cell
- any face whose image flags change
- any face whose owner identity changes
- any face whose area, normal, or centroid can change because a neighbor moved

Halo closure:

1. seed with dirty cells
2. add every one-hop neighbor across dirty faces
3. add extra neighbors when periodic wrap or rung-transition semantics are present
4. stop only when every dirty cell has enough geometry/context to rebuild local connectivity and face data

### Incremental rebuild acceptance

Incremental rebuild is not complete until all of the following pass:

- active-only face/CSR rebuild matches full rebuild on the active region
- volume closure survives active-only rebuild
- scheduler-selected active cells match the intended timebin probe
- final fields from active-only rebuild agree with full-sync reference within the documented same-precision tolerance

## 5. Work breakdown

### WBS-A. Freeze and document the 3-D contract

Status: mostly done

Deliverables:

- keep `build_arepo_tessellation_3d` as the single public seam
- keep `TessellationReference3D` canonical sort/debug contract
- keep `ArepoMeshArrays3D` output stable

Exit criteria:

- no new mesh implementation bypasses the canonical face-key normalization
- all gates compare through the same normalized contract

### WBS-B. Certify the CPU Delaunay reference

Status: partial

Tasks:

1. Run `:arepo_delaunay_reference` against the AREPO bridge at `N4`, `N8`, then `N12`.
2. Separate perturbed-point success from exact-lattice degeneracy behavior.
3. Record missing/extra/duplicate faces, image-shift mismatches, update-target mismatches, and volume/area/center deltas.
4. Decide whether promotion needs exact AREPO tie-breaking or geometry-equivalence comparison on lattice starts.

Exit criteria:

- Delaunay reference passes the non-degenerate bridge gates
- lattice degeneracy policy is explicit and artifact-backed

### WBS-C. Finish the backend-resident SoA path

Status: partial

Done already:

- periodic image generation
- dense candidate generation
- circumcenter recompute
- backend transfers for SoA payloads

Remaining kernels:

1. active candidate compaction scan/pack
2. halo compaction scan/pack
3. backend predicate classification for tetra/location work
4. fallback-ticket emission for ambiguous predicates
5. edge/tetra incident-ring enumeration
6. backend face polygon/ring emission
7. face dedup and compact face-table build
8. backend volume/center accumulation
9. backend CSR row counts, scans, and fill
10. backend canonical gather for parity comparisons

Exit criteria:

- a backend-resident tessellator path exists from points to compact face/CSR arrays
- host debug buffers are optional, not required for production rebuild

### WBS-D. Implement robust predicates with explicit CPU fallback

Status: scaffold only

Tasks:

1. Port the fast/error-bounded `Orient3d` and `InSphere` algebra with deterministic operand order.
2. Add ambiguous-result ticketing.
3. Add CPU exact replay for ambiguous tickets.
4. Compare GPU-accepted decisions against CPU exact on sampled gates.
5. Fail closed on mismatch.

Exit criteria:

- no ambiguous topology-changing decision is resolved silently on GPU
- mismatch counters are zero on certified gates

### WBS-E. Implement incremental and hierarchical rebuild

Status: not done

Tasks:

1. represent dirty cells/faces explicitly
2. grow one-hop halo plus periodic/rung closure
3. rebuild only touched local topology
4. preserve face ownership/image identity under partial rebuild
5. fall back to full rebuild on closure or consistency failure

Exit criteria:

- active-cell rebuild gates pass on hierarchy fixtures and moving Noh-style probes

### WBS-F. Metal/CPU parity and replay promotion

Status: pending

Tasks:

1. compare CPU backend vs KA CPU vs Metal on SoA arrays
2. compare compact face/CSR arrays after canonical sorting
3. compare final hydro fields after native rebuild on CPU and Metal
4. carry parity through repeated drift/rebuild cycles, not only one-step smoke runs

Exit criteria:

- exact index-array parity
- floating geometry and field parity within documented same-precision tolerance
- no backend-specific topology divergence

## 6. Gate map to use during promotion

### Geometry and rebuild gates

| Gate | Purpose | Promotion use |
| --- | --- | --- |
| `arepo_geometry_gate_3d.jl` | AREPO-exported geometry conversion and small hydro replay on CPU/Metal | keep as the upstream contract check for the hydro-facing mesh arrays |
| `arepo_native_rebuild_trace_gate_3d.jl` | direct topology/geometry comparison between AREPO trace rows and PowerFoam rebuild output | primary certification gate for local periodic fallback and later Delaunay path |
| `arepo_tessellator_rebuild_gate_matrix.jl` | prints the staged N4/N8/N12 and repeated-drift command matrix | use as the promotion checklist, not just documentation |
| `native_rebuild_gate_3d.jl` | small native rebuild smoke gate | keep as a fast local contract sanity check |

### Hierarchy and moving-mesh gates

| Gate | Purpose | Promotion use |
| --- | --- | --- |
| `arepo_hierarchy_gate_3d.jl` | hydro timebin parity and active-list schedule checks | required before claiming partial rebuild support |
| `arepo_noh3d_smoke_gate.jl` | moving local-periodic geometry plus hierarchy hot-path probe | useful for converging geometry and hierarchy interactions |
| `native_moving_solver_matrix_3d.jl` | repeated native Julia rebuild + CPU/Metal hydro comparisons | final local rebuild replay gate once topology is backend-resident |

### Component parity gates that protect adjacent semantics

| Gate | Purpose | Why it matters |
| --- | --- | --- |
| `arepo_gradient_parity_3d.jl` | CPU/Metal gradient parity | tessellator promotion must not regress geometry consumers |
| `arepo_face_trace_gate_3d.jl` | face-level predictor/flux parity | protects moving-face orientation and geometry semantics |
| `arepo_mesh_velocity_gate_3d.jl` | mesh velocity reconstruction parity | protects nearest-image and face-center semantics |
| `arepo_initial_state_gate_3d.jl` | state round-trip parity | useful for separating mesh bugs from state-import bugs |

## 7. Acceptance criteria for "production-ready tessellator"

The rewrite should not be considered complete until all of the following are true:

1. `build_arepo_tessellation_3d(...; algorithm=:arepo_delaunay_reference)` or its promoted successor passes the AREPO bridge geometry/rebuild gates on the targeted resolutions.
2. Degenerate lattice behavior has an explicit, artifact-backed tie-breaking or geometry-equivalence policy.
3. CPU backend and Metal backend produce the same compact topology after canonical sorting, with only documented same-precision float differences.
4. Predicate ambiguity is visible through counters and exact CPU replay, with zero silent topology divergences.
5. The tessellator supports hierarchical active-cell rebuild with halo closure and full-rebuild fallback.
6. The rebuilt geometry can drive existing predictor, flux, mesh-velocity, and final-field gates without regression.
7. Promotion is based on repeated drift/rebuild cycles, not only one initial mesh or one smoke step.

## 8. Recommended execution order

1. Certify the CPU Delaunay reference on bridge gates before writing more backend topology code.
2. Decide the degeneracy/tie-breaking gate policy before claiming N4/N8 topology parity.
3. Finish scan/compact and CSR backend kernels before trying to optimize throughput.
4. Add predicate fallback tickets before any backend tetra-topology edits are trusted.
5. Land hierarchical active-cell rebuild only after full-sync backend parity is green.
6. Promote to the production moving-mesh path only after `native_moving_solver_matrix_3d.jl` and hierarchy gates stay green on CPU and Metal.

## 9. Key recommendations

- Keep the current local periodic halfspace path as the production fallback until the Delaunay path is gate-certified.
- Treat lattice degeneracy as a first-class semantic task, not as cleanup after performance work.
- Make backend predicate ambiguity explicit and expensive rather than silent and fast.
- Use the existing 3-D gate stack as the source of truth; 2-D artifacts are supporting evidence, not promotion evidence.
- Do not start with multirank/MPI semantics. Single-rank periodic correctness plus hierarchy-aware incremental rebuild is the right first complete target.
