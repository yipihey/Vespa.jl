# AREPO Tessellator Port Data Layout

This is the backend-friendly layout for the first production tessellator pass.
It stays close to `point`, `tetra`, `face`, `tessellation`, and the 3-D
`ArepoMeshArrays3D` contract already used by `hydro3d.jl`.

## 1) Delaunay points

| buffer | residency | shape | notes |
| --- | --- | --- | --- |
| `dp_x, dp_y, dp_z` | backend-resident production | `Ndp` | generator positions |
| `dp_vx, dp_vy, dp_vz` | backend-resident production | `Ndp` | only if mesh motion is fused with rebuild |
| `dp_task, dp_index` | backend-resident production | `Ndp` | AREPO ownership and stable id |
| `dp_original_index, dp_timebin` | backend-resident production | `Ndp` | parity and active-list gating |
| `dp_image_flags` | backend-resident production | `Ndp` | periodic/image ownership recovery |
| `dp_active, dp_exported` | scan/compact transient | `Ndp` | local/halo filter and export selection |
| `dp_perm, dp_invperm` | scan/compact transient | `Ndp` | canonical reorder after compaction |
| `dp_debug_xx, dp_debug_yy, dp_debug_zz, dp_debug_ix, dp_debug_iy, dp_debug_iz` | CPU-only debug | `Ndp` | traced / high-precision mirrors from `point` |

## 2) Tetrahedra and adjacency

| buffer | residency | shape | notes |
| --- | --- | --- | --- |
| `tet_p0, tet_p1, tet_p2, tet_p3` | backend-resident production | `Ndt` | oriented Delaunay vertices |
| `tet_t0, tet_t1, tet_t2, tet_t3` | backend-resident production | `Ndt` | adjacent tetrahedra |
| `tet_s0, tet_s1, tet_s2, tet_s3` | backend-resident production | `Ndt` | opposite-vertex slot in neighbor |
| `tet_deleted` | backend-resident production | `Ndt` | matches `t[0] == -1` deletion state |
| `tet_circum_x, tet_circum_y, tet_circum_z` | backend-resident production | `Ndt` | circumcenters / local predicates |
| `tet_face_seed` | scan/compact transient | `Ndt` | one bit/slot per candidate face emission |
| `tet_perm` | scan/compact transient | `Ndt` | stable order after rebuild or flip rounds |
| `tet_debug_flip_kind, tet_debug_predicate_state` | CPU-only debug | `Ndt` | counts and fallback tracing |

## 3) Candidate / halo points

| buffer | residency | shape | notes |
| --- | --- | --- | --- |
| `cand_dp_*` | backend-resident production | subset of `Ndp` | active local points used in rebuild |
| `halo_dp_*` | backend-resident production | subset of `Ndp` | ghost / export points needed for periodic or boundary closure |
| `cand_keep, halo_keep` | scan/compact transient | subset of `Ndp` | mark-sweep before prefix compaction |
| `cand_count, halo_count` | scan/compact transient | scalar / per-rank | used to size packed working sets |
| `cand_owner_key` | scan/compact transient | subset of `Ndp` | canonical repartition key for stable packing |
| `halo_image_shift` | backend-resident production | subset of `Ndp` | periodic image metadata for face ownership |

## 4) Voronoi face production

| buffer | residency | shape | notes |
| --- | --- | --- | --- |
| `face_c1, face_c2` | backend-resident production | `Nvf` | 1-based local cells; `c2 <= 0` means foreign/boundary |
| `face_area` | backend-resident production | `Nvf` | hydro face area |
| `face_cx, face_cy, face_cz` | backend-resident production | `Nvf` | face centroid |
| `face_nx, face_ny, face_nz` | backend-resident production | `Nvf` | unit normal from `c1` to `c2` |
| `face_vx, face_vy, face_vz` | backend-resident production | `Nvf` | moving-face velocity, if available |
| `face_owner_task, face_owner_index` | backend-resident production | `Nvf` | update-target ownership and export routing |
| `face_image_flags` | backend-resident production | `Nvf` | periodic/image bookkeeping |
| `face_emit_counts` | scan/compact transient | `Nvf` or `Ndt` | per-tet face candidates before dedup |
| `face_keep, face_duplicate` | scan/compact transient | `Nvf` | prune duplicates and zero-area faces |
| `face_debug_verts, face_debug_nv` | CPU-only debug | ragged | exact polygon rings for gate comparison |

## 5) Compact face table and CSR

| buffer | residency | shape | notes |
| --- | --- | --- | --- |
| `geom.c1, geom.c2` | backend-resident production | `Nvf` | compact face endpoints |
| `geom.cell_face_offsets` | backend-resident production | `Ncells + 1` | 1-based CSR row offsets |
| `geom.cell_faces` | backend-resident production | `2*Nvf` | face ids per cell incidence |
| `geom.cell_face_signs` | backend-resident production | `2*Nvf` | `-1` for `c1`, `+1` for `c2` |
| `geom.volume` | backend-resident production | `Ncells` | cell volumes |
| `geom.face_area` | backend-resident production | `Nvf` | copy of `face_area` after compaction |
| `geom.normal_x, geom.normal_y, geom.normal_z` | backend-resident production | `Nvf` | canonical face normals |
| `csr_row_counts` | scan/compact transient | `Ncells` | prefix-sum input |
| `csr_face_perm` | scan/compact transient | `2*Nvf` | stable gather order into `cell_faces` |
| `csr_debug_cell_faces` | CPU-only debug | `2*Nvf` | verbose incidence traces |

## 6) Canonical sort keys for CPU/GPU parity

1. **Points**: `(dp_task, dp_original_index, dp_image_flags, dp_index)`.
   Preserve stable input order only as a final tie-breaker.
2. **Tetrahedra**: `(sort(tet_p0..tet_p3), tet_deleted, tet_face_seed)`.
   Keep orientation in the stored `p0..p3` slots; canonical identity comes from
   the sorted vertex set.
3. **Faces**: `(min(face_c1, face_c2), max(face_c1, face_c2), face_image_flags,
   face_owner_task, face_owner_index)`.
   Boundary faces sort after interior faces when `face_c2 <= 0`.
4. **CSR rows**: sort each cell row by the same face key above, then by signed
   incidence (`-1` before `+1`) to make CPU and GPU gathers identical.

## 7) Practical rule

Keep the production path as SoA only.  Any ragged polygons, replay traces,
exact-predicate logs, or flip diagnostics stay in CPU-only debug buffers and
never enter the kernel-facing layout.
