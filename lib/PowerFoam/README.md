# PowerFoam.jl

`PowerFoam.jl` is a small 2-D prototype for AREPO-compatible power/Laguerre
meshes.  It is deliberately dependency-light and exposes geometry in an
AREPO-shaped surface: cell polygons, neighbor pairs, face lengths, face centers,
outward normals, and segment vertices.

The first target is controlled comparison:

- generate ordinary Voronoi meshes by setting all weights to zero;
- generate weighted power meshes by assigning per-generator weights;
- import AREPO's exact 2-D polygon export with `from_arepo_polygons`;
- compare mesh-quality and reconstruction metrics before touching AREPO's C
  mesh builder.

```julia
using PowerFoam

patch = refine_patch(16, 16; refine_radius = 0.18)
vor = power_diagram(PowerSites2D(patch.points))
relaxed = relax_weights(patch.points, patch.target_areas;
                        smooth_strength = 0.5, smooth_passes = 1).mesh

aligned = relax_points_velocity_alignment(patch.points;
                                          velocity_field = xy -> (xy[1] - 0.5, xy[2] - 0.5),
                                          strength = 0.12, steps = 6).mesh

mesh_quality(vor)
mesh_quality(relaxed)
face_velocity_alignment(aligned; velocity_field = xy -> (xy[1] - 0.5, xy[2] - 0.5))
arepo_face_table(relaxed)
```

AREPO comparison path:

```julia
polys = ArepoLib.get_voronoi_2d(handle)
cells = ArepoLib.get_cell_field(handle, :center)[:, 1:2]
mesh = from_arepo_polygons(polys; generators = cells)
```

The importer does not reconstruct the mesh; it preserves AREPO's own polygon
vertices.  That gives a direct baseline for testing whether the power-diagram
prototype changes geometry, reconstruction conditioning, and refinement-boundary
quality in useful ways.

AREPO-shaped hydro kernel prototype:

```julia
using PowerFoam

pts = [0.25 0.5;
       0.75 0.5]
mesh = power_diagram(PowerSites2D(pts))
geom = arepo_mesh_arrays(mesh; T = Float64)
state = euler_state_2d(mesh; rho = [1.0, 2.0], pressure = 1.0, gamma = 1.4)
finite_volume_step_2d!(state, geom; dt = 0.01, gamma = 1.4, riemann = :hll)
```

The first hydro rung is a first-order 2-D Euler update over the AREPO-like face
table.  It uses KernelAbstractions kernels split into face-flux computation and
cell-wise CSR flux gathering, so the same code path runs on CPU arrays and on
GPU arrays staged with a backend such as Metal.  It is a solver/data-layout seam,
not yet AREPO's full moving-mesh reconstruction.

Moving-mesh ALE step:

```julia
state = euler_state_2d(mesh; rho = [1.0, 2.0], pressure = 1.0, gamma = 1.4)
moved = moving_mesh_step_2d!(state, mesh;
                             dt = 0.01, gamma = 1.4,
                             mesh_velocity = [0.1 0.0; 0.1 0.0],
                             riemann = :hll)
mesh = moved.mesh
```

`moving_mesh_step_2d!` computes ALE fluxes on the old moving faces, advects the
generators, rebuilds the bounded Voronoi mesh, and divides the updated conserved
cell integrals by the new cell volumes.  The face/update kernels still run
through KernelAbstractions; the current mesh rebuild is host-side.

Paper-inspired examples:

```bash
julia --project=lib/PowerFoam lib/PowerFoam/examples/exponential_disk_regularization.jl
julia --project=lib/PowerFoam lib/PowerFoam/examples/sedov_blast_mesh.jl
julia --project=lib/PowerFoam lib/PowerFoam/examples/convergent_shock_semilagrangian.jl
julia --project=lib/PowerFoam lib/PowerFoam/examples/moving_mesh_contact.jl
julia --project=lib/PowerFoam lib/PowerFoam/examples/turbulence_gpu_parity_2d.jl 12 5.0 0.02
julia --project=lib/PowerFoam lib/PowerFoam/examples/turbulence_gpu_parity_3d.jl 12 5.0 0.02
julia --project=lib/PowerFoam lib/PowerFoam/examples/native_rebuild_gate_3d.jl 3 0.02 periodic
AREPO_LIB=/Users/tabel/Projects/arepo/libarepo3d.dylib \
  julia --project=/private/tmp/powerfoam_arepo_env \
  lib/PowerFoam/examples/arepo_geometry_gate_3d.jl 12 0.001 hll 8
AREPO_LIB=/Users/tabel/Projects/arepo/libarepo3d.dylib \
  julia --project=/private/tmp/powerfoam_arepo_env \
  lib/PowerFoam/examples/arepo_solver_matrix_3d.jl 12 0.001 8,16,32 hll,llf
JULIA_NUM_THREADS=4 POWERFOAM_METAL_STORAGE=shared POWERFOAM_DIAGNOSTICS=final \
JULIA_LOAD_PATH=@:lib/PowerFoam:@stdlib \
  julia --project=lib/MultiCode/test \
  lib/PowerFoam/examples/native_moving_solver_matrix_3d.jl 16 0.001 8 hll 1 reconstruct
JULIA_NUM_THREADS=4 POWERFOAM_METAL_STORAGE=shared POWERFOAM_DIAGNOSTICS=final \
JULIA_LOAD_PATH=@:lib/PowerFoam:@stdlib \
  julia --project=lib/MultiCode/test \
  lib/PowerFoam/examples/native_moving_solver_matrix_3d.jl 16 0.001 8 llf 1 reconstruct
JULIA_NUM_THREADS=4 POWERFOAM_METAL_STORAGE=shared POWERFOAM_DIAGNOSTICS=final \
POWERFOAM_REBUILD=gpu_fixed JULIA_LOAD_PATH=@:lib/PowerFoam:@stdlib \
  julia --project=lib/MultiCode/test \
  lib/PowerFoam/examples/native_moving_solver_matrix_3d.jl 32 0.001 8 hll,llf 1 reconstruct
JULIA_NUM_THREADS=4 POWERFOAM_METAL_STORAGE=shared POWERFOAM_DIAGNOSTICS=final \
POWERFOAM_REBUILD=gpu_local JULIA_LOAD_PATH=@:lib/PowerFoam:@stdlib \
  julia --project=lib/MultiCode/test \
  lib/PowerFoam/examples/native_moving_solver_matrix_3d.jl 16 0.001 8 hll,llf 1 reconstruct
JULIA_NUM_THREADS=4 POWERFOAM_METAL_STORAGE=shared POWERFOAM_DIAGNOSTICS=final \
POWERFOAM_REBUILD=gpu_compact JULIA_LOAD_PATH=@:lib/PowerFoam:@stdlib \
  julia --project=lib/MultiCode/test \
  lib/PowerFoam/examples/native_moving_solver_matrix_3d.jl 16 0.001 8 hll,llf 1 reconstruct
```

These examples write SVG diagnostics under `lib/PowerFoam/examples/out/`.

The turbulence parity example writes CSV and Markdown reports under
`lib/PowerFoam/examples/out/turbulence_gpu_parity_2d/`.  Set
`POWERFOAM_BACKEND=metal` and run it from an environment containing Metal.jl to
exercise the Apple GPU path.  This is currently a 2-D bounded-mesh GPU parity
gate; use the 3-D gates below for moving-face and dynamic-rebuild comparisons.

The 3-D turbulence parity example is the next rung: it uses a periodic Cartesian
Voronoi-equivalent face table with the same 3-D HLL/LLF flux/update kernels that
the AREPO 3-D face-ring importer feeds.  It validates the GPU hydro path before
claiming full moving-Voronoi AREPO parity.

The AREPO geometry gate initializes the stock 3-D turbulence case with the
3-D AREPO library, exports the live Voronoi face rings through
`ArepoLib.get_voronoi_3d`, converts them to `ArepoMeshArrays3D`, and runs one
or more tiny PowerFoam HLL/LLF steps on CPU and Metal.  The optional fourth
argument is the first-order step count.  The same report also runs a one-step
reconstructed predictor gate using AREPO's exported hydro gradients, records the
AREPO-style hydro Courant timestep, and asks AREPO's native 3-D tessellator to
advance once and re-export the rebuilt mesh.  That last rebuild is still hosted
by AREPO for the production-size turbulence gate.

The AREPO solver matrix reruns that geometry gate for a list of Julia Riemann
solver choices and one or more step counts, then writes a combined GPU
comparison.  It is the right artifact for measuring immediate solver-choice
impact on AREPO's moving face table.  It is not yet a full end-to-end native GPU
AREPO evolution because that particular comparison still uses AREPO's
production-size 3-D tessellator for the rebuild row.

The native rebuild gate exercises the first Julia 3-D mesh rebuild paths.
`bounded_voronoi_mesh_arrays_3d` clips each small diagnostic cell by all
pairwise bisectors and the unit-box boundary.  `periodic_voronoi_mesh_arrays_3d`
uses periodic generator images and keeps periodic duplicate faces, matching the
topology expected by the turbulence face-table path on small meshes.  Both
convert to the same `ArepoMeshArrays3D` layout and can drive one
`moving_mesh_step_3d!` ALE update.  These rebuilds are deliberately all-pairs
contract gates rather than the final AREPO-scale Delaunay-backed tessellator.
`local_periodic_voronoi_mesh_arrays_3d` adds the first scalable native rebuild
rung for near-lattice periodic turbulence boxes by clipping against a local
periodic bin stencil.  `native_moving_solver_matrix_3d.jl` uses that producer to
rebuild every step in Julia and compare solver choices on CPU and Metal without
calling AREPO's tessellator.  Its last argument selects the hydro rung:
`first` uses the first-order ALE update, while `reconstruct` computes limited
gradients from the native face table and uses the predictor before updating into
the newly rebuilt volumes.  The report includes end-to-end CPU and Metal timing;
at this stage those timings include the host-side rebuild and host/device
staging, not just GPU kernel time.  Set `POWERFOAM_REBUILD=gpu_fixed` to keep
the initial topology, face areas, and cell volumes fixed while the backend
advects generators and refreshes face centers, normals, and mesh velocities.
That mode is a GPU-resident moving-geometry performance rung, not the final
topology-changing local Voronoi tessellator.  Set `POWERFOAM_REBUILD=gpu_local`
for the first topology-changing GPU port: it keeps a fixed-capacity local
candidate graph, clips each candidate bisector against the local periodic
halfspace stencil on the backend, and represents inactive candidate faces with
zero area.  It also accumulates cell volumes from active faces on the backend.
Set `POWERFOAM_ACTIVE_CELLS=gradients` or `all` to exercise the fixed-stride
device active-face traversal experiment; it is intentionally off by default
because the current indirect active-list loops are slower than the CSR-capacity
walk on the N16 Metal gate.  The N12 reconstructed runs remain useful small
correctness gates.  The current exact local-rebuild GPU break-even gate is N16,
the GPU-fixed rebuild crossover is N24, and the default GPU-local N16 gate
reaches about 1.30x for HLL and 1.12x for LLF with four Julia threads, final
diagnostics, and shared Metal storage; see
`examples/out/native_moving_solver_matrix_3d/gpu_break_even_summary.md`.
Set `POWERFOAM_REBUILD=gpu_compact` for the first device-side compact
face/CSR rebuild baseline.  It compacts active candidate faces and rebuilds the
cell-face CSR on the backend with chunked hierarchical scans and no atomics.
The compact CSR row rebuild now uses per-cell parallel incident-face counting
before the chunk-local row scan; set `POWERFOAM_COMPACT_CELL_SCAN_MODE=chunked`
to recover the older launch-fused scan for A/B profiling.
It also keeps separate old/new compact geometry buffers, so gradients and
fluxes use the old face table while the conservative update writes into the
newly advected cell volumes.  On the current N16 reconstructed gate it reaches
about 2.21x for HLL and 1.57x for LLF, and the cleaner N24 gate reaches about
2.13x for HLL and 2.03x for LLF, with exact CPU/Metal final fields.  The local
candidate face clipper uses lane-local scratch with a default Metal workgroup
of 16; set `POWERFOAM_FACE_CLIP_WORKGROUP` to tune that occupancy knob.
`POWERFOAM_PLANE_CULL=true` keeps the same halfspace result but avoids full
polygon clipping when a plane leaves the current polygon wholly inside or
wholly outside.  `POWERFOAM_MESH_WORK_STATS=true` records operation counts for
newly advected compact rebuilds, including dirty cells/faces, planes per face,
and the inside/empty/clipped plane fractions.  The dirty mask is controlled by
`POWERFOAM_DIRTY_MOTION_THRESHOLD` and is the current scaffold for topology
coherence and hierarchical active-cell rebuilds.  `POWERFOAM_CANDIDATE_TIER`
is an experimental lattice-near candidate stencil knob (`full`, `axis_edge`,
or `axial`); the default remains `full`.
After each hydro step, the newly advected compact geometry rotates into the
old-geometry slot for the next step; the next step refreshes only compact face
velocities before rebuilding the next geometry.  That is the GPU-resident
geometry cadence needed for hierarchical timestepping.

AREPO Sedov proxy:

```bash
julia --project=lib/PowerFoam lib/PowerFoam/examples/arepo_sedov_proxy/generate_tables.jl 64
python lib/PowerFoam/examples/arepo_sedov_proxy/write_arepo_cases.py \
  --tables lib/PowerFoam/examples/arepo_sedov_proxy/out \
  --dest /private/tmp/powerfoam_arepo_sedov \
  --arepo /Users/tabel/Projects/arepo
python lib/PowerFoam/examples/arepo_sedov_proxy/profile_snapshots.py \
  --root /private/tmp/powerfoam_arepo_sedov --snapshot 1
```

This first hydrodynamic rung uses PowerFoam to prepare matched generator
layouts and exact cell-area masses, then lets AREPO build its normal Voronoi
mesh.  It is a proxy for the eventual weighted-cell backend, not a replacement
for it.

AREPO Noh proxy:

```bash
julia --project=lib/PowerFoam lib/PowerFoam/examples/arepo_noh_proxy/generate_tables.jl 64
python lib/PowerFoam/examples/arepo_noh_proxy/write_arepo_cases.py \
  --tables lib/PowerFoam/examples/arepo_noh_proxy/out \
  --dest /private/tmp/powerfoam_arepo_noh_64 \
  --arepo /Users/tabel/Projects/arepo
python lib/PowerFoam/examples/arepo_noh_proxy/profile_snapshots.py \
  --root /private/tmp/powerfoam_arepo_noh_64 --snapshot 3
```

This follows AREPO's `noh_2d` example constants and plots every cell's radial
density against the analytic solution, which is the diagnostic most sensitive
to mesh-induced Noh noise.

The Noh table generator also has an experimental flow-alignment pass:

```bash
julia --project=lib/PowerFoam lib/PowerFoam/examples/arepo_noh_proxy/generate_tables.jl \
  64 2.0 0.18 0.26 exact 0.12 6
```

The final two arguments are generator alignment strength and number of alignment
steps.  This moves the PowerFoam generators so interior face normals prefer
either radial or tangential orientation relative to the Noh inflow; the default
strength is `0.0`, preserving the original mesh.
