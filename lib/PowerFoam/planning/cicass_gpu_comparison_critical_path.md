# PowerFoam → CICASS 4-code comparison on GPU: critical path

Date: 2026-06-23

**Goal.** Get PowerFoam to (A) give a real GPU single-step speedup estimate vs MPI-Arepo for
the grid-build + hydro, then (B) run the *identical* CICASS streaming-IC cosmology setup
(z=1000→20, 64³ fixed-res) as a 5th code in the MultiCode comparison alongside
Enzo-CPU / Enzo-GPU / RAMSES / Arepo.

This is the critical path only. Deep per-component detail already lives in:
`tessellator_mesh_workbreakdown.md`, `cosmology_gravity_workbreakdown.md`, `pm_gravity_audit.md`,
`hydro_parity_workbreakdown.md`, `arepo_jl_full_rewrite_master_plan.md`. This doc orders them
toward the two goals and states the acceptance gate for each.

## Measured starting point (2026-06-23, this box: EPYC 7763 + RTX A6000)
- PowerFoam grid-build (`local_periodic_voronoi_mesh_arrays_3d`) = ~14k cells/s, full step
  (`moving_mesh_step_3d!`, rebuild=:local+hydro) = ~7k cells/s — **CPU host, single process**
  (serial; threads didn't help). Extrapolated 64³: build ~19s, step ~37s.
- Arepo 64³: ~12–18s/step (16–64 MPI ranks); Voronoi build ~7s/step @16 ranks.
- ⇒ PowerFoam is currently ~2–3× slower; the grid build (Arepo's 78% `findpoints` bottleneck) is
  NOT on GPU. Hydro kernels ARE KA/GPU but **Metal-tested only**; no CUDA in src/examples.
- Hydro/mesh PARITY is certified via **trace replay** (fed Arepo geometry) at N4/N8; the native
  `run_step!` still consumes Arepo trace metadata (Blocking Gap #1 in the parity audit).

---

## Phase 0 — CUDA backend bring-up  (PREREQUISITE, small, ~0.5–1 day)
KA is backend-agnostic but only Metal has been exercised. Make the A6000 usable.
- Add a `maybe_cuda_backend()` (mirror `maybe_metal_backend()` in
  `examples/native_moving_solver_matrix_3d.jl`); select via `POWERFOAM_BACKEND=cuda`
  (`using CUDA; CUDABackend()`).
- Run the existing GPU hydro gates (`arepo_runtime_hydro_smoke`, `native_moving_solver_matrix_3d`,
  `arepo_solver_matrix_3d`) on CUDA; fix any unsupported ops (atomics, scalar indexing, Int type
  widths — watch `@atomic`, dynamic `getindex` on device).
- **Gate:** KA hydro step on CUDA matches the CPU-backend result to f32 round-off on the N8 mesh.
- **Deliverable:** a one-line GPU hydro throughput number (Mcell/s) on the A6000.

## Phase 1 — KA Delaunay/Voronoi rebuild on GPU  (THE CRUX, large, ~2–4 weeks)
This is the bottleneck and the whole point. The high-level builder is CPU/host; the
`tessellation3d.jl` SoA KA pieces exist but don't yet form a complete rebuild that "replaces the
current rebuild path" (per the file header). Building blocks present (all take a backend `be`):
`periodic_point_images_soa_3d`, `dense_candidate_pairs_soa_3d`, `pack_candidate_stencil_soa_3d`,
`candidate_tetra_predicates_soa_3d`. Missing: the full Delaunay construction (insertion/flip or
the chosen direct predicate path) → Voronoi extraction → `ArepoMeshArrays3D`, on-device.
- Complete the SoA KA Delaunay → Voronoi pipeline; expose a build entry point
  `local_periodic_voronoi_mesh_arrays_3d(be, points; ...)` that runs on CPU-KA and CUDA.
- Close the parity audit's **Blocking Gaps #2/#3** (native update-target face table + topology-
  equivalent rebuild) for the KA path.
- **Gates:** (a) KA-backend rebuild is topology-equivalent (faces/areas/normals/CSR) to the CPU
  rebuild on the N4/N8 trace-gate set (extend `arepo_tessellator_rebuild_gate_matrix`); (b) runs
  on CUDA at 32³/64³; (c) bit-/round-off-stable.
- **Deliverable → answers the user's question:** single-step grid-build + hydro on the A6000 at
  64³ → the real GPU speedup vs MPI-Arepo. *Phases 0+1 are the minimum to get the estimate.*

## Phase 2 — Self-driven `run_step!`  (medium-large, ~1–2 weeks)
Remove the dependence on Arepo trace metadata (Blocking Gap #1): construct the pass-local
snapshot sequence, post-drift generator positions, old/new volumes, and gas-cell reorder
boundaries from PowerFoam's own scheduler.
- **Gate:** standalone `run_step!` reproduces the traced 3-D decaying-turbulence run to the
  current parity level (~5e-13 HLL/LLF) WITHOUT consuming Arepo trace rows.

## Phase 3 — Cosmology: DM particles + self-gravity + comoving  (large, ~2–4 weeks)
None of this exists yet (PowerFoam is gas/mesh only; gravity = tiny-N scaffold + PM oracle,
periodic NOT implemented; cosmology = expansion-factor coefficients only).
- **DM particles:** collisionless N-body set (positions/velocities/mass), CIC deposit.
- **Self-gravity (periodic):** wire `PoissonKernels` (already a dep — periodic FFT Poisson,
  GPU) to deposit gas+DM → solve φ → accel on particles + gas. Reuse the Vespa-native pattern
  (`PoissonKernels` + the deposit/interp used by the KA cosmology stack). See `pm_gravity_audit.md`,
  `cosmology_gravity_workbreakdown.md`.
- **Comoving integration:** super-comoving kick-drift + the expansion factor in the timestep
  (coefficients already in `arepo_cosmology_coefficients.jl`).
- **Gates:** Zel'dovich pancake exact-growth (mirror `Vespa/test_zeldovich_pancake.jl`); linear
  growth D(a) on large scales matches analytic to ~%.

## Phase 4 — CICASS IC + MultiCode wiring  (medium, ~1 week)
- Read the shared CICASS realization (gas δ_b + DM displaced particles + streaming velocity) into
  PowerFoam — reuse `CICASSLib`/the Gadget IC reader the Arepo driver uses; apply the proper
  pressure-suppressed `gas_vel` (NOT CDM velocity — the bug we just fixed in RAMSES).
- Write `lib/MultiCode/examples/cicass_powerfoam_pk.jl` mirroring `cicass_arepo_pk.jl`: inject ICs,
  evolve z=1000→20 at fixed resolution, measure baryon+DM P(k) at the CIC_ZOUT redshifts, write
  `cicass_powerfoam_pk_<TAG>.dat` in the comparison format. Add MultiCode registration.
- **Gates:** (a) on a non-moving (fixed-mesh) baryon test, agrees with Enzo/Arepo; (b) DM tracks
  the CICASS linear growth like RAMSES-fine/Enzo (~1% at z=20); (c) appears in `compare_5way.py`
  (extend to 6-way) tracking the others.

---

## Recommended order & decision points
1. **Phases 0 + 1 first** — they deliver the speedup estimate the comparison decision hinges on.
   If the GPU Voronoi rebuild does NOT beat ~16-rank MPI-Arepo (Voronoi parallelizes worse than
   stencil hydro — expect a more modest win than the >1000 Mcell/s hydro kernels), reconsider
   before investing in Phases 2–4.
2. Phase 2 (self-driven step) is independent of cosmology and de-risks everything downstream.
3. Phases 3–4 only after 0–2 are green.

## Effort summary
| phase | scope | size | unlocks |
|---|---|---|---|
| 0 | CUDA backend bring-up | S | GPU hydro number |
| 1 | KA Delaunay/Voronoi rebuild on GPU | **L** | **the speedup estimate** |
| 2 | self-driven run_step! | M–L | standalone runs |
| 3 | DM + self-gravity + comoving | L | cosmology |
| 4 | CICASS IC + MultiCode wiring | M | 5th code in the comparison |
