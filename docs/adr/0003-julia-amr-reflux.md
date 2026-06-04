# ADR-0003: Conservative `:julia` hydro under Enzo AMR (the SubgridFluxes contract)

- **Status:** DONE (1D) — Part A (EnzoNG boundary-flux recording) + Part B (the
  bridge write + orchestration) implemented and proven exactly conservative. The
  flux bridge conserves the composite mass/energy to **round-off** on a static
  multi-level (5-level) hierarchy while the waves are interior to the refined
  region (`7.9e-16` vs `1.3e-3` with the correction disabled). Validated by
  `lib/EnzoLib/test/test_julia_reflux.jl`. **Follow-up #2 (ND face planes) is now
  DONE** — the Julia plane assembly rasterizes 2D/3D coarse–fine face planes
  (`test_julia_reflux_2d.jl`: 2D Sod-AMR strip conserves to round-off `5.9e-15`).
  **Parent-ghost coupling is DONE in 1D AND ND** — EnzoNG consumes Enzo's
  parent-interpolated ghosts at a subgrid's coarse–fine faces; the ND fix selects
  ParentGhost per (axis, side) so a subgrid's real domain-boundary faces keep their
  domain BC (2D subtest C: parent-ghost ON `1.47e-6` vs broken blanket `1.5e-4` vs
  no-reflux `1.2e-3`). The residual ~1e-6 is the boundary-interpolation accuracy,
  NOT the flux bridge.
- **Date:** 2026-06-03
- **Builds on:** ADR-0002 (method-slot registry), the ND single-grid `EnzoBackend`
  (`b15d99a2`), and the `set_acceleration_field` bridge (`5d917e0c`).

---

## Context

ND single-grid is done: EnzoNG's unchanged driver runs on a live 2D/3D Enzo grid
(round-trip identity exact, validated on NohProblem2D). The remaining piece for
"`:julia` physics under AMR" is **conservative coarse–fine coupling** when a
`:julia` hydro slot replaces `SolveHydroEquations` on a refined hierarchy.

Today `evolve_level!` gates Enzo's flux machinery
(`clear/create/finalize_fluxes`, `update_from_finer`) to `eng.hydro === :enzo`,
because a `:julia` hydro does not fill Enzo's flux registers and `finalize_fluxes`
then segfaults on the uninitialised `fluxes` arrays. This ADR records the exact
contract to lift that gate **conservatively** — not a non-conservative half-version
(which gives silently wrong coarse-grid totals at coarse–fine boundaries).

## The Enzo flux structures (verified)

- `struct fluxes` (`src/enzo/Fluxes.h`): per box-face flux storage —
  `LeftFluxes[field][dim]` / `RightFluxes[field][dim]` (the flux arrays) plus
  `Left/RightFlux{Start,End}GlobalIndex[dim][dim]` (the face extents in **global**
  index space at that level's zone width).
- `grid::CorrectForRefinedFluxes(InitialFluxes, RefinedFluxes, …)` is the
  consumer: it replaces the coarse grid's state change at the interface with the
  difference `(RefinedFlux − InitialFlux)` — restoring conservation.
- `SolveHydroEquations` fills BOTH roles as it sweeps:
  - the COARSE grid's flux at the cells under each subgrid → `SubgridFluxesEstimate[grid][subgrid]` (the `InitialFluxes`),
  - each grid's outer-boundary flux → its `BoundaryFluxes` (the `RefinedFluxes`,
    accumulated over the fine grid's subcycles via the register `scale`).

So conservative `:julia` AMR requires EnzoNG to produce **both** flux sets in
Enzo's exact format and extents.

## Design

Two halves: (A) EnzoNG records per-face boundary fluxes; (B) a bridge writes them
into Enzo's `fluxes` at matched global-index extents.

### A. EnzoNG-side flux recording — DONE

`BoundaryFluxRegister{NV,T}` (`src/reflux.jl`): `accumulate_flux!`/`step!` accept a
`bflux` sink; the two boundary `_flux_face!` methods call `_bflux_capture!` with
`scale=½dt` per SSP-RK2 stage, accumulating `∫F·area dt` keyed by `(axis, side,
boundary-cell)` — exactly the `Left/RightFluxes[field][dim]` plane Enzo wants, and
consistent with the gas update (same ½(F₁+F₂)·dt). Default `nothing` ⇒ the f64 path
is untouched. Validated by the conservation identity `Δ(total mass) == Σ_lo bflux[ρ]
− Σ_hi bflux[ρ]` to round-off (`test/test_boundary_flux.jl`, 5.5e-17 over 20 steps).

### B. Verified write target (traced from the Enzo source)

- The fine grid's flux lives in its `grid::BoundaryFluxes` member (`fluxes*`);
  `update_from_finer` reads it via `GetProjectedBoundaryFluxes(parent, refined)`
  which projects it to the COARSE index space, then `CorrectForRefinedFluxes(
  InitialFluxes=SubgridFluxesEstimate[coarse][subgrid], RefinedFluxes=that
  projection, …)` applies `coarse += (RefinedFlux − InitialFlux)`.
- `Grid_SolveHydroEquations.C:328-348` is the exact format to match: per `(field,
  dim)`, `LeftFluxes[field][dim] = new float[plane_size]` over the face plane, with
  `Left/RightFluxStartGlobalIndex[dim][j]` the plane corner in **global** zone index
  at this level. EnzoNG's `bflux` (axis,side,cell)→∫F·area dt maps to it as
  axis=dim, :lo→Left, :hi→Right, cell→(plane index, global-offset by the grid's
  GridStartIndex/GridLeftEdge). Need bridges for the grid's global start + the
  subgrid boxes to compute the extents.
- **The failure mode is NOT silent given the gate:** a wrong index/sign/unit shows
  as the `test_reflux` conservation drift jumping from ~1e-13 (round-off) to ~1e-3
  (~0.45%, the documented reflux signature) — so part B is validatable by running a
  2-level `:julia` AMR and asserting mass/energy conservation. Build B against that
  assertion, not by eyeballing the index math.

### B. The bridge (mirrors set_acceleration_field)

Grid methods + bridge fns + bindings:
- `problem_grid_left/right_edge(h, gi)` and `problem_grid_start_index(h, gi)` →
  the grid's global-index origin + ghost offset, so EnzoNG can compute the global
  `Left/RightFluxStartGlobalIndex` for each boundary plane.
- `problem_num_subgrids(h, level, gi)` + `problem_subgrid_box(h, level, gi, si)` →
  the coarse grid's subgrid extents (which boundary planes are coarse–fine
  interfaces, for the coarse `InitialFluxes`).
- `problem_set_boundary_flux(h, level, gi, field, dim, side, double *plane)` →
  write one boundary-flux plane into the grid's `BoundaryFluxes` (allocating like
  `set_acceleration` does), with the matching `*GlobalIndex` set from (B).
- `problem_set_subgrid_flux(h, level, gi, si, field, dim, side, double *plane)` →
  the coarse `InitialFluxes` at a subgrid face.

### Orchestration (`evolve_level!`)

Replace the blanket `ef = eng.hydro === :enzo` gate with: for `:julia` hydro,
still run `create_fluxes`/`finalize_fluxes`/`update_from_finer`, but the `:julia`
hydro hook (after its per-grid solve) writes the recorded boundary + subgrid
fluxes via (B). Then Enzo's `update_from_finer`/`CorrectForRefinedFluxes` does the
projection AND the conservation correction using EnzoNG's fluxes — identical
machinery, EnzoNG's numbers.

### Prerequisite (cheap): grid→level + multi-grid slot

`problem_grid_level(h, gi)` (the EMProblem grid list matches `LevelArray` by
pointer via `collect_grids`, so a per-level membership scan works) so the `:julia`
hydro hook iterates every grid on the level, building/caching an `EnzoGridMesh`
per grid index (the benchmark hooks already do handle-aware rebuilds).

## Verification

A 2-level refined Sod / Sedov with `hydro=:julia` must conserve mass/energy to the
same `~1e-13` the `:enzo` AMR path does (`test_reflux.jl` is the template). The
decisive check: disabling (B) takes the drift from round-off to ~1e-3 (the same
signature reflux has on EnzoNG's own composite AMR), proving the correction is what
restores conservation.

## What was built (the implementation, verified)

The contract above was implemented exactly. Producer/consumer were traced from the
Enzo source and the index mapping was validated end-to-end against a conservation
assertion (not by eyeballing):

- **C-ABI bridge** (`Grid_EnzoModulesFixture.C` grid methods + `enzomodules_problem_
  bridge.C` `extern "C"` fns + `session.jl` bindings; rebuilt via
  `build_grid_darwin.sh`): `EnzoModulesGlobalStart` (active-region global zone
  index), `EnzoModulesGridEdge` (per-grid physical edges → level-dependent cell
  width), `BoundaryFlux{Size,Extent,Set,Get}`, and `problem_{num_subgrids,
  subgrid_flux_extent,subgrid_flux_size,set/get_subgrid_flux,grid_index_on_level}`.
- **Two flux sets filled** (exactly what `SolveHydroEquations` fills): the coarse
  `SubgridFluxesEstimate[level][i][sub]` — proper subgrids = the coarse
  InitialFluxes at the coarse–fine faces (EnzoNG's INTERIOR flux there, looked up
  by the subgrid's coarse-index extents), and the **last** entry = the grid's own
  outer flux, which `FinalizeFluxes`→`AddToBoundaryFluxes` accumulates into the
  grid's `BoundaryFluxes` (giving the correct temporal accumulation across
  subcycles for free). This last-entry path is required: `AddToBoundaryFluxes`
  derefs every baryon field of it, so ALL fields must be allocated (mapped value or
  zero) — the segfault the old gate avoided.
- **Units / sign / field map** (verified): Enzo stores `F·dt/dx` (a conserved-
  density change), so `enzo_value = bflux/V_cell` (bflux = `∫F·area dt`); +axis
  sign throughout; EnzoNG conserved component → Enzo BaryonField via the mesh's
  role map (`cdi→di`, `cei→ei`, `cmom[d]→vi[d]`).
- **Orchestration** (`session.jl`): `EngineConfig(reflux=true)` lifts the
  `ef = hydro===:enzo` gate to also run `clear/create/update_from_finer/finalize`
  for a conservative `:julia` hydro; the hook (`test_julia_reflux.jl`) iterates the
  grids on each level, runs EnzoNG's driver, and writes the fluxes. Enzo's own
  `UpdateFromFinerGrids`/`CorrectForRefinedFluxes` then restore conservation —
  identical machinery, EnzoNG's numbers.

**Result:** static 5-level Sod, waves interior — composite mass drift `7.9e-16`
(WITH) vs `1.3e-3` (WITHOUT). Per-step: round-off every step until a wave sits ON a
coarse–fine boundary, where it jumps to ~1e-3 — the signature of a small *relative*
flux error scaled by flux magnitude, i.e. EnzoNG's Outflow ghost at the fine
boundary, not the bridge.

## Follow-ups

- **ND face planes — DONE.** The C++ bridge was already ND-general (sizes/extents
  over `GridRank`). The Julia plane assembly is now ND too: `EnzoNG.bflux_plane`
  (`src/reflux.jl`) rasterizes one `(dim, side)` face plane in Enzo's exact
  linearization (column-major over the orthogonal dims, dim-0 fastest, flux dim
  collapsed — the `Grid_CorrectForRefinedFluxes.C:460` `FluxIndex`), mapping each
  plane cell's global index to EnzoNG's per-cell boundary/interior flux register
  (orthogonal dims map straight `g−g0+1`; the flux-dim key follows the verified 1D
  mapping). `test_julia_reflux.jl`'s `_write_fluxes!` is now `EnzoGridMesh{R}`
  generic. **Result (2D Sod-AMR strip, `test_julia_reflux_2d.jl`, waves interior):**
  composite mass drift `5.9e-15` / energy `3.3e-14` (WITH) vs `1.2e-3` / `3.0e-3`
  (WITHOUT) — round-off, on genuine 60-cell and 4-cell face planes. 1D unchanged
  (bit-identical `7.9e-16`).
- **Parent-ghost coupling (accuracy + the residual conservation). — DONE.** EnzoNG
  now consumes Enzo's parent-interpolated ghost zones at a subgrid's coarse–fine
  faces instead of an Outflow copy. Mechanism: a `ParentGhost{F}` BC
  (`MeshInterface`) carrying a closure `(axis, side, cell) -> W_prim`; the driver's
  `_boundary_ghost` resolves it (the existing Outflow/Reflecting/Periodic paths are
  byte-unchanged — core stays 203/203). The closure reads the live Enzo grid's
  already-interpolated ghost zone one zone OUTWARD of the boundary active cell
  (`enzo_parent_ghost` in `EnzoBackend`, using the existing `problem_get_field`
  flat array that already includes ghosts — no new C-ABI bridge function needed),
  returns it as a conserved role-ordered tuple, and the hook converts to primitive.
  Applied only on level>0 grids (the root's outer faces are real domain BCs).
  **Result (subtest C):** end-to-end mass drift `1.79e-5` (Outflow) → `8.05e-6`
  (parent-ghost), a ~2.2× reduction; energy `2.05e-5` → `9.13e-6`. The flux bridge
  stays exactly conservative (subtest A unchanged at `7.9e-16`) — this is the
  boundary-ACCURACY half, not the conservation half. The sign/index is gated
  honestly: the negative control (reading the ghost on the WRONG side) makes drift
  WORSE than Outflow (`2.63e-5`), so a wrong index/sign FAILS the assertion. (The
  residual `~8e-6` is the coarse↔fine interpolation accuracy itself — EnzoNG reads
  the innermost interpolated layer; Enzo's multi-layer ghost + its own boundary
  reconstruction differ at higher order. Reaching round-off would require matching
  Enzo's reconstruction exactly, which is beyond consuming the parent ghost.)
- **ND parent-ghost — DONE.** The reader itself (`enzo_parent_ghost` in
  `EnzoBackend`) was already ND-general: the driver invokes the `ParentGhost` BC at
  every boundary cell's `CartesianIndex`, and the closure reads one zone outward via
  `m.strides[axis]` — so the (D−1)-plane of interpolated ghosts is read cell-by-cell
  for free in any D. The actual ND bug was in the BC SELECTION: the old
  `_apply_parent_ghost!` blanket-replaced ALL of a level>0 grid's outer faces with
  ParentGhost. In ND a subgrid's faces are a MIX — the 2D Sod strip's x-faces are
  coarse–fine interfaces, but its y-faces span the full domain and ARE the (Outflow)
  domain boundary. Putting a parent ghost on those domain faces broke conservation
  (the static-interior 2D drift blew up from round-off to `~1.5e-4`, almost the
  no-reflux `~1.2e-3`). The fix decides PER (axis, side): a face whose grid edge
  coincides with the domain edge (`problem_grid_edge` vs the root grid's extent)
  keeps the real domain BC; an interior face gets ParentGhost. In 1D every level>0
  grid is interior on both faces, so this reduces EXACTLY to the blanket replacement
  (1D bit-identical: static `7.894919286223351e-16`, end-to-end subtest C `8.07e-6`
  vs Outflow `1.79e-5`, ratio 2.22×). **Result (2D, `test_julia_reflux_2d.jl`
  subtest C, static interior):** parent-ghost ON mass drift `1.47e-6` / energy
  `1.87e-6` — `~100×` better than the broken blanket version (`1.5e-4`) and `~800×`
  below the no-reflux signature (`1.2e-3`). NOT round-off (the raster-only path stays
  `5.9e-15`): the residual is the same coarse↔fine interpolation accuracy the 1D
  end-to-end parent-ghost shows — EnzoNG reads the innermost interpolated layer,
  Enzo's multi-layer reconstruction differs at higher order. The flux bridge stays
  EXACTLY conservative; this is the boundary-ACCURACY half in ND.

## Why this is a dedicated effort, not a tail-end task

The flux extents are in **global** index space with per-(dim,side) planes and a
coarse/fine role split; an off-by-one in the `*GlobalIndex` mapping yields *silent*
non-conservation, not a crash. It needs the EnzoNG recording side, ~4 bridge
functions + a dylib rebuild, and the reflux-conservation test as the gate. Doing it
right is worth a focused pass; a non-conservative shortcut is worse than not doing
it.
