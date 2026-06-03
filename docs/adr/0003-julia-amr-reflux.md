# ADR-0003: Conservative `:julia` hydro under Enzo AMR (the SubgridFluxes contract)

- **Status:** Proposed (design; implementation is a dedicated effort)
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

### A. EnzoNG-side flux recording (it currently discards face fluxes)

`accumulate_flux!` computes `F·area` per face and folds it straight into `acc`,
keeping nothing. Add an optional **boundary-flux sink**: when the cell on one side
of a face is OUTSIDE the active region (an outer-boundary face) record `(field, dim,
side, active-index, F)` — EnzoNG already has the machinery shape for this in its
own `FluxRegister`/`reflux.jl` (used for composite AMR), so reuse the per-face
capture, keyed here by Enzo grid + face instead of the composite mesh. The result
is a per-(dim,side) plane of `nvars` fluxes over the grid's boundary — exactly the
`Left/RightFluxes[field][dim]` Enzo wants.

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

## Why this is a dedicated effort, not a tail-end task

The flux extents are in **global** index space with per-(dim,side) planes and a
coarse/fine role split; an off-by-one in the `*GlobalIndex` mapping yields *silent*
non-conservation, not a crash. It needs the EnzoNG recording side, ~4 bridge
functions + a dylib rebuild, and the reflux-conservation test as the gate. Doing it
right is worth a focused pass; a non-conservative shortcut is worse than not doing
it.
