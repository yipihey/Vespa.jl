# CLAUDE.md — EnzoNG.jl

Next-generation Enzo: shared-memory AMR astrophysics, **orchestration in Julia**
over a **swappable AMR substrate behind one interface**. This file is the working
guide for developing in this tree. Read `docs/adr/0001-architecture.md` for the
"why"; this file is the "how + gotchas".

## What exists (status)

- **Hydro, two backends, 70/70 tests green.** Ghost-free finite-volume HLLC +
  PLM + SSP-RK2, run identically on `RefMesh` (pure-Julia oracle) and
  `HGBackend` (HierarchicalGrids.jl adapter). Cross-backend agreement test
  proves identical physics.
- **Hierarchical AMR works on HGBackend.** Dynamic regridding via a Julia
  `RefinementPolicy`; conservative remap-on-refine; hanging-node coarse↔fine
  sub-face fluxes. Validated by refined Sod (conservation + convergence vs a
  uniform-fine RefMesh) and the 2D Sedov blast (conservation, symmetry,
  R∝t^½ growth).
- **Inline visualization (`lib/EnzoViz`).** Taps `evolve!`'s callback, rasterizes
  snapshots, renders via the veusz fork's GPU Vello painter, emits a
  self-contained interactive web page per problem. 15/15 EnzoViz tests green.

Not yet done (per ADR build sequence): MHD constrained transport (next, highest
risk), self-gravity, cooling/Grackle, cosmology; then Rust/GPU backends where
measured; Metal milestone-2.

## Layout

```
lib/MeshInterface  THE SEAM. AbstractMeshBackend + generic fns only (no methods):
                   topology, integer-exact geometry, neighbor-with-BC, for_each_cell,
                   for_each_face (hanging-node aware), cell-average fields + layouts,
                   restrict!/prolong!, Instrumented{B}. Solver code depends ONLY here.
lib/RefMesh        Uniform pure-Julia backend + correctness oracle. CartesianIndex handles.
lib/HGBackend      Adapter over HierarchicalGrids.jl (the target substrate). Int handles.
                   AMR, hanging nodes, AdaptiveField remap. Validated vs RefMesh.
lib/EnzoViz        Visualization (PythonCall → veusz/Vello → <veusz-figure> page).
                   Heavy Python/Rust/WASM toolchain isolated here; core stays headless.
src/               Science layer (deps: MeshInterface ONLY). eos, riemann (HLLC),
                   reconstruct (PLM/minmod), driver (Simulation/evolve!/step!/regrid!),
                   problem (Problem = source code), diagnostics, exact_riemann.
problems/          Problem specs as source (sod_shock_tube.jl, sedov_blast.jl).
test/              Core suite (RefMesh + HGBackend). Its own Project.toml + [sources].
```

The seam is enforced by package boundaries: `src/` cannot name a concrete backend
(it would be a missing-import compile error). Backends are injected at the
`Simulation` constructor. Keep it that way.

## How to run things (environment is non-obvious)

- **Julia is `juliaup`-managed; the binary is NOT on the non-interactive PATH.**
  Use the absolute path:
  `~/.julia/juliaup/julia-1.12.6+0.aarch64.apple.darwin14/bin/julia`
- **Core tests:** `<julia> --project=test test/runtests.jl` (≈7 min; the 256-cell
  Sod and 2D Sedov dominate).
- **EnzoViz tests:** `<julia> --project=lib/EnzoViz/test lib/EnzoViz/test/runtests.jl`
  (needs the Python env recorded — see EnzoViz section).
- **`[sources]` rule (bit us repeatedly):** Julia honors `[sources]` ONLY from the
  *active root* project, and every `[sources]` key must ALSO appear in
  `[deps]`/`[extras]` of that same project. That's why there is a dedicated
  `test/Project.toml` (and `lib/EnzoViz/test/Project.toml`) listing every
  path-dev'd subpackage **plus the unregistered `R3D` git dep** (a transitive dep
  of HierarchicalGrids). `Pkg.test()`'s sandbox does NOT inherit root `[sources]`
  — run `runtests.jl` directly against `--project=…/test`, not `Pkg.test()`.

## Shell gotcha (cost hours)

The shell is **zsh**. An unmatched glob (e.g. `~/Applications/Julia*`) aborts the
WHOLE command with `nomatch` AND cancels every other tool call batched in the same
turn. Always `setopt NULL_GLOB` at the top of scripts, or avoid bare globs.
Symptom: a batch of tool calls all "Cancelled" with one real error.

## Backend contract notes (for new backends / kernels)

- Cell handle is **opaque** — `CartesianIndex` (RefMesh) or `Int` id (HGBackend).
  Never assume one. Index field views with the handle; get geometry via
  `cell_center`/`cell_width`/`cell_volume`.
- **No ghost cells in the interface.** Boundaries resolve per-face via
  `neighbor(b, cell, axis, side; bcs)` → `Interior(cell)` (incl. periodic wrap) or
  `DomainBoundary(bc)`. The solver synthesizes ghost states from the BC.
- **`for_each_face(f, b; bcs)` is where AMR topology lives.** It emits each unique
  (sub)face once with `(left::NeighborRef, right::NeighborRef, axis, area)`; for
  coarse↔fine it emits per-fine-cell sub-faces carrying the fine area, so the
  flux-divergence driver is conservative across level jumps for free. New backends
  must dedup faces correctly (emit conforming from the hi side; coarse↔fine from
  the fine side).
- **`level_of` is RELATIVE to the base grid** (0 = base, +1 per refinement), not
  HG's absolute tree level. This is the AMR-policy semantics; HGBackend subtracts
  `base_level`. (Getting this wrong silently disables refinement.)

## HierarchicalGrids.jl (the substrate) gotchas

- Local at `/Users/tabel/Projects/HierarchicalGrids.jl`; pulled in via
  `[sources]` path. Depends on **R3D**, an unregistered git package
  (`github.com/yipihey/r3djl`, subdir `R3D.jl`) — must be declared+sourced in any
  root project that uses HGBackend.
- `refine_cells!(mesh, ids, masks)` — `split_masks` is **positional, not keyword**.
- `Document()`/widget creation: HG itself is fine; this note is for veusz (below).
- Cell-average fields: `AdaptiveField` only wraps **`PolynomialFieldSet`**, not
  `CellAverageFieldSet`. HGBackend therefore stores degree-0 `BernsteinBasis{D,0}`
  polynomial fields (the single coeff IS the cell mean) to get HG's tested
  conservative remap-on-refine. Don't "simplify" this to CellAverageFieldSet — it
  loses the remap.

## EnzoViz / veusz gotchas (all hard-won)

- **PythonCall interpreter:** EnzoViz points PythonCall at a recorded interpreter
  via `lib/EnzoViz/.python-path` (written by `setup_env.jl`), with
  `JULIA_CONDAPKG_BACKEND=Null`. Set `JULIA_PYTHONCALL_EXE` + `QT_QPA_PLATFORM=offscreen`
  BEFORE `using EnzoViz`/PythonCall init.
- **Two mandatory init steps before any veusz Document/capture**, or it aborts:
  1. import `veusz.setting` + `veusz.widgets` (widget classes self-register;
     otherwise `Document()` → `KeyError: 'document'`).
  2. construct a `QApplication` (`veusz.qtall.QApplication`, NOT `QtWidgets`)
     before any capture, or Qt SIGABRTs at `QFontDatabase`. Even offscreen.
  Both live in `lib/EnzoViz/src/capture.jl::_pymods()`.
- **Real veusz API (the snippet in the original request was wrong):**
  - `veusz.paint.qt_capture.capture_document_scene(doc, page; pagesize_px, dpi)` → scene JSON.
  - `veusz.paint._paint_ext.render_scene_to_png(scene_bytes, w, h, (r,g,b,a), backend)`
    — lives in `_paint_ext`, NOT `qt_capture`. Backends: `available_backends()` →
    `["tiny-skia","vello"]` (vello = GPU).
  - 2D data: `CommandInterface.SetData2D(name, data; xrange/yrange/xcent/ycent)`.
    There is **no `SetData2DXY`**.
- **Which veusz checkout:** EnzoViz runs against `/Users/tabel/Projects/veusz`
  (git remote `yipihey/veusz`) — it has the **built** Vello ext
  (`veusz/paint/_paint_ext.abi3.so`), a working `.venv` (Python 3.14), AND the
  `veusz-tauri/` WASM `<veusz-figure>` frontend (built `dist-embed/veusz-embed.js`).
  `setup_env.jl` takes the fork path as an arg and records whatever interpreter
  has a working `_paint_ext`; default is this checkout's `.venv`. (Note:
  `/Users/tabel/Research/codes/embeded_veusz/veusz` is a *different Julia project*,
  not a veusz checkout — ignore it for EnzoViz.) If you move to a newer veusz
  checkout, it must provide both the compiled `_paint_ext` (`uv pip install -e`,
  needs the Rust toolchain) and a built embed frontend.
- **Perf contract:** build the veusz `Document` ONCE; per snapshot only push
  arrays (`SetData`/`SetData2D`) + capture. Never rebuild the doc per frame.

## Background-run discipline

Long Julia runs: launch with `run_in_background`, poll a log file for a sentinel
(`===== EXIT $? =====`), and parse with python for robustness (ANSI/`@info` noise).
macOS prefs spew `Error interpreting item … in settings file` from Qt — filter it.
Don't mark a task complete until you've SEEN the green summary; this session
repeatedly marked things done prematurely and had to reopen them.
