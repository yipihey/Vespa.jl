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

## The live-Enzo bridge (EnzoLib) — `:enzo` slots, `:remote` worker, MPI

`lib/EnzoLib` is a native binding to a **live Enzo hierarchy** through a C-ABI
bridge (`EnzoModules/src/enzomodules_problem_bridge.C` +
`src/enzo/Grid_EnzoModulesFixture.C` ↔ `lib/EnzoLib/src/session.jl`). It lets a
Julia-driven `EvolveLevel` call Enzo's own certified kernels per step (the
`:enzo` method slots / full-replication mode), and lets a `:julia` slot mutate the
same Enzo memory Enzo's kernels operate on. Built into `libenzomodules_grid.dylib`
(links the full Enzo `.so`); `EnzoLib.grid_available()` gates every live call.

**One call macro, two transports (`@xcall`).** Every bridge call goes through
`@xcall(:c_symbol, Ret, (Argtypes…), args…)` in `session.jl`, which dispatches on
`EnzoLib.backend()`:
- `:local` (default) — in-process `ccall`. The fast path and the differential oracle.
- `:remote` — a **subprocess worker** over a control channel + shared file (ADR-0005).
  This exists because hosting the **MPI** Enzo (gcc/libstdc++) *inside* the Julia
  process (libc++) aborts in C++ static init — a runtime collision. A separate
  worker process carries no Julia runtime, so the collision cannot occur. Switch
  with `EnzoLib.connect_worker!(cmd; shm) … disconnect_worker!()`.

**Single source of truth = the manifest.** `EnzoLib.manifest()` parses the
`@xcall` sites out of `session.jl`; it generates (a) the Julia RPC marshalling and
(b) the C++ worker's typed dispatch (`EnzoModules/tools/gen_worker_dispatch.jl` →
`enzomodules_worker_dispatch.inc`). A `contract_hash()` (FNV-1a over the surface) is
checked at the worker handshake, so a worker built from a different `session.jl`
than the client is **refused, not silently corrupt**. *After adding/changing an
`@xcall`, the workers must be rebuilt* (the build does this) so the hash matches.

**Building the bridge** (stage-2 only; reuses a cached `libenzo`). `julia` must be
resolvable — juliaup's binary is **not** on the non-interactive PATH, so pass
`JULIA=`:
```
JULIA=~/.julia/juliaup/<ver>/bin/julia bash EnzoModules/deps/build_grid_darwin.sh        # serial (default)
JULIA=<jl> MPITRAMPOLINE_DIR=<artifact> bash EnzoModules/deps/build_grid_darwin.sh mpi    # MPI flavor
```
- `<artifact>` = `julia --project=lib/EnzoLib/test -e 'import MPItrampoline_jll as T; print(T.artifact_dir)'`
  (the script's own `is_available()` probe is false on this depot, hence the explicit dir).
- Artifacts: serial `libenzomodules_grid.dylib` + `enzomodules_worker`; MPI
  `libenzomodules_grid_mpi.dylib` + `enzomodules_worker_mpi`. Both bridge dylibs
  coexist; `ENV["ENZONG_ENZO_MPI"]=="1"` selects the MPI one for in-process loads.
- **Gotcha (fixed in-script but know it):** stage-1 sets the `Make.config` MPI flag
  only when it *builds* `libenzo`; with `libenzo` cached, the bridge inherits the
  *last* flavor's flag. The script now re-pins `make use-mpi-{yes,no}` before
  extracting DFLAGS, else a serial bridge build picks up a stale `-DUSE_MPI` and
  fails the link on `_MPI_*`.

**Running under MPI** (the subprocess worker, mpiexec'd N ranks). The Julia client
loads **no** MPI; it spawns `mpiexec -n N enzomodules_worker_mpi <shm> <mpi-bridge>`.
Rank 0 owns the control channel and `MPI_Bcast`s each command so collective bridge
calls (`session_init`→`CommunicationPartitionGrid`, `set_boundary`, `compute_dt`,
`update_from_finer`) run in lockstep. The opt-in gate (skips cleanly if the
toolchain is absent):
```
MPITRAMPOLINE_LIB=$HOME/opt/mpiwrapper/lib/libmpiwrapper.so \
  julia --project=lib/EnzoLib/test lib/EnzoLib/test/test_mpi_worker.jl
```
It asserts `num_ranks==2`, that the hierarchy distributed (`grid_owners=[0,1]`), and
**global mass conservation** (`session_global_field_integral` = per-rank local tile
sum, `CommunicationAllSumValues`-reduced; the distributed total equals the serial
total to round-off). The serial `:local`≡`:remote` parity oracle is
`test_rpc_parity.jl` (in `runtests.jl`). **Design + full rationale: `docs/adr/0005-…`
(subprocess boundary) and `docs/adr/0004-…` (Enzo-substrate MPI). The conservative
`:julia`-under-AMR path is `docs/adr/0003-…`.**

## CodeBridge + MultiCode (the multi-code framework, ADR-0006)

`lib/CodeBridge` is the shared legacy-wrapper substrate (extracted from EnzoLib,
hash-invariant — the prebuilt C++ workers still handshake): `LazyLib` multi-flavor
loading, `Bridge`, the `@xcall` macro (resolves the calling module's `const
BRIDGE`), manifest/contract-hash, and the worker RPC. **EnzoLib, RamsesLib
(RamsesNG.jl), and ArepoLib (Arepo.jl) are all clients** — their cross-repo
`[sources]` point back here, so changes to CodeBridge affect three repos.
Wire-protocol invariants live in `lib/CodeBridge/test` (29 tests incl. a
compiled-C-fixture local≡remote parity oracle; zero-arg calls and `Ref`/Matrix
buffers are covered — they bit earlier).

`lib/MultiCode` is the cross-code layer: `CellSet` canonical state + per-code
`extract`/`inject!` adapters (Phase 2), the PPMKernels-in-RAMSES guest slot
(Phase 3; `device=:metal` for f32 GPU), Moray-as-a-service + the conservative
deposit/sample exchange + Moray-inside-Arepo (Phase 4), RAMSES-RT wrapped +
the Moray-vs-RAMSES-RT cross-check + RAMSES-RT-inside-Enzo (Phases 4–5).
Run: `<julia> --project=lib/MultiCode/test lib/MultiCode/test/runtests.jl`
(~5 min; needs the Enzo grid dylib, mini-ramses `bin64h` AND `bin64hrt` libs,
and the sibling arepo `libarepo.dylib`). Reports land in `reports/multicode/`.
Multi-worker (N codes at once): `test/multicode/test_multicode_workers.jl`
(own Project.toml).

Hard-won gotchas:
- **RAMSES lib flavors:** `bin64s` = gravity-only (no hydro symbols), `bin64h` =
  hydro (`RAMSES_LIB` must point here for Sod), `bin64hrt` = hydro+RT
  (`RAMSES_LIB_RT`; build `make NDIM=3 HYDRO=1 GRAV=1 RT=1 NRTGRP=1 NION=1
  libramses`, and `make clean` first when flags change — stale `.mod` files).
- **RAMSES-RT driving:** call `rt_neq_updates!` once after `init` (fills the σ·c
  chemistry tables; without it photons stream but nothing ionizes). Ion
  fractions are density-weighted: `xHII = uold₆/uold₁`. Keep RT point sources at
  a cell center — a corner source's CIC cloud clips to 1/8 (upstream bug).
- **Arepo is a per-process singleton:** a second `init` in one process crashes
  its allocator. Any run after the first must use a worker
  (`run_arepo_sod(worker=true)` / `CodeBridge.connect_worker!`).
- **Periodic step ICs carry a mirror Riemann problem at the wrap seam** — the
  Sod comparison uses t̂=0.1, a double-length RAMSES domain, windowed profiles.
- **Moray/chemistry outer dt:** photons subcycle internally but the chemistry
  advances once per outer cycle — cap the outer dt (0.25 Myr) or the
  ionization lags the radiation.

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
