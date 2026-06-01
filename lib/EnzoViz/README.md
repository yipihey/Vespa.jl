# EnzoViz — inline AMR visualization for EnzoNG

EnzoViz turns a running EnzoNG simulation into a **self-contained interactive web
page** per problem: a time slider, play/pause, and live-editable colormap / min /
max / log-color, rendered through the [yipihey/veusz](https://github.com/yipihey/veusz)
fork's GPU **Vello** painter and its `<veusz-figure>` WASM component.

It lives under `lib/` like the mesh backends, so the core `EnzoNG` package stays
headless and dependency-clean. EnzoNG exposes only a generic `callback` hook in
`evolve!`; EnzoViz supplies a writer that snapshots field data each cadence step.

## How it works

* Build the veusz figure **once** (1D → stacked line plots; 2D → per-AMR-level
  image overlays on shared axes). Per snapshot, only **push arrays** into the
  existing datasets and capture the abstract scene — the document is never
  rebuilt (the perf contract).
* `veusz.paint.qt_capture.capture_document_scene` → scene JSON;
  `veusz.paint._paint_ext.render_scene_to_png(scene, w, h, (1,1,1,1), backend)`
  rasterizes (`"vello"` GPU, `"tiny-skia"` CPU fallback — auto-selected).
* `finalize!` writes `run_scenes.json` (`{meta, frames:[scene,…]}`) and an
  `index.html` hosting `<veusz-figure>`. Colormap/min/max/log-color re-map live
  in the browser (WASM) — no server.

## Setup (uv)

EnzoViz drives a local Python that has the veusz fork importable. The painless
path reuses the fork's prebuilt `.venv` (Vello already compiled); otherwise `uv`
builds a dedicated one:

```bash
julia lib/EnzoViz/setup_env.jl            # records the interpreter in .python-path
# or: julia lib/EnzoViz/setup_env.jl /path/to/veusz-fork
```

Then point PythonCall at it (set before `using EnzoViz`):

```julia
ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
ENV["JULIA_PYTHONCALL_EXE"]   = strip(read("lib/EnzoViz/.python-path", String))
ENV["QT_QPA_PLATFORM"]        = "offscreen"   # headless capture
```

One-time, for the interactive page, build the embed bundle in the fork:

```bash
cd /Users/tabel/Projects/veusz/veusz-tauri && pnpm install && pnpm run build:embed
```

(`finalize!` copies `veusz-embed.js` + the WASM assets next to the page; the WASM
is already built in the fork.)

## Usage

```julia
using EnzoNG, RefMesh, EnzoViz
prob = sod_problem_defaults(n = 256)
sim  = Simulation(UniformMesh(prob.dims, prob.domain), prob)
viz  = VizSession(sim; outdir = "out/Sod", every = 5)        # fields default: ρ, p, |v|
evolve!(sim; callback = writer(viz), callback_every = viz.every)
finalize!(viz)        # → out/Sod/{run_scenes.json, index.html, veusz-embed.js, wasm/}
```

AMR (2D) is identical — pass a `RefinementPolicy` to `evolve!`; EnzoViz adds a
per-level image widget whenever the mesh refines deeper, so finer levels overlay
coarser ones.

## Tests

```bash
julia --project=lib/EnzoViz/test -e 'using Pkg; Pkg.instantiate()'
julia --project=lib/EnzoViz/test lib/EnzoViz/test/runtests.jl
```
