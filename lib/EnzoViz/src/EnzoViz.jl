"""
    EnzoViz

Inline visualization for Vespa: as a simulation runs, capture field snapshots
and render them through the **yipihey/veusz** fork's GPU **Vello** painter,
producing for each problem a self-contained, interactive web page (time slider,
play/pause, live-editable colormap / vmin / vmax / log-color) powered by the
fork's `<veusz-figure>` WASM component.

This package is deliberately isolated under `lib/` (like the mesh backends): it
carries the heavy Python/Vello/WASM toolchain so the core `Vespa` package and
its headless test suite stay dependency-clean. Vespa exposes only a generic
`callback` hook in `evolve!`; EnzoViz supplies a writer callback.

## Pipeline (verified against the local fork)

  * `veusz.document.Document()` + `CommandInterface` build the figure **once**
    (the perf contract — never rebuilt per frame).
  * per snapshot: push arrays via `SetData` / `SetData2D`, then
    `veusz.paint.qt_capture.capture_document_scene(doc, 0; pagesize_px, dpi)`
    → scene JSON **bytes**.
  * `veusz.paint._paint_ext.render_scene_to_png(scene, w, h, (1,1,1,1), backend)`
    rasterizes (backend `"vello"` → GPU, `"tiny-skia"` → CPU fallback).
  * after the run, emit `run_scenes.json` (`{meta, frames:[scene,…]}`) + an
    `index.html` hosting `<veusz-figure>`.

## Usage

```julia
using Vespa, RefMesh, EnzoViz
prob = sod_problem_defaults(n = 256)
sim  = Simulation(UniformMesh(prob.dims, prob.domain), prob)
viz  = VizSession(sim; outdir = "out/Sod", every = 5)
evolve!(sim; callback = writer(viz), callback_every = viz.every)
finalize!(viz)        # writes run_scenes.json + index.html
```

## Live serve mode

For *continuous* editing of settings the browser can't change on its own (axis
range, **log axis**), build with `record = true` and start the server:

```julia
viz = VizSession(sim; outdir = "out/Sod", every = 5, record = true)
evolve!(sim; callback = writer(viz), callback_every = viz.every)
srv = serve(viz; port = 8080, open = true)   # http://127.0.0.1:8080/
# … the page's "server render" panel fetches /png re-rendered via Vello …
close(srv)
```
"""
module EnzoViz

using PythonCall
using JSON3
using Vespa
using MeshInterface

export VizSession, writer, snapshot!, finalize!, recapture, serve, python_interpreter

# ── Python interpreter selection ────────────────────────────────────────────
# EnzoViz points PythonCall at a recorded interpreter (`.python-path`, written by
# setup_env.jl) instead of CondaPkg, so we drive the fork's prebuilt Vello env
# directly. This must be set BEFORE PythonCall initializes CPython (the caller
# sets JULIA_PYTHONCALL_EXE / JULIA_CONDAPKG_BACKEND=Null from this path).

"Path to the Python interpreter EnzoViz uses (from `.python-path`, or env override)."
function python_interpreter()
    override = get(ENV, "ENZOVIZ_PYTHON", "")
    isempty(override) || return override
    pth = joinpath(@__DIR__, "..", ".python-path")
    isfile(pth) && return strip(read(pth, String))
    return ""   # fall back to PythonCall's default (CondaPkg) if unset
end

include("rasterize.jl")
include("capture.jl")
include("page.jl")
include("serve.jl")

end # module
