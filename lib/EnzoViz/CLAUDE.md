# CLAUDE.md — lib/EnzoViz

Inline visualization: PythonCall → veusz fork (GPU Vello) → `<veusz-figure>` WASM
page. See the root `EnzoNG.jl/CLAUDE.md` "EnzoViz / veusz gotchas" — that section
is the canonical reference. This file is the local file map.

## Files
- `src/EnzoViz.jl`   — module; `python_interpreter()` reads `.python-path`.
- `src/capture.jl`   — `_pymods()` (the mandatory Qt+widget init lives here),
                       `VizSession`, `snapshot!`, `writer`, `render_png`,
                       document builders (1D line / 2D per-level images).
- `src/rasterize.jl` — `raster1d` (sorted ρ/p/|v| arrays); `raster2d_levels`
                       (one regular grid per AMR level, NaN-masked, overlaid).
- `src/page.jl`      — `finalize!` → `run_scenes.json` + `index.html` + copies the
                       embed bundle/WASM next to the page.
- `setup_env.jl`     — records a Python interpreter with a working veusz `_paint_ext`
                       (reuses a built `.venv` if present, else `uv venv` + `uv pip
                       install -e <veusz fork>`). Arg = fork path.
- `assets/`          — copied embed bundle (`veusz-embed.js`) + `wasm/` (gitignored).
- `test/`            — env+capture smoke (both backends), 1D Sod page, 2D Sedov AMR page.

## Public API
```julia
viz = VizSession(sim; outdir, fields=(:density,:pressure,:speed), size, every, backend=:auto)
evolve!(sim; callback=writer(viz), callback_every=viz.every[, policy=…])
finalize!(viz)            # writes the interactive page
EnzoViz.render_png(viz, i)  # quick-look PNG of frame i (Vello/tiny-skia)
```

## First-time setup
```
<julia> lib/EnzoViz/setup_env.jl /path/to/built/veusz   # writes .python-path
```
Then set `JULIA_CONDAPKG_BACKEND=Null`, `JULIA_PYTHONCALL_EXE=$(cat .python-path)`,
`QT_QPA_PLATFORM=offscreen` before `using EnzoViz`.

## Editability split (by veusz design)
- Live in browser (WASM, no server): time/animation, **colormap, vmin, vmax,
  log-color**.
- Bakes at capture (needs re-capture / presets): axis *range*, log-*position* axis.

## Serve mode (`src/serve.jl`)
`serve(viz; port)` (needs `VizSession(...; record=true)`) starts a dependency-free
HTTP server (Julia `Sockets` stdlib — no HTTP.jl) hosting `outdir` plus live
endpoints: `/recapture` (fresh scene JSON with new settings) and `/png`
(server-side Vello re-render — the path that adds **log axis**, which WASM can't
do alone). `record=true` retains per-frame rasters so frames re-render without
re-running the sim; `recapture(viz, i; colormap, vmin, vmax, logcolor, logaxis)`
is the underlying call. All Python access goes through `viz.lock` (one shared
veusz document). serve.jl defines a local `partition` helper — keep it module-
internal (it shadows Base/PythonCall `partition`; harmless, but don't export).
