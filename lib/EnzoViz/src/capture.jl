# VizSession: a long-lived Python veusz document driven per snapshot. Build the
# figure ONCE; per snapshot push arrays + capture the scene. This is the perf
# contract — the document/widgets are never rebuilt, only dataset bytes change.

# Lazily-held Python module handles (resolved on first VizSession construction so
# that PythonCall's interpreter selection has happened).
const _PY = Dict{Symbol,Any}()

function _pymods()
    if isempty(_PY)
        # Drive veusz fully headless: offscreen Qt platform for capture.
        if !haskey(ENV, "QT_QPA_PLATFORM")
            ENV["QT_QPA_PLATFORM"] = "offscreen"
        end
        # A QApplication must exist before any QFontDatabase / QPainter use, or
        # Qt aborts (SIGABRT) the moment capture touches font metrics — even in
        # offscreen mode. veusz exposes `QApplication` directly on `qtall` (the
        # daemon's `qt.QApplication` pattern). Construct or reuse one process-wide.
        qt = pyimport("veusz.qtall")
        inst = qt.QApplication.instance()
        _PY[:qapp] = pyis(inst, pybuiltins.None) ? qt.QApplication(pylist([""])) : inst
        # Widget classes self-register into the widget factory only when these
        # modules are imported; without them `document.Document()` raises
        # `KeyError: 'document'`. Import for the side effect before anything else.
        pyimport("veusz.setting")
        pyimport("veusz.widgets")
        _PY[:document] = pyimport("veusz.document")
        _PY[:cap]      = pyimport("veusz.paint.qt_capture")
        _PY[:px]       = pyimport("veusz.paint._paint_ext")
        _PY[:np]       = pyimport("numpy")
    end
    return _PY
end

"Pick the best available paint backend (GPU vello if built, else CPU tiny-skia)."
function _choose_backend(requested::Symbol)
    avail = Set(string.(collect(_pymods()[:px].available_backends())))
    if requested === :auto
        return "vello" in avail ? "vello" : (first(avail))
    end
    s = string(requested)
    s in avail || error("EnzoViz: backend $s not available; have $(collect(avail))")
    return s
end

"""
    VizSession(sim; outdir, fields=(:density,:pressure,:speed),
               size=(900,650), every=5, backend=:auto, dpi=96.0)

Bind a simulation to a freshly-built veusz figure matched to the problem's rank
(1D line plots or 2D per-AMR-level images). Holds the live Python document,
command interface, accumulated scene frames, and snapshot metadata.
"""
mutable struct VizSession
    sim::Any
    outdir::String
    fields::NTuple{N,Symbol} where {N}
    size::Tuple{Int,Int}
    dpi::Float64
    every::Int
    backend::String
    rank::Int
    doc::Any
    ci::Any
    frames::Vector{Any}        # captured scene JSON (parsed) per snapshot
    times::Vector{Float64}
    ranges::Dict{Symbol,Tuple{Float64,Float64}}  # pinned per-field colour/axis range
    nlevels::Int
    record::Bool               # also retain per-frame rasters (for live re-capture)
    rasters::Vector{Any}       # per-frame rasterized data when `record` (else empty)
    lock::ReentrantLock        # serializes access to the single Python document
end

function VizSession(sim; outdir::AbstractString,
                    fields = FIELD_KEYS,
                    size::Tuple{Int,Int} = (900, 650),
                    every::Integer = 5,
                    backend::Symbol = :auto,
                    dpi::Real = 96.0,
                    record::Bool = false)
    mods = _pymods()
    mkpath(outdir)
    R = Vespa.rank(sim.backend)
    R in (1, 2) || error("EnzoViz supports 1D and 2D problems (got rank $R)")
    be = _choose_backend(backend)
    flds = Tuple(Symbol.(collect(fields)))

    # Pin per-field ranges from the initial state so frames stay comparable.
    ranges = Dict{Symbol,Tuple{Float64,Float64}}()
    for f in flds
        lo, hi = field_range(sim, f)
        pad = (hi - lo) * 0.05 + eps()
        ranges[f] = (lo - pad, hi + pad)
    end

    doc = mods[:document].Document()
    ci = mods[:document].CommandInterface(doc)
    nlev = R == 2 ? Vespa.max_level(sim.backend) + 1 : 0
    if R == 1
        _build_doc_1d!(ci, flds, sim, ranges)
    else
        _build_doc_2d!(ci, flds, sim, ranges, nlev)
    end

    # The 2D layout is a vertical stack of square panels (doc is 22cm wide ×
    # 8cm·nf tall). Force the render canvas to that aspect from the requested
    # width, so panels never get squished (titles/x-ticks crammed) by a
    # mismatched `size` hint. 1D keeps the size as given.
    size = R == 2 ? _stack_canvas(size, length(flds)) : size

    return VizSession(sim, String(outdir), flds, size, Float64(dpi), Int(every),
                      be, R, doc, ci, Any[], Float64[], ranges, nlev,
                      record, Any[], ReentrantLock())
end

# ── document builders (run ONCE) ────────────────────────────────────────────

function _build_doc_1d!(ci, fields, sim, ranges)
    dom = Vespa.domain(sim.backend)[1]
    ci.To("/")
    ci.Add("page", name = "page1")
    ci.To("/page1")
    ci.Set("width", "20cm")
    ci.Set("height", "8cm")
    # one stacked graph per field (grid layout)
    ci.Add("grid", name = "grid1", autoadd = false)
    ci.To("/page1/grid1")
    ci.Set("rows", length(fields))
    ci.Set("columns", 1)
    for f in fields
        g = "g_$(f)"
        fname = string(f)
        ci.Add("graph", name = g, autoadd = false)
        ci.To("/page1/grid1/$g")
        ci.Add("axis", name = "x")
        ci.Add("axis", name = "y", direction = "vertical")
        ci.Set("x/min", dom[1]); ci.Set("x/max", dom[2])
        lo, hi = ranges[f]
        ci.Set("y/min", lo); ci.Set("y/max", hi)
        ci.Set("y/label", fname)
        ci.Add("xy", name = fname)
        ci.Set("$fname/xData", "x")
        ci.Set("$fname/yData", fname)
        ci.Set("$fname/marker", "none")
        ci.To("/page1/grid1")
    end
    # seed datasets so widgets resolve
    x, cols = raster1d(sim, fields)
    ci.SetData("x", _pyarray(x))
    for f in fields
        ci.SetData(string(f), _pyarray(cols[f]))
    end
    return nothing
end

# Page geometry for the 2D stack (cm). Keep in sync with `_build_doc_2d!`.
const _STACK_W_CM = 22.0
const _STACK_H_CM = 8.0   # per panel

# Render canvas (px) for an `nf`-panel vertical stack: preserve width, set height
# to the document's cm aspect so panels render undistorted.
function _stack_canvas(size::Tuple{Int,Int}, nf::Integer)
    w = size[1]
    h = round(Int, w * (_STACK_H_CM * nf) / _STACK_W_CM)
    return (w, h)
end

function _build_doc_2d!(ci, fields, sim, ranges, nlev)
    dom = Vespa.domain(sim.backend)
    ci.To("/")
    ci.Add("page", name = "page1")
    ci.To("/page1")
    ci.Set("width", string(_STACK_W_CM) * "cm")
    ci.Set("height", string(_STACK_H_CM * length(fields)) * "cm")
    ci.Add("grid", name = "grid1", autoadd = false)
    ci.To("/page1/grid1")
    ci.Set("rows", length(fields))
    ci.Set("columns", 1)
    grids = raster2d_levels(sim, fields)
    first_g = "g_$(first(fields))"
    domw = dom[1][2] - dom[1][1]
    domh = dom[2][2] - dom[2][1]
    nf = length(fields)
    for (k, f) in enumerate(fields)
        g = "g_$(f)"
        ci.Add("graph", name = g, autoadd = false)
        ci.To("/page1/grid1/$g")
        # Fix data→pixel aspect so physical coordinates map 1:1 (round features
        # stay round): aspect = domain_width/domain_height (1 for a square domain).
        ci.Set("aspect", domw / domh)
        # Reserve room on the right of every panel for the colorbar + tick
        # numbers, and a little on top for the field-name title.
        _try_set(ci, "rightMargin", "2.6cm")
        _try_set(ci, "topMargin", "0.8cm")
        ci.Add("axis", name = "x")
        ci.Add("axis", name = "y", direction = "vertical")
        # Axes are SPATIAL coords, shared across every panel: identical range on
        # all, the field name lives on the colorbar (not the y-axis). Link panels
        # 2..N to the first panel's axes so the scale is shared (and zoom/pan
        # stays in sync in the live editor).
        ci.Set("x/min", dom[1][1]); ci.Set("x/max", dom[1][2])
        ci.Set("y/min", dom[2][1]); ci.Set("y/max", dom[2][2]); ci.Set("y/label", "y")
        if k > 1
            _try_set(ci, "x/match", "/page1/grid1/$first_g/x")
            _try_set(ci, "y/match", "/page1/grid1/$first_g/y")
        end
        # Shared x-axis: only the BOTTOM panel shows the x tick labels + "x"
        # label; the upper panels suppress them so the column reads as one shared
        # horizontal axis (the scales are identical via the match above).
        if k == nf
            ci.Set("x/label", "x")
        else
            _try_set(ci, "x/TickLabels/hide", true)
            ci.Set("x/label", "")
        end
        lo, hi = ranges[f]
        # one image widget per refinement level (coarse first, finer overlays)
        for lg in grids
            iw = "$(f)_L$(lg.level)"
            ci.Add("image", name = iw)
            ci.Set("$iw/data", iw)
            ci.Set("$iw/min", lo); ci.Set("$iw/max", hi)
            ci.Set("$iw/colorMap", "viridis")
        end
        # A colorbar per panel so the colours mean something. It must sit OUTSIDE
        # the plot (manual position just past the right edge) — with horzPosn
        # 'right' it draws inside and is hidden behind the image. It carries the
        # numeric scale; the field name goes on a panel title (below) since the
        # colorbar's own axis label clips against the page edge.
        cb = "cb_$(f)"
        ci.Add("colorbar", name = cb, image = "$(f)_L0")
        ci.Set("$cb/direction", "vertical")
        ci.Set("$cb/vertPosn", "centre")
        _try_set(ci, "$cb/horzPosn", "manual")
        _try_set(ci, "$cb/horzManual", 1.04)
        _try_set(ci, "$cb/width", "0.45cm")
        # Panel title = field name, top-left inside the graph.
        tl = "title_$(f)"
        ci.Add("label", name = tl)
        ci.Set("$tl/label", string(f))
        _try_set(ci, "$tl/xPos", 0.5)
        _try_set(ci, "$tl/yPos", 1.10)
        _try_set(ci, "$tl/alignHorz", "centre")
        _try_set(ci, "$tl/alignVert", "top")
        _try_set(ci, "$tl/Text/size", "13pt")
        ci.To("/page1/grid1")
    end
    # seed the per-level datasets
    _push_2d!(ci, grids, fields)
    return nothing
end

# ── per-snapshot data push + capture ────────────────────────────────────────

@inline _pyarray(v::AbstractVector{<:Real}) = _pymods()[:np].asarray(collect(Float64, v))
@inline _pymatrix(m::AbstractMatrix{<:Real}) = _pymods()[:np].asarray(collect(Float64, m))

function _push_1d!(ci, sim, fields)
    x, cols = raster1d(sim, fields)
    _push_1d_data!(ci, x, cols, fields)
    return nothing
end

# Push pre-rasterized 1D arrays (so re-capture can replay a stored frame).
function _push_1d_data!(ci, x, cols, fields)
    ci.SetData("x", _pyarray(x))
    for f in fields
        ci.SetData(string(f), _pyarray(cols[f]))
    end
    return nothing
end

function _push_2d!(ci, grids, fields)
    for lg in grids
        for f in fields
            ci.SetData2D("$(f)_L$(lg.level)", _pymatrix(lg.data[f]);
                         xcent = _pyarray(lg.xcent), ycent = _pyarray(lg.ycent))
        end
    end
    return nothing
end

"""
    snapshot!(viz)

Capture the current simulation state as one frame: push field arrays into the
existing datasets, capture the scene, and accumulate it. AMR meshes whose level
count grew get fresh per-level image widgets added on demand.
"""
function snapshot!(viz::VizSession)
    lock(viz.lock) do
        mods = _pymods()
        local raster
        if viz.rank == 1
            x, cols = raster1d(viz.sim, viz.fields)
            _push_1d_data!(viz.ci, x, cols, viz.fields)
            raster = (kind = :d1, x = x, cols = cols)
        else
            # If AMR added levels since the doc was built, extend the template.
            nlev_now = Vespa.max_level(viz.sim.backend) + 1
            if nlev_now > viz.nlevels
                _extend_levels_2d!(viz, viz.nlevels, nlev_now)
                viz.nlevels = nlev_now
            end
            grids = raster2d_levels(viz.sim, viz.fields)
            _push_2d!(viz.ci, grids, viz.fields)
            raster = (kind = :d2, grids = grids)
        end
        push!(viz.frames, JSON3.read(_scene_string(_capture(viz))))
        push!(viz.times, viz.sim.t)
        viz.record && push!(viz.rasters, raster)
    end
    return viz
end

# Capture the current document state to a scene-JSON python bytes object.
@inline function _capture(viz::VizSession)
    w, h = viz.size
    return _pymods()[:cap].capture_document_scene(viz.doc, 0;
                pagesize_px = pytuple((w, h)), dpi = pytuple((viz.dpi, viz.dpi)))
end

# Per-field min/max over a recorded raster (ignoring the NaN AMR mask in 2D).
function _raster_extrema!(acc, raster, fields)
    if raster.kind === :d1
        for f in fields
            v = raster.cols[f]
            isempty(v) && continue
            lo, hi = acc[f]
            acc[f] = (min(lo, minimum(v)), max(hi, maximum(v)))
        end
    else
        for lg in raster.grids, f in fields
            lo, hi = acc[f]
            for x in lg.data[f]
                isnan(x) && continue
                x < lo && (lo = x); x > hi && (hi = x)
            end
            acc[f] = (lo, hi)
        end
    end
    return acc
end

"""
    refit_ranges!(viz)

Recompute each field's colour/axis range from the **actual recorded data** across
all snapshots and re-capture every frame so the images and colorbars span the
real dynamic range. Without this the ranges stay pinned to the initial state —
which is degenerate for blast problems (uniform t=0 ⇒ flat colour, blank
colorbar). Requires `record=true`; a no-op otherwise.
"""
function refit_ranges!(viz::VizSession)
    (viz.record && !isempty(viz.rasters)) || return viz
    lock(viz.lock) do
        acc = Dict(f => (Inf, -Inf) for f in viz.fields)
        for r in viz.rasters
            _raster_extrema!(acc, r, viz.fields)
        end
        for f in viz.fields
            lo, hi = acc[f]
            isfinite(lo) && isfinite(hi) || continue
            hi <= lo && (hi = lo + (abs(lo) + 1.0) * 1e-6)  # avoid degenerate span
            pad = (hi - lo) * 0.02
            viz.ranges[f] = (lo - pad, hi + pad)
        end
        # Re-apply ranges to the live widgets and re-capture each stored frame.
        _apply_settings!(viz)
        empty!(viz.frames)
        for r in viz.rasters
            if r.kind === :d1
                _push_1d_data!(viz.ci, r.x, r.cols, viz.fields)
            else
                _push_2d!(viz.ci, r.grids, viz.fields)
            end
            push!(viz.frames, JSON3.read(_scene_string(_capture(viz))))
        end
    end
    return viz
end

# Add image widgets for newly-created refinement levels (AMR grew deeper).
function _extend_levels_2d!(viz, oldn, newn)
    ci = viz.ci
    lo_hi = viz.ranges
    for f in viz.fields
        for lev in oldn:(newn - 1)
            ci.To("/page1/grid1/g_$(f)")
            iw = "$(f)_L$(lev)"
            ci.Add("image", name = iw)
            ci.Set("$iw/data", iw)
            lo, hi = lo_hi[f]
            ci.Set("$iw/min", lo); ci.Set("$iw/max", hi)
            ci.Set("$iw/colorMap", "viridis")
        end
    end
    ci.To("/")
    return nothing
end

# capture_document_scene returns python bytes; get a Julia String of the JSON.
function _scene_string(scene)
    return pyconvert(String, scene.decode("utf-8"))
end

"""
    writer(viz) -> Function

Adapt a `VizSession` into the `callback(sim, stage)` form `evolve!` expects:
snapshot on `:init` and every `:step` invocation (the loop already gates by
`callback_every`); `:final` is captured too so the last state is always present.
"""
function writer(viz::VizSession)
    return function (sim, stage)
        if stage === :init || stage === :step || stage === :final
            snapshot!(viz)
        end
        return nothing
    end
end

"""
    render_png(viz, frame_index; backend=viz.backend) -> Vector{UInt8}

Rasterize an accumulated frame's scene to PNG bytes (quick-look). `frame_index`
is 1-based into `viz.frames`.
"""
function render_png(viz::VizSession, i::Integer; backend = viz.backend)
    mods = _pymods()
    scene_bytes = codeunits(JSON3.write(viz.frames[i]))
    w, h = viz.size
    png = mods[:px].render_scene_to_png(pybytes(scene_bytes), w, h,
            pytuple((1.0, 1.0, 1.0, 1.0)), backend)
    return pyconvert(Vector{UInt8}, png)
end

"""
    render_all_pngs!(viz, dir; prefix="frame_") -> Vector{String}

Rasterize every accumulated frame to `<dir>/<prefix>NNNN.png` (Vello, 1-based,
zero-padded) and return the file names (relative). This is the flipbook the
static page animates over — instant, no WASM/Pyodide, works over plain http.
"""
function render_all_pngs!(viz::VizSession, dir::AbstractString; prefix = "frame_")
    mkpath(dir)
    n = length(viz.frames)
    pad = max(4, ndigits(n))
    names = String[]
    for i in 1:n
        fn = prefix * lpad(i, pad, '0') * ".png"
        write(joinpath(dir, fn), render_png(viz, i))
        push!(names, fn)
    end
    return names
end

"""
    save_frame_vsz(viz, i, path)

Write a **self-contained** `.vsz` for frame `i` (data embedded), suitable for the
fork's `<veusz-figure src=…>` live WASM editor. Requires `record=true` (the
per-frame raster is re-pushed into the live document, then saved). Pins the
page size so the embedded figure matches the flipbook.
"""
function save_frame_vsz(viz::VizSession, i::Integer, path::AbstractString)
    viz.record || error("save_frame_vsz requires VizSession(...; record=true)")
    lock(viz.lock) do
        r = viz.rasters[i]
        if r.kind === :d1
            _push_1d_data!(viz.ci, r.x, r.cols, viz.fields)
        else
            _push_2d!(viz.ci, r.grids, viz.fields)
        end
        viz.ci.Save(path)
    end
    return path
end

# ── live re-capture (serve mode) ─────────────────────────────────────────────
# The browser WASM re-maps colour live (colormap/vmin/vmax/log-COLOR). Things it
# can NOT do client-side — because they bake into the captured scene at layout
# time — are the *axis* range and the **log-position axis** scaling. `recapture`
# re-pushes a stored frame's raster into the live document with those settings
# applied and returns a fresh scene. Requires `record=true` at construction.

# Set a setting, swallowing veusz errors for paths/settings that don't apply to
# a given widget (e.g. a colour setting on a level that wasn't created). Keeps a
# re-render robust to per-widget setting quirks.
@inline function _try_set(ci, path, val)
    try
        ci.Set(path, val)
    catch
        # ignore: setting not applicable to this widget
    end
    return nothing
end

"""
Apply display settings to every plottable widget for a re-render.

2D: colormap + min/max per image widget (one per AMR level). **log-color for 2D
is a client-side WASM capability**, so we don't bake it here — the genuine
server-only knob is the 1D **log axis**. 1D: y-range + log-y per graph.
"""
function _apply_settings!(viz::VizSession; colormap = "viridis",
                          vmin = nothing, vmax = nothing,
                          logcolor::Bool = false, logaxis::Bool = false)
    ci = viz.ci
    for f in viz.fields
        lo = vmin === nothing ? viz.ranges[f][1] : vmin
        hi = vmax === nothing ? viz.ranges[f][2] : vmax
        if viz.rank == 2
            for lev in 0:(viz.nlevels - 1)
                iw = "/page1/grid1/g_$(f)/$(f)_L$(lev)"
                _try_set(ci, "$iw/colorMap", colormap)
                _try_set(ci, "$iw/min", lo)
                _try_set(ci, "$iw/max", hi)
                logcolor && _try_set(ci, "$iw/colorScaling", "log")
            end
        else
            g = "/page1/grid1/g_$(f)"
            _try_set(ci, "$g/y/min", lo)
            _try_set(ci, "$g/y/max", hi)
            _try_set(ci, "$g/y/log", logaxis)
        end
    end
    return nothing
end

"""
    recapture(viz, frame_index; colormap, vmin, vmax, logcolor, logaxis) -> scene JSON String

Re-render a stored frame with new display settings, including the **log axis**
(1D) / log color-scaling that the browser cannot change on its own. Returns the
fresh scene JSON (string). Requires the session was built with `record=true`.
Thread-safe (serializes on the session lock).
"""
function recapture(viz::VizSession, i::Integer;
                   colormap = "viridis", vmin = nothing, vmax = nothing,
                   logcolor::Bool = false, logaxis::Bool = false)
    viz.record || error("recapture requires VizSession(...; record=true)")
    1 <= i <= length(viz.rasters) ||
        error("frame $i out of range (have $(length(viz.rasters)))")
    return lock(viz.lock) do
        r = viz.rasters[i]
        if r.kind === :d1
            _push_1d_data!(viz.ci, r.x, r.cols, viz.fields)
        else
            _push_2d!(viz.ci, r.grids, viz.fields)
        end
        _apply_settings!(viz; colormap = colormap, vmin = vmin, vmax = vmax,
                         logcolor = logcolor, logaxis = logaxis)
        _scene_string(_capture(viz))
    end
end
