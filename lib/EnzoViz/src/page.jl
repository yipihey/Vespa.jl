# After a run, emit a self-contained interactive page. The animation is a **PNG
# flipbook**: one Vello-rendered PNG per snapshot, animated by an HTML/JS time
# slider + play/pause (instant, no WASM/Pyodide, works over plain http). For
# live editing (colour, zoom, log axis) the page mounts the fork's real
# `<veusz-figure src="frame_NNNN.vsz">` element on demand for the current frame.
#
# Note: the fork's `<veusz-figure>` takes a single `.vsz` document URL (`src`)
# and renders it via Pyodide+Vello/WebGPU — it has no multi-frame/`scenes` API,
# so time animation is the page's job, not the element's.

const _EMBED_JS = "veusz-embed.js"   # the fork's built <veusz-figure> bundle
const _FRAME_DIR = "frames"          # per-frame PNGs + .vsz live here

"""
    finalize!(viz; with_vsz=true) -> String

Render every frame to `<outdir>/frames/frame_NNNN.png` (the flipbook), optionally
save a self-contained `<outdir>/frames/frame_NNNN.vsz` per frame for the live
editor (needs `record=true`; silently skipped otherwise), write `run_scenes.json`
(metadata + frame manifest), copy the embed bundle + WASM, and write the
flipbook `index.html`. Returns the outdir. Idempotent.
"""
function finalize!(viz::VizSession; with_vsz::Bool = true)
    out = viz.outdir
    mkpath(out)
    name = viz.sim.problem.name
    framedir = joinpath(out, _FRAME_DIR)

    # 0) refit colour/axis ranges to the actual evolved data (when recording) so
    #    images and colorbars span the real dynamic range, not the (often
    #    degenerate) initial state. Re-captures the frames in place.
    refit_ranges!(viz)

    # 1) the flipbook: one Vello PNG per snapshot.
    pngs = render_all_pngs!(viz, framedir)

    # 2) per-frame self-contained .vsz for the on-demand WASM editor.
    vszs = String[]
    can_vsz = with_vsz && viz.record
    if can_vsz
        pad = max(4, ndigits(length(viz.frames)))
        for i in 1:length(viz.frames)
            fn = "frame_" * lpad(i, pad, '0') * ".vsz"
            save_frame_vsz(viz, i, joinpath(framedir, fn))
            push!(vszs, fn)
        end
    end

    # 3) run_scenes.json: metadata + a lightweight per-frame manifest (relative
    #    paths + time). The captured scenes themselves are NOT embedded — the
    #    flipbook plays the PNGs and the editor loads the per-frame .vsz on
    #    demand, so embedding every scene here just bloated the file to 100s of MB.
    meta = (
        name = name,
        rank = viz.rank,
        fields = collect(String.(viz.fields)),
        nlevels = viz.nlevels,
        times = viz.times,
        size = collect(viz.size),
        ranges = Dict(string(k) => collect(v) for (k, v) in viz.ranges),
        editable = can_vsz,
    )
    frames = [(
        index = i,
        time = (i <= length(viz.times) ? viz.times[i] : nothing),
        png = joinpath(_FRAME_DIR, pngs[i]),
        vsz = (i <= length(vszs) ? joinpath(_FRAME_DIR, vszs[i]) : nothing),
    ) for i in 1:length(pngs)]
    open(joinpath(out, "run_scenes.json"), "w") do io
        JSON3.write(io, (; meta = meta, frames = frames))
    end

    wheelname = _copy_embed_assets(out)
    write(joinpath(out, "index.html"),
          _index_html(viz, name, pngs, vszs, can_vsz, wheelname))
    @info "EnzoViz page written" outdir = out frames = length(viz.frames)
    return out
end

# Copy the prebuilt <veusz-figure> bundle + WASM from the fork (or this package's
# assets/) so the page is self-contained. Returns the veusz wheel's canonical
# basename that was copied (or "" if none found).
function _copy_embed_assets(out::AbstractString)
    assets = joinpath(@__DIR__, "..", "assets")
    fork = get(ENV, "ENZOVIZ_VEUSZ", "/Users/tabel/Projects/veusz")
    distembed = joinpath(fork, "veusz-tauri", "dist-embed")

    # 1) the embed JS bundle
    for cand in (joinpath(assets, _EMBED_JS),
                 joinpath(distembed, "veusz-embed.js"),
                 joinpath(distembed, "veusz-embed.iife.js"))
        if isfile(cand)
            cp(cand, joinpath(out, _EMBED_JS); force = true)
            break
        end
    end
    # 2) the WASM assets directory (vello painter)
    for cand in (joinpath(assets, "wasm"), joinpath(distembed, "wasm"))
        if isdir(cand)
            dst = joinpath(out, "wasm")
            isdir(dst) && rm(dst; recursive = true)
            cp(cand, dst)
            break
        end
    end
    # 3) a veusz wheel — Pyodide micropip-installs it so `import veusz` works in
    #    the browser (the live editor fails with ModuleNotFoundError without it).
    #    Keep the CANONICAL wheel filename (`veusz-X.Y.Z-py3-none-any.whl`):
    #    micropip parses the filename and rejects a renamed one. Returns the
    #    basename copied (or "" if no wheel found).
    for dir in (joinpath(assets, "wheels"), joinpath(distembed, "wheels"),
                joinpath(fork, "dist"))
        isdir(dir) || continue
        whls = sort(filter(f -> endswith(f, ".whl") && startswith(basename(f), "veusz"),
                           readdir(dir; join = true)))
        if !isempty(whls)
            wheelname = basename(whls[end])
            cp(whls[end], joinpath(out, wheelname); force = true)
            return wheelname
        end
    end
    return ""
end

# ── flipbook page (static; the default `finalize!` output) ───────────────────

function _index_html(viz::VizSession, name::AbstractString, pngs, vszs, can_vsz::Bool,
                     wheelname::AbstractString = "")
    w, h = viz.size
    has_bundle = isfile(joinpath(viz.outdir, _EMBED_JS))
    has_wheel  = !isempty(wheelname) && isfile(joinpath(viz.outdir, wheelname))
    can_edit   = can_vsz && has_bundle && has_wheel
    pngs_js = "[" * join(("\"$p\"" for p in [joinpath(_FRAME_DIR, p) for p in pngs]), ",") * "]"
    vsz_js  = "[" * join(("\"$v\"" for v in [joinpath(_FRAME_DIR, v) for v in vszs]), ",") * "]"
    times_js = "[" * join((string(t) for t in viz.times), ",") * "]"
    editbtn = can_edit ? """
      <button id="editbtn" type="button">Edit this frame (live)</button>""" : ""
    # The shipped embed bundle omits the wgpu limit shim, so add it ourselves:
    # recent Chrome dropped `maxInterStageShaderComponents` from the WebGPU spec,
    # but the bundled wgpu still requests it, making requestDevice reject. Strip
    # any requiredLimit the adapter doesn't actually report.
    gpushim = can_edit ? raw"""<script>
    (function(){var A=self.GPUAdapter;if(!A||!A.prototype)return;
      var o=A.prototype.requestDevice;
      A.prototype.requestDevice=function(d){
        if(d&&d.requiredLimits){var s=this.limits,c={};
          for(var k in d.requiredLimits){if(s&&s[k]!==undefined)c[k]=d.requiredLimits[k];}
          d=Object.assign({},d,{requiredLimits:c});}
        return o.call(this,d);};})();
    </script>""" : ""
    embedscript = can_edit ?
        gpushim * """<script type="module" src="$(_EMBED_JS)"></script>""" : ""
    return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Vespa — $(name)</title>
  <style>
    body { font: 15px system-ui, sans-serif; margin: 1.5rem; color:#222; }
    h1 { font-size: 1.3rem; }
    img#frame { display:block; max-width:100%; height:auto; border:1px solid #ddd; image-rendering:auto; }
    .bar { display:flex; gap:1rem; align-items:center; flex-wrap:wrap; margin:.8rem 0; }
    input[type=range] { width: 360px; }
    .tag { font-size:.82rem; color:#666; }
    button { font: inherit; padding:.3rem .7rem; }
    #editwrap { margin-top:1rem; }
    .vz-loading { display:flex; align-items:center; gap:.6rem; color:#555;
                  margin:.6rem 0; font-size:.95rem; }
    .vz-spinner { width:1.1rem; height:1.1rem; border:2px solid #ccc;
                  border-top-color:#36c; border-radius:50%;
                  animation:vz-spin .8s linear infinite; }
    @keyframes vz-spin { to { transform:rotate(360deg); } }
  </style>
  $(embedscript)
</head>
<body>
  <h1>Vespa — $(name) <span class="tag">$(length(pngs)) frames · Vello flipbook</span></h1>
  <div class="bar">
    <button id="play" type="button">▶ Play</button>
    <input id="slider" type="range" min="1" max="$(length(pngs))" value="1">
    <span id="label"></span>
    $(editbtn)
  </div>
  <img id="frame" width="$(w)" height="$(h)" alt="frame">
  <div id="editwrap"></div>

  <script>
    const pngs = $(pngs_js), vsz = $(vsz_js), times = $(times_js);
    const img = document.getElementById("frame");
    const slider = document.getElementById("slider");
    const label = document.getElementById("label");
    const playBtn = document.getElementById("play");
    let idx = 0, playing = false, timer = null;

    function show(i) {
      idx = Math.max(0, Math.min(pngs.length - 1, i));
      img.src = pngs[idx];
      slider.value = idx + 1;
      const t = times[idx] !== undefined ? "  t = " + Number(times[idx]).toPrecision(4) : "";
      label.textContent = "frame " + (idx + 1) + "/" + pngs.length + t;
    }
    slider.addEventListener("input", () => show(slider.valueAsNumber - 1));
    function step() { show((idx + 1) % pngs.length); }
    function play() {
      playing = !playing;
      playBtn.textContent = playing ? "⏸ Pause" : "▶ Play";
      if (playing) { timer = setInterval(step, 120); } else { clearInterval(timer); }
    }
    playBtn.addEventListener("click", play);
    document.addEventListener("keydown", e => {
      if (e.key === "ArrowRight") show(idx + 1);
      else if (e.key === "ArrowLeft") show(idx - 1);
      else if (e.key === " ") { e.preventDefault(); play(); }
    });
    show(0);

    // On-demand live editor: mount the fork's <veusz-figure> for the current
    // frame's self-contained .vsz (real WASM colour/zoom editing). This renders
    // via Vello/WebGPU in the browser — if WebGPU is unavailable the figure
    // canvas stays blank (the embed only logs to console), so we check first
    // and show a clear message instead.
    const editbtn = document.getElementById("editbtn");
    let editing = false;
    function closeEditor() {
      editing = false;
      document.getElementById("editwrap").innerHTML = "";
      // restore the flipbook view + controls
      img.style.display = "";
      slider.disabled = false;
      playBtn.disabled = false;
      editbtn.textContent = "Edit this frame (live)";
      show(idx);   // refresh the static frame
    }
    if (editbtn) editbtn.addEventListener("click", async () => {
      // Toggle: a second click (now "Close editor") returns to the flipbook.
      if (editing) { closeEditor(); return; }
      const wrap = document.getElementById("editwrap");
      wrap.innerHTML = "";
      let gpuOk = false;
      try { gpuOk = !!(navigator.gpu && await navigator.gpu.requestAdapter()); } catch (e) { gpuOk = false; }
      if (!gpuOk) {
        wrap.innerHTML = '<p style="color:#a00;max-width:46rem">' +
          'The live editor renders with <b>WebGPU</b>, which this browser does not expose. ' +
          'Open this page in <b>Chrome</b> or <b>Safari\\u00a026+</b> (or enable WebGPU) to edit ' +
          'colour, zoom, and log axes interactively. The animated PNG flipbook above works everywhere.' +
          '</p>';
        return;
      }
      // Pause animation and hide the static flipbook image so the live editor
      // REPLACES it rather than stacking a second figure underneath.
      if (playing) play();
      editing = true;
      img.style.display = "none";
      slider.disabled = true;
      playBtn.disabled = true;
      editbtn.textContent = "Close editor";
      const fig = document.createElement("veusz-figure");
      fig.setAttribute("src", new URL(vsz[idx], location.href).href);
      fig.setAttribute("width", "$(w)");
      fig.setAttribute("height", "$(h)");
      fig.setAttribute("eager", "true");
      // Pyodide micropip-installs the veusz wheel (canonical PEP 427 filename —
      // micropip rejects a renamed wheel); the Vello WASM is served locally so
      // the editor is self-contained (only Pyodide core + numpy come from CDN).
      // wasm-base MUST be an absolute/resolvable URL: the embed does a dynamic
      // import(`\${base}/veusz_paint_wasm.js`), and a bare "wasm/…" specifier
      // throws "Failed to resolve module specifier" (→ silent blank canvas).
      fig.setAttribute("veusz-wheel", new URL("$(wheelname)", location.href).href);
      fig.setAttribute("wasm-base", new URL("wasm", location.href).href);

      // The figure's own status ("Ready") fires before micropip finishes
      // installing the veusz wheel and parsing the embedded data, so the panel
      // can sit blank for ~15-30s the first time with no feedback. Show our own
      // spinner until the WebGPU canvas actually appears, then remove it.
      const loading = document.createElement("div");
      loading.className = "vz-loading";
      loading.innerHTML = '<span class="vz-spinner"></span>' +
        '<span>Booting in-browser editor — first time loads Pyodide + Veusz ' +
        '(~15–30 s)…</span>';
      wrap.appendChild(loading);
      wrap.appendChild(fig);
      const t0 = Date.now();
      const poll = setInterval(() => {
        if (fig.querySelector("canvas")) {
          clearInterval(poll); loading.remove();
        } else if (Date.now() - t0 > 90000) {
          clearInterval(poll);
          loading.querySelector("span:last-child").textContent =
            "Still loading… if this persists, check the browser console (WebGPU required).";
        }
      }, 400);
    });
  </script>
</body>
</html>
"""
end

# ── serve-mode page: flipbook + live server re-render (adds log axis) ─────────

function _serve_html(viz::VizSession, name::AbstractString)
    w, h = viz.size
    nf = length(viz.frames)
    f1 = first(viz.fields)
    lo, hi = viz.ranges[f1]
    return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Vespa (live) — $(name)</title>
  <style>
    body { font: 15px system-ui, sans-serif; margin: 1.5rem; color:#222; }
    h1 { font-size: 1.3rem; }
    img#frame { display:block; max-width:100%; height:auto; border:1px solid #ddd; }
    .bar { display:flex; gap:.8rem; align-items:center; flex-wrap:wrap; margin:.6rem 0; }
    .bar label { display:flex; gap:.35rem; align-items:center; }
    input[type=range] { width: 360px; }
    .tag { font-size:.82rem; color:#666; }
  </style>
</head>
<body>
  <h1>Vespa — $(name) <span class="tag">live serve · server-side Vello re-render</span></h1>
  <div class="bar">
    <button id="play" type="button">▶ Play</button>
    <input id="slider" type="range" min="1" max="$(nf)" value="1">
    <span id="label"></span>
  </div>
  <div class="bar">
    <label>colormap
      <select id="cmap"><option>viridis</option><option>plasma</option>
      <option>inferno</option><option>magma</option><option>grey</option>
      <option>spectrum</option></select></label>
    <label>min <input id="vmin" type="number" step="any" value="$(lo)" style="width:9ch"></label>
    <label>max <input id="vmax" type="number" step="any" value="$(hi)" style="width:9ch"></label>
    <label>log color <input id="logc" type="checkbox"></label>
    <label>log axis <input id="logax" type="checkbox"></label>
  </div>
  <img id="frame" width="$(w)" height="$(h)" alt="server-rendered frame">

  <script>
    const NF = $(nf);
    const img = document.getElementById("frame");
    const slider = document.getElementById("slider");
    const label = document.getElementById("label");
    const playBtn = document.getElementById("play");
    const q = id => document.getElementById(id);
    let idx = 0, playing = false, timer = null;

    function url(i) {
      const p = new URLSearchParams({
        frame: i + 1, colormap: q("cmap").value,
        vmin: q("vmin").value, vmax: q("vmax").value,
        logcolor: q("logc").checked ? 1 : 0, logaxis: q("logax").checked ? 1 : 0,
        _: Date.now(),
      });
      return "/png?" + p.toString();
    }
    function show(i) {
      idx = Math.max(0, Math.min(NF - 1, i));
      img.src = url(idx);
      slider.value = idx + 1;
      label.textContent = "frame " + (idx + 1) + "/" + NF;
    }
    slider.addEventListener("input", () => show(slider.valueAsNumber - 1));
    ["cmap","vmin","vmax","logc","logax"].forEach(id => {
      q(id).addEventListener("input", () => show(idx));
      q(id).addEventListener("change", () => show(idx));
    });
    function step() { show((idx + 1) % NF); }
    playBtn.addEventListener("click", () => {
      playing = !playing;
      playBtn.textContent = playing ? "⏸ Pause" : "▶ Play";
      if (playing) timer = setInterval(step, 200); else clearInterval(timer);
    });
    show(0);
  </script>
</body>
</html>
"""
end
