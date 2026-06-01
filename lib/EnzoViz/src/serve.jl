# Live "serve" mode: a dependency-free HTTP server (Julia `Sockets` stdlib) that
# upgrades a finalized page to *continuous* editing. The static page already does
# time/animation + colormap/vmin/vmax/log-color entirely client-side (WASM); the
# server adds the things the browser CANNOT do on its own — re-rendering a frame
# with a different **axis range** or **log axis/color-scaling** — by replaying the
# stored raster through the live veusz document and returning a fresh scene.
#
# No external HTTP dependency: we speak just enough HTTP/1.1 for a localhost tool.
# Requires the `VizSession` was built with `record=true`.

using Sockets

# ── tiny HTTP plumbing ───────────────────────────────────────────────────────

struct Request
    method::String
    path::String
    query::Dict{String,String}
end

function _read_request(io)::Union{Request,Nothing}
    line = readline(io)
    isempty(line) && return nothing
    parts = split(line, ' ')
    length(parts) >= 2 || return nothing
    method, target = parts[1], parts[2]
    # drain headers
    while true
        h = readline(io)
        (isempty(h) || h == "\r") && break
    end
    path, q = occursin('?', target) ? split(target, '?'; limit = 2) : (target, "")
    query = Dict{String,String}()
    if !isempty(q)
        for kv in split(q, '&')
            k, _, v = partition(kv, '=')
            query[_urldecode(k)] = _urldecode(v)
        end
    end
    return Request(String(method), String(path), query)
end

partition(s, c) = (i = findfirst(==(c), s); i === nothing ? (s, "", "") :
                   (s[1:prevind(s, i)], c, s[nextind(s, i):end]))

function _urldecode(s)
    s = replace(s, '+' => ' ')
    io = IOBuffer()
    i = firstindex(s)
    while i <= lastindex(s)
        c = s[i]
        if c == '%' && i + 2 <= lastindex(s)
            write(io, parse(UInt8, s[i+1:i+2]; base = 16))
            i += 3
        else
            write(io, c)
            i = nextind(s, i)
        end
    end
    return String(take!(io))
end

const _MIME = Dict(".html" => "text/html; charset=utf-8",
                   ".json" => "application/json",
                   ".js" => "text/javascript",
                   ".wasm" => "application/wasm",
                   ".png" => "image/png",
                   ".css" => "text/css")
_mime(path) = get(_MIME, lowercase(splitext(path)[2]), "application/octet-stream")

function _respond(io, status, body; ctype = "text/plain", extra = "")
    b = body isa AbstractString ? Vector{UInt8}(codeunits(body)) : body
    write(io, "HTTP/1.1 $status\r\n")
    write(io, "Content-Type: $ctype\r\n")
    write(io, "Content-Length: $(length(b))\r\n")
    write(io, "Access-Control-Allow-Origin: *\r\n")
    isempty(extra) || write(io, extra)
    write(io, "Connection: close\r\n\r\n")
    write(io, b)
    return nothing
end

_qget(q, k, default) = get(q, k, default)
_qbool(q, k) = get(q, k, "0") in ("1", "true", "on", "yes")
_qnum(q, k) = (v = get(q, k, ""); isempty(v) ? nothing : tryparse(Float64, v))

# ── server ───────────────────────────────────────────────────────────────────

"""
    serve(viz; host="127.0.0.1", port=8080, open=false) -> server

Start a localhost HTTP server that hosts `viz.outdir` (the finalized page +
assets) and adds a live `/recapture` endpoint for continuous axis/log editing.
The session must have been built with `record=true`. Returns the listening
`TCPServer`; call `close(server)` to stop. Runs the accept loop on a task.

Endpoints:
  * `GET /` , `/index.html`  → the **serve-mode** page (static page + live panel).
  * `GET /<file>`            → any asset under `outdir` (run_scenes.json, the
                               embed bundle, wasm/…).
  * `GET /recapture?frame=N&colormap=&vmin=&vmax=&logcolor=&logaxis=`
                             → fresh scene JSON for that frame with the requested
                               display settings applied (server-side re-render).
  * `GET /meta`              → small JSON: nframes, rank, fields, times.
"""
function serve(viz::VizSession; host::AbstractString = "127.0.0.1",
               port::Integer = 8080, open::Bool = false)
    viz.record ||
        error("serve requires VizSession(...; record=true) so frames can be re-rendered")
    finalize!(viz)                                   # ensure assets are on disk
    write(joinpath(viz.outdir, "index.html"), _serve_html(viz, viz.sim.problem.name))

    server = Sockets.listen(Sockets.getaddrinfo(host), Int(port))
    @info "EnzoViz serving" url = "http://$host:$port/" outdir = viz.outdir frames = length(viz.frames)
    errormonitor(@async _accept_loop(server, viz))
    if open
        try; run(`open "http://$host:$port/"`); catch; end
    end
    return server
end

function _accept_loop(server, viz)
    while isopen(server)
        local sock
        try
            sock = Sockets.accept(server)
        catch
            break          # server closed
        end
        @async _handle(sock, viz)
    end
end

function _handle(sock, viz)
    try
        req = _read_request(sock)
        req === nothing && return
        if req.path == "/recapture"
            _serve_recapture(sock, viz, req.query)
        elseif req.path == "/png"
            _serve_png(sock, viz, req.query)
        elseif req.path == "/meta"
            meta = Dict("nframes" => length(viz.frames), "rank" => viz.rank,
                        "fields" => collect(String.(viz.fields)),
                        "times" => viz.times, "nlevels" => viz.nlevels)
            _respond(sock, "200 OK", JSON3.write(meta); ctype = "application/json")
        else
            _serve_file(sock, viz, req.path)
        end
    catch e
        try; _respond(sock, "500 Internal Server Error", "error: $(e)"); catch; end
    finally
        close(sock)
    end
end

function _serve_recapture(sock, viz, q)
    frame = something(tryparse(Int, _qget(q, "frame", "1")), 1)
    frame = clamp(frame, 1, length(viz.rasters))
    scene = recapture(viz, frame;
        colormap = _qget(q, "colormap", "viridis"),
        vmin = _qnum(q, "vmin"), vmax = _qnum(q, "vmax"),
        logcolor = _qbool(q, "logcolor"), logaxis = _qbool(q, "logaxis"))
    _respond(sock, "200 OK", scene; ctype = "application/json")
end

# Server-side render of a frame with full settings applied (the robust path:
# colormap / vmin / vmax / log-color AND log-axis, all baked by re-capture, then
# rasterized through Vello). Returns PNG bytes.
function _serve_png(sock, viz, q)
    frame = something(tryparse(Int, _qget(q, "frame", "1")), 1)
    frame = clamp(frame, 1, length(viz.rasters))
    scene = recapture(viz, frame;
        colormap = _qget(q, "colormap", "viridis"),
        vmin = _qnum(q, "vmin"), vmax = _qnum(q, "vmax"),
        logcolor = _qbool(q, "logcolor"), logaxis = _qbool(q, "logaxis"))
    w, h = viz.size
    png = lock(viz.lock) do
        pyconvert(Vector{UInt8}, _pymods()[:px].render_scene_to_png(
            pybytes(codeunits(scene)), w, h, pytuple((1.0, 1.0, 1.0, 1.0)), viz.backend))
    end
    _respond(sock, "200 OK", png; ctype = "image/png",
             extra = "Cache-Control: no-store\r\n")
    return nothing
end

function _serve_file(sock, viz, path)
    rel = path == "/" ? "index.html" : lstrip(path, '/')
    # prevent path traversal
    full = normpath(joinpath(viz.outdir, rel))
    if !startswith(full, normpath(viz.outdir)) || !isfile(full)
        _respond(sock, "404 Not Found", "not found: $path")
        return
    end
    _respond(sock, "200 OK", read(full); ctype = _mime(full))
    return nothing
end
