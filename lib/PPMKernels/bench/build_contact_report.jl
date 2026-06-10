using Printf

const OUTDIR = joinpath(@__DIR__, "contact_out")
const METRICS = joinpath(OUTDIR, "contact_metrics.csv")
const REPORT = joinpath(OUTDIR, "contact_report.html")

const COLORS = Dict(
    "Hancock-PLM" => "#8a8a8a",
    "Hancock-PPM-tr-2shk" => "#4c78a8",
    "Local-PPM-tr-2shk" => "#f58518",
    "Local-PPM-exact-label-THINC" => "#54a24b",
    "Local-PPM-carried-label-THINC" => "#e45756",
    "PPM-DirectEuler" => "#b279a2",
    "PPML-trace" => "#111111",
)

const ORDER = [
    "Hancock-PLM",
    "Hancock-PPM-tr-2shk",
    "Local-PPM-tr-2shk",
    "Local-PPM-exact-label-THINC",
    "Local-PPM-carried-label-THINC",
    "PPM-DirectEuler",
    "PPML-trace",
]

slug_to_solver(s) = replace(s, "_" => "-")
solver_to_slug(s) = replace(s, "-" => "_")

function read_csv(path)
    rows = split(read(path, String), '\n'; keepempty = false)
    header = split(first(rows), ',')
    data = [split(r, ',') for r in rows[2:end]]
    return header, data
end

function read_profiles(kind)
    header, data = read_csv(joinpath(OUTDIR, "$(kind)_profiles.csv"))
    x = Float64[parse(Float64, r[1]) for r in data]
    exact = Float64[parse(Float64, r[2]) for r in data]
    series = Dict{String,Vector{Float64}}()
    for (j, h) in enumerate(header[3:end])
        series[slug_to_solver(h)] = Float64[parse(Float64, r[j+2]) for r in data]
    end
    return x, exact, series
end

function read_metrics()
    header, data = read_csv(METRICS)
    idx = Dict(h => i for (i, h) in enumerate(header))
    out = Dict{Tuple{String,String},Dict{String,Float64}}()
    for r in data
        k = (r[idx["case"]], r[idx["solver"]])
        out[k] = Dict(
            "l1" => parse(Float64, r[idx["l1"]]),
            "width" => parse(Float64, r[idx["width_cells"]]),
            "over" => parse(Float64, r[idx["overshoot"]]),
            "under" => parse(Float64, r[idx["undershoot"]]),
            "pwig" => parse(Float64, r[idx["pressure_wiggle"]]),
            "wall" => parse(Float64, r[idx["wall_s"]]),
        )
    end
    return out
end

function esc(s)
    replace(s, "&" => "&amp;", "<" => "&lt;", ">" => "&gt;", "\"" => "&quot;")
end

function polyline(xs, ys, xmin, xmax, ymin, ymax, w, h, pad)
    pts = String[]
    for (x, y) in zip(xs, ys)
        px = pad + (x - xmin) / (xmax - xmin) * (w - 2pad)
        py = h - pad - (y - ymin) / (ymax - ymin) * (h - 2pad)
        push!(pts, @sprintf("%.2f,%.2f", px, py))
    end
    return join(pts, " ")
end

function svg_lines(title, x, exact, series; xrange = (0.0, 1.0), yrange = (0.9, 2.1),
                   residual = false, width = 900, height = 360)
    pad = 48
    xmin, xmax = xrange
    ymin, ymax = yrange
    sel = findall(i -> xmin <= x[i] <= xmax, eachindex(x))
    xs = x[sel]
    io = IOBuffer()
    println(io, """<svg viewBox="0 0 $width $height" class="plot" role="img" aria-label="$(esc(title))">""")
    println(io, """<rect x="0" y="0" width="$width" height="$height" fill="white"/>""")
    println(io, """<text x="$pad" y="24" class="plot-title">$(esc(title))</text>""")
    println(io, """<line x1="$pad" y1="$(height-pad)" x2="$(width-pad)" y2="$(height-pad)" class="axis"/>""")
    println(io, """<line x1="$pad" y1="$pad" x2="$pad" y2="$(height-pad)" class="axis"/>""")
    for yt in range(ymin, ymax; length = 5)
        py = height - pad - (yt - ymin) / (ymax - ymin) * (height - 2pad)
        println(io, @sprintf("""<line x1="%d" y1="%.2f" x2="%d" y2="%.2f" class="grid"/>""", pad, py, width-pad, py))
        println(io, @sprintf("""<text x="8" y="%.2f" class="tick">%.3g</text>""", py+4, yt))
    end
    for xt in range(xmin, xmax; length = 5)
        px = pad + (xt - xmin) / (xmax - xmin) * (width - 2pad)
        println(io, @sprintf("""<text x="%.2f" y="%d" class="tick" text-anchor="middle">%.2f</text>""", px, height-16, xt))
    end
    ex = exact[sel]
    if !residual
        pts = polyline(xs, ex, xmin, xmax, ymin, ymax, width, height, pad)
        println(io, """<polyline points="$pts" class="line exact"/>""")
    end
    for solver in ORDER
        haskey(series, solver) || continue
        yraw = series[solver][sel]
        ys = residual ? yraw .- ex : yraw
        pts = polyline(xs, ys, xmin, xmax, ymin, ymax, width, height, pad)
        color = COLORS[solver]
        lw = occursin("PPML", solver) ? 2.8 : 2.0
        println(io, """<polyline points="$pts" fill="none" stroke="$color" stroke-width="$lw" stroke-linejoin="round" stroke-linecap="round"/>""")
    end
    println(io, "</svg>")
    return String(take!(io))
end

function bar_svg(title, metrics, kind, field; width = 900, height = 300)
    pad = 58
    vals = [get(metrics, (kind, s), Dict(field => NaN))[field] for s in ORDER]
    vmax = maximum(filter(isfinite, vals)) * 1.12
    io = IOBuffer()
    println(io, """<svg viewBox="0 0 $width $height" class="plot" role="img" aria-label="$(esc(title))">""")
    println(io, """<rect x="0" y="0" width="$width" height="$height" fill="white"/>""")
    println(io, """<text x="$pad" y="24" class="plot-title">$(esc(title))</text>""")
    println(io, """<line x1="$pad" y1="$(height-pad)" x2="$(width-pad)" y2="$(height-pad)" class="axis"/>""")
    println(io, """<line x1="$pad" y1="$pad" x2="$pad" y2="$(height-pad)" class="axis"/>""")
    bw = (width - 2pad) / length(ORDER) * 0.68
    for (i, solver) in enumerate(ORDER)
        v = vals[i]
        isfinite(v) || continue
        cx = pad + (i - 0.5) / length(ORDER) * (width - 2pad)
        bh = v / vmax * (height - 2pad)
        y = height - pad - bh
        println(io, @sprintf("""<rect x="%.2f" y="%.2f" width="%.2f" height="%.2f" fill="%s"/>""",
                              cx - bw/2, y, bw, bh, COLORS[solver]))
        println(io, @sprintf("""<text x="%.2f" y="%.2f" class="bar-label" text-anchor="middle">%.3g</text>""", cx, y - 5, v))
        println(io, @sprintf("""<text x="%.2f" y="%d" class="solver-label" text-anchor="end" transform="rotate(-42 %.2f %d)">%s</text>""",
                              cx, height - 20, cx, height - 20, esc(solver)))
    end
    println(io, "</svg>")
    return String(take!(io))
end

function metrics_table(metrics, kind)
    io = IOBuffer()
    println(io, """<table><thead><tr><th>Solver</th><th>L1(rho)</th><th>Width cells</th><th>Overshoot</th><th>Undershoot</th><th>Pressure wiggle</th><th>Wall s</th></tr></thead><tbody>""")
    for solver in ORDER
        m = get(metrics, (kind, solver), nothing)
        m === nothing && continue
        println(io, @sprintf("""<tr><td><span class="swatch" style="background:%s"></span>%s</td><td>%.4e</td><td>%.0f</td><td>%.3e</td><td>%.3e</td><td>%.3e</td><td>%.2f</td></tr>""",
                              COLORS[solver], esc(solver), m["l1"], m["width"], m["over"], m["under"], m["pwig"], m["wall"]))
    end
    println(io, "</tbody></table>")
    return String(take!(io))
end

function section(kind, label, metrics)
    x, exact, series = read_profiles(kind)
    zoom = kind == "contact" ? (0.36, 0.64) : (0.18, 0.57)
    io = IOBuffer()
    println(io, """<section><h2>$label</h2>""")
    println(io, metrics_table(metrics, kind))
    println(io, """<div class="grid2">""")
    println(io, svg_lines("$label density profile", x, exact, series; yrange = (0.86, 2.16)))
    println(io, svg_lines("$label zoom near contacts", x, exact, series; xrange = zoom, yrange = (0.86, 2.16)))
    println(io, svg_lines("$label density residual", x, exact, series; yrange = (-0.08, 0.08), residual = true))
    println(io, bar_svg("$label transition width", metrics, kind, "width"))
    println(io, bar_svg("$label density overshoot", metrics, kind, "over"))
    println(io, bar_svg("$label density undershoot", metrics, kind, "under"))
    println(io, "</div></section>")
    return String(take!(io))
end

function main()
    metrics = read_metrics()
    html = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Local PPM Contact Preservation Report</title>
<style>
body { margin: 0; font: 15px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #1f2328; background: #f6f7f9; }
main { max-width: 1180px; margin: 0 auto; padding: 28px; }
h1 { margin: 0 0 8px; font-size: 30px; }
h2 { margin-top: 34px; border-top: 1px solid #d8dee4; padding-top: 24px; }
.lede { max-width: 900px; color: #57606a; }
.callout { background: #fff; border: 1px solid #d8dee4; border-left: 5px solid #e45756; padding: 14px 16px; margin: 18px 0; border-radius: 6px; }
.grid2 { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; align-items: start; }
.plot { width: 100%; border: 1px solid #d8dee4; border-radius: 6px; background: white; }
.axis { stroke: #57606a; stroke-width: 1; }
.grid { stroke: #eaeef2; stroke-width: 1; }
.tick, .solver-label, .bar-label { fill: #57606a; font-size: 11px; }
.plot-title { fill: #24292f; font-size: 15px; font-weight: 650; }
.line { fill: none; stroke-width: 1.5; }
.exact { stroke: #111; stroke-width: 1.2; stroke-dasharray: 4 4; opacity: .7; }
table { width: 100%; border-collapse: collapse; background: white; border: 1px solid #d8dee4; border-radius: 6px; overflow: hidden; margin: 12px 0 16px; }
th, td { padding: 7px 9px; border-bottom: 1px solid #eaeef2; text-align: right; font-variant-numeric: tabular-nums; }
th:first-child, td:first-child { text-align: left; }
.swatch { display: inline-block; width: 11px; height: 11px; border-radius: 2px; margin-right: 7px; vertical-align: -1px; }
@media (max-width: 900px) { .grid2 { grid-template-columns: 1fr; } main { padding: 16px; } }
</style>
</head>
<body>
<main>
<h1>Local PPM Contact Preservation</h1>
<p class="lede">Periodic Mach-4.23 contact advection at nx=128 after 10 box crossings. Dashed black lines are the exact density profiles. The key comparison is default local PPM, exact-label THINC as an upper bound, carried-label THINC with in-solver rho*a and rho*a^2 moment updates, and PPML as the contact-memory reference.</p>
<div class="callout"><strong>Summary:</strong> carried label moments inside local PPM narrow the contact from 10 to 8 cells and improve L1, but they add a small density ringing penalty. Exact label information does better, and PPML remains much sharper with no density overshoot.</div>
$(section("contact", "Single Contact", metrics))
$(section("top_hat", "Top-Hat Contacts", metrics))
</main>
</body>
</html>
"""
    open(REPORT, "w") do io
        write(io, html)
    end
    println(REPORT)
end

main()
