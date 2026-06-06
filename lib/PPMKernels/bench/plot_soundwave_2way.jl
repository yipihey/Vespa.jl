# Focused 2-way: Hancock-PPM-tr-2shk vs Hancock-PPM-HLLC on the advected acoustic wave.
# Same IC/metrics as compare_soundwave.jl. Renders a 3-panel PNG (manual raster → `sips`):
#   [HLLC vs exact] [tr-2shk vs exact] [both finals OVERLAID] so the difference is visible.
# Run from lib/PPMKernels:
#   <julia> --project=test bench/plot_soundwave_2way.jl [nx] [k] [nperiods] [amp]

using PPMKernels, KernelAbstractions, Printf
try; @eval using Metal; catch err; end
const _P = PPMKernels
const NG = 4; const GAMMA = 1.4; const CS0 = 1.0; const U0 = 1.0

nx   = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 128
KW   = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 4
NPER = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 10.0
AMP  = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 1e-3
const NY = 8
be = _P.backend(_P.has_backend(:metal) ? :metal : :cpu); const T = Float32
dev(a) = _P.to_device(be, a, T)

function wave_ic()
    nbx = nx + 2NG; nby = NY + 2NG; dims = (nbx, nby, nby); N = nbx * nby * nby; dx = 1.0 / nx
    d = Vector{Float64}(undef, N); u = similar(d); pr = similar(d)
    idx(i, j, k) = i + nbx * (j - 1) + nbx * nby * (k - 1); p0 = CS0^2 / GAMMA
    for k in 1:nby, j in 1:nby, i in 1:nbx
        x = (i - NG - 0.5) / nx; s = AMP * sinpi(2 * KW * x); q = idx(i, j, k)
        d[q] = 1.0 + s; u[q] = U0 + CS0 * s; pr[q] = p0 + CS0^2 * s
    end
    eint = pr ./ ((GAMMA - 1) .* d); etot = eint .+ 0.5 .* u .^ 2; vz = zeros(N)
    (; d, u, vy = vz, vz, eint, etot, dims, dx, N, nbx, nby)
end
vxline(hvx, ic) = (jc = NG + NY ÷ 2; [Float64(hvx[i + ic.nbx*(jc-1) + ic.nbx*ic.nby*(jc-1)]) for i in (NG+1):(ic.nbx-NG)])

# discrete Fourier coefficient at m wavelengths (for distortion metric)
function fmode(prof, m)
    n = length(prof); re = 0.0; im = 0.0
    @inbounds for j in 0:n-1; θ = 2π * m * j / n; re += prof[j+1]*cos(θ); im -= prof[j+1]*sin(θ); end
    complex(re, im) * 2 / n
end

# Riemann-ISOLATING pair: predictor=:trace held fixed in BOTH; only HLL vs two-shock differs.
const SOLVERS = ["Hancock-PPM-trace", "Hancock-PPM-tr-2shk"]

function run_vx(name, ic, dt, nsteps)
    dims = ic.dims; dx = ic.dx; N = ic.N
    pbc5(a, b, c, dd, ee) = _P.fill_periodic!(dims, NG, a, b, c, dd, ee)
    D = dev(ic.d); S1 = dev(ic.d .* ic.u); S2 = dev(zeros(N)); S3 = dev(zeros(N)); Tau = dev(ic.d .* ic.etot)
    step! = name == "Hancock-PPM-trace" ?
            (o) -> _P.muscl_hancock_step_3d!(D, S1, S2, S3, Tau, dims, NG; dt = dt, gamma = GAMMA, dx = dx, order = o, bc! = pbc5, recon = :ppm, predictor = :trace) :
            (o) -> _P.muscl_hancock_step_3d!(D, S1, S2, S3, Tau, dims, NG; dt = dt, gamma = GAMMA, dx = dx, order = o, bc! = pbc5, recon = :ppm, predictor = :trace, riemann = :twoshock)
    _P.with_pool() do
        for s in 0:nsteps; step!(isodd(s) ? (3, 2, 1) : (1, 2, 3)); end
    end
    vxline(_P.to_host(S1) ./ _P.to_host(D), ic)
end

# ── tiny raster line-plotter → PPM → PNG ──────────────────────────────────────
setpix!(img, px, py, rgb) = (H = size(img, 2); W = size(img, 3); (1 <= px <= W && 1 <= py <= H) && (img[:, py, px] .= rgb))
function seg!(img, x0, y0, x1, y1, rgb)
    n = max(1, ceil(Int, max(abs(x1 - x0), abs(y1 - y0))))
    for s in 0:n
        t = s / n; px = round(Int, x0 + (x1 - x0) * t); py = round(Int, y0 + (y1 - y0) * t)
        setpix!(img, px, py, rgb); setpix!(img, px, py + 1, rgb)
    end
end
curve!(img, xs, ys, X, Y, rgb) = for i in 1:length(xs)-1
    seg!(img, X(xs[i]), Y(ys[i]), X(xs[i+1]), Y(ys[i+1]), rgb)
end

ic = wave_ic()
vx0 = vxline(ic.u, ic)
vmax = U0 + CS0 + CS0 * AMP; tfinal = NPER * 1.0 / (U0 + CS0)
dt = 0.3 * ic.dx / vmax; nsteps = ceil(Int, tfinal / dt)
@printf("sound wave %d×%d²  k=%d (%.0f c/λ)  A=%.0e  %.0f crossings (%d steps) [%s]\n",
        nx, NY, KW, nx / KW, AMP, NPER, nsteps, _P.has_backend(:metal) ? "metal" : "cpu")

finals = Dict{String,Vector{Float64}}()
p2p0 = maximum(vx0) - minimum(vx0)
for name in SOLVERS
    finals[name] = run_vx(name, ic, dt, nsteps)
    v = finals[name]; amp = (maximum(v) - minimum(v)) / p2p0
    cf = fmode(v .- U0, KW); harm = sqrt(sum(abs2, fmode(v .- U0, m) for m in (2KW, 3KW, 4KW))) / abs(cf)
    @printf("  %-20s amp(p2p)=%.3f  distort(→□)=%.3f\n", name, amp, harm)
end

# δvx = vx − U0
ymax = maximum(maximum(abs.(v .- U0)) for v in values(finals)); ymax = max(ymax, AMP) * 1.1
xs = collect(1:nx)
pw, ph, mx, my = 360, 220, 50, 30
cols = 3; W = cols * pw + (cols + 1) * mx; H = ph + 2 * my
img = fill(0xf8 |> UInt8, 3, H, W)
gray = UInt8[150, 150, 150]; red = UInt8[210, 60, 50]; blue = UInt8[40, 90, 210]; axc = UInt8[210, 210, 210]
panels = ["Hancock-PPM-trace", "Hancock-PPM-tr-2shk", "OVERLAY"]
for (idxp, title) in enumerate(panels)
    c = idxp - 1; x0 = mx + c * (pw + mx); y0 = my
    X(xx) = x0 + (xx - 1) / (nx - 1) * pw
    Y(yy) = y0 + ph / 2 - (yy / ymax) * (ph / 2 - 6)
    for px in x0:x0+pw; setpix!(img, px, round(Int, Y(0.0)), axc); end       # baseline
    curve!(img, xs, vx0 .- U0, X, Y, gray)                                   # exact (gray)
    if title == "OVERLAY"
        curve!(img, xs, finals["Hancock-PPM-trace"]   .- U0, X, Y, red)      # trace+HLL red
        curve!(img, xs, finals["Hancock-PPM-tr-2shk"] .- U0, X, Y, blue)     # trace+2shk blue
    else
        curve!(img, xs, finals[title] .- U0, X, Y, title == "Hancock-PPM-trace" ? red : blue)
    end
end
outdir = mkpath(joinpath(@__DIR__, "turb_out")); ppm = joinpath(outdir, "soundwave_2way.ppm"); png = joinpath(outdir, "soundwave_2way_$(Int(NPER))x.png")
open(ppm, "w") do io
    write(io, "P6\n$W $H\n255\n")
    for py in 1:H, px in 1:W; write(io, img[1, py, px], img[2, py, px], img[3, py, px]); end
end
try; run(`sips -s format png $ppm --out $png`); rm(ppm); catch e; @info "sips failed" e; end
println("wrote $png")
println("panels L→R: trace+HLL (red) | trace+2shk (blue) | overlay (gray=exact) — predictor=:trace FIXED, only Riemann differs ; y-scale ±$(round(ymax/AMP,digits=2))·A")
