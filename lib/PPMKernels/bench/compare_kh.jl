# Kelvin-Helmholtz benchmark following Lecoanet et al. 2015
# (arXiv:1509.03630), using the production 3D PPM kernels.
#
# This matches the paper's smooth periodic initial condition, domain geometry,
# gamma, pressure, density-jump parameter, and snapshot times.  The current
# PPMKernels solvers are Euler solvers, so this is the no-explicit-diffusion
# analogue of the paper setup, not the viscous/conductive Re=1e5 Navier-Stokes
# reference solution.
#
# Run:
#   julia --project=lib/PPMKernels/test lib/PPMKernels/bench/compare_kh.jl [N] [drat] [solver_substring]
#
# Examples:
#   julia --project=lib/PPMKernels/test lib/PPMKernels/bench/compare_kh.jl 32 1
#   julia --project=lib/PPMKernels/test lib/PPMKernels/bench/compare_kh.jl 64 0 Local

using PPMKernels, KernelAbstractions, Printf, Statistics
try
    @eval using Metal
catch err
    @info "Metal not loadable - CPU fallback" err
end

const _P = PPMKernels
const NG = 4
const NZ = 8
const GAMMA = 5 / 3
const P0 = 10.0
const UFLOW = 1.0
const AMP = 0.01
const A_SHEAR = 0.05
const SIGMA = 0.2
const Z1 = 0.5
const Z2 = 1.5
const CFL = 0.8
const SNAP_TIMES = (2.0, 4.0, 6.0, 8.0)

n = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 32
drat = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 1.0
solver_filter = length(ARGS) >= 3 ? ARGS[3] : ""

bkname = _P.has_backend(:metal) ? :metal : :cpu
be = _P.backend(bkname)
const T = Float32
dev(a) = _P.to_device(be, a, T)

const SOLVERS = [
    "Hancock-PLM",
    "Hancock-PPM-tr-2shk",
    "Local-PPM-tr-2shk",
    "PPM-DirectEuler",
    "PPML-trace",
]

qidx(i, j, k, dims) = i + dims[1] * (j - 1) + dims[1] * dims[2] * (k - 1)

function kh_ic(n, drat)
    nx = n
    ny = 2n
    nbx = nx + 2NG
    nby = ny + 2NG
    nzb = NZ + 2NG
    dims = (nbx, nby, nzb)
    Ntot = prod(dims)
    dx = 1.0 / n
    d = Vector{Float64}(undef, Ntot)
    vx = Vector{Float64}(undef, Ntot)
    vy = Vector{Float64}(undef, Ntot)
    vz = zeros(Float64, Ntot)
    dye = Vector{Float64}(undef, Ntot)

    @inbounds for k in 1:nzb, j in 1:nby, i in 1:nbx
        x = mod((i - NG - 0.5) / n, 1.0)
        z = mod((j - NG - 0.5) / n, 2.0)
        t1 = tanh((z - Z1) / A_SHEAR)
        t2 = tanh((z - Z2) / A_SHEAR)
        ρ = 1.0 + drat * 0.5 * (t1 - t2)
        ux = UFLOW * (t1 - t2 - 1.0)
        uz = AMP * sin(2.0 * pi * x) *
             (exp(-((z - Z1)^2) / SIGMA^2) + exp(-((z - Z2)^2) / SIGMA^2))
        c = 0.5 * (t2 - t1 + 2.0)
        q = qidx(i, j, k, dims)
        d[q] = ρ
        vx[q] = ux
        vy[q] = uz
        dye[q] = c
    end

    eint = fill(P0, Ntot) ./ ((GAMMA - 1) .* d)
    etot = similar(eint)
    @inbounds for q in eachindex(etot)
        etot[q] = eint[q] + 0.5 * (vx[q]^2 + vy[q]^2 + vz[q]^2)
    end
    return (; d, vx, vy, vz, dye, eint, etot, dims, N = Ntot, dx, nx, ny)
end

function initial_max_signal(ic)
    s = 0.0
    @inbounds for q in eachindex(ic.d)
        cs = sqrt(GAMMA * P0 / ic.d[q])
        s = max(s, abs(ic.vx[q]) + abs(ic.vy[q]) + cs)
    end
    return s
end

function active_fields(D, S1, S2, S3, Tau, ic)
    ρ = Matrix{Float64}(undef, ic.nx, ic.ny)
    vx = similar(ρ)
    vy = similar(ρ)
    pr = similar(ρ)
    k = NG + max(1, NZ ÷ 2)
    @inbounds for j0 in 1:ic.ny, i0 in 1:ic.nx
        q = qidx(i0 + NG, j0 + NG, k, ic.dims)
        r = Float64(D[q])
        u = Float64(S1[q]) / r
        v = Float64(S2[q]) / r
        w = Float64(S3[q]) / r
        τ = Float64(Tau[q])
        ρ[i0, j0] = r
        vx[i0, j0] = u
        vy[i0, j0] = v
        pr[i0, j0] = (GAMMA - 1) * (τ - 0.5 * r * (u*u + v*v + w*w))
    end
    return (; rho = ρ, vx, vy, pressure = pr)
end

function vorticity_z(vx, vy, dx)
    w = similar(vx)
    nx, ny = size(vx)
    @inbounds for j in 1:ny, i in 1:nx
        ip = i == nx ? 1 : i + 1
        im = i == 1 ? nx : i - 1
        jp = j == ny ? 1 : j + 1
        jm = j == 1 ? ny : j - 1
        w[i, j] = (vy[ip, j] - vy[im, j]) / (2dx) - (vx[i, jp] - vx[i, jm]) / (2dx)
    end
    return w
end

function lower_half(a, ic)
    a[:, 1:ic.nx]
end

function symmetry_error(a, ic)
    # Lecoanet et al. reflect-and-shift symmetry: z -> 2-z, x -> x+1/2.
    nx, ny = size(a)
    shift = nx ÷ 2
    err = 0.0
    norm = 0.0
    @inbounds for j in 1:ny, i in 1:nx
        ii = ((i - 1 + shift) % nx) + 1
        jj = ny - j + 1
        err += abs2(a[i, j] - a[ii, jj])
        norm += abs2(a[i, j])
    end
    return sqrt(err / max(norm, eps()))
end

function kh_metrics(f, wall, ic)
    ρ = f.rho
    p = f.pressure
    ω = vorticity_z(f.vx, f.vy, ic.dx)
    Δρ = max(drat, eps())
    mixed = count(x -> 1.0 + 0.1 * Δρ < x < 1.0 + 0.9 * Δρ, ρ) / length(ρ)
    key = mean(@. 0.5 * ρ * f.vy^2)
    enstrophy = mean(abs2, ω)
    pv = maximum(abs.(p .- P0)) / P0
    sym = symmetry_error(ρ, ic)
    return (; mixed, key, enstrophy, pv, sym, rhomin = minimum(ρ), rhomax = maximum(ρ),
              pmin = minimum(p), pmax = maximum(p), wall)
end

function setup_solver(name, ic)
    dims = ic.dims
    pbc(fs...) = _P.fill_periodic!(dims, NG, fs...)

    D = dev(ic.d)
    S1 = dev(ic.d .* ic.vx)
    S2 = dev(ic.d .* ic.vy)
    S3 = dev(ic.d .* ic.vz)
    Tau = dev(ic.d .* ic.etot)
    Ge = dev(ic.d .* ic.eint)
    direct = name == "PPM-DirectEuler"
    if direct
        E = dev(ic.etot)
        GeP = dev(ic.eint)
        Vx = dev(ic.vx)
        Vy = dev(ic.vy)
        Vz = dev(ic.vz)
        Z = dev(zeros(ic.N))
        pbc6(a, b, c, d, e, f) = _P.fill_periodic!(dims, NG, a, b, c, d, e, f)
        step_direct! = (dt, o) -> _P.ppm_step_3d!(D, E, GeP, Vx, Vy, Vz, Z, Z, Z, dims, NG;
            dt, gamma = GAMMA, dx = ic.dx, order = o, bc! = pbc6,
            idual = 1, iflatten = 3, isteep = 0, idiff = 0, gravity = 0, eta2 = 0.1)
        return (; step! = step_direct!, direct = true, D, E, Vx, Vy, Vz)
    end

    st = name == "PPML-trace" ? _P.ppml_alloc_state(D, dims, NG) : nothing
    st === nothing || _P.ppml_init_state!(st, D, S1, S2, S3, Tau; gamma = GAMMA, ge = Ge)

    step! = if name == "Hancock-PLM"
        (dt, o) -> _P.muscl_hancock_step_3d!(D, S1, S2, S3, Tau, dims, NG;
            dt, gamma = GAMMA, dx = ic.dx, order = o, bc! = pbc, ge = Ge, recon = :plm)
    elseif name == "Hancock-PPM-tr-2shk"
        (dt, o) -> _P.muscl_hancock_step_3d!(D, S1, S2, S3, Tau, dims, NG;
            dt, gamma = GAMMA, dx = ic.dx, order = o, bc! = pbc, ge = Ge,
            recon = :ppm, predictor = :trace, riemann = :twoshock)
    elseif name == "Local-PPM-tr-2shk"
        (dt, o) -> _P.muscl_hancock_step_3d!(D, S1, S2, S3, Tau, dims, NG;
            dt, gamma = GAMMA, dx = ic.dx, order = o, bc! = pbc, ge = Ge,
            face_periodic = true, recon = :ppm_local, predictor = :trace, riemann = :twoshock)
    elseif name == "PPML-trace"
        (dt, o) -> _P.ppml_step_3d!(D, S1, S2, S3, Tau, dims, NG;
            state = st, dt, gamma = GAMMA, dx = ic.dx, order = o, ge = Ge,
            face_periodic = true, predictor = :trace)
    else
        error("unknown solver $name")
    end
    return (; step!, direct = false, D, S1, S2, S3, Tau)
end

function solver_fields(solver, ic)
    hD = _P.to_host(solver.D)
    if solver.direct
        hE = _P.to_host(solver.E)
        hVx = _P.to_host(solver.Vx)
        hVy = _P.to_host(solver.Vy)
        hVz = _P.to_host(solver.Vz)
        hS1 = hD .* hVx
        hS2 = hD .* hVy
        hS3 = hD .* hVz
        hTau = hD .* hE
        return active_fields(hD, hS1, hS2, hS3, hTau, ic)
    end
    hS1 = _P.to_host(solver.S1)
    hS2 = _P.to_host(solver.S2)
    hS3 = _P.to_host(solver.S3)
    hTau = _P.to_host(solver.Tau)
    return active_fields(hD, hS1, hS2, hS3, hTau, ic)
end

function run_solver(name, ic, base_dt, snap_times)
    solver = setup_solver(name, ic)
    rows = NamedTuple[]
    t = 0.0
    total_wall = 0.0
    total_steps = 0
    _P.with_pool() do
        for target in snap_times
            nseg = max(1, ceil(Int, (target - t) / base_dt))
            dt = (target - t) / nseg
            wall = @elapsed for s in 1:nseg
                total_steps += 1
                solver.step!(dt, isodd(total_steps) ? (1, 2, 3) : (3, 2, 1))
            end
            total_wall += wall
            t = target
            f = solver_fields(solver, ic)
            push!(rows, (; time = target, steps = total_steps, fields = f,
                          metrics = kh_metrics(f, total_wall, ic)))
        end
    end
    return rows
end

esc(s) = replace(string(s), "&" => "&amp;", "<" => "&lt;", ">" => "&gt;", "\"" => "&quot;")
lerp(a, b, t) = a + (b - a) * clamp(t, 0.0, 1.0)

function density_color(x)
    r, g, b = density_rgb(x)
    return "rgb($r,$g,$b)"
end

function density_rgb(x)
    lo = 1.0
    hi = 1.0 + max(drat, 1e-6)
    t = clamp((x - lo) / (hi - lo), 0.0, 1.0)
    r = round(Int, lerp(18, 180, t))
    g = round(Int, lerp(72, 32, t))
    b = round(Int, lerp(140, 28, t))
    return (r, g, b)
end

function diverging_color(x, lim)
    r, g, b = diverging_rgb(x, lim)
    return "rgb($r,$g,$b)"
end

function diverging_rgb(x, lim)
    t = lim <= 0 ? 0.5 : clamp(0.5 + 0.5 * x / lim, 0.0, 1.0)
    if t < 0.5
        q = t / 0.5
        r = round(Int, lerp(40, 246, q))
        g = round(Int, lerp(90, 246, q))
        b = round(Int, lerp(170, 246, q))
    else
        q = (t - 0.5) / 0.5
        r = round(Int, lerp(246, 178, q))
        g = round(Int, lerp(246, 34, q))
        b = round(Int, lerp(246, 34, q))
    end
    return (r, g, b)
end

function sampled_heatmap(a; max_n = 256)
    nx, ny = size(a)
    sx = max(1, ceil(Int, nx / max_n))
    sy = max(1, ceil(Int, ny / max_n))
    xs = 1:sx:nx
    ys = 1:sy:ny
    return a[xs, ys]
end

function heatmap_svg(title, a; mode = :density, width = 320, height = 320)
    b = sampled_heatmap(a)
    nx, ny = size(b)
    sx = width / nx
    sy = height / ny
    lim = mode === :vorticity ? quantile(abs.(vec(b)), 0.98) : 1.0
    io = IOBuffer()
    println(io, """<figure><figcaption>$(esc(title))</figcaption><svg viewBox="0 0 $width $height" class="heat" role="img" aria-label="$(esc(title))">""")
    @inbounds for j in 1:ny, i in 1:nx
        y = height - j * sy
        c = mode === :density ? density_color(b[i, j]) : diverging_color(b[i, j], lim)
        print(io, """<rect x="$(round((i - 1) * sx; digits = 3))" y="$(round(y; digits = 3))" width="$(ceil(sx + 0.01))" height="$(ceil(sy + 0.01))" fill="$c"/>""")
    end
    println(io, "</svg></figure>")
    return String(take!(io))
end

function safe_slug(s)
    replace(lowercase(string(s)), r"[^a-z0-9]+" => "_")
end

function write_ppm(path, a; mode = :density, max_n = 512)
    b = sampled_heatmap(a; max_n)
    nx, ny = size(b)
    lim = mode === :vorticity ? quantile(abs.(vec(b)), 0.98) : 1.0
    open(path, "w") do io
        write(io, "P6\n$nx $ny\n255\n")
        @inbounds for j in ny:-1:1, i in 1:nx
            r, g, bl = mode === :density ? density_rgb(b[i, j]) : diverging_rgb(b[i, j], lim)
            write(io, UInt8(clamp(r, 0, 255)))
            write(io, UInt8(clamp(g, 0, 255)))
            write(io, UInt8(clamp(bl, 0, 255)))
        end
    end
    return path
end

function convert_ppm_to_png(ppm, png)
    magick = Sys.which("magick")
    if magick !== nothing
        run(`$magick $ppm $png`)
    else
        convert = Sys.which("convert")
        if convert !== nothing
        run(`$convert $ppm $png`)
        else
            sips = Sys.which("sips")
            sips === nothing && error("need ImageMagick convert or macOS sips to write PNG report images")
            run(`$sips -s format png $ppm --out $png`)
        end
    end
    rm(ppm; force = true)
    return png
end

function write_heatmap_png(imgdir, stem, a; mode = :density)
    ppm = joinpath(imgdir, stem * ".ppm")
    png = joinpath(imgdir, stem * ".png")
    write_ppm(ppm, a; mode)
    convert_ppm_to_png(ppm, png)
    return basename(png)
end

short_name(name) =
    name == "Hancock-PLM" ? "PLM" :
    name == "Hancock-PPM-tr-2shk" ? "PPM" :
    name == "Local-PPM-tr-2shk" ? "Local" :
    name == "PPM-DirectEuler" ? "Direct" :
    name == "PPML-trace" ? "PPML" : name

function write_metrics(outdir, rows, ic, base_dt)
    path = joinpath(outdir, "kh_lecoanet_metrics.csv")
    open(path, "w") do io
        println(io, "solver,N,ny,drat,t,steps,base_dt,cfl,backend,mixed_area,vertical_ke,enstrophy,pressure_wiggle,symmetry_error,rho_min,rho_max,p_min,p_max,wall_s")
        for r in rows, s in r.snapshots
            m = s.metrics
            @printf(io, "%s,%d,%d,%.12g,%.12g,%d,%.12g,%.12g,%s,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.6f\n",
                    r.solver, ic.nx, ic.ny, drat, s.time, s.steps, base_dt, CFL, bkname,
                    m.mixed, m.key, m.enstrophy, m.pv, m.sym, m.rhomin, m.rhomax,
                    m.pmin, m.pmax, m.wall)
        end
    end
    return path
end

function write_report(outdir, rows, ic, base_dt)
    html = joinpath(outdir, "kh_lecoanet_report.html")
    imgdir = mkpath(joinpath(outdir, "kh_lecoanet_png"))
    open(html, "w") do io
        println(io, """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Lecoanet KH Solver Comparison</title>
<style>
body { margin: 0; font: 14px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #172033; background: #f7f8fb; }
main { max-width: 1280px; margin: 0 auto; padding: 28px; }
h1 { margin: 0 0 8px; font-size: 26px; }
h2 { margin-top: 30px; font-size: 18px; }
.meta, .note { color: #526071; margin-bottom: 18px; line-height: 1.45; }
table { border-collapse: collapse; width: 100%; background: white; border: 1px solid #d8dee9; }
th, td { padding: 7px 9px; border-bottom: 1px solid #e5e9f0; text-align: right; }
th:first-child, td:first-child { text-align: left; }
th { color: #334155; background: #eef2f7; font-weight: 600; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 14px; }
figure { margin: 0; background: white; border: 1px solid #d8dee9; padding: 8px; }
figcaption { font-weight: 600; margin-bottom: 6px; color: #334155; }
.heat { width: 100%; height: auto; display: block; image-rendering: pixelated; }
</style>
</head>
<body><main>
<h1>Lecoanet et al. KH Solver Comparison</h1>
<div class="meta">arXiv:1509.03630 setup: N=$(ic.nx), domain 1 x 2, gamma=5/3, P0=10, uflow=1, A=0.01, a=0.05, sigma=0.2, drat=$(drat), snapshot times $(join(SNAP_TIMES, ", ")), backend=$(bkname)/$(T), CFL=$(CFL), base dt=$(Printf.@sprintf("%.4e", base_dt)).</div>
<div class="note">Important: PPMKernels currently evolves the Euler equations here. The paper's converged reference cases include explicit viscosity, thermal diffusion, and dye diffusion with Re=1e5 or 1e6. This report is therefore the paper-matched inviscid/no-explicit-diffusion analogue.</div>
<table>
<thead><tr><th>solver</th><th>t</th><th>mixed area</th><th>vertical KE</th><th>enstrophy</th><th>pressure wiggle</th><th>sym err</th><th>rho min</th><th>rho max</th><th>wall s</th></tr></thead>
<tbody>
""")
        for r in rows, s in r.snapshots
            m = s.metrics
            println(io, "<tr><td>$(esc(r.solver))</td><td>$(Printf.@sprintf("%.1f", s.time))</td><td>$(Printf.@sprintf("%.4f", m.mixed))</td><td>$(Printf.@sprintf("%.4e", m.key))</td><td>$(Printf.@sprintf("%.4e", m.enstrophy))</td><td>$(Printf.@sprintf("%.4e", m.pv))</td><td>$(Printf.@sprintf("%.4e", m.sym))</td><td>$(Printf.@sprintf("%.4f", m.rhomin))</td><td>$(Printf.@sprintf("%.4f", m.rhomax))</td><td>$(Printf.@sprintf("%.2f", m.wall))</td></tr>")
        end
        println(io, "</tbody></table>")
        for time in SNAP_TIMES
            println(io, "<h2>Density, lower half, t=$(Printf.@sprintf("%.1f", time))</h2><section class=\"grid\">")
            for r in rows
                s = only(filter(x -> x.time == time, r.snapshots))
                stem = "rho_t$(safe_slug(time))_$(safe_slug(r.solver))"
                img = write_heatmap_png(imgdir, stem, lower_half(s.fields.rho, ic); mode = :density)
                println(io, """<figure><figcaption>$(esc(r.solver))</figcaption><img class="heat" src="kh_lecoanet_png/$(esc(img))" alt="$(esc(r.solver)) density t=$(Printf.@sprintf("%.1f", time))"></figure>""")
            end
            println(io, "</section><h2>Vorticity, lower half, t=$(Printf.@sprintf("%.1f", time))</h2><section class=\"grid\">")
            for r in rows
                s = only(filter(x -> x.time == time, r.snapshots))
                ω = vorticity_z(s.fields.vx, s.fields.vy, ic.dx)
                stem = "vort_t$(safe_slug(time))_$(safe_slug(r.solver))"
                img = write_heatmap_png(imgdir, stem, lower_half(ω, ic); mode = :vorticity)
                println(io, """<figure><figcaption>$(esc(r.solver))</figcaption><img class="heat" src="kh_lecoanet_png/$(esc(img))" alt="$(esc(r.solver)) vorticity t=$(Printf.@sprintf("%.1f", time))"></figure>""")
            end
            println(io, "</section>")
        end
        println(io, "</main></body></html>")
    end
    return html
end

ic = kh_ic(n, drat)
base_dt = CFL * ic.dx / initial_max_signal(ic)
outdir = mkpath(joinpath(@__DIR__, "kh_out"))

@printf("\nLecoanet KH analogue — N=%d ny=%d drat=%.3g times=%s backend=%s/%s CFL=%.2f base_dt=%.4e\n",
        ic.nx, ic.ny, drat, join(SNAP_TIMES, ","), bkname, T, CFL, base_dt)
@printf("%-21s %-5s %-10s %-11s %-11s %-10s %-10s %-9s %-9s %-8s\n",
        "solver", "t", "mixed", "KE_y", "enstrophy", "pwiggle", "symerr", "rho_min", "rho_max", "wall")
println("-"^112)

rows = NamedTuple[]
for solver in SOLVERS
    !isempty(solver_filter) && !occursin(solver_filter, solver) && continue
    try
        snaps = run_solver(solver, ic, base_dt, SNAP_TIMES)
        for s in snaps
            m = s.metrics
            @printf("%-21s %-5.1f %-10.4f %-11.4e %-11.4e %-10.3e %-10.3e %-9.4f %-9.4f %-8.2f\n",
                    solver, s.time, m.mixed, m.key, m.enstrophy, m.pv,
                    m.sym, m.rhomin, m.rhomax, m.wall)
        end
        push!(rows, (; solver, short = short_name(solver), snapshots = snaps))
        flush(stdout)
    catch err
        @printf("%-21s failed: %s\n", solver, sprint(showerror, err))
        flush(stdout)
    end
end

metrics_path = write_metrics(outdir, rows, ic, base_dt)
html_path = write_report(outdir, rows, ic, base_dt)
println("\nwrote $metrics_path")
println("wrote $html_path")
