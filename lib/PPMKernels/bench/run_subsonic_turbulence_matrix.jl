# GPU subsonic decaying-turbulence matrix for the KA hydro path.
#
# Run:
#   julia --project=lib/PPMKernels/test \
#     lib/PPMKernels/bench/run_subsonic_turbulence_matrix.jl [n] [mach] [tfinal] [solvers] [outdir]
#
# Example:
#   julia --project=lib/PPMKernels/test \
#     lib/PPMKernels/bench/run_subsonic_turbulence_matrix.jl 128 0.3 1.0 \
#     plm-hll,ppm-hll,localppm-hll,plm-llf,plm-hllc
#
# The hydro hot loop stays on the selected KernelAbstractions backend. Host copies
# happen only for sparse diagnostics and final PNG/report output.

using PPMKernels, KernelAbstractions, Printf, Random, LinearAlgebra, Statistics
const KA = KernelAbstractions
try
    @eval using Metal
catch err
    @info "Metal not loadable; falling back to CPU backend" err
end

const P = PPMKernels
const NG = 4
const GAMMA = 1.4
const CS0 = 1.0
const COURANT = 0.3
const TDEV = Float32

const SOLVERS = Dict(
    "plm-hll"      => (; label = "PLM+HLL",       recon = :plm,       riemann = :hll,  predictor = :hancock, face_periodic = false),
    "ppm-hll"      => (; label = "PPM+HLL",       recon = :ppm,       riemann = :hll,  predictor = :hancock, face_periodic = false),
    "localppm-hll" => (; label = "LocalPPM+HLL",  recon = :ppm_local, riemann = :hll,  predictor = :trace,   face_periodic = true),
    "plm-llf"      => (; label = "PLM+LLF",       recon = :plm,       riemann = :llf,  predictor = :hancock, face_periodic = false),
    "ppm-llf"      => (; label = "PPM+LLF",       recon = :ppm,       riemann = :llf,  predictor = :hancock, face_periodic = false),
    "plm-hllc"     => (; label = "PLM+HLLC",      recon = :plm,       riemann = :hllc, predictor = :hancock, face_periodic = false),
    "ppm-hllc"     => (; label = "PPM+HLLC",      recon = :ppm,       riemann = :hllc, predictor = :hancock, face_periodic = false),
)

safe_slug(s) = replace(lowercase(string(s)), r"[^a-z0-9]+" => "_")
esc(s) = replace(string(s), "&" => "&amp;", "<" => "&lt;", ">" => "&gt;", "\"" => "&quot;")
lerp(a, b, t) = a + (b - a) * clamp(t, 0.0, 1.0)

function turbulence_ic(n::Int; mach, seed = 271, kmin = 2, kmax = 3, specidx = 4.0)
    Random.seed!(seed)
    nb = n + 2NG
    dx = 1.0 / n
    x = Float64[(i - NG - 0.5) * dx for i in 1:nb]
    vx = zeros(nb, nb, nb)
    vy = zeros(nb, nb, nb)
    vz = zeros(nb, nb, nb)
    kmin2 = kmin * kmin
    kmax2 = kmax * kmax
    modes = [(kx, ky, kz) for kx in -kmax:kmax, ky in -kmax:kmax, kz in -kmax:kmax
             if kmin2 <= kx*kx + ky*ky + kz*kz <= kmax2]
    twopi = 2.0 * pi
    for (kx, ky, kz) in modes
        kk = sqrt(float(kx*kx + ky*ky + kz*kz))
        amp = kk^(-specidx / 2)
        kh = (kx / kk, ky / kk, kz / kk)
        a = randn(3)
        adot = a[1] * kh[1] + a[2] * kh[2] + a[3] * kh[3]
        ax = a[1] - adot * kh[1]
        ay = a[2] - adot * kh[2]
        az = a[3] - adot * kh[3]
        an = sqrt(ax * ax + ay * ay + az * az)
        an < 1e-12 && continue
        ax *= amp / an
        ay *= amp / an
        az *= amp / an
        phase = twopi * rand()
        @inbounds for k in 1:nb, j in 1:nb
            base = kz * x[k] + ky * x[j]
            for i in 1:nb
                s = cos(twopi * (kx * x[i] + base) + phase)
                vx[i, j, k] += ax * s
                vy[i, j, k] += ay * s
                vz[i, j, k] += az * s
            end
        end
    end
    intr = (NG + 1):(nb - NG)
    v2 = 0.0
    nc = 0
    @inbounds for k in intr, j in intr, i in intr
        v2 += vx[i, j, k]^2 + vy[i, j, k]^2 + vz[i, j, k]^2
        nc += 1
    end
    fac = mach * CS0 / sqrt(v2 / nc)
    isfinite(fac) || error("turbulence_ic produced zero/non-finite velocity variance at n=$n")
    vx .*= fac
    vy .*= fac
    vz .*= fac
    eint0 = (CS0^2 / GAMMA) / (GAMMA - 1)
    ncell = nb^3
    d = ones(ncell)
    vxf = vec(vx)
    vyf = vec(vy)
    vzf = vec(vz)
    eint = fill(eint0, ncell)
    etot = eint .+ 0.5 .* (vxf .^ 2 .+ vyf .^ 2 .+ vzf .^ 2)
    return (; d, vx = vxf, vy = vyf, vz = vzf, eint, etot, dims = (nb, nb, nb), dx, ncell)
end

function stage_state(ic, be)
    dev(a) = P.to_device(be, a, TDEV)
    D = dev(ic.d)
    S1 = dev(ic.d .* ic.vx)
    S2 = dev(ic.d .* ic.vy)
    S3 = dev(ic.d .* ic.vz)
    Tau = dev(ic.d .* ic.etot)
    Ge = dev(ic.d .* ic.eint)
    return (; D, S1, S2, S3, Tau, Ge)
end

function host_fields(st)
    D = P.to_host(st.D)
    S1 = P.to_host(st.S1)
    S2 = P.to_host(st.S2)
    S3 = P.to_host(st.S3)
    Tau = P.to_host(st.Tau)
    return (; rho = D, vx = S1 ./ D, vy = S2 ./ D, vz = S3 ./ D, etot = Tau ./ D)
end

idx(i, j, k, nx, ny) = i + nx * (j - 1) + nx * ny * (k - 1)
wrap_index(i, lo, hi) = i > hi ? lo : (i < lo ? hi : i)

function diagnostics(f, dims, dx)
    nx, ny, nz = dims
    lo = NG + 1
    hi = nx - NG
    mass = 0.0
    ke = 0.0
    te = 0.0
    v2sum = 0.0
    cs2sum = 0.0
    rhosum = 0.0
    rho2sum = 0.0
    grad_rho2 = 0.0
    grad_v2 = 0.0
    dmin = Inf
    dmax = -Inf
    emin = Inf
    nc = 0
    @inbounds for k in lo:hi, j in lo:hi, i in lo:hi
        q = idx(i, j, k, nx, ny)
        rho = f.rho[q]
        vx = f.vx[q]
        vy = f.vy[q]
        vz = f.vz[q]
        v2 = vx * vx + vy * vy + vz * vz
        eint = f.etot[q] - 0.5 * v2
        ke_cell = 0.5 * rho * v2
        mass += rho
        ke += ke_cell
        te += rho * f.etot[q]
        v2sum += v2
        cs2sum += GAMMA * (GAMMA - 1) * max(eint, 0.0)
        rhosum += rho
        rho2sum += rho * rho
        dmin = min(dmin, rho)
        dmax = max(dmax, rho)
        emin = min(emin, eint)
        ip = wrap_index(i + 1, lo, hi)
        jp = wrap_index(j + 1, lo, hi)
        kp = wrap_index(k + 1, lo, hi)
        qx = idx(ip, j, k, nx, ny)
        qy = idx(i, jp, k, nx, ny)
        qz = idx(i, j, kp, nx, ny)
        grad_rho2 += (f.rho[qx] - rho)^2 + (f.rho[qy] - rho)^2 + (f.rho[qz] - rho)^2
        grad_v2 += (f.vx[qx] - vx)^2 + (f.vy[qx] - vy)^2 + (f.vz[qx] - vz)^2
        grad_v2 += (f.vx[qy] - vx)^2 + (f.vy[qy] - vy)^2 + (f.vz[qy] - vz)^2
        grad_v2 += (f.vx[qz] - vx)^2 + (f.vy[qz] - vy)^2 + (f.vz[qz] - vz)^2
        nc += 1
    end
    rho_mean = rhosum / nc
    rho_rms = sqrt(max(rho2sum / nc - rho_mean^2, 0.0)) / rho_mean
    dV = dx^3
    return (; mass = mass * dV, KE = ke * dV, TE = te * dV,
            vrms = sqrt(v2sum / nc),
            mach = sqrt(v2sum / max(cs2sum, eps(Float64))),
            rho_mean, rho_rms, rho_rough = sqrt(grad_rho2 / (3nc)) / rho_mean,
            v_rough = sqrt(grad_v2 / (3nc)), dmin, dmax, emin)
end

function density_slice(f, dims)
    nx, ny, nz = dims
    lo = NG + 1
    hi = nx - NG
    k = (lo + hi) ÷ 2
    a = Matrix{Float64}(undef, hi - lo + 1, hi - lo + 1)
    @inbounds for j in lo:hi, i in lo:hi
        a[i - lo + 1, j - lo + 1] = f.rho[idx(i, j, k, nx, ny)]
    end
    return a
end

function density_rgb(x, lo, hi)
    t = hi > lo ? clamp((x - lo) / (hi - lo), 0.0, 1.0) : 0.5
    if t < 0.5
        q = t / 0.5
        r = round(Int, lerp(22, 244, q))
        g = round(Int, lerp(58, 238, q))
        b = round(Int, lerp(120, 184, q))
    else
        q = (t - 0.5) / 0.5
        r = round(Int, lerp(244, 176, q))
        g = round(Int, lerp(238, 34, q))
        b = round(Int, lerp(184, 44, q))
    end
    return UInt8(clamp(r, 0, 255)), UInt8(clamp(g, 0, 255)), UInt8(clamp(b, 0, 255))
end

function write_density_png(path, a; lo = minimum(a), hi = maximum(a), max_n = 768)
    nx, ny = size(a)
    sx = max(1, ceil(Int, nx / max_n))
    sy = max(1, ceil(Int, ny / max_n))
    xs = collect(1:sx:nx)
    ys = collect(1:sy:ny)
    ppm = path * ".ppm"
    open(ppm, "w") do io
        write(io, "P6\n$(length(xs)) $(length(ys))\n255\n")
        @inbounds for jj in reverse(ys), ii in xs
            r, g, b = density_rgb(a[ii, jj], lo, hi)
            write(io, r, g, b)
        end
    end
    sips = Sys.which("sips")
    if sips !== nothing
        run(`$sips -s format png $ppm --out $path`)
        rm(ppm; force = true)
    else
        mv(ppm, path; force = true)
    end
    return path
end

function run_one(spec, ic, be, tfinal; diag_count = 8, max_steps = 100000)
    st = stage_state(ic, be)
    dims = ic.dims
    dx = ic.dx
    ws = similar(st.D)
    pbc(f...) = P.fill_periodic!(dims, NG, f...)
    step!(dt, order) = P.muscl_hancock_step_3d!(st.D, st.S1, st.S2, st.S3, st.Tau, dims, NG;
        dt = dt, gamma = GAMMA, dx = dx, order = order, bc! = pbc, ge = st.Ge,
        recon = spec.recon, riemann = spec.riemann, predictor = spec.predictor,
        face_periodic = spec.face_periodic)

    hist = NamedTuple[]
    f0 = host_fields(st)
    d0 = diagnostics(f0, dims, dx)
    isfinite(d0.KE) && isfinite(d0.rho_rms) && isfinite(d0.emin) ||
        error("non-finite initial diagnostics for $(spec.label): KE=$(d0.KE) rho_rms=$(d0.rho_rms) emin=$(d0.emin)")
    push!(hist, (; t = 0.0, step = 0, wall = 0.0, d0...))
    t = 0.0
    s = 0
    next_diag = tfinal / diag_count
    wall = @elapsed P.with_pool() do
        while t < tfinal - 1e-9 && s < max_steps
            P.fill_periodic!(dims, NG, st.D, st.S1, st.S2, st.S3, st.Tau, st.Ge)
            vmax = P.max_wavespeed(ws, st.D, st.S1, st.S2, st.S3, st.Tau; gamma = GAMMA)
            isfinite(vmax) && vmax > 0 ||
                error("non-finite CFL speed for $(spec.label) at step=$s t=$t: vmax=$vmax")
            dt = min(COURANT * dx / vmax, tfinal - t)
            step!(dt, isodd(s) ? (3, 2, 1) : (1, 2, 3))
            t += dt
            s += 1
            if t >= next_diag - 1e-12 || t >= tfinal - 1e-9
                f = host_fields(st)
                d = diagnostics(f, dims, dx)
                isfinite(d.KE) && isfinite(d.rho_rms) && isfinite(d.emin) ||
                    error("non-finite diagnostics for $(spec.label) at step=$s t=$t")
                push!(hist, (; t, step = s, wall = 0.0, d...))
                next_diag += tfinal / diag_count
                @printf("  %-13s step=%5d t=%.4f dt=%.3e KE/KE0=%.5f rho_rms=%.4e rough=%.4e\n",
                        spec.label, s, t, dt, d.KE / d0.KE, d.rho_rms, d.rho_rough)
                flush(stdout)
            end
        end
    end
    ff = host_fields(st)
    df = diagnostics(ff, dims, dx)
    isfinite(df.KE) && isfinite(df.rho_rms) && isfinite(df.emin) ||
        error("non-finite final diagnostics for $(spec.label): KE=$(df.KE) rho_rms=$(df.rho_rms) emin=$(df.emin)")
    return (; spec, hist, final = df, initial = d0, fields = ff,
            wall, steps = s, mcell_s = (dims[1] - 2NG)^3 * max(s, 1) / max(wall, eps()) / 1e6)
end

function write_outputs(outdir, results, ic, backend_name, mach, tfinal, slices)
    mkpath(outdir)
    imgdir = mkpath(joinpath(outdir, "png"))
    allvals = reduce(vcat, [vec(a) for a in values(slices)])
    slo = minimum(allvals)
    shi = maximum(allvals)
    image_names = Dict{String,String}()
    for (key, a) in slices
        name = safe_slug(key) * "_rho_tf.png"
        write_density_png(joinpath(imgdir, name), a; lo = slo, hi = shi)
        image_names[key] = joinpath("png", name)
    end
    open(joinpath(outdir, "summary.csv"), "w") do io
        println(io, "solver,recon,riemann,predictor,n,mach0,tfinal,steps,wall_s,mcell_s,ke_retained,rho_rms,rho_rough,v_rough,mass_err,te_err,dmin,dmax,emin")
        for r in results
            d0 = r.initial
            df = r.final
            sp = r.spec
            @printf(io, "%s,%s,%s,%s,%d,%.8g,%.8g,%d,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8e,%.8e,%.8g,%.8g,%.8g\n",
                    sp.label, sp.recon, sp.riemann, sp.predictor, ic.dims[1] - 2NG, mach, tfinal,
                    r.steps, r.wall, r.mcell_s, df.KE / d0.KE, df.rho_rms, df.rho_rough,
                    df.v_rough, abs(df.mass - d0.mass) / d0.mass, abs(df.TE - d0.TE) / d0.TE,
                    df.dmin, df.dmax, df.emin)
        end
    end
    open(joinpath(outdir, "history.csv"), "w") do io
        println(io, "solver,t,step,ke_retained,vrms,mach,rho_rms,rho_rough,v_rough,dmin,dmax,emin")
        for r in results, h in r.hist
            @printf(io, "%s,%.8g,%d,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g\n",
                    r.spec.label, h.t, h.step, h.KE / r.initial.KE, h.vrms, h.mach,
                    h.rho_rms, h.rho_rough, h.v_rough, h.dmin, h.dmax, h.emin)
        end
    end
    open(joinpath(outdir, "index.html"), "w") do io
        println(io, """
<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>GPU Subsonic Turbulence Matrix</title>
<style>
body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:24px;background:#f7f8fb;color:#141820}
table{border-collapse:collapse;width:100%;background:white}th,td{padding:7px 9px;border-bottom:1px solid #dfe3ea;text-align:right}th:first-child,td:first-child{text-align:left}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px;margin-top:18px}
figure{margin:0;background:white;border:1px solid #dfe3ea}figcaption{padding:8px 10px;font-weight:600}img{width:100%;display:block;image-rendering:auto}
.meta{color:#4b5563}.note{background:white;border-left:4px solid #475569;padding:10px 12px;margin:14px 0}
</style></head><body>
<h1>GPU Subsonic Turbulence Matrix</h1>
<p class="meta">n=$(ic.dims[1] - 2NG), Mach0=$(mach), tfinal=$(tfinal), backend=$(backend_name), precision=$(TDEV)</p>
<div class="note">Final density slices use one shared color scale: rho=$(round(slo; digits=6)) to $(round(shi; digits=6)). Lower density RMS and lower roughness indicate stronger damping of cell-scale compressive structure.</div>
<table><thead><tr><th>solver</th><th>KE retained</th><th>rho RMS</th><th>rho rough</th><th>v rough</th><th>Mcell/s</th><th>mass err</th></tr></thead><tbody>
""")
        for r in results
            df = r.final
            d0 = r.initial
            @printf(io, "<tr><td>%s</td><td>%.5f</td><td>%.4e</td><td>%.4e</td><td>%.4e</td><td>%.1f</td><td>%.2e</td></tr>\n",
                    esc(r.spec.label), df.KE / d0.KE, df.rho_rms, df.rho_rough,
                    df.v_rough, r.mcell_s, abs(df.mass - d0.mass) / d0.mass)
        end
        println(io, "</tbody></table><div class=\"grid\">")
        for r in results
            key = r.spec.label
            println(io, "<figure><figcaption>$(esc(key))</figcaption><img src=\"$(esc(image_names[key]))\" alt=\"$(esc(key)) final density\"></figure>")
        end
        println(io, "</div></body></html>")
    end
    return outdir
end

function main()
    n = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 128
    mach = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 0.3
    tfinal = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 1.0
    solver_arg = length(ARGS) >= 4 ? ARGS[4] : "plm-hll,ppm-hll,localppm-hll,plm-llf"
    outdir = length(ARGS) >= 5 ? ARGS[5] : joinpath(@__DIR__, "turb_out", "subsonic_gpu_n$(n)")
    solver_keys = split(solver_arg, ",")
    bad = [k for k in solver_keys if !haskey(SOLVERS, k)]
    isempty(bad) || error("unknown solver(s): $(join(bad, ", ")). Known: $(join(sort(collect(Base.keys(SOLVERS))), ", "))")
    backend_name = Symbol(get(ENV, "PPM_BACKEND", P.has_backend(:metal) ? "metal" : "cpu"))
    P.has_backend(backend_name) || error("PPM backend :$backend_name is not available")
    be = P.backend(backend_name)
    @printf("\nSubsonic decaying turbulence matrix: n=%d Mach0=%.3f tfinal=%.3f backend=%s precision=%s\n",
            n, mach, tfinal, backend_name, TDEV)
    @printf("solvers: %s\n\n", join(solver_keys, ", "))
    ic = turbulence_ic(n; mach)
    results = NamedTuple[]
    slices = Dict{String,Matrix{Float64}}()
    for key in solver_keys
        spec = SOLVERS[key]
        @printf("Running %s (recon=%s riemann=%s predictor=%s)\n", spec.label, spec.recon, spec.riemann, spec.predictor)
        r = run_one(spec, ic, be, tfinal)
        push!(results, r)
        slices[spec.label] = density_slice(r.fields, ic.dims)
        @printf("  done: steps=%d wall=%.2fs %.1f Mcell/s KE/KE0=%.5f rho_rms=%.4e rough=%.4e\n\n",
                r.steps, r.wall, r.mcell_s, r.final.KE / r.initial.KE, r.final.rho_rms, r.final.rho_rough)
        flush(stdout)
    end
    write_outputs(outdir, results, ic, backend_name, mach, tfinal, slices)
    println("wrote $(joinpath(outdir, "index.html"))")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
