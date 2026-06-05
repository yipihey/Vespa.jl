# Advected linear sound-wave accuracy comparison across all PLM/PPM/PPML solvers, GPU.
#
# A small-amplitude RIGHT-GOING acoustic eigenmode on a uniform background that is itself
# translated at +1 sound speed (u0 = cs), so the wave travels at u0 + cs = 2·cs and
# returns to its starting profile after each box crossing (t = L/(2cs) = 0.5). The exact
# solution is therefore a pure translation — any change is NUMERICAL. We measure, via the
# discrete Fourier mode at the wave's fundamental wavenumber:
#   • amplitude retention |c_k|_f / |c_k|_0   (numerical DISSIPATION; 1 = lossless)
#   • phase error (fraction of a wavelength)  (numerical DISPERSION / phase-speed error)
#   • harmonic distortion sqrt(Σ_{m≥2}|c_mk|²)/|c_k|  (waveform ASYMMETRY / steepening;
#     a pure sine has zero — energy leaking to 2k,3k,… means the wave went lop-sided)
# The +cs translation makes the upwinding act on a net flow, so asymmetric numerical
# diffusion (leading vs trailing edge) shows up here that a static (u0=0) wave would hide.
#
# Run:  <julia> --project=test bench/compare_soundwave.jl [nx] [k] [nperiods] [amp]
#   e.g. ... bench/compare_soundwave.jl 128 4 10 1e-3

using PPMKernels, KernelAbstractions, Printf
try; @eval using Metal; catch err; @info "Metal not loadable — CPU fallback" err; end
const _P = PPMKernels
const NG = 4; const GAMMA = 1.4; const CS0 = 1.0; const U0 = 1.0   # background = +1 sound speed

nx       = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 128
KW       = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 4             # wavelengths in the box
NPER     = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 10.0      # box translations to run
AMP      = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 1e-3
const NY = 8                                                       # thin transverse (1-D wave)
bkname = _P.has_backend(:metal) ? :metal : :cpu
be = _P.backend(bkname); const T = Float32
dev(a) = _P.to_device(be, a, T)

# ── right-going acoustic eigenmode IC (primitive), uniform in y,z ─────────────
# δp = cs²·δρ, δu = cs·δρ/ρ0 (right-going r₅); background u0 = cs added on top.
function wave_ic()
    nbx = nx + 2NG; nby = NY + 2NG; dims = (nbx, nby, nby); N = nbx * nby * nby; dx = 1.0 / nx
    d = Vector{Float64}(undef, N); u = similar(d); pr = similar(d)
    idx(i, j, k) = i + nbx * (j - 1) + nbx * nby * (k - 1)
    p0 = CS0^2 / GAMMA                                             # cs = sqrt(γp0/ρ0), ρ0=1
    for k in 1:nby, j in 1:nby, i in 1:nbx
        x = (i - NG - 0.5) / nx; s = AMP * sinpi(2 * KW * x); q = idx(i, j, k)
        d[q] = 1.0 + s; u[q] = U0 + CS0 * s; pr[q] = p0 + CS0^2 * s
    end
    eint = pr ./ ((GAMMA - 1) .* d); etot = eint .+ 0.5 .* u .^ 2
    vz = zeros(N)
    return (; d, u, vy = vz, vz, eint, etot, dims, dx, N, nbx, nby)
end

# density line along x at the transverse centre, interior only
function line(hρ, ic)
    nbx = ic.nbx; nby = ic.nby; jc = NG + NY ÷ 2; kc = jc
    [Float64(hρ[i + nbx * (jc - 1) + nbx * nby * (kc - 1)]) for i in (NG+1):(nbx-NG)]
end

# discrete Fourier coefficient of `prof` (length n) at `m` wavelengths (complex amplitude)
function fmode(prof, m)
    n = length(prof); re = 0.0; im = 0.0
    @inbounds for j in 0:n-1
        θ = 2π * m * j / n; re += prof[j+1] * cos(θ); im -= prof[j+1] * sin(θ)
    end
    return complex(re, im) * 2 / n
end

# wave metrics: amplitude retention, phase error (frac of wavelength), harmonic distortion
function wave_metrics(prof_f, prof_0)
    c0 = fmode(prof_0, KW); cf = fmode(prof_f, KW)
    amp = abs(cf) / abs(c0)
    dphi = angle(cf) - angle(c0); dphi = mod(dphi + π, 2π) - π          # wrap to (−π,π]
    phase = dphi / (2π)                                                # in wavelengths
    harm = sqrt(sum(abs2, fmode(prof_f, m) for m in (2KW, 3KW, 4KW))) / abs(cf)
    (; amp, phase, harm)
end

const SOLVERS = ["RK2 (PLM)", "Hancock-PLM", "Hancock-PPM", "PPM-DirectEuler", "PPML-trace", "PPML-Hancock"]

function run_solver(name, ic, dt, nsteps)
    dims = ic.dims; dx = ic.dx; N = ic.N
    pbc5(a, b, c, dd, ee) = _P.fill_periodic!(dims, NG, a, b, c, dd, ee)
    pbc6(a, b, c, dd, ee, ff) = _P.fill_periodic!(dims, NG, a, b, c, dd, ee, ff)
    if name == "PPM-DirectEuler"
        d = dev(ic.d); e = dev(ic.etot); ge = dev(ic.eint)
        vx = dev(ic.u); vy = dev(ic.vy); vz = dev(ic.vz); z = dev(zeros(N))
        full!(o) = _P.ppm_step_3d!(d, e, ge, vx, vy, vz, z, z, z, dims, NG; dt = dt, gamma = GAMMA, dx = dx,
                                   order = o, bc! = pbc6, idual = 0, iflatten = 3, isteep = 0, idiff = 0, gravity = 0, eta2 = 0.1)
        tw = _P.with_pool() do
            full!((1, 2, 3))
            @elapsed for s in 1:nsteps; full!(isodd(s) ? (1, 2, 3) : (3, 2, 1)); end
        end
        return (_P.to_host(d), tw)
    end
    D = dev(ic.d); S1 = dev(ic.d .* ic.u); S2 = dev(zeros(N)); S3 = dev(zeros(N)); Tau = dev(ic.d .* ic.etot)
    st = startswith(name, "PPML") ? _P.ppml_alloc_state(D, dims, NG) : nothing
    st === nothing || _P.ppml_init_state!(st, D, S1, S2, S3, Tau; gamma = GAMMA)
    step! = if name == "RK2 (PLM)"
        (o) -> _P.muscl_step_3d!(D, S1, S2, S3, Tau, dims, NG; dt = dt, gamma = GAMMA, dx = dx, bc! = pbc5)
    elseif name == "Hancock-PLM"
        (o) -> _P.muscl_hancock_step_3d!(D, S1, S2, S3, Tau, dims, NG; dt = dt, gamma = GAMMA, dx = dx, order = o, bc! = pbc5, recon = :plm)
    elseif name == "Hancock-PPM"
        (o) -> _P.muscl_hancock_step_3d!(D, S1, S2, S3, Tau, dims, NG; dt = dt, gamma = GAMMA, dx = dx, order = o, bc! = pbc5, recon = :ppm)
    elseif name == "PPML-trace"
        (o) -> _P.ppml_step_3d!(D, S1, S2, S3, Tau, dims, NG; state = st, dt = dt, gamma = GAMMA, dx = dx, order = o, face_periodic = true, predictor = :trace)
    else
        (o) -> _P.ppml_step_3d!(D, S1, S2, S3, Tau, dims, NG; state = st, dt = dt, gamma = GAMMA, dx = dx, order = o, face_periodic = true, predictor = :hancock)
    end
    tw = _P.with_pool() do
        step!((1, 2, 3))
        @elapsed for s in 1:nsteps; step!(isodd(s) ? (3, 2, 1) : (1, 2, 3)); end
    end
    return (_P.to_host(D), tw)
end

# ── driver ───────────────────────────────────────────────────────────────────
ic = wave_ic()
prof0 = line(ic.d, ic)
vmax = U0 + CS0 + CS0 * AMP
tfinal = NPER * 1.0 / (U0 + CS0)                                   # NPER box translations
dt = 0.3 * ic.dx / vmax; nsteps = ceil(Int, tfinal / dt)
ppw = nx / KW
@printf("\nAdvected sound wave — %d×%d²  k=%d (%.0f cells/λ)  A=%.0e  u0=cs ⇒ travels 2cs\n", nx, NY, KW, ppw, AMP)
@printf("→ %.0f box-translations (t=%.3f), fixed dt=%.2e ⇒ %d steps  [%s/%s]\n",
        NPER, tfinal, dt, nsteps, bkname, T)
@printf("\n%-17s %-11s %-13s %-13s %-8s\n", "solver", "amp kept", "phase err(λ)", "asym/distort", "wall(s)")
println("-"^66)
for name in SOLVERS
    try
        hρ, tw = run_solver(name, ic, dt, nsteps)
        any(isnan, hρ) && (@printf("%-17s  NaN\n", name); continue)
        m = wave_metrics(line(hρ, ic), prof0)
        @printf("%-17s %-11.4f %-+13.2e %-13.2e %-8.1f\n", name, m.amp, m.phase, m.harm, tw)
        flush(stdout)
    catch err
        @printf("%-17s  (failed: %s)\n", name, sprint(showerror, err)); flush(stdout)
    end
end
println()
