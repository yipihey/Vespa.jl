# Supersonic decaying-turbulence dissipation comparison across all PLM/PPM hydro
# solvers, GPU only. Solenoidal IC at a target RMS Mach, evolved to a fraction of the
# box crossing time (t_cross = L / v_rms); we report the FINAL RMS Mach number — a
# less-dissipative solver retains more turbulent velocity (higher final Mach). Dual
# energy is on for all (essential at high Mach), and a SINGLE fixed dt (from the IC's
# max wavespeed) is shared by every solver, so the only difference is the numerics.
#
# Run:  <julia> --project=test bench/compare_turb_dissipation.jl [n] [Mach] [t/t_cross]
#   e.g. ... bench/compare_turb_dissipation.jl 128 5 0.5

using PPMKernels, KernelAbstractions, Printf, Random, LinearAlgebra
const KA = KernelAbstractions
try; @eval using Metal; catch err; @info "Metal not loadable — CPU fallback" err; end
const _P = PPMKernels
const NG = 4; const GAMMA = 1.4; const CS0 = 1.0

n     = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 128
MACH  = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 5.0
TFRAC = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 0.5
bkname = _P.has_backend(:metal) ? :metal : :cpu
be = _P.backend(bkname); const T = Float32
dev(a) = _P.to_device(be, a, T)

# ── solenoidal spectral IC (primitive): ρ=1, P=ρcs²/γ, v ⊥ k normalised to v_rms ──
function turb_ic(n; mach, seed = 271, kmin = 2, kmax = 3, specidx = 4.0)
    Random.seed!(seed); nb = n + 2NG; dx = 1.0 / n
    X = Float64[(i - NG - 0.5) * dx for i in 1:nb]
    vx = zeros(nb, nb, nb); vy = similar(vx); vz = similar(vx)
    modes = [(kx, ky, kz) for kx in -kmax:kmax, ky in -kmax:kmax, kz in -kmax:kmax
             if kmin^2 <= kx^2 + ky^2 + kz^2 <= kmax^2]
    twopi = 2.0 * pi
    for (kx, ky, kz) in modes
        kk = sqrt(float(kx^2 + ky^2 + kz^2)); amp = kk^(-specidx / 2)
        khx = kx / kk; khy = ky / kk; khz = kz / kk
        a = randn(3); ad = a[1]*khx + a[2]*khy + a[3]*khz
        a1 = a[1] - ad*khx; a2 = a[2] - ad*khy; a3 = a[3] - ad*khz
        na = sqrt(a1*a1 + a2*a2 + a3*a3); na < 1e-6 && continue
        a1 *= amp/na; a2 *= amp/na; a3 *= amp/na
        ph = twopi * rand()
        @inbounds for c in 1:nb, b in 1:nb
            base = kz * X[c] + ky * X[b]
            for aa in 1:nb
                s = cos(twopi * (kx * X[aa] + base) + ph)
                vx[aa, b, c] += a1 * s; vy[aa, b, c] += a2 * s; vz[aa, b, c] += a3 * s
            end
        end
    end
    ix = (NG + 1):(nb - NG); s2 = 0.0
    @inbounds for k in ix, j in ix, i in ix
        s2 += vx[i, j, k]^2 + vy[i, j, k]^2 + vz[i, j, k]^2
    end
    vr = sqrt(s2 / length(ix)^3)
    f = mach * CS0 / vr; vx .*= f; vy .*= f; vz .*= f
    eint0 = (CS0^2 / GAMMA) / (GAMMA - 1)                       # P=ρcs²/γ ⇒ eint = P/((γ-1)ρ)
    N = nb^3; d = ones(N); vxf = vec(vx); vyf = vec(vy); vzf = vec(vz)
    eint = fill(eint0, N); etot = eint .+ 0.5 .* (vxf .^ 2 .+ vyf .^ 2 .+ vzf .^ 2)
    return (; d, vx = vxf, vy = vyf, vz = vzf, eint, etot, dims = (nb, nb, nb), dx, N)
end

# ── RMS-Mach / KE diagnostics from host primitive arrays (interior only) ──────
function diag(ρ, vx, vy, vz, eint, dims, dx)
    nx, ny, nz = dims; v2s = 0.0; cs2s = 0.0; ke = 0.0; m = 0.0; nc = 0; emin = Inf
    @inbounds for k in (NG+1):(nz-NG), j in (NG+1):(ny-NG), i in (NG+1):(nx-NG)
        q = i + nx * (j - 1) + nx * ny * (k - 1)
        v2 = vx[q]^2 + vy[q]^2 + vz[q]^2; e = eint[q]
        v2s += v2; cs2s += GAMMA * (GAMMA - 1) * max(e, 0.0); ke += 0.5 * ρ[q] * v2
        m += ρ[q]; emin = min(emin, e); nc += 1
    end
    (; mach = sqrt(v2s / cs2s), vrms = sqrt(v2s / nc), ke = ke * dx^3, mass = m * dx^3, emin)
end

# vmax (max signal speed) over the IC for the shared CFL timestep
function vmax_ic(ic)
    vm = 0.0
    @inbounds for q in 1:ic.N
        vm = max(vm, sqrt(ic.vx[q]^2 + ic.vy[q]^2 + ic.vz[q]^2) + sqrt(GAMMA * (GAMMA - 1) * ic.eint[q]))
    end
    vm
end

const SOLVERS = ["RK2 (PLM)", "Hancock-PLM", "Hancock-PPM", "PPM-DirectEuler", "PPML-trace", "PPML-Hancock"]

# evolve one solver `nsteps` of fixed `dt`; return (final-diag, walltime, mcell/s, nan)
function run_solver(name, ic, dt, nsteps)
    dims = ic.dims; dx = ic.dx; N = ic.N; nb = dims[1]
    pbc(f...) = _P.fill_periodic!(dims, NG, f...)
    # PPM-DirectEuler works on primitive arrays; everyone else on the conserved set.
    if name == "PPM-DirectEuler"
        d = dev(ic.d); e = dev(ic.etot); ge = dev(ic.eint)
        vx = dev(ic.vx); vy = dev(ic.vy); vz = dev(ic.vz); z = dev(zeros(N))
        # NOTE: DirectEuler has no bc! hook, so it only gets a per-STEP periodic refill
        # (not between its 3 internal sweeps) ⇒ a small conservation handicap vs the
        # conserved-variable solvers; its absolute numbers are not strictly comparable.
        full!(o) = begin
            _P.fill_periodic!(dims, NG, d, e, ge, vx, vy, vz)
            _P.ppm_step_3d!(d, e, ge, vx, vy, vz, z, z, z, dims, NG; dt = dt, gamma = GAMMA, dx = dx,
                            order = o, idual = 1, iflatten = 3, isteep = 0, idiff = 0, gravity = 0, eta2 = 0.1)
        end
        tw = _P.with_pool() do
            full!((1, 2, 3))                                   # warm
            @elapsed for s in 1:nsteps; full!(isodd(s) ? (1, 2, 3) : (3, 2, 1)); end
        end
        p = (_P.to_host(d), _P.to_host(vx), _P.to_host(vy), _P.to_host(vz), _P.to_host(ge))
        return (diag(p..., dims, dx), tw, n^3 * nsteps / tw / 1e6, any(isnan, p[1]))
    end
    # conserved staging (dual energy: Ge = ρ·eint)
    D = dev(ic.d); S1 = dev(ic.d .* ic.vx); S2 = dev(ic.d .* ic.vy); S3 = dev(ic.d .* ic.vz)
    Tau = dev(ic.d .* ic.etot); Ge = dev(ic.d .* ic.eint)
    st = startswith(name, "PPML") ? _P.ppml_alloc_state(D, dims, NG) : nothing
    st === nothing || _P.ppml_init_state!(st, D, S1, S2, S3, Tau; gamma = GAMMA, ge = Ge)
    step! = if name == "RK2 (PLM)"
        (o) -> _P.muscl_step_3d!(D, S1, S2, S3, Tau, dims, NG; dt = dt, gamma = GAMMA, dx = dx, bc! = pbc, ge = Ge)
    elseif name == "Hancock-PLM"
        (o) -> _P.muscl_hancock_step_3d!(D, S1, S2, S3, Tau, dims, NG; dt = dt, gamma = GAMMA, dx = dx, order = o, bc! = pbc, ge = Ge, recon = :plm)
    elseif name == "Hancock-PPM"
        (o) -> _P.muscl_hancock_step_3d!(D, S1, S2, S3, Tau, dims, NG; dt = dt, gamma = GAMMA, dx = dx, order = o, bc! = pbc, ge = Ge, recon = :ppm)
    elseif name == "PPML-trace"
        (o) -> _P.ppml_step_3d!(D, S1, S2, S3, Tau, dims, NG; state = st, dt = dt, gamma = GAMMA, dx = dx, order = o, ge = Ge, face_periodic = true, predictor = :trace)
    else  # PPML-Hancock
        (o) -> _P.ppml_step_3d!(D, S1, S2, S3, Tau, dims, NG; state = st, dt = dt, gamma = GAMMA, dx = dx, order = o, ge = Ge, face_periodic = true, predictor = :hancock)
    end
    tw = _P.with_pool() do
        step!((1, 2, 3))                                       # warm up (compile + prime pool/state)
        @elapsed for s in 1:nsteps; step!(isodd(s) ? (3, 2, 1) : (1, 2, 3)); end
    end
    hD = _P.to_host(D)
    p = (hD, _P.to_host(S1) ./ hD, _P.to_host(S2) ./ hD, _P.to_host(S3) ./ hD, _P.to_host(Ge) ./ hD)
    return (diag(p..., dims, dx), tw, n^3 * nsteps / tw / 1e6, any(isnan, p[1]))
end

# ── driver ───────────────────────────────────────────────────────────────────
ic = turb_ic(n; mach = MACH)
d0 = diag(ic.d, ic.vx, ic.vy, ic.vz, ic.eint, ic.dims, ic.dx)
tcross = 1.0 / d0.vrms                                          # L / v_rms
tfinal = TFRAC * tcross
vm = vmax_ic(ic); dt = 0.2 * ic.dx / vm; nsteps = ceil(Int, tfinal / dt)
@printf("\nSupersonic decaying turbulence — %d³  Mach0=%.2f  → %.2f t_cross (t=%.4f)  [%s/%s]\n",
        n, MACH, TFRAC, tfinal, bkname, T)
@printf("IC: v_rms=%.3f  Mach_rms=%.3f  KE=%.4g ;  fixed dt=%.2e ⇒ %d steps (vmax=%.2f)\n",
        d0.vrms, d0.mach, d0.ke, dt, nsteps, vm)
@printf("\n%-17s %-10s %-10s %-9s %-10s %-9s %-8s\n",
        "solver", "Mach_f", "v_rms_f", "KE diss%", "Δmass/M", "wall(s)", "Mcell/s")
println("-"^76)
for name in SOLVERS
    try
        (df, tw, mcs, nan) = run_solver(name, ic, dt, nsteps)
        @printf("%-17s %-10.3f %-10.3f %-9.1f %-10.1e %-9.1f %-8.1f%s\n",
                name, df.mach, df.vrms, 100 * (1 - df.ke / d0.ke), abs(df.mass - d0.mass) / d0.mass,
                tw, mcs, nan ? "  NaN!" : (df.emin < 0 ? "  e<0" : ""))
        flush(stdout)
    catch err
        @printf("%-17s  (failed: %s)\n", name, sprint(showerror, err)); flush(stdout)
    end
end
println()
