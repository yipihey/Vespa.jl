# An ACTUAL decaying-turbulence run (Enzo ProblemType 106 / HydroMethod=3, HD_RK):
# a solenoidal spectral velocity IC (the way Enzo's Turbulence_Generator builds it
# — E(k) ∝ k^-4 over a few low-k shells, normalized to a target RMS Mach number),
# evolved on a triply-PERIODIC box with a CFL-limited timestep and NO forcing, so
# the turbulence decays. We integrate forward with the Metal MUSCL solver and track
# the physics: kinetic-energy decay, mass/total-energy conservation, RMS Mach.
#
# Run:  <juliaup-julia> --project=test bench/run_decaying_turbulence.jl [n] [solver] [mach] [tfinal]
#   solver ∈ {hancock, rk2};  e.g.  ... 128 hancock 1.0 1.0

using PPMKernels, KernelAbstractions, Printf, Random, LinearAlgebra
const KA = KernelAbstractions
try; @eval using Metal; catch err; @info "Metal not loadable — running on CPU" err; end

# ── Enzo-style solenoidal spectral turbulence IC (primitive fields, periodic) ──
# Divergence-free velocity = Σ_k A_k ê⊥(k) cos(2π k·x + φ_k), A_k ∝ |k|^(-specidx/2)
# (specidx=4 ⇒ |v_k|∝k^-2 ⇒ E(k)∝k^-4, Enzo's Larson-relation default), then the
# whole field is normalized so v_rms = mach·c_s. ρ uniform, P = ρ c_s²/γ.
function turbulence_ic(n::Int, ng::Int; mach, gamma, cs = 1.0, seed = 271,
                       kmin = 2, kmax = 3, specidx = 4.0)
    Random.seed!(seed)
    N = n + 2ng; dx = 1.0 / n
    X = Float64[(i - ng - 0.5) * dx for i in 1:N]          # periodic cell-centre coord
    vx = zeros(N, N, N); vy = zeros(N, N, N); vz = zeros(N, N, N)
    modes = [(kx, ky, kz) for kx in -kmax:kmax, ky in -kmax:kmax, kz in -kmax:kmax
             if kmin^2 <= kx^2 + ky^2 + kz^2 <= kmax^2]
    for (kx, ky, kz) in modes
        kk = sqrt(kx^2 + ky^2 + kz^2)
        amp = kk^(-specidx / 2)                            # |v_k| ∝ k^(-specidx/2)
        kh = (kx, ky, kz) ./ kk
        a = randn(3); a .-= dot(a, collect(kh)) .* collect(kh)   # project ⊥ k ⇒ solenoidal
        na = norm(a); na < 1e-12 && continue; a ./= na
        φ = 2π * rand(); a1, a2, a3 = amp .* a
        @inbounds for c in 1:N
            kzc = kz * X[c]
            for b in 1:N
                kyb = ky * X[b] + kzc
                for aa in 1:N
                    s = cos(2π * (kx * X[aa] + kyb) + φ)
                    vx[aa, b, c] += a1 * s; vy[aa, b, c] += a2 * s; vz[aa, b, c] += a3 * s
                end
            end
        end
    end
    # normalize to target Mach over the interior
    intr = (ng + 1):(N - ng)
    v2 = 0.0; nc = 0
    @inbounds for c in intr, b in intr, aa in intr
        v2 += vx[aa, b, c]^2 + vy[aa, b, c]^2 + vz[aa, b, c]^2; nc += 1
    end
    vrms = sqrt(v2 / nc); f = mach * cs / vrms
    vx .*= f; vy .*= f; vz .*= f
    P0 = cs^2 / gamma                                      # P = ρ c_s²/γ, ρ=1
    eint0 = P0 / (gamma - 1)                               # specific internal energy
    d = ones(N * N * N)
    vxf = vec(vx); vyf = vec(vy); vzf = vec(vz)
    etot = eint0 .+ 0.5 .* (vxf .^ 2 .+ vyf .^ 2 .+ vzf .^ 2)
    return (; d, vx = vxf, vy = vyf, vz = vzf, etot, dims = (N, N, N), dx, N)
end

# ── interior diagnostics: mass, kinetic / internal / total energy, RMS Mach ───
function diagnostics(D, S1, S2, S3, Tau, dims, ng, dx, gamma)
    nx, ny, nz = dims
    hD = PPMKernels.to_host(D); h1 = PPMKernels.to_host(S1); h2 = PPMKernels.to_host(S2)
    h3 = PPMKernels.to_host(S3); hT = PPMKernels.to_host(Tau)
    dV = dx^3; mass = 0.0; KE = 0.0; IE = 0.0; TE = 0.0; v2 = 0.0; cs2 = 0.0; dmin = Inf; dmax = -Inf
    @inbounds for k in (ng+1):(nz-ng), j in (ng+1):(ny-ng), i in (ng+1):(nx-ng)
        q = i + nx * (j - 1) + nx * ny * (k - 1)
        d = hD[q]; ke = 0.5 * (h1[q]^2 + h2[q]^2 + h3[q]^2) / d
        ie = hT[q] - ke                                    # ρ·eint = τ − ½ρ|v|²
        p = (gamma - 1) * ie
        mass += d; KE += ke; IE += ie; TE += hT[q]
        v2 += (h1[q]^2 + h2[q]^2 + h3[q]^2) / d^2; cs2 += gamma * p / d
        dmin = min(dmin, d); dmax = max(dmax, d)
    end
    (; mass = mass * dV, KE = KE * dV, IE = IE * dV, TE = TE * dV,
       mach = sqrt(v2 / cs2), dmin, dmax)
end

# ── density mid-plane slice → PNG (jet-ish colormap, via PPM + macOS `sips`) ───
function save_density_png(D, dims, ng, path; gamma = 1.4)
    nx, ny, nz = dims; h = PPMKernels.to_host(D)
    k = nz ÷ 2
    sl = [Float64(h[i + nx * (j - 1) + nx * ny * (k - 1)]) for i in (ng+1):(nx-ng), j in (ng+1):(ny-ng)]
    lo, hi = extrema(sl); rng = hi > lo ? hi - lo : 1.0
    jet(t) = (clamp(1.5 - abs(4t - 3), 0, 1), clamp(1.5 - abs(4t - 2), 0, 1), clamp(1.5 - abs(4t - 1), 0, 1))
    m = size(sl, 1)
    ppm = path * ".ppm"
    open(ppm, "w") do io
        write(io, "P6\n$m $m\n255\n")
        for j in 1:m, i in 1:m                              # row-major for PPM
            r, g, b = jet((sl[i, j] - lo) / rng)
            write(io, UInt8(round(255r)), UInt8(round(255g)), UInt8(round(255b)))
        end
    end
    try; run(`sips -s format png $ppm --out $path`); rm(ppm); catch; end
    return (lo, hi)
end

# ── driver ────────────────────────────────────────────────────────────────────
n      = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 128
solver = length(ARGS) >= 2 ? Symbol(ARGS[2]) : :hancock
mach   = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 1.0
tfinal = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 1.0
recon  = length(ARGS) >= 5 ? Symbol(ARGS[5]) : :plm          # :plm | :ppm (hancock only)
tag    = solver === :hancock ? "$(solver)_$(recon)" : String(Symbol(solver))
const NG = 3; const GAMMA = 1.4; const COURANT = 0.3
const T = Float32
backend_name = PPMKernels.has_backend(:metal) ? :metal : :cpu

@printf("\nDecaying turbulence: %d^3  solver=%s  Mach0=%.2f  γ=%.1f  → t=%.2f  [%s/%s]\n",
        n, solver, mach, GAMMA, tfinal, backend_name, T)
ic = turbulence_ic(n, NG; mach = mach, gamma = GAMMA)
dims = ic.dims; dx = ic.dx; ngc = dims[1] - 2NG
be = PPMKernels.backend(backend_name)
dev(a) = PPMKernels.to_device(be, a, T)
D = similar(dev(ic.d)); S1 = similar(D); S2 = similar(D); S3 = similar(D); Tau = similar(D)
PPMKernels.prim_to_cons!(D, S1, S2, S3, Tau, dev(ic.d), dev(ic.vx), dev(ic.vy), dev(ic.vz), dev(ic.etot))
ws = similar(D)
bcfn(d, s1, s2, s3, tau) = PPMKernels.fill_periodic!(dims, NG, d, s1, s2, s3, tau)

step!(dt, order) = solver === :rk2 ?
    PPMKernels.muscl_step_3d!(D, S1, S2, S3, Tau, dims, NG; dt = dt, gamma = GAMMA, dx = dx, bc! = bcfn) :
    PPMKernels.muscl_hancock_step_3d!(D, S1, S2, S3, Tau, dims, NG; dt = dt, gamma = GAMMA, dx = dx,
                                      order = order, bc! = bcfn, recon = recon)

outdir = mkpath(joinpath(@__DIR__, "turb_out"))
d0 = diagnostics(D, S1, S2, S3, Tau, dims, NG, dx, GAMMA)
save_density_png(D, dims, NG, joinpath(outdir, "rho_$(tag)_t0.png"))
@printf("%-6s %-9s %-9s %-11s %-11s %-11s %-9s %-9s\n",
        "step", "t", "dt", "KE", "IE", "TE", "Mach", "ρ[min,max]")
@printf("%-6d %-9.4f %-9s %-11.5g %-11.5g %-11.5g %-9.4f %.3f/%.3f\n",
        0, 0.0, "—", d0.KE, d0.IE, d0.TE, d0.mach, d0.dmin, d0.dmax)
hist = [(0.0, d0.KE, d0.IE, d0.TE, d0.mass, d0.mach)]

PPMKernels.with_pool() do
    t = 0.0; s = 0; KE0 = d0.KE
    twall = @elapsed while t < tfinal - 1e-9 && s < 5000
        PPMKernels.fill_periodic!(dims, NG, D, S1, S2, S3, Tau)
        vmax = PPMKernels.max_wavespeed(ws, D, S1, S2, S3, Tau; gamma = GAMMA)
        dt = min(COURANT * dx / vmax, tfinal - t)
        step!(dt, isodd(s) ? (3, 2, 1) : (1, 2, 3))
        t += dt; s += 1
        if s % 25 == 0 || t >= tfinal - 1e-9
            d = diagnostics(D, S1, S2, S3, Tau, dims, NG, dx, GAMMA)
            push!(hist, (t, d.KE, d.IE, d.TE, d.mass, d.mach))
            @printf("%-6d %-9.4f %-9.2e %-11.5g %-11.5g %-11.5g %-9.4f %.3f/%.3f\n",
                    s, t, dt, d.KE, d.IE, d.TE, d.mach, d.dmin, d.dmax)
            (isnan(d.KE) || isinf(d.KE)) && (println("  NaN/Inf — aborting"); break)
        end
    end
    df = diagnostics(D, S1, S2, S3, Tau, dims, NG, dx, GAMMA)
    @printf("\nfinished %d steps in %.1fs (%.0f Mcell/s avg)\n", s, twall, n^3 * s / twall / 1e6)
    @printf("KE: %.5g → %.5g  (%.1f%% dissipated)\n", KE0, df.KE, 100 * (1 - df.KE / KE0))
    @printf("mass conservation:  Δ/M = %.2e\n", abs(df.mass - d0.mass) / d0.mass)
    @printf("total-energy cons:  Δ/E = %.2e   (KE→IE: ΔIE=%.4g, −ΔKE=%.4g)\n",
            abs(df.TE - d0.TE) / d0.TE, df.IE - d0.IE, KE0 - df.KE)
end

save_density_png(D, dims, NG, joinpath(outdir, "rho_$(tag)_tf.png"))
open(joinpath(outdir, "ke_decay_$(tag).csv"), "w") do io
    println(io, "t,KE,IE,TE,mass,mach")
    for (t, ke, ie, te, m, ma) in hist
        @printf(io, "%.6f,%.8g,%.8g,%.8g,%.8g,%.6f\n", t, ke, ie, te, m, ma)
    end
end
println("wrote slices + ke_decay_$(tag).csv to $outdir")
