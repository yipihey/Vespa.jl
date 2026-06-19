# E1b (pilot) — full-replication of the hydro from Julia: drive a Sod shock tube
# end-to-end using ONLY the certified legacy `ppm_sweep_1d` kernel (the numerical
# core of Enzo's Eulerian PPM), looped from Julia. Mirrors EnzoModules'
# examples/ppm_sod.py, but via the native ccall binding.
#
# E1d (pilot) — mix-and-match parity: the SAME Sod problem run with Vespa's
# native Julia HLLC+PLM scheme. Both the legacy-PPM and the Julia-HLLC solutions
# match the exact Riemann solution, i.e. a Julia method and the legacy method are
# interchangeable on the same physics. (Running the Julia kernel directly on Enzo
# *grid memory* via an EnzoBackend is the next step and needs the full Enzo .so.)

const NGHOST = 3

# Evolve a Sod tube to `tfinal` using the legacy PPM sweep; return (x, ρ) active.
function legacy_ppm_sod(; n = 200, tfinal = 0.2, cfl = 0.4, γ = 1.4)
    idim = n + 2 * NGHOST
    i1, i2 = NGHOST + 1, NGHOST + n          # 1-based active range
    dx = 1.0 / n
    xc(k) = (k - 0.5) * dx                    # center of active cell k = 1..n
    d = zeros(idim); e = zeros(idim); u = zeros(idim)
    v = zeros(idim); w = zeros(idim); p = zeros(idim)
    for k in 1:n
        ρ0, p0 = xc(k) < 0.5 ? (1.0, 1.0) : (0.125, 0.1)
        idx = NGHOST + k
        d[idx] = ρ0; p[idx] = p0
        e[idx] = p0 / ((γ - 1) * ρ0)          # total specific energy (v=0)
    end
    refresh!(a) = (for g in 1:NGHOST; a[g] = a[i1]; a[i2 + g] = a[i2]; end)  # outflow
    t = 0.0
    while t < tfinal * (1 - 1e-12)
        for a in (d, e, u, v, w); refresh!(a); end
        @. p = (γ - 1) * d * (e - 0.5 * (u^2 + v^2 + w^2))
        refresh!(p)
        amax = 0.0
        for k in i1:i2
            amax = max(amax, abs(u[k]) + sqrt(γ * max(p[k], 0.0) / d[k]))  # floor: cold-gas guard
        end
        dt = min(cfl * dx / amax, tfinal - t)
        EnzoLib.ppm_sweep_1d!(d, e, u, v, w, p; i1 = i1, i2 = i2, dx = dx, dt = dt, gamma = γ)
        t += dt
    end
    return [xc(k) for k in 1:n], d[i1:i2]
end

# L1 density error of (x, ρ) vs the exact Riemann solution at time t.
function sod_l1(x, ρ; t = 0.2, γ = 1.4)
    WL = (1.0, 0.0, 1.0); WR = (0.125, 0.0, 0.1)       # exact_riemann_sample wants (ρ,u,p)
    err = 0.0
    for i in eachindex(x)
        ρe = exact_riemann_sample(WL, WR, γ, (x[i] - 0.5) / t)[1]
        err += abs(ρ[i] - ρe)
    end
    return err / length(x)
end

@testset "E1b: legacy PPM Sod (full replication) matches exact Riemann" begin
    x, ρ = legacy_ppm_sod(n = 200)
    l1 = sod_l1(x, ρ)
    @info "legacy ppm_sweep_1d Sod" cells = 200 L1_density = l1
    @test l1 < 3e-3                                   # README reports ~1.5e-3
    @test all(>(0), ρ)                                # positive, finite
end

@testset "E1d: Vespa Julia HLLC ≡ legacy PPM (both match exact Riemann)" begin
    # Native Vespa run, same problem.
    prob = sod_problem_defaults(n = 200)
    sim = Simulation(UniformMesh(prob.dims, prob.domain), prob)
    evolve!(sim)
    jf = dump_fields(sim)
    l1_julia = sod_l1(jf.x, jf.density)
    x, ρleg = legacy_ppm_sod(n = 200, tfinal = prob.tfinal)
    l1_legacy = sod_l1(x, ρleg)
    @info "mix-and-match parity" L1_julia_HLLC = l1_julia L1_legacy_PPM = l1_legacy
    @test l1_julia < 6e-3                             # Julia HLLC+PLM (2nd order) matches truth
    @test l1_legacy < 3e-3                            # legacy PPM (sharper) matches truth
    # Two different schemes on the same problem agree to within their combined
    # truncation error (a problem-level parity, not bitwise — see plan).
    @test maximum(abs.(jf.density .- ρleg)) < 0.05
end
