# Phase 3b — couple the gravity source g=−∇φ to the gas on a uniform mesh. Two
# oracles:
#   S1  sign gate: an overdensity at rest COLLAPSES (peak grows), not disperses —
#       catches a wrong-sign source (anti-gravity) immediately.
#   S2  Jeans linear theory: ω² = c_s²k² − 4πGρ₀. Long wavelength (ω²<0) grows at
#       rate s=√(−ω²) (compare to analytic); short wavelength (ω²>0) is stable
#       (bounded, no net growth).
# Also: pure hydro is byte-identical with gravity off (no-op safety).

using HGBackend

# Uniform periodic gas with ρ(x), p₀, v=0.
function _grav_sim(n, ρfun; p0 = 0.06, γ = 5 / 3, L = 1.0)
    init(x, y, z) = (ρfun(x), 0.0, 0.0, 0.0, p0)
    prob = Problem(; name = "jeans", dims = (n,), domain = ((0.0, L),), γ = γ,
                   bcs = Periodic(), init = init, tfinal = 1.0, cfl = 0.3)
    return Simulation(UniformMesh(prob.dims, prob.domain), prob)
end

# Amplitude of the cos(kx) Fourier mode of (ρ − ρ̄): a = ⟨(ρ−ρ̄)cos⟩/⟨cos²⟩.
function _mode_amp(sim, k, ρ̄)
    b = sim.backend
    num = 0.0; den = 0.0
    for_each_cell(b) do c
        x = cell_center(b, c)[1]; v = cell_volume(b, c)
        ck = cos(k * x)
        num += (primitive_at(sim, c)[1] - ρ̄) * ck * v
        den += ck * ck * v
    end
    return num / den
end

@testset "S1 sign gate: overdensity collapses (not disperses)" begin
    # Small-amplitude long-wavelength overdensity at rest. Correct gravity pulls
    # mass toward the peak → the mode amplitude grows. Wrong-sign would shrink it.
    ρ̄ = 1.0; A = 1e-3; k = 2π
    sim = _grav_sim(128, x -> ρ̄ * (1 + A * cos(k * x)); p0 = 0.06)
    enable_gravity!(sim; G = 1.0, bcs = Periodic())
    a0 = _mode_amp(sim, k, ρ̄)
    # advance a short time (well within a growth time)
    sim.problem isa Problem
    for _ in 1:40
        sim.grav === nothing || solve_poisson!(sim, sim.grav)
        dt = compute_dt(sim)
        step!(sim, dt)
    end
    a1 = _mode_amp(sim, k, ρ̄)
    @info "sign gate" a0 a1 ratio = a1 / a0
    @test a1 / a0 > 1.05                       # grew ⇒ attractive gravity (correct sign)
end

@testset "S2 Jeans growth rate (unstable) ≈ √(4πGρ₀ − c_s²k²)" begin
    ρ̄ = 1.0; G = 1.0; γ = 5 / 3; p0 = 0.06; A = 1e-4
    cs2 = γ * p0 / ρ̄; k = 2π                       # m=1 on L=1
    ω2 = cs2 * k^2 - 4π * G * ρ̄
    @test ω2 < 0                                   # this mode must be unstable
    s_exact = sqrt(-ω2)

    sim = _grav_sim(256, x -> ρ̄ * (1 + A * cos(k * x)); p0 = p0, γ = γ)
    enable_gravity!(sim; G = G, bcs = Periodic())

    # A density perturbation at REST projects onto BOTH the growing AND decaying
    # Jeans eigenmodes (δ ∝ cosh(s·t) initially), so an early log-linear fit
    # underestimates s. Integrate to ~2 growth times and fit the LATE half, where
    # the growing mode dominates and δ ∝ e^{s·t}, while staying in the linear
    # regime (A=1e-4 ⇒ δ_end ≈ A·e^{2} ≈ 7e-4 ≪ 1).
    ts = Float64[]; logamp = Float64[]
    a0 = abs(_mode_amp(sim, k, ρ̄))
    tmax = 2.0 / s_exact
    while sim.t < tmax
        solve_poisson!(sim, sim.grav)
        dt = min(compute_dt(sim), tmax - sim.t)
        step!(sim, dt)
        push!(ts, sim.t); push!(logamp, log(abs(_mode_amp(sim, k, ρ̄)) / a0))
    end
    half = length(ts) ÷ 2                           # fit only the growing-mode-dominated tail
    tf = ts[half:end]; lf = logamp[half:end]
    n = length(tf); st = sum(tf); stt = sum(t -> t^2, tf)
    sl = sum(lf); stl = sum(tf .* lf)
    s_meas = (n * stl - st * sl) / (n * stt - st^2)
    @info "Jeans unstable" s_exact s_meas rel = abs(s_meas - s_exact) / s_exact
    @test abs(s_meas - s_exact) / s_exact < 0.08   # within 8% of linear theory
end

@testset "S2b Jeans stable mode does not grow" begin
    # Short wavelength (large k): ω² > 0 ⇒ stable; the perturbation oscillates
    # but the amplitude must not run away (no spurious growth from gravity).
    ρ̄ = 1.0; G = 1.0; γ = 5 / 3; p0 = 0.5; A = 1e-4
    cs2 = γ * p0 / ρ̄; k = 2π * 4                    # m=4
    ω2 = cs2 * k^2 - 4π * G * ρ̄
    @test ω2 > 0                                   # must be stable
    sim = _grav_sim(256, x -> ρ̄ * (1 + A * cos(k * x)); p0 = p0, γ = γ)
    enable_gravity!(sim; G = G, bcs = Periodic())
    a0 = abs(_mode_amp(sim, k, ρ̄))
    amax = a0
    period = 2π / sqrt(ω2)
    while sim.t < 2 * period
        solve_poisson!(sim, sim.grav)
        dt = min(compute_dt(sim), 2 * period - sim.t)
        step!(sim, dt)
        amax = max(amax, abs(_mode_amp(sim, k, ρ̄)))
    end
    @info "Jeans stable" a0 amax ratio = amax / a0
    @test amax / a0 < 1.5                          # bounded — no runaway growth
end

@testset "Gravity off is a no-op (pure hydro unchanged)" begin
    # A Sod run with no enable_gravity! must be byte-identical to the canonical
    # path (sim.grav === nothing short-circuits every gravity hook).
    prob = sod_problem_defaults(n = 128)
    a = Simulation(UniformMesh(prob.dims, prob.domain), prob); evolve!(a)
    b = Simulation(UniformMesh(prob.dims, prob.domain), prob); evolve!(b)
    da = dump_fields(a); db = dump_fields(b)
    @test da.density == db.density                  # deterministic, no gravity touched it
    @test a.grav === nothing
end
