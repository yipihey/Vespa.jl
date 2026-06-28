# Phase 3 (P3.1): comoving dark-matter particle dynamics. The KDK push
# (`push_particles!`) uses the comoving drift `x += v·dt/a` and the semi-implicit
# Hubble-drag half-kick `v ← ((1−c)v + (g/a)·h)/(1+c)`, c = ½(ȧ/a)h — Enzo's
# particle scheme, consistent with the gas (`apply_expansion_terms!`). The gate is
# the collisionless Zel'dovich pancake: particles on a Lagrangian lattice ξ,
# displaced by the analytic map x = ξ − (A/k)sin(kξ) with A(a) = a·(1+z_c)/(1+z_i)
# and the matching peculiar velocity. In EdS the growing mode is D ∝ a, so the
# displacement amplitude must grow ∝ a (and track the analytic map before caustic).

using RefMesh

# advance the cosmology + particles together (expansion-limited steps).
function _evolve_cosmo_particles!(sim, a_target; maxstep = 400)
    solve_poisson!(sim, sim.grav)
    n = 0
    while sim.cosmo.a < a_target && n < maxstep
        dt = Vespa.expansion_dt(sim.cosmo)
        push_particles!(sim, dt)
        sim.t += dt
        Vespa.set_expansion!(sim.cosmo, sim.cosmo.t_initial + sim.t)
        n += 1
    end
    return n
end

@testset "Cosmology P3.1: comoving DM particle Zel'dovich growth" begin
    zi, zc, kx, n = 20.0, 1.0, 2π, 12
    Aof(a) = a * (1 + zc) / (1 + zi)
    vamp(a) = -sqrt(2 / 3) * sqrt(a) * (1 + zc) / ((1 + zi) * kx)
    xmap(ξ, a) = ξ - Aof(a) / kx * sin(kx * ξ)
    dom = ntuple(_ -> (0.0, 1.0), 3)

    prob = Problem(name = "zp", dims = (n, n, n), domain = dom, γ = 5 / 3, bcs = Periodic(),
                   init = (x, y, z) -> (1.0, 0.0, 0.0, 0.0, 1e-6), tfinal = 1.0)
    sim = Simulation(UniformMesh((n, n, n), dom), prob)
    enable_cosmology!(sim; OmegaMatter = 1.0, OmegaLambda = 0.0, HubbleConstantNow = 0.5,
                      ComovingBoxSize = 64.0, InitialRedshift = zi, FinalRedshift = 6.0,
                      MaxExpansionRate = 0.02)

    ξx = Float64[]; px = Float64[]; py = Float64[]; pz = Float64[]; vx = Float64[]
    for i in 0:n-1, j in 0:n-1, k in 0:n-1
        ξ = (i + 0.5) / n
        push!(ξx, ξ); push!(px, mod(xmap(ξ, 1.0), 1.0))
        push!(py, (j + 0.5) / n); push!(pz, (k + 0.5) / n)
        push!(vx, vamp(1.0) * sin(kx * ξ))
    end
    np = length(px)
    ps = enable_particles!(sim; px = px, py = py, pz = pz,
                           vx = vx, vy = zeros(np), vz = zeros(np), m = fill(1.0 / np, np))

    sdisp(p) = maximum(abs.(mod.(p .- ξx .+ 0.5, 1.0) .- 0.5))   # peak signed displacement
    disp0 = sdisp(ps.px)
    @test isapprox(disp0, Aof(1.0) / kx; rtol = 0.05)            # IC matches the map

    _evolve_cosmo_particles!(sim, 2.5)
    af = sim.cosmo.a
    dispf = sdisp(ps.px)

    @test af > 2.4                                              # actually expanded
    # growing mode D ∝ a: displacement amplitude grows ∝ a
    @test isapprox(dispf / disp0, af; rtol = 0.05)
    # and tracks the analytic Zel'dovich amplitude A(a)/k before caustic
    @test isapprox(dispf, Aof(af) / kx; rtol = 0.06)
end
