# Phase 3 (P3.2): Compton drag — inverse-Compton coupling of the gas temperature to
# the CMB (T_cmb = 2.725(1+z), rate Γ_C ∝ (1+z)⁴). `apply_compton_drag!` relaxes the
# gas thermal energy toward T_cmb semi-implicitly (density + kinetic energy
# untouched). Gate: starting hot (T0 ≫ T_cmb) at z=20, the temperature follows the
# analytic exponential T(t) = T_cmb + (T0−T_cmb)·e^{−Γ_C t}, and a single huge step
# is stable (no overshoot below T_cmb).

using RefMesh

@testset "Cosmology P3.2: Compton drag relaxes T_gas → T_cmb" begin
    zi, γ, μ, n = 20.0, 5 / 3, 0.6, 8
    dom = ntuple(_ -> (0.0, 1.0), 3)
    prob = Problem(name = "cb", dims = (n, n, n), domain = dom, γ = γ, bcs = Periodic(),
                   init = (x, y, z) -> (1.0, 0.0, 0.0, 0.0, 1.0), tfinal = 1.0)

    function fresh()
        s = Simulation(UniformMesh((n, n, n), dom), prob)
        enable_cosmology!(s; OmegaMatter = 1.0, OmegaLambda = 0.0, HubbleConstantNow = 0.5,
                          ComovingBoxSize = 64.0, InitialRedshift = zi, FinalRedshift = 6.0,
                          gravity = false)
        return s
    end
    Tof(sim, u) = Vespa._cell_temperature(sim, Vespa.get_U(sim.sv, _cell1(sim)), γ, μ, u.velocity)[1]
    _cell1(sim) = first((cc = Any[]; (for_each_cell(sim.backend) do c; push!(cc, c); nothing end); cc))
    setT!(sim, T, e_to_T) = for_each_cell(sim.backend) do c
        Vespa.set_U!(sim.sv, c, (1.0, 0.0, 0.0, 0.0, T / e_to_T)); return nothing
    end

    sim = fresh()
    u = cosmology_units(sim.cosmo, sim.cosmo.a)
    z = Vespa.redshift_at_scale_factor(sim.cosmo, sim.cosmo.a)
    Tcmb = Vespa.T_CMB0_K * (1 + z)
    e_to_T = (γ - 1) * μ * Vespa.M_H_CGS / Vespa.K_B_CGS * u.velocity^2
    ΓC = 8 * Vespa.SIGMA_T_CGS * Vespa.A_RAD_CGS * Tcmb^4 / (3 * Vespa.M_E_CGS * Vespa.C_CGS)

    T0 = 1000.0
    setT!(sim, T0, e_to_T)
    @test isapprox(Tof(sim, u), T0; rtol = 1e-10)      # temperature round-trips through units
    @test Tcmb < T0

    # relax for 0.7 e-folds in 80 semi-implicit steps → analytic exponential
    t_total_s = 0.7 / ΓC; M = 80; dt_code = (t_total_s / M) / u.time
    for _ in 1:M
        apply_compton_drag!(sim, dt_code; mu = μ)
    end
    Tana = Tcmb + (T0 - Tcmb) * exp(-ΓC * t_total_s)
    @test isapprox(Tof(sim, u), Tana; rtol = 0.02)     # follows the Compton rate (measured ~0.3%)
    @test Tof(sim, u) < T0 && Tof(sim, u) > Tcmb       # relaxed toward, not past, the CMB

    # density + momentum untouched by the drag
    U = Vespa.get_U(sim.sv, _cell1(sim))
    @test U[1] == 1.0 && U[2] == 0.0

    # stability: one enormous step cannot overshoot below T_cmb
    sim2 = fresh()
    setT!(sim2, T0, e_to_T)
    apply_compton_drag!(sim2, (1e6 / ΓC) / u.time; mu = μ)     # Γ_C·dt = 1e6 ≫ 1
    @test Tof(sim2, u) >= Tcmb * (1 - 1e-6)
    @test isapprox(Tof(sim2, u), Tcmb; rtol = 1e-3)    # → T_cmb in the stiff limit
end
