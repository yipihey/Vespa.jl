# Phase C2 — couple the comoving background to the gas: the semi-implicit Hubble
# drag, the a-scaled CFL, and the expansion timestep limiter, on a UNIFORM box
# (no spatial structure ⇒ the 1/a flux divergence vanishes, isolating the drag).
# Oracles (γ=5/3, gravity off):
#   X1  peculiar velocity redshifts: v ∝ 1/a  (momentum drag, rate ȧ/a)
#   X2  uniform adiabatic gas cools: T ∝ a^{-2}  (energy drag, rate 2ȧ/a, γ=5/3)
#   X3  cosmology off is a no-op (sim.cosmo === nothing short-circuits every hook)

first_cell(sim) = (c = Ref{Any}(nothing);
                   for_each_cell(sim.backend) do cell; c[] === nothing && (c[] = cell); end;
                   c[])

function _is_uniform(sim; rtol = 1e-9)
    ρ0 = nothing; ok = true
    for_each_cell(sim.backend) do cell
        ρ = primitive_at(sim, cell)[1]
        ρ0 === nothing ? (ρ0 = ρ) : (isapprox(ρ, ρ0; rtol = rtol) || (ok = false))
    end
    return ok
end

# Uniform periodic box in code units; cosmology attached separately (gravity off).
function _uniform_cosmo_sim(n, ρ0, v0, p0; zi = 4.0, zf = 0.0, γ = 5 / 3)
    init(x, y, z) = (ρ0, v0, 0.0, 0.0, p0)
    prob = Problem(; name = "expansion", dims = (n,), domain = ((0.0, 1.0),), γ = γ,
                   bcs = Periodic(), init = init, tfinal = 1.0, cfl = 0.3)  # tfinal ignored
    sim = Simulation(UniformMesh(prob.dims, prob.domain), prob)
    enable_cosmology!(sim; OmegaMatter = 1.0, OmegaLambda = 0.0, HubbleConstantNow = 0.5,
                      ComovingBoxSize = 1.0, InitialRedshift = zi, FinalRedshift = zf,
                      gravity = false)
    return sim
end

@testset "X1: peculiar velocity redshifts as v ∝ 1/a" begin
    v0 = 1e-3
    sim = _uniform_cosmo_sim(16, 1.0, v0, 0.06)
    @test isapprox(sim.cosmo.a, 1.0; rtol = 1e-10)        # start at z_i
    evolve!(sim)
    af = sim.cosmo.a
    @test af > 4.5                                         # z_i=4,z_f=0 ⇒ a: 1→5
    # box stayed uniform (drag only); compare the live velocity to v0/a.
    v_meas = primitive_at(sim, first_cell(sim))[2]
    @info "X1 velocity redshift" a_final = af v_expected = v0 / af v_meas
    @test isapprox(v_meas, v0 / af; rtol = 1e-2)
    @test _is_uniform(sim)
end

@testset "X2: adiabatic gas cools as T ∝ a^{-2} (γ=5/3)" begin
    p0 = 0.1; ρ0 = 1.0
    sim = _uniform_cosmo_sim(16, ρ0, 0.0, p0)
    evolve!(sim)
    af = sim.cosmo.a
    p_meas = primitive_at(sim, first_cell(sim))[5]
    # T ∝ p/ρ ∝ p (ρ fixed): expect p ∝ a^{-2}.
    @info "X2 adiabatic cooling" a_final = af p_expected = p0 / af^2 p_meas
    @test isapprox(p_meas, p0 / af^2; rtol = 1.5e-2)
    @test _is_uniform(sim)
end

@testset "X3: cosmology off is a no-op (pure hydro unchanged)" begin
    prob = sod_problem_defaults(n = 128)
    a = Simulation(UniformMesh(prob.dims, prob.domain), prob); evolve!(a)
    b = Simulation(UniformMesh(prob.dims, prob.domain), prob); evolve!(b)
    @test dump_fields(a).density == dump_fields(b).density   # deterministic, cosmo untouched
    @test a.cosmo === nothing
end
