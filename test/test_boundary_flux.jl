# ADR-0003 part A: EnzoNG records the per-stage flux through each domain-boundary
# face (∫F·area dt) into a BoundaryFluxRegister. On the EnzoBackend a grid's outer
# faces ARE the coarse–fine interface, so this is the fine grid's RefinedFluxes (and
# a coarse grid's InitialFluxes at a subgrid) that Enzo's CorrectForRefinedFluxes
# consumes for conservation. The recording is correct iff the time-integrated
# boundary flux EXACTLY accounts for the change in total conserved mass:
#   Δ(total mass) == Σ_lo bflux[ρ] − Σ_hi bflux[ρ].

function _bflux_conservation_run()
    prob = Problem(; name = "ramp", dims = (64,), domain = ((0.0, 1.0),), γ = 1.4, bcs = Outflow(),
                   init = (x, y, z) -> (1.0 + x, 1.0, 0.0, 0.0, 2.0),   # ramp + flow ⇒ Δmass ≠ 0
                   tfinal = 1.0, cfl = 0.3)
    sim = Simulation(UniformMesh(prob.dims, prob.domain), prob)
    di = density_index(sim.model); V = 1 / 64
    mass() = sum(sim.sv[di][I] for I in CartesianIndices((64,))) * V
    breg = EnzoNG._bflux_register(sim)
    m0 = mass(); tot_lo = 0.0; tot_hi = 0.0; nonzero = false
    for _ in 1:20
        empty!(breg.flux)
        step!(sim, 5e-4; bflux = breg)
        nonzero |= any(v -> v[di] != 0, values(breg.flux))
        tot_lo += sum((v[di] for ((ax, sd, c), v) in breg.flux if sd === :lo); init = 0.0)
        tot_hi += sum((v[di] for ((ax, sd, c), v) in breg.flux if sd === :hi); init = 0.0)
    end
    return (Δmass = mass() - m0, lo_minus_hi = tot_lo - tot_hi, nonzero = nonzero,
            entries = length(breg.flux))
end

@testset "boundary-flux recording (ADR-0003 part A): conservation identity" begin
    r = _bflux_conservation_run()
    @test r.entries == 2                       # both domain ends recorded (1D)
    @test r.nonzero                            # a real, nonzero boundary flux
    @test abs(r.Δmass) > 1e-3                  # the run actually changes total mass
    @info "boundary-flux conservation" Δmass = r.Δmass lo_minus_hi = r.lo_minus_hi
    @test isapprox(r.Δmass, r.lo_minus_hi; atol = 1e-12)   # ∫boundary flux ≡ Δmass (to round-off)
end
