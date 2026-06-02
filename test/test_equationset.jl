# The EquationSet abstraction in action: a SECOND model with a different conserved
# variable LAYOUT (total energy first, then density, then momenta) must produce
# bit-identical physics to the default ideal hydro. This proves the driver,
# reconstruction, fluxes, and source terms are driven by the model's roles
# (density_index / momentum_indices / energy_index / nvars / kernels), not by any
# hardcoded variable choice — exactly the flexibility the design now provides.

# A reordered ideal-hydro equation set: conserved (E, ρ, ρvx, ρvy, ρvz). It reuses
# the ideal-hydro kernels by permuting to/from the standard (ρ, ρvx, ρvy, ρvz, E)
# order, so the arithmetic is identical — only the storage layout differs.
struct IdealHydroEF <: EnzoNG.EquationSet
    γ::Float64
end
@inline _to_std(U)   = (U[2], U[3], U[4], U[5], U[1])   # EF → standard order
@inline _from_std(U) = (U[5], U[1], U[2], U[3], U[4])   # standard → EF order

EnzoNG.nvars(::IdealHydroEF) = 5
EnzoNG.conserved_names(::IdealHydroEF) =
    (:total_energy, :density, :momentum_x, :momentum_y, :momentum_z)
EnzoNG.density_index(::IdealHydroEF) = 2
EnzoNG.momentum_indices(::IdealHydroEF) = (3, 4, 5)
EnzoNG.energy_index(::IdealHydroEF) = 1
EnzoNG.cons2prim(m::IdealHydroEF, U) = _from_std(EnzoNG.cons2prim(_to_std(U), m.γ))
EnzoNG.prim2cons(m::IdealHydroEF, W) = _from_std(EnzoNG.prim2cons(_to_std(W), m.γ))
EnzoNG.sound_speed(m::IdealHydroEF, W) = EnzoNG.sound_speed(_to_std(W), m.γ)
EnzoNG.riemann_flux(m::IdealHydroEF, WL, WR, axis::Int) =
    _from_std(EnzoNG.hllc_flux(_to_std(WL), _to_std(WR), m.γ, axis))

# Sorted density profile, read via the model's role index (so it's layout-agnostic).
function _dens_profile(sim)
    s = cell_samples(sim); sort!(s; by = t -> t[1][1])
    di = density_index(sim.model)
    return [t[2][di] for t in s]
end

@testset "EquationSet: a reordered variable layout gives identical physics" begin
    prob = sod_problem_defaults(n = 128)
    # the EF model's primitive order is (p, ρ, vx, vy, vz), so its IC is reordered
    ef_init(x, y, z) = x < 0.5 ? (1.0, 1.0, 0.0, 0.0, 0.0) : (0.1, 0.125, 0.0, 0.0, 0.0)
    prob_ef = Problem(; name = "sod-ef", dims = prob.dims, domain = prob.domain, γ = prob.γ,
                      bcs = prob.bcs, init = ef_init, tfinal = prob.tfinal, cfl = prob.cfl)
    a = Simulation(UniformMesh(prob.dims, prob.domain), prob)
    b = Simulation(UniformMesh(prob.dims, prob.domain), prob_ef; model = IdealHydroEF(prob.γ))
    @test a.model isa IdealHydro && nvars(a.model) == 5
    @test b.model isa IdealHydroEF
    @test density_index(b.model) == 2 && energy_index(b.model) == 1   # genuinely reordered
    evolve!(a); evolve!(b)
    @test _dens_profile(a) == _dens_profile(b)        # bit-identical despite the layout
    @test conserved_totals(a).mass ≈ conserved_totals(b).mass rtol = 1e-14
end
