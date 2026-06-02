# Phase C3 — the Zel'dovich pancake: the integration gate for comoving hydro +
# comoving self-gravity + expansion. A sinusoidal perturbation collapses toward a
# caustic; before the caustic the gas follows the analytic Zel'dovich growing-mode
# solution exactly (pressureless limit), so it pins the full coupling — the 1/a
# flux scaling, the 4πG/a overdensity Poisson, and the Hubble drag — at once. A
# wrong gravity factor would show up as the wrong collapse amplitude at a_f.
#   Z1  evolved ρ(x), v(x) match the analytic Zel'dovich profile at a_f (L1, few %)
#   Z2  mass conserved to round-off; the perturbation actually grew (collapse)
#   Z3  cross-backend: RefMesh ≡ HGBackend

using HGBackend

const ZP_zi = 20.0      # initial redshift
const ZP_zc = 1.0       # collapse redshift (caustic)
const ZP_zf = 3.0       # stop pre-caustic: a_f = 21/4 = 5.25 ⇒ A = 0.5
const ZP_kx = 2π

_cells(sim) = (v = Any[]; for_each_cell(sim.backend) do c; push!(v, c); end; v)
_sorted_density(sim) = (s = cell_samples(sim); sort!(s; by = t -> t[1][1]);
                        [t[2][1] for t in s])

function _run_pancake(make_mesh; n = 64)
    prob, cosmo_kwargs = zeldovich_pancake(; n = n, FinalRedshift = ZP_zf,
                                           InitialRedshift = ZP_zi, CollapseRedshift = ZP_zc)
    sim = Simulation(make_mesh(prob.dims, prob.domain), prob)
    enable_cosmology!(sim; cosmo_kwargs..., gravity = true)
    evolve!(sim)
    return sim
end

# Volume-weighted L1 of a per-cell scalar against the analytic Zel'dovich field.
function _pancake_L1(sim, a, sel)
    b = sim.backend
    err = 0.0; ref = 0.0
    for_each_cell(b) do c
        x = cell_center(b, c)[1]; v = cell_volume(b, c)
        ρa, va = zeldovich_state(x, a; zi = ZP_zi, zc = ZP_zc, kx = ZP_kx)
        sim_val = sel(primitive_at(sim, c))
        ana_val = sel((ρa, va))
        err += abs(sim_val - ana_val) * v
        ref += abs(ana_val) * v
    end
    return err / ref
end

@testset "Z1: evolved gas matches the analytic Zel'dovich profile at a_f" begin
    sim = _run_pancake(UniformMesh; n = 64)
    a_f = sim.cosmo.a
    @test isapprox(a_f, 5.25; rtol = 1e-3)                 # z_i=20 → z_f=3
    L1ρ = _pancake_L1(sim, a_f, W -> W[1])                 # density
    L1v = _pancake_L1(sim, a_f, W -> W[2])                 # peculiar velocity
    @info "Zeldovich L1" a_f redshift = redshift(sim.cosmo) L1_density = L1ρ L1_velocity = L1v
    @test L1ρ < 0.05                                       # density within ~5%
    @test L1v < 0.06                                       # velocity within ~6%
end

@testset "Z2: mass conserved, perturbation grew (collapse)" begin
    sim = _run_pancake(UniformMesh; n = 64)
    # rebuild the IC to get the starting totals/peak on the same mesh
    prob, ck = zeldovich_pancake(; n = 64, FinalRedshift = ZP_zf,
                                 InitialRedshift = ZP_zi, CollapseRedshift = ZP_zc)
    sim0 = Simulation(UniformMesh(prob.dims, prob.domain), prob)
    @test isapprox(conserved_totals(sim).mass, conserved_totals(sim0).mass; rtol = 1e-9)
    ρmax = maximum(primitive_at(sim, c)[1] for c in _cells(sim))
    ρmax0 = maximum(primitive_at(sim0, c)[1] for c in _cells(sim0))
    @info "Zeldovich growth" rho_peak_initial = ρmax0 rho_peak_final = ρmax analytic_peak = 2.0
    @test ρmax > 1.5 && ρmax > 1.3 * ρmax0                 # collapsed: peak grew toward 2
end

@testset "Z3: cross-backend RefMesh ≡ HGBackend" begin
    sρ = _sorted_density(_run_pancake(UniformMesh; n = 64))
    hρ = _sorted_density(_run_pancake(HGMesh; n = 64))
    @test length(sρ) == length(hρ)
    @test maximum(abs.(sρ .- hρ)) < 1e-9
end
