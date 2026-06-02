# Phase 3a — the composite Poisson solver, with NO hydro coupling. The discrete
# FV Laplacian is the structural twin of the hydro flux divergence; these oracles
# pin its correctness before gravity is coupled to the gas:
#   O1  Laplacian symmetry + null space (cheapest gate, no solve)
#   O2  periodic Fourier-mode Poisson vs the analytic φ (2nd-order convergence)
#   O3  cross-backend φ agreement (RefMesh ≡ HGBackend)
# Operator: A = −∇²·V (SPD); solve ∇²φ = 4πG ρ via CG. G is code units.

using HGBackend

# Build a Simulation whose density field is ρ(x) (uniform p, zero v) so the
# gravity solver has a controlled RHS. Periodic gas BCs (φ gets its own below).
function _density_sim(make_mesh, dims, dom, ρfun; γ = 5 / 3)
    init(x, y, z) = (ρfun(x, y, z), 0.0, 0.0, 0.0, 1.0)
    prob = Problem(; name = "grav_poisson", dims = dims, domain = dom, γ = γ,
                   bcs = Periodic(), init = init, tfinal = 1.0, cfl = 0.4)
    return Simulation(make_mesh(dims, dom), prob)
end

@testset "O1: Laplacian is symmetric and annihilates constants" begin
    # On a small uniform periodic mesh, A=−∇²·V must satisfy ⟨x,Ay⟩=⟨Ax,y⟩ and
    # A·1 = 0. This localizes the per-face stencil math (sign/area/distance)
    # before any solve. (No volume weight in the dot — A carries the geometry.)
    dims = (16,); dom = ((0.0, 1.0),)
    sim = _density_sim(UniformMesh, dims, dom, (x, y, z) -> 1.0)
    grav = enable_gravity!(sim; G = 1.0, bcs = Periodic())
    b = sim.backend

    # two arbitrary fields x, y in the φ store + scratch
    setf(v, f) = for_each_cell(b) do c; v[c] = f(cell_center(b, c)[1]); end
    setf(grav.phiv, x -> sin(2π * x) + 0.3cos(6π * x))
    setf(grav.rv,   x -> cos(2π * x) - 0.7sin(4π * x))
    Ax = grav.Apv; Ay = grav.bv
    apply_laplacian!(sim, grav, grav.phiv, Ax)
    apply_laplacian!(sim, grav, grav.rv, Ay)
    xAy = EnzoNG.dot_cells(sim, grav.phiv, Ay)
    yAx = EnzoNG.dot_cells(sim, grav.rv, Ax)
    @test isapprox(xAy, yAx; rtol = 1e-12, atol = 1e-12)   # symmetry

    # A·1 = 0 (constant null space of the periodic Laplacian)
    for_each_cell(b) do c; grav.pv[c] = 1.0; end
    apply_laplacian!(sim, grav, grav.pv, Ax)
    maxabs = 0.0; for_each_cell(b) do c; maxabs = max(maxabs, abs(Ax[c])); end
    @test maxabs < 1e-12

    # negative-semidefinite: ⟨x, Ax⟩ ≥ 0 for A=−∇²·V (energy of the gradient)
    setf(grav.phiv, x -> sin(2π * x))
    apply_laplacian!(sim, grav, grav.phiv, Ax)
    @test EnzoNG.dot_cells(sim, grav.phiv, Ax) > 0
end

# Analytic periodic Poisson: ρ = ρ̄ + A cos(kx), k = 2πm/L on [0,L].
# ∇²φ = 4πG ρ with zero-mean RHS ⇒ φ = −(4πG A / k²) cos(kx)  (zero mean).
function _fourier_l1(make_mesh, n; m = 1, A = 0.1, ρ̄ = 1.0, G = 1.0, L = 1.0)
    k = 2π * m / L
    sim = _density_sim(make_mesh, (n,), ((0.0, L),), (x, y, z) -> ρ̄ + A * cos(k * x))
    grav = enable_gravity!(sim; G = G, bcs = Periodic(), tol = 1e-12, maxiter = 5000)
    its, rr = solve_poisson!(sim, grav)
    b = sim.backend
    coef = -4π * G * A / k^2
    err = 0.0; vol = 0.0
    for_each_cell(b) do c
        x = cell_center(b, c)[1]; v = cell_volume(b, c)
        φe = coef * cos(k * x)                       # analytic (zero-mean)
        err += abs(grav.phiv[c] - φe) * v; vol += v
    end
    return err / vol, its, rr
end

@testset "O2: periodic Fourier-mode Poisson → analytic φ, 2nd-order" begin
    e64, it64, rr64 = _fourier_l1(UniformMesh, 64)
    e128, _, _      = _fourier_l1(UniformMesh, 128)
    @info "Poisson Fourier L1" n64 = e64 n128 = e128 ratio = e64 / e128 iters = it64 relres = rr64
    @test rr64 < 1e-10                       # CG converged
    @test e64 < 5e-3                         # accurate at N=64
    @test e64 / e128 > 3.5                   # ~4× drop on doubling N (2nd order)
end

@testset "O3: cross-backend φ (RefMesh ≡ HGBackend)" begin
    # Same Fourier problem on both substrates: φ must agree to tight tolerance.
    solve_on(make_mesh) = begin
        sim = _density_sim(make_mesh, (64,), ((0.0, 1.0),),
                           (x, y, z) -> 1.0 + 0.1 * cos(2π * x))
        grav = enable_gravity!(sim; G = 1.0, bcs = Periodic(), tol = 1e-12, maxiter = 5000)
        solve_poisson!(sim, grav)
        [grav.phiv[c] for c in (begin v = Any[]; for_each_cell(sim.backend) do cc; push!(v, cc); end; v end)]
    end
    φref = solve_on(UniformMesh)
    φhg  = solve_on(HGMesh)
    @test length(φref) == length(φhg)
    @test maximum(abs.(φref .- φhg)) < 1e-9
end

@testset "O2b: 2D periodic Poisson → analytic φ" begin
    # ρ = 1 + A cos(kx)cos(ky); ∇²φ = 4πGρ ⇒ φ = −(4πG A/(2k²)) cos(kx)cos(ky).
    n = 64; L = 1.0; A = 0.1; G = 1.0; k = 2π / L
    sim = _density_sim(HGMesh, (n, n), ((0.0, L), (0.0, L)),
                       (x, y, z) -> 1.0 + A * cos(k * x) * cos(k * y))
    grav = enable_gravity!(sim; G = G, bcs = Periodic(), tol = 1e-11, maxiter = 5000)
    _, rr = solve_poisson!(sim, grav)
    b = sim.backend
    coef = -4π * G * A / (2k^2)
    err = 0.0; vol = 0.0
    for_each_cell(b) do c
        ctr = cell_center(b, c); v = cell_volume(b, c)
        φe = coef * cos(k * ctr[1]) * cos(k * ctr[2])
        err += abs(grav.phiv[c] - φe) * v; vol += v
    end
    @info "Poisson 2D L1" l1 = err / vol relres = rr
    @test rr < 1e-9
    @test err / vol < 1e-2
end
