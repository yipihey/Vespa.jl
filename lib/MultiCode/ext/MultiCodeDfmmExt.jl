# ── the dfmm engine (ADR-0006 Phase 5: library convergence) ───────────────────
#
# dfmm — the dual-frame moment method, the first solver born INSIDE the new
# ecosystem (variational, symplectic, cold-limit collisionless unification,
# built on HierarchicalGrids) — enters the comparison harness as an ENGINE:
# the same Sod Riemann problem the legacy codes run, advanced on dfmm's own
# Lagrangian segments, certified against the same exact-Riemann oracle.
#
# A package EXTENSION (the MultiCode extension-ifying pattern): `using dfmm`
# activates it; MultiCode core carries only the `run_dfmm_sod` stub.  dfmm's
# heavy dependency tree (Makie, Enzyme, …) never burdens the legacy-only user.

module MultiCodeDfmmExt

using MultiCode
using MultiCode: SodSpec
using dfmm

# segment-centered primitives (ρ, u, x) from the HG mesh — the harness's
# profile shape (a compact in-package version of the A1 experiment extractor)
function _dfmm_profile(mesh_HG)
    mesh = dfmm.mesh1d_from_HG(mesh_HG)
    N = dfmm.n_segments(mesh)
    rho = zeros(N); u = zeros(N); xc = zeros(N)
    @inbounds for j in 1:N
        seg = mesh.segments[j]
        rho[j] = dfmm.segment_density(mesh, j)
        jr = j == N ? 1 : j + 1
        u[j] = (seg.state.u + mesh.segments[jr].state.u) / 2
        wrap = (j == N) ? mesh.L_box : 0.0
        xc[j] = (seg.state.x + mesh.segments[jr].state.x + wrap) / 2
    end
    return (rho = rho, u = u, x = xc)
end

function MultiCode.run_dfmm_sod(spec::SodSpec = SodSpec(gamma = 5 / 3, t = 0.2);
                                N::Integer = 200, tau::Real = 1e-3, cfl::Real = 0.3,
                                sigma_x0::Real = 0.02)
    isapprox(spec.gamma, 5 / 3; atol = 1e-12) ||
        error("run_dfmm_sod: the dfmm closure is γ = 5/3 (got $(spec.gamma))")
    spec.x0 == 0.5 || error("run_dfmm_sod: the mirror construction assumes x0 = 0.5")
    Γ = spec.gamma
    # the spec's Riemann problem on [0,1], mirrored to a [0,2] periodic domain
    # (the mirror seam at x = 1 and the wrap at x = 0/2 carry NO jump)
    x0 = ((0:N-1) .+ 0.5) ./ N
    rhoh = ifelse.(x0 .< spec.x0, spec.rhoL, spec.rhoR)
    uh = ifelse.(x0 .< spec.x0, spec.uL, spec.uR)
    Ph = ifelse.(x0 .< spec.x0, spec.pL, spec.pR)
    rho = vcat(rhoh, reverse(rhoh)); u = vcat(uh, -reverse(uh)); P = vcat(Ph, reverse(Ph))
    n = 2N; dx = 2.0 / n
    mesh = dfmm.DetMeshHG_from_arrays(collect((0:n-1) .* dx), u,
                                      fill(Float64(sigma_x0), n), zeros(n),
                                      log.(P ./ rho .^ Γ);
                                      Δm = rho .* dx, Pps = copy(P),
                                      L_box = 2.0, bc = :periodic)
    mass0 = dfmm.total_mass_HG(mesh)
    cs_max = sqrt(Γ * maximum(P) / minimum(rho))
    nsteps = ceil(Int, spec.t / (cfl * dx / cs_max))
    dt = spec.t / nsteps
    t0 = time()
    for _ in 1:nsteps
        dfmm.det_step_HG!(mesh, dt; tau = tau)
    end
    seconds = time() - t0
    full = _dfmm_profile(mesh)
    keep = findall(x -> 0.0 <= x < 1.0, full.x)        # the un-mirrored window
    profile = (x = full.x[keep], rho = full.rho[keep], u = full.u[keep], scatter = 0.0)
    return (profile = profile, t = spec.t, steps = nsteps, seconds = seconds,
            mass = dfmm.total_mass_HG(mesh), mass0 = mass0,
            momentum = dfmm.total_momentum_HG(mesh),
            diag = (N = Int(N), tau = Float64(tau), cfl = Float64(cfl),
                    sigma_x0 = Float64(sigma_x0)),
            free = () -> nothing)
end

end # module
