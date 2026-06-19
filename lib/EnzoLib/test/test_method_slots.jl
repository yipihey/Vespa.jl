# ADR-0002 method-slot registry — Phase B (hydro :julia slot) and Phase C
# (gravity :julia slot) run through the registry on the LIVE Enzo hierarchy.
#
# Phase B: EngineConfig(hydro=:julia) routes the hydro step to a hook that runs
# Vespa's unchanged driver on the live grid (via EnzoBackend); certified vs the
# exact Riemann oracle (Vespa HLLC ≠ Enzo PPM, so physics-level not bit-for-bit).
#
# Guarded on grid_available() (needs the Session bridge library).

using EnzoBackend
import MeshInterface

# Phase B hook: an Vespa-driver hydro slot, (h, level, dt) as the registry wants.
# Lazily builds the EnzoGridMesh + Simulation on the live handle; the EquationSet
# model supplies the conserved-role field map (variable choice is the model's).
function julia_hydro_hook(; γ = 1.4, nghost = 3, domain = ((0.0, 1.0),))
    cache = Ref{Any}(nothing)
    model = IdealHydro(γ)
    return function (h, level, dt)
        if cache[] === nothing
            mesh = EnzoGridMesh(h; grid = 0, nghost = nghost, domain = domain,
                                cons_density = density_index(model),
                                cons_momentum = momentum_indices(model),
                                cons_energy = energy_index(model))
            N = MeshInterface.n_cells(mesh)
            prob = Problem(; name = "slot-hydro", dims = (N,), domain = domain, γ = γ,
                           bcs = Outflow(), tfinal = 1.0, cfl = 0.4,
                           init = (x, y, z) -> (1.0, 0.0, 0.0, 0.0, 1.0))   # overwritten by sync
            cache[] = (mesh, Simulation(mesh, prob; model = model))
        end
        mesh, sim = cache[]
        sync_from_enzo!(sim.sv, mesh)     # Enzo → conserved
        step!(sim, dt)                    # Vespa's unchanged driver, through the seam
        sync_to_enzo!(mesh, sim.sv)       # conserved → Enzo
        return nothing
    end
end

if !EnzoLib.grid_available()
    @info "Session bridge not built — skipping method-slot registry tests"
else
    const SLOT_RUN = normpath(joinpath(@__DIR__, "..", "..", "..", "..", "run"))

    @testset "Phase A: EngineConfig + run_slot wiring" begin
        # all-:enzo from flags == full replication; :off slots are no-ops.
        cfg = EnzoLib.engine_from_flags(; gravity = true, cosmology = true)
        @test cfg.gravity == :enzo
        @test cfg.comoving_expansion == :enzo
        @test cfg.cooling == :off
        @test cfg.hydro == :enzo
        # a :julia slot with no hook is an error (caught before any bridge call)
        bad = EnzoLib.EngineConfig(; hydro = :julia)
        @test_throws ErrorException EnzoLib.run_slot(:hydro, bad, Ptr{Cvoid}(0), 0, 1.0)
    end

    @testset "Phase B: hydro=:julia slot (Vespa driver on live Enzo grid)" begin
        pf = joinpath(SLOT_RUN, "Hydro", "Hydro-1D", "Toro-1-ShockTube", "Toro-1-ShockTube.enzo")
        engine = EnzoLib.EngineConfig(; hydro = :julia,
                                      hooks = Dict{Symbol,Function}(:hydro => julia_hydro_hook()))
        dj = EnzoLib.run_amr(pf; reader = EnzoLib.read_density, engine = engine, regrid = false)
        N = length(dj); dx = 1.0 / N
        x = [(k - 0.5) * dx for k in 1:N]
        WL = (1.0, 0.75, 1.0); WR = (0.125, 0.0, 0.1)        # Toro-1, disc 0.3, t=0.2
        ρx(xi) = exact_riemann_sample(WL, WR, 1.4, (xi - 0.3) / 0.2)[1]
        l1 = sum(abs(dj[i] - ρx(x[i])) for i in 1:N) / N
        @info "Phase B: hydro=:julia via registry vs exact Riemann" cells = N L1 = l1
        @test all(isfinite, dj) && all(>(0), dj)
        @test l1 < 0.03
    end

    # Phase C gravity slot: Vespa's matrix-free CG Poisson solver runs on the LIVE
    # Enzo grid's baryon density. Returns the convergence record + the solved field
    # so the test can certify it. (Full coupling into :enzo hydro needs a
    # set-AccelerationField bridge — ADR-0002's remaining integration step; here we
    # certify the NEW solver port runs correctly on live Enzo memory.)
    function julia_gravity_probe(; G = 1.0, nghost = 3, domain = ((0.0, 1.0),), γ = 1.4)
        cache = Ref{Any}(nothing); rec = Ref{Any}(nothing)
        model = IdealHydro(γ)
        hook = function (h, level, dt)
            if cache[] === nothing
                mesh = EnzoGridMesh(h; grid = 0, nghost = nghost, domain = domain,
                                    cons_density = density_index(model),
                                    cons_momentum = momentum_indices(model),
                                    cons_energy = energy_index(model))
                N = MeshInterface.n_cells(mesh)
                prob = Problem(; name = "slot-grav", dims = (N,), domain = domain, γ = γ,
                               bcs = Periodic(), tfinal = 1.0, cfl = 0.4,
                               init = (x, y, z) -> (1.0, 0.0, 0.0, 0.0, 1.0))
                sim = Simulation(mesh, prob; model = model)
                grav = enable_gravity!(sim; G = G, bcs = Periodic())
                cache[] = (mesh, sim, grav)
            end
            mesh, sim, grav = cache[]
            sync_from_enzo!(sim.sv, mesh)                 # live Enzo ρ → conserved state
            iters, relres = solve_poisson!(sim, grav)     # Vespa CG on live density
            rec[] = (iters = iters, relres = relres, sim = sim, grav = grav, mesh = mesh)
            return nothing
        end
        return hook, rec
    end

    @testset "Phase C: gravity=:julia slot (Vespa Poisson on live Enzo density)" begin
        # ZeldovichPancake = a 1D single-grid self-gravitating baryon field; its IC
        # carries the Zel'dovich density perturbation, so the Poisson RHS is nontrivial.
        pf = abspath(joinpath(SLOT_RUN, "Cosmology", "ZeldovichPancake", "ZeldovichPancake.enzo"))
        hook, rec = julia_gravity_probe(; G = 1.0, domain = ((0.0, 1.0),))
        cfg = EnzoLib.EngineConfig(; gravity = :julia, hooks = Dict{Symbol,Function}(:gravity => hook))
        cd(EnzoLib._workdir(pf)) do
            h = EnzoLib.session_init(pf)
            try
                EnzoLib.run_slot(:gravity, cfg, h, 0, 1.0)   # registry dispatch → the :julia hook
            finally
                EnzoLib.free_problem(h)
            end
        end
        r = rec[]
        ρ = r.sim.sv[density_index(IdealHydro(1.4))]
        φ = MeshInterface.field_view(r.mesh, r.grav.phi, :phi)
        N = MeshInterface.n_cells(r.mesh)
        ρv = [ρ[CartesianIndex(i)] for i in 1:N]
        φv = [φ[CartesianIndex(i)] for i in 1:N]
        # Pearson correlation ρ↔φ: gravity puts potential WELLS (low φ) at overdensities.
        ρc = ρv .- sum(ρv) / N; φc = φv .- sum(φv) / N
        corr = sum(ρc .* φc) / sqrt(sum(abs2, ρc) * sum(abs2, φc) + 1e-300)
        @info "Phase C: Vespa Poisson on live Enzo density" cells = N iters = r.iters relres = r.relres ρ_φ_corr = corr
        @test r.relres < 1e-6                 # CG converged on the live Enzo density
        @test r.iters < 500
        @test maximum(ρv) - minimum(ρv) > 1e-6  # the live density really is perturbed
        @test corr < -0.5                       # potential anti-correlates with density (wells at peaks)
    end
end
