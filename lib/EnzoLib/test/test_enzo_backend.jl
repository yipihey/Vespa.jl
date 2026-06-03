# E5 — the full seam-level EnzoBackend. EnzoNG's UNCHANGED Simulation/driver
# (`step!` → accumulate_flux! → HLLC/PLM/SSP-RK2) runs through the MeshInterface
# seam on an `EnzoGridMesh` that represents the live Enzo grid, with the conserved
# state synced to/from Enzo's BaryonField around each step. Enzo owns init / BC /
# CFL / advance; EnzoNG's full driver owns the hydro, via the seam.
#
# Guarded on grid_available() (needs the heavy Session bridge library).

using EnzoBackend

const SEAM_PROB = normpath(joinpath(@__DIR__, "..", "..", "..", "..",
                                    "run", "Hydro", "Hydro-1D", "Toro-1-ShockTube",
                                    "Toro-1-ShockTube.enzo"))

# Hydro slot: run EnzoNG's full driver through the seam on the live Enzo grid.
# Lazily builds the EnzoGridMesh + Simulation from the live handle on first call.
function enzo_seam_hydro_slot(; γ = 1.4, nghost = 3)
    cache = Ref{Any}(nothing)
    model = IdealHydro(γ)                 # the equation set drives the field map below
    return function (h, dt)
        if cache[] === nothing
            mesh = EnzoGridMesh(h; nghost = nghost, domain = ((0.0, 1.0),),
                                cons_density = density_index(model),    # roles FROM the model
                                cons_momentum = momentum_indices(model),
                                cons_energy = energy_index(model))
            N = MeshInterface.n_cells(mesh)
            prob = Problem(; name = "enzo-seam", dims = (N,), domain = ((0.0, 1.0),),
                           γ = γ, bcs = Outflow(),
                           init = (x, y, z) -> (1.0, 0.0, 0.0, 0.0, 1.0),  # overwritten by sync
                           tfinal = 1.0, cfl = 0.4)
            cache[] = (mesh, Simulation(mesh, prob; model = model))
        end
        mesh, sim = cache[]
        sync_from_enzo!(sim.sv, mesh)      # Enzo → conserved (model-driven roles)
        step!(sim, dt)                     # EnzoNG's unchanged driver, through the seam
        sync_to_enzo!(mesh, sim.sv)        # conserved → Enzo
        return nothing
    end
end

if !EnzoLib.grid_available()
    @info "Session bridge not built — skipping seam-level EnzoBackend test"
else
    @testset "E5: EnzoNG driver through the seam on a live Enzo grid" begin
        dj = EnzoLib.session_evolve_density(SEAM_PROB, enzo_seam_hydro_slot())
        N = length(dj); dx = 1.0 / N
        x = [(k - 0.5) * dx for k in 1:N]
        WL = (1.0, 0.75, 1.0); WR = (0.125, 0.0, 0.1)        # Toro-1, disc at 0.3, t=0.2
        ρexact(xi) = exact_riemann_sample(WL, WR, 1.4, (xi - 0.3) / 0.2)[1]
        l1 = sum(abs(dj[i] - ρexact(x[i])) for i in 1:N) / N
        @info "EnzoNG-through-seam vs exact Riemann" cells = N L1_density = l1
        @test all(isfinite, dj) && all(>(0), dj)
        @test l1 < 0.03                                       # EnzoNG's driver on Enzo memory matches truth
    end

    # ND single-grid: the seam adapter over a 2D Enzo grid. The column-major
    # flat-index map must be exact (round-trip identity) and EnzoNG's unchanged
    # 2D driver must run on it conservatively.
    @testset "EnzoBackend 2D single-grid (NohProblem2D)" begin
        pf = normpath(joinpath(@__DIR__, "..", "..", "..", "..",
                               "run", "Hydro", "Hydro-2D", "NohProblem2D", "NohProblem2D.enzo"))
        dom2 = ((0.0, 1.0), (0.0, 1.0)); model = IdealHydro(5 / 3)
        cd(EnzoLib._workdir(pf)) do
            h = EnzoLib.session_init(pf); EnzoLib.session_set_boundary(h, 0)
            try
                mesh = EnzoGridMesh(h; grid = 0, domain = dom2,
                                    cons_density = density_index(model),
                                    cons_momentum = momentum_indices(model), cons_energy = energy_index(model))
                @test MeshInterface.rank(mesh) == 2
                @test mesh.active == (100, 100) && mesh.strides == (1, 106)
                prob = Problem(; name = "noh2d", dims = mesh.active, domain = dom2, γ = 5 / 3, bcs = Outflow(),
                               init = (x, y, z) -> (1.0, 0.0, 0.0, 0.0, 1.0), tfinal = 1.0, cfl = 0.3)
                sim = Simulation(mesh, prob; model = model)
                ρ0 = copy(EnzoLib.problem_get_field(h, mesh.di, 0))
                sync_from_enzo!(sim.sv, mesh); sync_to_enzo!(mesh, sim.sv)
                @test EnzoLib.problem_get_field(h, mesh.di, 0) == ρ0          # round-trip identity, bit-for-bit
                di = density_index(model)
                m0 = sum(sim.sv[di][I] for I in CartesianIndices(mesh.active))
                step!(sim, 1e-4)                                              # EnzoNG's 2D driver on the live grid
                d = [sim.sv[di][I] for I in CartesianIndices(mesh.active)]
                m1 = sum(d)
                @test all(isfinite, d) && all(>(0), d)
                @test abs(m1 - m0) / m0 < 1e-3                               # ~conservative (outflow edges)
            finally
                EnzoLib.free_problem(h)
            end
        end
    end
end
