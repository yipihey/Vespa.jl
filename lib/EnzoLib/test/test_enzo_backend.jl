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
    return function (h, dt)
        if cache[] === nothing
            mesh = EnzoGridMesh(h; γ = γ, nghost = nghost, domain = ((0.0, 1.0),))
            N = MeshInterface.n_cells(mesh)
            prob = Problem(; name = "enzo-seam", dims = (N,), domain = ((0.0, 1.0),),
                           γ = γ, bcs = Outflow(),
                           init = (x, y, z) -> (1.0, 0.0, 0.0, 0.0, 1.0),  # overwritten by sync
                           tfinal = 1.0, cfl = 0.4)
            cache[] = (mesh, Simulation(mesh, prob))
        end
        mesh, sim = cache[]
        sync_from_enzo!(sim.sv, mesh)      # Enzo → conserved
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
end
