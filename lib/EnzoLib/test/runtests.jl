using Test
using EnzoLib, EnzoFixtures
using Vespa, MeshInterface, RefMesh
using EnzoBackend

@testset "Vespa ↔ Enzo integration (Phase 1)" begin
    include("test_parity_ppm.jl")        # E1c: legacy ppm_sweep_1d replay ≡ golden fixtures
    include("test_replicate_sod.jl")     # E1b/E1d: legacy-PPM replication + Julia-HLLC parity
    include("test_session_replication.jl") # E2: Julia EvolveLevel ≡ Enzo EvolveHierarchy (if grid lib built)
    include("test_julia_slot_swap.jl")     # E3: Julia HLLC hydro slot on live Enzo grid (if grid lib built)
    include("test_amr_replication.jl")     # E4: Julia recursive EvolveLevel ≈ Enzo AMR (if grid lib built)
    include("test_enzo_suite.jl")          # SUITE: Enzo test problems (Sod/Toro-AMR/BrioWu) ≡ Enzo (if grid lib built)
    include("test_enzo_backend.jl")        # E5: Vespa driver through the seam on a live Enzo grid (if grid lib built)
    include("test_method_slots.jl")        # ADR-0002: method-slot registry, hydro/gravity :julia slots (if grid lib built)
    include("test_set_acceleration.jl")    # set_acceleration bridge: :julia gravity → :enzo hydro coupling primitive
    include("test_julia_reflux.jl")        # ADR-0003 part B: conservative :julia hydro under AMR (SubgridFluxes bridge)
    include("test_julia_reflux_2d.jl")     # ADR-0003 follow-up #2: ND face-plane raster — 2D conservation (if grid lib built)
    include("test_local_ppm_method.jl")    # HydroMethod=10 selects conservative one-ghost local PPM
    include("test_local_ppm_amr.jl")       # standard 1-D/2-D Sod + 3-D Noh AMR analytic tests
    # Phase C subgrid gravity runs in its OWN process: a prior session's leaked
    # Enzo globals empty the GravityTest deposit (GMF = NaN) in a shared host.
    @testset "Phase C: subgrid gravity (isolated process)" begin
        tf = joinpath(@__DIR__, "test_gravity_subgrid_slot.jl")
        @test success(pipeline(`$(Base.julia_cmd()) --project=$(Base.active_project()) $tf`;
                               stdout = stdout, stderr = stderr))
    end
    include("test_rpc_parity.jl")          # ADR-0005: local ≡ remote bridge parity (the differential oracle)
end
