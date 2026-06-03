using Test
using EnzoLib, EnzoFixtures
using EnzoNG, MeshInterface, RefMesh

@testset "EnzoNG ↔ Enzo integration (Phase 1)" begin
    include("test_parity_ppm.jl")        # E1c: legacy ppm_sweep_1d replay ≡ golden fixtures
    include("test_replicate_sod.jl")     # E1b/E1d: legacy-PPM replication + Julia-HLLC parity
    include("test_session_replication.jl") # E2: Julia EvolveLevel ≡ Enzo EvolveHierarchy (if grid lib built)
    include("test_julia_slot_swap.jl")     # E3: Julia HLLC hydro slot on live Enzo grid (if grid lib built)
    include("test_amr_replication.jl")     # E4: Julia recursive EvolveLevel ≈ Enzo AMR (if grid lib built)
    include("test_enzo_suite.jl")          # SUITE: Enzo test problems (Sod/Toro-AMR/BrioWu) ≡ Enzo (if grid lib built)
    include("test_enzo_backend.jl")        # E5: EnzoNG driver through the seam on a live Enzo grid (if grid lib built)
end
