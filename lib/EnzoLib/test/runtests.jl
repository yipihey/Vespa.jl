using Test
using EnzoLib, EnzoFixtures
using EnzoNG, MeshInterface, RefMesh

@testset "EnzoNG ↔ Enzo integration (Phase 1)" begin
    include("test_parity_ppm.jl")        # E1c: legacy ppm_sweep_1d replay ≡ golden fixtures
    include("test_replicate_sod.jl")     # E1b/E1d: legacy-PPM replication + Julia-HLLC parity
end
