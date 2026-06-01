using Test
using MeshInterface
using RefMesh
using EnzoNG

@testset "EnzoNG — hydro on two backends + AMR" begin
    include("test_interface.jl")      # backend-contract conformance (RefMesh)
    include("test_sod.jl")            # Sod vs exact Riemann (RefMesh)
    include("test_layout_swap.jl")    # layout independence (P3)
    include("test_instrument.jl")     # measurement at the seam (P10)
    include("test_hgbackend.jl")      # HGBackend conformance + cross-backend oracle
    include("test_amr.jl")            # hierarchical AMR on HGBackend (conserv. + convergence)
    include("test_sedov.jl")          # 2D Sedov blast: dynamic AMR, symmetry, growth law
    include("test_subcycle.jl")       # AMR time subcycling Phase 1 (per-level dt, no-op invariance)
    include("test_reflux.jl")         # AMR subcycling Phase 2 (coarse–fine flux register conservation)
end
