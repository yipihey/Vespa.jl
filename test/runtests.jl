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
    include("test_gravity_poisson.jl") # self-gravity Phase 3a (composite CG Poisson vs analytic)
    include("test_gravity_hydro.jl")   # self-gravity Phase 3b (g source: sign gate, Jeans growth)
    include("test_gravity_amr.jl")     # self-gravity Phase 3c (AMR + subcycle: conservation, regrid)
    include("test_cosmology_units.jl") # cosmology C1 (Enzo-compatible units + Friedmann a(t))
    include("test_cosmology_expansion.jl") # cosmology C2 (Hubble drag: v∝1/a, T∝a⁻²)
    include("test_zeldovich_pancake.jl")   # cosmology C3 (comoving hydro+gravity vs analytic Zel'dovich)
end
