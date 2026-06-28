using Test
using MeshInterface
using RefMesh
using Vespa

@testset "Vespa — hydro on two backends + AMR" begin
    include("test_equationset.jl")    # EquationSet: reordered variable layout ≡ identical physics
    include("test_interface.jl")      # backend-contract conformance (RefMesh)
    include("test_sod.jl")            # Sod vs exact Riemann (RefMesh)
    include("test_layout_swap.jl")    # layout independence (P3)
    include("test_instrument.jl")     # measurement at the seam (P10)
    include("test_hgbackend.jl")      # HGBackend conformance + cross-backend oracle
    include("test_amr.jl")            # hierarchical AMR on HGBackend (conserv. + convergence)
    include("test_sedov.jl")          # 2D Sedov blast: dynamic AMR, symmetry, growth law
    include("test_subcycle.jl")       # AMR time subcycling Phase 1 (per-level dt, no-op invariance)
    include("test_reflux.jl")         # AMR subcycling Phase 2 (coarse–fine flux register conservation)
    include("test_amr_ka_flux.jl")    # P1 re-platform: batched-per-face flux backend ≡ native, round-off AMR
    include("test_boundary_flux.jl")  # ADR-0003 part A: boundary-flux recording (∫F·area dt ≡ Δmass)
    include("test_gravity_poisson.jl") # self-gravity Phase 3a (composite CG Poisson vs analytic)
    include("test_gravity_hydro.jl")   # self-gravity Phase 3b (g source: sign gate, Jeans growth)
    include("test_gravity_amr.jl")     # self-gravity Phase 3c (AMR + subcycle: conservation, regrid)
    include("test_particles.jl")       # P2.1: DM particle self-gravity (CIC deposit + KDK push)
    include("test_cosmology_units.jl") # cosmology C1 (Enzo-compatible units + Friedmann a(t))
    include("test_cosmology_expansion.jl") # cosmology C2 (Hubble drag: v∝1/a, T∝a⁻²)
    include("test_zeldovich_pancake.jl")   # cosmology C3 (comoving hydro+gravity vs analytic Zel'dovich)
    include("test_cosmology_particles.jl") # P3.1 (comoving DM particle KDK: Zel'dovich growth D∝a)
    include("test_compton_drag.jl")        # P3.2 (Compton drag: T_gas → T_cmb at the analytic rate)
    include("test_cosmology_native.jl")    # P4 (native DM structure formation: gravity+expansion+AMR, no Enzo)
    include("test_refine.jl")          # AMR refinement indicators (Jeans-length + DM particle-count)
end
