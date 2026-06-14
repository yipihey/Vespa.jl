# Calibrate the GPU root Poisson solve against Enzo's certified root FFT, on the
# live CICASS cosmology grid.  Enzo's root gravity solves ∇²φ = 4π·GMF where the
# GravitatingMassField (GMF) already carries the cosmological normalization
# (1/a, density units) from PrepareDensityField.  We:
#   1. session_gravity(0)         → Enzo's correct root φ  (read it)
#   2. session_prepare_density(0) → GMF (the source)
#   3. GPU-solve ∇²φ = 4π·(GMF-mean) and compare to Enzo's φ.
# A match (ratio≈1, corr≈1) means the GMF+4π GPU solve is the correct, cosmology-
# aware replacement for the hand-built G=1 δ in gravity_gpu!.

using EnzoLib, MultiCode, CICASSLib, PoissonKernels, Printf, Statistics
include(joinpath(@__DIR__, "..", "..", "EnzoLib", "examples", "sb_metal_amr.jl"))

const BOX=0.128; const ZSTART=1000.0; const OMEGA_M=0.27
const GD = joinpath(homedir(),"Research","codes","grackle","input","CloudyData_noUVB.h5")

chem = """
RadiativeCooling=1
use_grackle=1
with_radiative_cooling=1
MultiSpecies=3
equilibrium_h2_intermediates=1
neutral_helium=1
equilibrium_deuterium=1
cmb_dissociation=1
cmb_recombination=1
CaseBRecombination=1
grackle_data_file=$(GD)
DualEnergyFormalism=1
"""

res = MultiCode.run_cicass_enzo(; boxlength=BOX, zstart=ZSTART, ngrid=128,
                                omega_m=OMEGA_M, param_extra=chem)
h = res.handle; dims=res.dims; act=res.act; N=res.n
try
    EnzoLib.session_rebuild(h, 0)
    bep = PoissonKernels.backend(BE)
    cube(v) = (m = round(Int, length(v)^(1/3)); reshape(Float64.(v), m, m, m))
    actcube(A) = (off=(size(A,1)-N)÷2; Array(@view A[(off+1):(off+N),(off+1):(off+N),(off+1):(off+N)]))
    # 1) Enzo's certified root solve
    EnzoLib.session_gravity(h, 0)
    g = EnzoLib.problem_grid_index_on_level(h, 0, 0)
    pe_full = cube(EnzoLib.problem_get_potential(h, g))
    @printf("potential field dims=%d^3  active N=%d\n", size(pe_full,1), N)
    phi_enzo = actcube(pe_full)

    # 2) GMF source (deposit only)
    EnzoLib.session_prepare_density(h, 0)
    gmf = cube(EnzoLib.problem_get_gravitating_mass(h, g))
    @printf("GMF dims=%d^3\n", size(gmf,1))
    src = actcube(gmf)
    @printf("GMF active: mean=%.6e  min=%.4e max=%.4e\n", mean(src), minimum(src), maximum(src))
    src .-= mean(src)

    # 3) GPU solve ∇²φ = 4π·(GMF-mean) on the unit box
    for gc in (4π, 1.0, 2π, 4π/OMEGA_M)
        φ = PoissonKernels.device_zeros(bep, T, (N,N,N))
        PoissonKernels.fft_poisson_root_gpu!(φ, dev(bep, src); G=gc, a=1.0, boxsize=1.0)
        pg = Float64.(PoissonKernels.to_host(φ))
        # compare to enzo (both mean-zero)
        pe = phi_enzo .- mean(phi_enzo); pg .-= mean(pg)
        # best-fit scale enzo ≈ s·gpu
        s = sum(pe.*pg)/sum(pg.*pg)
        corr = sum(pe.*pg)/sqrt(sum(pe.^2)*sum(pg.^2))
        @printf("  G=%-8.4f : enzo/gpu scale=%.4f  corr=%.5f  rms(enzo)=%.3e rms(gpu)=%.3e\n",
                gc, s, corr, sqrt(mean(pe.^2)), sqrt(mean(pg.^2)))
    end
finally
    res.free()
end
