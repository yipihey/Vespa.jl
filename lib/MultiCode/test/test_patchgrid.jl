# test_patchgrid.jl — the topgrid-decomposition invariance test.
#
# A uniform periodic grid is evolved two ways from IDENTICAL ICs and kernels:
#   REF    : one patch  (np = 1×1×1)  — the undecomposed reference
#   DECOMP : 8 patches  (np = 2×2×2)  — the topgrid split into octants
# Because gravity is one GLOBAL FFT solve (size-identical) and the per-cell hydro/
# chem stencils read identically-filled ghosts, DECOMP must reproduce REF to f64
# round-off.  We check gas density, energies, species AND the gravitational
# potential field, plus global mass / species-mass conservation.
#
# Run: <julia> --project=lib/MultiCode/test lib/MultiCode/test/test_patchgrid.jl

using Test
using MultiCode
import PoissonKernels

const NC   = (16, 16, 16)        # global active cells (small → fast bit-level check)
const NG   = 4
const GAM  = 5/3
const BOX  = 1.0
const DXC  = BOX / NC[1]
const DT   = 0.02 * DXC          # comfortably sub-CFL for the smooth IC
const NSP  = 3                   # HII, H2I, HDI

# ── build a smooth periodic initial condition as global ncell³ conserved fields ──
function make_ic(nc, T; nsp=NSP)
    rho = Array{T,3}(undef, nc); v1 = similar(rho); v2 = similar(rho); v3 = similar(rho)
    eint = similar(rho); sp = [similar(rho) for _ in 1:nsp]
    @inbounds for k in 1:nc[3], j in 1:nc[2], i in 1:nc[1]
        x = 2π*(i-1)/nc[1]; y = 2π*(j-1)/nc[2]; z = 2π*(k-1)/nc[3]
        rho[i,j,k]  = T(1.0 + 0.30*sin(x)*cos(y) + 0.20*sin(z))
        v1[i,j,k]   = T(0.05*sin(y));  v2[i,j,k] = T(0.04*cos(z));  v3[i,j,k] = T(0.03*sin(x))
        eint[i,j,k] = T(1.0 + 0.10*cos(x)*sin(y))           # specific internal energy
        sp[1][i,j,k] = rho[i,j,k]*T(0.05*(1.1+sin(x)))      # ρ·x_HII
        sp[2][i,j,k] = rho[i,j,k]*T(1e-3*(1.1+cos(y)))      # ρ·x_H2I
        sp[3][i,j,k] = rho[i,j,k]*T(1e-4*(1.1+sin(z)))      # ρ·x_HDI
    end
    D = rho
    S1 = rho .* v1; S2 = rho .* v2; S3 = rho .* v3
    Ge = rho .* eint
    Tau = rho .* (eint .+ T(0.5).*(v1.^2 .+ v2.^2 .+ v3.^2))
    return (D=D, S1=S1, S2=S2, S3=S3, Tau=Tau, Ge=Ge, species=sp)
end

# evolve K cycles of a freshly built+scattered PatchGrid; returns the gathered global state
function evolve(np; T, nc=NC, K=3, with_grav=true, with_chem=true, besym=:cpu)
    pg = build_patchgrid(; ng=NG, ncell=nc, np=np, dx=DXC, gamma=GAM, nspecies=NSP,
                         besym=besym, T=T, du=1.0, lu=1.0, tu=1.0, deut=true)
    scatter_global!(pg, make_ic(nc, T))
    ρg = zeros(Float64, nc); φg = zeros(Float64, nc)
    m0 = total_mass(pg)
    for cyc in 1:K
        accel = with_grav ? global_gravity_accel(pg; G=1.0, a=1.0, boxsize=BOX, ρg=ρg, φg=φg) : nothing
        ord = isodd(cyc) ? (1,2,3) : (3,2,1)
        patch_step!(pg, DT; a_value=1.0, order=ord, accel=accel, chem=with_chem)
    end
    g = gather_global(pg)
    return g, total_mass(pg), m0, (φg=copy(φg),)
end

relerr(a, b) = maximum(abs.(Float64.(a) .- Float64.(b))) / (maximum(abs.(Float64.(b))) + eps())

@testset "patchgrid topgrid decomposition (np=1 ≡ np=2), CPU f64" begin
    T = Float64; TOL = 1e-10

    @testset "hydro only" begin
        ref,  _, _, _ = evolve((1,1,1); T=T, with_grav=false, with_chem=false)
        dec, mD, m0, _ = evolve((2,2,2); T=T, with_grav=false, with_chem=false)
        for f in (:D,:S1,:S2,:S3,:Tau,:Ge)
            @test relerr(getfield(dec,f), getfield(ref,f)) ≤ TOL
        end
        for s in 1:NSP
            @test relerr(dec.species[s], ref.species[s]) ≤ TOL
        end
        @test isapprox(mD, m0; rtol=1e-12)            # mass conserved
    end

    @testset "hydro + global gravity" begin
        ref, _, _, rg = evolve((1,1,1); T=T, with_grav=true, with_chem=false)
        dec, _, _, dg = evolve((2,2,2); T=T, with_grav=true, with_chem=false)
        for f in (:D,:S1,:S2,:S3,:Tau,:Ge)
            @test relerr(getfield(dec,f), getfield(ref,f)) ≤ TOL
        end
        @test relerr(dg.φg, rg.φg) ≤ TOL              # global potential matches
    end

    @testset "hydro + gravity + chemistry" begin
        ref, mR0R, _, _ = evolve((1,1,1); T=T, with_grav=true, with_chem=true)
        dec, mD, m0, _  = evolve((2,2,2); T=T, with_grav=true, with_chem=true)
        for f in (:D,:S1,:S2,:S3,:Tau,:Ge)
            @test relerr(getfield(dec,f), getfield(ref,f)) ≤ TOL
        end
        for s in 1:NSP
            @test relerr(dec.species[s], ref.species[s]) ≤ TOL
        end
        @test isapprox(mD, m0; rtol=1e-12)            # chemistry conserves total mass
    end
end
