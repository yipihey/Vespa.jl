# test_patchgrid_packed.jl — persistent UInt16 packed-species storage (build_patchgrid packed_species=true).
#
# The patch stores the log2-packed mass FRACTION Xᵢ=ρxᵢ/ρ (ChemistryKernels log2_species.jl convention); I/O
# stays in ρ·xᵢ. CPU-only checks (no GPU): (1) scatter→gather ρxᵢ round-trips through the UInt16 quantum;
# (2) the :ppm sweep advects packed species (decode Xᵢ·ρ→ρXᵢ, advect, re-encode per axis) — hydro fields
# match f32 exactly, species within the quantum; (3) the packed chem path (solve_chem_device_u16!) matches
# the f32 chem path (solve_chem_device!) to the packing quantization. The FVGK packed ADVECTION (CUDA-only)
# lives in test_fvgk_solver.jl.
# Run: <julia> --project=lib/MultiCode/test lib/MultiCode/test/test_patchgrid_packed.jl

using Test
using MultiCode
import ChemistryKernels

const NC = (8, 8, 8); const NG = 2; const NSP = 2
const DU, LU, TU = 1.0e-24, 3.0e24, 3.0e15
const AV = 1.0 / (1 + 50.0); const GAM = 5.0/3.0

function make_ic(nc, nsp)
    D=zeros(nc); S1=zeros(nc); S2=zeros(nc); S3=zeros(nc); Tau=zeros(nc); Ge=zeros(nc)
    sp=[zeros(nc) for _ in 1:nsp]
    eint = 1.0e11 / (LU/TU)^2
    for k in 1:nc[3], j in 1:nc[2], i in 1:nc[1]
        ρ = 1.0 + 0.3*sin(2π*i/nc[1])*cos(2π*j/nc[2]); u = 0.1
        D[i,j,k]=ρ; S1[i,j,k]=ρ*u; Tau[i,j,k]=ρ*(eint+0.5*u^2); Ge[i,j,k]=ρ*eint
        x1 = clamp(1e-3*(1+0.5*sin(2π*j/nc[2])), 1e-6, 0.5); x2 = 1e-6
        sp[1][i,j,k]=ρ*x1; sp[2][i,j,k]=ρ*x2          # store ρ·xᵢ
    end
    return (D=D, S1=S1, S2=S2, S3=S3, Tau=Tau, Ge=Ge, species=sp)
end
mkpg(packed) = build_patchgrid(; ng=NG, ncell=NC, np=(1,1,1), dx=1.0, gamma=GAM, nspecies=NSP,
                               besym=:cpu, T=Float64, du=DU, lu=LU, tu=TU, deut=false, packed_species=packed)

@testset "packed UInt16 species storage (CPU)" begin
    ic = make_ic(NC, NSP)

    @testset "storage type + ρxᵢ round-trip through the UInt16 quantum" begin
        pg = mkpg(true)
        @test pg.packed
        @test eltype(pg.patches[1].species[1]) == UInt16
        scatter_global!(pg, ic); g = gather_global(pg)
        for q in 1:NSP
            xq  = ic.species[q] ./ ic.D                       # the fraction that gets quantized
            xqg = g.species[q]  ./ g.D
            @test maximum(abs.(xqg .- xq) ./ (xq .+ eps())) < 3e-3    # log2-UInt16 quantum ≈ 0.12%
            @test all(isfinite, g.species[q])
        end
    end

    @testset "packed PPM hydro advection ≡ f32 (within UInt16 quantization)" begin
        pgp = mkpg(true);  scatter_global!(pgp, ic)
        pgf = mkpg(false); scatter_global!(pgf, ic)
        dt = 1e-3
        patch_step!(pgp, dt; a_value=AV, solver=:ppm, do_hydro=true, do_chem=false, chem=true)
        patch_step!(pgf, dt; a_value=AV, solver=:ppm, do_hydro=true, do_chem=false, chem=true)
        gp = gather_global(pgp); gf = gather_global(pgf)
        # colours are PASSIVE ⇒ the hydro fields must be identical regardless of packing
        @test maximum(abs.(gp.D   .- gf.D))   / maximum(abs.(gf.D))   < 1e-10
        @test maximum(abs.(gp.Tau .- gf.Tau)) / maximum(abs.(gf.Tau)) < 1e-10
        # the advected species agree within the UInt16 quantum (decode·ρ → advect → encode/ρ per axis)
        for q in 1:NSP
            @test all(isfinite, gp.species[q])
            xp = gp.species[q] ./ gp.D; xf = gf.species[q] ./ gf.D
            @test maximum(abs.(xp .- xf) ./ (xf .+ eps())) < 2e-2
        end
    end

    @testset "packed chem ≡ f32 chem (within UInt16 quantization)" begin
        pgp = mkpg(true);  scatter_global!(pgp, ic)
        pgf = mkpg(false); scatter_global!(pgf, ic)
        dt = 1e-2
        patch_step!(pgp, dt; a_value=AV, do_hydro=false, do_chem=true, chem=true)
        patch_step!(pgf, dt; a_value=AV, do_hydro=false, do_chem=true, chem=true)
        gp = gather_global(pgp); gf = gather_global(pgf)
        for q in 1:NSP
            @test all(isfinite, gp.species[q])
            @test maximum(abs.(gp.species[q] .- gf.species[q]) ./ (abs.(gf.species[q]) .+ eps())) < 1.5e-2
        end
        @test maximum(abs.(gp.Ge .- gf.Ge) ./ (abs.(gf.Ge) .+ eps())) < 1e-3   # cooling matches
    end
end
