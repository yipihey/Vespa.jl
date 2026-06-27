# test_fvgk_solver.jl — the FiniteVolumeGodunovKA hydro solver in PatchGrid (solver=:fvgk).
#
# FVGK is an unsplit single-grid Godunov solver; the adapter (MultiCodeFVGKExt) assembles the
# patch interiors into one global periodic grid, steps the fast transpiled CTU kernel once, and
# disperses back — decomposition-invariant for any np.
#
# Checks (CUDA / Float32):
#   1. ADAPTER bit-identity: solver=:fvgk (np=1) ≡ a STANDALONE Grid3DCuMarch on the same IC
#      (max|Δ| = 0) — proves the gather/scatter repack is exact.
#   2. DECOMPOSITION INVARIANCE: np=2×2×2 ≡ np=1×1×1 bit-identically (the multi-patch criterion).
#   3. DUAL ENERGY: Ge re-derived as Tau − ½(S1²+S2²+S3²)/D, positive.
#   4. GRAVITY interop: np=2 ≡ np=1 with the global-FFT gravity kick (KDK around the FVGK hydro).
#   5. CONSERVATION + physics sanity vs PPMKernels.
#
# Needs a CUDA GPU + nvcc; skips cleanly otherwise.
# Run: <julia> --project=lib/MultiCode/test lib/MultiCode/test/test_fvgk_solver.jl

using Test
using MultiCode
import PPMKernels
import PoissonKernels

const _HAVE_CUDA = try
    using CUDA; CUDA.functional() && PPMKernels.has_backend(:cuda)
catch
    false
end

if !_HAVE_CUDA
    @info "test_fvgk_solver: CUDA not functional — skipping FVGK solver tests"
else
    using FiniteVolumeGodunovKA          # triggers MultiCodeFVGKExt
    const FV = FiniteVolumeGodunovKA

    const NC  = (32, 32, 32)
    const NG  = 4
    const GAM = 5f0/3f0
    const BOX = 1f0
    const DXC = BOX / NC[1]
    const DT  = 0.02f0 * DXC              # sub-CFL → one CTU substep per call
    const K   = 3

    # smooth periodic IC as global conserved fields (Float32, no species)
    function make_ic(nc)
        rho = Array{Float32,3}(undef, nc); v1 = similar(rho); v2 = similar(rho); v3 = similar(rho); eint = similar(rho)
        @inbounds for k in 1:nc[3], j in 1:nc[2], i in 1:nc[1]
            x = 2f0π*(i-1)/nc[1]; y = 2f0π*(j-1)/nc[2]; z = 2f0π*(k-1)/nc[3]
            rho[i,j,k]  = 1f0 + 0.30f0*sin(x)*cos(y) + 0.20f0*sin(z)
            v1[i,j,k]   = 0.05f0*sin(y); v2[i,j,k] = 0.04f0*cos(z); v3[i,j,k] = 0.03f0*sin(x)
            eint[i,j,k] = 1f0 + 0.10f0*cos(x)*sin(y)
        end
        D = rho; S1 = rho.*v1; S2 = rho.*v2; S3 = rho.*v3
        Tau = rho .* (eint .+ 0.5f0 .* (v1.^2 .+ v2.^2 .+ v3.^2)); Ge = rho .* eint
        return (D=D, S1=S1, S2=S2, S3=S3, Tau=Tau, Ge=Ge, species=Vector{Array{Float32,3}}())
    end

    "K patch_step!s at tiling `np` with `solver`; returns gathered conserved fields + (m0, m1)."
    function run_patch(np, solver; with_grav=false)
        pg = build_patchgrid(; ng=NG, ncell=NC, np=np, dx=DXC, gamma=GAM, nspecies=0,
                             besym=:cuda, T=Float32, deut=false)
        scatter_global!(pg, make_ic(NC))
        ρg = zeros(Float64, NC); φg = zeros(Float64, NC); m0 = total_mass(pg)
        for cyc in 1:K
            accel = with_grav ? global_gravity_accel(pg; G=1.0, a=1.0, boxsize=Float64(BOX), ρg=ρg, φg=φg) : nothing
            patch_step!(pg, DT; a_value=1.0, order=(1,2,3), accel=accel, chem=false, solver=solver)
        end
        return gather_global(pg), m0, total_mass(pg)
    end

    "standalone FVGK Grid3DCuMarch on the same IC; conserved (D,S1,S2,S3,Tau) host arrays."
    function run_standalone()
        ic = make_ic(NC)
        U0 = [ (ic.D[i,j,k], ic.S1[i,j,k], ic.S2[i,j,k], ic.S3[i,j,k], ic.Tau[i,j,k])
               for i in 1:NC[1], j in 1:NC[2], k in 1:NC[3] ]
        g = FV.Grid3DCuMarch(FV.Euler(γ=GAM), U0; dx=Float32(DXC))
        for _ in 1:K; FV.run_ctus!(g, DT, 1); end
        Rh = Array(g.R); VOL = prod(NC); blk(c) = reshape(Rh[(c-1)*VOL+1 : c*VOL], NC...)
        return (D=blk(1), S1=blk(2), S2=blk(3), S3=blk(4), Tau=blk(5))
    end

    relerr(a, b) = maximum(abs.(Float64.(a) .- Float64.(b))) / (maximum(abs.(Float64.(b))) + eps())

    @testset "FVGK solver in PatchGrid (CUDA f32)" begin
        fv1, m0, m1 = run_patch((1,1,1), :fvgk)

        @testset "adapter ≡ standalone Grid3DCuMarch (np=1, bit-identical)" begin
            st = run_standalone()
            for f in (:D, :S1, :S2, :S3, :Tau)
                @test maximum(abs.(getfield(fv1, f) .- getfield(st, f))) == 0f0
            end
        end

        @testset "decomposition invariance (np=2×2×2 ≡ np=1×1×1, bit-identical)" begin
            dec, mD, mD0 = run_patch((2,2,2), :fvgk)
            for f in (:D, :S1, :S2, :S3, :Tau, :Ge)
                @test maximum(abs.(getfield(dec, f) .- getfield(fv1, f))) == 0f0
            end
            @test isapprox(mD, mD0; rtol=1e-5)
        end

        @testset "dual energy Ge ≡ Tau − kinetic, positive" begin
            ge_expect = fv1.Tau .- 0.5f0 .* (fv1.S1.^2 .+ fv1.S2.^2 .+ fv1.S3.^2) ./ fv1.D
            @test relerr(fv1.Ge, ge_expect) < 1e-4
            @test all(>(0f0), fv1.Ge)
        end

        @testset "conservation + finiteness" begin
            @test isapprox(m1, m0; rtol=1e-5)
            @test all(isfinite, fv1.D) && all(>(0f0), fv1.D)
            @test all(isfinite, fv1.Tau) && all(>(0f0), fv1.Tau)
        end

        @testset "decomposition invariance + global gravity (KDK)" begin
            rg, _, _ = run_patch((1,1,1), :fvgk; with_grav=true)
            dg, _, _ = run_patch((2,2,2), :fvgk; with_grav=true)
            for f in (:D, :S1, :S2, :S3, :Tau)
                @test all(isfinite, getfield(rg, f))
                @test relerr(getfield(dg, f), getfield(rg, f)) ≤ 1e-5
            end
        end

        @testset "physics sanity vs PPMKernels" begin
            pp, _, _ = run_patch((1,1,1), :ppm)
            @test all(isfinite, pp.D) && all(>(0f0), pp.D)
            @test relerr(fv1.D, pp.D)   < 0.05
            @test relerr(fv1.Tau, pp.Tau) < 0.05
        end
    end

    # ── passive species (colours) carried as EulerColors extra conserved vars ──────────
    # species ρ·xᵢ ride the hydro mass flux: x₁ uniform (CMA: stays uniform), x₂ a blob (advects).
    function make_ic_sp(nc, nsp)
        b = make_ic(nc); ρ = b.D; sp = Vector{Array{Float32,3}}()
        for q in 1:nsp
            x = similar(ρ)
            @inbounds for k in 1:nc[3], j in 1:nc[2], i in 1:nc[1]
                frac = q == 1 ? 0.40f0 :
                       0.10f0 + 0.05f0*q + 0.20f0*exp(-(Float32(i-nc[1]÷2)^2 + Float32(j-nc[2]÷2)^2)/8f0)
                x[i,j,k] = ρ[i,j,k] * frac      # store ρ·xᵢ
            end
            push!(sp, x)
        end
        return (D=b.D, S1=b.S1, S2=b.S2, S3=b.S3, Tau=b.Tau, Ge=b.Ge, species=sp)
    end
    function run_patch_sp(np, nsp)
        pg = build_patchgrid(; ng=NG, ncell=NC, np=np, dx=DXC, gamma=GAM, nspecies=nsp,
                             besym=:cuda, T=Float32, deut=false)
        ic = make_ic_sp(NC, nsp); scatter_global!(pg, ic)
        for _ in 1:K
            patch_step!(pg, DT; a_value=1.0, order=(1,2,3), accel=nothing, chem=false, solver=:fvgk)
        end
        return gather_global(pg), ic
    end

    @testset "FVGK passive species (EulerColors colours)" begin
        nsp = 2
        g1, ic = run_patch_sp((1,1,1), nsp)
        for q in 1:nsp
            @test all(isfinite, g1.species[q])
            @test isapprox(sum(Float64.(g1.species[q])), sum(Float64.(ic.species[q])); rtol=1e-4)  # conservation
        end
        @test g1.species[2] != ic.species[2]                                    # x₂ actually advected
        @test maximum(abs.(g1.species[1] ./ g1.D .- 0.40f0)) < 1f-4             # x₁ uniform stays uniform (CMA)
        # decomposition invariance: np=2×2×2 ≡ np=1×1×1 bit-identically, hydro AND species
        g2, _ = run_patch_sp((2,2,2), nsp)
        for f in (:D, :S1, :S2, :S3, :Tau)
            @test maximum(abs.(getfield(g2, f) .- getfield(g1, f))) == 0f0
        end
        for q in 1:nsp
            @test maximum(abs.(g2.species[q] .- g1.species[q])) == 0f0
        end
    end

    # ── persistent UInt16 packed-species storage (build_patchgrid packed_species=true) ──
    function run_patch_sp_packed(np, nsp)
        pg = build_patchgrid(; ng=NG, ncell=NC, np=np, dx=DXC, gamma=GAM, nspecies=nsp,
                             besym=:cuda, T=Float32, deut=false, packed_species=true)
        ic = make_ic_sp(NC, nsp); scatter_global!(pg, ic)
        for _ in 1:K
            patch_step!(pg, DT; a_value=1.0, order=(1,2,3), accel=nothing, chem=false, solver=:fvgk)
        end
        return gather_global(pg), ic
    end

    @testset "FVGK packed UInt16 species advection ≡ f32 (within quantum)" begin
        nsp = 2
        gp, ic = run_patch_sp_packed((1,1,1), nsp)
        gf, _  = run_patch_sp((1,1,1), nsp)                     # f32-storage reference, same IC
        for q in 1:nsp
            @test all(isfinite, gp.species[q])
            @test isapprox(sum(Float64.(gp.species[q])), sum(Float64.(ic.species[q])); rtol=3e-3)   # conservation
            xp = gp.species[q] ./ gp.D; xf = gf.species[q] ./ gf.D     # compare FRACTIONS
            @test maximum(abs.(xp .- xf) ./ (xf .+ eps())) < 2f-2      # per-step repack quantum, K steps
        end
        @test maximum(abs.(gp.species[1] ./ gp.D .- 0.40f0)) < 2f-3    # x₁ uniform stays uniform (CMA)
        # packed storage is decomposition-invariant too (np=2 ≡ np=1, bit-identical UInt16)
        gp2, _ = run_patch_sp_packed((2,2,2), nsp)
        for q in 1:nsp
            @test maximum(abs.(gp2.species[q] .- gp.species[q])) == 0f0
        end
    end
end
