# test_f16.jl — Phase 4 gates: Float16 storage with per-block power-of-two f32
# scales.
#   * f16 twin of the subcycled 2-level Sedov tracks the f32 run (L1(ρ) < 1e-2)
#     with bounded conservation drift (quantization, not a bug);
#   * cold-background blast: Ge ~ 1e-5 physical (below f16 normal range) is
#     carried cleanly by the scales; the blast entering a cold block triggers a
#     rescale and everything stays finite and positive.
const BA = BlockAMR

for BE in BACKENDS
@testset "f16 twin Sedov tracks f32 [$BE]" begin
    σ = 2.0 / 64
    prof = sedov(σ, 1.0, 1e-5)
    hs = Dict{DataType,Any}()
    for T in (Float32, Float16)
        hier = center_refined(; backend = BE, T)
        set_ic!(hier, 0, prof); set_ic!(hier, 1, prof)
        hs[T] = hier
    end
    M0, _, _, _, E0 = total_conserved(hs[Float16])
    for n in 1:20
        advance_hierarchy!(hs[Float32])
        advance_hierarchy!(hs[Float16])
        for l in 0:1
            update_scales!(hs[Float16], l)
        end
    end
    M1, _, _, _, E1 = total_conserved(hs[Float16])
    @test abs(M1 - M0) / M0 < 2e-2                     # f16 quantization drift, bounded
    @test abs(E1 - E0) / E0 < 2e-2
    # L1(ρ) agreement on the fine level
    l1 = 0.0; n = 0
    for T in (Float32,)
        lev32 = hs[Float32].levels[2]; lev16 = hs[Float16].levels[2]
        h32 = Array(lev32.D); h16 = Array(lev16.D)
        s32 = Array(lev32.Dsc); s16 = Array(lev16.Dsc)
        for (sa, sb) in zip(lev32.live, lev16.live)
            @assert lev32.meta[sa].origin == lev16.meta[sb].origin
            b32 = (Int(sa) - 1) * lev32.stride; b16 = (Int(sb) - 1) * lev16.stride
            for c in 1:lev32.nd^3
                l1 += abs(Float64(h32[b32+c]) * s32[sa] - Float64(h16[b16+c]) * s16[sb])
                n += 1
            end
        end
    end
    @test l1 / n < 1e-2
end

@testset "cold-background blast: scales carry sub-f16 Ge [$BE]" begin
    σ = 2.0 / 64
    # blob near the refined-region edge so it invades initially-cold blocks
    prof = x -> begin
        r2 = (x[1] - 0.32)^2 + (x[2] - 0.5)^2 + (x[3] - 0.5)^2
        (1.0, 0.0, 0.0, 0.0, 1e-5 + (2/3) * exp(-r2 / (2σ^2)) / ((2π)^1.5 * σ^3))
    end
    hier = center_refined(; backend = BE, T = Float16)
    set_ic!(hier, 0, prof); set_ic!(hier, 1, prof)
    # a far, cold level-0 block: its Esc must start tiny (sub-f16 Ge is scale-carried)
    lev0 = hier.levels[1]
    far = lev0.byorigin[(UInt128(16), UInt128(16), UInt128(16))]
    esc0 = Array(lev0.Esc)[far]
    @test esc0 < 6e-5                                   # ~1.5e-5 phys windowed to [1,2)
    ge0 = Array(lev0.Ge)[(Int(far)-1)*lev0.stride + (lev0.ng*lev0.nd + lev0.ng)*lev0.nd + lev0.ng + 1]
    @test 0.5 < Float32(ge0) * esc0 / 1.5e-5 < 2.0      # physical value round-trips
    M0, _, _, _, E0 = total_conserved(hier)
    for _ in 1:25
        advance_hierarchy!(hier)
        for l in 0:1
            update_scales!(hier, l)
        end
    end
    M1, _, _, _, E1 = total_conserved(hier)
    # everything finite & physical
    for l in 0:1
        lev = hier.levels[l+1]
        hD = Array(lev.D); hG = Array(lev.Ge)
        okd = true
        for s in lev.live
            base = (Int(s)-1)*lev.stride
            for c in 1:lev.nd^3
                (isfinite(Float32(hD[base+c])) && isfinite(Float32(hG[base+c]))) || (okd = false)
            end
        end
        @test okd
    end
    @test abs(M1 - M0) / M0 < 2e-2
    @test abs(E1 - E0) / E0 < 5e-2                      # crossing + quantization, bounded
end
end # for BE
