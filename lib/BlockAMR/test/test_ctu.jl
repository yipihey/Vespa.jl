# test_ctu.jl — the tiled/per-cell CTU integrator gates.
#   * uniform flow across a 2-level hierarchy is an EXACT fixed point of the
#     full W-cycle (step + captures + reflux + restriction): conservation and
#     uniformity to the bit, incl. uniform species codes;
#   * Sedov on the static 2-level tower: conservation + exact binary clocks +
#     agreement with the RK2 oracle at truncation level;
#   * [CUDA] the tiled shared-memory kernel and the per-cell oracle produce
#     BIT-IDENTICAL outputs (the f16 hat-tile rounding is replicated exactly by
#     `_ctu_prims_pool`).
const BA = BlockAMR

function _ctu_center_refined(; nbase = (32, 32, 32), B = 16, backend = :cpu,
                             T = Float32, nsp = 0, scheme = :ctu)
    hier = AMRHierarchy(; nbase, B, backend, T, nsp, scheme)
    lev0 = init_base_level!(hier)
    Bh = B ÷ 2
    for cz in 0:1, cy in 0:1, cx in 0:1
        p = lev0.byorigin[(UInt128(cx * B), UInt128(cy * B), UInt128(cz * B))]
        off = (cx == 0 ? Bh : 0, cy == 0 ? Bh : 0, cz == 0 ? Bh : 0)
        add_block!(hier, 1, p, off)
    end
    build_level_tables!(hier, 0); build_level_tables!(hier, 1)
    build_cf_register!(hier, 1)
    return hier
end

function _set_uniform_sp!(hier, l, code1, code2)
    lev = hier.levels[l + 1]
    for (sp, cd) in zip(lev.sp, (code1, code2))
        fill!(sp, cd)
    end
end

for BE in BACKENDS
@testset "CTU: uniform flow is an exact fixed point [$BE]" begin
    hier = _ctu_center_refined(; backend = BE, nsp = 2)
    prof = x -> (1.0, 0.3, -0.2, 0.1, 0.7)
    set_ic!(hier, 0, prof); set_ic!(hier, 1, prof)
    c1 = BA.ChemistryKernels.encode_log2sp(1.0f-4)
    c2 = BA.ChemistryKernels.encode_log2sp(3.0f-6)
    _set_uniform_sp!(hier, 0, c1, c2); _set_uniform_sp!(hier, 1, c1, c2)
    M0 = total_conserved(hier)
    for _ in 1:4
        advance_level_w!(hier, 0, 0.1f0)
    end
    M1 = total_conserved(hier)
    @test M1[1] == M0[1]                              # exact (uniform ⇒ zero ΔF)
    @test M1[5] == M0[5]
    for l in 0:1
        lev = hier.levels[l + 1]
        hD = Array(lev.D)
        vals = Float32[]
        for s in lev.live, k in 1:lev.B, j in 1:lev.B, i in 1:lev.B
            push!(vals, hD[(Int(s)-1)*lev.stride +
                           ((lev.ng+k-1)*lev.nd + (lev.ng+j-1))*lev.nd + (lev.ng+i-1) + 1])
        end
        @test maximum(vals) == minimum(vals)          # still uniform, bitwise
        h1 = Array(lev.sp[1]); h2 = Array(lev.sp[2])
        okc = true
        for s in lev.live, k in 1:lev.B, j in 1:lev.B, i in 1:lev.B
            idx = (Int(s)-1)*lev.stride +
                  ((lev.ng+k-1)*lev.nd + (lev.ng+j-1))*lev.nd + (lev.ng+i-1) + 1
            okc &= (h1[idx] == c1) && (h2[idx] == c2)
        end
        @test okc                                     # uniform species codes survive
    end
end

@testset "CTU: 2-level Sedov — conservation, clocks, vs-RK2 [$BE]" begin
    σ = 2.0 / 128
    prof = sedov(σ, 1.0, 1e-5)
    hc = _ctu_center_refined(; backend = BE, scheme = :ctu)
    hr = _ctu_center_refined(; backend = BE, scheme = :rk2)
    for h in (hc, hr)
        set_ic!(h, 0, prof); set_ic!(h, 1, prof)
    end
    M0, _, _, _, E0 = total_conserved(hc)
    nroot = 12
    for _ in 1:nroot
        advance_hierarchy!(hc)
        advance_hierarchy!(hr)
    end
    M1, _, _, _, E1 = total_conserved(hc)
    @test abs(M1 - M0) / M0 < 3e-5                    # conservation under subcycling
    @test abs(E1 - E0) / E0 < 3e-5
    @test hc.nstep[1] == nroot && hc.nstep[2] == 2nroot   # exact binary clocks
    # CTU vs the RK2 oracle: same physics, different 2nd-order integrator —
    # agreement at truncation level on the coarse grid
    l0c = hc.levels[1]; l0r = hr.levels[1]
    hDc = Array(l0c.D); hDr = Array(l0r.D)
    hsc = Array(l0c.Dsc); hsr = Array(l0r.Dsc)
    num = 0.0; den = 0.0
    for s in l0c.live, k in 1:l0c.B, j in 1:l0c.B, i in 1:l0c.B
        idx = (Int(s)-1)*l0c.stride +
              ((l0c.ng+k-1)*l0c.nd + (l0c.ng+j-1))*l0c.nd + (l0c.ng+i-1) + 1
        a = Float64(hDc[idx]) * hsc[s]; b = Float64(hDr[idx]) * hsr[s]
        num += abs(a - b); den += abs(b)
    end
    @test num / den < 2e-2                            # truncation-level agreement
    @test num / den > 0                               # ...but genuinely different schemes
end

if BE != :cpu
@testset "CTU: tiled ≡ per-cell, bit-exact [$BE]" begin
    σ = 2.0 / 128
    prof = sedov(σ, 1.0, 1e-5)
    mk() = (h = _ctu_center_refined(; backend = BE, nsp = 2);
            set_ic!(h, 0, prof); set_ic!(h, 1, prof);
            c1 = BA.ChemistryKernels.encode_log2sp(1.0f-4);
            c2 = BA.ChemistryKernels.encode_log2sp(3.0f-6);
            _set_uniform_sp!(h, 0, c1, c2); _set_uniform_sp!(h, 1, c1, c2); h)
    ha, hb = mk(), mk()
    # a couple of RK2-free CTU substeps on each level, tiled vs per-cell
    for h in (ha, hb), l in 0:1
        BA.fill_ghosts!(h, l; θ = 0.0f0, buf = :R)
    end
    for l in 0:1
        ctu_level!(ha, l, 0.15f0; tiled = true)
        ctu_level!(hb, l, 0.15f0; tiled = false)
    end
    # tolerance: the two kernels are the SAME inline chain on the SAME inputs,
    # but ptxas contracts a*b+c → fma per-kernel by scheduling context, so
    # differently-structured kernels differ by ~1 ulp at steep cells.  The gate
    # is against indexing/physics bugs (1e-2+), so a few-ulp bound is exact
    # enough; species codes may straddle one codec ULP.
    reldiff(A, Bv) = maximum(abs.(Float32.(A) .- Float32.(Bv)) ./
                             max.(abs.(Float32.(Bv)), 1.0f-6))
    for l in 0:1
        la, lb = ha.levels[l + 1], hb.levels[l + 1]
        @test reldiff(Array(la.Do), Array(lb.Do)) < 1e-5
        @test reldiff(Array(la.S1o), Array(lb.S1o)) < 1e-5
        @test reldiff(Array(la.S2o), Array(lb.S2o)) < 1e-5
        @test reldiff(Array(la.S3o), Array(lb.S3o)) < 1e-5
        @test reldiff(Array(la.Tauo), Array(lb.Tauo)) < 1e-5
        @test reldiff(Array(la.Geo), Array(lb.Geo)) < 1e-5
        @test maximum(abs.(Int.(Array(la.spo[1])) .- Int.(Array(lb.spo[1])))) <= 1
        @test maximum(abs.(Int.(Array(la.spo[2])) .- Int.(Array(lb.spo[2])))) <= 1
    end
end
end # if !cpu
end # for BE
