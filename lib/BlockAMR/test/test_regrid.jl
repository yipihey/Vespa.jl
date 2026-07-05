# test_regrid.jl — Phase 2 gates: dynamic regridding under a 45°-advected
# density pulse crossing block boundaries and the periodic wrap.
#   * composite conservation holds EXACTLY through every regrid generation
#     (restriction before de-refine + PC prolongation on refine are conservative);
#   * refinement tracks the pulse (the densest cell is always on the fine level);
#   * ≥ 3 create/destroy generations + freelist churn + nesting after every rebuild.
const BA = BlockAMR

for BE in BACKENDS
@testset "dynamic regrid tracks advected pulse [$BE]" begin
    hier = AMRHierarchy(; nbase = (32, 32, 32), B = 16, backend = BE, cap0 = 2)
    init_base_level!(hier)
    pol = BlockRefinementPolicy(; dthresh = 1.05, nbuf = 2, every = 4, lmax = 1)
    # Gaussian pulse ρ = 1 + 0.5·exp(−r²/2σ²) advecting along (1,1,0)/√2·0.5
    σ = 0.05; vx = 0.5; vy = 0.5
    x0 = (0.30, 0.30, 0.50)
    prof(t) = x -> begin
        r2 = sum(d -> (mod(x[d] - (x0[d] + (d == 1 ? vx : d == 2 ? vy : 0.0) * t) + 0.5, 1.0)
                       - 0.5)^2, 1:3)
        (1.0 + 0.5 * exp(-r2 / (2σ^2)), vx, vy, 0.0, 1.0)
    end
    set_ic!(hier, 0, prof(0.0))
    build_level_tables!(hier, 0)
    regrid!(hier, pol)                                # initial refinement over the pulse
    @test !isempty(hier.levels[2].live)
    @test check_nesting(hier)
    # re-init BOTH levels from the analytic profile (fine level gets true IC)
    set_ic!(hier, 0, prof(0.0)); set_ic!(hier, 1, prof(0.0))
    M0, _, _, _, E0 = total_conserved(hier)
    t = 0.0; gens = Set{BA.Origin}(); freed = 0
    prevorgs = Set(hier.levels[2].meta[s].origin for s in hier.levels[2].live)
    for n in 1:120
        dt = compute_dt(hier)
        hierarchy_rk2_step!(hier, dt); t += dt
        if n % pol.every == 0
            Mpre, _, _, _, Epre = total_conserved(hier)
            regrid!(hier, pol)
            @test check_nesting(hier)
            Mpost, _, _, _, Epost = total_conserved(hier)
            # regrid itself is exactly conservative (restrict-then-free + PC prolong)
            @test abs(Mpost - Mpre) / Mpre < 1e-6
            @test abs(Epost - Epre) / abs(Epre) < 1e-6
            orgs = Set(hier.levels[2].meta[s].origin for s in hier.levels[2].live)
            freed += length(setdiff(prevorgs, orgs))
            union!(gens, setdiff(orgs, prevorgs))
            prevorgs = orgs
        end
    end
    M1, _, _, _, E1 = total_conserved(hier)
    @test abs(M1 - M0) / M0 < 3e-5                    # whole run incl. all regrids
    @test abs(E1 - E0) / abs(E0) < 3e-5
    @test length(gens) >= 3                           # ≥3 creation generations
    @test freed >= 3                                  # ...and destructions
    @test !isempty(hier.levels[2].freelist) || freed > 0   # slot recycling exercised
    # the pulse peak must live on the fine level and sit near the advected center
    lev1 = hier.levels[2]; hD = Array(lev1.D)
    dmax = 0.0f0; gbest = (0, 0, 0)
    for s in lev1.live
        m = lev1.meta[s]; base = (Int(s) - 1) * lev1.stride
        for k in 1:lev1.B, j in 1:lev1.B, i in 1:lev1.B
            v = hD[base + ((lev1.ng+k-1)*lev1.nd + (lev1.ng+j-1))*lev1.nd + (lev1.ng+i-1) + 1]
            v > dmax && (dmax = v;
                gbest = (Int(m.origin[1]) + i - 1, Int(m.origin[2]) + j - 1, Int(m.origin[3]) + k - 1))
        end
    end
    @test dmax > 1.2                                  # pulse survived (diffused but present)
    xc = ntuple(d -> mod(x0[d] + (d == 1 ? vx : d == 2 ? vy : 0.0) * t, 1.0), 3)
    dxf = BA.level_dx(hier, 1)
    dist = sqrt(sum(d -> (mod((gbest[d] + 0.5) * dxf - xc[d] + 0.5, 1.0) - 0.5)^2, 1:3))
    @test dist < 4σ                                   # tracked through wrap + boundaries
end

@testset "slot compaction: pure data movement + Morton order [$BE]" begin
    # twin hierarchies through the same advected-pulse regrid churn — the ONLY
    # difference is the Morton compaction pass, so every field must agree
    # bit-exactly per block origin (compaction is data movement, nothing else).
    σ = 0.05; vx = 0.5; vy = 0.5; x0 = (0.30, 0.30, 0.50)
    prof = x -> begin
        r2 = sum(d -> (mod(x[d] - x0[d] + 0.5, 1.0) - 0.5)^2, 1:3)
        (1.0 + 0.5 * exp(-r2 / (2σ^2)), vx, vy, 0.0, 1.0)
    end
    pol = BlockRefinementPolicy(; dthresh = 1.05, nbuf = 2, every = 4, lmax = 1)
    mk() = (h = AMRHierarchy(; nbase = (32, 32, 32), B = 16, backend = BE, cap0 = 2);
            init_base_level!(h); set_ic!(h, 0, prof); build_level_tables!(h, 0); h)
    ha, hb = mk(), mk()
    regrid!(ha, pol; compact = true); regrid!(hb, pol; compact = false)
    for h in (ha, hb)
        set_ic!(h, 0, prof); set_ic!(h, 1, prof)
    end
    dtsame = true
    for n in 1:60
        dt = compute_dt(ha)
        dtsame &= (dt == compute_dt(hb))
        hierarchy_rk2_step!(ha, dt); hierarchy_rk2_step!(hb, dt)
        if n % pol.every == 0
            regrid!(ha, pol; compact = true)
            regrid!(hb, pol; compact = false)
            @test check_nesting(ha)
        end
    end
    @test dtsame
    # compacted invariants: dense Morton-sorted slots, empty freelist
    lev = ha.levels[2]; n1 = length(lev.live)
    @test n1 > 0
    @test lev.live == Int32.(1:n1)
    @test issorted([BA._morton(lev.meta[s], lev.B) for s in lev.live])
    @test isempty(lev.freelist)
    @test !compact_level!(ha, 1)                      # idempotent once compacted
    # bit-exact agreement, keyed by origin, every gas field, both levels
    blockmap(lv, fld) = begin
        h = Array(getfield(lv, fld))
        Dict(lv.meta[s].origin =>
             [h[(Int(s)-1)*lv.stride + ((lv.ng+k-1)*lv.nd + (lv.ng+j-1))*lv.nd + (lv.ng+i-1) + 1]
              for i in 1:lv.B, j in 1:lv.B, k in 1:lv.B]
             for s in lv.live)
    end
    for lidx in 1:2, fld in (:D, :S1, :S2, :S3, :Tau, :Ge)
        ma = blockmap(ha.levels[lidx], fld); mb = blockmap(hb.levels[lidx], fld)
        @test keys(ma) == keys(mb)
        @test all(ma[k] == mb[k] for k in keys(ma))
    end
end
end # for BE
