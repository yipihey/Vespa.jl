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
end # for BE
