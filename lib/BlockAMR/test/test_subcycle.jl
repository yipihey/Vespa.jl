# test_subcycle.jl — Phase 3 gates: strict 2:1 W-cycle (fine-first, frozen
# coarse ghosts, ONE global λ).
#   * conservation to round-off WITH subcycling, incl. a shock crossing the C/F
#     interface (weights ¼/½ are the load-bearing claim);
#   * exact integer time bookkeeping: nstep[l] = 2^l per root step;
#   * padding the hierarchy with an empty level changes nothing bitwise.
const BA = BlockAMR

# 3-level static tower: level 1 over parent cells [8,24)³ (as center_refined),
# level 2 over level-1 cells [24,40)³ (strictly nested).
function tower3(; backend = :cpu)
    hier = center_refined(; backend)
    lev1 = hier.levels[2]
    for s in lev1.live
        m = lev1.meta[s]
        off = ntuple(d -> m.origin[d] == UInt128(16) ? 8 : 0, 3)
        add_block!(hier, 2, s, off)
    end
    @test check_nesting(hier)
    for l in 0:2
        build_level_tables!(hier, l)
        l >= 1 && build_cf_register!(hier, l)
    end
    return hier
end

for BE in BACKENDS
@testset "2:1 W-cycle: 3-level Sedov conservation + clocks [$BE]" begin
    σ = 2.0 / 128
    prof = sedov(σ, 1.0, 1e-5)
    hier = tower3(; backend = BE)
    for l in 0:2
        set_ic!(hier, l, prof)
    end
    M0, _, _, _, E0 = total_conserved(hier)
    nroot = 12
    for _ in 1:nroot
        advance_hierarchy!(hier)
    end
    M1, _, _, _, E1 = total_conserved(hier)
    @test abs(M1 - M0) / M0 < 3e-5
    @test abs(E1 - E0) / E0 < 3e-5
    @test hier.nstep[1] == nroot                     # exact binary clocks
    @test hier.nstep[2] == 2nroot
    @test hier.nstep[3] == 4nroot
end

@testset "2:1 shock crossing C/F: reflux weights are right [$BE]" begin
    σ = 2.0 / 64
    prof = x -> begin
        r2 = (x[1] - 0.30)^2 + (x[2] - 0.5)^2 + (x[3] - 0.5)^2
        (1.0, 0.0, 0.0, 0.0, 1e-5 + (2/3) * exp(-r2 / (2σ^2)) / ((2π)^1.5 * σ^3))
    end
    hier = center_refined(; backend = BE)            # 2 levels, subcycled now
    set_ic!(hier, 0, prof); set_ic!(hier, 1, prof)
    Ma, _, _, _, Ea = total_conserved(hier)
    for _ in 1:40
        advance_hierarchy!(hier)
    end
    Mb, _, _, _, Eb = total_conserved(hier)
    @test abs(Mb - Ma) / Ma < 2e-6                   # mass: round-off through crossing
    @test abs(Eb - Ea) / Ea < 2e-6                   # energy: the ¼/½ weight gate
end

@testset "empty-level padding is a bitwise no-op [$BE]" begin
    σ = 2.0 / 64
    prof = sedov(σ, 1.0, 1e-5)
    mk() = begin
        h = center_refined(; backend = BE)
        set_ic!(h, 0, prof); set_ic!(h, 1, prof)
        h
    end
    h1 = mk()
    h2 = mk(); BA.ensure_level!(h2, 2)               # empty level 2 appended
    build_level_tables!(h2, 2); build_cf_register!(h2, 2)
    for _ in 1:8
        advance_hierarchy!(h1); advance_hierarchy!(h2)
    end
    same = true
    for il in 1:2, f in (:D, :S1, :S2, :S3, :Tau, :Ge)
        Array(getfield(h1.levels[il], f)) == Array(getfield(h2.levels[il], f)) ||
            (same = false)
    end
    @test same
end
end # for BE
