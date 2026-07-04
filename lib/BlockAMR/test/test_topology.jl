# test_topology.jl — exact-integer topology: origin chains beyond f64, periodic
# wrap (incl. single-block self-wrap), nesting validator, slot pool churn.
const BA = BlockAMR

@testset "origin exactness to 70 levels (UInt128 vs BigInt)" begin
    nbase = (512, 512, 512)
    # walk a 70-level chain of child placements with pseudo-random offsets and
    # verify the UInt128 arithmetic against exact BigInt (period 512·2^70 ≈ 2^79).
    B = 16; Bh = B - B ÷ 2
    org  = (UInt128(496), UInt128(0), UInt128(272))     # a level-0 block origin
    borg = BigInt.(Int.(org))
    for l in 1:70
        P  = BA.level_period(nbase, l)
        bP = BigInt(512) * BigInt(2)^l
        off = ntuple(d -> Int16(mod(7l + 3d, Bh + 1)), 3)
        org  = BA.child_origin(org, off, P)
        borg = ntuple(d -> mod(2 * (borg[d] + Int(off[d])), bP), 3)
        @test all(BigInt(org[d]) == borg[d] for d in 1:3)
    end
    @test any(o -> o > UInt128(2)^60, org)              # genuinely beyond f64 exactness
    # f64 CANNOT represent these origins exactly (the point of integer topology)
    @test Float64(org[1]) != org[1] || Float64(org[3]) != org[3]
end

@testset "wrapped-interval overlap + axis images" begin
    P = UInt128(32)
    @test BA.overlap1(UInt128(30), 4, UInt128(0), 4, P)        # wraps into [0,2)
    @test !BA.overlap1(UInt128(10), 4, UInt128(20), 4, P)
    @test BA.overlap1(UInt128(0), 32, UInt128(17), 3, P)       # full-axis interval
    imgs = BA.axis_images(Int128(-2), 20, Int128(0), 16, Int128(16))
    # single block per axis (P=16): images at s=−16 → [−2,0), s=0 → [0,16), s=+16 → [16,18)
    @test length(imgs) == 3
    @test (Int128(-2), 2, Int128(-16)) in imgs
    @test (Int128(0), 16, Int128(0)) in imgs
    @test (Int128(16), 2, Int128(16)) in imgs
end

@testset "overlapping_blocks: tilemap + periodic (2×2×2 base)" begin
    hier = AMRHierarchy(; nbase = (32, 32, 32), B = 16)
    lev = init_base_level!(hier)
    @test length(lev.live) == 8
    # a query box centered on the far corner touches all 8 blocks periodically
    hits = overlapping_blocks(lev, (Int128(-2), Int128(-2), Int128(-2)), (4, 4, 4))
    @test sort(hits) == sort(lev.live)
    # a strictly interior box touches exactly one block
    hits = overlapping_blocks(lev, (Int128(2), Int128(2), Int128(2)), (4, 4, 4))
    @test length(hits) == 1
end

@testset "nesting validator accepts / rejects" begin
    hier = AMRHierarchy(; nbase = (32, 32, 32), B = 16)
    init_base_level!(hier)
    # a child anywhere in a fully tiled level 0 is properly nested (periodic coverage)
    s1 = add_block!(hier, 1, hier.levels[1].live[1], (0, 0, 0))
    @test check_nesting(hier)
    # level-2 child strictly interior to the lone level-1 block: covers parent
    # cells [4,12), grown margin [3,13) ⊂ [0,16) → properly nested
    s2 = add_block!(hier, 2, s1, (4, 4, 4))
    @test check_nesting(hier)
    remove_block!(hier, 2, s2)
    # at offset (0,0,0) the grown margin [−1,9) pokes out of the lone level-1
    # block → violation (the parent's level-1 face/edge/corner siblings are missing)
    s3 = add_block!(hier, 2, s1, (0, 0, 0))
    @test !check_nesting(hier)
    remove_block!(hier, 2, s3)
    @test check_nesting(hier)
end

@testset "slot alloc/free/grow preserves data" begin
    hier = AMRHierarchy(; nbase = (32, 32, 32), B = 16, cap0 = 2)
    lev = init_base_level!(hier)                        # 8 blocks > cap0 → forced growth
    @test lev.cap >= 8
    for (n, s) in enumerate(lev.live)
        fill!(blockview(lev, s, :D), Float32(n))
    end
    ref = [Array(blockview(lev, s, :D))[1] for s in lev.live]
    BA._grow!(lev, lev.cap * 2)                         # explicit growth must preserve
    @test [Array(blockview(lev, s, :D))[1] for s in lev.live] == ref
    # free + realloc reuses the slot
    s1 = add_block!(hier, 1, lev.live[1], (2, 2, 2))
    remove_block!(hier, 1, s1)
    s2 = add_block!(hier, 1, lev.live[1], (3, 3, 3))
    @test s2 == s1                                       # freelist reuse
    lev1 = hier.levels[2]
    @test length(lev1.live) == 1 && BA.isalive(lev1.meta[s2])
end
