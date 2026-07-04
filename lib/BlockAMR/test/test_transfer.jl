# test_transfer.jl — batched data-motion geometry: sibling ghost fill (incl.
# single-block self-wrap), PC prolongation (parent lookup), conservative
# restriction, and the prolong→restrict identity.  Backend-generic: all staging
# goes through whole-pool host arrays (no scalar device indexing).
const BA = BlockAMR

# value encodings, exact in Float32 / representable in UInt16
vgas(g) = Float32(mod(g[1], 19) + 20 * mod(g[2], 19) + 400 * mod(g[3], 19) + 1)
vsp(g)  = UInt16(mod(g[1] + 3g[2] + 7g[3], 60000) + 1)

# stage f(global cell) into the ACTIVE cells of every live block of `lev`
function stage!(lev, fld::Symbol, f)
    h = Array(getfield(lev, fld))
    nd = lev.nd; ng = lev.ng; B = lev.B
    for s in lev.live
        m = lev.meta[s]; base = (Int(s) - 1) * lev.stride
        for k in 1:B, j in 1:B, i in 1:B
            g = ntuple(d -> BA.wrapc(Int128(m.origin[d]) + (i, j, k)[d] - 1, lev.P[d]), 3)
            h[base + ((ng+k-1) * nd + (ng+j-1)) * nd + (ng+i-1) + 1] = f(g)
        end
    end
    copyto!(getfield(lev, fld), h)
end

# expected value at stored cell u (1-based) of block s: `fine` if the wrapped
# global cell lies in a live same-level block, else parent PC lookup via `coarse`
function expected(lev, s, u, fine, coarse)
    m = lev.meta[s]
    gu = ntuple(d -> Int128(m.origin[d]) + u[d] - 1 - lev.ng, 3)     # unwrapped
    g  = ntuple(d -> BA.wrapc(gu[d], lev.P[d]), 3)
    for c in lev.live                                                # same-level owner?
        cm = lev.meta[c]
        all(d -> mod(Int128(g[d]) - Int128(cm.origin[d]), Int128(lev.P[d])) < lev.B, 1:3) &&
            return fine(g)
    end
    return coarse(ntuple(d -> BA.wrapc(fld(gu[d], 2), lev.P[d] >> 1), 3))
end

for BE in BACKENDS
@testset "sibling ghost fill, 2×2×2 base [$BE]" begin
    hier = AMRHierarchy(; nbase = (32, 32, 32), B = 16, backend = BE, nsp = 1)
    lev = init_base_level!(hier)
    stage!(lev, :D, vgas); stage!(lev, :Tau, g -> 2 * vgas(g))
    # species staged through the same pattern (u16)
    hs = Array(lev.sp[1])
    for s in lev.live
        m = lev.meta[s]; base = (Int(s) - 1) * lev.stride
        for k in 1:16, j in 1:16, i in 1:16
            g = ntuple(d -> BA.wrapc(Int128(m.origin[d]) + (i, j, k)[d] - 1, lev.P[d]), 3)
            hs[base + ((lev.ng+k-1) * lev.nd + (lev.ng+j-1)) * lev.nd + (lev.ng+i-1) + 1] = vsp(g)
        end
    end
    copyto!(lev.sp[1], hs)
    BA.build_level_tables!(hier, 0)
    fill_ghosts!(hier, 0)
    hD = Array(lev.D); hT = Array(lev.Tau); hS = Array(lev.sp[1])
    ok = 0; bad = 0
    for s in lev.live
        m = lev.meta[s]; base = (Int(s) - 1) * lev.stride
        for k in 1:lev.nd, j in 1:lev.nd, i in 1:lev.nd
            g = ntuple(d -> BA.wrapc(Int128(m.origin[d]) + (i, j, k)[d] - 1 - lev.ng, lev.P[d]), 3)
            idx = base + ((k-1) * lev.nd + (j-1)) * lev.nd + (i-1) + 1
            (hD[idx] == vgas(g) && hT[idx] == 2 * vgas(g) && hS[idx] == vsp(g)) ?
                (ok += 1) : (bad += 1)
        end
    end
    @test bad == 0
    @test ok == 8 * lev.nd^3
end

@testset "self-wrap ghost fill, single base block [$BE]" begin
    hier = AMRHierarchy(; nbase = (16, 16, 16), B = 16, backend = BE)
    lev = init_base_level!(hier)
    @test length(lev.live) == 1
    stage!(lev, :D, vgas)
    BA.build_level_tables!(hier, 0)
    fill_ghosts!(hier, 0)
    hD = Array(lev.D)
    bad = 0
    for k in 1:lev.nd, j in 1:lev.nd, i in 1:lev.nd
        g = ntuple(d -> BA.wrapc(Int128((i, j, k)[d] - 1 - lev.ng), lev.P[d]), 3)
        hD[((k-1) * lev.nd + (j-1)) * lev.nd + (i-1) + 1] == vgas(g) || (bad += 1)
    end
    @test bad == 0
end

@testset "level-1 ghosts: sibling + PC parent prolongation [$BE]" begin
    hier = AMRHierarchy(; nbase = (32, 32, 32), B = 16, backend = BE)
    lev0 = init_base_level!(hier)
    p0 = lev0.byorigin[(UInt128(0), UInt128(0), UInt128(0))]
    c1 = add_block!(hier, 1, p0, (0, 0, 0))     # fine active [0,16)³
    c2 = add_block!(hier, 1, p0, (8, 0, 0))     # fine active [16,32)×[0,16)²
    lev1 = hier.levels[2]
    vpar(g)  = Float32(1000 + mod(g[1], 13) + 15 * mod(g[2], 13) + 250 * mod(g[3], 13))
    vfine(g) = vgas(g)
    stage!(lev0, :D, vpar)
    stage!(lev1, :D, vfine)
    BA.build_level_tables!(hier, 0); BA.build_level_tables!(hier, 1)
    fill_ghosts!(hier, 0)
    fill_ghosts!(hier, 1)
    hD = Array(lev1.D)
    bad = 0; nsib = 0; npro = 0
    for s in lev1.live
        base = (Int(s) - 1) * lev1.stride
        for k in 1:lev1.nd, j in 1:lev1.nd, i in 1:lev1.nd
            want = expected(lev1, s, (i, j, k), vfine, vpar)
            got  = hD[base + ((k-1) * lev1.nd + (j-1)) * lev1.nd + (i-1) + 1]
            got == want || (bad += 1)
        end
    end
    @test bad == 0
end

@testset "restriction: 2³ mean + prolong→restrict identity [$BE]" begin
    hier = AMRHierarchy(; nbase = (32, 32, 32), B = 16, backend = BE)
    lev0 = init_base_level!(hier)
    p0 = lev0.byorigin[(UInt128(0), UInt128(0), UInt128(0))]
    c1 = add_block!(hier, 1, p0, (2, 4, 6))
    lev1 = hier.levels[2]
    stage!(lev1, :D, vgas)
    BA.build_level_tables!(hier, 1)
    restrict_level!(hier, 1)
    hC = Array(lev0.D)
    m1 = lev1.meta[c1]
    bad = 0
    for k in 0:7, j in 0:7, i in 0:7                    # B÷2 covered coarse cells
        # coarse cell (in parent stored frame) ← mean of 8 fine cells
        exp8 = 0.0f0
        for ck in 0:1, cj in 0:1, ci in 0:1
            g = ntuple(d -> BA.wrapc(Int128(m1.origin[d]) + 2 * (i, j, k)[d] + (ci, cj, ck)[d],
                                     lev1.P[d]), 3)
            exp8 += vgas(g)
        end
        exp8 *= 0.125f0
        di = lev0.ng + Int(m1.offset[1]) + i; dj = lev0.ng + Int(m1.offset[2]) + j
        dk = lev0.ng + Int(m1.offset[3]) + k
        base = (Int(p0) - 1) * lev0.stride
        got = hC[base + (dk * lev0.nd + dj) * lev0.nd + di + 1]
        isapprox(got, exp8; rtol = 1e-6) || (bad += 1)
    end
    @test bad == 0
    # PC-prolong the fine interior from the parent, restrict back → parent unchanged
    stage!(lev0, :D, vgas)
    tab = BA.to_device_table(lev1.be, BA.build_prolong_jobs(lev1, lev0; interior = true))
    BA._run_prolong!(lev1.be, tab, lev1.D, lev0.D, lev0.D, 0.0f0, lev1.nd, lev1.stride)
    restrict_level!(hier, 1)
    hC2 = Array(lev0.D)
    m = lev0.meta[p0]; base = (Int(p0) - 1) * lev0.stride
    bad = 0
    for k in 0:7, j in 0:7, i in 0:7
        di = lev0.ng + Int(m1.offset[1]) + i; dj = lev0.ng + Int(m1.offset[2]) + j
        dk = lev0.ng + Int(m1.offset[3]) + k
        g = ntuple(d -> BA.wrapc(Int128(m.origin[d]) +
                                 (Int(m1.offset[d]) + (i, j, k)[d]), lev0.P[d]), 3)
        got = hC2[base + (dk * lev0.nd + dj) * lev0.nd + di + 1]
        got == vgas(g) || (bad += 1)
    end
    @test bad == 0
end
end # for BE
