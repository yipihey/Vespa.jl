# test_ct_memory.jl — CT memory layout and block-size independence.
#   * CT has no ψ pools, scratch = 34 lanes (35 with the shock switch), the base
#     level is allocated exactly (no growth slack);
#   * the pre-update scratch lives in the O pools (ct.jl), so R ghost rings hold
#     scratch after a level step: a regrid must prolong from refilled rings (same
#     answer however often the rings were dirtied), conserve mass and keep ∇·B;
#   * a uniform level gives bitwise the same state for any block size (one block
#     covering the short axis included: a periodic block is its own neighbour).
const BA = BlockAMR

function ctm_box(; BE, B, nbase = (256, 64), bc = ((:outflow, :outflow), :periodic), sw = false, kw...)
    h = AMRHierarchy(; nbase, B, backend = BE, physics = :mhd, T = Float32, divb = :ct, bc,
                     shock_switch = sw, kw...)
    init_base_level!(h); build_level_tables!(h, 0)
    ctm_ic!(h, 0)
    return h
end
function ctm_ic!(h, l)
    lev = h.levels[l + 1]; dx = BA.level_dx(h, l); γ = h.gamma; b0 = 0.1
    BA.set_ct_faces_from_Az!(h, l, (x, y) -> -b0 * x)
    n = lev.cap * lev.stride; a = [zeros(n) for _ in 1:10]
    for s in lev.live, j in 0:lev.B-1, i in 0:lev.B-1
        o = lev.meta[s].origin; idx = (Int(s)-1)*lev.stride + (j+lev.ng)*lev.nd + (i+lev.ng) + 1
        x = (Int(o[1]) + i + 0.5) * dx; y = (Int(o[2]) + j + 0.5) * dx
        ρ = 1 + 40exp(-((x - 0.3)^2 + (y - 0.125)^2) / 0.03^2) + 0.1sin(2π * y / 0.25)
        u = 0.8 + 0.1sin(2π * y / 0.25); v = 0.05cos(2π * x); p = 1.0
        vals = (ρ, ρ*u, ρ*v, 0.0, p/(γ-1) + 0.5ρ*(u^2+v^2) + 0.5b0^2, p/(γ-1), 0.0, b0, 0.0, 0.0)
        for f in 1:10; a[f][idx] = vals[f]; end
    end
    BA.encode_mhd_from_host!(lev, a...)
end
# decoded active state by global cell (l, X, Y)
function ctm_state(h)
    out = Dict{NTuple{3,Int},NTuple{9,Float64}}()
    for l in 0:length(h.levels)-1
        lev = h.levels[l + 1]; isempty(lev.live) && continue
        G = map(Array, BA.gasfields(lev)); C = map(Array, BA.cfields(lev)); Bz = Array(lev.Bz)
        sc = map(Array, (lev.Dsc, lev.Ssc, lev.Ssc, lev.Ssc, lev.Esc, lev.Gsc))
        for s in lev.live, j in 0:lev.B-1, i in 0:lev.B-1
            o = lev.meta[s].origin; idx = (Int(s)-1)*lev.stride + (j+lev.ng)*lev.nd + (i+lev.ng) + 1
            out[(l, Int(o[1]) + i, Int(o[2]) + j)] =
                (ntuple(q -> Float64(G[q][idx]) * Float64(sc[q][s]), 6)..., Float64(C[1][idx]),
                 Float64(C[2][idx]), Float64(Bz[idx]))
        end
    end
    return out
end

for BE in BACKENDS
@testset "CT memory + block size [$BE]" begin
    # layout
    h = ctm_box(; BE, B = 16)
    lev = h.levels[1]
    @test isempty(lev.Psi) && isempty(lev.Psio)
    @test lev.cap == length(lev.live)                       # exact base allocation
    advance_hierarchy!(h)
    @test length(lev.tabs[:mrec]) == 34 * lev.cap * lev.stride
    hs = ctm_box(; BE, B = 16, sw = true); advance_hierarchy!(hs)
    @test length(hs.levels[1].tabs[:mrec]) == 35 * hs.levels[1].cap * hs.levels[1].stride

    # uniform level: bitwise independent of the block size, B up to the short axis
    ref = ctm_box(; BE, B = 16)
    for _ in 1:5; advance_hierarchy!(ref); end
    s0 = ctm_state(ref)
    for B in (32, 64)
        hb = ctm_box(; BE, B)
        for _ in 1:5; advance_hierarchy!(hb); end
        @test ctm_state(hb) == s0
    end

    # AMR: a regrid after dirty rings (ct.jl) — finite, ∇·B at round-off, and the
    # checkpoint round-trip keeps the file's ψ column for a level without the pool
    for B in (16, 32)
        ha = ctm_box(; BE, B)
        pol = BlockRefinementPolicy(; dthresh = 3.0, nbuf = 2, every = 1, lmax = 2)
        regrid!(ha, pol)
        for l in 1:length(ha.levels)-1; isempty(ha.levels[l+1].live) || ctm_ic!(ha, l); end
        for _ in 1:12; advance_hierarchy!(ha); regrid!(ha, pol); end
        @test length(ha.levels) >= 2 && !isempty(ha.levels[2].live)
        st = ctm_state(ha)
        @test all(v -> all(isfinite, v) && v[1] > 0, values(st))
        @test BA.max_divb_ct(ha) < 1e-4
        f = tempname(); save_checkpoint(f, ha)
        h2, _ = load_checkpoint(f; backend = BE)
        @test ctm_state(h2) == st
        rm(f; force = true)
    end
end
end # for BE
