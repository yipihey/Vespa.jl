# test_mhd_de.jl — MHD dual energy (entropy form, mhd.jl / ct.jl), CT + GLM.
#   * selection: thermal energy from E − KE − ½B² only where it exceeds η1·E and
#     0.1·½B², else the advected Ge;
#   * ±10 c_s double rarefaction, first step: the advected entropy gives the
#     isentropic pressure while the total energy is heated 10× at the start-up
#     discontinuity;
#   * Mach-100 advection of a density wave (thermal fraction 1e-4, Ge selected):
#     uniform pressure kept to < 0.5 % over 40 steps; mass and momentum still
#     conserved to round-off;
#   * a cell seeded with E − KE − ½B² < 0 (the truncation error that pinned the
#     HVC wake at the 10 K floor) keeps a positive pressure and gets E rebuilt;
#   * dual-energy parameters survive a checkpoint round-trip; old checkpoints
#     (no flag) restore with it off.
const BA = BlockAMR

# 2-D MHD box (64×64 unless nbase given), fields from prof(x, y) = (ρ, u, v, p, Bx, By)
function de_box(; BE, divb, prof, nbase = (64, 64), bc = :periodic, de = true, Az = nothing)
    h = AMRHierarchy(; nbase, B = 16, backend = BE, physics = :mhd, T = Float64, divb, bc,
                     dual_energy = de)
    init_base_level!(h); build_level_tables!(h, 0)
    lev = h.levels[1]; n = lev.cap * lev.stride; a = [zeros(n) for _ in 1:10]
    dx = BA.level_dx(h, 0); γ = h.gamma
    for s in lev.live, j in 0:lev.B-1, i in 0:lev.B-1
        o = lev.meta[s].origin; idx = (Int(s)-1)*lev.stride + (j+lev.ng)*lev.nd + (i+lev.ng) + 1
        ρ, u, v, p, bx, by = prof((Int(o[1]) + i + 0.5) * dx, (Int(o[2]) + j + 0.5) * dx)
        vals = (ρ, ρ*u, ρ*v, 0.0, p/(γ-1) + 0.5ρ*(u^2 + v^2) + 0.5(bx^2 + by^2), p/(γ-1),
                bx, by, 0.0, 0.0)
        for f in 1:10; a[f][idx] = vals[f]; end
    end
    BA.encode_mhd_from_host!(lev, a...)
    divb === :ct && Az !== nothing && BA.set_ct_faces_from_Az!(h, 0, Az)
    return h
end

# per-cell (i, j) → (ρ, mx, my, p(E), p(Ge), E)
function de_cells(h)
    lev = h.levels[1]; γ = h.gamma
    D, S1, S2, E, G, Bx, By, Bz = map(Array, (lev.D, lev.S1, lev.S2, lev.Tau, lev.Ge, lev.Bx, lev.By, lev.Bz))
    ds, ss, es, gs = map(Array, (lev.Dsc, lev.Ssc, lev.Esc, lev.Gsc))
    out = Dict{NTuple{2,Int},NTuple{6,Float64}}()
    for s in lev.live, j in 0:lev.B-1, i in 0:lev.B-1
        o = lev.meta[s].origin; idx = (Int(s)-1)*lev.stride + (j+lev.ng)*lev.nd + (i+lev.ng) + 1
        ρ = D[idx]*ds[s]; m1 = S1[idx]*ss[s]; m2 = S2[idx]*ss[s]; e = E[idx]*es[s]
        pE = (γ-1)*(e - 0.5(m1^2 + m2^2)/ρ - 0.5(Bx[idx]^2 + By[idx]^2 + Bz[idx]^2))
        out[(Int(o[1]) + i, Int(o[2]) + j)] = (ρ, m1, m2, pE, (γ-1)*G[idx]*gs[s], e)
    end
    return out
end

@testset "MHD dual energy: selection rule" begin
    T = Float64; η1 = 1e-3
    @test BA._de_total(10.0, 4.0, 1.0, 0.7, η1) == 10.0          # thermal 5 of 10: trusted
    @test BA._de_total(10.0, 9.995, 0.0, 0.7, η1) ≈ 10.695       # 5e-4 < η1·E: Ge
    @test BA._de_total(10.0, 3.0, 7.5, 0.7, η1) ≈ 11.2           # negative: Ge
    @test BA._de_total(10.0, 0.0, 9.5, 0.7, η1) ≈ 10.2           # β ≈ 0.035 < 0.07: Ge
    @test BA._de_total(10.0, 0.0, 8.0, 0.7, η1) == 10.0          # β ≈ 0.17: trusted
end

for BE in BACKENDS
@testset "MHD dual energy [$BE]" begin
    for divb in (:ct, :glm)
        # ±10 c_s double rarefaction (x), transverse B: first step
        U = 10.0; p0 = 0.4; b = 0.1; γ = 5/3
        h = de_box(; BE, divb, nbase = (256, 16), bc = Dict(1 => :outflow, 2 => :periodic),
                   prof = (x, y) -> (1.0, x < 0.5 ? -U : U, 0.0, p0, 0.0, b), Az = (x, y) -> -b * x)
        advance_hierarchy!(h)
        c = de_cells(h)[(127, 3)]
        @test isapprox(c[5], p0 * c[1]^γ; rtol = 1e-6)          # advected entropy: isentropic
        @test c[4] > 5 * p0 * c[1]^γ                            # total energy: start-up heating
        @test all(v -> v[5] > 0, values(de_cells(h)))

        # Mach-100 advection of a density wave, β = 100: Ge selected everywhere
        B0 = sqrt(2 / 100); u = 100 * sqrt(γ)
        prof = (x, y) -> (1 + 0.5sin(2π * x) * sin(2π * y), u, 0.5u, 1.0, B0, 0.0)
        h = de_box(; BE, divb, prof, Az = (x, y) -> B0 * y)
        c0 = de_cells(h)
        M0 = sum(v[1] for v in values(c0)); P0 = sum(v[2] for v in values(c0))
        for _ in 1:40; advance_hierarchy!(h); end
        c1 = de_cells(h)
        @test maximum(abs(v[5] - 1) for v in values(c1)) < 5e-3
        @test isapprox(sum(v[1] for v in values(c1)), M0; rtol = 1e-12)
        @test isapprox(sum(v[2] for v in values(c1)), P0; rtol = 1e-12)

        # uniform magnetized state; one cell with E − KE − ½B² < 0 (Ge positive)
        h = de_box(; BE, divb, nbase = (32, 32), prof = (x, y) -> (1.0, 0.3, 0.0, 1e-3, 0.0, 0.3),
                   Az = (x, y) -> -0.3 * x)
        lev = h.levels[1]; s = lev.byorigin[(UInt128(0), UInt128(0), UInt128(0))]
        idx = (Int(s)-1)*lev.stride + (5+lev.ng)*lev.nd + (5+lev.ng) + 1
        E = Array(lev.Tau); es = Array(lev.Esc)
        E[idx] -= 2 * (1e-3 / (γ - 1)) / es[s]                  # thermal part → −p/(γ−1)
        copyto!(lev.Tau, E)
        @test de_cells(h)[(5, 5)][4] < 0
        for _ in 1:5; advance_hierarchy!(h); end
        cs = de_cells(h)
        @test all(v -> v[5] > 0, values(cs))
        @test maximum(abs(v[5] / 1e-3 - 1) for v in values(cs)) < 0.05
        @test cs[(6, 5)][4] > 0                                 # E rebuilt downstream

        # without dual energy the same seed leaves a non-positive pressure behind
        hn = de_box(; BE, divb, nbase = (32, 32), de = false,
                    prof = (x, y) -> (1.0, 0.3, 0.0, 1e-3, 0.0, 0.3), Az = (x, y) -> -0.3 * x)
        ln = hn.levels[1]; En = Array(ln.Tau); En[idx] -= 2 * (1e-3 / (γ - 1)) / Array(ln.Esc)[s]
        copyto!(ln.Tau, En)
        for _ in 1:5; advance_hierarchy!(hn); end
        @test minimum(v[4] for v in values(de_cells(hn))) < 1e-4
    end

    # checkpoint round-trip keeps the dual-energy settings
    h = AMRHierarchy(; nbase = (32, 32), B = 16, backend = BE, physics = :mhd, T = Float64,
                     divb = :ct, dual_energy = true, de_eta1 = 2e-3, de_eta2 = 0.2)
    init_base_level!(h); build_level_tables!(h, 0)
    f = tempname()
    save_checkpoint(f, h)
    h2, _ = load_checkpoint(f; backend = BE)
    @test h2.mhd.de && h2.mhd.eta1 == 2e-3 && h2.mhd.eta2 == 0.2
end
end # for BE
