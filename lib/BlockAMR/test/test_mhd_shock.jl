# test_mhd_shock.jl — the MHD shock switch (PLUTO's MULTID shock flattening, mhd.jl)
# against the carbuncle: Quirk's odd-even test, a Mach-6 planar shock running down
# a channel with a 1e-3 density perturbation on the centre row, B = 0 (HLLD ≡ HLLC).
#   * HLLD alone lets transverse velocity grow behind the shock (odd-even
#     decoupling, ≫ HLL's level); with `shock_switch` it stays at HLL's level;
#   * the switch leaves smooth flow untouched (no flags: bitwise the HLLD result).
const BA = BlockAMR
const SHOCK_SMOOTH_E = Ref{Any}(nothing); const SHOCK_SMOOTH_D = Ref{Any}(nothing)

function quirk_box(; BE, divb, sw, riemann = :hlld, nx = 400, ny = 32)
    γ = 5/3; M = 6.0; ρ2 = (γ+1)*M^2 / ((γ-1)*M^2 + 2); p2 = 1 + 2γ/(γ+1)*(M^2 - 1)
    Us = M * sqrt(γ); u2 = Us * (1 - 1/ρ2)
    h = AMRHierarchy(; nbase = (nx, ny), B = 16, box = 1.0, backend = BE, physics = :mhd, T = Float64,
                     divb, riemann, shock_switch = sw, bc = ((:inflow, :outflow), :periodic),
                     inflow = Dict((1, :lo) => (ρ2, u2, 0.0, 0.0, p2, 0.0, 0.0, 0.0)))
    init_base_level!(h); build_level_tables!(h, 0)
    lev = h.levels[1]; n = lev.cap * lev.stride; a = [zeros(n) for _ in 1:10]; dx = 1.0 / nx
    for s in lev.live, j in 0:lev.B-1, i in 0:lev.B-1
        o = lev.meta[s].origin; idx = (Int(s)-1)*lev.stride + (j+lev.ng)*lev.nd + (i+lev.ng) + 1
        x = (Int(o[1]) + i + 0.5) * dx; J = Int(o[2]) + j
        ρ, u, p = x < 0.05 ? (ρ2, u2, p2) : ((J == ny ÷ 2 ? 1 + 1e-3 : 1.0), 0.0, 1.0)
        vals = (ρ, ρ*u, 0.0, 0.0, p/(γ-1) + 0.5ρ*u^2, p/(γ-1), 0.0, 0.0, 0.0, 0.0)
        for f in 1:10; a[f][idx] = vals[f]; end
    end
    BA.encode_mhd_from_host!(lev, a...)
    t = 0.0; while t < 0.6 / Us; t += advance_hierarchy!(h); end
    S1, S2, D = Array(lev.S1), Array(lev.S2), Array(lev.D); ss, ds = Array(lev.Ssc), Array(lev.Dsc)
    vy = maximum(abs(S2[(Int(s)-1)*lev.stride + (j+lev.ng)*lev.nd + (i+lev.ng) + 1] * ss[s] /
                     (D[(Int(s)-1)*lev.stride + (j+lev.ng)*lev.nd + (i+lev.ng) + 1] * ds[s]))
             for s in lev.live, j in 0:lev.B-1, i in 0:lev.B-1)
    return vy / u2, h
end

for BE in BACKENDS
@testset "MHD shock switch (Quirk odd-even, B = 0) [$BE]" begin
    for divb in (:ct, :glm)
        v0, _ = quirk_box(; BE, divb, sw = false)
        v1, _ = quirk_box(; BE, divb, sw = true)
        vh, _ = quirk_box(; BE, divb, sw = false, riemann = :hll)
        @test v0 > 30 * vh                     # HLLD (≡ HLLC): odd-even decoupling grows
        @test v1 < 5 * vh                      # switched: HLL's level
        @test v1 < v0 / 10
    end
    # smooth flow: nothing flagged ⇒ identical to the unswitched run
    for sw in (false, true)
        h = AMRHierarchy(; nbase = (32, 32), B = 16, backend = BE, physics = :mhd, T = Float64,
                         divb = :ct, shock_switch = sw)
        init_base_level!(h); build_level_tables!(h, 0)
        lev = h.levels[1]; n = lev.cap * lev.stride; a = [zeros(n) for _ in 1:10]; γ = h.gamma
        for s in lev.live, j in 0:lev.B-1, i in 0:lev.B-1
            o = lev.meta[s].origin; idx = (Int(s)-1)*lev.stride + (j+lev.ng)*lev.nd + (i+lev.ng) + 1
            x = (Int(o[1]) + i + 0.5) / 32; y = (Int(o[2]) + j + 0.5) / 32
            ρ = 1 + 0.2sin(2π * x) * cos(2π * y)
            vals = (ρ, ρ, 0.5ρ, 0.0, 1/(γ-1) + 0.5ρ*1.25, 1/(γ-1), 0.0, 0.0, 0.0, 0.0)
            for f in 1:10; a[f][idx] = vals[f]; end
        end
        BA.encode_mhd_from_host!(lev, a...)
        for _ in 1:20; advance_hierarchy!(h); end
        sw ? (@test Array(lev.Tau) == SHOCK_SMOOTH_E[] && Array(lev.D) == SHOCK_SMOOTH_D[]) :
             (SHOCK_SMOOTH_E[] = Array(lev.Tau); SHOCK_SMOOTH_D[] = Array(lev.D))
    end
end
end # for BE
