# test_ct.jl — constrained-transport MHD (physics = :mhd, divb = :ct), 2-D.
# Scheme cross-check vs NestedMHD's CT mode: bench/xcheck_nestedmhd.jl (≤ 3.3e-8,
# face B included).  Gates here (Float64 unless noted, every backend):
#   * ∇·B (face) stays at round-off across C/F, walls, inflow/outflow, the W-cycle,
#     E_z correction + face restriction, and dynamic regrid (divergence-free new
#     blocks, destruction, compaction);
#   * uniform magnetized flow is an exact fixed point; Orszag–Tang with AMR
#     conserves M, E, p, ∫B to round-off; reflecting box (Bn = 0 walls);
#   * CP Alfvén wave 2nd order; checkpoint bit-identical; CPU ≡ GPU; Float32.
# Requires test_2d.jl / test_mhd.jl helpers (static2d, cellmap2, KIDS_*, set_mhd_ic!).
const BA = BlockAMR

"Cells from `prof` with B_cc = the face averages of the A_z faces; E consistent."
function set_ct_ic!(h, prof, Az)
    for l in 0:length(h.levels)-1
        lev = h.levels[l + 1]
        isempty(lev.live) && continue
        set_ct_faces_from_Az!(h, l, Az)
        bx = Array(lev.Bx); by = Array(lev.By)
        set_mhd_ic!(h, l, prof)                  # gas (+ cell B from prof, replaced below)
        copyto!(lev.Bx, bx); copyto!(lev.By, by)
        hD = Array(lev.D); s1 = Array(lev.S1); s2 = Array(lev.S2); s3 = Array(lev.S3)
        hG = Array(lev.Ge); hz = Array(lev.Bz); hE = Array(lev.Tau)
        for q in eachindex(hE)
            hD[q] > 0 || continue
            hE[q] = hG[q] + 0.5 * (s1[q]^2 + s2[q]^2 + s3[q]^2) / hD[q] +
                    0.5 * (bx[q]^2 + by[q]^2 + hz[q]^2)
        end
        copyto!(lev.Tau, hE)
    end
end

ctot(x) = (25 / (36π), -sin(2π * x[2]), sin(2π * x[1]), 0.0, 5 / (12π), 0.0, 0.0, 0.0)
ctAz(x, y) = (1 / sqrt(4π)) * (cos(4π * x) / (4π) + cos(2π * y) / (2π))
const KIDS_C4 = (((16, 16), (8, 8)), ((32, 16), (0, 8)), ((16, 32), (8, 0)), ((32, 32), (0, 0)))

for BE in BACKENDS
@testset "CT: uniform magnetized flow — fixed point, ∇·B = 0 (walls, in/out, C/F) [$BE]" begin
    for (bc, st) in ((((:inflow, :outflow), :periodic), (1.0, 0.7, 0.2, 0.1, 1.0, 0.5, -0.3, 0.4)),
                     ((:periodic, :reflecting), (1.0, 0.7, 0.0, 0.1, 1.0, 0.5, 0.0, 0.4)))
        h = static2d(; nbase = (64, 32), backend = BE, kids = KIDS_ALL, physics = :mhd, divb = :ct,
                     T = Float64, bc, inflow = Dict((1, :lo) => st))
        set_ct_ic!(h, x -> st, (x, y) -> st[6] * y - st[7] * x)
        r0 = [cellmap2(h, l, f) for l in 0:1, f in (:D, :S1, :Tau, :Bx, :Bz)]
        for _ in 1:6; advance_hierarchy!(h); end
        worst = 0.0
        for l in 0:1, (fi, f) in enumerate((:D, :S1, :Tau, :Bx, :Bz)), (g, v) in cellmap2(h, l, f)
            worst = max(worst, abs(v - r0[l + 1, fi][g]) / max(abs(r0[l + 1, fi][g]), 1e-12))
        end
        @test worst < 1e-12
        @test max_divb(h) < 1e-13
    end
end

@testset "CT: Orszag–Tang with AMR — round-off conservation and ∇·B [$BE]" begin
    h = static2d(; nbase = (64, 64), backend = BE, kids = KIDS_C4, physics = :mhd, divb = :ct,
                 T = Float64)
    set_ct_ic!(h, ctot, ctAz)
    @test max_divb(h) < 1e-14
    a = total_conserved(h); ba = total_bflux(h)
    t = 0.0
    while t < 0.2; t += advance_hierarchy!(h); end
    b = total_conserved(h); bb = total_bflux(h)
    @test abs(b[1] - a[1]) / a[1] < 1e-12
    @test abs(b[5] - a[5]) / a[5] < 1e-12
    @test abs(b[2] - a[2]) < 1e-13 && abs(b[3] - a[3]) < 1e-13
    @test all(abs(bb[d] - ba[d]) < 1e-13 for d in 1:3)
    @test max_divb(h) < 1e-13
end

@testset "CT: reflecting box (Bn = 0 walls) — blast, ∇·B, M/E [$BE]" begin
    σ = 0.03
    prof = x -> (1.0, 0.0, 0.0, 0.0,
                 1e-2 + 0.4 * exp(-((x[1] - 0.36)^2 + (x[2] - 0.22)^2) / (2σ^2)) / (2π * σ^2),
                 0.0, 0.0, 0.1)
    # field tangential to every wall: a closed loop A_z ∝ sin(πx)·sin(2πy/Ly) vanishes on the walls
    Az = (x, y) -> 0.02 * sin(π * x) * sin(2π * y)
    h = static2d(; nbase = (64, 32), backend = BE, kids = KIDS_ALL, physics = :mhd, divb = :ct,
                 T = Float64, bc = :reflecting)
    set_ct_ic!(h, prof, Az)
    M0, _, _, _, E0 = total_conserved(h)
    for _ in 1:40; advance_hierarchy!(h); end
    M1, _, _, _, E1 = total_conserved(h)
    @test abs(M1 - M0) / M0 < 1e-11 && abs(E1 - E0) / E0 < 1e-11
    @test max_divb(h) < 1e-13
end

@testset "CT: dynamic regrid — field loop on a moving blob, ∇·B through regrids [$BE]" begin
    A0 = 1e-3; R0 = 0.12
    Az = (x, y) -> (r = sqrt((mod(x - 0.3 + 0.5, 1.0) - 0.5)^2 + (y - 0.25)^2); r < R0 ? A0 * (R0 - r) : 0.0)
    prof = x -> begin
        r2 = (mod(x[1] - 0.3 + 0.5, 1.0) - 0.5)^2 + (x[2] - 0.25)^2
        (1.0 + exp(-r2 / (2 * 0.06^2)), 2.0, 0.6, 0.0, 1.0, 0.0, 0.0, 0.0)
    end
    h = AMRHierarchy(; nbase = (64, 32), B = 16, backend = BE, cap0 = 4, physics = :mhd, divb = :ct,
                     T = Float64, bc = (:periodic, :reflecting))
    init_base_level!(h); build_level_tables!(h, 0)
    set_ct_ic!(h, prof, Az)
    pol = BlockRefinementPolicy(; dthresh = 1.3, nbuf = 2, every = 2, lmax = 1)
    regrid!(h, pol); set_ct_ic!(h, prof, Az)
    M0 = total_conserved(h)[1]; worst = 0.0; gens = Set{BA.Origin}()
    for n in 1:120
        advance_hierarchy!(h)
        if n % pol.every == 0
            regrid!(h, pol); @test check_nesting(h)
            union!(gens, (h.levels[2].meta[s].origin for s in h.levels[2].live))
            worst = max(worst, max_divb(h))
        end
    end
    @test length(gens) > length(h.levels[2].live)   # blocks were created AND destroyed
    @test worst < 1e-13
    @test abs(total_conserved(h)[1] - M0) / M0 < 1e-12
end

@testset "CT: CP Alfvén wave, 2nd order [$BE]" begin
    A = 0.1
    cp(t) = x -> (1.0, 0.0, -A * sin(2π * (x[1] - t)), -A * cos(2π * (x[1] - t)), 0.1,
                  1.0, A * sin(2π * (x[1] - t)), A * cos(2π * (x[1] - t)))
    Az = (x, y) -> y + A * cos(2π * x) / (2π)              # Bx = 1, By = A sin 2πx
    errs = map((32, 64)) do N
        h = static2d(; nbase = (N, 16), backend = BE, physics = :mhd, divb = :ct, T = Float64)
        set_ct_ic!(h, cp(0.0), Az)
        t = 0.0
        while t < 0.5; t += advance_hierarchy!(h); end
        c = cellmap2(h, 0, :By); dx = BA.level_dx(h, 0)
        sum(abs(v - A * sin(2π * ((g[1] + 0.5) * dx - t))) for (g, v) in c) / length(c)
    end
    @test errs[2] < 2e-3
    @test log2(errs[1] / errs[2]) > 1.7
end

@testset "CT: checkpoint round-trip, bit-identical continuation [$BE]" begin
    st = (1.0, 0.5, 0.0, 0.0, 1.0, 0.3, 0.2, 0.0)
    ha = static2d(; nbase = (64, 64), backend = BE, kids = KIDS_C4, physics = :mhd, divb = :ct,
                  T = Float64, bc = ((:inflow, :outflow), :periodic), inflow = Dict((1, :lo) => st))
    set_ct_ic!(ha, ctot, ctAz)
    for _ in 1:3; advance_hierarchy!(ha); end
    ckf = tempname() * ".jls"
    save_checkpoint(ckf, ha)
    hb, _ = load_checkpoint(ckf; backend = BE)
    @test hb.mhd.ct && hb.mhd == ha.mhd
    for _ in 1:3; advance_hierarchy!(ha); advance_hierarchy!(hb); end
    for l in 0:1, f in (:D, :S1, :Tau, :Bx, :By, :FBx, :FBy)
        @test cellmap2(ha, l, f) == cellmap2(hb, l, f)
    end
    @test max_divb(hb) < 1e-13
    rm(ckf; force = true)
end

@testset "CT: Float32 storage — ∇·B at the f32 floor [$BE]" begin
    h = static2d(; nbase = (64, 64), backend = BE, kids = KIDS_C4, physics = :mhd, divb = :ct)
    set_ct_ic!(h, ctot, ctAz)
    t = 0.0
    while t < 0.2; t += advance_hierarchy!(h); end
    @test max_divb(h) < 1e-5
    @test all(isfinite, Array(h.levels[2].FBx))
end
end # for BE

if :cuda in BACKENDS
@testset "CT: CPU ≡ CUDA (OT + C/F, Float64)" begin
    hs = map((:cpu, :cuda)) do be
        h = static2d(; nbase = (64, 64), backend = be, kids = KIDS_C4, physics = :mhd, divb = :ct,
                     T = Float64)
        set_ct_ic!(h, ctot, ctAz)
        for _ in 1:20; advance_hierarchy!(h); end
        h
    end
    a = composite_rho(hs[1]); b = composite_rho(hs[2])
    @test maximum(abs(b[g] - v) for (g, v) in a) < 1e-9     # f64 fma contraction at shocks
end
end

@testset "CT: 3-D requests error clearly" begin
    @test_throws ErrorException AMRHierarchy(; nbase = (32, 32, 32), physics = :mhd, divb = :ct)
end
