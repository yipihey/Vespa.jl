# test_mhd.jl — GLM-MHD on the block hierarchy (physics = :mhd; the NestedMHD
# scheme — see bench/xcheck_nestedmhd.jl for the scheme cross-check against
# NestedMHD itself, which agrees to the f32 λ rounding, ~2e-8 after 90 steps).
#   * uniform magnetized flow is an exact fixed point across C/F with walls and
#     inflow/outflow faces (reflux registers stay zero);
#   * Orszag–Tang with a static fine patch: M, E, p, ∫B to round-off; ∇·B with
#     AMR stays comparable to the uniform grid (GLM cleaning across C/F);
#   * circularly polarized Alfvén wave vs exact, 2nd-order convergence;
#   * Brio–Wu: y-uniform, x ≡ y transposed, AMR vs uniform-fine;
#   * closed reflecting box MHD blast; dynamic regrid with M/E/∫B conservation;
#   * checkpoint round-trip bit-identical; CPU ≡ GPU; 3-D smoke.
# Requires test_2d.jl's helpers (static2d, cellmap2, KIDS_*, composite_rho).
const BA = BlockAMR

function set_mhd_ic!(hier, l, prof)
    lev = hier.levels[l + 1]; dx = BA.level_dx(hier, l); n = lev.cap * lev.stride
    h = [zeros(Float64, n) for _ in 1:10]; γ = hier.gamma
    Bz = BA._Bax(lev, 3); kg = BA._ngax(lev, 3)
    for s in lev.live
        m = lev.meta[s]; base = (Int(s) - 1) * lev.stride
        for k in 1:Bz, j in 1:lev.B, i in 1:lev.B
            g = ntuple(d -> BA.wrapc(Int128(m.origin[d]) + (i, j, k)[d] - 1, lev.P[d]), 3)
            x = ntuple(d -> (Float64(g[d]) + 0.5) * dx, 3)
            ρ, u, v, w, P, bx, by, bz = prof(x)
            idx = base + ((kg + k - 1) * lev.nd + (lev.ng + j - 1)) * lev.nd + (lev.ng + i - 1) + 1
            ge = P / (γ - 1)
            vals = (ρ, ρ * u, ρ * v, ρ * w, ge + 0.5ρ * (u^2 + v^2 + w^2) + 0.5 * (bx^2 + by^2 + bz^2),
                    ge, bx, by, bz, 0.0)
            for f in 1:10; h[f][idx] = vals[f]; end
        end
    end
    BA.encode_mhd_from_host!(lev, h...)
end
setallm!(h, prof) = for l in 0:length(h.levels)-1
    isempty(h.levels[l + 1].live) || set_mhd_ic!(h, l, prof)
end

const B0OT = 1 / sqrt(4π)
otv(x) = (25 / (36π), -sin(2π * x[2]), sin(2π * x[1]), 0.0, 5 / (12π),
          -B0OT * sin(2π * x[2]), B0OT * sin(4π * x[1]), 0.0)

for BE in BACKENDS
@testset "MHD: uniform magnetized flow is a fixed point (walls, in/outflow, C/F) [$BE]" begin
    st = (1.0, 0.7, 0.0, 0.1, 1.0, 0.5, 0.0, 0.4)    # Bn = 0 on the y walls: reflect-consistent
    h = static2d(; nbase = (64, 32), backend = BE, kids = KIDS_ALL, physics = :mhd,
                 bc = ((:inflow, :outflow), :reflecting), inflow = Dict((1, :lo) => st))
    setallm!(h, x -> st)
    r0 = [cellmap2(h, l, f) for l in 0:1, f in (:D, :S1, :Tau, :Bx, :Bz)]
    for _ in 1:6; advance_hierarchy!(h); end
    worst = 0.0
    for l in 0:1, (fi, f) in enumerate((:D, :S1, :Tau, :Bx, :Bz))
        for (g, v) in cellmap2(h, l, f)
            worst = max(worst, abs(v - r0[l + 1, fi][g]) / abs(r0[l + 1, fi][g]))
        end
    end
    @test worst < 1e-6
    @test maximum(abs, Array(h.levels[2].tabs[:cfreg])) < 1e-6
end

@testset "MHD: Orszag–Tang with AMR — conservation + ∇·B control [$BE]" begin
    hu = static2d(; nbase = (64, 64), backend = BE, physics = :mhd)
    ha = static2d(; nbase = (64, 64), backend = BE, physics = :mhd,
                  kids = (((16, 16), (8, 8)), ((32, 16), (0, 8)), ((16, 32), (8, 0)), ((32, 32), (0, 0))))
    setallm!(hu, otv); setallm!(ha, otv)
    a = total_conserved(ha); ba = total_bflux(ha)
    tu = 0.0; ta = 0.0
    while ta < 0.2; ta += advance_hierarchy!(ha); end
    while tu < 0.2; tu += advance_hierarchy!(hu); end
    b = total_conserved(ha); bb = total_bflux(ha)
    @test abs(b[1] - a[1]) / a[1] < 1e-6
    @test abs(b[5] - a[5]) / a[5] < 1e-6
    @test abs(b[2] - a[2]) < 1e-7 && abs(b[3] - a[3]) < 1e-7
    @test all(abs(bb[d] - ba[d]) < 1e-7 for d in 1:3)
    du = max_divb(hu); da = max_divb(ha)
    @test da < 0.2 && da < 2.5du                        # C/F does not break the cleaning
    @test all(isfinite, Array(ha.levels[2].Psi))
end

@testset "MHD: CP Alfvén wave vs exact, 2nd order (HLLD) [$BE]" begin
    A = 0.1
    cp(t) = x -> (1.0, 0.0, -A * sin(2π * (x[1] - t)), -A * cos(2π * (x[1] - t)), 0.1,
                  1.0, A * sin(2π * (x[1] - t)), A * cos(2π * (x[1] - t)))
    errs = map((32, 64)) do N
        h = static2d(; nbase = (N, 16), backend = BE, physics = :mhd, box = 1.0)
        setallm!(h, cp(0.0))
        t = 0.0
        while t < 0.5; t += advance_hierarchy!(h); end
        c = cellmap2(h, 0, :By); dx = BA.level_dx(h, 0)
        sum(abs(v - A * sin(2π * ((g[1] + 0.5) * dx - t))) for (g, v) in c) / length(c)
    end
    @test errs[2] < 2e-3
    @test log2(errs[1] / errs[2]) > 1.7
end

@testset "MHD: Brio–Wu — y-uniform, x ≡ y, AMR vs uniform-fine [$BE]" begin
    L = (1.0, 0.0, 0.0, 0.0, 1.0, 0.75, 1.0, 0.0); R = (0.125, 0.0, 0.0, 0.0, 0.1, 0.75, -1.0, 0.0)
    bwx = x -> x[1] < 0.5 ? L : R
    # transposed: Bx ↔ By, so the normal field is still 0.75
    tr(W) = (W[1], 0.0, 0.0, 0.0, W[5], W[7], W[6], 0.0)
    bwy = x -> tr(x[2] < 0.5 ? L : R)
    kx = (((48, 0), (0, 0)), ((48, 0), (8, 0)), ((48, 0), (0, 8)), ((48, 0), (8, 8)),
          ((64, 0), (0, 0)), ((64, 0), (8, 0)), ((64, 0), (0, 8)), ((64, 0), (8, 8)))
    ky = map(k -> ((k[1][2], k[1][1]), (k[2][2], k[2][1])), kx)
    run(nbase, kids, bc, prof, box) = (h = static2d(; nbase, kids, bc, gamma = 2.0, backend = BE,
                                                    physics = :mhd, box);
                                       setallm!(h, prof); t = 0.0;
                                       while t < 0.1; t += advance_hierarchy!(h); end; h)
    # transverse faces PERIODIC: the transverse field is normal to them (a reflecting
    # wall would flip it — a conducting wall needs Bn = 0)
    hx = run((128, 16), kx, ((:outflow, :outflow), :periodic), bwx, 1.0)
    hy = run((16, 128), ky, (:periodic, (:outflow, :outflow)), bwy, 1 / 8)
    hr = run((256, 16), (), ((:outflow, :outflow), :periodic), bwx, 1.0)
    cx = composite_rho(hx); cy = composite_rho(hy)
    @test maximum(abs(v - cx[(g[1], 0)]) for (g, v) in cx) < 1e-6
    @test maximum(abs(v - cy[(g[2], g[1])]) for (g, v) in cx) < 1e-5
    # fine region (x ∈ [0.375, 0.625)) vs the uniform 256-cell run at nearly the same time
    rf = cellmap2(hr, 0, :D); f1 = cellmap2(hx, 1, :D)
    @test sum(abs(v - rf[(g[1], 0)]) for (g, v) in f1 if g[2] == 0) / 64 < 0.03
end

@testset "MHD: closed reflecting box blast (W-cycle + C/F + walls) [$BE]" begin
    σ = 0.03
    prof = x -> (1.0, 0.0, 0.0, 0.0,
                 1e-2 + 0.4 * exp(-((x[1] - 0.36)^2 + (x[2] - 0.22)^2) / (2σ^2)) / (2π * σ^2),
                 0.3, 0.2, 0.1)
    h = static2d(; nbase = (64, 32), backend = BE, kids = KIDS_ALL, physics = :mhd,
                 bc = :reflecting)
    setallm!(h, prof)
    M0, _, _, pz0, E0 = total_conserved(h)
    for _ in 1:40; advance_hierarchy!(h); end
    M1, _, _, pz1, E1 = total_conserved(h)
    @test abs(M1 - M0) / M0 < 1e-5
    @test abs(E1 - E0) / E0 < 1e-5
    @test all(isfinite, Array(h.levels[1].D))
end

@testset "MHD: dynamic regrid — magnetized blob, M/E/∫B exact through regrids [$BE]" begin
    σ = 0.05
    prof = x -> begin
        r2 = (mod(x[1] - 0.3 + 0.5, 1.0) - 0.5)^2 + (x[2] - 0.25)^2
        (1.0 + exp(-r2 / (2σ^2)), 2.0, 0.0, 0.0, 1.0, 0.2, 0.1 * exp(-r2 / (2σ^2)), 0.0)
    end
    h = AMRHierarchy(; nbase = (64, 32), B = 16, backend = BE, cap0 = 4, physics = :mhd,
                     bc = (:periodic, :reflecting))
    init_base_level!(h); build_level_tables!(h, 0); set_mhd_ic!(h, 0, prof)
    pol = BlockRefinementPolicy(; dthresh = 1.3, nbuf = 2, every = 2, lmax = 1)
    regrid!(h, pol); setallm!(h, prof)
    @test !isempty(h.levels[2].live)
    M0, _, _, _, E0 = total_conserved(h); B0 = total_bflux(h)
    gens = Set{BA.Origin}()
    for n in 1:40
        advance_hierarchy!(h)
        if n % pol.every == 0
            regrid!(h, pol); @test check_nesting(h)
            union!(gens, (h.levels[2].meta[s].origin for s in h.levels[2].live))
        end
    end
    M1, _, _, _, E1 = total_conserved(h); B1 = total_bflux(h)
    @test abs(M1 - M0) / M0 < 1e-5 && abs(E1 - E0) / E0 < 1e-5
    @test abs(B1[1] - B0[1]) < 1e-6 && abs(B1[2] - B0[2]) < 1e-6
    @test length(gens) > length(h.levels[2].live)      # the patch moved with the blob
end

@testset "MHD: checkpoint round-trip, bit-identical continuation [$BE]" begin
    st = (1.0, 0.5, 0.0, 0.0, 1.0, 0.3, 0.2, 0.0)
    prof = x -> (1.0 + 0.5 * exp(-((x[1] - 0.45)^2 + (x[2] - 0.25)^2) / 0.002),
                 0.5, 0.0, 0.0, 1.0, 0.3, 0.2, 0.0)
    ha = static2d(; nbase = (64, 32), backend = BE, kids = KIDS_ALL, physics = :mhd,
                  bc = ((:inflow, :outflow), :periodic), inflow = Dict((1, :lo) => st))
    setallm!(ha, prof)
    for _ in 1:3; advance_hierarchy!(ha); end
    ckf = tempname() * ".jls"
    save_checkpoint(ckf, ha)
    hb, _ = load_checkpoint(ckf; backend = BE)
    @test hb.physics === :mhd && hb.mhd == ha.mhd && hb.inflow == ha.inflow
    for _ in 1:3; advance_hierarchy!(ha); advance_hierarchy!(hb); end
    for l in 0:1, f in (:D, :S1, :Tau, :Bx, :By, :Psi)
        @test cellmap2(ha, l, f) == cellmap2(hb, l, f)
    end
    rm(ckf; force = true)
end

@testset "MHD: 3-D smoke — centre patch, periodic, conservation [$BE]" begin
    h = AMRHierarchy(; nbase = (32, 32, 32), B = 16, backend = BE, physics = :mhd)
    lev0 = init_base_level!(h)
    for cz in 0:1, cy in 0:1, cx in 0:1
        add_block!(h, 1, lev0.byorigin[(UInt128(16cx), UInt128(16cy), UInt128(16cz))],
                   (cx == 0 ? 8 : 0, cy == 0 ? 8 : 0, cz == 0 ? 8 : 0))
    end
    build_level_tables!(h, 0); build_level_tables!(h, 1); build_cf_register!(h, 1)
    prof = x -> (1.0 + 0.2 * sin(2π * x[1]) * cos(2π * x[3]), sin(2π * x[2]), -sin(2π * x[1]),
                 0.3 * sin(2π * x[3]), 1.0, 0.2 * cos(2π * x[2]), 0.2 * sin(2π * x[3]), 0.2)
    setallm!(h, prof)
    a = total_conserved(h); ba = total_bflux(h)
    for _ in 1:6; advance_hierarchy!(h); end
    b = total_conserved(h); bb = total_bflux(h)
    @test abs(b[1] - a[1]) / a[1] < 1e-5 && abs(b[5] - a[5]) / a[5] < 1e-5
    @test all(abs(bb[d] - ba[d]) < 1e-6 for d in 1:3)
end
end # for BE

if :cuda in BACKENDS
@testset "MHD: CPU ≡ CUDA (OT + C/F + walls)" begin
    hs = map((:cpu, :cuda)) do be
        h = static2d(; nbase = (64, 32), backend = be, kids = KIDS_ALL, physics = :mhd,
                     bc = ((:reflecting, :outflow), :reflecting))
        setallm!(h, x -> (otv(x)[1:4]..., otv(x)[5] + 0.1, otv(x)[6:8]...))
        for _ in 1:20; advance_hierarchy!(h); end
        h
    end
    a = composite_rho(hs[1]); b = composite_rho(hs[2])
    @test sum(abs(b[g] - v) for (g, v) in a) / sum(values(a)) < 1e-5
end
end

@testset "MHD: unsupported combinations error clearly" begin
    @test_throws ErrorException AMRHierarchy(; nbase = (32, 32), physics = :mhd, T = Float16)
    @test_throws ErrorException AMRHierarchy(; nbase = (32, 32), physics = :mhd, nsp = 1)
    h = AMRHierarchy(; nbase = (32, 32), physics = :mhd)
    init_base_level!(h); build_level_tables!(h, 0)
    @test_throws ErrorException hierarchy_rk2_step!(h, 1e-3)
end
