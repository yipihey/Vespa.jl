# test_2d.jl — 2-D block AMR (B×B×1 blocks, flat z) and non-periodic domain
# faces (outflow / reflecting / inflow on any face), on every backend.
#   * BC ghost fill is EXACT per rule, incl. the x-then-y corner composition;
#   * uniform flow is a fixed point with fine blocks touching walls and
#     inflow/outflow faces (RK2 + CTU, W-cycle) — reflux registers stay zero;
#   * closed reflecting box: blast crossing C/F and hitting walls conserves
#     mass + energy to f32 round-off (RK2 + CTU, W-cycle);
#   * elongated Sod (128×16) vs the exact Riemann solution; Sod along x ≡ Sod
#     along y (transposed box) bitwise; 1-D solution stays y-uniform;
#   * AMR fine region matches a uniform-fine 2-D reference;
#   * dynamic regrid: a blob advected out through an outflow face (no wrap
#     artifacts at the inflow face, fine level vanishes after it leaves);
#     KH band in a periodic-x / reflecting-y box (mass, energy, x-momentum);
#   * f16 storage tracks the f32 twin; checkpoint round-trip continues
#     bit-identically; CPU ≡ GPU; a 3-D smoke of the same BCs.
const BA = BlockAMR

# active-cell map (global level (i,j)) → physical value of field `f`
function cellmap2(h, l, f)
    lev = h.levels[l + 1]; a = Array(getfield(lev, f))
    sc = f === :D ? Array(lev.Dsc) : f in (:S1, :S2, :S3) ? Array(lev.Ssc) :
         f === :Tau ? Array(lev.Esc) : Array(lev.Gsc)
    out = Dict{NTuple{2,Int},Float64}()
    for s in lev.live
        m = lev.meta[s]; base = (Int(s) - 1) * lev.stride
        for j in 1:lev.B, i in 1:lev.B
            idx = base + (lev.ng + j - 1) * lev.nd + (lev.ng + i - 1) + 1
            out[(Int(m.origin[1]) + i - 1, Int(m.origin[2]) + j - 1)] = Float64(a[idx]) * sc[s]
        end
    end
    return out
end

# static 2-level 2-D hierarchy; `kids` = [(parent base-block origin (x,y), offset (ox,oy))]
function static2d(; nbase, B = 16, bc = :periodic, inflow = Dict(), scheme = :rk2,
                  T = Float32, gamma = 5/3, kids = (), backend = :cpu, box = 1.0,
                  physics = :hydro, riemann = :hlld)
    hier = AMRHierarchy(; nbase, B, backend, bc, inflow, scheme, T, gamma, box, physics, riemann)
    lev0 = init_base_level!(hier)
    for (po, off) in kids
        p = lev0.byorigin[(UInt128(po[1]), UInt128(po[2]), UInt128(0))]
        add_block!(hier, 1, p, (off[1], off[2], 0))
    end
    @test check_nesting(hier)
    build_level_tables!(hier, 0)
    if !isempty(kids)
        build_level_tables!(hier, 1); build_cf_register!(hier, 1)
    end
    return hier
end

# fine patch over parent cells [24,40)×[8,24) of a (64,32) base, plus two
# corner patches TOUCHING the domain faces ([0,8)² and [56,64)×[24,32))
const KIDS_CENTER = (((16, 0), (8, 8)), ((32, 0), (0, 8)), ((16, 16), (8, 0)), ((32, 16), (0, 0)))
const KIDS_CORNERS = (((0, 0), (0, 0)), ((48, 16), (8, 8)))
const KIDS_ALL = (KIDS_CENTER..., KIDS_CORNERS...)

setall!(h, prof) = for l in 0:length(h.levels)-1
    isempty(h.levels[l + 1].live) || set_ic!(h, l, prof)
end

# ── exact Riemann solver (Toro ch. 4) for the Sod gates ──────────────────────
function _fK(p, ρK, pK, γ)
    AK = 2 / ((γ + 1) * ρK); BK = (γ - 1) / (γ + 1) * pK; cK = sqrt(γ * pK / ρK)
    p > pK ? ((p - pK) * sqrt(AK / (p + BK)), sqrt(AK / (BK + p)) * (1 - (p - pK) / (2(BK + p)))) :
             (2cK / (γ - 1) * ((p / pK)^((γ - 1) / (2γ)) - 1), (p / pK)^(-(γ + 1) / (2γ)) / (ρK * cK))
end
function riemann_exact(ξ, L, R, γ)                  # ξ = (x − x0)/t; L, R = (ρ, u, p)
    ρL, uL, pL = L; ρR, uR, pR = R
    cL = sqrt(γ * pL / ρL); cR = sqrt(γ * pR / ρR)
    p = 0.5(pL + pR)
    for _ in 1:100
        fL, dL = _fK(p, ρL, pL, γ); fR, dR = _fK(p, ρR, pR, γ)
        pn = max(p - (fL + fR + uR - uL) / (dL + dR), 1e-12)
        done = abs(pn - p) < 1e-14p; p = pn; done && break
    end
    fL, _ = _fK(p, ρL, pL, γ); fR, _ = _fK(p, ρR, pR, γ)
    u = 0.5(uL + uR) + 0.5(fR - fL); g1 = (γ - 1) / (γ + 1)
    if ξ <= u
        if p > pL
            SL = uL - cL * sqrt((γ + 1) / (2γ) * p / pL + (γ - 1) / (2γ))
            return ξ < SL ? (ρL, uL, pL) : (ρL * (p / pL + g1) / (g1 * p / pL + 1), u, p)
        end
        cs = cL * (p / pL)^((γ - 1) / (2γ))
        ξ < uL - cL && return (ρL, uL, pL)
        ξ > u - cs && return (ρL * (p / pL)^(1 / γ), u, p)
        cf = 2 / (γ + 1) * (cL + (γ - 1) / 2 * (uL - ξ))
        return (ρL * (cf / cL)^(2 / (γ - 1)), 2 / (γ + 1) * (cL + (γ - 1) / 2 * uL + ξ),
                pL * (cf / cL)^(2γ / (γ - 1)))
    else
        if p > pR
            SR = uR + cR * sqrt((γ + 1) / (2γ) * p / pR + (γ - 1) / (2γ))
            return ξ > SR ? (ρR, uR, pR) : (ρR * (p / pR + g1) / (g1 * p / pR + 1), u, p)
        end
        cs = cR * (p / pR)^((γ - 1) / (2γ))
        ξ > uR + cR && return (ρR, uR, pR)
        ξ < u + cs && return (ρR * (p / pR)^(1 / γ), u, p)
        cf = 2 / (γ + 1) * (cR - (γ - 1) / 2 * (uR - ξ))
        return (ρR * (cf / cR)^(2 / (γ - 1)), 2 / (γ + 1) * (-cR + (γ - 1) / 2 * uR + ξ),
                pR * (cf / cR)^(2γ / (γ - 1)))
    end
end

# composite ρ map at the finest resolution present (level-0 cells not covered by
# level 1 are expanded 2×2) — keyed on level-1 cell indices
function composite_rho(h)
    c0 = cellmap2(h, 0, :D)
    c1 = length(h.levels) > 1 ? cellmap2(h, 1, :D) : Dict{NTuple{2,Int},Float64}()
    out = copy(c1)
    for ((i, j), v) in c0, b in 0:1, a in 0:1
        g = (2i + a, 2j + b)
        haskey(out, g) || (out[g] = v)
    end
    return out
end

for BE in BACKENDS
@testset "2-D BC ghost fill is exact per rule (corners compose x→y) [$BE]" begin
    γ = 5/3
    combos = (((:outflow, :reflecting), (:inflow, :outflow)),
              ((:reflecting, :inflow), :periodic),
              (:periodic, (:reflecting, :inflow)),
              ((:inflow, :inflow), (:reflecting, :reflecting)))
    instate = Dict((1, :lo) => (1.5, 0.7, -0.2, 0.1, 2.0), (1, :hi) => (0.8, -0.3, 0.4, 0.0, 0.5),
                   (2, :lo) => (1.2, 0.1, 0.9, -0.1, 1.1), (2, :hi) => (0.6, 0.0, -0.5, 0.2, 0.9))
    for bc in combos
        hier = AMRHierarchy(; nbase = (32, 48), B = 16, backend = BE, bc, gamma = γ,
                            inflow = instate)
        lev = init_base_level!(hier); build_level_tables!(hier, 0)
        N = (32, 48)
        fval(f, gx, gy) = f == 1 ? 1 + 0.001 * (gx + 37gy) : f == 2 ? 0.01gx - 0.2 :
                          f == 3 ? 0.02gy + 0.1 : f == 4 ? 0.03 :
                          f == 5 ? 3 + 0.001gx : 2 + 0.002gy
        n = lev.cap * lev.stride
        h = [zeros(Float64, n) for _ in 1:6]
        for s in lev.live, j in 0:15, i in 0:15
            m = lev.meta[s]; gx = Int(m.origin[1]) + i; gy = Int(m.origin[2]) + j
            idx = (Int(s) - 1) * lev.stride + (j + 2) * lev.nd + (i + 2) + 1
            for f in 1:6; h[f][idx] = fval(f, gx, gy); end
        end
        BA.encode_from_host!(lev, h...)
        fill_ghosts!(hier, 0)
        inval(ax, side, f) = begin
            ρ, u, v, w, P = hier.inflow[ax][side]
            ge = P / (γ - 1)
            (ρ, ρ * u, ρ * v, ρ * w, ge + 0.5ρ * (u^2 + v^2 + w^2), ge)[f]
        end
        # expected ghost value, x rule inside, y rule outside (fill order)
        function ex_x(f, gx, gy)
            if 0 <= gx < N[1] || hier.bc[1][1] === :periodic
                return fval(f, mod(gx, N[1]), gy)
            end
            side = gx < 0 ? 1 : 2; k = hier.bc[1][side]
            k === :inflow && return inval(1, side, f)
            sx = k === :outflow ? (gx < 0 ? 0 : N[1] - 1) : (gx < 0 ? -1 - gx : 2N[1] - 1 - gx)
            return (k === :reflecting && f == 2 ? -1 : 1) * fval(f, sx, gy)
        end
        function ex(f, gx, gy)
            if 0 <= gy < N[2] || hier.bc[2][1] === :periodic
                return ex_x(f, gx, mod(gy, N[2]))
            end
            side = gy < 0 ? 1 : 2; k = hier.bc[2][side]
            k === :inflow && return inval(2, side, f)
            sy = k === :outflow ? (gy < 0 ? 0 : N[2] - 1) : (gy < 0 ? -1 - gy : 2N[2] - 1 - gy)
            return (k === :reflecting && f == 3 ? -1 : 1) * ex_x(f, gx, sy)
        end
        worst = 0.0
        H = [Array(getfield(lev, F)) for F in (:D, :S1, :S2, :S3, :Tau, :Ge)]
        for s in lev.live, j in 0:lev.nd-1, i in 0:lev.nd-1
            (2 <= i < 18 && 2 <= j < 18) && continue           # active
            m = lev.meta[s]; gx = Int(m.origin[1]) + i - 2; gy = Int(m.origin[2]) + j - 2
            idx = (Int(s) - 1) * lev.stride + j * lev.nd + i + 1
            for f in 1:6
                worst = max(worst, abs(Float64(H[f][idx]) - ex(f, gx, gy)) /
                                   max(abs(ex(f, gx, gy)), 1.0))
            end
        end
        @test worst < 1e-6
    end
end

@testset "2-D uniform flow is a fixed point with walls + inflow/outflow [$BE]" begin
    st = (1.0, 0.7, 0.0, 0.0, 1.0)                   # flow parallel to the y walls
    for scheme in (:rk2, :ctu)
        hier = static2d(; nbase = (64, 32), backend = BE, scheme, kids = KIDS_ALL,
                        bc = ((:inflow, :outflow), :reflecting),
                        inflow = Dict((1, :lo) => st))
        setall!(hier, x -> st)
        r0 = [cellmap2(hier, l, f) for l in 0:1, f in (:D, :S1, :Tau)]
        for _ in 1:6
            advance_hierarchy!(hier)
        end
        worst = 0.0
        for l in 0:1, (fi, f) in enumerate((:D, :S1, :Tau))
            c = cellmap2(hier, l, f)
            for (g, v) in c
                worst = max(worst, abs(v - r0[l + 1, fi][g]) / abs(r0[l + 1, fi][g]))
            end
        end
        @test worst < 1e-6
        @test maximum(abs, Array(hier.levels[2].tabs[:cfreg])) < 1e-6
        @test hier.nstep[2] == 2hier.nstep[1]
    end
end

@testset "2-D closed reflecting box: blast conserves M, E (RK2 + CTU) [$BE]" begin
    σ = 0.03
    prof = x -> begin
        r2 = (x[1] - 0.36)^2 + (x[2] - 0.22)^2
        (1.0, 0.0, 0.0, 0.0, 1e-3 + 0.4 * exp(-r2 / (2σ^2)) / (2π * σ^2))
    end
    for scheme in (:rk2, :ctu)
        hier = static2d(; nbase = (64, 32), backend = BE, scheme, kids = KIDS_ALL,
                        bc = :reflecting)
        setall!(hier, prof)
        M0, px0, py0, pz0, E0 = total_conserved(hier)
        for _ in 1:50
            advance_hierarchy!(hier)
        end
        M1, px1, py1, pz1, E1 = total_conserved(hier)
        @test abs(M1 - M0) / M0 < 3e-5
        @test abs(E1 - E0) / E0 < 3e-5
        @test pz1 == 0                               # flat z: no z momentum ever
        # the blast has reached the walls (the BCs carried real load)
        c0 = cellmap2(hier, 0, :Tau)
        @test maximum(v for ((i, j), v) in c0 if j == 0) > 2 * (1e-3 / (2 / 3))
    end
end

@testset "2-D elongated Sod (128×16) vs exact; x ≡ y transposed [$BE]" begin
    γ = 1.4; L = (1.0, 0.0, 1.0); R = (0.125, 0.0, 0.1); tend = 0.2
    # fine level over x ∈ [0.375, 0.625): touches both reflecting y walls
    kx = (((48, 0), (0, 0)), ((48, 0), (8, 0)), ((48, 0), (0, 8)), ((48, 0), (8, 8)),
          ((64, 0), (0, 0)), ((64, 0), (8, 0)), ((64, 0), (0, 8)), ((64, 0), (8, 8)))
    ky = map(k -> ((k[1][2], k[1][1]), (k[2][2], k[2][1])), kx)
    run(nbase, kids, bc, ax) = begin                # box = x length; unit-length Sod axis
        h = static2d(; nbase, kids, bc, gamma = γ, backend = BE, box = ax == 1 ? 1.0 : 1 / 8)
        prof = x -> (W = x[ax] < 0.5 ? L : R; (W[1], 0.0, 0.0, 0.0, W[3]))
        setall!(h, prof)
        t = 0.0
        while t < tend - 1e-12
            dt = min(compute_dt(h), tend - t)
            hierarchy_rk2_step!(h, dt); t += dt
        end
        h
    end
    hx = run((128, 16), kx, ((:outflow, :outflow), :reflecting), 1)
    hy = run((16, 128), ky, (:reflecting, (:outflow, :outflow)), 2)
    cx = composite_rho(hx); cy = composite_rho(hy)
    # 1-D solution stays exactly y-uniform (walls + C/F included)
    @test maximum(abs(v - cx[(g[1], 0)]) for (g, v) in cx) < 1e-6
    # Sod along y ≡ Sod along x (transposed), cell by cell (bitwise on CPU;
    # GPU fma contraction differs per axis path → ulp-level)
    @test maximum(abs(v - cy[(g[2], g[1])]) for (g, v) in cx) < 1e-5
    # vs the exact solution (fine-cell centers on the composite grid)
    dxf = BA.level_dx(hx, 1)
    l1 = 0.0
    for i in 0:255
        x = (i + 0.5) * dxf
        l1 += abs(cx[(i, 0)] - riemann_exact((x - 0.5) / tend, L, R, γ)[1]) * dxf
    end
    @test l1 < 0.012
end

@testset "2-D AMR fine region matches uniform-fine reference [$BE]" begin
    σ = 0.03
    prof = x -> begin
        r2 = (x[1] - 0.5)^2 + (x[2] - 0.25)^2
        (1.0, 0.0, 0.0, 0.0, 1e-5 + 0.3 * exp(-r2 / (2σ^2)) / (2π * σ^2))
    end
    hier = static2d(; nbase = (64, 32), backend = BE, kids = KIDS_CENTER)
    setall!(hier, prof)
    ref = static2d(; nbase = (128, 64), backend = BE)
    set_ic!(ref, 0, prof)
    for _ in 1:25
        dt = compute_dt(hier)
        hierarchy_rk2_step!(hier, dt); hierarchy_rk2_step!(ref, dt)
    end
    amr = cellmap2(hier, 1, :D); rf = cellmap2(ref, 0, :D)
    @test length(amr) == 32^2
    @test sum(abs(v - rf[g]) for (g, v) in amr) / length(amr) < 3e-3
end

@testset "2-D regrid: blob exits through the outflow face [$BE]" begin
    st = (1.0, 2.0, 0.0, 0.0, 1.0)                  # supersonic (M ≈ 1.55) inflow
    hier = AMRHierarchy(; nbase = (64, 32), B = 16, backend = BE, cap0 = 4,
                        bc = ((:inflow, :outflow), :periodic), inflow = Dict((1, :lo) => st))
    init_base_level!(hier); build_level_tables!(hier, 0)
    σ = 0.04
    prof = x -> (1.0 + exp(-((x[1] - 0.3)^2 + (x[2] - 0.25)^2) / (2σ^2)), 2.0, 0.0, 0.0, 1.0)
    set_ic!(hier, 0, prof)
    pol = BlockRefinementPolicy(; dthresh = 1.2, nbuf = 2, every = 2, lmax = 1)
    regrid!(hier, pol); setall!(hier, prof)
    @test !isempty(hier.levels[2].live)
    M0 = total_conserved(hier)[1]
    t = 0.0; nreg = 0; bad = 0; maxpeak = 0.0
    while t < 0.6
        t += advance_hierarchy!(hier)
        if hier.nstep[1] % pol.every == 0
            regrid!(hier, pol); nreg += 1
            @test check_nesting(hier)
            # no refinement may appear next to the INFLOW face (a wrap artifact)
            # once the blob has moved on (it started at x = 0.3)
            for s in hier.levels[2].live
                x = Float64(hier.levels[2].meta[s].origin[1]) * BA.level_dx(hier, 1)
                (t > 0.1 && x < 0.15) && (bad += 1)
            end
        end
        t < 0.3 && (maxpeak = max(maxpeak, maximum(values(cellmap2(hier, 1, :D)))))
    end
    @test nreg >= 20 && bad == 0
    @test maxpeak > 1.5                              # tracked on the fine level
    @test isempty(hier.levels[2].live)              # gone: the blob left the box
    c0 = cellmap2(hier, 0, :D)
    @test maximum(abs(v - 1.0) for v in values(c0)) < 1e-3   # back to the inflow state
    @test abs(total_conserved(hier)[1] - 0.5) < 1e-4         # = ρ_in × area
end

@testset "2-D KH band, periodic x + reflecting y, dynamic regrid (RK2 + CTU) [$BE]" begin
    prof = x -> begin
        s = 0.5 * (tanh((x[2] - 0.19) / 0.01) - tanh((x[2] - 0.31) / 0.01))
        (1.0 + s, -0.5 + s, 0.02 * sin(4π * x[1]), 0.0, 2.5)
    end
    for scheme in (:rk2, :ctu)
        hier = AMRHierarchy(; nbase = (64, 32), B = 16, backend = BE, cap0 = 4, scheme,
                            bc = (:periodic, :reflecting))
        init_base_level!(hier); build_level_tables!(hier, 0)
        set_ic!(hier, 0, prof)
        pol = BlockRefinementPolicy(; dthresh = 1.3, nbuf = 2, every = 2, lmax = 1)
        regrid!(hier, pol); setall!(hier, prof)
        @test !isempty(hier.levels[2].live)
        M0, px0, _, _, E0 = total_conserved(hier)
        for n in 1:40
            advance_hierarchy!(hier)
            if n % pol.every == 0
                regrid!(hier, pol)
                @test check_nesting(hier)
            end
        end
        M1, px1, _, _, E1 = total_conserved(hier)
        @test abs(M1 - M0) / M0 < 3e-5
        @test abs(E1 - E0) / E0 < 3e-5
        @test abs(px1 - px0) < 3e-5 * M0             # walls exert no x force
        @test !isempty(hier.levels[2].live)
    end
end

@testset "2-D f16 storage tracks the f32 twin [$BE]" begin
    σ = 0.03
    prof = x -> begin
        r2 = (x[1] - 0.36)^2 + (x[2] - 0.22)^2
        (1.0, 0.0, 0.0, 0.0, 1e-3 + 0.4 * exp(-r2 / (2σ^2)) / (2π * σ^2))
    end
    mk(T) = (h = static2d(; nbase = (64, 32), backend = BE, kids = KIDS_ALL, T,
                          bc = ((:reflecting, :outflow), :reflecting));
             setall!(h, prof); h)
    h32 = mk(Float32); h16 = mk(Float16)
    for _ in 1:30
        advance_hierarchy!(h32); advance_hierarchy!(h16)
        for l in 0:1; BA.update_scales!(h16, l); end
    end
    a = composite_rho(h32); b = composite_rho(h16)
    @test sum(abs(b[g] - v) for (g, v) in a) / sum(values(a)) < 1e-2
end

@testset "2-D checkpoint round-trip with inflow BCs [$BE]" begin
    st = (1.0, 2.0, 0.0, 0.0, 1.0)
    σ = 0.04
    prof = x -> (1.0 + exp(-((x[1] - 0.45)^2 + (x[2] - 0.25)^2) / (2σ^2)), 2.0, 0.0, 0.0, 1.0)
    ha = static2d(; nbase = (64, 32), backend = BE, kids = KIDS_ALL, scheme = :ctu,
                  bc = ((:inflow, :outflow), :reflecting),
                  inflow = Dict((1, :lo) => st))
    setall!(ha, prof)
    for _ in 1:3; advance_hierarchy!(ha); end
    ckf = tempname() * ".jls"
    save_checkpoint(ckf, ha)
    hb, _ = load_checkpoint(ckf; backend = BE)
    @test hb.dim == 2 && hb.bc == ha.bc && hb.inflow == ha.inflow && hb.nbase == ha.nbase
    for _ in 1:3
        advance_hierarchy!(ha); advance_hierarchy!(hb)
    end
    for l in 0:1, f in (:D, :S1, :S2, :Tau, :Ge)
        @test cellmap2(ha, l, f) == cellmap2(hb, l, f)
    end
    rm(ckf; force = true)
end

if BE != :cpu
@testset "2-D CTU: tiled (16×16) ≡ per-cell [$BE]" begin
    σ = 0.03
    prof = x -> begin
        r2 = (x[1] - 0.36)^2 + (x[2] - 0.22)^2
        (1.0, 0.2, -0.1, 0.05, 1e-3 + 0.4 * exp(-r2 / (2σ^2)) / (2π * σ^2))
    end
    mk() = begin
        h = AMRHierarchy(; nbase = (64, 32), B = 16, backend = BE, nsp = 2, scheme = :ctu,
                         bc = ((:inflow, :outflow), :reflecting),
                         inflow = Dict((1, :lo) => (1.0, 0.2, -0.1, 0.05, 1e-3)))
        lev0 = init_base_level!(h)
        for (po, off) in KIDS_ALL
            add_block!(h, 1, lev0.byorigin[(UInt128(po[1]), UInt128(po[2]), UInt128(0))],
                       (off[1], off[2], 0))
        end
        build_level_tables!(h, 0); build_level_tables!(h, 1); build_cf_register!(h, 1)
        setall!(h, prof)
        for l in 0:1, (q, c) in enumerate((1.0f-4, 3.0f-6))
            fill!(h.levels[l + 1].sp[q], BA.ChemistryKernels.encode_log2sp(c))
        end
        h
    end
    ha, hb = mk(), mk()
    for h in (ha, hb), l in 0:1
        BA.fill_ghosts!(h, l; θ = 0.0f0, buf = :R)
    end
    for l in 0:1
        ctu_level!(ha, l, 0.15f0; tiled = true)
        ctu_level!(hb, l, 0.15f0; tiled = false)
    end
    # same inline chain on the same inputs; ptxas fma contraction differs per
    # kernel structure → ulp-level (see the 3-D twin in test_ctu.jl)
    reldiff(A, Bv) = maximum(abs.(Float32.(A) .- Float32.(Bv)) ./ max.(abs.(Float32.(Bv)), 1.0f-6))
    for l in 0:1
        la, lb = ha.levels[l + 1], hb.levels[l + 1]
        for f in (:Do, :S1o, :S2o, :S3o, :Tauo, :Geo)
            @test reldiff(Array(getfield(la, f)), Array(getfield(lb, f))) < 1e-5
        end
        @test maximum(abs.(Int.(Array(la.spo[1])) .- Int.(Array(lb.spo[1])))) <= 1
    end
end
end # if !cpu
end # for BE

if :cuda in BACKENDS
@testset "2-D CPU ≡ CUDA (W-cycle, walls + C/F, RK2 + CTU)" begin
    σ = 0.03
    prof = x -> begin
        r2 = (x[1] - 0.36)^2 + (x[2] - 0.22)^2
        (1.0, 0.1, -0.05, 0.0, 1e-3 + 0.4 * exp(-r2 / (2σ^2)) / (2π * σ^2))
    end
    for scheme in (:rk2, :ctu)
        hs = map((:cpu, :cuda)) do be
            h = static2d(; nbase = (64, 32), backend = be, kids = KIDS_ALL, scheme,
                         bc = ((:reflecting, :outflow), :reflecting))
            setall!(h, prof)
            for _ in 1:20; advance_hierarchy!(h); end
            h
        end
        a = composite_rho(hs[1]); b = composite_rho(hs[2])
        # f32 round-off (GPU fma contraction) amplified only at shock fronts
        @test sum(abs(b[g] - v) for (g, v) in a) / sum(values(a)) < 1e-5
        @test maximum(abs(b[g] - v) / v for (g, v) in a) < 1e-2
    end
end
end

for BE in BACKENDS
@testset "3-D domain BCs: uniform inflow/outflow + reflecting-box blast [$BE]" begin
    mk(bc, inflow) = begin
        h = AMRHierarchy(; nbase = (32, 32, 32), B = 16, backend = BE, bc, inflow)
        lev0 = init_base_level!(h)
        for cz in 0:1, cy in 0:1, cx in 0:1          # centre patch (8 children)
            p = lev0.byorigin[(UInt128(16cx), UInt128(16cy), UInt128(16cz))]
            add_block!(h, 1, p, (cx == 0 ? 8 : 0, cy == 0 ? 8 : 0, cz == 0 ? 8 : 0))
        end
        @test check_nesting(h)
        build_level_tables!(h, 0); build_level_tables!(h, 1); build_cf_register!(h, 1)
        h
    end
    st = (1.0, 0.6, 0.0, 0.3, 1.0)
    h = mk(((:inflow, :outflow), :reflecting, :periodic), Dict((1, :lo) => st))
    setall!(h, x -> st)
    for _ in 1:3; advance_hierarchy!(h); end
    worst = 0.0
    for l in 0:1                                     # every ACTIVE cell still == st
        lev = h.levels[l + 1]; hD = Array(lev.D); hS = Array(lev.S1)
        for s in lev.live, k in 1:16, j in 1:16, i in 1:16
            idx = (Int(s) - 1) * lev.stride + ((k + 1) * lev.nd + (j + 1)) * lev.nd + (i + 1) + 1
            worst = max(worst, abs(hD[idx] - 1.0), abs(hS[idx] - 0.6))
        end
    end
    @test worst < 1e-6
    @test maximum(abs, Array(h.levels[2].tabs[:cfreg])) < 1e-6
    σ = 2 / 64
    hb = mk(:reflecting, Dict())
    setall!(hb, x -> (1.0, 0.0, 0.0, 0.0,
                      1e-3 + 0.05 * exp(-sum((x .- (0.4, 0.45, 0.5)) .^ 2) / (2σ^2)) / σ^3))
    M0, _, _, _, E0 = total_conserved(hb)
    for _ in 1:15; advance_hierarchy!(hb); end
    M1, _, _, _, E1 = total_conserved(hb)
    @test abs(M1 - M0) / M0 < 3e-5 && abs(E1 - E0) / E0 < 3e-5
end
end # for BE
