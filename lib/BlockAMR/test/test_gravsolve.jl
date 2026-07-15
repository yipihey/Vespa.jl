# test_gravsolve.jl — native level Poisson: manufactured solution
# φ* = cos2πx·cos2πy·cos2πz, source ρ = ∇²φ*/coef = −12π²φ*/coef.  Level 0 holds
# the exact φ*; levels 1–2 solve with trilinearly-prolonged Dirichlet BCs and
# batched red-black sweeps over the block union.  Gates: converges to φ* within
# discretization+BC error; residual drops orders of magnitude; warm restart is
# nearly free; the pool kick matches −∇φ*.
const BA = BlockAMR

for BE in BACKENDS
@testset "level Poisson vs manufactured solution [$BE]" begin
    coef = 1.0
    φex(x) = cos(2π*x[1]) * cos(2π*x[2]) * cos(2π*x[3])
    ρex(x) = -12π^2 * φex(x) / coef
    hier = center_refined(; backend = BE, T = Float32)
    lev1 = hier.levels[2]
    s1 = lev1.live[1]
    add_block!(hier, 2, s1, (4, 4, 4))                    # a level-2 block inside
    for l in 0:2
        build_level_tables!(hier, l)
    end
    @test check_nesting(hier)
    # stage: D = ρex on every level (through the scale path), φ* on level 0
    for l in 0:2
        lev = hier.levels[l + 1]
        n = lev.cap * lev.stride
        hD = zeros(n); hz = zeros(n); hone = fill(1.0, n)
        dx = BA.level_dx(hier, l)
        for s in lev.live
            m = lev.meta[s]; base = (Int(s)-1)*lev.stride
            for k in 1:lev.B, j in 1:lev.B, i in 1:lev.B
                x = ntuple(d -> (Float64(m.origin[d]) + (i,j,k)[d] - 0.5) * dx, 3)
                hD[base + ((lev.ng+k-1)*lev.nd + (lev.ng+j-1))*lev.nd + (lev.ng+i-1) + 1] = ρex(x)
            end
        end
        BA.encode_from_host!(lev, hD, hz, hz, hz, hone, hone)
    end
    lev0 = hier.levels[1]
    hphi = zeros(Float32, lev0.cap * lev0.stride)
    for s in lev0.live
        m = lev0.meta[s]; base = (Int(s)-1)*lev0.stride
        for k in 1:lev0.nd, j in 1:lev0.nd, i in 1:lev0.nd
            g = ntuple(d -> (Float64(m.origin[d]) + (i,j,k)[d] - 1 - lev0.ng + 0.5) / 32, 3)
            hphi[base + ((k-1)*lev0.nd + (j-1))*lev0.nd + (i-1) + 1] = Float32(φex(g))
        end
    end
    copyto!(lev0.phi, hphi)
    # solve level 1 then level 2 (top-down), cold start from parent
    r1a = solve_gravity_level!(hier, 1; source_coef = coef, nsweep = 2, init = :parent)
    r1b = solve_gravity_level!(hier, 1; source_coef = coef, nsweep = 400)
    @test r1b < 1e-3 * max(r1a, 1e-20)                    # residual collapsed
    r2 = solve_gravity_level!(hier, 2; source_coef = coef, nsweep = 400, init = :parent)
    # φ accuracy on level 1 (interior; discretization + BC-interp limited)
    for (l, tol) in ((1, 0.02), (2, 0.02))
        lev = hier.levels[l + 1]
        hp = Array(lev.phi); dx = BA.level_dx(hier, l)
        worst = 0.0
        for s in lev.live
            m = lev.meta[s]; base = (Int(s)-1)*lev.stride
            for k in 1:lev.B, j in 1:lev.B, i in 1:lev.B
                x = ntuple(d -> (Float64(m.origin[d]) + (i,j,k)[d] - 0.5) * dx, 3)
                got = hp[base + ((lev.ng+k-1)*lev.nd + (lev.ng+j-1))*lev.nd + (lev.ng+i-1) + 1]
                worst = max(worst, abs(got - φex(x)))
            end
        end
        @test worst < tol                                  # |φ*| ≤ 1
    end
    # masked MULTIGRID reaches the same solution from a COLD start, sourced by
    # the same GAS density (the level D pools) — the gas-self-gravity MG gate.
    fill!(hier.levels[2].phi, 0.0f0); fill!(hier.levels[3].phi, 0.0f0)
    solve_gravity_level_mg!(hier, 1; source_coef = coef, ncycle = 6)
    solve_gravity_level_mg!(hier, 2; source_coef = coef, ncycle = 6)
    for (l, tol) in ((1, 0.02), (2, 0.02))
        lev = hier.levels[l + 1]
        hp = Array(lev.phi); dx = BA.level_dx(hier, l)
        worst = 0.0
        for s in lev.live
            m = lev.meta[s]; base = (Int(s)-1)*lev.stride
            for k in 1:lev.B, j in 1:lev.B, i in 1:lev.B
                x = ntuple(d -> (Float64(m.origin[d]) + (i,j,k)[d] - 0.5) * dx, 3)
                got = hp[base + ((lev.ng+k-1)*lev.nd + (lev.ng+j-1))*lev.nd + (lev.ng+i-1) + 1]
                worst = max(worst, abs(got - φex(x)))
            end
        end
        @test worst < tol                              # MG ≡ converged RB, gas source
    end
    # SPLIT-CLUSTER (Schwarz) path: a tiny budget forces recursive bisection down
    # to single-block parts — seam rings warm-start from the neighbor part's φ
    # via the sibling exchange, so repeated outer passes must reach the same
    # solution as the unsplit dense solve (the percolated-union scaling path).
    withenv("BAM_MGCLUSTER_GIB" => "1e-5") do
        delete!(hier.levels[2].tabs, :mgd); delete!(hier.levels[3].tabs, :mgd)
        fill!(hier.levels[2].phi, 0.0f0); fill!(hier.levels[3].phi, 0.0f0)
        for _ in 1:8
            solve_gravity_level_mg!(hier, 1; source_coef = coef, ncycle = 2)
        end
        for _ in 1:8
            solve_gravity_level_mg!(hier, 2; source_coef = coef, ncycle = 2)
        end
        for (l, tol) in ((1, 0.03), (2, 0.03))
            lev = hier.levels[l + 1]
            hp = Array(lev.phi); dx = BA.level_dx(hier, l)
            worst = 0.0
            for s in lev.live
                m = lev.meta[s]; base = (Int(s)-1)*lev.stride
                for k in 1:lev.B, j in 1:lev.B, i in 1:lev.B
                    x = ntuple(d -> (Float64(m.origin[d]) + (i,j,k)[d] - 0.5) * dx, 3)
                    got = hp[base + ((lev.ng+k-1)*lev.nd + (lev.ng+j-1))*lev.nd + (lev.ng+i-1) + 1]
                    worst = max(worst, abs(got - φex(x)))
                end
            end
            @test worst < tol                          # split Schwarz ≡ unsplit MG
        end
        delete!(hier.levels[2].tabs, :mgd); delete!(hier.levels[3].tabs, :mgd)
    end
    # re-solve with RB so the warm-restart check below sees its expected state
    solve_gravity_level!(hier, 1; source_coef = coef, nsweep = 400)

    # warm restart: a few sweeps keep the residual small
    rw = solve_gravity_level!(hier, 1; source_coef = coef, nsweep = 5)
    @test rw < 5 * r1b + 1e-6
    # pool kick ≈ −∇φ* (uniform ρ=1 block for a clean momentum read)
    lev = hier.levels[2]
    n = lev.cap * lev.stride
    hone = fill(1.0, n); hz = zeros(n)
    BA.encode_from_host!(lev, hone, hz, hz, hz, hone, hone)
    dt2 = 1e-3
    grav_kick_level_pool!(hier, 1, dt2)
    hS = Array(lev.S1); hssc = Array(lev.Ssc); dx = BA.level_dx(hier, 1)
    worst = 0.0
    for s in lev.live[1:1]
        m = lev.meta[s]; base = (Int(s)-1)*lev.stride
        for (ci,cj,ck) in ((8,8,8), (4,11,6))
            x = ntuple(d -> (Float64(m.origin[d]) + (ci,cj,ck)[d] - 0.5) * dx, 3)
            aex = 2π * sin(2π*x[1]) * cos(2π*x[2]) * cos(2π*x[3])   # −∂xφ*
            idx = base + ((ck+lev.ng-1)*lev.nd + (cj+lev.ng-1))*lev.nd + (ci+lev.ng-1) + 1
            got = Float64(hS[idx]) * hssc[s] / dt2
            worst = max(worst, abs(got - aex))
        end
    end
    @test worst < 0.15 * 2π                                # h²-limited gradient
end
end # for BE
