# test_hydro.jl — Phase 1 gates: the fused batched SSP-RK2 PLM+HLLC hydro on a
# static 2-level hierarchy.
#   * uniform flow across the C/F interface is preserved BIT-EXACTLY (the
#     recompute-reflux cancellation gate);
#   * Sedov on 2 levels: mass/energy to f32 round-off with reflux ON, visibly
#     worse with reflux OFF; blast mirror symmetry across block boundaries;
#   * fine-region L1 agreement with an equivalent single-level uniform-fine run.
const BA = BlockAMR

# set (ρ,u,v,w,P) from an analytic profile on every ACTIVE cell (Ge = P/(γ−1))
function set_ic!(hier, l, prof)
    lev = hier.levels[l + 1]
    dx = BA.level_dx(hier, l)
    h = Dict(f => Array(getfield(lev, f)) for f in (:D, :S1, :S2, :S3, :Tau, :Ge))
    γ = hier.gamma
    for s in lev.live
        m = lev.meta[s]; base = (Int(s) - 1) * lev.stride
        for k in 1:lev.B, j in 1:lev.B, i in 1:lev.B
            g = ntuple(d -> BA.wrapc(Int128(m.origin[d]) + (i, j, k)[d] - 1, lev.P[d]), 3)
            x = ntuple(d -> (Float64(g[d]) + 0.5) * dx, 3)
            ρ, u, v, w, P = prof(x)
            idx = base + ((lev.ng+k-1) * lev.nd + (lev.ng+j-1)) * lev.nd + (lev.ng+i-1) + 1
            ge = P / (γ - 1)
            h[:D][idx] = ρ; h[:S1][idx] = ρ*u; h[:S2][idx] = ρ*v; h[:S3][idx] = ρ*w
            h[:Ge][idx] = ge
            h[:Tau][idx] = ge + 0.5 * ρ * (u^2 + v^2 + w^2)
        end
    end
    for f in (:D, :S1, :S2, :S3, :Tau, :Ge)
        copyto!(getfield(lev, f), h[f])
    end
end

# static center refinement: 8 level-1 children covering parent cells [B/2,3B/2)³
function center_refined(; nbase = (32, 32, 32), B = 16, backend = :cpu)
    hier = AMRHierarchy(; nbase, B, backend)
    lev0 = init_base_level!(hier)
    Bh = B ÷ 2
    for cz in 0:1, cy in 0:1, cx in 0:1
        p = lev0.byorigin[(UInt128(cx * B), UInt128(cy * B), UInt128(cz * B))]
        off = (cx == 0 ? Bh : 0, cy == 0 ? Bh : 0, cz == 0 ? Bh : 0)
        add_block!(hier, 1, p, off)
    end
    @test check_nesting(hier)
    build_level_tables!(hier, 0); build_level_tables!(hier, 1)
    build_cf_register!(hier, 1)
    return hier
end

sedov(σ, E0, P0) = x -> begin
    r2 = sum((x .- 0.5) .^ 2)
    Pb = P0 + (2/3) * E0 * exp(-r2 / (2σ^2)) / ((2π)^1.5 * σ^3)   # (γ−1)·e_blob
    (1.0, 0.0, 0.0, 0.0, Pb)
end

for BE in BACKENDS
@testset "uniform flow across C/F is bit-exact [$BE]" begin
    hier = center_refined(; backend = BE)
    prof = x -> (1.0, 0.3, 0.2, 0.1, 1.0)
    set_ic!(hier, 0, prof); set_ic!(hier, 1, prof)
    r0 = [Array(getfield(hier.levels[il], f))
          for il in 1:2, f in (:D, :S1, :S2, :S3, :Tau, :Ge)]
    for _ in 1:5
        hierarchy_rk2_step!(hier, 0.4 * BA.level_dx(hier, 1) / 2.0)
    end
    r1 = [Array(getfield(hier.levels[il], f))
          for il in 1:2, f in (:D, :S1, :S2, :S3, :Tau, :Ge)]
    # ACTIVE cells: preserved to a few ulps.  (Bit-exactness is NOT expected of
    # the fields themselves: the dual-energy sync legitimately rewrites Ge as
    # Tau−KE (± ulp of the IC's independent rounding), and the 2³ restriction
    # mean of 8 identical values can round in its running sum.  The bit-exact
    # claim lives in the reflux register — asserted below.)
    worst = 0.0
    for il in 1:2
        lev = hier.levels[il]
        for (fi, f) in enumerate((:D, :S1, :S2, :S3, :Tau, :Ge))
            for s in lev.live
                base = (Int(s) - 1) * lev.stride
                for k in 1:lev.B, j in 1:lev.B, i in 1:lev.B
                    idx = base + ((lev.ng+k-1) * lev.nd + (lev.ng+j-1)) * lev.nd + (lev.ng+i-1) + 1
                    a = r0[il, fi][idx]; b = r1[il, fi][idx]
                    worst = max(worst, abs(Float64(a) - Float64(b)) / max(abs(Float64(a)), 1e-30))
                end
            end
        end
    end
    @test worst < 1e-6
    # ... and the reflux register is exactly zero after apply (recompute cancels)
    @test all(iszero, Array(hier.levels[2].tabs[:cfreg]))
end

@testset "2-level Sedov: conservation + symmetry [$BE]" begin
    σ = 2.0 / 64
    hier = center_refined(; backend = BE)
    prof = sedov(σ, 1.0, 1e-5)
    set_ic!(hier, 0, prof); set_ic!(hier, 1, prof)
    M0, px0, py0, pz0, E0t = total_conserved(hier)
    nst = 0
    for _ in 1:30
        dt = compute_dt(hier)
        hierarchy_rk2_step!(hier, dt); nst += 1
    end
    M1, px1, py1, pz1, E1t = total_conserved(hier)
    @test abs(M1 - M0) / M0 < 2e-5                       # f32 round-off random walk
    @test abs(E1t - E0t) / E0t < 2e-5
    @test max(abs(px1), abs(py1), abs(pz1)) < 2e-4 * sqrt(2 * M0 * E0t)
    # mirror symmetry of ρ on the fine level across the box center
    lev1 = hier.levels[2]; hD = Array(lev1.D)
    val = Dict{NTuple{3,Int},Float32}()
    for s in lev1.live
        m = lev1.meta[s]; base = (Int(s) - 1) * lev1.stride
        for k in 1:lev1.B, j in 1:lev1.B, i in 1:lev1.B
            g = (Int(m.origin[1]) + i - 1, Int(m.origin[2]) + j - 1, Int(m.origin[3]) + k - 1)
            val[g] = hD[base + ((lev1.ng+k-1)*lev1.nd + (lev1.ng+j-1))*lev1.nd + (lev1.ng+i-1) + 1]
        end
    end
    worst = 0.0
    for (g, v) in val
        gm = (63 - g[1], g[2], g[3])                     # mirror in x (fine grid is [16,48)³ of 64)
        haskey(val, gm) || continue
        worst = max(worst, abs(v - val[gm]) / max(abs(v), 1e-6))
    end
    @test worst < 2e-3
end

@testset "shock crossing C/F: reflux on=round-off, off=broken [$BE]" begin
    # blob NEAR the interface so the blast actually crosses it within the run;
    # this is the scenario where the flux correction carries real weight.
    σ = 2.0 / 64
    prof = x -> begin
        r2 = (x[1] - 0.30)^2 + (x[2] - 0.5)^2 + (x[3] - 0.5)^2
        (1.0, 0.0, 0.0, 0.0, 1e-5 + (2/3) * exp(-r2 / (2σ^2)) / ((2π)^1.5 * σ^3))
    end
    drift = Float64[]
    for rf in (true, false)
        hier = center_refined(; backend = BE)
        set_ic!(hier, 0, prof); set_ic!(hier, 1, prof)
        Ma, _, _, _, Ea = total_conserved(hier)
        for _ in 1:60
            hierarchy_rk2_step!(hier, compute_dt(hier); reflux = rf)
        end
        Mb, _, _, _, Eb = total_conserved(hier)
        push!(drift, abs(Eb - Ea) / Ea)
    end
    @test drift[1] < 2e-6                      # reflux ON: round-off even at crossing
    @test drift[2] > 1e3 * max(drift[1], 1e-12)   # OFF: catastrophically worse (~3%)
end

@testset "fine region matches uniform-fine reference [$BE]" begin
    σ = 2.0 / 64
    prof = sedov(σ, 1.0, 1e-5)
    hier = center_refined(; backend = BE)
    set_ic!(hier, 0, prof); set_ic!(hier, 1, prof)
    ref = AMRHierarchy(; nbase = (64, 64, 64), B = 16, backend = BE)
    init_base_level!(ref); build_level_tables!(ref, 0)
    set_ic!(ref, 0, prof)
    for _ in 1:25
        dt = compute_dt(hier)                            # AMR dt drives both
        hierarchy_rk2_step!(hier, dt)
        hierarchy_rk2_step!(ref, dt)
    end
    # gather ρ on the fine lattice
    getrho(h, il) = begin
        lev = h.levels[il]; hD = Array(lev.D); out = Dict{NTuple{3,Int},Float32}()
        for s in lev.live
            m = lev.meta[s]; base = (Int(s) - 1) * lev.stride
            for k in 1:lev.B, j in 1:lev.B, i in 1:lev.B
                g = (Int(m.origin[1]) + i - 1, Int(m.origin[2]) + j - 1, Int(m.origin[3]) + k - 1)
                out[g] = hD[base + ((lev.ng+k-1)*lev.nd + (lev.ng+j-1))*lev.nd + (lev.ng+i-1) + 1]
            end
        end
        out
    end
    amr = getrho(hier, 2); rf = getrho(ref, 1)
    l1 = 0.0; n = 0
    for (g, v) in amr
        l1 += abs(Float64(v) - Float64(rf[g])); n += 1
    end
    @test n == 32^3
    @test l1 / n < 3e-3                                   # C/F boundary influence only
end
end # for BE
