# test_collapse.jl — END-TO-END: self-gravitating Jeans collapse with the FULL
# stack: f16 storage + per-block scales, dynamic regridding to lmax=2, strict
# 2:1 subcycling, native level Poisson gravity (periodic base solve + Dirichlet
# level solves), hysteresis rescaling.  Gates: the peak density GROWS (vs a
# pressure-dispersal no-gravity control), refinement follows the collapse to
# the deepest level, everything stays finite, nesting holds throughout.
const BA = BlockAMR

peak_rho(hier) = begin
    best = 0.0
    for l in 0:length(hier.levels)-1
        lev = hier.levels[l+1]
        isempty(lev.live) && continue
        hD = Array(lev.D); hsc = Array(lev.Dsc)
        for s in lev.live
            base = (Int(s)-1)*lev.stride
            for k in 1:lev.B, j in 1:lev.B, i in 1:lev.B
                v = Float64(hD[base + ((lev.ng+k-1)*lev.nd + (lev.ng+j-1))*lev.nd + (lev.ng+i-1) + 1]) * hsc[s]
                v > best && (best = v)
            end
        end
    end
    best
end

for BE in BACKENDS
@testset "Jeans collapse end-to-end (f16, AMR, native gravity) [$BE]" begin
    σ = 0.06; A = 3.0; P0 = 0.05
    prof = x -> begin
        r2 = sum((x .- 0.5).^2)
        (1.0 + A * exp(-r2 / (2σ^2)), 0.0, 0.0, 0.0, P0)
    end
    rho_mean = 1.0 + A * (2π)^1.5 * σ^3          # box mean of the blob (V = 1)
    coef = 82.0                                  # 4πG: strongly Jeans-unstable
    results = Float64[]
    for grav in (true, false)
        hier = AMRHierarchy(; nbase = (32, 32, 32), B = 16, backend = BE,
                            T = Float16, cfl = 0.3)
        init_base_level!(hier); build_level_tables!(hier, 0)
        set_ic!(hier, 0, prof)
        pol = BlockRefinementPolicy(; dthresh = 2.0, nbuf = 2, every = 4, lmax = 2)
        regrid!(hier, pol)
        @test check_nesting(hier)
        for l in 1:length(hier.levels)-1
            isempty(hier.levels[l+1].live) || set_ic!(hier, l, prof)
        end
        ρ0 = peak_rho(hier)
        sg = grav ? (coef = coef, nsweep = 25, nsweep0 = 150, rho_mean = rho_mean) : nothing
        for n in 1:40
            advance_hierarchy!(hier; selfgrav = sg)
            for l in 0:length(hier.levels)-1
                update_scales!(hier, l)
            end
            if n % pol.every == 0
                regrid!(hier, pol)
                @test check_nesting(hier)
            end
        end
        ρ1 = peak_rho(hier)
        push!(results, ρ1 / ρ0)
        if grav
            @test isfinite(ρ1)
            # collapse reached the deepest allowed level
            @test length(hier.levels) >= 3 && !isempty(hier.levels[3].live)
        end
    end
    @test results[1] > 1.5           # gravity: peak grows (collapse under way)
    @test results[2] < 1.1           # control: pressure disperses the blob
    @test results[1] > 2 * results[2]
end
end # for BE
