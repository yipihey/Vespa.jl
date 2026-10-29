# micro_eta.jl — is the dual-energy sync threshold η actually effective/plumbed?
# 2-level hierarchy, KE-dominated uniform state, then CORRUPT Ge to 10x its
# consistent value (Ge != Tau-KE).  One ctu step at η=0 (sync ON: should pull
# Ge -> Tau-KE ~ consistent) vs η=1 (sync OFF: Ge stays ~10x).  Uniform otherwise,
# so ONLY the sync can differ.  ratio≈10 => sync effective; ≈1 => no-op/dead.
using BlockAMR
const BA = BlockAMR

function set_ic!(hier, l, prof)
    lev = hier.levels[l + 1]
    dx = BA.level_dx(hier, l); n = lev.cap * lev.stride
    h = Dict(f => zeros(Float64, n) for f in (:D,:S1,:S2,:S3,:Tau,:Ge)); γ = hier.gamma
    for s in lev.live
        m = lev.meta[s]; base = (Int(s)-1)*lev.stride
        for k in 1:lev.B, j in 1:lev.B, i in 1:lev.B
            ρ,u,v,w,P = prof
            idx = base + ((lev.ng+k-1)*lev.nd + (lev.ng+j-1))*lev.nd + (lev.ng+i-1) + 1
            ge = P/(γ-1)
            h[:D][idx]=ρ; h[:S1][idx]=ρ*u; h[:S2][idx]=ρ*v; h[:S3][idx]=ρ*w
            h[:Ge][idx]=ge; h[:Tau][idx]=ge + 0.5*ρ*(u^2+v^2+w^2)
        end
    end
    BA.encode_from_host!(lev, h[:D],h[:S1],h[:S2],h[:S3],h[:Tau],h[:Ge])
end

function center_refined(; backend=:cpu, T=Float32)
    hier = AMRHierarchy(; nbase=(32,32,32), B=16, backend, T, nsp=0, scheme=:ctu)
    lev0 = BA.init_base_level!(hier); Bh = 8
    for cz in 0:1, cy in 0:1, cx in 0:1
        p = lev0.byorigin[(UInt128(cx*16),UInt128(cy*16),UInt128(cz*16))]
        BA.add_block!(hier, 1, p, (cx==0 ? Bh : 0, cy==0 ? Bh : 0, cz==0 ? Bh : 0))
    end
    BA.build_level_tables!(hier,0); BA.build_level_tables!(hier,1); BA.build_cf_register!(hier,1)
    return hier
end

function run_eta(η)
    hier = center_refined()
    # KE-dominated: ρ=1,v=(0.3,-0.2,0.1) KE=0.07 ; P=1.6e-5 -> ge=2.4e-5 (internal/total~3.4e-4)
    prof = (1.0, 0.3, -0.2, 0.1, 1.6e-5)
    set_ic!(hier, 0, prof); set_ic!(hier, 1, prof)
    for l in 0:1
        lev = hier.levels[l+1]; lev.Ge .*= 10          # corrupt INPUT R-buffer Ge
    end
    BA.ctu_level!(hier, 0, 0.15f0; η = Float32(η))
    lev = hier.levels[1]
    s = Int(lev.live[1]); base=(s-1)*lev.stride
    idx = base + ((lev.ng)*lev.nd + lev.ng)*lev.nd + lev.ng + 1
    gsc = Array(lev.Gsc)[s]
    return Float32(Array(lev.Geo)[idx]) * gsc
end

g0 = run_eta(0.0); g1 = run_eta(1.0)
println("Ge (interior L1 cell) after 1 CTU step  [consistent~2.4e-5, corrupted~2.4e-4]:")
println("  η=0.0 (sync ON): ", g0)
println("  η=1.0 (sync OFF): ", g1)
println("  ratio g1/g0 = ", g1/g0, "   (≈10 => sync EFFECTIVE; ≈1 => sync no-op/plumbing dead)")
