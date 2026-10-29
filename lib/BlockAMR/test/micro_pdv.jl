# micro_pdv.jl — is the BAM_NO_PDV toggle actually live in BOTH ctu kernels?
# Non-uniform IC with a velocity divergence (u = k*x => divv != 0), one ctu step
# with BAM_NO_PDV="0" (pdV ON) vs "1" (pdV OFF).  If Ge differs, the toggle is
# live.  Tests CPU per-cell kernel and (if CUDA) the tiled GPU kernel.
using BlockAMR
try; @eval using CUDA; catch; end
const BA = BlockAMR

function set_ic_grad!(hier, l)
    lev = hier.levels[l+1]; dx = BA.level_dx(hier, l); n = lev.cap*lev.stride
    h = Dict(f => zeros(Float64, n) for f in (:D,:S1,:S2,:S3,:Tau,:Ge)); γ = hier.gamma
    for s in lev.live
        m = lev.meta[s]; base=(Int(s)-1)*lev.stride
        for k in 1:lev.B, j in 1:lev.B, i in 1:lev.B
            gx = BA.wrapc(Int128(m.origin[1])+i-1, lev.P[1])
            x = (Float64(gx)+0.5)*dx
            ρ=1.0; u=0.5*x; v=0.0; w=0.0; P=1.6e-5     # divv = du/dx = 0.5 != 0
            idx = base + ((lev.ng+k-1)*lev.nd + (lev.ng+j-1))*lev.nd + (lev.ng+i-1)+1
            ge=P/(γ-1)
            h[:D][idx]=ρ; h[:S1][idx]=ρ*u; h[:S2][idx]=ρ*v; h[:S3][idx]=ρ*w
            h[:Ge][idx]=ge; h[:Tau][idx]=ge+0.5*ρ*(u^2+v^2+w^2)
        end
    end
    BA.encode_from_host!(lev, h[:D],h[:S1],h[:S2],h[:S3],h[:Tau],h[:Ge])
end

function center_refined(; backend=:cpu, T=Float32)
    hier = AMRHierarchy(; nbase=(32,32,32), B=16, backend, T, nsp=0, scheme=:ctu)
    lev0 = BA.init_base_level!(hier); Bh=8
    for cz in 0:1, cy in 0:1, cx in 0:1
        p = lev0.byorigin[(UInt128(cx*16),UInt128(cy*16),UInt128(cz*16))]
        BA.add_block!(hier, 1, p, (cx==0 ? Bh : 0, cy==0 ? Bh : 0, cz==0 ? Bh : 0))
    end
    BA.build_level_tables!(hier,0); BA.build_level_tables!(hier,1); BA.build_cf_register!(hier,1)
    return hier
end

function meanGe(hier)
    lev = hier.levels[1]; hGeo = Array(lev.Geo); gsc = Array(lev.Gsc)
    tot=0.0; cnt=0
    for s in lev.live
        base=(Int(s)-1)*lev.stride
        for k in 1:lev.B, j in 1:lev.B, i in 1:lev.B
            idx = base + ((lev.ng+k-1)*lev.nd+(lev.ng+j-1))*lev.nd+(lev.ng+i-1)+1
            tot += Float64(hGeo[idx])*Array(lev.Gsc)[Int(s)]; cnt+=1
        end
    end
    return tot/cnt
end

function run_toggle(backend, tiled)
    res = Dict()
    for flag in ("0","1")
        ENV["BAM_NO_PDV"]=flag
        hier = center_refined(; backend)
        set_ic_grad!(hier, 0); set_ic_grad!(hier, 1)
        BA.ctu_level!(hier, 0, 0.15f0; tiled=tiled)
        res[flag]=meanGe(hier)
    end
    delete!(ENV,"BAM_NO_PDV")
    return res["0"], res["1"]
end

on,off = run_toggle(:cpu, false)
println("CPU per-cell: mean Ge  pdV_ON=$on  pdV_OFF=$off  Δ=$(on-off)  rel=$((on-off)/on)")
if BA.has_backend(:cuda)
    on,off = run_toggle(:cuda, true)
    println("GPU tiled   : mean Ge  pdV_ON=$on  pdV_OFF=$off  Δ=$(on-off)  rel=$((on-off)/on)")
else
    println("CUDA not available; skipped tiled kernel test")
end
