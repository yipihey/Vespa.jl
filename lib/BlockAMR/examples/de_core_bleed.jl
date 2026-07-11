# de_core_bleed.jl — SMALL fast reproducer for the deep-core dual-energy thermal
# bleed seen in the L20 cosmology run (L9 core 220K → L10 95K → toward CMB).
#
# A self-gravitating Jeans collapse on a tiny 32³ box, f16 storage + per-block
# scales, regrid to deep levels, native gravity — NO cosmology lead-in, no
# particles, no chem.  The collapse goes SUPERSONIC and gravity SUSTAINS the
# infall onto a stagnant core (the real run's configuration), refining deep so
# the finest level subcycles 2^L× per root step.
#
# ADIABATIC PHYSICS (no cooling): the collapsing core must HEAT via pdV
# compression → its specific internal energy e must RISE.  THE DUAL-ENERGY BUG
# shows as the deep-core e FAILING to rise / going cold: the stagnant center
# shares a block with the supersonic-infall skirt, so the block's Tau scale esc
# is pinned by the skirt KE, the center's small thermal Tau quantizes coarsely,
# and the sync nG=Tau−KE overwrites the accurate advected Ge → the core bleeds.
#
# Run (CPU, ~1-3 min — no GPU needed, no contention with the big run):
#   julia --project=lib/BlockAMR/test lib/BlockAMR/examples/de_core_bleed.jl
# Env: BAM_LMAX (refine depth, default 6), BAM_NSTEPS, BAM_A (blob amplitude),
#      BAM_P0 (initial pressure = coldness), BAM_COEF (4πG).
#
# SIGNATURE of the bug: "core e/e0" at the deepest level drops < 1 (cold) while
# ρ_peak climbs.  After the fix: core e/e0 RISES with ρ (adiabatic heating).

using BlockAMR, Printf
const BA = BlockAMR
try; @eval using CUDA; catch; end
BE = (get(ENV, "BAM_BACKEND", "cpu") == "cuda" && BA.has_backend(:cuda)) ? :cuda : :cpu

const γ    = 5f0/3f0
const A    = parse(Float64, get(ENV, "BAM_A", "8.0"))     # blob overdensity
const σ    = 0.06
const P0   = parse(Float64, get(ENV, "BAM_P0", "0.02"))   # low P ⇒ cold, Jeans-unstable
const COEF = parse(Float64, get(ENV, "BAM_COEF", "120.0")) # 4πG (strong ⇒ fast runaway)
const LMAX = parse(Int,     get(ENV, "BAM_LMAX", "6"))
const NSTP = parse(Int,     get(ENV, "BAM_NSTEPS", "160"))
const NBASE = 32; const B = 16

prof(x) = begin
    r2 = (x[1]-0.5)^2 + (x[2]-0.5)^2 + (x[3]-0.5)^2
    (1.0 + A*exp(-r2/(2σ^2)), 0.0, 0.0, 0.0, P0)
end
const RHO_MEAN = 1.0 + A*(2π)^1.5*σ^3
const E0 = P0 / ((γ-1) * (1.0 + A))                       # specific int. energy at the initial core

# inline set_ic! (same as test/test_hydro.jl)
function set_ic_local!(hier, l)
    lev = hier.levels[l + 1]; isempty(lev.live) && return
    dx = BA.level_dx(hier, l); n = lev.cap * lev.stride
    h = Dict(f => zeros(Float64, n) for f in (:D,:S1,:S2,:S3,:Tau,:Ge))
    for s in lev.live
        m = lev.meta[s]; base = (Int(s)-1)*lev.stride
        for k in 1:lev.B, j in 1:lev.B, i in 1:lev.B
            g = ntuple(d -> BA.wrapc(Int128(m.origin[d]) + (i,j,k)[d] - 1, lev.P[d]), 3)
            x = ntuple(d -> (Float64(g[d]) + 0.5) * dx, 3)
            ρ,u,v,w,P = prof(x)
            idx = base + ((lev.ng+k-1)*lev.nd + (lev.ng+j-1))*lev.nd + (lev.ng+i-1) + 1
            ge = P/(γ-1)
            h[:D][idx]=ρ; h[:S1][idx]=ρ*u; h[:S2][idx]=ρ*v; h[:S3][idx]=ρ*w
            h[:Ge][idx]=ge; h[:Tau][idx]=ge + 0.5*ρ*(u^2+v^2+w^2)
        end
    end
    BA.encode_from_host!(lev, h[:D],h[:S1],h[:S2],h[:S3],h[:Tau],h[:Ge])
end

# density peak + the specific internal energy e THERE, scanning the deepest level
function core_state(hier)
    ρbest = 0.0; ecore = NaN; Lbest = 0
    for l in 0:length(hier.levels)-1
        lev = hier.levels[l+1]; isempty(lev.live) && continue
        hD=Array(lev.D); hG=Array(lev.Ge); hDsc=Array(lev.Dsc); hGsc=Array(lev.Gsc)
        for s in lev.live
            base=(Int(s)-1)*lev.stride
            for k in 1:lev.B, j in 1:lev.B, i in 1:lev.B
                idx=base+((lev.ng+k-1)*lev.nd+(lev.ng+j-1))*lev.nd+(lev.ng+i-1)+1
                ρ=Float64(hD[idx])*hDsc[s]
                if ρ>ρbest
                    ρbest=ρ; Lbest=l; ecore=Float64(hG[idx])*hGsc[s]/max(ρ,1e-30)
                end
            end
        end
    end
    return ρbest, ecore, Lbest
end

hier = AMRHierarchy(; nbase=(NBASE,NBASE,NBASE), B, backend=BE, T=Float16, cfl=0.3, scheme=:ctu)
init_base_level!(hier); build_level_tables!(hier, 0)
set_ic_local!(hier, 0)
pol = BlockRefinementPolicy(; dthresh=2.0, nbuf=2, every=4, lmax=LMAX)
regrid!(hier, pol)
for l in 1:length(hier.levels)-1; set_ic_local!(hier, l); end
ρ0,e0i,_ = core_state(hier)
println("de_core_bleed: backend=$BE LMAX=$LMAX A=$A P0=$P0 coef=$COEF  e0=$(round(E0,sigdigits=4))")
println("adiabatic: core e should RISE with ρ; BLEED = e/e0 drops < 1 as ρ climbs")
@printf("step %3d  Lmax=%d  ρ_peak=%.3g  core_e/e0=%.4f\n", 0, 0, ρ0, e0i/E0)

function run!(hier, pol)
    sg = (coef=COEF, nsweep=25, nsweep0=150, rho_mean=RHO_MEAN)
    maxseen = 0
    for st in 1:NSTP
        advance_hierarchy!(hier; selfgrav=sg)
        for l in 0:length(hier.levels)-1; update_scales!(hier, l); end
        st % pol.every == 0 && regrid!(hier, pol)
        ρ,e,L = core_state(hier)
        if L > maxseen || st % 20 == 0
            maxseen = max(maxseen, L)
            @printf("step %3d  Lmax=%d  ρ_peak=%.3g  core_e/e0=%.4f  %s\n",
                    st, L, ρ, e/E0, e/E0 < 0.7 ? "◄ COLD (bleed?)" : ""); flush(stdout)
        end
        isfinite(ρ) || (println("NONFINITE at step $st"); break)
    end
end
run!(hier, pol)
ρf,ef,Lf = core_state(hier)
@printf("FINAL  Lmax=%d  ρ_peak=%.3g  core_e/e0=%.4f  (%s)\n", Lf, ρf, ef/E0,
        ef/E0 < 0.7 ? "BLEED reproduced (core cold despite compression)" :
                      "core heated adiabatically (no bleed)")
