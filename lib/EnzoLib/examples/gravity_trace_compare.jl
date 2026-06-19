# Per-substep CPU(Enzo)-vs-GPU(Vespa) gravity divergence tracer.
#
# Reads an evolved SB state (warm up WARM cycles with pure Enzo), then traces ONE
# top-grid step through the live recursive evolve_level!. At EVERY level/subcycle the
# :gravity slot runs BOTH Enzo's session_gravity AND our composite_gravity! on the
# IDENTICAL pre-gravity state, and compares — per grid — the GravitatingMassField
# (source), the potential, and the acceleration field; plus the acceleration
# interpolated to ALL particles (root). Enzo's accel is restored after each compare so
# the traced trajectory stays on the certified reference (divergences don't compound).
#
# Reuses the EXACT production gravity (composite_gravity!) by including sb_composite_run.jl
# with SB_NO_MAIN=1. Run both backends to confirm CPU≡GPU AND both vs Enzo:
#   ENZOMODULES_GRID_LIB=<f32> SB_NO_MAIN=1 BACKEND=cpu   <jl> --project=lib/PPMKernels/test lib/EnzoLib/examples/gravity_trace_compare.jl [WARM]
#   ENZOMODULES_GRID_LIB=<f32> SB_NO_MAIN=1 BACKEND=metal <jl> --project=lib/PPMKernels/test lib/EnzoLib/examples/gravity_trace_compare.jl [WARM]
ENV["SB_NO_MAIN"]="1"
include(joinpath(@__DIR__, "sb_composite_run.jl"))   # composite_gravity!, ppm_hydro!, helpers, consts, GS
using Printf

slope(o,e)=(s=sum(vec(o).^2); s==0 ? 0.0 : sum(vec(o).*vec(e))/s)
corr(o,e)=(om=o.-sum(o)/length(o);em=e.-sum(e)/length(e);d=sqrt(sum(om.^2)*sum(em.^2)); d==0 ? 1.0 : sum(om.*em)/d)
reldiff(o,e)=(s=maximum(abs,e); s==0 ? 0.0 : maximum(abs,o.-e)/s)

const WARM = length(ARGS)>=1 ? parse(Int,ARGS[1]) : 6

# snapshot the active accel of every grid on a level → Dict gidx ⇒ (a1,a2,a3)
function snap_accel(h, level)
    n=EnzoLib.session_num_grids_on_level(h,level); d=Dict{Int,NTuple{3,Array{Float64,3}}}()
    for i in 0:n-1
        g=EnzoLib.problem_grid_index_on_level(h,level,i); gd=Tuple(Int.(EnzoLib.problem_grid_dims(h,g)))
        d[g]=ntuple(c->active_of(EnzoLib.problem_get_acceleration(h,c-1,g),gd),3)
    end; d
end
restore_accel!(h,snap)= for (g,a) in snap, c in 0:2
    gd=Tuple(Int.(EnzoLib.problem_grid_dims(h,g))); EnzoLib.problem_set_acceleration(h,c,place_active(a[c+1],gd);grid=g)
end

# CIC-interp (cell-centered) of an active accel field on the ROOT to all particle positions
function interp_root(a, Nc, pos)
    out=Vector{Float64}(undef,size(pos,1))
    @inbounds for p in 1:size(pos,1)
        gx=mod(pos[p,1],1.0)*Nc-0.5;gy=mod(pos[p,2],1.0)*Nc-0.5;gz=mod(pos[p,3],1.0)*Nc-0.5
        i=floor(Int,gx);fx=gx-i;j=floor(Int,gy);fy=gy-j;k=floor(Int,gz);fz=gz-k; w(q)=mod(q,Nc)+1
        out[p]=a[w(i),w(j),w(k)]*(1-fx)*(1-fy)*(1-fz)+a[w(i+1),w(j),w(k)]*fx*(1-fy)*(1-fz)+
               a[w(i),w(j+1),w(k)]*(1-fx)*fy*(1-fz)+a[w(i+1),w(j+1),w(k)]*fx*fy*(1-fz)+
               a[w(i),w(j),w(k+1)]*(1-fx)*(1-fy)*fz+a[w(i+1),w(j),w(k+1)]*fx*(1-fy)*fz+
               a[w(i),w(j+1),w(k+1)]*(1-fx)*fy*fz+a[w(i+1),w(j+1),w(k+1)]*fx*fy*fz
    end; out
end

const _sub = Dict{Int,Int}()   # subcycle counter per level

# comparison gravity hook: run Enzo, snapshot; run ours, snapshot; diff; restore Enzo
function grav_compare!(h, level, dt)
    sc = (_sub[level] = get(_sub,level,-1)+1)
    # (1) Enzo reference gravity → GMF, potential, grid accel
    EnzoLib.session_gravity(h, level)
    enzo_acc = snap_accel(h, level)
    enzo_gmf = Dict(g=>copy(EnzoLib.problem_get_gravitating_mass(h,g)) for g in keys(enzo_acc))
    enzo_pot = Dict(g=>copy(EnzoLib.problem_get_potential(h,g))        for g in keys(enzo_acc))
    a,_ = EnzoLib.session_cosmology(h); afac[]=a
    # (2) our composite gravity (sets GravState at level 0; overwrites grid accel)
    composite_gravity!(h, level, dt)
    our_acc = snap_accel(h, level)
    # (3) compare per grid (accel x-component is representative; report worst over grids)
    gks=collect(keys(enzo_acc)); ng=length(gks)
    accc=[corr(our_acc[g][1],enzo_acc[g][1]) for g in gks]
    accs=[slope(our_acc[g][1],enzo_acc[g][1]) for g in gks]
    accr=[reldiff(our_acc[g][1],enzo_acc[g][1]) for g in gks]
    nc(g)=prod(Tuple(Int.(EnzoLib.problem_grid_dims(h,g))).-2NG)   # active cell count
    @printf("L%d.s%d  grids=%-4d ACCEL_x: corr[min %.4f med %.4f] slope[med %.3f] maxreldiff[max %.3f]\n",
            level, sc, ng, minimum(accc), sort(accc)[(ng+1)÷2], sort(accs)[(ng+1)÷2], maximum(accr))
    if level>=1
        rg0=EnzoLib.problem_grid_index_on_level(h,0,0); Ncr=first(Tuple(Int.(EnzoLib.problem_grid_dims(h,rg0))))-2NG; Nf=2*Ncr
        sizes=[nc(g) for g in gks]
        bin(mask)=(c=accc[mask];s=accs[mask];n=count(mask); n==0 ? (0,NaN,NaN) : (n,sort(c)[(n+1)÷2],sort(s)[(n+1)÷2]))
        bs=bin(sizes.<1000); bm=bin(1000 .<=sizes.<8000); bb=bin(sizes.>=8000)
        @printf("        by size  small<10³(n=%d cor%.3f slp%.3f) mid(n=%d cor%.3f slp%.3f) big≥20³(n=%d cor%.3f slp%.3f)\n",
                bs[1],bs[2],bs[3], bm[1],bm[2],bm[3], bb[1],bb[2],bb[3])
        # the single biggest subgrid — this is the one that sets |v|max (cluster core)
        qb=argmax(sizes); g=gks[qb]; na=Tuple(Int.(EnzoLib.problem_grid_dims(h,g))).-2NG
        @printf("        BIGGEST g=%d na=%s (cells=%d): accel corr=%.4f slope=%.4f reldiff=%.3f\n",
                g, string(na), sizes[qb], accc[qb], accs[qb], accr[qb])
        for q in sortperm(accc)[1:min(5,ng)]
            g=gks[q]; na=Tuple(Int.(EnzoLib.problem_grid_dims(h,g))).-2NG
            l,_=EnzoLib.problem_grid_edge(h,g); le=Float64.(l)
            al=ntuple(dd->le[dd]*Nf,3)   # integer ⟺ aligned to the 2Nc deposit grid (Nf from ROOT)
            @printf("          worst g=%d na=%s cor=%.3f slp=%.2f rel=%.2f le·Nf=(%.2f,%.2f,%.2f)\n",
                    g,string(na),accc[q],accs[q],accr[q],al[1],al[2],al[3])
        end
    end
    # root: GMF + potential + ALL-particle accel (our φr/δr vs Enzo)
    if level==0
        g0=first(keys(enzo_acc)); gd=Tuple(Int.(EnzoLib.problem_grid_dims(h,g0))); Nc=gd[1]-2NG
        buf=(size(enzo_gmf[g0],1)-Nc)÷2; R=buf+1:buf+Nc
        egmf=enzo_gmf[g0][R,R,R]; epot=enzo_pot[g0][R,R,R]   # accessors return 3D arrays
        gs=GS[]
        @printf("        ROOT GMF(δr vs Enzo): corr=%.5f slope=%.4f | POT(φr vs Enzo): corr=%.5f slope=%.4f\n",
                corr(gs.δr,egmf), slope(gs.δr,egmf), corr(gs.φr,epot), slope(gs.φr,epot))
        pos=gs.posall
        ea=interp_root(enzo_acc[g0][1],Nc,pos); oa=interp_root(our_acc[g0][1],Nc,pos)
        @printf("        ALL-PARTICLE accel_x (root-interp, N=%d): corr=%.5f slope=%.4f maxreldiff=%.4f\n",
                length(ea), corr(oa,ea), slope(oa,ea), reldiff(oa,ea))
    end
    # (4) restore Enzo's accel — trajectory stays on the certified reference
    restore_accel!(h, enzo_acc)
    flush(stdout)
    return nothing
end

function main_trace()
    EnzoLib.grid_available() || error("grid dylib not built")
    rundir=joinpath(SBDIR,"trace_$(BE)"); mkpath(rundir)
    for f in readdir(SBDIR)
        (startswith(f,"SB_") && (occursin("Grid",f)||occursin("Particle",f))) || continue
        lnk=joinpath(rundir,f); islink(lnk)||isfile(lnk)||symlink(joinpath(SBDIR,f),lnk)
    end
    pf=joinpath(rundir,"run.enzo")
    write(pf, replace(read(joinpath(SBDIR,"SB_amr.enzo"),String),
                      r"GreensFunctionMaxNumber.*"=>"GreensFunctionMaxNumber   = 30\nNumberOfGhostZones        = 4"))
    @printf("GRAVITY TRACE  backend=%s  warmup=%d Enzo cycles, then 1 traced top step\n", BE, WARM); flush(stdout)
    cd(rundir) do
        h=EnzoLib.session_init(pf); h==C_NULL && error("session_init failed")
        try
            EnzoLib.session_rebuild(h,0)
            allenzo=EnzoLib.EngineConfig(; hydro=:enzo, gravity=:enzo, comoving_expansion=:enzo)
            for c in 1:WARM
                afac[]=first(EnzoLib.session_cosmology(h))
                EnzoLib.evolve_level!(h,0,0.0; engine=allenzo, regrid=true); EnzoLib.session_rebuild(h,0)
            end
            a=first(EnzoLib.session_cosmology(h)); afac[]=a
            ng=[EnzoLib.session_num_grids_on_level(h,l) for l in 0:3]
            @printf("evolved state: t=%.5f a=%.4f grids/level=%s — TRACING ONE TOP STEP\n",
                    EnzoLib.session_time(h), a, string(ng)); flush(stdout)
            # traced step: all-Enzo physics EXCEPT the gravity slot, which compares ours vs Enzo
            cmp=EnzoLib.EngineConfig(; hydro=:enzo, gravity=:julia, comoving_expansion=:enzo,
                                     hooks=Dict{Symbol,Function}(:gravity=>grav_compare!))
            EnzoLib.evolve_level!(h,0,0.0; engine=cmp, regrid=true)
            println("TRACE DONE"); flush(stdout)
        finally
            EnzoLib.free_problem(h)
        end
    end
end
main_trace()
