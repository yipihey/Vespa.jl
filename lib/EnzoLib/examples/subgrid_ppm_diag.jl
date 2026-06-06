# Lightweight: WHY does our PPM give ΔD=0 on a subgrid (vs Enzo's ~1e-5)?
# Read a real subgrid's inputs (accel, velocity) and run our PPM with gravity off/on.
# Run: ENZOMODULES_GRID_LIB=<f32> <jl> --project=lib/PPMKernels/test lib/EnzoLib/examples/subgrid_ppm_diag.jl
using EnzoLib, PPMKernels, Printf
try; @eval using Metal; catch; end
const SB="/Users/tabel/Projects/enzo-dev/run/CosmologySimulation/SantaBarbaraCluster/SB_amr.enzo"
const NG=4; const GAMMA=5/3; const iD,iV1,iV2,iV3,iTE,iGE=0,1,2,3,4,5
active(flat,gd)=(n=gd.-2NG; Array(reshape(Float64.(flat),gd...)[NG+1:NG+n[1],NG+1:NG+n[2],NG+1:NG+n[3]]))

cd(dirname(SB)) do
    pf=joinpath(dirname(SB),"SB_ppmdiag.enzo")
    write(pf, replace(read(SB,String), r"GreensFunctionMaxNumber.*"=>"GreensFunctionMaxNumber=30\nNumberOfGhostZones=4"))
    h=EnzoLib.session_init(pf); EnzoLib.session_set_boundary(h,0); EnzoLib.session_rebuild(h,0)
    EnzoLib.session_gravity(h,0); EnzoLib.session_gravity(h,1)
    nL1=EnzoLib.session_num_grids_on_level(h,1)
    idxs=[EnzoLib.problem_grid_index_on_level(h,1,i) for i in 0:nL1-1]
    g=argmax(gi->prod(Int.(EnzoLib.problem_grid_dims(h,gi))), idxs)
    gd=Tuple(Int.(EnzoLib.problem_grid_dims(h,g)))
    EnzoLib.session_set_boundary(h,1)
    be=PPMKernels.backend(:cpu); T=Float64
    f(fi)=PPMKernels.to_device(be,EnzoLib.problem_get_field(h,fi,g),T)
    D0=active(EnzoLib.problem_get_field(h,iD,g),gd)
    gxh=EnzoLib.problem_get_acceleration(h,0,g); gyh=EnzoLib.problem_get_acceleration(h,1,g); gzh=EnzoLib.problem_get_acceleration(h,2,g)
    dt=EnzoLib.session_compute_dt(h,1)
    @printf("subgrid dims=%s, dt=%.4e\n", string(gd), dt)
    @printf("INPUTS: max|gx|=%.3e |gy|=%.3e |gz|=%.3e ; max|vx|=%.3e |D-0.1|=%.3e\n",
            maximum(abs,gxh),maximum(abs,gyh),maximum(abs,gzh),
            maximum(abs,active(EnzoLib.problem_get_field(h,iV1,g),gd)), maximum(abs,D0.-0.1))
    for grav in (0,1)
        d=f(iD);e=f(iTE);ge=f(iGE);vx=f(iV1);vy=f(iV2);vz=f(iV3)
        gx = grav==1 ? PPMKernels.to_device(be,gxh,T) : PPMKernels.device_zeros(be,T,gd)
        gy = grav==1 ? PPMKernels.to_device(be,gyh,T) : PPMKernels.device_zeros(be,T,gd)
        gz = grav==1 ? PPMKernels.to_device(be,gzh,T) : PPMKernels.device_zeros(be,T,gd)
        PPMKernels.ppm_step_3d!(d,e,ge,vx,vy,vz,gx,gy,gz,gd,NG; dt=dt,gamma=GAMMA,order=(1,2,3),gravity=grav,idual=1)
        ΔD=active(Float64.(PPMKernels.to_host(d)),gd).-D0
        Δv=active(Float64.(PPMKernels.to_host(vx)),gd).-active(EnzoLib.problem_get_field(h,iV1,g),gd)
        @printf("  gravity=%d : ΔD L2=%.3e (max=%.2e) ; Δvx L2=%.3e (max=%.2e)\n",
                grav, sqrt(sum(abs2,ΔD)),maximum(abs,ΔD), sqrt(sum(abs2,Δv)),maximum(abs,Δv))
    end
    # PROPOSED FIX: operator-split — pre-kick v by g·dt/2, pure hydro (gravity=0), post-kick
    let d=f(iD),e=f(iTE),ge=f(iGE),vx=f(iV1),vy=f(iV2),vz=f(iV3)
        gx=PPMKernels.to_device(be,gxh,T);gy=PPMKernels.to_device(be,gyh,T);gz=PPMKernels.to_device(be,gzh,T)
        vx.+=gx.*(dt/2); vy.+=gy.*(dt/2); vz.+=gz.*(dt/2)
        PPMKernels.ppm_step_3d!(d,e,ge,vx,vy,vz,gx,gy,gz,gd,NG; dt=dt,gamma=GAMMA,order=(1,2,3),gravity=0,idual=1)
        vx.+=gx.*(dt/2); vy.+=gy.*(dt/2); vz.+=gz.*(dt/2)
        ΔD=active(Float64.(PPMKernels.to_host(d)),gd).-D0
        @printf("  OP-SPLIT (pre/post kick) ΔD L2=%.3e (max=%.2e)  [Enzo target ~3.6e-4]\n",
                sqrt(sum(abs2,ΔD)),maximum(abs,ΔD))
    end
    EnzoLib.free_problem(h)
end
