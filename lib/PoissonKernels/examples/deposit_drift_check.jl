# Test the deposit time-centering hypothesis: Enzo's PrepareDensityField(When=0.5)
# deposits particles DRIFTED to t+½dt (x += (½dt/a)·v, Grid_UpdateParticlePosition
# Coefficient=dt/a).  Our CIC uses undrifted positions ⇒ corr 0.70.  Scan the drift
# coefficient c (x' = x + c·(dt/a)·v) and find the c that maximizes corr vs Enzo's GMF.
# c≈0.5 ⇒ the drift is the cause (then composite_gravity! drifts the CIC likewise).
# Run: ENZOMODULES_GRID_LIB=<f32> <jl> --project=lib/PoissonKernels/test lib/PoissonKernels/examples/deposit_drift_check.jl
using PoissonKernels, EnzoLib, Printf
const SB="/Users/tabel/Projects/enzo-dev/run/CosmologySimulation/SantaBarbaraCluster/SB_amr.enzo"
corr(o,e)=(om=o.-sum(o)/length(o);em=e.-sum(e)/length(e);sum(om.*em)/(sqrt(sum(om.^2)*sum(em.^2))+1e-300))
function cic_drift(pos,vel,N,disp)            # CIC of (pos + disp·vel) onto N³, periodic
    ρ=zeros(N,N,N)
    @inbounds for p in 1:size(pos,1)
        gx=(pos[p,1]+disp*vel[p,1])*N-0.5; gy=(pos[p,2]+disp*vel[p,2])*N-0.5; gz=(pos[p,3]+disp*vel[p,3])*N-0.5
        i0=floor(Int,gx);fx=gx-i0;j0=floor(Int,gy);fy=gy-j0;k0=floor(Int,gz);fz=gz-k0
        for (di,wi) in ((0,1-fx),(1,fx)),(dj,wj) in ((0,1-fy),(1,fy)),(dk,wk) in ((0,1-fz),(1,fz))
            ρ[mod(i0+di,N)+1,mod(j0+dj,N)+1,mod(k0+dk,N)+1]+=wi*wj*wk
        end
    end
    ρ
end
cd(dirname(SB)) do
    h=EnzoLib.session_init(SB); EnzoLib.session_set_boundary(h,0); EnzoLib.session_rebuild(h,0)
    dt=EnzoLib.session_compute_dt(h,0); EnzoLib.session_set_dt(h,dt,0)   # set dt so the deposit drift is active
    EnzoLib.session_gravity(h,0)                                        # Enzo deposit at t+½dt → root GMF
    a,z=EnzoLib.session_cosmology(h)
    gmf=EnzoLib.problem_get_gravitating_mass(h,0); gd=size(gmf); Nc=128; b=(gd[1]-Nc)÷2
    gA=gmf[b+1:b+Nc,b+1:b+Nc,b+1:b+Nc]; gz0=gA.-sum(gA)/length(gA)
    pos=EnzoLib.read_particles(h); vel=EnzoLib.read_particle_velocities(h)
    @printf("dt=%.4e a=%.4f z=%.3f  max|v|=%.3e  drift(½dt/a)·max|v| in cells=%.3f\n",
            dt,a,z,maximum(abs,vel),0.5*dt/a*maximum(abs,vel)*Nc)
    for c in 0.0:0.1:0.7
        ρ=cic_drift(pos,vel,Nc,c*dt/a); zc=ρ.-sum(ρ)/length(ρ)
        @printf("  drift c=%.2f : corr(our CIC, Enzo GMF)=%.4f\n", c, corr(zc,gz0))
    end
    EnzoLib.free_problem(h)
end
