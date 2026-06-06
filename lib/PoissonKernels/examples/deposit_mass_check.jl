using PoissonKernels, EnzoLib, Printf
const SB="/Users/tabel/Projects/enzo-dev/run/CosmologySimulation/SantaBarbaraCluster/SB_amr.enzo"
corr(o,e)=(om=o.-sum(o)/length(o);em=e.-sum(e)/length(e);sum(om.*em)/(sqrt(sum(om.^2)*sum(em.^2))+1e-300))
function cic(pos,N,m)
    ρ=zeros(N,N,N)
    @inbounds for p in 1:size(pos,1)
        gx=pos[p,1]*N-0.5;gy=pos[p,2]*N-0.5;gz=pos[p,3]*N-0.5
        i0=floor(Int,gx);fx=gx-i0;j0=floor(Int,gy);fy=gy-j0;k0=floor(Int,gz);fz=gz-k0
        w=m[p]
        for (di,wi) in ((0,1-fx),(1,fx)),(dj,wj) in ((0,1-fy),(1,fy)),(dk,wk) in ((0,1-fz),(1,fz))
            ρ[mod(i0+di,N)+1,mod(j0+dj,N)+1,mod(k0+dk,N)+1]+=w*wi*wj*wk
        end
    end
    ρ
end
cd(dirname(SB)) do
    h=EnzoLib.session_init(SB);EnzoLib.session_set_boundary(h,0);EnzoLib.session_rebuild(h,0)
    EnzoLib.session_gravity(h,0)
    gmf=EnzoLib.problem_get_gravitating_mass(h,0);gd=size(gmf);Nc=128;b=(gd[1]-Nc)÷2
    gA=gmf[b+1:b+Nc,b+1:b+Nc,b+1:b+Nc];gz0=gA.-sum(gA)/length(gA)
    pos=EnzoLib.read_particles(h); mass=EnzoLib.read_particle_masses(h)
    @printf("npart=%d  mass: min=%.4e max=%.4e mean=%.4e std/mean=%.3e\n",
            length(mass),minimum(mass),maximum(mass),sum(mass)/length(mass),
            sqrt(sum((mass.-sum(mass)/length(mass)).^2)/length(mass))/(sum(mass)/length(mass)))
    ρ1=cic(pos,Nc,ones(length(mass)));z1=ρ1.-sum(ρ1)/length(ρ1)
    ρm=cic(pos,Nc,mass);zm=ρm.-sum(ρm)/length(ρm)
    @printf("weight-1 CIC vs GMF: corr=%.4f\n", corr(z1,gz0))
    @printf("mass-wt  CIC vs GMF: corr=%.4f\n", corr(zm,gz0))
    EnzoLib.free_problem(h)
end
