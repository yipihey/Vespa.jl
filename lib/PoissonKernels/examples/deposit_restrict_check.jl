# Test the AMR-recursion hypothesis for the root deposit: SB_amr refines the whole box,
# so Enzo deposits particles onto the FINE (256³) subgrid mesh and RESTRICTS to the
# root (128³), vs our direct 128³ CIC.  Compare both to Enzo's root GMF.
# Run: ENZOMODULES_GRID_LIB=<f32> <jl> --project=lib/PoissonKernels/test lib/PoissonKernels/examples/deposit_restrict_check.jl
using PoissonKernels, EnzoLib, Printf
const SB="/Users/tabel/Projects/enzo-dev/run/CosmologySimulation/SantaBarbaraCluster/SB_amr.enzo"
corr(o,e)=(om=o.-sum(o)/length(o);em=e.-sum(e)/length(e);sum(om.*em)/(sqrt(sum(om.^2)*sum(em.^2))+1e-300))
function cic(pos,N)
    ρ=zeros(N,N,N)
    @inbounds for p in 1:size(pos,1)
        gx=pos[p,1]*N-0.5;gy=pos[p,2]*N-0.5;gz=pos[p,3]*N-0.5
        i0=floor(Int,gx);fx=gx-i0;j0=floor(Int,gy);fy=gy-j0;k0=floor(Int,gz);fz=gz-k0
        for (di,wi) in ((0,1-fx),(1,fx)),(dj,wj) in ((0,1-fy),(1,fy)),(dk,wk) in ((0,1-fz),(1,fz))
            ρ[mod(i0+di,N)+1,mod(j0+dj,N)+1,mod(k0+dk,N)+1]+=wi*wj*wk
        end
    end
    ρ
end
restrict2(f)=(N=size(f,1)÷2; [sum(@view f[2i-1:2i,2j-1:2j,2k-1:2k]) for i in 1:N,j in 1:N,k in 1:N])  # 2³ sum
cd(dirname(SB)) do
    h=EnzoLib.session_init(SB); EnzoLib.session_set_boundary(h,0); EnzoLib.session_rebuild(h,0)
    EnzoLib.session_gravity(h,0); a,_=EnzoLib.session_cosmology(h)
    gmf=EnzoLib.problem_get_gravitating_mass(h,0); gd=size(gmf); Nc=128; b=(gd[1]-Nc)÷2
    gA=gmf[b+1:b+Nc,b+1:b+Nc,b+1:b+Nc]; gz0=gA.-sum(gA)/length(gA)
    pos=EnzoLib.read_particles(h)
    direct=cic(pos,Nc); dz=direct.-sum(direct)/length(direct)
    fine=cic(pos,2Nc); rest=restrict2(fine); rz=rest.-sum(rest)/length(rest)         # 256³ → 128³
    @printf("direct 128³ CIC      vs Enzo GMF: corr=%.4f\n", corr(dz,gz0))
    @printf("256³ CIC → restrict  vs Enzo GMF: corr=%.4f\n", corr(rz,gz0))
    EnzoLib.free_problem(h)
end
