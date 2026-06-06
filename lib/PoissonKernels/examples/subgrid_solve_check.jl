# Isolate the subgrid MG solve: feed Enzo's OWN subgrid GravitatingMassField (source)
# AND Enzo's OWN subgrid potential at the boundary ring, solve with our multigrid,
# compare interior φ to Enzo's PotentialField. Identical source+boundary ⇒ any mismatch
# is purely our MG operator/normalization; a match ⇒ the run's +6% was the BOUNDARY
# interpolation (our trilinear root-φ ≠ Enzo's parent-potential interpolation).
# Run: ENZOMODULES_GRID_LIB=<f32> <jl> --project=lib/PoissonKernels/test lib/PoissonKernels/examples/subgrid_solve_check.jl
using PoissonKernels, EnzoLib, Printf
const PK = PoissonKernels
const SB = "/Users/tabel/Projects/enzo-dev/run/CosmologySimulation/SantaBarbaraCluster/SB_amr.enzo"
slope(o,e)=sum(vec(o).*vec(e))/sum(vec(o).^2)
corr(o,e)=(om=o.-sum(o)/length(o);em=e.-sum(e)/length(e);sum(om.*em)/(sqrt(sum(om.^2)*sum(em.^2))+1e-300))

cd(dirname(SB)) do
    h=EnzoLib.session_init(SB); EnzoLib.session_set_boundary(h,0); EnzoLib.session_rebuild(h,0)
    EnzoLib.session_gravity(h,0); EnzoLib.session_gravity(h,1)        # populate subgrid GMF + potential
    a,_ = EnzoLib.session_cosmology(h); NG=3
    n=EnzoLib.session_num_grids_on_level(h,1)
    idxs=[EnzoLib.problem_grid_index_on_level(h,1,i) for i in 0:n-1]
    g=argmax(gi->prod(Int.(EnzoLib.problem_grid_dims(h,gi))), idxs)
    gd=Tuple(Int.(EnzoLib.problem_grid_dims(h,g))); na=gd.-2NG
    l,r=EnzoLib.problem_grid_edge(h,g); dx=(Float64(r[1])-Float64(l[1]))/na[1]
    gmf=EnzoLib.problem_get_gravitating_mass(h,g); pot=EnzoLib.problem_get_potential(h,g); gdm=size(gmf)
    bx=(gdm[1]-na[1])÷2; by=(gdm[2]-na[2])÷2; bz=(gdm[3]-na[3])÷2
    @printf("subgrid g: na=%s GMFdims=%s buffer=(%d,%d,%d) dx=%.5e a=%.4f\n", string(na),string(gdm),bx,by,bz,dx,a)
    # active + 1 ring (na+2) blocks
    Rx=bx:bx+na[1]+1; Ry=by:by+na[2]+1; Rz=bz:bz+na[3]+1     # 1-based, includes the ring just outside active
    src = gmf[bx+1:bx+na[1], by+1:by+na[2], bz+1:bz+na[3]]    # active source (na³)
    potR = pot[Rx,Ry,Rz]                                       # Enzo potential, active+ring ((na+2)³)
    d=(na[1]+2,na[2]+2,na[3]+2)
    # our MG: Dirichlet boundary = Enzo's potential ring; rhs = fac·(coef·src), coef=1/a
    be=PK.backend(:cpu); T=Float64
    sol=zeros(T,d); sol[1,:,:].=potR[1,:,:]; sol[end,:,:].=potR[end,:,:]
    sol[:,1,:].=potR[:,1,:]; sol[:,end,:].=potR[:,end,:]; sol[:,:,1].=potR[:,:,1]; sol[:,:,end].=potR[:,:,end]
    fac=dx^2*(d[1]-1)*(d[2]-1)*(d[3]-1)
    rhs=zeros(T,d); @inbounds rhs[2:d[1]-1,2:d[2]-1,2:d[3]-1] .= fac.*(src./a)
    sd=PK.to_device(be,sol,T); rd=PK.to_device(be,rhs,T)
    PK.vcycle_solve!(sd,rd; cycle=:W, rtol=1e-8, maxcycles=60, dirichlet=true)
    φo=Float64.(PK.to_host(sd))[2:d[1]-1,2:d[2]-1,2:d[3]-1]
    φe=potR[2:d[1]-1,2:d[2]-1,2:d[3]-1]
    @printf("(A) 1-ring + Enzo-ring boundary:  corr=%.5f  slope=%.5f\n", corr(φo,φe), slope(φo,φe))

    # (B) BUFFERED solve like Enzo: full GMF mesh (52³), Dirichlet boundary at the buffer
    #     edge = trilinear(root φ).  Tests whether a COARSE parent boundary 6 cells out
    #     (with the buffer resolving local structure) reproduces Enzo's active potential.
    rootpot = EnzoLib.problem_get_potential(h, 0)               # Enzo root potential (buffered)
    rgd=size(rootpot); rb=(rgd[1]-128)÷2; Nc=128
    φroot = rootpot[rb+1:rb+Nc, rb+1:rb+Nc, rb+1:rb+Nc]         # active root φ (128³)
    function scc(x,y,z)                                          # cell-centered trilinear, periodic
        gx=x*Nc-0.5; gy=y*Nc-0.5; gz=z*Nc-0.5
        i0=floor(Int,gx);fx=gx-i0;j0=floor(Int,gy);fy=gy-j0;k0=floor(Int,gz);fz=gz-k0; w(i)=mod(i,Nc)+1
        @inbounds φroot[w(i0),w(j0),w(k0)]*(1-fx)*(1-fy)*(1-fz)+φroot[w(i0+1),w(j0),w(k0)]*fx*(1-fy)*(1-fz)+
         φroot[w(i0),w(j0+1),w(k0)]*(1-fx)*fy*(1-fz)+φroot[w(i0+1),w(j0+1),w(k0)]*fx*fy*(1-fz)+
         φroot[w(i0),w(j0),w(k0+1)]*(1-fx)*(1-fy)*fz+φroot[w(i0+1),w(j0),w(k0+1)]*fx*(1-fy)*fz+
         φroot[w(i0),w(j0+1),w(k0+1)]*(1-fx)*fy*fz+φroot[w(i0+1),w(j0+1),w(k0+1)]*fx*fy*fz
    end
    D=gdm                                                       # full buffered GMF size (52³)
    leb=(Float64(l[1])-bx*dx, Float64(l[2])-by*dx, Float64(l[3])-bz*dx)  # buffered-mesh left edge
    solB=zeros(T,D)
    for k in 1:D[3],j in 1:D[2],i in 1:D[1]
        (i==1||i==D[1]||j==1||j==D[2]||k==1||k==D[3]) &&
            (solB[i,j,k]=scc(mod(leb[1]+(i-0.5)*dx,1.0),mod(leb[2]+(j-0.5)*dx,1.0),mod(leb[3]+(k-0.5)*dx,1.0)))
    end
    facB=dx^2*(D[1]-1)*(D[2]-1)*(D[3]-1)
    rhsB=zeros(T,D); @inbounds rhsB[2:D[1]-1,2:D[2]-1,2:D[3]-1] .= facB.*(gmf[2:D[1]-1,2:D[2]-1,2:D[3]-1]./a)
    sdB=PK.to_device(be,solB,T); rdB=PK.to_device(be,rhsB,T)
    PK.vcycle_solve!(sdB,rdB; cycle=:W, rtol=1e-8, maxcycles=80, dirichlet=true)
    φB=Float64.(PK.to_host(sdB))[bx+1:bx+na[1], by+1:by+na[2], bz+1:bz+na[3]]   # active
    φeA=pot[bx+1:bx+na[1], by+1:by+na[2], bz+1:bz+na[3]]
    @printf("(B) BUFFERED + coarse parent boundary:  corr=%.5f  slope(enzo/ours)=%.5f\n", corr(φB,φeA), slope(φB,φeA))
    EnzoLib.free_problem(h)
end
