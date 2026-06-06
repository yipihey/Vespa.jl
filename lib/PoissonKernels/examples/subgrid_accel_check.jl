# Full subgrid gravity chain vs Enzo's OWN subgrid AccelerationField (no fitting).
# Source = OUR level-1 deposit (verified corr/slope 1.0 vs Enzo GMF), boundary =
# trilinear(root φ) on the buffered mesh (production convention). Solve with our MG,
# compute accel with several del conventions, compare to Enzo's AccelerationField.
# The convention that gives corr 1.0 / slope 1.0 is the right one.
# Run: ENZOMODULES_GRID_LIB=<f32> <jl> --project=lib/PoissonKernels/test lib/PoissonKernels/examples/subgrid_accel_check.jl
using PoissonKernels, EnzoLib, Printf
const PK = PoissonKernels
const SB = "/Users/tabel/Projects/enzo-dev/run/CosmologySimulation/SantaBarbaraCluster/SB_amr.enzo"
const OMEGA_CDM=0.9
slope(o,e)=sum(vec(o).*vec(e))/sum(vec(o).^2)
corr(o,e)=(om=o.-sum(o)/length(o);em=e.-sum(e)/length(e);sum(om.*em)/(sqrt(sum(om.^2)*sum(em.^2))+1e-300))
ciccc!(rho,pos,N,idx)=begin
    @inbounds for p in idx
        gx=mod(pos[p,1],1.0)*N-0.5;gy=mod(pos[p,2],1.0)*N-0.5;gz=mod(pos[p,3],1.0)*N-0.5
        i=floor(Int,gx);fx=gx-i;j=floor(Int,gy);fy=gy-j;k=floor(Int,gz);fz=gz-k
        i0=mod(i,N)+1;i1=mod(i+1,N)+1;j0=mod(j,N)+1;j1=mod(j+1,N)+1;k0=mod(k,N)+1;k1=mod(k+1,N)+1
        rho[i0,j0,k0]+=(1-fx)*(1-fy)*(1-fz);rho[i1,j0,k0]+=fx*(1-fy)*(1-fz)
        rho[i0,j1,k0]+=(1-fx)*fy*(1-fz);rho[i1,j1,k0]+=fx*fy*(1-fz)
        rho[i0,j0,k1]+=(1-fx)*(1-fy)*fz;rho[i1,j0,k1]+=fx*(1-fy)*fz
        rho[i0,j1,k1]+=(1-fx)*fy*fz;rho[i1,j1,k1]+=fx*fy*fz
    end; rho
end

cd(dirname(SB)) do
    h=EnzoLib.session_init(SB); EnzoLib.session_set_boundary(h,0); EnzoLib.session_rebuild(h,0)
    EnzoLib.session_gravity(h,0); EnzoLib.session_gravity(h,1)
    a,_=EnzoLib.session_cosmology(h); NG=3
    pos=EnzoLib.read_particles(h); Npart=size(pos,1); mass=EnzoLib.read_particle_masses(h)
    L0=findall(<(4.0),mass); L1=findall(>(4.0),mass)
    g0=EnzoLib.problem_grid_index_on_level(h,0,0); gd0=Tuple(Int.(EnzoLib.problem_grid_dims(h,g0))); Nc=gd0[1]-2NG
    Nf=2Nc; meanc=Npart/Nf^3
    dmL1=ciccc!(zeros(Nf,Nf,Nf),pos,Nf,L1)
    # root overdensity δ (same field the root FFT solved) for buffer-fill (parent interpolation)
    rsum2(f)=(N=size(f,1)÷2; [sum(@view f[2i-1:2i,2j-1:2j,2k-1:2k]) for i in 1:N,j in 1:N,k in 1:N])
    OMEGA_B=0.1
    gasR=reshape(Float64.(EnzoLib.read_density(h;grid=g0)),gd0...)[NG+1:NG+Nc,NG+1:NG+Nc,NG+1:NG+Nc]
    gasbar=sum(gasR)/length(gasR)
    dmroot=ciccc!(zeros(Nc,Nc,Nc),pos,Nc,L0).+rsum2(dmL1); dmbar=sum(dmroot)/length(dmroot)
    δroot=gasR.*(OMEGA_B/gasbar).+dmroot.*(OMEGA_CDM/dmbar); δroot./=(sum(δroot)/length(δroot)); δroot.-=1.0
    sccδ(x,y,z)=begin                         # trilinear sample of cell-centered periodic δroot (Nc³)
        gx=x*Nc-0.5;gy=y*Nc-0.5;gz=z*Nc-0.5
        i0=floor(Int,gx);fx=gx-i0;j0=floor(Int,gy);fy=gy-j0;k0=floor(Int,gz);fz=gz-k0; w(i)=mod(i,Nc)+1
        @inbounds δroot[w(i0),w(j0),w(k0)]*(1-fx)*(1-fy)*(1-fz)+δroot[w(i0+1),w(j0),w(k0)]*fx*(1-fy)*(1-fz)+
         δroot[w(i0),w(j0+1),w(k0)]*(1-fx)*fy*(1-fz)+δroot[w(i0+1),w(j0+1),w(k0)]*fx*fy*(1-fz)+
         δroot[w(i0),w(j0),w(k0+1)]*(1-fx)*(1-fy)*fz+δroot[w(i0+1),w(j0),w(k0+1)]*fx*(1-fy)*fz+
         δroot[w(i0),w(j0+1),w(k0+1)]*(1-fx)*fy*fz+δroot[w(i0+1),w(j0+1),w(k0+1)]*fx*fy*fz
    end
    # root φ (active) for the parent boundary
    rootpot=EnzoLib.problem_get_potential(h,0); rgd=size(rootpot); rb=(rgd[1]-Nc)÷2
    φroot=rootpot[rb+1:rb+Nc,rb+1:rb+Nc,rb+1:rb+Nc]
    scc(x,y,z)=begin
        gx=x*Nc-0.5;gy=y*Nc-0.5;gz=z*Nc-0.5
        i0=floor(Int,gx);fx=gx-i0;j0=floor(Int,gy);fy=gy-j0;k0=floor(Int,gz);fz=gz-k0; w(i)=mod(i,Nc)+1
        @inbounds φroot[w(i0),w(j0),w(k0)]*(1-fx)*(1-fy)*(1-fz)+φroot[w(i0+1),w(j0),w(k0)]*fx*(1-fy)*(1-fz)+
         φroot[w(i0),w(j0+1),w(k0)]*(1-fx)*fy*(1-fz)+φroot[w(i0+1),w(j0+1),w(k0)]*fx*fy*(1-fz)+
         φroot[w(i0),w(j0),w(k0+1)]*(1-fx)*(1-fy)*fz+φroot[w(i0+1),w(j0),w(k0+1)]*fx*(1-fy)*fz+
         φroot[w(i0),w(j0+1),w(k0+1)]*(1-fx)*fy*fz+φroot[w(i0+1),w(j0+1),w(k0+1)]*fx*fy*fz
    end
    be=PK.backend(:cpu); T=Float64
    n=EnzoLib.session_num_grids_on_level(h,1)
    idxs=[EnzoLib.problem_grid_index_on_level(h,1,i) for i in 0:n-1]
    g=argmax(gi->prod(Int.(EnzoLib.problem_grid_dims(h,gi))), idxs)
    gd=Tuple(Int.(EnzoLib.problem_grid_dims(h,g))); na=gd.-2NG
    l,r=EnzoLib.problem_grid_edge(h,g); le=Float64.(l); dx=(Float64(r[1])-le[1])/na[1]
    bw=6; D=(na[1]+2bw,na[2]+2bw,na[3]+2bw); o=ntuple(dd->round(Int,le[dd]*Nf),3)
    ccb(aa,ii)=le[aa]-bw*dx+(ii-0.5)*dx
    inactive(i,j,k)=(bw<i<=bw+na[1] && bw<j<=bw+na[2] && bw<k<=bw+na[3])
    mydep(i,j,k)=OMEGA_CDM*(dmL1[mod(o[1]-bw+i-1,Nf)+1,mod(o[2]-bw+j-1,Nf)+1,mod(o[3]-bw+k-1,Nf)+1]/meanc-1.0)
    # Enzo's full buffered GMF (reference source)
    egmf=EnzoLib.problem_get_gravitating_mass(h,g)
    # three source variants on the buffered mesh D
    srcA=[mydep(i,j,k) for i in 1:D[1],j in 1:D[2],k in 1:D[3]]                 # my deposit everywhere (current prod)
    srcB=egmf                                                                  # Enzo GMF (correct buffer) — reference
    srcC=[inactive(i,j,k) ? mydep(i,j,k) :                                      # my deposit in active …
          sccδ(mod(ccb(1,i),1.0),mod(ccb(2,j),1.0),mod(ccb(3,k),1.0))          # … trilinear(root δ) in buffer
          for i in 1:D[1],j in 1:D[2],k in 1:D[3]]
    # boundary = trilinear(root φ) on the buffered edge (production convention)
    solBC=zeros(T,D)
    for k in 1:D[3],j in 1:D[2],i in 1:D[1]
        (i==1||i==D[1]||j==1||j==D[2]||k==1||k==D[3]) &&
            (solBC[i,j,k]=scc(mod(ccb(1,i),1.0),mod(ccb(2,j),1.0),mod(ccb(3,k),1.0)))
    end
    fac=dx^2*(D[1]-1)*(D[2]-1)*(D[3]-1)
    φeA=EnzoLib.problem_get_potential(h,g)[bw+1:bw+na[1],bw+1:bw+na[2],bw+1:bw+na[3]]
    actv(flat)=(Array(reshape(Float64.(flat),gd...)[NG+1:NG+na[1],NG+1:NG+na[2],NG+1:NG+na[3]]))
    eA1=actv(EnzoLib.problem_get_acceleration(h,0,g))
    @printf("subgrid na=%s dx=%.5e a=%.4f  |enzo accel|max=%.3e\n", string(na),dx,a,maximum(abs,eA1))
    Rb=(bw+1:bw+na[1], bw+1:bw+na[2], bw+1:bw+na[3])
    for (nm,src) in [("A my-deposit everywhere",srcA),("B Enzo-GMF (ref)",srcB),("C my-active + root-δ buffer",srcC)]
        rhs=zeros(T,D); @inbounds rhs[2:D[1]-1,2:D[2]-1,2:D[3]-1] .= fac.*(src[2:D[1]-1,2:D[2]-1,2:D[3]-1]./a)
        sd=PK.to_device(be,copy(solBC),T); rd=PK.to_device(be,rhs,T)
        PK.vcycle_solve!(sd,rd; cycle=:W, rtol=1e-8, maxcycles=80, dirichlet=true)
        φB=Float64.(PK.to_host(sd))
        d1=PK.device_zeros(be,T,na);d2=similar(d1);d3=similar(d1)
        PK.comp_accel!(d1,d2,d3, PK.to_device(be,φB,T); iflag=1, start=(bw,bw,bw), del=(dx,dx,dx))
        o1=Float64.(PK.to_host(d1))./a^2
        @printf("  [%-26s] φ corr=%.5f slope=%.4f | accelx corr=%.5f slope=%.4f |our|max=%.3e\n",
                nm, corr(φB[Rb...],φeA), slope(φB[Rb...],φeA), corr(o1,eA1), slope(o1,eA1), maximum(abs,o1))
    end
end
