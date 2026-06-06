# Compare OUR level-1 deposit-based subgrid source to Enzo's OWN subgrid
# GravitatingMassField (the definitive amplitude/convention check — no fitting).
# We build dmL1 = ciccc!(level-1 particles @2Nc) exactly as the production run,
# extract the buffered block under each subgrid, and test several candidate source
# forms against Enzo's GMF (corr + slope). The slope that gives 1.0 is the right
# convention; corr tells us the shape already matches.
# Run: ENZOMODULES_GRID_LIB=<f32> <jl> --project=lib/PoissonKernels/test lib/PoissonKernels/examples/my_subgrid_source_check.jl
using PoissonKernels, EnzoLib, Printf
const PK = PoissonKernels
const SB = "/Users/tabel/Projects/enzo-dev/run/CosmologySimulation/SantaBarbaraCluster/SB_amr.enzo"
const OMEGA_B=0.1; const OMEGA_CDM=0.9
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
    pos=EnzoLib.read_particles(h); Npart=size(pos,1)
    mass=EnzoLib.read_particle_masses(h)
    L1=findall(>(4.0),mass); NL1=length(L1)
    g0=EnzoLib.problem_grid_index_on_level(h,0,0); gd0=Tuple(Int.(EnzoLib.problem_grid_dims(h,g0))); Nc=gd0[1]-2NG
    Nf=2Nc
    dmL1=ciccc!(zeros(Nf,Nf,Nf),pos,Nf,L1)
    meanc_npart=Npart/Nf^3; meanc_nl1=NL1/Nf^3
    @printf("Nc=%d Nf=%d Npart=%d NL1=%d  meanc(Npart)=%.4f meanc(NL1)=%.4f  a=%.4f\n",
            Nc,Nf,Npart,NL1,meanc_npart,meanc_nl1,a)
    # Enzo GMF mean over whole root (density units)
    gmf0=EnzoLib.problem_get_gravitating_mass(h,0); @printf("mean(Enzo root GMF)=%.5f\n", sum(gmf0)/length(gmf0))

    n=EnzoLib.session_num_grids_on_level(h,1)
    idxs=[EnzoLib.problem_grid_index_on_level(h,1,i) for i in 0:n-1]
    # test the 3 biggest subgrids
    order=sort(idxs; by=gi->-prod(Int.(EnzoLib.problem_grid_dims(h,gi))))
    for g in order[1:min(3,length(order))]
        gd=Tuple(Int.(EnzoLib.problem_grid_dims(h,g))); na=gd.-2NG
        l,r=EnzoLib.problem_grid_edge(h,g); le=Float64.(l)
        dx=(Float64(r[1])-le[1])/na[1]
        gmf=EnzoLib.problem_get_gravitating_mass(h,g); gdm=size(gmf)
        bx=(gdm[1]-na[1])÷2; by=(gdm[2]-na[2])÷2; bz=(gdm[3]-na[3])÷2
        gmfA=gmf[bx+1:bx+na[1],by+1:by+na[2],bz+1:bz+na[3]]   # active GMF (na³)
        o=ntuple(dd->round(Int,le[dd]*Nf),3)                  # active block origin in 2Nc deposit
        myblk=[dmL1[mod(o[1]+i-1,Nf)+1,mod(o[2]+j-1,Nf)+1,mod(o[3]+k-1,Nf)+1] for i in 1:na[1],j in 1:na[2],k in 1:na[3]]
        @printf("\nsubgrid na=%s GMFdims=%s buf=(%d,%d,%d)  mean(EnzoGMF active)=%.5f  mean(myblk count)=%.4f\n",
                string(na),string(gdm),bx,by,bz,sum(gmfA)/length(gmfA),sum(myblk)/length(myblk))
        # candidate source forms vs Enzo's active GMF
        cands = [
          ("dmL1/meanc_npart"            , myblk./meanc_npart),
          ("Ωcdm·dmL1/meanc_npart"       , OMEGA_CDM.*myblk./meanc_npart),
          ("Ωcdm·(dmL1/meanc_npart - 1)" , OMEGA_CDM.*(myblk./meanc_npart .- 1)),
          ("dmL1/meanc_nl1"              , myblk./meanc_nl1),
          ("Ωcdm·dmL1/meanc_nl1"         , OMEGA_CDM.*myblk./meanc_nl1),
        ]
        for (nm,src) in cands
            @printf("  %-30s corr=%.5f  slope(enzo/ours)=%.4f\n", nm, corr(src,gmfA), slope(src,gmfA))
        end
    end
end
