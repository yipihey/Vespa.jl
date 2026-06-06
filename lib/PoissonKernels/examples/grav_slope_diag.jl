# Pin down the ~0.84 gravity-amplitude slope (EnzoNG accel ~19% stronger than Enzo)
# by decomposing the root Poisson source against Enzo's own root acceleration.
# Tests the "OmegaB / gas not correctly integrated" hypothesis: compare slopes for
# gas+DM, DM-only, gas-only, and physical-sum source compositions.
#
# Run: ENZOMODULES_GRID_LIB=<f32> <jl> --project=lib/PPMKernels/test lib/PoissonKernels/examples/grav_slope_diag.jl
using PoissonKernels, EnzoLib, Printf
const PK = PoissonKernels
const SB = "/Users/tabel/Projects/enzo-dev/run/CosmologySimulation/SantaBarbaraCluster/SB_amr.enzo"
const OMEGA_B = 0.1; const OMEGA_CDM = 0.9

cic!(rho,pos,N)=begin
    @inbounds for p in 1:size(pos,1)
        gx=mod(pos[p,1],1.0)*N;gy=mod(pos[p,2],1.0)*N;gz=mod(pos[p,3],1.0)*N
        i=floor(Int,gx);fx=gx-i;j=floor(Int,gy);fy=gy-j;k=floor(Int,gz);fz=gz-k
        i0=mod(i,N)+1;i1=mod(i+1,N)+1;j0=mod(j,N)+1;j1=mod(j+1,N)+1;k0=mod(k,N)+1;k1=mod(k+1,N)+1
        rho[i0,j0,k0]+=(1-fx)*(1-fy)*(1-fz);rho[i1,j0,k0]+=fx*(1-fy)*(1-fz)
        rho[i0,j1,k0]+=(1-fx)*fy*(1-fz);rho[i1,j1,k0]+=fx*fy*(1-fz)
        rho[i0,j0,k1]+=(1-fx)*(1-fy)*fz;rho[i1,j0,k1]+=fx*(1-fy)*fz
        rho[i0,j1,k1]+=(1-fx)*fy*fz;rho[i1,j1,k1]+=fx*fy*fz
    end; rho
end
function corr_slope(x,y)
    xv=vec(x);yv=vec(y);mx=sum(xv)/length(xv);my=sum(yv)/length(yv);sxy=0.;sxx=0.;syy=0.
    @inbounds for i in eachindex(xv,yv);dx=xv[i]-mx;dy=yv[i]-my;sxy+=dx*dy;sxx+=dx*dx;syy+=dy*dy;end
    (sxy/sqrt(sxx*syy+1e-300), sxy/(sxx+1e-300))
end

cd(dirname(SB)) do
    h=EnzoLib.session_init(SB); EnzoLib.session_set_boundary(h,0); EnzoLib.session_rebuild(h,0)
    be=PK.backend(:cpu); T=Float64
    gd=Tuple(Int.(EnzoLib.problem_grid_dims(h,0))); NG=(gd[1]-128)÷2; N=128
    act(flat)=Array(reshape(Float64.(flat),gd...)[NG+1:NG+N,NG+1:NG+N,NG+1:NG+N])
    EnzoLib.session_gravity(h,0)
    ea=(act(EnzoLib.problem_get_acceleration(h,0,0)),act(EnzoLib.problem_get_acceleration(h,1,0)),act(EnzoLib.problem_get_acceleration(h,2,0)))
    ev=vcat(vec.(ea)...)
    gas=act(EnzoLib.read_density(h;grid=0)); posall=EnzoLib.read_particles(h)
    dm=cic!(zeros(N,N,N),posall,N)
    gasbar=sum(gas)/length(gas); dmbar=sum(dm)/length(dm)
    @printf("SB_amr root: %d subgrids, gas mean=%.4f (Ω_b=%.2f), %d particles, dm mean(count)=%.4f\n",
            EnzoLib.session_num_grids_on_level(h,1), gasbar, OMEGA_B, size(posall,1), dmbar)

    # accel from a given zero-mean source δ, via FFT + padded comp_accel; slope vs Enzo
    function accel_slope(δ; label)
        φ=Array{T,3}(undef,N,N,N); PK.fft_poisson_root!(φ, δ; G=1.0,a=1.0,boxsize=1.0)
        φp=Array{T,3}(undef,gd...)
        @inbounds for k in 1:gd[3],j in 1:gd[2],i in 1:gd[1]; φp[i,j,k]=φ[mod(i-NG-1,N)+1,mod(j-NG-1,N)+1,mod(k-NG-1,N)+1]; end
        a1=PK.device_zeros(be,T,(N,N,N));a2=similar(a1);a3=similar(a1)
        PK.comp_accel!(a1,a2,a3,PK.to_device(be,φp,T);iflag=1,start=(NG,NG,NG),del=(1/N,1/N,1/N))
        ov=vcat(vec(PK.to_host(a1)),vec(PK.to_host(a2)),vec(PK.to_host(a3)))
        r,s=corr_slope(ov,ev)
        @printf("  %-28s corr=%.4f  slope(enzo/ours)=%.4f\n", label, r, s)
    end

    zm(x)=(y=copy(x); y .-= sum(y)/length(y); y)
    # current composite source
    accel_slope(zm(gas.*(OMEGA_B/gasbar) .+ dm.*(OMEGA_CDM/dmbar)); label="gas·Ωb/ḡ + dm·Ωcdm/d̄  (current)")
    # DM only
    accel_slope(zm(dm.*(OMEGA_CDM/dmbar)); label="DM only (×Ωcdm)")
    # gas only
    accel_slope(zm(gas.*(OMEGA_B/gasbar)); label="gas only (×Ωb)")
    # physical sum: gas raw (mean already Ω_b) + DM normalized to Ω_cdm — no /gasbar
    accel_slope(zm(gas .+ dm.*(OMEGA_CDM/dmbar)); label="gas(raw) + dm·Ωcdm/d̄")
    # total-matter (DM carries ALL mass, gas omitted) — tests double-count
    accel_slope(zm(dm./dmbar); label="DM as total matter (no gas)")
    # gas + DM each as full overdensity then averaged by Ω (δ = Ωb·δ_gas + Ωcdm·δ_dm)
    accel_slope(OMEGA_B.*zm(gas./gasbar) .+ OMEGA_CDM.*zm(dm./dmbar); label="Ωb·δ_gas + Ωcdm·δ_dm")

    # Enzo deposits subgrid particles at FINE res then projects to root (smoother).
    # Mimic it: deposit all at 256³, restrict (2×2×2 average) to 128³.
    glb=cic!(zeros(2N,2N,2N),posall,2N)
    dmproj=Array{Float64,3}(undef,N,N,N)
    @inbounds for k in 1:N,j in 1:N,i in 1:N
        s=0.0; for dk in 0:1,dj in 0:1,di in 0:1; s+=glb[2i-1+di,2j-1+dj,2k-1+dk]; end; dmproj[i,j,k]=s/8
    end
    accel_slope(zm(gas.*(OMEGA_B/gasbar) .+ dmproj.*(OMEGA_CDM/(sum(dmproj)/length(dmproj)))); label="DM 256³-deposit → projected to 128³")

    # Refined-vs-unrefined split: is our root accel right where Enzo is ALSO coarse
    # (unrefined cells) and only "off" in refined cells (where Enzo's root accel is a
    # placeholder, real force from the subgrid)? Mask root cells covered by a subgrid.
    refined = falses(N,N,N)
    for i in 0:EnzoLib.session_num_grids_on_level(h,1)-1
        g=EnzoLib.problem_grid_index_on_level(h,1,i); l,r=EnzoLib.problem_grid_edge(h,g)
        a=ntuple(d->clamp(round(Int,Float64(l[d])*N)+1,1,N),3); b=ntuple(d->clamp(round(Int,Float64(r[d])*N),1,N),3)
        @inbounds refined[a[1]:b[1],a[2]:b[2],a[3]:b[3]] .= true
    end
    δfull=zm(gas.*(OMEGA_B/gasbar).+dm.*(OMEGA_CDM/dmbar))
    φ=Array{T,3}(undef,N,N,N); PK.fft_poisson_root!(φ,δfull;G=1.0,a=1.0,boxsize=1.0)
    φp=Array{T,3}(undef,gd...)
    @inbounds for k in 1:gd[3],j in 1:gd[2],i in 1:gd[1]; φp[i,j,k]=φ[mod(i-NG-1,N)+1,mod(j-NG-1,N)+1,mod(k-NG-1,N)+1]; end
    a1=PK.device_zeros(be,T,(N,N,N));a2=similar(a1);a3=similar(a1)
    PK.comp_accel!(a1,a2,a3,PK.to_device(be,φp,T);iflag=1,start=(NG,NG,NG),del=(1/N,1/N,1/N))
    o1=PK.to_host(a1);o2=PK.to_host(a2);o3=PK.to_host(a3)
    mu=.!refined
    @printf("  refined fraction of root = %.1f%%\n", 100*sum(refined)/length(refined))
    ru,su=corr_slope(vcat(o1[mu],o2[mu],o3[mu]), vcat(ea[1][mu],ea[2][mu],ea[3][mu]))
    rr,sr=corr_slope(vcat(o1[refined],o2[refined],o3[refined]), vcat(ea[1][refined],ea[2][refined],ea[3][refined]))
    @printf("  UNREFINED cells: corr=%.4f slope=%.4f  |  REFINED cells: corr=%.4f slope=%.4f\n", ru,su,rr,sr)
    EnzoLib.free_problem(h)
end

# Control: SAME source/accel WITHOUT rebuild (no subgrids, all particles on root).
# If slope→~1.0, the 0.84 is purely the AMR coarse↔fine reference, not our source.
println("\n--- control: NO rebuild (single root grid, all particles on root) ---")
cd(dirname(SB)) do
    h=EnzoLib.session_init(SB); EnzoLib.session_set_boundary(h,0)   # NO session_rebuild
    be=PK.backend(:cpu); T=Float64
    gd=Tuple(Int.(EnzoLib.problem_grid_dims(h,0))); NG=(gd[1]-128)÷2; N=128
    act(flat)=Array(reshape(Float64.(flat),gd...)[NG+1:NG+N,NG+1:NG+N,NG+1:NG+N])
    EnzoLib.session_gravity(h,0)
    ea=(act(EnzoLib.problem_get_acceleration(h,0,0)),act(EnzoLib.problem_get_acceleration(h,1,0)),act(EnzoLib.problem_get_acceleration(h,2,0)))
    ev=vcat(vec.(ea)...)
    gas=act(EnzoLib.read_density(h;grid=0)); dm=cic!(zeros(N,N,N),EnzoLib.read_particles(h),N)
    gasbar=sum(gas)/length(gas); dmbar=sum(dm)/length(dm)
    δ=copy(gas.*(OMEGA_B/gasbar) .+ dm.*(OMEGA_CDM/dmbar)); δ .-= sum(δ)/length(δ)
    φ=Array{T,3}(undef,N,N,N); PK.fft_poisson_root!(φ,δ;G=1.0,a=1.0,boxsize=1.0)
    φp=Array{T,3}(undef,gd...)
    @inbounds for k in 1:gd[3],j in 1:gd[2],i in 1:gd[1]; φp[i,j,k]=φ[mod(i-NG-1,N)+1,mod(j-NG-1,N)+1,mod(k-NG-1,N)+1]; end
    a1=PK.device_zeros(be,T,(N,N,N));a2=similar(a1);a3=similar(a1)
    PK.comp_accel!(a1,a2,a3,PK.to_device(be,φp,T);iflag=1,start=(NG,NG,NG),del=(1/N,1/N,1/N))
    r,s=corr_slope(vcat(vec(PK.to_host(a1)),vec(PK.to_host(a2)),vec(PK.to_host(a3))),ev)
    @printf("  NO-rebuild root: %d subgrids, corr=%.4f slope(enzo/ours)=%.4f\n",
            EnzoLib.session_num_grids_on_level(h,1), r, s)
    EnzoLib.free_problem(h)
end
