# Task 2 end-to-end on a LIVE refined hierarchy: does our composite gravity
# (root FFT → parent-φ trilinear interp boundary → subgrid multigrid → comp_accel)
# match Enzo's OWN subgrid gravity on the 737 level-1 SB subgrids?
#
# Enzo reference: session_gravity(0) then session_gravity(1) populate the
# AccelerationField on the root + every subgrid (PrepareDensityField + SolveForPotential
# + ComputeAccelerations). We snapshot the subgrid accelerations, then recompute them
# with OUR kernels and report structure (Pearson corr) + best-fit slope per subgrid.
#
# Run: <julia> --project=lib/PPMKernels/test (ENZOMODULES_GRID_LIB=<f32>) \
#        lib/PoissonKernels/examples/sb_subgrid_gravity_check.jl
using PoissonKernels, EnzoLib, Printf
const PK = PoissonKernels
const SB = "/Users/tabel/Projects/enzo-dev/run/CosmologySimulation/SantaBarbaraCluster/SB_amr.enzo"
const OMEGA_B = 0.1; const OMEGA_CDM = 0.9
const G_GRAV = 1.0

active_of(flat, gd, ng) = (N=gd.-2ng; reshape(Float64.(flat), gd[1],gd[2],gd[3])[ng+1:ng+N[1], ng+1:ng+N[2], ng+1:ng+N[3]])

# trilinear sample of cell-centered periodic root φ (N³, cell i center=(i-0.5)/N)
function sample_cc(φr, N, x, y, z)
    gx=x*N-0.5; gy=y*N-0.5; gz=z*N-0.5
    i0=floor(Int,gx); fx=gx-i0; j0=floor(Int,gy); fy=gy-j0; k0=floor(Int,gz); fz=gz-k0
    w(i)=mod(i,N)+1
    @inbounds (φr[w(i0),w(j0),w(k0)]*(1-fx)*(1-fy)*(1-fz)+φr[w(i0+1),w(j0),w(k0)]*fx*(1-fy)*(1-fz)+
     φr[w(i0),w(j0+1),w(k0)]*(1-fx)*fy*(1-fz)+φr[w(i0+1),w(j0+1),w(k0)]*fx*fy*(1-fz)+
     φr[w(i0),w(j0),w(k0+1)]*(1-fx)*(1-fy)*fz+φr[w(i0+1),w(j0),w(k0+1)]*fx*(1-fy)*fz+
     φr[w(i0),w(j0+1),w(k0+1)]*(1-fx)*fy*fz+φr[w(i0+1),w(j0+1),w(k0+1)]*fx*fy*fz)
end
# CIC of ALL particles onto the root N³ (after rebuild, particles migrate to subgrids,
# so deposit(grid=0) is INCOMPLETE — the root source needs every particle).
function cic_all!(rho, pos, N)
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
# NON-periodic CIC of particles onto a subgrid's active n³ count field (mass=1), using
# the deposit's own cell-edge convention g=(pos-le)/dx but CLAMPED (no wrap) — the live
# bridge deposit periodic-wraps the active region, which misplaces edge particles on a
# (non-periodic) subgrid and corrupts the source. δ_dm = count/(N_part·dx³) is units-
# consistent with the root's count-based dm0/dmbar (uniform ⇒ 1 on both).
function cic_sub!(cnt, pos, le, dx, n)
    @inbounds for p in 1:size(pos,1)
        g0=(pos[p,1]-le[1])/dx; g1=(pos[p,2]-le[2])/dx; g2=(pos[p,3]-le[3])/dx
        (g0< -1||g0>n[1]||g1< -1||g1>n[2]||g2< -1||g2>n[3]) && continue
        i0=floor(Int,g0);fx=g0-i0;j0=floor(Int,g1);fy=g1-j0;k0=floor(Int,g2);fz=g2-k0
        for (di,wi) in ((0,1-fx),(1,fx)), (dj,wj) in ((0,1-fy),(1,fy)), (dk,wk) in ((0,1-fz),(1,fz))
            i=i0+di; j=j0+dj; k=k0+dk
            (1<=i+1<=n[1] && 1<=j+1<=n[2] && 1<=k+1<=n[3]) || continue
            cnt[i+1,j+1,k+1]+=wi*wj*wk
        end
    end; cnt
end
function corr_slope(x,y)
    xv=vec(x);yv=vec(y);mx=sum(xv)/length(xv);my=sum(yv)/length(yv);sxy=0.0;sxx=0.0;syy=0.0
    @inbounds for i in eachindex(xv,yv); dx=xv[i]-mx;dy=yv[i]-my;sxy+=dx*dy;sxx+=dx*dx;syy+=dy*dy; end
    (sxy/sqrt(sxx*syy+1e-300), sxy/(sxx+1e-300))
end

cd(dirname(SB)) do
    h = EnzoLib.session_init(SB); EnzoLib.session_set_boundary(h,0); EnzoLib.session_rebuild(h,0)
    be = PK.backend(:cpu); T=Float64
    gd0=Tuple(Int.(EnzoLib.problem_grid_dims(h,0))); NG=(gd0[1]-128)÷2; N=gd0[1]-2NG
    nsub = EnzoLib.session_num_grids_on_level(h,1)
    @printf("SB_amr: %d level-1 subgrids, root dims=%s (NG=%d). Composite gravity vs Enzo:\n", nsub, string(gd0), NG)

    # ── Enzo reference: gravity on root + subgrids; snapshot subgrid accel ──
    EnzoLib.session_gravity(h,0)
    er0=(active_of(EnzoLib.problem_get_acceleration(h,0,0),gd0,NG),   # root accel BEFORE level-1 gravity
         active_of(EnzoLib.problem_get_acceleration(h,1,0),gd0,NG),
         active_of(EnzoLib.problem_get_acceleration(h,2,0),gd0,NG))
    EnzoLib.session_gravity(h,1)
    sub_idx = [EnzoLib.problem_grid_index_on_level(h,1,i) for i in 0:nsub-1]
    enzo_a = Dict{Int,NTuple{3,Array{Float64,3}}}()
    for g in sub_idx
        gd=Tuple(Int.(EnzoLib.problem_grid_dims(h,g)))
        enzo_a[g]=(active_of(EnzoLib.problem_get_acceleration(h,0,g),gd,NG),
                   active_of(EnzoLib.problem_get_acceleration(h,1,g),gd,NG),
                   active_of(EnzoLib.problem_get_acceleration(h,2,g),gd,NG))
    end

    # ── our root FFT potential (cell-centered, active N³) + global means ──
    gas0=active_of(EnzoLib.read_density(h;grid=0),gd0,NG)
    posall=EnzoLib.read_particles(h); Npart=size(posall,1)       # ALL particles (root + subgrid source)
    dm0 =cic_all!(zeros(N,N,N), posall, N)
    gasbar=sum(gas0)/length(gas0); dmbar=sum(dm0)/length(dm0)
    δ0 = gas0.*(OMEGA_B/gasbar) .+ dm0.*(OMEGA_CDM/dmbar); δ0 ./= (sum(δ0)/length(δ0)); δ0 .-= 1.0
    # global mean DM density ρ̄_dm (= total particle mass, Vbox=1) in deposit units, so
    # the per-subgrid deposit (mass/cellvol) gives the SAME ρ_dm/ρ̄_dm ratio as the root's
    # count-based dm0/dmbar. m_p recovered from the root's own particles' deposit.
    nrp = EnzoLib.problem_num_particles(h,0)
    rdep = active_of(EnzoLib.deposit_particle_density(h;grid=0),gd0,NG)
    m_p = sum(rdep)*(1.0/N)^3/max(nrp,1)
    rho_dm_bar = m_p*sum(dm0)                  # total DM mass = ρ̄_dm
    φr=Array{T,3}(undef,N,N,N)
    PK.fft_poisson_root!(φr, δ0; G=G_GRAV, a=1.0, boxsize=1.0)   # ∇²φ=(G/a)δ on [0,1]³
    # EFFICIENT subgrid source: deposit ALL particles onto ONE global level-1 grid
    # (2N³) once; each subgrid (cells aligned to the level-1 mesh) extracts its block.
    # Replaces the O(Npart·Nsub) per-subgrid CIC with a single O(Npart) pass.
    Nf = 2N
    glb1 = cic_all!(zeros(Float64,Nf,Nf,Nf), posall, Nf)        # count per fine (level-1) cell
    # sanity: our root accel vs Enzo's root accel (should be ~0.995 like sb_root_gravity)
    let
        φpad=Array{T,3}(undef,gd0...)
        @inbounds for k in 1:gd0[3],j in 1:gd0[2],i in 1:gd0[1]
            φpad[i,j,k]=φr[mod(i-NG-1,N)+1,mod(j-NG-1,N)+1,mod(k-NG-1,N)+1]; end
        ar1=PK.device_zeros(be,T,(N,N,N));ar2=similar(ar1);ar3=similar(ar1)
        PK.comp_accel!(ar1,ar2,ar3,PK.to_device(be,φpad,T); iflag=1, start=(NG,NG,NG), del=(1.0/N,1.0/N,1.0/N))
        ov=vcat(vec(PK.to_host(ar1)),vec(PK.to_host(ar2)),vec(PK.to_host(ar3)))
        r0,s0=corr_slope(ov, vcat(vec(er0[1]),vec(er0[2]),vec(er0[3])))     # vs pre-level1 root accel
        er1=(active_of(EnzoLib.problem_get_acceleration(h,0,0),gd0,NG),
             active_of(EnzoLib.problem_get_acceleration(h,1,0),gd0,NG),
             active_of(EnzoLib.problem_get_acceleration(h,2,0),gd0,NG))
        r1,s1=corr_slope(ov, vcat(vec(er1[1]),vec(er1[2]),vec(er1[3])))     # vs post-level1 root accel
        @printf("  [sanity] root accel vs Enzo: pre-lvl1 corr=%.4f slope=%.3e | post-lvl1 corr=%.4f slope=%.3e\n", r0,s0,r1,s1)
    end

    # ── per-subgrid composite solve, compare to Enzo ──
    picks = nsub <= 24 ? (1:nsub) : round.(Int, range(1, nsub, length=24))
    cs=Float64[]; cpar=Float64[]; ss=Float64[]; print("  sampling $(length(picks)) subgrids: ")
    for ii in picks
        g = sub_idx[ii]
        gd=Tuple(Int.(EnzoLib.problem_grid_dims(h,g))); n=gd.-2NG
        le=zeros(Float64,3); re=zeros(Float64,3)
        l,r = EnzoLib.problem_grid_edge(h,g); le=Float64.(l); re=Float64.(r)
        dx = (re[1]-le[1])/n[1]
        # solve grid = active + 1 boundary ring (Dirichlet from parent φ)
        d = (n[1]+2, n[2]+2, n[3]+2)
        sol=zeros(T,d); rhs=zeros(T,d)
        # boundary: parent (root) φ interpolated to each ring cell center
        cc(a,i)=le[a]+(i-1-1+0.5)*dx        # solve-cell i (1-based, i=2 is first active) center; i=1 is ghost ring
        φpar=Array{T,3}(undef,d)                      # parent φ sampled on the WHOLE solve grid
        for k in 1:d[3], j in 1:d[2], i in 1:d[1]
            φpar[i,j,k]=sample_cc(φr,N,mod(cc(1,i),1.0),mod(cc(2,j),1.0),mod(cc(3,k),1.0))
        end
        for k in 1:d[3], j in 1:d[2], i in 1:d[1]      # Dirichlet boundary = parent φ
            (i==1||i==d[1]||j==1||j==d[2]||k==1||k==d[3]) && (sol[i,j,k]=φpar[i,j,k])
        end
        # source on active region (interior of solve grid): δ_sub, global-mean normalized
        gas=active_of(EnzoLib.read_density(h;grid=g),gd,NG)
        # subgrid DM source = block of the global level-1 deposit aligned to this subgrid
        # (the per-subgrid deposit(grid=g) is incomplete — needs ALL particles; this is the
        # efficient all-particle version: one global O(Npart) deposit, then cheap extraction).
        o1=round(Int,le[1]*Nf); o2=round(Int,le[2]*Nf); o3=round(Int,le[3]*Nf)
        cnt=[glb1[mod(o1+i-1,Nf)+1, mod(o2+j-1,Nf)+1, mod(o3+k-1,Nf)+1] for i in 1:n[1], j in 1:n[2], k in 1:n[3]]
        ρdm = cnt ./ (Npart*dx^3)                                        # ρ_dm/ρ̄_dm (uniform⇒1)
        δs = gas.*(OMEGA_B/gasbar) .+ ρdm.*OMEGA_CDM .- 1.0             # ρ/ρ̄-1, units-consistent w/ root
        fac = dx^2*(d[1]-1)*(d[2]-1)*(d[3]-1)*G_GRAV
        @inbounds rhs[2:d[1]-1,2:d[2]-1,2:d[3]-1] .= fac .* δs
        sd=PK.to_device(be,sol,T); rd=PK.to_device(be,rhs,T)
        PK.vcycle_solve!(sd,rd; cycle=:W, rtol=1e-8, maxcycles=80, dirichlet=true)
        φs=PK.to_host(sd)
        # comp_accel on the solve grid → accel on the active interior; start=(1,1,1) reads the ring
        a1=PK.device_zeros(be,T,n);a2=similar(a1);a3=similar(a1)
        PK.comp_accel!(a1,a2,a3, PK.to_device(be,φs,T); iflag=1, start=(1,1,1), del=(dx,dx,dx))
        oa=(PK.to_host(a1),PK.to_host(a2),PK.to_host(a3))
        # parent-only accel (no MG, no source): isolates boundary-interp + comp_accel
        p1=PK.device_zeros(be,T,n);p2=similar(p1);p3=similar(p1)
        PK.comp_accel!(p1,p2,p3, PK.to_device(be,φpar,T); iflag=1, start=(1,1,1), del=(dx,dx,dx))
        pa=(PK.to_host(p1),PK.to_host(p2),PK.to_host(p3))
        ea=enzo_a[g]
        ev=vcat(vec(ea[1]),vec(ea[2]),vec(ea[3]))
        r,s   = corr_slope(vcat(vec(oa[1]),vec(oa[2]),vec(oa[3])), ev)
        rp,sp = corr_slope(vcat(vec(pa[1]),vec(pa[2]),vec(pa[3])), ev)
        push!(cs, r); push!(cpar, rp); push!(ss, s); print(".")
    end
    println()
    @printf("  subgrid accel vs Enzo:  composite  median corr=%.4f  median SLOPE(enzo/ours)=%.4f  [min %.4f, max %.4f]\n",
            sort(cs)[(end+1)÷2], sort(ss)[(end+1)÷2], minimum(cs), maximum(cs))
    @printf("                          parent-only median corr=%.4f  [min %.4f, max %.4f]  (isolates boundary+comp_accel)\n",
            sort(cpar)[(end+1)÷2], minimum(cpar), maximum(cpar))
    EnzoLib.free_problem(h)
end
