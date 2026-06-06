# Santa Barbara cluster with EnzoNG composite gravity (root FFT + per-subgrid
# multigrid) AND PPM hydro as :julia slots, under Enzo's OWN AMR hierarchy.
# This is the validated composite-gravity path (sb_subgrid_gravity_check.jl: corr
# 0.95 vs Enzo on 737 real subgrids) wired into the live time loop.
#
#   :gravity → level 0: δ (gas + ALL-particle DM) → fft_poisson_root! → store φ_root
#              level≥1: per subgrid, source = all-particle deposit (global grid for
#                       level 1) → parent-φ interpolated Dirichlet boundary →
#                       vcycle_solve!(dirichlet=true) → comp_accel → AccelerationField
#   :hydro   → Enzo PPM (gravity-coupled, dual-energy), per grid, on BACKEND.
#
# Run (env BACKEND=cpu|metal, ENZOMODULES_GRID_LIB=<f32>):
#   <julia> --project=lib/PPMKernels/test lib/EnzoLib/examples/sb_composite_run.jl [cycles]

using EnzoLib, PPMKernels, PoissonKernels, Printf
try; @eval using Metal; catch; end

const SBDIR = "/Users/tabel/Projects/enzo-dev/run/CosmologySimulation/SantaBarbaraCluster"
const NG = 4                       # ppm_step_3d! ghost zones (param file forced to 4)
const GAMMA = 5/3
const OMEGA_B = 0.1; const OMEGA_CDM = 0.9
const iD, iV1, iV2, iV3, iTE, iGE = 0, 1, 2, 3, 4, 5
const BE = Symbol(get(ENV, "BACKEND", "cpu"))
const T  = Float32
const GFIX = get(ENV, "GFIX", "0") == "1"   # operator-split gravity (couples g into the mass flux; insufficient alone)
const afac = Ref(1.0)                        # expansion factor a (proper cell width = a·comoving dx)
const GRAVA = get(ENV, "GRAVA", "1") == "1"  # apply Enzo's comoving gravity coupling to the composite accel
# DERIVED (not fit) from enzo_gravity_steps.jl against Enzo's own GravitatingMassField/PotentialField:
#   Poisson coef = G_eff/a with G_eff=1.0 EXACTLY (Enzo's 4π cancels the FFT/Green's norm; corr=1.0),
#   accel gradient uses proper cell width del=a·dx (corr=1.0).  Net: accel ×= 1.0/a².  Constant is 1.0.
const GRAVK = parse(Float64, get(ENV, "GRAVK", "1.0"))
gravfac() = GRAVA ? GRAVK/afac[]^2 : 1.0     # ≡ coef=1/a (potential) × 1/a (proper-dx gradient)
const GMFSRC = get(ENV, "GMFSRC", "0") == "1"  # option (b): use Enzo's GravitatingMassField as the root source
const SUBENZO = get(ENV, "SUBENZO", "0") == "1"  # isolation: use Enzo's subgrid accel directly (skip our subgrid solve)
                                               # (our verified-exact GPU Poisson+accel ⇒ matches Enzo's gravity)
const _step = Ref(0)

active_of(flat, gd) = (n=gd.-2NG; Array(reshape(Float64.(flat), gd...)[NG+1:NG+n[1], NG+1:NG+n[2], NG+1:NG+n[3]]))
place_active(act, gd) = (full=zeros(Float64, gd...); n=size(act); full[NG+1:NG+n[1],NG+1:NG+n[2],NG+1:NG+n[3]].=act; vec(full))
function pad_periodic(φ)
    n=size(φ,1); M=n+2NG; full=Array{Float64,3}(undef,M,M,M)
    @inbounds for k in 1:M,j in 1:M,i in 1:M
        full[i,j,k]=φ[mod(i-NG-1,n)+1,mod(j-NG-1,n)+1,mod(k-NG-1,n)+1]; end
    full
end
cic!(rho,pos,Nc)=begin
    @inbounds for p in 1:size(pos,1)
        gx=mod(pos[p,1],1.0)*Nc;gy=mod(pos[p,2],1.0)*Nc;gz=mod(pos[p,3],1.0)*Nc
        i=floor(Int,gx);fx=gx-i;j=floor(Int,gy);fy=gy-j;k=floor(Int,gz);fz=gz-k
        i0=mod(i,Nc)+1;i1=mod(i+1,Nc)+1;j0=mod(j,Nc)+1;j1=mod(j+1,Nc)+1;k0=mod(k,Nc)+1;k1=mod(k+1,Nc)+1
        rho[i0,j0,k0]+=(1-fx)*(1-fy)*(1-fz);rho[i1,j0,k0]+=fx*(1-fy)*(1-fz)
        rho[i0,j1,k0]+=(1-fx)*fy*(1-fz);rho[i1,j1,k0]+=fx*fy*(1-fz)
        rho[i0,j0,k1]+=(1-fx)*(1-fy)*fz;rho[i1,j0,k1]+=fx*(1-fy)*fz
        rho[i0,j1,k1]+=(1-fx)*fy*fz;rho[i1,j1,k1]+=fx*fy*fz
    end; rho
end
# cell-centered CIC (matches Enzo cic_deposit: gx=pos·N-0.5) over particle subset idx
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
rsum2(f)=(N=size(f,1)÷2; [sum(@view f[2i-1:2i,2j-1:2j,2k-1:2k]) for i in 1:N,j in 1:N,k in 1:N])  # 2³ restrict-sum
# level-split deposit onto N³: level-0 particles @N, level-1 @2N restrict-summed (matches Enzo GMF, corr 1.0)
function leveldep(pos, mass, N)
    L0=findall(<(4.0),mass); L1=findall(>(4.0),mass)   # mass 0.9⇒L0, 7.2⇒L1 (ratio 8=2³ = per-level deposit scaling)
    ciccc!(zeros(N,N,N),pos,N,L0) .+ rsum2(ciccc!(zeros(2N,2N,2N),pos,2N,L1))
end
function sample_cc(φr,Nc,x,y,z)            # trilinear sample of cell-centered periodic φr (Nc³)
    gx=x*Nc-0.5;gy=y*Nc-0.5;gz=z*Nc-0.5
    i0=floor(Int,gx);fx=gx-i0;j0=floor(Int,gy);fy=gy-j0;k0=floor(Int,gz);fz=gz-k0; w(i)=mod(i,Nc)+1
    @inbounds (φr[w(i0),w(j0),w(k0)]*(1-fx)*(1-fy)*(1-fz)+φr[w(i0+1),w(j0),w(k0)]*fx*(1-fy)*(1-fz)+
     φr[w(i0),w(j0+1),w(k0)]*(1-fx)*fy*(1-fz)+φr[w(i0+1),w(j0+1),w(k0)]*fx*fy*(1-fz)+
     φr[w(i0),w(j0),w(k0+1)]*(1-fx)*(1-fy)*fz+φr[w(i0+1),w(j0),w(k0+1)]*fx*(1-fy)*fz+
     φr[w(i0),w(j0+1),w(k0+1)]*(1-fx)*fy*fz+φr[w(i0+1),w(j0+1),w(k0+1)]*fx*fy*fz)
end

# shared gravity state across a cycle's level-0/level-≥1 hook calls
mutable struct GravState
    φr::Array{Float64,3}; Nc::Int; gasbar::Float64; Npart::Int
    posall::Matrix{Float64}; glb1::Array{Float64,3}    # global level-1 deposit (cell-ctr CIC, level-1 particles @2Nc)
    NL1::Int                                            # level-1 particle count (for the subgrid overdensity normalization)
    δr::Array{Float64,3}                                # root overdensity δ (Nc³) — fills the subgrid gravity buffer (Enzo's parent interpolation)
end
const GS = Ref{GravState}()

# ── :gravity slot — composite (root FFT + per-subgrid MG) ──
function composite_gravity!(h, level, dt)
    bep = PoissonKernels.backend(BE)
    if level == 0
        g = EnzoLib.problem_grid_index_on_level(h,0,0)
        gd = Tuple(Int.(EnzoLib.problem_grid_dims(h,g))); Nc = gd[1]-2NG
        gas = active_of(EnzoLib.read_density(h;grid=g), gd)
        posall = EnzoLib.read_particles(h); Npart = size(posall,1)
        gasbar = sum(gas)/length(gas)
        # Level-split deposit matching Enzo's GravitatingMassField (corr 1.0, verified):
        #   level-0 particles (mass<4) @root res; level-1 (mass>4) @2× res. The mass values
        #   (0.9, 7.2; ratio 8=2³) encode the grid level, NOT physical mass — all weight-1.
        massall = EnzoLib.read_particle_masses(h)
        L0=findall(<(4.0),massall); L1=findall(>(4.0),massall); NL1=length(L1)
        dmL1 = ciccc!(zeros(2Nc,2Nc,2Nc), posall, 2Nc, L1)         # level-1 @2Nc (subgrid source + root via restrict)
        if GMFSRC
            EnzoLib.session_prepare_density(h, 0)
            gmf = EnzoLib.problem_get_gravitating_mass(h, 0); buf=(size(gmf,1)-Nc)÷2; Rr=buf+1:buf+Nc
            δ = gmf[Rr,Rr,Rr]; δ = δ .- sum(δ)/length(δ)
        else
            dm = ciccc!(zeros(Nc,Nc,Nc), posall, Nc, L0) .+ rsum2(dmL1)   # root source (corr 1.0)
            dmbar = sum(dm)/length(dm)
            δ = gas.*(OMEGA_B/gasbar) .+ dm.*(OMEGA_CDM/dmbar); δ ./= (sum(δ)/length(δ)); δ .-= 1.0
        end
        # root Poisson on the GPU (device radix-2 FFT); φr to host for subgrid boundaries
        φd = PoissonKernels.device_zeros(bep, T, (Nc,Nc,Nc))
        PoissonKernels.fft_poisson_root_gpu!(φd, PoissonKernels.to_device(bep, Array{T,3}(δ), T); G=1.0, a=1.0, boxsize=1.0)
        φr = Float64.(PoissonKernels.to_host(φd))
        GS[] = GravState(φr, Nc, gasbar, Npart, posall, dmL1, NL1, Array{Float64,3}(δ))   # δr fills subgrid gravity buffer
        # root acceleration
        φf = pad_periodic(φr); dev=PoissonKernels.to_device(bep,φf,T)
        a1=PoissonKernels.device_zeros(bep,T,(Nc,Nc,Nc));a2=similar(a1);a3=similar(a1)
        PoissonKernels.comp_accel!(a1,a2,a3,dev; iflag=1, start=(NG,NG,NG), del=(1.0/Nc,1.0/Nc,1.0/Nc))
        _ga = T(gravfac())                    # Enzo comoving gravity coupling: accel ×= G_enzo/a²
        a1.*=_ga; a2.*=_ga; a3.*=_ga
        EnzoLib.problem_set_acceleration(h,0,place_active(Float64.(PoissonKernels.to_host(a1)),gd);grid=g)
        EnzoLib.problem_set_acceleration(h,1,place_active(Float64.(PoissonKernels.to_host(a2)),gd);grid=g)
        EnzoLib.problem_set_acceleration(h,2,place_active(Float64.(PoissonKernels.to_host(a3)),gd);grid=g)
        return nothing
    end
    isassigned(GS) || return nothing
    gs = GS[]; Nc = gs.Nc
    n = EnzoLib.session_num_grids_on_level(h, level)
    # subgrid→root accel projection (gravity analogue of update_from_finer): in
    # refined root cells the force should come from the SUBGRID (fine) solve, not the
    # full-res root solve (Enzo reduces the root force in refined regions). Read the
    # root accel, overwrite covered cells with the 2×2×2-averaged subgrid accel.
    rg = EnzoLib.problem_grid_index_on_level(h,0,0)
    rgd = Tuple(Int.(EnzoLib.problem_grid_dims(h,rg)))
    Ra = level==1 ? [active_of(EnzoLib.problem_get_acceleration(h,d,rg),rgd) for d in 0:2] : nothing
    GMFSRC && EnzoLib.session_prepare_density(h, level)   # option (b): deposit-only subgrid GravitatingMassFields (no Enzo solve)
    SUBENZO && return nothing                      # isolation: keep Enzo's subgrid accel (skip our solve)
    # ---- build each subgrid's source + parent-φ boundary (host), GROUP by active size ----
    groups = Dict{NTuple{3,Int}, Vector{Any}}()
    for i in 0:n-1
        g = EnzoLib.problem_grid_index_on_level(h, level, i)
        gd = Tuple(Int.(EnzoLib.problem_grid_dims(h,g))); na = gd.-2NG
        l,r = EnzoLib.problem_grid_edge(h,g); le=Float64.(l); dx=(Float64(r[1])-le[1])/na[1]
        gas = active_of(EnzoLib.read_density(h;grid=g), gd)
        # bw = solve-mesh buffer each side. GMFSRC: Enzo's gravity buffer (≈6) so the coarse
        # parent boundary sits far from the active region (Enzo's convention); else 1-ring.
        local d, bw, srcfull
        if GMFSRC
            gmf = EnzoLib.problem_get_gravitating_mass(h, g); d=size(gmf)
            bw=(d[1]-na[1])÷2; srcfull=gmf                       # FULL buffered source (incl buffer mass)
        elseif level == 1
            # OUR level-1 deposit (corr 1.0 vs Enzo GMF) on the BUFFERED mesh (bw=6, Enzo's
            # convention) → coarse parent boundary sits 6 cells out. Source = δ_DM·Ωcdm
            # (gas uniform at high z absorbs into the constant); gs.glb1 = level-1 CIC @2Nc.
            bw=6; Nf=2Nc; o=ntuple(dd->round(Int,le[dd]*Nf),3); d=(na[1]+2bw,na[2]+2bw,na[3]+2bw)
            meanc=gs.Npart/Nf^3                                  # GLOBAL mean count per fine cell (consistent w/ root dmbar)
            # ACTIVE cells: our level-1 deposit overdensity (corr/slope 1.0 vs Enzo GMF, verified).
            # BUFFER cells: trilinear(root δ) — Enzo fills the gravity buffer by interpolating the
            # PARENT density, not the fine deposit. Raw level-1 CIC in the buffer (sparse at the
            # subgrid edge) over-steepens φ → accel max 1.43× too big; root-δ buffer → 0.98× (verified).
            ccbf(aa,ii)=le[aa]-bw*dx+(ii-0.5)*dx
            inact(i,j,k)=(bw<i<=bw+na[1] && bw<j<=bw+na[2] && bw<k<=bw+na[3])
            srcfull=[inact(i,j,k) ?
                       OMEGA_CDM*(gs.glb1[mod(o[1]-bw+i-1,Nf)+1,mod(o[2]-bw+j-1,Nf)+1,mod(o[3]-bw+k-1,Nf)+1]/meanc-1.0) :
                       sample_cc(gs.δr,Nc,mod(ccbf(1,i),1.0),mod(ccbf(2,j),1.0),mod(ccbf(3,k),1.0))
                     for i in 1:d[1],j in 1:d[2],k in 1:d[3]]
        else
            cnt=zeros(Float64,na...)                              # level≥2: per-particle CIC (1-ring)
            @inbounds for p in 1:gs.Npart
                g0=(gs.posall[p,1]-le[1])/dx;g1=(gs.posall[p,2]-le[2])/dx;g2=(gs.posall[p,3]-le[3])/dx
                (g0< -1||g0>na[1]||g1< -1||g1>na[2]||g2< -1||g2>na[3]) && continue
                i0=floor(Int,g0);fx=g0-i0;j0=floor(Int,g1);fy=g1-j0;k0=floor(Int,g2);fz=g2-k0
                for (di,wi) in ((0,1-fx),(1,fx)),(dj,wj) in ((0,1-fy),(1,fy)),(dk,wk) in ((0,1-fz),(1,fz))
                    ii=i0+di+1;jj=j0+dj+1;kk=k0+dk+1
                    (1<=ii<=na[1]&&1<=jj<=na[2]&&1<=kk<=na[3]) && (cnt[ii,jj,kk]+=wi*wj*wk)
                end
            end
            ρdm = cnt ./ (gs.Npart*dx^3); bw=1; d=(na[1]+2,na[2]+2,na[3]+2)
            srcfull=zeros(Float64,d); srcfull[2:d[1]-1,2:d[2]-1,2:d[3]-1] .= gas.*(OMEGA_B/gs.gasbar) .+ ρdm.*OMEGA_CDM .- 1.0
        end
        sol=zeros(Float64,d); rhs=zeros(Float64,d)
        ccb(aa,ii)=le[aa]-bw*dx+(ii-0.5)*dx                       # buffered-mesh cell centers
        for k in 1:d[3],j in 1:d[2],ii in 1:d[1]
            (ii==1||ii==d[1]||j==1||j==d[2]||k==1||k==d[3]) &&
                (sol[ii,j,k]=sample_cc(gs.φr,Nc,mod(ccb(1,ii),1.0),mod(ccb(2,j),1.0),mod(ccb(3,k),1.0)))
        end
        fac=dx^2*(d[1]-1)*(d[2]-1)*(d[3]-1)*1.0
        @inbounds rhs[2:d[1]-1,2:d[2]-1,2:d[3]-1] .= fac.*(srcfull[2:d[1]-1,2:d[2]-1,2:d[3]-1])
        push!(get!(groups, d, Any[]), (g, gd, na, le, dx, sol, rhs, bw))
    end
    # ---- BATCHED MG solve per size-group, ON CPU ----
    # Enzo AMR is wildly heterogeneous (≈560 distinct sizes among 737 subgrids), so
    # exact-size GPU batching degenerates to singletons + per-group GPU launch/alloc
    # overhead. Tiny grids have NO launch overhead on the CPU KA backend, so the
    # subgrid MG runs on CPU (batched per size-group, threaded); the big root stays GPU.
    bsub = PoissonKernels.backend(:cpu); Tc = Float64
    for (d, items) in groups
        NB=length(items); na=items[1][3]; dx=items[1][5]; bw=items[1][8]
        solB=Array{Tc,4}(undef,d...,NB); rhsB=Array{Tc,4}(undef,d...,NB)
        for (b,it) in enumerate(items); @views solB[:,:,:,b].=it[6]; rhsB[:,:,:,b].=it[7]; end
        PoissonKernels.vcycle_batched!(solB,rhsB; cycle=:W, ncyc=20, dirichlet=true)
        a1=Array{Tc,4}(undef,na...,NB);a2=similar(a1);a3=similar(a1)
        PoissonKernels.comp_accel_batched!(a1,a2,a3,solB; iflag=1, start=(bw,bw,bw), del=(dx,dx,dx))
        _ga = gravfac(); (a1.*=_ga; a2.*=_ga; a3.*=_ga)   # Enzo comoving gravity coupling: accel ×= G_enzo/a²
        a1h=a1;a2h=a2;a3h=a3
        for (b,(g,gd,na2,le,dx2,_,_)) in enumerate(items)
            sa=(a1h[:,:,:,b],a2h[:,:,:,b],a3h[:,:,:,b])
            EnzoLib.problem_set_acceleration(h,0,place_active(sa[1],gd);grid=g)
            EnzoLib.problem_set_acceleration(h,1,place_active(sa[2],gd);grid=g)
            EnzoLib.problem_set_acceleration(h,2,place_active(sa[3],gd);grid=g)
            if Ra !== nothing && all(iseven, na2)
                o=ntuple(dd->round(Int,le[dd]*Nc),3); nr=ntuple(dd->na2[dd]÷2,3)
                @inbounds for c in 1:3, K in 1:nr[3], J in 1:nr[2], I in 1:nr[1]
                    s=0.0; for dk in 0:1,dj in 0:1,di in 0:1; s+=sa[c][2I-1+di,2J-1+dj,2K-1+dk]; end
                    Ra[c][o[1]+I,o[2]+J,o[3]+K]=s/8
                end
            end
        end
    end
    # write the refined-cell-corrected root acceleration back
    if Ra !== nothing
        for d in 0:2; EnzoLib.problem_set_acceleration(h,d,place_active(Ra[d+1],rgd);grid=rg); end
    end
    return nothing
end

# per-level hydro wall time (root vs subgrid) for the perf dissection
const _hroot = Ref(0.0); const _hsub = Ref(0.0); const _hio = Ref(0.0)

# ── :hydro slot — Enzo PPM, per grid ──
function ppm_hydro!(h, level, dt)
    _t0 = time_ns()
    bep = PPMKernels.backend(BE)
    n = EnzoLib.session_num_grids_on_level(h, level)
    order = isodd(_step[]) ? (3,2,1) : (1,2,3); _step[] += 1
    for i in 0:n-1
        g = EnzoLib.problem_grid_index_on_level(h, level, i)
        gd = Tuple(Int.(EnzoLib.problem_grid_dims(h, g))); na = gd .- 2NG
        # PROPER cell width = a·(comoving dx) — Enzo's PPM uses CellWidthTemp=a·CellWidth
        # (Grid_SolveHydroEquations.C:406). Without this the mass flux dt/dx is ~N× too
        # small and the gas never compresses (velocity, gravity-sourced, looks fine).
        l,r = EnzoLib.problem_grid_edge(h,g); dxc = (Float64(r[1])-Float64(l[1]))/na[1]
        dxp = afac[] * dxc
        _ti = time_ns()
        f(fi) = PPMKernels.to_device(bep, EnzoLib.problem_get_field(h, fi, g), T)
        d,e,ge = f(iD),f(iTE),f(iGE); vx,vy,vz = f(iV1),f(iV2),f(iV3)
        gx = PPMKernels.to_device(bep, EnzoLib.problem_get_acceleration(h,0,g), T)
        gy = PPMKernels.to_device(bep, EnzoLib.problem_get_acceleration(h,1,g), T)
        gz = PPMKernels.to_device(bep, EnzoLib.problem_get_acceleration(h,2,g), T)
        _hio[] += (time_ns()-_ti)/1e9
        if GFIX                          # Strang-split: ½ g-kick → pure hydro → ½ g-kick (g enters the mass flux)
            vx.+=gx.*(T(dt)/2); vy.+=gy.*(T(dt)/2); vz.+=gz.*(T(dt)/2)
            PPMKernels.ppm_step_3d!(d,e,ge,vx,vy,vz,gx,gy,gz,gd,NG; dt=dt,gamma=GAMMA,order=order,gravity=0,idual=1,dx=dxp)
            vx.+=gx.*(T(dt)/2); vy.+=gy.*(T(dt)/2); vz.+=gz.*(T(dt)/2)
        else
            PPMKernels.ppm_step_3d!(d,e,ge,vx,vy,vz,gx,gy,gz,gd,NG; dt=dt,gamma=GAMMA,order=order,gravity=1,idual=1,dx=dxp)
        end
        _ti = time_ns()
        wr(fi,a)=EnzoLib.problem_set_field(h,fi,Float64.(PPMKernels.to_host(a));grid=g)
        wr(iD,d);wr(iTE,e);wr(iGE,ge);wr(iV1,vx);wr(iV2,vy);wr(iV3,vz)
        _hio[] += (time_ns()-_ti)/1e9
    end
    el=(time_ns()-_t0)/1e9; level==0 ? (_hroot[]+=el) : (_hsub[]+=el)
    return nothing
end

# MODE=enzong (default): EnzoNG :julia composite gravity + PPM (on BACKEND).
# MODE=enzo: the all-:enzo reference (native PPM + FFT gravity). Same evolve_level!,
# same AMR — isolates the swapped kernels for the parity/perf comparison.
const MODE = Symbol(get(ENV, "MODE", "enzong"))

function main()
    maxcyc = length(ARGS)>=1 ? parse(Int,ARGS[1]) : 200
    EnzoLib.grid_available() || error("grid dylib not built")
    # per-mode working dir with symlinked ICs (so the two runs never collide on output)
    rundir = joinpath(SBDIR, "run_$(MODE)_$(BE)")
    mkpath(rundir)
    for f in readdir(SBDIR)
        (startswith(f,"SB_") && (occursin("Grid",f)||occursin("Particle",f))) || continue
        lnk=joinpath(rundir,f); islink(lnk)||isfile(lnk)||symlink(joinpath(SBDIR,f),lnk)
    end
    pf = joinpath(rundir,"run.enzo")
    write(pf, replace(read(joinpath(SBDIR,"SB_amr.enzo"),String),
                      r"GreensFunctionMaxNumber.*"=>"GreensFunctionMaxNumber   = 30\nNumberOfGhostZones        = 4"))
    # MODE=enzo: pure Enzo (native PPM + FFT gravity). MODE=enzong: EnzoNG GPU PPM
    # hydro with Enzo's OWN gravity (option C) — identical gravity+AMR removes gravity
    # as a confound and isolates the hydro port + the GPU win (the launch-bound tiny-
    # subgrid GPU gravity is a perf loss anyway; the real GPU win is PPM on the root).
    # MODE=enzong_g: the experimental EnzoNG composite gravity too (parity ~0.87 under AMR).
    prb = EnzoLib.SlotProbe()
    eng = MODE === :enzo ?
        EnzoLib.EngineConfig(; hydro=:enzo, gravity=:enzo, comoving_expansion=:enzo, probe=prb) :
      MODE === :enzong_g ?
        EnzoLib.EngineConfig(; hydro=:julia, gravity=:julia, comoving_expansion=:enzo, reflux=false, probe=prb,
                             hooks=Dict{Symbol,Function}(:hydro=>ppm_hydro!, :gravity=>composite_gravity!)) :
        EnzoLib.EngineConfig(; hydro=:julia, gravity=:enzo, comoving_expansion=:enzo, probe=prb,
                             hooks=Dict{Symbol,Function}(:hydro=>ppm_hydro!))
    @printf("SB_amr MODE=%s backend=%s — %d cycles (dir %s)\n", MODE, BE, maxcyc, basename(rundir)); flush(stdout)
    cd(rundir) do
        h = EnzoLib.session_init(pf); h==C_NULL && error("session_init failed")
        try
            EnzoLib.session_rebuild(h,0)
            m0 = EnzoLib.session_global_field_integral(h,0)
            @printf("%-4s %-9s %-16s %-9s %-9s %-9s %-9s\n","cyc","t","grids/level","ρmax","|v|max","TEmean","Δmass/M"); flush(stdout)
            cyc=0; t0=time(); pcyc=Float64[]
            while cyc<maxcyc
                _c0=time_ns()
                afac[] = first(EnzoLib.session_cosmology(h))   # a for the proper cell width in ppm_hydro!
                EnzoLib.evolve_level!(h,0,0.0; engine=eng, regrid=(get(ENV,"REGRID","1")=="1"))
                push!(pcyc,(time_ns()-_c0)/1e9)
                EnzoLib.session_rebuild(h,0)
                ρ=EnzoLib.problem_get_field(h,iD,0)
                v1=EnzoLib.problem_get_field(h,iV1,0); v2=EnzoLib.problem_get_field(h,iV2,0); v3=EnzoLib.problem_get_field(h,iV3,0)
                te=EnzoLib.problem_get_field(h,iTE,0)
                vmax=sqrt(maximum(@. v1^2+v2^2+v3^2)); m=EnzoLib.session_global_field_integral(h,0)
                ngl=Int[EnzoLib.session_num_grids_on_level(h,l) for l in 0:3]
                @printf("%-4d %-9.5f %-16s %-9.5f %-9.2e %-9.2e %-9.1e a=%.4f\n", cyc, EnzoLib.session_time(h),
                        string(ngl), maximum(ρ), vmax, sum(te)/length(te), abs(m-m0)/m0, afac[]); flush(stdout)
                any(isnan,ρ) && (println("  NaN — abort"); break)
                cyc+=1
            end
            @printf("%s: ran %d cycles in %.0f s\n", cyc>=maxcyc ? "DONE" : "ABORTED", cyc, time()-t0)
            # ── perf dissection (median per-cycle; per-slot from SlotProbe; hydro root/sub split) ──
            ps = EnzoLib.probe_summary(prb)
            med(v)=isempty(v) ? 0.0 : sort(v)[(length(v)+1)÷2]
            gms = haskey(ps,:gravity) ? ps[:gravity].total_ns/1e6/max(cyc,1) : 0.0
            hms = haskey(ps,:hydro)   ? ps[:hydro].total_ns/1e6/max(cyc,1)   : 0.0
            @printf("PERF: median %.2f s/cyc | per-cyc avg: gravity %.0f ms, hydro %.0f ms\n", med(pcyc), gms, hms)
            @printf("PERF: hydro split (avg/cyc): root %.0f ms, subgrids %.0f ms, bridge-IO+transfers %.0f ms\n",
                    1000*_hroot[]/max(cyc,1), 1000*_hsub[]/max(cyc,1), 1000*_hio[]/max(cyc,1))
        finally
            EnzoLib.free_problem(h)
        end
    end
end
get(ENV, "SB_NO_MAIN", "0") == "0" && main()   # gate: include this file (SB_NO_MAIN=1) to reuse composite_gravity!/ppm_hydro! without running
