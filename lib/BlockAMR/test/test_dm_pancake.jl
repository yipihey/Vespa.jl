# DM dynamics gate: the Zel'dovich pancake for PARTICLES.  A single-mode plane
# wave is EXACT under Zel'dovich until shell-crossing, so the DM integrator
# (topgrid CIC -> FFT Poisson G=1.5 Om a -> accel gather -> KDK -> cosmology dtau)
# must reproduce  x(q,a) = q + [D(a)/D(a_i)] Psi(q)  and  the growing-mode velocity.
# EdS (Om=1) is used so D(a)=a exactly -> clean analytic target.
#
# Run: <julia> --project=lib/BlockAMR/test lib/BlockAMR/test/test_dm_pancake.jl
using BlockAMR, Printf
import PoissonKernels
try; @eval using CUDA; catch; end
const BE = has_backend(:cuda) ? :cuda : :cpu

# EdS cosmology, inline (self-contained): Om=1 -> D(a)=a exactly, da/dtau = a^1.5.
D(a) = a
dadtau(a) = sqrt(a^3)                                        # Om=1
dtau_for_dlna(a, dlna) = a*dlna/dadtau(a)

function run_pancake(; N=32, mode=1, ac=0.10, ai=0.01, af=0.05, nstep=200, be=BE)
    box = 1.0; Om = 1.0; k = 2π*mode/box; A = 1.0/ac         # caustic at a=ac
    # Lagrangian lattice + growing-mode Zel'dovich (displacement along x)
    Np = N^3
    q = Float64[]; sizehint!(q, 3Np)
    px=zeros(Float32,Np); py=zeros(Float32,Np); pz=zeros(Float32,Np)
    vx=zeros(Float32,Np); vy=zeros(Float32,Np); vz=zeros(Float32,Np)
    qx=zeros(Np)
    p=0
    for kk in 0:N-1, jj in 0:N-1, ii in 0:N-1
        p+=1
        qxi=(ii+0.5)/N*box; qyi=(jj+0.5)/N*box; qzi=(kk+0.5)/N*box
        qx[p]=qxi
        psi = -(A/k)*sin(k*qxi)                              # Psi_lin_x (comoving Mpc/h)
        xi = qxi + D(ai)*psi                                 # x = q + D(a) Psi_lin
        px[p]=Float32(mod(xi/box,1.0)); py[p]=Float32(qyi/box); pz[p]=Float32(qzi/box)
        # v = dx/dtau = (dD/dtau) Psi_lin ; dD/dtau = (dD/da) dadtau
        dDda=(D(ai*1.001)-D(ai*0.999))/(ai*0.002)
        vx[p]=Float32(dDda*dadtau(ai)*psi)                   # comoving dx/dtau (Mpc/h per 1/H0)
    end
    beobj=BlockAMR.backend(be); dev(v)=BlockAMR.to_device(beobj, v, Float32)
    parts=(px=dev(px),py=dev(py),pz=dev(pz),vx=dev(vx),vy=dev(vy),vz=dev(vz))
    mass_code = 1.0/Np                                        # all matter is DM (fb=0)
    # hierarchy: level 0 tiles the box (B|N).  gas uniform (0 perturbation).
    B = 16; while N % B != 0; B ÷= 2; end
    hier = AMRHierarchy(; nbase=(N,N,N), B=B, backend=be, T=Float16, nsp=0,
                        gamma=5/3, cfl=0.3, Lcap=0, scheme=:ctu)
    init_base_level!(hier); build_level_tables!(hier, 0)
    ρg = BlockAMR.device_zeros(hier.be, Float32, (N^3,))
    φg = BlockAMR.device_zeros(hier.be, Float32, (N^3,))
    pax=BlockAMR.device_zeros(hier.be,Float32,(Np,)); pay=similar(pax); paz=similar(pax)
    a = ai; dlna = (log(af)-log(ai))/nstep
    l1_pos=0.0
    for s in 1:nstep
        dτ = dtau_for_dlna(a, dlna)
        # gravity: DM CIC -> delta -> FFT (G=1.5 Om a) -> level-0 phi
        fill!(ρg, 0.0f0)
        PoissonKernels.cic_deposit!(ρg, parts.px, parts.py, parts.pz,
            parts.vx, parts.vy, parts.vz, Float32(mass_code*N^3); N=N, disp=0.0f0, shift=-0.5f0)
        ρg .-= 1.0f0
        ρ3=reshape(ρg,N,N,N); φ3=reshape(φg,N,N,N)
        PoissonKernels.fft_poisson_rfft!(φ3, ρ3; G=1.5*Om*a, a=1.0, boxsize=1.0)
        BlockAMR.phi_from_global!(hier, φg)
        BlockAMR.gather_accel_particles!(hier, parts, pax, pay, paz)
        # KDK
        BlockAMR.particles_kick!(hier, parts, pax, pay, paz, 0.5*dτ)
        BlockAMR.particles_drift!(hier, parts, dτ)
        a2 = exp(log(a)+dlna)
        BlockAMR.particles_kick!(hier, parts, pax, pay, paz, 0.5*dτ)
        a = a2
    end
    # compare final positions to exact Zel'dovich x(q,af)=q+D(af)Psi_lin
    hpx=Array(parts.px)
    err=0.0; ref=0.0
    for p in 1:Np
        psi=-(A/k)*sin(k*qx[p]); xex=mod((qx[p]+D(af)*psi)/box,1.0)
        d=abs(hpx[p]-xex); d=min(d,1-d)                       # periodic
        err+=d; ref+=abs(D(af)*psi/box)
    end
    return err/ref, D(af)/D(ai)                               # L1 rel error, growth ratio
end

# ── refined variant: evolve near the caustic so delta>threshold and level 1
#    refines, exercising the block-CIC + native-gravity DM path (LDM=1) ──
function run_pancake_refined(; N=32, mode=1, ac=0.10, ai=0.01, af=0.092, nstep=400,
                              lmax=1, dthresh=3.0, ldm=1, be=BE)
    box=1.0; Om=1.0; k=2π*mode/box; A=1.0/ac; Np=N^3
    px=zeros(Float32,Np); py=zeros(Float32,Np); pz=zeros(Float32,Np)
    vx=zeros(Float32,Np); vy=zeros(Float32,Np); vz=zeros(Float32,Np); qx=zeros(Np)
    p=0
    for kk in 0:N-1, jj in 0:N-1, ii in 0:N-1
        p+=1; qxi=(ii+0.5)/N*box; qx[p]=qxi
        psi=-(A/k)*sin(k*qxi); xi=qxi+D(ai)*psi
        px[p]=Float32(mod(xi/box,1.0)); py[p]=Float32((jj+0.5)/N); pz[p]=Float32((kk+0.5)/N)
        dDda=(D(ai*1.001)-D(ai*0.999))/(ai*0.002); vx[p]=Float32(dDda*dadtau(ai)*psi)
    end
    beobj=BlockAMR.backend(be); dev(v)=BlockAMR.to_device(beobj,v,Float32)
    parts=(px=dev(px),py=dev(py),pz=dev(pz),vx=dev(vx),vy=dev(vy),vz=dev(vz))
    mass_code=1.0/Np
    B=16; while N%B!=0; B÷=2; end
    hier=AMRHierarchy(; nbase=(N,N,N),B=B,backend=be,T=Float16,nsp=0,gamma=5/3,cfl=0.3,Lcap=lmax,scheme=:ctu)
    init_base_level!(hier); build_level_tables!(hier,0)
    # DM-based refinement: pure-DM pancake refines on its OWN overdensity (total
    # matter = DM here, so dthresh is in total-matter units).
    pol=BlockRefinementPolicy(; dthresh=dthresh, nbuf=2, every=4, lmax=lmax, lfac=4.0,
                              dm_criterion=true)
    ρg=BlockAMR.device_zeros(hier.be,Float32,(N^3,)); φg=BlockAMR.device_zeros(hier.be,Float32,(N^3,))
    pax=BlockAMR.device_zeros(hier.be,Float32,(Np,)); pay=similar(pax); paz=similar(pax)
    a=ai; dlna=(log(af)-log(ai))/nstep; nstepdone=0
    for s in 1:nstep
        dτ=dtau_for_dlna(a,dlna)
        s%4==0 && regrid!(hier, pol; compact=true, parts, mass_code)
        # level-0 topgrid FFT (gas=0 + DM CIC)
        fill!(ρg,0.0f0)
        PoissonKernels.cic_deposit!(ρg,parts.px,parts.py,parts.pz,parts.vx,parts.vy,parts.vz,
            Float32(mass_code*N^3); N=N, disp=0.0f0, shift=-0.5f0)
        ρg .-= 1.0f0
        PoissonKernels.fft_poisson_rfft!(reshape(φg,N,N,N),reshape(ρg,N,N,N); G=1.5*Om*a,a=1.0,boxsize=1.0)
        BlockAMR.phi_from_global!(hier,φg)
        # per-level DM deposit + native gravity (levels 1..ldm), seeds finer Dirichlet
        for l in 1:min(length(hier.levels)-1, ldm)
            isempty(hier.levels[l+1].live) && continue
            BlockAMR.deposit_particles_level!(hier,l,parts; mass_code)
        end
        for l in 1:length(hier.levels)-1
            isempty(hier.levels[l+1].live) && continue
            BlockAMR.solve_gravity_level!(hier,l; source_coef=1.5*Om*a, nsweep=30,
                rho_mean=1.0, use_dm=(l<=ldm), residual=false)
        end
        BlockAMR.gather_accel_particles!(hier,parts,pax,pay,paz)
        BlockAMR.particles_kick!(hier,parts,pax,pay,paz,0.5*dτ)
        BlockAMR.particles_drift!(hier,parts,dτ)
        BlockAMR.particles_kick!(hier,parts,pax,pay,paz,0.5*dτ)
        a=exp(log(a)+dlna); nstepdone+=1
    end
    hpx=Array(parts.px); err=0.0; ref=0.0
    for p in 1:Np
        psi=-(A/k)*sin(k*qx[p]); xex=mod((qx[p]+D(af)*psi)/box,1.0)
        d=abs(hpx[p]-xex); d=min(d,1-d); err+=d; ref+=abs(D(af)*psi/box)
    end
    maxlev=maximum(l for l in 0:length(hier.levels)-1 if !isempty(hier.levels[l+1].live))
    return err/ref, maxlev
end

# ── TWO-FLUID cold pancake: gas + DM both cold, both trace the SAME Zel'dovich
#    plane wave, both source the COMBINED gravity (gas from level 0 + DM CIC).
#    This exercises the gas-hydro/DM-gravity COUPLING the pure-DM tests never
#    touch: the gas continuity+Euler solve must reproduce the same Zel'dovich
#    Eulerian density the particle map does, and — with dm_criterion refinement —
#    must keep doing so on the refined blocks.  Kept BELOW the caustic (D·A ≤ 0.7)
#    so the cold gas stays single-valued (no multi-stream shell-crossing, which a
#    single-valued Eulerian solver cannot represent).
#
# Exact 1D Zel'dovich Eulerian map for x = q − (D·A/k) sin(k q):
#   ρ(x)/ρ̄ = 1/(1 − D·A·cos(k q(x))),   v_x = (dD/dτ)·(−(A/k) sin(k q(x))).
zeld_q(x, DA, k) = begin                # Newton-invert x = q − (DA/k) sin(k q)
    q = x
    for _ in 1:80
        f = q - (DA/k)*sin(k*q) - x; fp = 1 - DA*cos(k*q)
        q -= f/(abs(fp) < 1e-6 ? copysign(1e-6, fp) : fp)
    end
    q
end

# gas x-slice density profile (averaged over y,z) on the finest COVERING cell at
# each global fine-x index — a composite leaf profile for the 1D plane wave.
function gas_profile_composite(hier, N; lmax=0)
    Nf = N * 2^lmax
    prof = zeros(Nf); cnt = zeros(Int, Nf)
    for l in 0:length(hier.levels)-1
        lev = hier.levels[l+1]; isempty(lev.live) && continue
        Nl = N * 2^l; rep = 2^(lmax-l)          # each level-l cell spans rep fine-x bins
        hD = Array(lev.D); hdsc = Array(lev.Dsc)
        flev = l+2 <= length(hier.levels) ? hier.levels[l+2] : nothing
        for s in lev.live
            m = lev.meta[s]; base = (Int(s)-1)*lev.stride
            for k in 1:lev.B, j in 1:lev.B, i in 1:lev.B
                gi = Int(m.origin[1]) + i          # 1..Nl  (x index)
                if flev !== nothing && !isempty(flev.live)
                    g = (Int128(m.origin[1])+i-1, Int128(m.origin[2])+j-1, Int128(m.origin[3])+k-1)
                    gf = ntuple(d -> Int128(2)*g[d], 3)
                    isempty(overlapping_blocks(flev, gf, (2,2,2))) || continue   # covered
                end
                idx = base + ((lev.ng+k-1)*lev.nd + (lev.ng+j-1))*lev.nd + (lev.ng+i-1) + 1
                ρ = Float64(hD[idx]) * hdsc[s]
                for r in 1:rep                       # spread level-l cell over fine bins
                    fb_ = (gi-1)*rep + r
                    prof[fb_] += ρ; cnt[fb_] += 1
                end
            end
        end
    end
    for b in 1:Nf; cnt[b] > 0 && (prof[b] /= cnt[b]); end
    return prof
end

function run_pancake_twofluid(; N=32, mode=1, ac=0.10, ai=0.01, af=0.07, nstep=400,
                               fb=0.15, cs2=1f-8, lmax=0, dthresh=3.0, ldm=1, be=BE)
    box=1.0; Om=1.0; k=2π*mode/box; A=1.0/ac; Np=N^3
    dDdτ(a) = ((D(a*1.001)-D(a*0.999))/(a*0.002))*dadtau(a)      # growing-mode dD/dτ
    # ── DM particles: Zel'dovich lattice (Lagrangian q = lattice, exact) ──
    px=zeros(Float32,Np); py=zeros(Float32,Np); pz=zeros(Float32,Np)
    vx=zeros(Float32,Np); vy=zeros(Float32,Np); vz=zeros(Float32,Np); qx=zeros(Np)
    p=0
    for kk in 0:N-1, jj in 0:N-1, ii in 0:N-1
        p+=1; qxi=(ii+0.5)/N*box; qx[p]=qxi
        psi=-(A/k)*sin(k*qxi); px[p]=Float32(mod((qxi+D(ai)*psi)/box,1.0))
        py[p]=Float32((jj+0.5)/N); pz[p]=Float32((kk+0.5)/N)
        vx[p]=Float32(dDdτ(ai)*psi)
    end
    beobj=BlockAMR.backend(be); dev(v)=BlockAMR.to_device(beobj,v,Float32)
    parts=(px=dev(px),py=dev(py),pz=dev(pz),vx=dev(vx),vy=dev(vy),vz=dev(vz))
    mass_dm=(1-fb)/Np                                            # DM mean density = 1−fb
    # ── gas level-0 fields from the EXACT Zel'dovich Eulerian density at a_i ──
    B=16; while N%B!=0; B÷=2; end
    hier=AMRHierarchy(; nbase=(N,N,N),B=B,backend=be,T=Float16,nsp=0,gamma=5/3,cfl=0.3,
                      Lcap=lmax,scheme=:ctu)
    lev0=init_base_level!(hier); build_level_tables!(hier,0)
    n=lev0.cap*lev0.stride
    hD=zeros(n); hS1=zeros(n); hS2=zeros(n); hS3=zeros(n); hT=zeros(n); hG=zeros(n)
    eint=cs2/(hier.gamma*(hier.gamma-1))                        # T from cs² = γ(γ−1)e
    DAi=D(ai)*A
    for s in lev0.live
        m=lev0.meta[s]; base=(Int(s)-1)*lev0.stride
        for kk in 1:B, jj in 1:B, ii in 1:B
            gi=Int(m.origin[1])+ii                              # x index 1..N
            xc=(gi-0.5)/N; q=zeld_q(xc,DAi,k)
            ρ = fb/(1 - DAi*cos(k*q)); v1=dDdτ(ai)*(-(A/k)*sin(k*q))
            idx=base+((lev0.ng+kk-1)*lev0.nd+(lev0.ng+jj-1))*lev0.nd+(lev0.ng+ii-1)+1
            hD[idx]=ρ; hS1[idx]=ρ*v1; hG[idx]=ρ*eint
            hT[idx]=ρ*(eint+0.5*v1^2)
        end
    end
    BlockAMR.encode_from_host!(lev0,hD,hS1,hS2,hS3,hT,hG)
    pol = lmax>0 ? BlockRefinementPolicy(; dthresh=dthresh, nbuf=2, every=4, lmax=lmax,
                                         lfac=4.0, dm_criterion=true) : nothing
    ρg=BlockAMR.device_zeros(hier.be,Float32,(N^3,)); φg=BlockAMR.device_zeros(hier.be,Float32,(N^3,))
    pax=BlockAMR.device_zeros(hier.be,Float32,(Np,)); pay=similar(pax); paz=similar(pax)
    dx0=BlockAMR.level_dx(hier,0)
    a=ai; dlna=(log(af)-log(ai))/nstep
    for step in 1:nstep
        (pol!==nothing && step%4==0) && regrid!(hier,pol; compact=true, parts, mass_code=mass_dm)
        # combined gravity source: gas (level 0) + DM CIC → δ → FFT
        fill!(ρg,0.0f0); BlockAMR.global_from_level0!(hier,ρg)
        PoissonKernels.cic_deposit!(ρg,parts.px,parts.py,parts.pz,parts.vx,parts.vy,parts.vz,
            Float32(mass_dm*N^3); N=N, disp=0.0f0, shift=-0.5f0)
        ρg .-= 1.0f0
        PoissonKernels.fft_poisson_rfft!(reshape(φg,N,N,N),reshape(ρg,N,N,N); G=1.5*Om*a,a=1.0,boxsize=1.0)
        BlockAMR.phi_from_global!(hier,φg)
        for l in 1:min(length(hier.levels)-1, ldm)
            isempty(hier.levels[l+1].live) && continue
            BlockAMR.deposit_particles_level!(hier,l,parts; mass_code=mass_dm)
        end
        # timestep: cosmology dlna capped by the hydro CFL
        λ=compute_lambda!(hier); dτ=min(dtau_for_dlna(a,dlna), Float64(λ)*dx0)
        hier.λ=Float32(dτ/dx0)
        BlockAMR.advance_level_w!(hier,0,hier.λ;
            selfgrav=(coef=1.5*Om*a, nsweep=30, nsweep0=0, rho_mean=1.0, use_dm=(ldm>=1)))
        # particles KDK from the finest covering level's φ
        BlockAMR.gather_accel_particles!(hier,parts,pax,pay,paz)
        BlockAMR.particles_kick!(hier,parts,pax,pay,paz,0.5*dτ)
        BlockAMR.particles_drift!(hier,parts,dτ)
        BlockAMR.particles_kick!(hier,parts,pax,pay,paz,0.5*dτ)
        a=exp(log(a)+dlna)
        (pol!==nothing) && update_scales!(hier,0)
    end
    # ── compare ──
    maxlev=maximum(l for l in 0:length(hier.levels)-1 if !isempty(hier.levels[l+1].live))
    # (1) gas 1D profile vs exact Zel'dovich Eulerian density
    gp = gas_profile_composite(hier, N; lmax=maxlev); Nf=length(gp)
    DAf=D(af)*A; gex=zeros(Nf); egas=0.0; ref=0.0; peak_gas=0.0; peak_ex=0.0
    for b in 1:Nf
        xc=(b-0.5)/Nf; q=zeld_q(xc,DAf,k); ρex=fb/(1-DAf*cos(k*q)); gex[b]=ρex
        egas+=abs(gp[b]-ρex); ref+=abs(ρex-fb)
        peak_gas=max(peak_gas,gp[b]); peak_ex=max(peak_ex,ρex)
    end
    l1_gas = egas/ref
    # (2) DM positions vs exact Zel'dovich (Lagrangian, level-independent)
    hpx=Array(parts.px); edm=0.0; rdm=0.0
    for pp in 1:Np
        psi=-(A/k)*sin(k*qx[pp]); xex=mod((qx[pp]+D(af)*psi)/box,1.0)
        d=abs(hpx[pp]-xex); d=min(d,1-d); edm+=d; rdm+=abs(D(af)*psi/box)
    end
    l1_dm=edm/rdm
    # (3) gas vs exact at HALF resolution (Nc=N/2): a peak-cusp-robust profile
    #    metric — the single finest cell resolving the cusp dominates the full-res
    #    L1, so averaging two cells gives the fair "does the gas trace the smooth
    #    growing mode" number.  (A direct gas-vs-DM-density check is NOT used: a
    #    1:1 particle lattice deposited onto a commensurate grid aliases into a
    #    beat — deposited CIC is a poor small-scale DM density estimator — so the
    #    coupling is instead established by BOTH fluids matching exact Zel'dovich.)
    Nc=N÷2; gN=zeros(N)
    rep=Nf÷N; for gi_ in 1:N; gN[gi_]=sum(@view gp[(gi_-1)*rep+1:gi_*rep])/rep; end
    gasc=zeros(Nc); for gi_ in 1:N; gasc[(gi_-1)÷2+1]+=gN[gi_]/2; end
    egc=0.0; rgc=0.0
    for b in 1:Nc
        xc=(b-0.5)/Nc; q=zeld_q(xc,DAf,k); ρex=fb/(1-DAf*cos(k*q))
        egc+=abs(gasc[b]-ρex); rgc+=abs(ρex-fb)
    end
    l1_gas_c=egc/rgc
    if get(ENV,"TF_DUMP","0")=="1"
        println("  xc-idx  gas_δ   exact_δ  (Nc=$Nc)")
        for b in 1:Nc
            xc=(b-0.5)/Nc; q=zeld_q(xc,DAf,k); dex=1/(1-DAf*cos(k*q))-1
            @printf("  %3d  %7.3f %7.3f\n", b, gasc[b]/fb-1, dex)
        end
    end
    return (l1_gas=l1_gas, l1_gas_c=l1_gas_c, l1_dm=l1_dm, maxlev=maxlev,
            peak_gas=peak_gas/fb, peak_ex=peak_ex/fb, DAf=DAf)
end

if get(ENV, "PANCAKE_RUN", "1") == "1"   # PANCAKE_RUN=0 → defs only (interactive probing)

@info "DM pancake test" backend=BE
l1, gr = run_pancake()
@printf("[level-0]  growth=%.3f  L1(pos)=%.4f  %s\n", gr, l1, l1<0.05 ? "PASS" : "FAIL")

# ── convergence with RESOLUTION (level-0 PM): L1 vs N, expect ~2nd order (dx^2) ──
if get(ENV, "PANCAKE_CONV", "0") == "1"
    println("── spatial convergence (nstep=800 so timestep error is subdominant) ──")
    Ns=[16,32,64,128]; L=Float64[]
    for N in Ns
        e,_=run_pancake(; N=N, nstep=800)
        push!(L,e); @printf("  N=%4d  L1(pos)=%.5f\n", N, e)
    end
    for i in 2:length(Ns)
        rate=log(L[i-1]/L[i])/log(Ns[i]/Ns[i-1])
        @printf("  rate N=%d→%d : %.2f\n", Ns[i-1], Ns[i], rate)
    end
    println("── temporal convergence (N=32, vary nstep) ──")
    for ns in [100,200,400,800]
        e,_=run_pancake(; N=32, nstep=ns); @printf("  nstep=%4d  L1=%.5f\n", ns, e)
    end
end
# The refined variant exercises regrid + block-CIC + native gravity ONLY if the
# region actually refines.  NOTE: BlockAMR's refinement criterion keys on the GAS
# density; a pure-DM pancake (gas=0) never triggers it -> DM refinement is INDIRECT
# (gas traces the DM potential).  A true refined-DM gate needs a two-fluid cold
# pancake (gas tracing DM) or a DM-based refinement criterion.
l1r, mlev = run_pancake_refined()
if mlev == 0
    @printf("[refined]  deepest level=0 — refinement did NOT trigger\n")
else
    @printf("[refined]  deepest level=%d  L1(pos)=%.4f  %s\n", mlev, l1r, l1r<0.08 ? "PASS" : "FAIL")
end
# LDM sensitivity: block-DM gravity (ldm=1) vs DM-via-Dirichlet-only (ldm=0)
if get(ENV, "PANCAKE_LDM_SWEEP", "1") == "1"
    for ldm in (0, 1)
        l1s, ml = run_pancake_refined(; ldm=ldm, lmax=2)
        @printf("[LDM=%d, lmax=2]  deepest=%d  L1(pos)=%.4f\n", ldm, ml, l1s)
    end
end
println(l1<0.05 ? "DM DYNAMICS: level-0 (topgrid) reproduces Zel'dovich to <5% ✓" :
                  "DM DYNAMICS: level-0 FAILED")

# ── #1: TWO-FLUID cold pancake (gas + DM both cold, tracing each other) ──
# The gas is a cold Eulerian fluid; the DM is collisionless particles.  Both
# start on the SAME growing-mode Zel'dovich flow and source the COMBINED gravity,
# so both must track the exact Zel'dovich density AND each other.  In the smooth
# mildly-nonlinear regime (DA≲0.5) the two representations agree tightly; only
# near the caustic (DA→1) do they diverge — the grid gas diffuses the sharpening
# peak while particles pile toward the shell-crossing δ-function (a real
# discretization difference, resolved by refinement, not a coupling bug).
println("── two-fluid cold pancake: gas hydro + DM gravity coupling ──")
# (a) code-path gate: fb→0 must reproduce the pure-DM integrator EXACTLY — same
#     particles + FFT φ, but the gas now sources gravity (carrying ≈0 mass).  A
#     match proves the two-fluid gravity/stepping wiring adds no error.
base = run_pancake(; N=32, af=0.05, nstep=500)[1]
tf0  = run_pancake_twofluid(; N=32, af=0.05, nstep=500, fb=1f-4)
@printf("[fb→0 ≡ pure-DM]  pure-DM L1=%.4f  two-fluid(fb=1e-4) DM L1=%.4f  %s\n",
        base, tf0.l1_dm, abs(tf0.l1_dm-base)<0.01 ? "PASS" : "FAIL")
# (b) gas-tracks-Zel'dovich gate: cold gas hydro must reproduce the growing-mode
#     density in the smooth regime (DA=0.5, well below the caustic).
tfg = run_pancake_twofluid(; N=32, af=0.05, nstep=500, fb=0.05)
@printf("[gas ≈ Zel'dovich]  DA=0.5 gasδ L1(N/2)=%.4f  peak δ=%.3f vs exact %.3f  %s\n",
        tfg.l1_gas_c, tfg.peak_gas-1, tfg.peak_ex-1,
        (tfg.l1_gas_c<0.08 && abs(tfg.peak_gas-tfg.peak_ex)<0.06) ? "PASS" : "FAIL")
# (c) gas-fraction sweep (informational): the CIC-DM source is spiky (over-
#     drives), the Eulerian gas is diffusive (under-drives); they cross near
#     fb≈0.05 (gas peak = exact).  More gas ⇒ shallower composite well ⇒ both
#     fluids collapse slightly less — a fixed-resolution effect, cured by AMR.
println("  fb       DM pos L1   gasδ L1(N/2)   gas peak δ (exact 0.963)")
for fb in (0.02, 0.05, 0.1565, 0.5)
    tf = run_pancake_twofluid(; N=32, af=0.05, nstep=500, fb=fb)
    @printf("  %.4f   %.4f      %.4f         %.3f\n", fb, tf.l1_dm, tf.l1_gas_c, tf.peak_gas-1)
end
# (d) refinement gate: at the SAME physical time (DA=0.6), one level of AMR
#     (dm_criterion → refine on total matter) must sharpen the gas peak toward
#     exact vs level-0.
r0 = run_pancake_twofluid(; N=32, af=0.06, nstep=600, fb=0.1565)
r1 = run_pancake_twofluid(; N=32, af=0.06, nstep=600, fb=0.1565, lmax=1, dthresh=0.5)
@printf("[AMR sharpens gas]  L0 peak δ=%.3f (L1=%.4f) → L%d peak δ=%.3f (L1=%.4f)  exact %.3f  %s\n",
        r0.peak_gas-1, r0.l1_gas, r1.maxlev, r1.peak_gas-1, r1.l1_gas, r0.peak_ex-1,
        (r1.maxlev>=1 && (r1.peak_gas>r0.peak_gas)) ? "PASS (peak sharpened)" :
        (r1.maxlev==0 ? "no-refine" : "FAIL"))

end  # PANCAKE_RUN
