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

@info "DM pancake test" backend=BE
l1, gr = run_pancake()
@printf("Zel'dovich pancake: growth D(af)/D(ai)=%.3f  |  L1(position) rel err = %.4f\n", gr, l1)
println(l1 < 0.05 ? "PASS: DM integrator reproduces Zel'dovich to <5%" :
                    "FAIL: L1=$(round(l1,digits=4)) — DM dynamics off")
