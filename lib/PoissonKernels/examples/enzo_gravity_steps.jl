# Step-isolation test of the gravity chain against Enzo's OWN fields (no fitting):
#   (1) Poisson: feed Enzo's GravitatingMassField (source) to our FFT solve, compare
#       to Enzo's PotentialField  → measures coef (should be 4π/a) exactly.
#   (2) Accel:   feed Enzo's PotentialField to our comp_accel, compare to Enzo's
#       AccelerationField           → measures the gradient convention (del=a·dx) exactly.
# Identical Enzo inputs in, so any mismatch is a pure convention/units fact, derivable.
# Run: ENZOMODULES_GRID_LIB=<f32> <jl> --project=lib/PoissonKernels/test lib/PoissonKernels/examples/enzo_gravity_steps.jl
using PoissonKernels, EnzoLib, Printf
try; @eval using Metal; catch; end
const PK = PoissonKernels
const SB = "/Users/tabel/Projects/enzo-dev/run/CosmologySimulation/SantaBarbaraCluster/SB_amr.enzo"
slope(o,e) = sum(vec(o).*vec(e))/sum(vec(o).^2)               # least-squares e ≈ slope·o
corr(o,e)=(om=o.-sum(o)/length(o); em=e.-sum(e)/length(e); sum(om.*em)/(sqrt(sum(om.^2)*sum(em.^2))+1e-300))

cd(dirname(SB)) do
    h = EnzoLib.session_init(SB); EnzoLib.session_set_boundary(h,0); EnzoLib.session_rebuild(h,0)
    EnzoLib.session_gravity(h,0)                                # Enzo: deposit→FFT→accel on the root
    a,z = EnzoLib.session_cosmology(h)
    gmf = EnzoLib.problem_get_gravitating_mass(h,0)             # Enzo's Poisson SOURCE
    pot = EnzoLib.problem_get_potential(h,0)                    # Enzo's solved POTENTIAL
    gd=Tuple(Int.(EnzoLib.problem_grid_dims(h,0))); NGb=3; Na=gd[1]-2NGb  # baryon active
    dims = size(gmf); buf=(dims[1]-Na)÷2                                   # gravity buffer each side
    @printf("GMF dims=%s  active N=%d  buffer=%d  a=%.5f  z=%.4f  mean(gmf)=%.5f\n",
            string(dims), Na, buf, a, z, sum(gmf)/length(gmf))
    # extract the active N³ block (periodic domain) from the buffered GMF + potential
    R = (buf+1):(buf+Na)
    gmfA = gmf[R,R,R]; potA = pot[R,R,R]; N=Na
    @printf("active: (N=%d pow2? %s)  Enzo φ range [%.3e,%.3e]\n", N, string((N&(N-1))==0), minimum(potA), maximum(potA))
    be = PK.backend(:cpu); T=Float64
    # (1) Poisson: our solve on Enzo's GMF (active), coef=4π/a (Enzo GravitationalConstant=4π)
    src = PK.to_device(be, gmfA, T); φ = PK.device_zeros(be, T, (N,N,N))
    PK.fft_poisson_root_gpu!(φ, src; G=4π, a=a, boxsize=1.0)
    φo = Float64.(PK.to_host(φ)); potz = potA .- sum(potA)/length(potA); φoz = φo .- sum(φo)/length(φo)
    @printf("(1) POISSON  our φ(4π/a · GMF) vs Enzo φ:  corr=%.5f  slope(enzo/ours)=%.5f\n",
            corr(φoz,potz), slope(φoz,potz))
    # (2) Accel: our comp_accel on Enzo's potential (active), del = a·(1/N) (Enzo CellSize=a·GMFCellSize)
    pd = PK.to_device(be, potA, T)
    d1=PK.device_zeros(be,T,(N-2,N-2,N-2));d2=similar(d1);d3=similar(d1)
    PK.comp_accel!(d1,d2,d3, pd; iflag=1, start=(1,1,1), del=(a/N,a/N,a/N))
    e1=EnzoLib.problem_get_acceleration(h,0,0); NG=NGb
    E1=reshape(Float64.(e1),gd...)[NG+2:NG+N-1,NG+2:NG+N-1,NG+2:NG+N-1]
    A1=Float64.(PK.to_host(d1))
    @printf("(2) ACCEL    our g(-∇φ_enzo, del=a/N) vs Enzo accel_x:  corr=%.5f  slope(enzo/ours)=%.5f\n",
            corr(A1,E1), slope(A1,E1))
    # (3) DEPOSIT: our composite source δ (gas·Ωb + CIC-DM·Ωcdm, zero-mean) vs Enzo's GMF (active)
    OMEGA_B=0.1; OMEGA_CDM=0.9
    gas = reshape(Float64.(EnzoLib.read_density(h;grid=0)),gd...)[NGb+1:NGb+N,NGb+1:NGb+N,NGb+1:NGb+N]
    pos = EnzoLib.read_particles(h); dm = zeros(N,N,N)
    @inbounds for p in 1:size(pos,1)
        g0=pos[p,1]*N-0.5;g1=pos[p,2]*N-0.5;g2=pos[p,3]*N-0.5
        i0=floor(Int,g0);fx=g0-i0;j0=floor(Int,g1);fy=g1-j0;k0=floor(Int,g2);fz=g2-k0
        for (di,wi) in ((0,1-fx),(1,fx)),(dj,wj) in ((0,1-fy),(1,fy)),(dk,wk) in ((0,1-fz),(1,fz))
            dm[mod(i0+di,N)+1,mod(j0+dj,N)+1,mod(k0+dk,N)+1]+=wi*wj*wk
        end
    end
    δ = gas.*(OMEGA_B/(sum(gas)/length(gas))) .+ dm.*(OMEGA_CDM/(sum(dm)/length(dm)))
    δ ./= (sum(δ)/length(δ)); δ .-= 1.0
    gz = gmfA .- sum(gmfA)/length(gmfA); zc(x)=x.-sum(x)/length(x)
    # variants to localize the shape mismatch
    dmz = zc(dm); gasz = zc(gas)
    # node-offset CIC (g0 = x·N, no -0.5)
    dmN = zeros(N,N,N)
    @inbounds for p in 1:size(pos,1)
        g0=pos[p,1]*N;g1=pos[p,2]*N;g2=pos[p,3]*N
        i0=floor(Int,g0);fx=g0-i0;j0=floor(Int,g1);fy=g1-j0;k0=floor(Int,g2);fz=g2-k0
        for (di,wi) in ((0,1-fx),(1,fx)),(dj,wj) in ((0,1-fy),(1,fy)),(dk,wk) in ((0,1-fz),(1,fz))
            dmN[mod(i0+di,N)+1,mod(j0+dj,N)+1,mod(k0+dk,N)+1]+=wi*wj*wk
        end
    end
    @printf("(3) DEPOSIT  gas+DM δ vs GMF:  corr=%.4f slope=%.4f\n", corr(δ,gz), slope(δ,gz))
    @printf("    pure DM (cell-ctr) vs GMF:  corr=%.4f\n", corr(dmz,gz))
    @printf("    pure DM (node)     vs GMF:  corr=%.4f\n", corr(zc(dmN),gz))
    @printf("    pure gas           vs GMF:  corr=%.4f  (gas std=%.3e)\n", corr(gasz,gz), sqrt(sum(gasz.^2)/length(gasz)))
    @printf("    npart=%d  gd=%s\n", size(pos,1), string(gd))
    # shift scan: is it a grid-alignment offset? cross-correlate dm vs GMF at integer shifts
    best=(-1.0,0,0,0)
    for sx in -2:2, sy in -2:2, sz in -2:2
        sh = circshift(dmz,(sx,sy,sz)); c=corr(sh,gz)
        c>best[1] && (best=(c,sx,sy,sz))
    end
    @printf("    DM shift-scan best corr=%.4f at shift=(%d,%d,%d)\n", best[1],best[2],best[3],best[4])
    EnzoLib.free_problem(h)
end
