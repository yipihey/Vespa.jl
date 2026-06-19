# Profile the Vespa :gravity slot sub-steps on the live SB hierarchy, to find
# where the ~145 ms/call goes (FFT? CIC deposit of 2M particles? bridge I/O?).
# Run: BACKEND=cpu|metal <jl> --project=lib/PoissonKernels/test lib/PoissonKernels/examples/prof_gravity.jl
using PoissonKernels, EnzoLib, Printf
try; @eval using Metal; catch; end
const PK = PoissonKernels
const SB = "/Users/tabel/Projects/enzo-dev/run/CosmologySimulation/SantaBarbaraCluster/SantaBarbaraCluster.enzo"
const NG = 4; const OMEGA_B=0.1; const OMEGA_CDM=0.9
const BE = Symbol(get(ENV,"BACKEND","cpu")); const T = Float32

active_of(flat, gd, N) = Array(reshape(Float64.(flat), gd[1],gd[2],gd[3])[NG+1:NG+N,NG+1:NG+N,NG+1:NG+N])
function pad_periodic(φ); N=size(φ,1); M=N+2NG; full=Array{Float64,3}(undef,M,M,M)
    @inbounds for k in 1:M,j in 1:M,i in 1:M; full[i,j,k]=φ[mod(i-NG-1,N)+1,mod(j-NG-1,N)+1,mod(k-NG-1,N)+1]; end; full; end
place_active(act,gd)=(full=zeros(Float64,gd[1],gd[2],gd[3]);N=size(act,1);full[NG+1:NG+N,NG+1:NG+N,NG+1:NG+N].=act;vec(full))
cic!(rho,pos,N)=begin
    @inbounds for p in 1:size(pos,1)
        gx=mod(pos[p,1],1.0)*N;gy=mod(pos[p,2],1.0)*N;gz=mod(pos[p,3],1.0)*N
        i=floor(Int,gx);fx=gx-i;j=floor(Int,gy);fy=gy-j;k=floor(Int,gz);fz=gz-k
        i0=mod(i,N)+1;i1=mod(i+1,N)+1;j0=mod(j,N)+1;j1=mod(j+1,N)+1;k0=mod(k,N)+1;k1=mod(k+1,N)+1
        rho[i0,j0,k0]+=(1-fx)*(1-fy)*(1-fz);rho[i1,j0,k0]+=fx*(1-fy)*(1-fz)
        rho[i0,j1,k0]+=(1-fx)*fy*(1-fz);rho[i1,j1,k0]+=fx*fy*(1-fz)
        rho[i0,j0,k1]+=(1-fx)*(1-fy)*fz;rho[i1,j0,k1]+=fx*(1-fy)*fz
        rho[i0,j1,k1]+=(1-fx)*fy*fz;rho[i1,j1,k1]+=fx*fy*fz
    end; rho; end

ms(f) = (f(); t=time_ns(); f(); (time_ns()-t)/1e6)  # one warmup + one timed

cd(dirname(SB)) do
    h = EnzoLib.session_init(SB); EnzoLib.session_set_boundary(h,0); EnzoLib.session_rebuild(h,0)
    bep = PK.backend(BE)
    g = EnzoLib.problem_grid_index_on_level(h,0,0)
    gd = Tuple(Int.(EnzoLib.problem_grid_dims(h,g))); N = gd[1]-2NG
    @printf("backend=%s  N=%d  grid dims=%s\n", BE, N, string(gd))
    t_rd  = ms(() -> EnzoLib.read_density(h; grid=g))
    t_rp  = ms(() -> EnzoLib.read_particles(h))
    pos = EnzoLib.read_particles(h)
    @printf("  read_density   : %7.1f ms\n", t_rd)
    @printf("  read_particles : %7.1f ms  (%d particles)\n", t_rp, size(pos,1))
    t_cic = ms(() -> cic!(zeros(N,N,N), pos, N))
    @printf("  CIC deposit    : %7.1f ms  (Julia, OLD path)\n", t_cic)
    t_dep = ms(() -> EnzoLib.deposit_particle_density(h; grid=g))
    @printf("  deposit (C++)  : %7.1f ms  (NEW path: replaces read_particles+CIC)\n", t_dep)
    gas = active_of(EnzoLib.read_density(h;grid=g),gd,N)
    dm = cic!(zeros(N,N,N),pos,N); dm .*= OMEGA_CDM/(sum(dm)/length(dm)); gas .*= OMEGA_B/(sum(gas)/length(gas))
    δ = gas.+dm; δ ./= (sum(δ)/length(δ)); δ .-= 1.0
    φh = Array{T,3}(undef,N,N,N); δT = Array{T,3}(δ)
    t_fft = ms(() -> PK.fft_poisson_root!(φh, δT; G=1.0,a=1.0,boxsize=1.0))      # host path (cached plan/Gk)
    @printf("  fft_poisson    : %7.1f ms  (host, cached plan)\n", t_fft)
    φf = PK.to_device(bep, pad_periodic(Float64.(φh)), T)
    a1=PK.device_zeros(bep,T,(N,N,N));a2=similar(a1);a3=similar(a1)
    t_acc = ms(() -> PK.comp_accel!(a1,a2,a3,φf;iflag=1,start=(NG,NG,NG),del=(1.0/N,1.0/N,1.0/N)))
    @printf("  comp_accel     : %7.1f ms\n", t_acc)
    aa = Float64.(PK.to_host(a1))
    t_wr = ms(() -> EnzoLib.problem_set_acceleration(h,0,place_active(aa,gd);grid=g))
    @printf("  set_accel(×3≈) : %7.1f ms  (one dim)\n", t_wr)
    EnzoLib.free_problem(h)
end
