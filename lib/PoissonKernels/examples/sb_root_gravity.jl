# Phase C1: does our composite root gravity match Enzo's on the live SB hierarchy?
#
# Enzo's gravity (session_gravity) deposits gas+DM → GravitatingMassField, FFT-solves
# the periodic root potential, and differences it to the AccelerationField. We rebuild
# the SAME chain with our Metal kernels: gas density + CIC-deposited DM → δ →
# fft_poisson_root! → comp_accel! → our acceleration. We compare the two acceleration
# fields by structure (Pearson correlation) and best-fit slope — high correlation
# validates the deposit→FFT→accel chain; the slope pins the units constant (G/a, cell
# size) that we then fix. (Not bit-tight: FFTW ≠ Enzo FFT and our CIC ≈ Enzo's deposit.)
#
# Run:  <julia> --project=lib/PoissonKernels/test lib/PoissonKernels/examples/sb_root_gravity.jl

using PoissonKernels, EnzoLib, Printf
try; @eval using Metal; catch; end
const PK = PoissonKernels
const SB = "/Users/tabel/Projects/enzo-dev/run/CosmologySimulation/SantaBarbaraCluster/SantaBarbaraCluster.enzo"
const OMEGA_B = 0.1; const OMEGA_CDM = 0.9

function cic!(rho, pos, N)
    @inbounds for p in 1:size(pos,1)
        gx=mod(pos[p,1],1.0)*N; gy=mod(pos[p,2],1.0)*N; gz=mod(pos[p,3],1.0)*N
        i=floor(Int,gx); fx=gx-i; j=floor(Int,gy); fy=gy-j; k=floor(Int,gz); fz=gz-k
        i0=mod(i,N)+1;i1=mod(i+1,N)+1;j0=mod(j,N)+1;j1=mod(j+1,N)+1;k0=mod(k,N)+1;k1=mod(k+1,N)+1
        rho[i0,j0,k0]+=(1-fx)*(1-fy)*(1-fz); rho[i1,j0,k0]+=fx*(1-fy)*(1-fz)
        rho[i0,j1,k0]+=(1-fx)*fy*(1-fz); rho[i1,j1,k0]+=fx*fy*(1-fz)
        rho[i0,j0,k1]+=(1-fx)*(1-fy)*fz; rho[i1,j0,k1]+=fx*(1-fy)*fz
        rho[i0,j1,k1]+=(1-fx)*fy*fz; rho[i1,j1,k1]+=fx*fy*fz
    end
    rho
end

# Pearson r and best-fit slope a (b≈0) of y ≈ a·x
function corr_slope(x, y)
    xv=vec(x); yv=vec(y); mx=sum(xv)/length(xv); my=sum(yv)/length(yv)
    sxy=0.0; sxx=0.0; syy=0.0
    @inbounds for i in eachindex(xv,yv)
        dx=xv[i]-mx; dy=yv[i]-my; sxy+=dx*dy; sxx+=dx*dx; syy+=dy*dy
    end
    (sxy/sqrt(sxx*syy), sxy/sxx)
end

cd(dirname(SB)) do
    h = EnzoLib.session_init(SB)
    try
        EnzoLib.session_set_boundary(h, 0)
        EnzoLib.session_gravity(h, 0)                      # Enzo: deposit + FFT root + accel
        gd = EnzoLib.problem_grid_dims(h, 0); ng = (gd[1]-128)÷2; N = 128
        a0 = ng+1; b0 = ng+N
        act(flat) = Array(reshape(Float64.(flat), gd[1],gd[2],gd[3])[a0:b0,a0:b0,a0:b0])
        ax_e = act(EnzoLib.problem_get_acceleration(h,0,0))
        ay_e = act(EnzoLib.problem_get_acceleration(h,1,0))
        az_e = act(EnzoLib.problem_get_acceleration(h,2,0))
        gas  = act(EnzoLib.read_density(h; grid=0))
        pos  = EnzoLib.read_particles(h)
        @printf("SB root %d³: Enzo |a|∈[%.2e,%.2e], gas mean=%.3f, %d particles\n",
                N, minimum(abs,ax_e), maximum(abs,ax_e), sum(gas)/length(gas), size(pos,1))

        # our gravitating overdensity (gas mean→Ω_b, DM mean→Ω_CDM), zero-mean
        dm = cic!(zeros(N,N,N), pos, N); dm .*= OMEGA_CDM/(sum(dm)/length(dm))
        g  = gas .* (OMEGA_B/(sum(gas)/length(gas)))
        δ  = (g .+ dm); δ ./= (sum(δ)/length(δ)); δ .-= 1.0

        BE = Symbol(get(ENV,"BACKEND","metal")); T = BE===:cpu ? Float64 : Float32
        be = PK.backend(BE)
        φ  = PK.device_zeros(be, T, (N,N,N))
        # FULL GPU root Poisson solve (device radix-2 FFT) — no CPU FFTW round-trip
        PK.fft_poisson_root_gpu!(φ, PK.to_device(be, T.(δ), T); G=1.0, a=1.0, boxsize=1.0)
        # our acceleration on the interior (start=1 reads φ[2..N-1]); iflag=1 symmetric
        d1=PK.device_zeros(be,T,(N-2,N-2,N-2)); d2=similar(d1); d3=similar(d1)
        PK.comp_accel!(d1,d2,d3, φ; iflag=1, start=(1,1,1), del=(T(1.0/N),T(1.0/N),T(1.0/N)))
        ax_o=PK.to_host(d1); ay_o=PK.to_host(d2); az_o=PK.to_host(d3)
        # compare on the same interior of the Enzo active region
        E1=ax_e[2:N-1,2:N-1,2:N-1]; E2=ay_e[2:N-1,2:N-1,2:N-1]; E3=az_e[2:N-1,2:N-1,2:N-1]
        for (nm,o,e) in (("x",ax_o,E1),("y",ay_o,E2),("z",az_o,E3))
            r,s = corr_slope(o, e)
            @printf("  accel %s: corr=%.4f  slope(enzo/ours)=%.3e\n", nm, r, s)
        end
    finally
        EnzoLib.free_problem(h)
    end
end
