# Validate fft_poisson_root_gpu! (device radix-2 FFT) vs the CPU FFTW oracle
# (fft_poisson_root!) and the analytic spectral solution, on CPU and Metal.
# Run: <julia> --project=lib/PoissonKernels/test lib/PoissonKernels/examples/gpu_fft_check.jl
using PoissonKernels, Printf
try; @eval using Metal; catch; end
const PK = PoissonKernels
rel(a,b)=sqrt(sum(abs2,a.-b)/sum(abs2,b))

# analytic: ρ = sin(2πx) ⇒ ∇²φ=ρ ⇒ φ = -ρ/(2π)²  (single mode, periodic)
function analytic(N)
    ρ=Array{Float64,3}(undef,N,N,N); φ=similar(ρ)
    @inbounds for k in 1:N,j in 1:N,i in 1:N
        x=(i-1)/N; s=sinpi(2x); ρ[i,j,k]=s; φ[i,j,k]=-s/(2π)^2; end
    ρ.-=sum(ρ)/length(ρ); ρ,φ
end

function check(be_name, T)
    PK.has_backend(be_name) || (return println("  [$be_name] not available"))
    be=PK.backend(be_name); N=64
    # 1) vs FFTW oracle on random zero-mean source
    ρh=rand(Float64,N,N,N); ρh.-=sum(ρh)/length(ρh)
    φ_fftw=Array{Float64,3}(undef,N,N,N); PK.fft_poisson_root!(φ_fftw, ρh; G=1.0,a=1.0,boxsize=1.0)
    ρd=PK.to_device(be,T.(ρh),T); φd=PK.device_zeros(be,T,(N,N,N))
    PK.fft_poisson_root_gpu!(φd, ρd; G=1.0,a=1.0,boxsize=1.0)
    φg=Float64.(PK.to_host(φd))
    @printf("  [%-5s %s] GPU-FFT vs FFTW: relL2=%.2e\n", be_name, T, rel(φg,φ_fftw))
    # 2) vs analytic single mode
    ρa,φa=analytic(N)
    ρd2=PK.to_device(be,T.(ρa),T); φd2=PK.device_zeros(be,T,(N,N,N))
    PK.fft_poisson_root_gpu!(φd2, ρd2; G=1.0,a=1.0,boxsize=1.0)
    φg2=Float64.(PK.to_host(φd2)); φg2.-=sum(φg2)/length(φg2)
    @printf("  [%-5s %s] GPU-FFT vs analytic: relL2=%.2e\n", be_name, T, rel(φg2,φa))
end

println("GPU root Poisson (radix-2 FFT) validation:")
check(:cpu, Float64)
check(:cpu, Float32)
check(:metal, Float32)
