"""
    PoissonKernelsCUDAExt

Package extension that lights up the CUDA (NVIDIA GPU) backend for
`PoissonKernels`. Loaded automatically when `CUDA` is present in the
environment. Registers the `:cuda` backend and specialises the device-array
helpers onto `CuArray`. CUDA supports both Float32 and Float64.
"""
module PoissonKernelsCUDAExt

using PoissonKernels
using CUDA
using PrecompileTools

function __init__()
    if CUDA.functional()
        PoissonKernels.register_backend!(:cuda, CUDABackend())
    end
end

PoissonKernels.device_zeros(::CUDABackend, ::Type{T}, dims::Dims) where {T} =
    CUDA.zeros(T, dims)

# Precompile the CUDA-specialised gravity kernels into this extension's pkgimage. A cold
# first call otherwise JITs ~19 s per process start (the rfft plan+spectral-operator path
# alone is ~15 s; two-grid ~3 s; deterministic Int32 deposit ~1 s). Types MATCH the CICASS
# production path so the runtime call hits the cache: Float32 fields/positions, Float16
# velocities (VEL16 half-drift), Int32 deposit accumulator. Guarded on a functional GPU so
# the ext still precompiles on GPU-less CI (there these kernels JIT at runtime, as before).
@setup_workload begin
    if CUDA.functional()
        @compile_workload begin
            N = 16; Np = 64
            ρ  = CUDA.zeros(Float32, N, N, N); φ  = CUDA.zeros(Float32, N, N, N)
            PoissonKernels.fft_poisson_rfft!(φ, ρ; G=1.0, a=1.0, boxsize=1.0)
            ρ2 = CUDA.zeros(Float32, N, N, N); φ2 = CUDA.zeros(Float32, N, N, N)
            PoissonKernels.fft_poisson_2grid!(φ2, ρ2; G=1.0, a=1.0, boxsize=1.0, nsweeps=2, prolong=:cubic)
            ρi = CUDA.zeros(Int32, N^3)
            px = CUDA.zeros(Float32, Np); py = CUDA.zeros(Float32, Np); pz = CUDA.zeros(Float32, Np)
            vx = CUDA.zeros(Float16, Np); vy = CUDA.zeros(Float16, Np); vz = CUDA.zeros(Float16, Np)
            PoissonKernels.cic_deposit_det!(ρi, px, py, pz, vx, vy, vz, 0.83f0; N=N, disp=0.0, shift=-0.5, qbits=23)
            CUDA.synchronize()
        end
    end
end

end # module
