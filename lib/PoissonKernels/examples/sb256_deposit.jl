# Step 1 of the SWIFT SantaBarbara-256 gravity test: read the SWIFT DM particles
# and CIC-deposit them into a 256³ overdensity δ, written as a raw Float64 binary.
#
# This runs in ITS OWN process using only HDF5.jl. It must NOT load EnzoLib —
# EnzoLib's grid dylib links Homebrew's libhdf5, and HDF5.jl loads HDF5_jll's
# libhdf5; two libhdf5 in one process abort (SIGABRT). The companion solver step
# (sb256_gravity.jl) reads the binary this writes and never loads HDF5.jl.
#
# Run:  <julia> --project=lib/PoissonKernels/test lib/PoissonKernels/examples/sb256_deposit.jl

using HDF5
using Printf

const IC = "/Users/tabel/Projects/enzo-dev/run/CosmologySimulation/SantaBarbara-256/SantaBarbara_256.hdf5"
const OUT = "/tmp/sb256_delta.bin"
const N = 256

function cic_deposit(C::AbstractMatrix{Float64}, mass::AbstractVector{Float64}, box::Float64, N::Int)
    @assert size(C, 1) == 3
    rho = zeros(Float64, N, N, N)
    s = N / box
    @inbounds for p in 1:size(C, 2)
        gx = mod(C[1, p] * s, Float64(N)); gy = mod(C[2, p] * s, Float64(N)); gz = mod(C[3, p] * s, Float64(N))
        m = mass[p]
        i = floor(Int, gx); fx = gx - i
        j = floor(Int, gy); fy = gy - j
        k = floor(Int, gz); fz = gz - k
        i0 = mod(i, N) + 1; i1 = mod(i + 1, N) + 1
        j0 = mod(j, N) + 1; j1 = mod(j + 1, N) + 1
        k0 = mod(k, N) + 1; k1 = mod(k + 1, N) + 1
        rho[i0, j0, k0] += m * (1 - fx) * (1 - fy) * (1 - fz); rho[i1, j0, k0] += m * fx * (1 - fy) * (1 - fz)
        rho[i0, j1, k0] += m * (1 - fx) * fy * (1 - fz);       rho[i1, j1, k0] += m * fx * fy * (1 - fz)
        rho[i0, j0, k1] += m * (1 - fx) * (1 - fy) * fz;       rho[i1, j0, k1] += m * fx * (1 - fy) * fz
        rho[i0, j1, k1] += m * (1 - fx) * fy * fz;             rho[i1, j1, k1] += m * fx * fy * fz
    end
    return rho
end

C, mass, box, Np = h5open(IC, "r") do f
    coords = read(f["PartType1/Coordinates"])
    coords = size(coords, 1) == 3 ? coords : permutedims(coords)
    m = read(f["PartType1/Masses"])
    bs = read_attribute(f["Header"], "BoxSize")
    b = bs isa AbstractArray ? Float64(bs[1]) : Float64(bs)
    (Float64.(coords), Float64.(m), b, size(coords, 2))
end
@printf("SWIFT SB-256: %d DM particles, box=%.3f, mass[1]=%.4e\n", Np, box, mass[1])

t = @elapsed (rho = cic_deposit(C, mass, box, N))
rhobar = sum(rho) / length(rho)
delta = rho ./ rhobar .- 1.0
@printf("CIC %d³ (%.1fs): ρ̄=%.4e  δ∈[%.3f, %.3f]  Σδ=%.2e\n",
        N, t, rhobar, minimum(delta), maximum(delta), sum(delta))

open(OUT, "w") do io
    write(io, Int64(N))
    write(io, delta)                 # N³ Float64, column-major
end
@printf("wrote %s (%d bytes)\n", OUT, filesize(OUT))
