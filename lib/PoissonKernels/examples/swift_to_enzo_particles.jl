# Convert the SWIFT SantaBarbara-256 DM particle IC (gadget-style HDF5) into Enzo
# CosmologySimulation particle IC files (SB256_ParticlePositions / _Velocities)
# referenced by run/CosmologySimulation/SantaBarbara-256/SantaBarbaraCluster.enzo.
#
# FORMAT (matched to the original 128³ SB_ParticlePositions): an HDF5 dataset of
# shape {3, Np} Float32 (component-major: all x, then all y, then all z), with the
# Enzo attributes Component_Rank=3, Component_Size=3·Np, Dimensions=[Np,3], Rank=1.
# Positions are normalized to [0,1) code units (SWIFT coord / BoxSize).
#
# STATUS: POSITIONS are converted faithfully and the output format is validated
# against the original Enzo IC (h5ls/h5dump). VELOCITIES are passed through with the
# SWIFT→Enzo unit factor left as `VEL_FACTOR` — the gadget→Enzo cosmology velocity
# convention (peculiar v, the a^½ factor, and Enzo's VelocityUnits) MUST be verified
# before a dynamical Enzo run. The Vespa Metal-gravity performance test does NOT use
# this converter (it reads the SWIFT file directly; velocities are irrelevant there).
#
# Run in its OWN process (HDF5.jl only — never alongside EnzoLib, see sb256_deposit.jl):
#   <julia> --project=lib/PoissonKernels/test lib/PoissonKernels/examples/swift_to_enzo_particles.jl

using HDF5
using Printf

const IC  = "/Users/tabel/Projects/enzo-dev/run/CosmologySimulation/SantaBarbara-256/SantaBarbara_256.hdf5"
const DST = "/Users/tabel/Projects/enzo-dev/run/CosmologySimulation/SantaBarbara-256"
const VEL_FACTOR = 1.0f0     # TODO: verify SWIFT→Enzo peculiar-velocity unit conversion

# Write a {3, Np} component-major Float32 dataset with Enzo's IC attributes.
# `M` is the Julia (Np, 3) array (column c = component c); HDF5.jl stores it as {3, Np}.
function write_enzo_component(path, name, M::Matrix{Float32})
    Np = size(M, 1)
    h5open(path, "w") do f
        dset = create_dataset(f, name, datatype(Float32), dataspace(Np, 3))
        write(dset, M)
        attributes(dset)["Component_Rank"] = Int32(3)
        attributes(dset)["Component_Size"] = Int32(Np * 3)
        attributes(dset)["Dimensions"]     = Int32[Np, 3]
        attributes(dset)["Rank"]           = Int32(1)
    end
end

coords, vels, box, Np = h5open(IC, "r") do f
    c = read(f["PartType1/Coordinates"]); c = size(c, 1) == 3 ? c : permutedims(c)   # (3, Np)
    v = read(f["PartType1/Velocities"]);  v = size(v, 1) == 3 ? v : permutedims(v)   # (3, Np)
    bs = read_attribute(f["Header"], "BoxSize"); b = bs isa AbstractArray ? Float64(bs[1]) : Float64(bs)
    (c, v, b, size(c, 2))
end
@printf("SWIFT SB-256: %d particles, box=%.3f\n", Np, box)

# positions → [0,1), Julia (Np,3) so HDF5 writes {3, Np} component-major
pos = Matrix{Float32}(undef, Np, 3)
@inbounds for d in 1:3, p in 1:Np
    pos[p, d] = Float32(mod(coords[d, p] / box, 1.0))
end
write_enzo_component(joinpath(DST, "SB256_ParticlePositions"), "SB256_ParticlePositions", pos)
@printf("wrote SB256_ParticlePositions {3, %d}, range [%.4f, %.4f]\n", Np, minimum(pos), maximum(pos))

vel = Matrix{Float32}(undef, Np, 3)
@inbounds for d in 1:3, p in 1:Np
    vel[p, d] = Float32(vels[d, p]) * VEL_FACTOR
end
write_enzo_component(joinpath(DST, "SB256_ParticleVelocities"), "SB256_ParticleVelocities", vel)
@printf("wrote SB256_ParticleVelocities {3, %d}  (VEL_FACTOR=%.3g — verify before a dynamical run)\n", Np, VEL_FACTOR)
