#!/usr/bin/env julia
# Write Enzo-format HDF5 initial-condition files for a CICASS realization.
#
# Runs as a SUBPROCESS (loads HDF5.jl but NOT libenzo — the two libhdf5 copies
# abort together if co-resident).  Reads raw little-endian binary dumps written by
# the parent (already in Enzo code units), emits the four CosmologySimulation IC
# datasets with the EXACT attributes Enzo's CosmologySimulationInitializeGrid reads
# (verified against the historical host files):
#   GridDensity       : dataspace (N,N,N)    Rank=3 Component_Rank=1 Dimensions=[N,N,N]
#   GridVelocities    : dataspace (3,N,N,N)  Rank=3 Component_Rank=3 Dimensions=[N,N,N,3]
#   ParticlePositions : dataspace (3,Np)     Rank=1 Component_Rank=3 Dimensions=[Np,3]
#   ParticleVelocities: dataspace (3,Np)     Rank=1 Component_Rank=3 Dimensions=[Np,3]
# (Enzo's particle reader requires Rank==1 with Dimensions[0]=Np; a 2-D rank made it
#  read 3 particles and overrun.)
#
#   usage: julia write_enzo_cicass_ic.jl <dir> <N> <Np>
# expects in <dir>:  density.f32 (N³, i fastest)   gridvel.f32 (N³·3, [cell,dim])
#                    partpos.f32 (Np·3, [part,dim]) partvel.f32 (Np·3, [part,dim])

using HDF5

dir = ARGS[1]; N = parse(Int, ARGS[2]); Np = parse(Int, ARGS[3])
rd(name, n) = Vector{Float32}(reinterpret(Float32, read(joinpath(dir, name)))[1:n])

# Write a dataset of on-disk shape `dspace` (C/HDF5 order) from a contiguous Float32
# buffer already laid out in that C order, plus Enzo's IC attributes.
function emit(path, dsname, cbuf::Vector{Float32}, dspace::Vector{Int},
              rank::Int, comp_rank::Int, dims::Vector{Int32})
    h5open(path, "w") do f
        # HDF5.jl reverses Julia dims → dataspace; give it the reversed-shape array so
        # the on-disk C-order equals `dspace` and the bytes are our `cbuf` verbatim.
        jl = reshape(cbuf, reverse(dspace)...)
        d = HDF5.create_dataset(f, dsname, datatype(Float32), dataspace(size(jl)))
        write(d, jl)
        attributes(d)["Rank"]           = Int32(rank)
        attributes(d)["Component_Rank"] = Int32(comp_rank)
        attributes(d)["Component_Size"] = Int32(length(cbuf) ÷ comp_rank)
        attributes(d)["Dimensions"]     = dims
    end
end

# density: on-disk (N,N,N), i fastest — our density.f32 is exactly that contiguous buffer
emit(joinpath(dir, "GridDensity"), "GridDensity", rd("density.f32", N^3),
     [N, N, N], 3, 1, Int32[N, N, N])

# velocity: on-disk (3,N,N,N) component-major.  gridvel.f32 is [cell,dim] (component
# blocks of N³, i fastest) → already the (3,N,N,N) C-order buffer (comp slowest).
emit(joinpath(dir, "GridVelocities"), "GridVelocities", rd("gridvel.f32", 3*N^3),
     [3, N, N, N], 3, 3, Int32[N, N, N, 3])

# particles: on-disk (3,Np) component-major.  partpos.f32 is [part,dim] (component
# blocks of Np) → the (3,Np) C-order buffer.
for (fn, ds) in (("partpos.f32", "ParticlePositions"), ("partvel.f32", "ParticleVelocities"))
    emit(joinpath(dir, ds), ds, rd(fn, 3*Np), [3, Np], 1, 3, Int32[Np, 3])
end

println("wrote Enzo CICASS ICs to $dir (N=$N, Np=$Np)")
