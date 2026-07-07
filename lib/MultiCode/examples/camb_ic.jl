# camb_ic.jl — reader for the correctly-normalized CAMB ICs written by
# tools/scout_ics/gen_camb_ic.py.  Produces the SAME NamedTuple as
# MusicIC.read_music_grafic so the BlockAMR driver consumes it unchanged
# (BAM_IC=camb, BAM_CAMB_IC=<path>).  Unlike MUSIC (which under-normalized the DM
# P(k) ~7× at z=1000: matter-only growth + a realization miss), these match the
# CAMB linear P_cdm(k,z_start) to sub-percent by construction — see the
# normalization audit in project_scout_planck18 memory.
#
# Binary (little-endian): Int64 n, Int64 Np, Float64 box[Mpc/h], Float64 zstart,
# then Float32: dm_pos[Np,3] (box frac, per-particle row-major x,y,z),
# dm_vel[Np,3] (km/s), gas_delta[n^3] (i-fastest), gas_vel[Np,3] (km/s).
module CambIC

"""
    read_camb_ic(path) -> NamedTuple

Read a gen_camb_ic.py binary into the CICASS/MUSIC-snapshot layout:
`gas_delta` (n³ vec, i-fastest), `gas_vel` (N×3 km/s), `dm_pos` (N×3 box fraction),
`dm_vel` (N×3 km/s peculiar), plus `n, box [Mpc/h], zinit, Np`.
"""
function read_camb_ic(path::AbstractString)
    open(path, "r") do io
        n  = read(io, Int64); Np = read(io, Int64)
        box = read(io, Float64); zstart = read(io, Float64)
        rd(cnt) = reshape(read!(io, Vector{Float32}(undef, cnt)), :)
        # dm_pos/vel written (Np,3) row-major → julia reshape(3,Np) is [axis,particle]
        dmp = permutedims(reshape(rd(3Np), 3, Int(Np)))     # (Np,3)
        dmv = permutedims(reshape(rd(3Np), 3, Int(Np)))
        gdelta = Float64.(rd(n^3))                          # i-fastest
        gv  = permutedims(reshape(rd(3Np), 3, Int(Np)))
        return (; n = Int(n), box = box, zinit = zstart,
                omega_m = 0.315, hconst = 0.674,
                gas_delta = gdelta, gas_vel = Float64.(gv),
                dm_pos = Float64.(dmp), dm_vel = Float64.(dmv), Np = Int(Np))
    end
end

end # module
