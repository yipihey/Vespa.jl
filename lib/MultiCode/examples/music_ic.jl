# MUSIC grafic2 → a CICASS-snapshot-compatible NamedTuple for the BlockAMR driver.
# Reads a MUSIC grafic level dir (ic_deltab, ic_velb*, ic_velc*, ic_posc*) and
# returns gas δ + velocities + DM positions/velocities in the same layout/units
# (physical peculiar km/s; positions in box fraction) the CICASS path provides.
module MusicIC
using Statistics
function _read_grafic(path)
    open(path, "r") do io
        read(io, Int32); n = Int(read(io, Int32)); read(io, Int32); read(io, Int32)
        hv = [read(io, Float32) for _ in 1:8]; read(io, Int32)
        A = Array{Float32}(undef, n, n, n); plane = Vector{Float32}(undef, n*n)
        for k in 1:n
            read(io, Int32); read!(io, plane); A[:, :, k] = reshape(plane, n, n); read(io, Int32)
        end
        return A, (; n, dx = hv[1], astart = hv[5], om = hv[6], h0 = hv[8])
    end
end

"""
    read_music_grafic(dir) -> snapshot NamedTuple

`dir` = a MUSIC grafic2 `level_XXX` directory.  Returns fields mirroring
`CICASSLib.read_snapshot`: box [Mpc/h], zinit, gas_delta (n³ vec, i-fastest),
gas_vel (N×3 km/s), dm_pos (N×3 box fraction), dm_vel (N×3 km/s), plus the
per-particle DM mass in 1e10 M⊙/h (for cross-checks).
"""
function read_music_grafic(dir::AbstractString)
    db, H = _read_grafic(joinpath(dir, "ic_deltab")); n = H.n
    box_mpch = n * H.dx * H.h0        # dx is Mpc (grafic); ×h0 → Mpc/h? h0 here is H0/100? see below
    vbx,_ = _read_grafic(joinpath(dir, "ic_velbx")); vby,_ = _read_grafic(joinpath(dir, "ic_velby")); vbz,_ = _read_grafic(joinpath(dir, "ic_velbz"))
    vcx,_ = _read_grafic(joinpath(dir, "ic_velcx")); vcy,_ = _read_grafic(joinpath(dir, "ic_velcy")); vcz,_ = _read_grafic(joinpath(dir, "ic_velcz"))
    pcx,_ = _read_grafic(joinpath(dir, "ic_poscx")); pcy,_ = _read_grafic(joinpath(dir, "ic_poscy")); pcz,_ = _read_grafic(joinpath(dir, "ic_poscz"))
    # h: grafic header h0 is H0 (e.g. 67.4) → h=0.674; box in Mpc/h = n*dx*h  (dx in Mpc)
    h = H.h0 > 2 ? H.h0/100 : H.h0
    box = n * H.dx * h                          # Mpc/h
    z = 1/H.astart - 1
    N = n^3
    # DM positions: Lagrangian grid (cell center) + displacement, in box fraction [0,1)
    # grafic arrays are A[i,j,k] with i fastest; particle p = (i,j,k)
    dm_pos = Array{Float32}(undef, N, 3); dm_vel = similar(dm_pos); gas_vel = similar(dm_pos)
    dxfrac = 1.0f0 / n                          # cell size in box fraction
    # grafic ic_posc* displacements are in Mpc/h (NOT Mpc — verified by the v/Ψ
    # growing-mode cross-check v_raw/Ψ_raw = a·f·H/h, and P(k) recovery vs CAMB).
    # box fraction = displacement[Mpc/h]/box[Mpc/h] = disp/box.  (The earlier
    # disp/(n·dx) divided by box in Mpc = box/h → under-displaced everything by h.)
    boxf = Float32(box)
    @inbounds for k in 1:n, j in 1:n, i in 1:n
        p = i + (j-1)*n + (k-1)*n^2
        dm_pos[p,1] = mod((i-0.5f0)*dxfrac + pcx[i,j,k]/boxf, 1.0f0)
        dm_pos[p,2] = mod((j-0.5f0)*dxfrac + pcy[i,j,k]/boxf, 1.0f0)
        dm_pos[p,3] = mod((k-0.5f0)*dxfrac + pcz[i,j,k]/boxf, 1.0f0)
        dm_vel[p,1] = vcx[i,j,k]; dm_vel[p,2] = vcy[i,j,k]; dm_vel[p,3] = vcz[i,j,k]
        gas_vel[p,1] = vbx[i,j,k]; gas_vel[p,2] = vby[i,j,k]; gas_vel[p,3] = vbz[i,j,k]
    end
    gas_delta = vec(db)                         # i-fastest already
    return (; n, box, zinit = z, omega_m = Float64(H.om), hconst = Float64(h),
            gas_delta, gas_vel, dm_pos, dm_vel, Np = N)
end
end # module
