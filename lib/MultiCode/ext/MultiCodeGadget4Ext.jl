# ── GADGET-4 halo finding as a harness service (the wrapper-registry on-ramp) ─
#
# The fifth package extension: GADGET-4's FOF+SUBFIND runs as a SERVICE on
# particles in MultiCode's conventions (N×3 rows, positions normalized to
# [0,1)³) — any engine's particle output becomes a halo catalogue.  The
# particle mass is set cosmologically consistently (m = Ωm·ρ_crit·box³/N) so
# FOF's linking length is 0.2× the mean spacing (the recorded G4 trap).
#
# `using Gadget4Lib` activates it.  G4 runs in a child process (it owns
# exit()/MPI) — the D2-correct transport for this code.

module MultiCodeGadget4Ext

using MultiCode
using Gadget4Lib
# extensions see only the parent's deps + triggers: HDF5 comes through Gadget4Lib
using Gadget4Lib.HDF5: h5open, attributes

function MultiCode.run_gadget4_halos(xp::AbstractMatrix; box_mpch::Real = 50.0,
                                     omega_m::Real = 0.308, redshift::Real = 0.0)
    Gadget4Lib.available() || error("the GADGET-4 capi bridge is not built")
    np = size(xp, 1)
    size(xp, 2) == 3 || error("run_gadget4_halos: xp must be N×3 (normalized rows)")
    pos = Matrix{Float64}(undef, 3, np)
    @inbounds for p in 1:np, d in 1:3
        pos[d, p] = mod(xp[p, d], 1.0) * box_mpch
    end
    vel = zeros(3, np)
    rho_crit = 27.7536                       # 1e10 M⊙/h per (Mpc/h)³ at H = 100h
    m = omega_m * rho_crit * box_mpch^3 / np
    d = mktempdir()
    snap = Gadget4Lib.write_snapshot(joinpath(d, "multicode.hdf5"), pos, vel;
                                     mass = m, boxsize = Float64(box_mpch),
                                     time = 1 / (1 + redshift), redshift = Float64(redshift))
    spec = GenicSpec(boxlength = Float64(box_mpch), nsample = 16, zfinal = Float64(redshift))
    r = Gadget4Lib.find_halos(snap, spec; workdir = mktempdir(), snapnum = 0)
    r.rc == 0 || error("GADGET-4 FOF/SUBFIND failed (rc=$(r.rc))")
    ngroups = nsub = 0
    lens = Int[]
    h5open(r.catalog, "r") do f
        ngroups = Int(read(attributes(f["Header"]), "Ngroups_Total")[])
        nsub = Int(read(attributes(f["Header"]), "Nsubhalos_Total")[])
        ngroups > 0 && (lens = Int.(read(f["Group/GroupLen"])))
    end
    return (ngroups = ngroups, nsubhalos = nsub, group_lens = lens,
            catalog = r.catalog, particle_mass = m, free = () -> nothing)
end

end # module
