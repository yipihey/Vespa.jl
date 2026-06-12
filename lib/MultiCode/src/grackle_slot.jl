# ── reduced-chemistry guest slot for RAMSES / Arepo ───────────────────────────
#
# Wires the code-neutral GrackleChem service (grackle_service.jl) onto a host
# code's gas state so it can run early-universe primordial chemistry advecting
# only TWO species, HII and H2I.  The host carries those two as density-weighted
# passive scalars (mass density rho*x), exactly the GrackleChem convention.
#
#   chem_init!(...)                 -- initialise the service for the host units
#   chem_step!(rho,eint,HII,H2I;..) -- generic core (the testable seam)
#   ramses_chem_step!(h,lev;..)     -- extract/inject for a RAMSES host
#
# Arepo uses the same chem_step! core once ArepoLib exposes the passive scalars
# (see the note in arepo_chem_step! below).

using .GrackleChem

const _CHEM_DATA_FILE = Ref{String}(
    joinpath(homedir(), "Research", "codes", "grackle", "input", "CloudyData_noUVB.h5"))

"""
    chem_init!(; hubble, Om, OL, a_value, fh, density_units, length_units,
                 time_units, data_file=<CloudyData_noUVB.h5>)

Initialise the reduced primordial-chemistry service for a host's code units
(`*_units` convert host code units → CGS; `hubble` = H0 in km/s/Mpc).
"""
function chem_init!(; hubble::Real, Om::Real, OL::Real, a_value::Real, fh::Real=0.76,
        density_units::Real, length_units::Real, time_units::Real,
        data_file::AbstractString=_CHEM_DATA_FILE[])
    GrackleChem.grackle_reduced_init!(; hubble=hubble, Om=Om, OL=OL, a_value=a_value,
        fh=fh, density_units=density_units, length_units=length_units,
        time_units=time_units, data_file=data_file)
end

"""
    chem_step!(rho, eint, HII, H2I; a_value, dt)

Advance the chemistry+cooling one step.  `eint` (specific internal energy),
`HII`, `H2I` (mass densities rho*x) are updated in place.  This is the code-
neutral core both RAMSES and Arepo call after extracting their gas state.
"""
chem_step!(rho, eint, HII, H2I; a_value, dt) =
    GrackleChem.grackle_reduced_step!(rho, eint, HII, H2I; a_value=a_value, dt=dt)

# ── RAMSES wiring ─────────────────────────────────────────────────────────────
# RAMSES stores uold = (rho, rho*u, E_total, [passive scalars rho*x ...]).  The
# two chemistry species live at hydro var indices `iHII`, `iH2I` (density-
# weighted).  Requires a RAMSES built with nvar >= max(iHII,iH2I) (e.g. -DNVAR=7
# with iHII=6, iH2I=7); RamsesLib.get_hydro/set_hydro already handle any ivar.

"""
    ramses_chem_step!(h, lev; dt, a_value, density_units, length_units,
                      time_units, iHII=6, iH2I=7)

Run one reduced-chemistry step on a RAMSES level: pull (rho, momentum, E_total)
and the two species via `RamsesLib.get_hydro_all`, form the specific internal
energy, call `chem_step!`, and write back E_total and the two species via
`RamsesLib.set_hydro!`.  `chem_init!` must have been called first.
"""
function ramses_chem_step!(h, lev::Integer; dt::Real, a_value::Real,
        density_units::Real, length_units::Real, time_units::Real,
        iHII::Integer=6, iH2I::Integer=7)
    ck, U = RamsesLib.get_hydro_all(h, :uold, lev)      # U :: noct × 8 × nvar
    nv = size(U, 3)
    nv >= max(iHII, iH2I) ||
        error("RAMSES nvar=$nv < $(max(iHII,iH2I)); rebuild with -DNVAR>=$(max(iHII,iH2I)) to carry HII,H2I")

    rho  = Float64.(vec(@view U[:, :, 1]))
    mx   = Float64.(vec(@view U[:, :, 2]))
    my   = Float64.(vec(@view U[:, :, 3]))
    mz   = Float64.(vec(@view U[:, :, 4]))
    Etot = Float64.(vec(@view U[:, :, 5]))
    HII  = Float64.(vec(@view U[:, :, iHII]))
    H2I  = Float64.(vec(@view U[:, :, iH2I]))

    r    = max.(rho, eps())
    kin  = 0.5 .* (mx.^2 .+ my.^2 .+ mz.^2) ./ r          # kinetic energy density
    eint = (Etot .- kin) ./ r                             # specific internal energy

    chem_step!(rho, eint, HII, H2I; a_value=a_value, dt=dt)

    Etot_new = eint .* rho .+ kin                         # cooled internal + same kinetic
    noct = size(U, 1)
    reshape8(v) = reshape(v, noct, 8)
    RamsesLib.set_hydro!(h, :uold, 5,    lev, ck, reshape8(Etot_new))
    RamsesLib.set_hydro!(h, :uold, iHII, lev, ck, reshape8(HII))
    RamsesLib.set_hydro!(h, :uold, iH2I, lev, ck, reshape8(H2I))
    return (; ncells = length(rho))
end

# ── Arepo wiring ──────────────────────────────────────────────────────────────
# Arepo carries the two species as primitive passive-scalar abundances x (the
# :scalars field added to ArepoLib; the bridge keeps the conserved
# PConservedScalars = x*Mass in sync so they advect with the Voronoi flux).
# Arepo stores utherm as specific internal energy directly, so -- unlike RAMSES
# -- no kinetic subtraction is needed.  Requires Arepo built with
# PASSIVE_SCALARS=2 (column 1 = x_HII, column 2 = x_H2I).

"""
    arepo_chem_step!(h; dt, a_value)

Run one reduced-chemistry step on all Arepo gas cells: read rho, utherm and the
two passive-scalar abundances, call `chem_step!` (converting to/from the density-
weighted convention), and write back utherm and the abundances.  `chem_init!`
must have been called first with Arepo's code units.
"""
function arepo_chem_step!(h; dt::Real, a_value::Real)
    rho  = Float64.(ArepoLib.get_cell_field(h, :rho))
    eint = Float64.(ArepoLib.get_cell_field(h, :utherm))      # specific internal energy
    sc   = ArepoLib.get_cell_field(h, :scalars)               # n×2 abundances [x_HII x_H2I]
    HII  = rho .* Float64.(@view sc[:, 1])                    # density-weighted rho*x
    H2I  = rho .* Float64.(@view sc[:, 2])

    chem_step!(rho, eint, HII, H2I; a_value=a_value, dt=dt)

    r = max.(rho, eps())
    ArepoLib.set_cell_field!(h, :utherm, eint)
    ArepoLib.set_cell_field!(h, :scalars, hcat(HII ./ r, H2I ./ r))
    return (; ncells = length(rho))
end
