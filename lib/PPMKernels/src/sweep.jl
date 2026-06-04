# ── Phase 3 — ppm_sweep_1d!: composed 1-D directional PPM sweep ───────────────
# Composes the six certified component kernels into one directional update, in
# the exact order of Enzo's `enzomodules_ppm_sweep_1d_full` (the golden reference):
#
#   calcdiss → inteuler → twoshock → flux_twoshock → euler
#
# No new physics — only orchestration + scratch allocation. The kernels' own
# `synchronize`s order the stages (flux_twoshock reads the OLD zone-centred state,
# euler then overwrites it). Index ranges follow the reference exactly: inteuler
# emits L/R states over i1..i2+1; twoshock resolves those i1..i2+1 INTERFACES
# (note the i2+1); flux_twoshock forms fluxes over i1..i2+1; euler updates the
# i1..i2 cells. `pslice` is an input (as in the reference), not recomputed here.

export ppm_sweep_1d!

"""
    ppm_sweep_1d!(dslice, eslice, geslice, uslice, vslice, wslice, pslice, grslice, dxi;
                  idim, i1, i2, dt, gamma, gravity=0, idual=0, eta1=0.0, eta2=0.0,
                  isteep=0, iflatten=0, idiff=0, ipresfree=0, pmin=1e-20, dfloor=0.0)
        -> (df, ef, uf)

One directional PPM hydro update of a 1-D slab. The six zone-centred slices are
updated IN PLACE; `pslice` (input) is the precomputed pressure; `grslice` the
gravitational acceleration (used when `gravity≠0`). Returns the density / energy /
normal-momentum face fluxes. Element type sets the working precision; all arrays
must live on the same backend. Mirrors `EnzoLib.ppm_sweep_1d_full!`.
"""
function ppm_sweep_1d!(dslice, eslice, geslice, uslice, vslice, wslice, pslice, grslice, dxi;
                       idim::Integer, i1::Integer, i2::Integer, dt::Real, gamma::Real,
                       gravity::Integer = 0, idual::Integer = 0, eta1::Real = 0.0,
                       eta2::Real = 0.0, isteep::Integer = 0, iflatten::Integer = 0,
                       idiff::Integer = 0, ipresfree::Integer = 0, pmin::Real = 1e-20,
                       dfloor::Real = 0.0)
    idim, i1, i2 = Int(idim), Int(i1), Int(i2)
    j1, j2 = 1, 1

    # 0. flattening + diffusion coefficients (1-D regime)
    diffcoef = _zlike(dslice); flatten = _zlike(dslice)
    if iflatten != 0 || idiff != 0
        calcdiss!(diffcoef, flatten, dslice, eslice, uslice, pslice;
                  idim = idim, i1 = i1, i2 = i2, j1 = j1, j2 = j2, gamma = gamma,
                  idiff = idiff, iflatten = iflatten)
    end

    # 1. PPM reconstruction → left/right interface states (i1..i2+1)
    rec = (; (f => _zlike(dslice) for f in
              (:dls, :drs, :pls, :prs, :gels, :gers, :uls, :urs, :vls, :vrs, :wls, :wrs))...)
    inteuler!(rec, dslice, pslice, uslice, vslice, wslice, geslice, grslice, dxi, flatten;
              idim = idim, i1 = i1, i2 = i2, j1 = j1, j2 = j2, dt = dt, gamma = gamma,
              eta2 = eta2, gravity = gravity, idual = idual, isteep = isteep,
              iflatten = iflatten, ipresfree = ipresfree)

    # 2. two-shock Riemann at each interface i1..i2+1  (note the i2+1)
    pbar = _zlike(dslice); ubar = _zlike(dslice)
    twoshock!(pbar, ubar, rec.dls, rec.drs, rec.pls, rec.prs, rec.uls, rec.urs;
              idim = idim, i1 = i1, i2 = i2 + 1, j1 = j1, j2 = j2, gamma = gamma,
              pmin = pmin, ipresfree = ipresfree)

    # 3. Eulerian fluxes (i1..i2+1)
    fx = (; (f => _zlike(dslice) for f in (:df, :ef, :uf, :vf, :wf, :gef, :ges))...)
    flux_twoshock!(fx, rec.dls, rec.drs, rec.pls, rec.prs, rec.gels, rec.gers,
                   rec.uls, rec.urs, rec.vls, rec.vrs, rec.wls, rec.wrs, pbar, ubar,
                   dslice, uslice, vslice, wslice, eslice, geslice, dxi, diffcoef;
                   idim = idim, i1 = i1, i2 = i2, j1 = j1, j2 = j2, dt = dt, gamma = gamma,
                   idiff = idiff, idual = idual)

    # 4. conservative update of the zone-centred state, in place (i1..i2)
    euler!(dslice, eslice, geslice, uslice, vslice, wslice,
           fx.df, fx.ef, fx.uf, fx.vf, fx.wf, fx.gef, fx.ges, grslice, dxi;
           idim = idim, i1 = i1, i2 = i2, j1 = j1, j2 = j2, dt = dt,
           gravity = gravity, idual = idual, dfloor = dfloor)

    return fx.df, fx.ef, fx.uf
end
