# ── Phase 2.6 — euler: conservative flux-divergence update ────────────────────
# Port of Enzo's `euler.F`. Advances the zone-centred state by the sweep-direction
# flux divergence (eq. 3.1), then the dual-energy gas-energy law (flux + source)
# and the gravity source (GRAVITY_METHOD1). Fully per-cell over i1..i2: each cell
# reads its own old state and the fluxes at i and i+1, and updates in place. The
# density is written LAST so the momentum/energy updates see the old density.
#
# The (disabled-by-default) second-order gravity correction is not ported.

export euler!

@inline _EU_tiny(::Type{T}) where {T} = T(1e-20)

@kernel function _eu_kernel!(dslice, eslice, geslice, uslice, vslice, wslice,
                             @Const(df), @Const(ef), @Const(uf), @Const(vf), @Const(wf),
                             @Const(gef), @Const(ges), @Const(grslice), @Const(dx),
                             idim::Int, istart::Int, j1::Int,
                             dt, gravity::Int, idual::Int, dfloor)
    gi, gj = @index(Global, NTuple)
    i = istart + gi - 1
    j = j1 + gj - 1
    idx = (j - 1) * idim + i
    ip = idx + 1
    T = eltype(dslice)
    @inbounds begin
        dold = dslice[idx]
        dnu = dold + (df[idx] - df[ip])
        if dfloor > zero(T)
            dnu = max(dnu, dfloor)
        end
        dnuinv = one(T) / dnu
        uold = uslice[idx]
        un = (uslice[idx] * dold + (uf[idx] - uf[ip])) * dnuinv
        vn = (vslice[idx] * dold + (vf[idx] - vf[ip])) * dnuinv
        wn = (wslice[idx] * dold + (wf[idx] - wf[ip])) * dnuinv
        en = max(T(0.1) * eslice[idx], (eslice[idx] * dold + (ef[idx] - ef[ip])) * dnuinv)
        uslice[idx] = un; vslice[idx] = vn; wslice[idx] = wn; eslice[idx] = en

        if idual == 1
            gen = max((geslice[idx] * dold + (gef[idx] - gef[ip]) + ges[idx]) * dnuinv,
                      T(0.5) * geslice[idx])
            geslice[idx] = gen
        end

        if gravity == 1                       # GRAVITY_METHOD1 (time-centred accel)
            gu = uslice[idx] + dt * grslice[idx] * T(0.5) * (dold * dnuinv + one(T))
            uslice[idx] = gu
            ge_ = eslice[idx] + dt * grslice[idx] * T(0.5) * (gu + uold * dold * dnuinv)
            eslice[idx] = max(ge_, _EU_tiny(T))
        end

        dslice[idx] = dnu                     # density updated last
    end
end

"""
    euler!(dslice, eslice, geslice, uslice, vslice, wslice,
           df, ef, uf, vf, wf, gef, ges, grslice, dx;
           idim, i1, i2, j1=1, j2=1, dt, gravity=0, idual=0, dfloor=0.0)
        -> (dslice, eslice, geslice, uslice, vslice, wslice)

Update the six zone-centred slices IN PLACE over `i1..i2` from the fluxes. `gef`/
`ges` are consumed only when `idual=1`; `grslice` only when `gravity=1`. Element
type sets precision.
"""
function euler!(dslice, eslice, geslice, uslice, vslice, wslice,
                df, ef, uf, vf, wf, gef, ges, grslice, dx;
                idim::Integer, i1::Integer, i2::Integer, j1::Integer = 1, j2::Integer = 1,
                dt::Real, gravity::Integer = 0, idual::Integer = 0, dfloor::Real = 0.0)
    be = KA.get_backend(dslice)
    T = eltype(dslice)
    nj = j2 - j1 + 1
    _eu_kernel!(be)(dslice, eslice, geslice, uslice, vslice, wslice,
                    df, ef, uf, vf, wf, gef, ges, grslice, dx,
                    Int(idim), Int(i1), Int(j1), T(dt), Int(gravity), Int(idual), T(dfloor);
                    ndrange = (i2 - i1 + 1, nj))
    KA.synchronize(be)
    return dslice, eslice, geslice, uslice, vslice, wslice
end
