# ── Phase 2.5 — flux_twoshock: Eulerian fluxes from the resolved states ───────
# Port of Enzo's `flux_twoshock.F`. Evaluates the time-averaged interface state
# (Colella 1982) by sampling the self-similar two-shock solution at x/t=0 — the
# RAREFACTION2 (linear-interpolation) branch — then forms the conservative fluxes
# (with optional artificial diffusion and the dual-energy gas-energy flux+source).
#
# Mostly per-cell over interfaces i1..i2+1. Two structural splits: (1) the bar
# state and the flux assembly are separate kernels to stay under Metal's
# 31-buffer cap; (2) the gas-energy SOURCE `ges = qc·pcent·(ub[i]−ub[i+1])` reads
# the neighbour `ub[i+1]`, so it is its own pass over i1..i2.
#
# The HLL fallback on negative density (`ifallback`) is an error path, not ported.

export flux_twoshock!

@inline _FX_tiny(::Type{T}) where {T} = T(1e-20)

# resolve the time-averaged bar state (pb,db,ub,eb) + upwinded vb,wb,geb
@kernel function _ft_resolve!(pb, db, ub, eb, vb, wb, geb,
                              @Const(dls), @Const(drs), @Const(pls), @Const(prs),
                              @Const(uls), @Const(urs), @Const(vls), @Const(vrs),
                              @Const(wls), @Const(wrs), @Const(gels), @Const(gers),
                              @Const(pbar), @Const(ubar),
                              idim::Int, istart::Int, j1::Int, gamma)
    gi, gj = @index(Global, NTuple)
    i = istart + gi - 1
    j = j1 + gj - 1
    idx = (j - 1) * idim + i
    T = eltype(pb)
    tn = _FX_tiny(T)
    @inbounds begin
        qa = (gamma + one(T)) / (T(2) * gamma)
        pbarv = pbar[idx]; ubarv = ubar[idx]
        sn = (-ubarv >= zero(T)) ? one(T) : -one(T)
        if sn < zero(T)                      # ubar>0 ⇒ upwind from the left
            u0 = uls[idx]; p0 = pls[idx]; d0 = dls[idx]
        else
            u0 = urs[idx]; p0 = prs[idx]; d0 = drs[idx]
        end
        c0 = sqrt(max(gamma * p0 / d0, tn))
        z0 = c0 * d0 * sqrt(max(one(T) + qa * (pbarv / p0 - one(T)), tn))
        dbar = one(T) / (one(T) / d0 - (pbarv - p0) / max(z0 * z0, tn))
        cbar = sqrt(max(gamma * pbarv / dbar, tn))
        if pbarv < p0
            l0 = u0 * sn + c0
            lbar = sn * ubarv + cbar
        else
            l0 = u0 * sn + z0 / d0
            lbar = l0
        end
        # RAREFACTION2: linear interpolation between end states at x/t = 0
        frac = l0 - lbar
        frac = frac < tn ? tn : frac
        frac = (zero(T) - lbar) / frac
        frac = min(max(frac, zero(T)), one(T))
        pbv = p0 * frac + pbarv * (one(T) - frac)
        dbv = d0 * frac + dbar * (one(T) - frac)
        ubv = u0 * frac + ubarv * (one(T) - frac)
        if lbar >= zero(T)                   # inside post-shock region
            pbv = pbarv; dbv = dbar; ubv = ubarv
        end
        if l0 < zero(T)                       # outside the wave entirely
            pbv = p0; dbv = d0; ubv = u0
        end
        # transverse + gas energy by upwinding on the resolved ub
        if ubv > zero(T)
            vbv = vls[idx]; wbv = wls[idx]; gebv = gels[idx]
        else
            vbv = vrs[idx]; wbv = wrs[idx]; gebv = gers[idx]
        end
        ebv = pbv / ((gamma - one(T)) * dbv) + T(0.5) * (ubv * ubv + vbv * vbv + wbv * wbv)
        pb[idx] = pbv; db[idx] = dbv; ub[idx] = ubv; eb[idx] = ebv
        vb[idx] = vbv; wb[idx] = wbv; geb[idx] = gebv
    end
end

# assemble fluxes df,ef,uf,vf,wf (+gef and pcent for the dual source)
@kernel function _ft_flux!(df, ef, uf, vf, wf, gef, pcent,
                           @Const(pb), @Const(db), @Const(ub), @Const(eb),
                           @Const(vb), @Const(wb), @Const(geb),
                           @Const(dslice), @Const(uslice), @Const(vslice),
                           @Const(wslice), @Const(eslice), @Const(geslice),
                           @Const(diffcoef), @Const(dx),
                           idim::Int, istart::Int, j1::Int, dt, idiff::Int, idual::Int, gamma)
    gi, gj = @index(Global, NTuple)
    i = istart + gi - 1
    j = j1 + gj - 1
    idx = (j - 1) * idim + i
    il = idx - 1
    T = eltype(df)
    @inbounds begin
        pbv = pb[idx]; dbv = db[idx]; ubv = ub[idx]
        upb = pbv * ubv
        dub = ubv * dbv
        if idiff != 0
            cf = diffcoef[idx]
            duub = dub * ubv + cf * (dslice[il] * uslice[il] - dslice[idx] * uslice[idx])
            duvb = dub * vb[idx] + cf * (dslice[il] * vslice[il] - dslice[idx] * vslice[idx])
            duwb = dub * wb[idx] + cf * (dslice[il] * wslice[il] - dslice[idx] * wslice[idx])
            dueb = dub * eb[idx] + cf * (dslice[il] * eslice[il] - dslice[idx] * eslice[idx])
            dub  = dub + cf * (dslice[il] - dslice[idx])             # must be last
            dugeb = dub * geb[idx] + cf * (dslice[il] * geslice[il] - dslice[idx] * geslice[idx])
        else
            duub = dub * ubv; duvb = dub * vb[idx]; duwb = dub * wb[idx]; dueb = dub * eb[idx]
            dugeb = dub * geb[idx]
        end
        qc = dt / dx[i]
        df[idx] = qc * dub
        ef[idx] = qc * (dueb + upb)
        uf[idx] = qc * (duub + pbv)
        vf[idx] = qc * duvb
        wf[idx] = qc * duwb
        if idual == 1
            gef[idx] = qc * dugeb
            pcent[idx] = max((gamma - one(T)) * geslice[idx] * dslice[idx], _FX_tiny(T))
        end
    end
end

# gas-energy source ges = qc·pcent·(ub[i]−ub[i+1]), over i1..i2 (needs ub[i+1])
@kernel function _ft_ges!(ges, @Const(pcent), @Const(ub), @Const(dx),
                          idim::Int, istart::Int, j1::Int, dt)
    gi, gj = @index(Global, NTuple)
    i = istart + gi - 1
    j = j1 + gj - 1
    idx = (j - 1) * idim + i
    T = eltype(ges)
    @inbounds ges[idx] = (dt / dx[i]) * pcent[idx] * (ub[idx] - ub[idx + 1])
end

"""
    flux_twoshock!(out, dls, drs, pls, prs, gels, gers, uls, urs, vls, vrs, wls, wrs,
                   pbar, ubar, dslice, uslice, vslice, wslice, eslice, geslice, dx, diffcoef;
                   idim, i1, i2, j1=1, j2=1, dt, gamma, idiff=0, idual=0)
        -> out

Fill `out`, a NamedTuple `(df,ef,uf,vf,wf,gef,ges)` of pre-zeroed flux slabs.
`gef`/`ges` are written only when `idual=1`; `ges` is valid over `i1..i2`.
See file header for scope.
"""
function flux_twoshock!(out, dls, drs, pls, prs, gels, gers, uls, urs, vls, vrs, wls, wrs,
                        pbar, ubar, dslice, uslice, vslice, wslice, eslice, geslice, dx, diffcoef;
                        idim::Integer, i1::Integer, i2::Integer, j1::Integer = 1, j2::Integer = 1,
                        dt::Real, gamma::Real, idiff::Integer = 0, idual::Integer = 0)
    be = KA.get_backend(dls)
    T  = eltype(dls)
    idim, i1, i2, j1 = Int(idim), Int(i1), Int(i2), Int(j1)
    nj = j2 - j1 + 1
    g, dtT = T(gamma), T(dt)
    pb = _zlike(dls); db = _zlike(dls); ub = _zlike(dls); eb = _zlike(dls)
    vb = _zlike(dls); wb = _zlike(dls); geb = _zlike(dls); pcent = _zlike(dls)

    _ft_resolve!(be)(pb, db, ub, eb, vb, wb, geb, dls, drs, pls, prs, uls, urs,
                     vls, vrs, wls, wrs, gels, gers, pbar, ubar,
                     idim, i1, j1, g; ndrange = (i2 - i1 + 2, nj))
    _ft_flux!(be)(out.df, out.ef, out.uf, out.vf, out.wf, out.gef, pcent,
                  pb, db, ub, eb, vb, wb, geb, dslice, uslice, vslice, wslice,
                  eslice, geslice, diffcoef, dx, idim, i1, j1, dtT, Int(idiff), Int(idual),
                  g; ndrange = (i2 - i1 + 2, nj))
    if idual == 1
        _ft_ges!(be)(out.ges, pcent, ub, dx, idim, i1, j1, dtT; ndrange = (i2 - i1 + 1, nj))
    end
    KA.synchronize(be)
    return out
end
