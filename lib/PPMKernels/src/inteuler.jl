# ── Phase 2.3b — inteuler: Eulerian PPM left/right interface states ───────────
# Port of Enzo's `inteuler.F`. Precomputes the grid/characteristic coefficients,
# reconstructs each field via [`intvar!`], then folds the §3 characteristic (β)
# corrections — with gravity, dual-energy fallback, floors and the pressure-free
# reset — into one fused per-cell "combine" kernel.
#
# SCOPE: the default reconstruction path `iconsrec=0, iposrec=0` (the
# conservative-reconstruction / positivity variants `calc_eigen`/`intprim`/
# `intpos` are deferred). Supported flags: gravity, idual, isteep, iflatten,
# ipresfree; ncolor=0 (colour advection deferred). Certified bit-faithful vs the
# Fortran with those same flags.

export inteuler!

@inline _IE_tiny(::Type{T}) where {T} = T(1e-20)

# ── grid coefficients c1..c6, dx2i (functions of dxi only; 1-D over i) ───────
@kernel function _ie_geom1!(c1, c2, @Const(dxi), idim::Int, istart::Int)
    gi = @index(Global, Linear)
    i = istart + gi - 1
    T = eltype(c1)
    @inbounds begin
        qa = dxi[i] / (dxi[i - 1] + dxi[i] + dxi[i + 1])
        c1[i] = qa * (T(2) * dxi[i - 1] + dxi[i]) / (dxi[i + 1] + dxi[i])
        c2[i] = qa * (T(2) * dxi[i + 1] + dxi[i]) / (dxi[i - 1] + dxi[i])
    end
end

@kernel function _ie_geom2!(c3, c4, c5, c6, dx2i, @Const(dxi), idim::Int, istart::Int)
    gi = @index(Global, Linear)
    i = istart + gi - 1
    T = eltype(c3)
    @inbounds begin
        qa = dxi[i - 2] + dxi[i - 1] + dxi[i] + dxi[i + 1]
        qb = dxi[i - 1] / (dxi[i - 1] + dxi[i])
        qc = (dxi[i - 2] + dxi[i - 1]) / (T(2) * dxi[i - 1] + dxi[i])
        qd = (dxi[i + 1] + dxi[i]) / (T(2) * dxi[i] + dxi[i - 1])
        qb = qb + T(2) * dxi[i] * qb / qa * (qc - qd)
        c3[i] = one(T) - qb
        c4[i] = qb
        c5[i] =  dxi[i]     / qa * qd
        c6[i] = -dxi[i - 1] / qa * qc
        dx2i[i] = T(0.5) / dxi[i]
    end
end

# ── per-slice characteristic distances char1/char2, cm/c0/cp (over i1-1..i2+1) ─
@kernel function _ie_slice!(char1, char2, cm, c0, cp, @Const(p), @Const(d), @Const(u),
                            @Const(dx2i), idim::Int, istart::Int, j1::Int,
                            gamma, dt, ipresfree::Int)
    gi, gj = @index(Global, NTuple)
    i = istart + gi - 1
    j = j1 + gj - 1
    idx = (j - 1) * idim + i
    T = eltype(char1)
    @inbounds begin
        cs = ipresfree == 1 ? _IE_tiny(T) : sqrt(gamma * p[idx] / d[idx])
        h = dx2i[i]
        char1[idx] = max(zero(T),  dt * (u[idx] + cs)) * h
        char2[idx] = max(zero(T), -dt * (u[idx] - cs)) * h
        cm[idx] = dt * (u[idx] - cs) * h
        c0[idx] = dt *  u[idx]       * h
        cp[idx] = dt * (u[idx] + cs) * h
    end
end

# ── steepening precompute (isteep): d2d/dxb then steepen, density-only ────────
@kernel function _ie_steep_d2d!(d2d, dxb, @Const(d), @Const(dxi),
                                idim::Int, istart::Int, j1::Int)
    gi, gj = @index(Global, NTuple)
    i = istart + gi - 1
    j = j1 + gj - 1
    idx = (j - 1) * idim + i
    T = eltype(d2d)
    @inbounds begin
        qa = dxi[i - 1] + dxi[i] + dxi[i + 1]
        t = (d[idx + 1] - d[idx]) / (dxi[i + 1] + dxi[i])
        d2d[idx] = (t - (d[idx] - d[idx - 1]) / (dxi[i] + dxi[i - 1])) / qa
        dxb[idx] = T(0.5) * (dxi[i] + dxi[i + 1])
    end
end

@kernel function _ie_steep!(steepen, @Const(d2d), @Const(dxb), @Const(d), @Const(p),
                            idim::Int, istart::Int, j1::Int, gamma)
    gi, gj = @index(Global, NTuple)
    i = istart + gi - 1
    j = j1 + gj - 1
    base = (j - 1) * idim
    idx = base + i
    T = eltype(steepen)
    @inbounds begin
        dm1 = d[idx - 1]; dp1 = d[idx + 1]
        qc = abs(dp1 - dm1) - T(0.01) * min(abs(dp1), abs(dm1))
        dxbm = dxb[idx - 1]; dxbi = dxb[idx]
        s1 = (d2d[idx - 1] - d2d[idx + 1]) * (dxbm^3 + dxbi^3) /
             ((dxbi + dxbm) * (dp1 - dm1 + _IE_tiny(T)))
        if d2d[idx + 1] * d2d[idx - 1] > zero(T); s1 = zero(T); end
        if qc <= zero(T); s1 = zero(T); end
        s2 = max(zero(T), min(T(20) * (s1 - T(0.05)), one(T)))
        qa = abs(dp1 - dm1) / min(dp1, dm1)
        qb = abs(p[idx + 1] - p[idx - 1]) / min(p[idx + 1], p[idx - 1])
        steepen[idx] = (gamma * T(0.1) * qa >= qb) ? s2 : zero(T)
    end
end

# The β-correction is split into small per-cell kernels because Apple GPUs cap a
# kernel at 31 argument buffers — the fused form needs ~50. Each piece stays well
# under the limit. p/u domain-of-dependence averages → raw left/right states →
# advected-state upwinding → dual fallback → floors/pressure-free.

# domain-of-dependence averages of p (eq. 3.5), over i1..i2+1
@kernel function _ie_pavg!(plm, prm, plp, prp, @Const(pl), @Const(pr),
                           @Const(dp), @Const(p6), @Const(cm), @Const(cp),
                           idim::Int, istart::Int, j1::Int)
    gi, gj = @index(Global, NTuple)
    i = istart + gi - 1; j = j1 + gj - 1
    idx = (j - 1) * idim + i; il = idx - 1
    T = eltype(plm); ft = _FT(T)
    @inbounds begin
        cmL = cm[il]; cmI = cm[idx]; cpL = cp[il]; cpI = cp[idx]
        plm[idx] = pr[il]  - cmL * (dp[il]  - (one(T) - ft * cmL) * p6[il])
        prm[idx] = pl[idx] - cmI * (dp[idx] + (one(T) + ft * cmI) * p6[idx])
        plp[idx] = pr[il]  - cpL * (dp[il]  - (one(T) - ft * cpL) * p6[il])
        prp[idx] = pl[idx] - cpI * (dp[idx] + (one(T) + ft * cpI) * p6[idx])
    end
end

# domain-of-dependence averages of u, over i1..i2+1
@kernel function _ie_uavg!(ulm, urm, ulp, urp, @Const(ul), @Const(ur),
                           @Const(du), @Const(u6), @Const(cm), @Const(cp),
                           idim::Int, istart::Int, j1::Int)
    gi, gj = @index(Global, NTuple)
    i = istart + gi - 1; j = j1 + gj - 1
    idx = (j - 1) * idim + i; il = idx - 1
    T = eltype(ulm); ft = _FT(T)
    @inbounds begin
        cmL = cm[il]; cmI = cm[idx]; cpL = cp[il]; cpI = cp[idx]
        ulm[idx] = ur[il]  - cmL * (du[il]  - (one(T) - ft * cmL) * u6[il])
        urm[idx] = ul[idx] - cmI * (du[idx] + (one(T) + ft * cmI) * u6[idx])
        ulp[idx] = ur[il]  - cpL * (du[il]  - (one(T) - ft * cpL) * u6[il])
        urp[idx] = ul[idx] - cpI * (du[idx] + (one(T) + ft * cpI) * u6[idx])
    end
end

# raw left states pls,uls,dls (eq. 3.6–3.7a), over i1..i2+1
@kernel function _ie_primary_l!(pls, uls, dls, @Const(pla), @Const(pl0), @Const(dla),
                                @Const(dl0), @Const(ula), @Const(plm), @Const(plp),
                                @Const(ulm), @Const(ulp), @Const(cm), @Const(c0),
                                @Const(cp), @Const(grslice), idim::Int, istart::Int,
                                j1::Int, gamma, dt, gravity::Int)
    gi, gj = @index(Global, NTuple)
    i = istart + gi - 1; j = j1 + gj - 1
    idx = (j - 1) * idim + i; il = idx - 1
    T = eltype(pls)
    @inbounds begin
        cla = sqrt(max(gamma * pla[idx] * dla[idx], zero(T)))
        f1 = one(T) / cla
        blp = (ula[idx] - ulp[idx]) + (pla[idx] - plp[idx]) * f1
        blm = (ula[idx] - ulm[idx]) - (pla[idx] - plm[idx]) * f1
        bl0 = (pla[idx] - pl0[idx]) * f1 * f1 + one(T) / dla[idx] - one(T) / dl0[idx]
        if gravity == 1
            g = T(0.25) * dt * (grslice[il] + grslice[idx]); blp -= g; blm -= g
        end
        f1 = T(0.5) / cla; blp = -blp * f1; blm = blm * f1
        if cp[il] <= zero(T); blp = zero(T); end
        if cm[il] <= zero(T); blm = zero(T); end
        if c0[il] <= zero(T); bl0 = zero(T); end
        pls[idx] = pla[idx] + (blp + blm) * cla * cla
        uls[idx] = ula[idx] + (blp - blm) * cla
        dls[idx] = one(T) / (one(T) / dla[idx] - (bl0 + blp + blm))
    end
end

# raw right states prs,urs,drs (eq. 3.6–3.7b), over i1..i2+1
@kernel function _ie_primary_r!(prs, urs, drs, @Const(pra), @Const(pr0), @Const(dra),
                                @Const(dr0), @Const(ura), @Const(prm), @Const(prp),
                                @Const(urm), @Const(urp), @Const(cm), @Const(c0),
                                @Const(cp), @Const(grslice), idim::Int, istart::Int,
                                j1::Int, gamma, dt, gravity::Int)
    gi, gj = @index(Global, NTuple)
    i = istart + gi - 1; j = j1 + gj - 1
    idx = (j - 1) * idim + i; il = idx - 1
    T = eltype(prs)
    @inbounds begin
        cra = sqrt(max(gamma * pra[idx] * dra[idx], zero(T)))
        f1 = one(T) / cra
        brp = (ura[idx] - urp[idx]) + (pra[idx] - prp[idx]) * f1
        brm = (ura[idx] - urm[idx]) - (pra[idx] - prm[idx]) * f1
        br0 = (pra[idx] - pr0[idx]) * f1 * f1 + one(T) / dra[idx] - one(T) / dr0[idx]
        if gravity == 1
            g = T(0.25) * dt * (grslice[il] + grslice[idx]); brp -= g; brm -= g
        end
        f1 = T(0.5) / cra; brp = -brp * f1; brm = brm * f1
        if cp[idx] >= zero(T); brp = zero(T); end
        if cm[idx] >= zero(T); brm = zero(T); end
        if c0[idx] >= zero(T); br0 = zero(T); end
        prs[idx] = pra[idx] + (brp + brm) * cra * cra
        urs[idx] = ura[idx] + (brp - brm) * cra
        drs[idx] = one(T) / (one(T) / dra[idx] - (br0 + brp + brm))
    end
end

# upwind-selected advected states v,w,(ge), over i1..i2+1
@kernel function _ie_advect!(vls, vrs, wls, wrs, gels, gers,
                             @Const(vla), @Const(vra), @Const(vl0), @Const(vr0),
                             @Const(wla), @Const(wra), @Const(wl0), @Const(wr0),
                             @Const(gela), @Const(gera), @Const(gel0), @Const(ger0),
                             @Const(uslice), idim::Int, istart::Int, j1::Int, idual::Int)
    gi, gj = @index(Global, NTuple)
    i = istart + gi - 1; j = j1 + gj - 1
    idx = (j - 1) * idim + i; il = idx - 1
    T = eltype(vls)
    @inbounds begin
        if uslice[il] <= zero(T)
            vls[idx] = vla[idx]; wls[idx] = wla[idx]
            if idual == 1; gels[idx] = gela[idx]; end
        else
            vls[idx] = vl0[idx]; wls[idx] = wl0[idx]
            if idual == 1; gels[idx] = gel0[idx]; end
        end
        if uslice[idx] >= zero(T)
            vrs[idx] = vra[idx]; wrs[idx] = wra[idx]
            if idual == 1; gers[idx] = gera[idx]; end
        else
            vrs[idx] = vr0[idx]; wrs[idx] = wr0[idx]
            if idual == 1; gers[idx] = ger0[idx]; end
        end
    end
end

# dual-energy fallback: discard corrections in hypersonic / runaway cells
@kernel function _ie_dual!(pls, prs, uls, urs, dls, drs,
                           @Const(pla), @Const(pra), @Const(dla), @Const(dra),
                           @Const(ula), @Const(ura), @Const(cm), @Const(c0), @Const(cp),
                           idim::Int, istart::Int, j1::Int, gamma, eta2)
    gi, gj = @index(Global, NTuple)
    i = istart + gi - 1; j = j1 + gj - 1
    idx = (j - 1) * idim + i; il = idx - 1
    T = eltype(pls)
    @inbounds begin
        if gamma * pla[idx] / dla[idx] < eta2 * ula[idx]^2 ||
           max(abs(cm[il]), abs(c0[il]), abs(cp[il])) < T(1e-3) || dls[idx] / dla[idx] > T(5)
            pls[idx] = pla[idx]; uls[idx] = ula[idx]; dls[idx] = dla[idx]
        end
        if gamma * pra[idx] / dra[idx] < eta2 * ura[idx]^2 ||
           max(abs(cm[idx]), abs(c0[idx]), abs(cp[idx])) < T(1e-3) || drs[idx] / dra[idx] > T(5)
            prs[idx] = pra[idx]; urs[idx] = ura[idx]; drs[idx] = dra[idx]
        end
    end
end

# floors on p,d and the pressure-free density reset, over i1..i2+1
@kernel function _ie_finalize!(pls, prs, dls, drs, @Const(dla), @Const(dra),
                               idim::Int, istart::Int, j1::Int, ipresfree::Int)
    gi, gj = @index(Global, NTuple)
    i = istart + gi - 1; j = j1 + gj - 1
    idx = (j - 1) * idim + i
    T = eltype(pls); tn = _IE_tiny(T)
    @inbounds begin
        pls[idx] = max(pls[idx], tn); prs[idx] = max(prs[idx], tn)
        dls[idx] = ipresfree == 1 ? dla[idx] : max(dls[idx], tn)
        drs[idx] = ipresfree == 1 ? dra[idx] : max(drs[idx], tn)
    end
end

# small helper: a zeroed scratch array shaped like `proto` (pooled when active)
_zlike(proto) = _scratch(proto, length(proto); zero = true)

"""
    inteuler!(out, dslice, pslice, uslice, vslice, wslice, geslice, grslice, dxi, flatten;
              idim, i1, i2, j1=1, j2=1, dt, gamma, eta2=0.0,
              gravity=0, idual=0, isteep=0, iflatten=0, ipresfree=0)

PPM Eulerian interface states. `out` is a NamedTuple of the twelve output slabs
`(dls,drs,pls,prs,gels,gers,uls,urs,vls,vrs,wls,wrs)` (pre-allocated, zeroed).
`flatten` is the calcdiss flattening slab (ignored unless `iflatten≠0`); pass a
zero slab otherwise. See file header for scope. Element type sets precision.
"""
function inteuler!(out, dslice, pslice, uslice, vslice, wslice, geslice, grslice,
                   dxi, flatten;
                   idim::Integer, i1::Integer, i2::Integer, j1::Integer = 1, j2::Integer = 1,
                   dt::Real, gamma::Real, eta2::Real = 0.0,
                   gravity::Integer = 0, idual::Integer = 0, isteep::Integer = 0,
                   iflatten::Integer = 0, ipresfree::Integer = 0)
    be = KA.get_backend(dslice)
    T  = eltype(dslice)
    idim, i1, i2, j1 = Int(idim), Int(i1), Int(i2), Int(j1)
    nj = j2 - j1 + 1
    g, dtT, eta2T = T(gamma), T(dt), T(eta2)

    # geometry coefficients (1-D)
    c1 = _zlike(dxi); c2 = _zlike(dxi); c3 = _zlike(dxi); c4 = _zlike(dxi)
    c5 = _zlike(dxi); c6 = _zlike(dxi); dx2i = _zlike(dxi)
    _ie_geom1!(be)(c1, c2, dxi, idim, i1 - 2; ndrange = i2 - i1 + 5)
    _ie_geom2!(be)(c3, c4, c5, c6, dx2i, dxi, idim, i1 - 1; ndrange = i2 - i1 + 4)

    # per-slice characteristic distances (2-D)
    char1 = _zlike(dslice); char2 = _zlike(dslice)
    cm = _zlike(dslice); c0 = _zlike(dslice); cp = _zlike(dslice)
    _ie_slice!(be)(char1, char2, cm, c0, cp, pslice, dslice, uslice, dx2i,
                   idim, i1 - 1, j1, g, dtT, Int(ipresfree); ndrange = (i2 - i1 + 3, nj))

    geom = (; c1, c2, c3, c4, c5, c6, char1, char2, c0)

    # steepening coefficients (density only)
    steepen = _zlike(dslice)
    if isteep != 0
        d2d = _zlike(dslice); dxb = _zlike(dslice)
        _ie_steep_d2d!(be)(d2d, dxb, dslice, dxi, idim, i1 - 2, j1; ndrange = (i2 - i1 + 5, nj))
        _ie_steep!(be)(steepen, d2d, dxb, dslice, pslice, idim, i1 - 1, j1, g;
                       ndrange = (i2 - i1 + 3, nj))
    end

    # reconstruct each field; keep p's and u's parabola for the β-correction
    mkout() = (_zlike(dslice), _zlike(dslice), _zlike(dslice), _zlike(dslice))
    mktmp() = (_zlike(dslice), _zlike(dslice), _zlike(dslice), _zlike(dslice))
    kw = (; idim, i1, i2, j1, j2, isteep, iflatten)

    od = mkout(); intvar!(od, mktmp(), dslice, geom, steepen, flatten; kw...)
    tp = mktmp(); op = mkout(); intvar!(op, tp, pslice, geom, steepen, flatten; kw..., isteep = 0)
    tu = mktmp(); ou = mkout(); intvar!(ou, tu, uslice, geom, steepen, flatten; kw..., isteep = 0)
    ov = mkout(); intvar!(ov, mktmp(), vslice, geom, steepen, flatten; kw..., isteep = 0)
    ow = mkout(); intvar!(ow, mktmp(), wslice, geom, steepen, flatten; kw..., isteep = 0)
    oge = mkout()
    if idual == 1
        intvar!(oge, mktmp(), geslice, geom, steepen, flatten; kw..., isteep = 0)
    end

    (dla, dra, dl0, dr0) = od
    (pla, pra, pl0, pr0) = op; (dp, pl, pr, p6) = tp
    (ula, ura, ul0, ur0) = ou; (du, ul, ur, u6) = tu
    (vla, vra, vl0, vr0) = ov
    (wla, wra, wl0, wr0) = ow
    (gela, gera, gel0, ger0) = oge

    # β-correction pipeline (split to respect Metal's 31-buffer kernel cap)
    plm = _zlike(dslice); prm = _zlike(dslice); plp = _zlike(dslice); prp = _zlike(dslice)
    ulm = _zlike(dslice); urm = _zlike(dslice); ulp = _zlike(dslice); urp = _zlike(dslice)
    nd = (i2 - i1 + 2, nj)
    gv = Int(gravity)
    _ie_pavg!(be)(plm, prm, plp, prp, pl, pr, dp, p6, cm, cp, idim, i1, j1; ndrange = nd)
    _ie_uavg!(be)(ulm, urm, ulp, urp, ul, ur, du, u6, cm, cp, idim, i1, j1; ndrange = nd)
    _ie_primary_l!(be)(out.pls, out.uls, out.dls, pla, pl0, dla, dl0, ula, plm, plp,
                       ulm, ulp, cm, c0, cp, grslice, idim, i1, j1, g, dtT, gv; ndrange = nd)
    _ie_primary_r!(be)(out.prs, out.urs, out.drs, pra, pr0, dra, dr0, ura, prm, prp,
                       urm, urp, cm, c0, cp, grslice, idim, i1, j1, g, dtT, gv; ndrange = nd)
    _ie_advect!(be)(out.vls, out.vrs, out.wls, out.wrs, out.gels, out.gers,
                    vla, vra, vl0, vr0, wla, wra, wl0, wr0, gela, gera, gel0, ger0,
                    uslice, idim, i1, j1, Int(idual); ndrange = nd)
    if idual == 1
        _ie_dual!(be)(out.pls, out.prs, out.uls, out.urs, out.dls, out.drs,
                      pla, pra, dla, dra, ula, ura, cm, c0, cp,
                      idim, i1, j1, g, eta2T; ndrange = nd)
    end
    _ie_finalize!(be)(out.pls, out.prs, out.dls, out.drs, dla, dra,
                      idim, i1, j1, Int(ipresfree); ndrange = nd)
    KA.synchronize(be)
    return out
end
