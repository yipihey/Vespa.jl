# ── Phase 2.4 — twoshock: two-shock approximate Riemann solver ────────────────
# Port of Enzo's `twoshock.F`. Resolves the interface pressure `pbar` and normal
# velocity `ubar` from the reconstructed left/right states via Newton iteration
# (van Leer 1979). Purely per-cell: each interface is an independent root-find,
# so one thread owns one interface and runs the (≤8-iteration) loop with a local
# convergence flag — the Fortran's `mask` early-exit, per cell.
#
# The gravity path is dead in the Fortran (`if gravity==99`) and the dual block
# is `#ifdef UNUSED`, so neither is ported. The convergence tolerance is
# precision-dependent, matching Enzo's CONFIG_BFLOAT_8 (1e-14) / _4 (1e-7).

export twoshock!

@inline _ts_tol(::Type{T}) where {T} = T === Float64 ? T(1e-14) : T(1e-7)
const _TS_NUMITER = 8

@kernel function _ts_kernel!(pbar, ubar, @Const(dls), @Const(drs), @Const(pls),
                             @Const(prs), @Const(uls), @Const(urs),
                             idim::Int, istart::Int, j1::Int,
                             gamma, pmin, ipresfree::Int)
    gi, gj = @index(Global, NTuple)
    i = istart + gi - 1
    j = j1 + gj - 1
    idx = (j - 1) * idim + i
    T = eltype(pbar)
    @inbounds begin
        plsv = pls[idx]; prsv = prs[idx]; dlsv = dls[idx]; drsv = drs[idx]
        ulsv = uls[idx]; ursv = urs[idx]
        if ipresfree == 1
            pbar[idx] = pmin
            ubar[idx] = T(0.5) * (ulsv + ursv)
        else
            qa = (gamma + one(T)) / (T(2) * gamma)
            gp1 = gamma + one(T)
            cl = sqrt(gamma * plsv * dlsv)
            cr = sqrt(gamma * prsv * drsv)
            ps = (cr * plsv + cl * prsv + cr * cl * (ulsv - ursv)) / (cr + cl)
            ps = max(ps, pmin)
            old_ps = ps
            conv = false
            tol = _ts_tol(T)
            ubl = zero(T); ubr = zero(T); dpdul = zero(T); dpdur = zero(T)
            for _n in 2:_TS_NUMITER
                if !conv
                    zl = cl * sqrt(one(T) + qa * (ps / plsv - one(T)))
                    zr = cr * sqrt(one(T) + qa * (ps / prsv - one(T)))
                    ubl = ulsv - (ps - plsv) / zl
                    ubr = ursv + (ps - prsv) / zr
                    dpdul = -T(4) * zl^3 / dlsv / (T(4) * zl^2 / dlsv - gp1 * (ps - plsv))
                    dpdur =  T(4) * zr^3 / drsv / (T(4) * zr^2 / drsv - gp1 * (ps - prsv))
                    ps = ps + (ubr - ubl) * dpdur * dpdul / (dpdur - dpdul)
                    ps = max(ps, pmin)
                    delta = ps - old_ps
                    old_ps = ps
                    if abs(delta / ps) < tol
                        conv = true
                    end
                end
            end
            if ps < pmin
                ps = min(plsv, prsv)
            end
            pbar[idx] = ps
            ubar[idx] = ubl + (ubr - ubl) * dpdur / (dpdur - dpdul)
        end
    end
end

"""
    twoshock!(pbar, ubar, dls, drs, pls, prs, uls, urs;
              idim, i1, i2, j1=1, j2=1, gamma, pmin=1e-20, ipresfree=0) -> (pbar, ubar)

Resolve interface `(pbar, ubar)` from the left/right states over the active
region. Element type `T = eltype(pbar)` sets precision (and selects the Newton
tolerance). Inputs are not modified.
"""
function twoshock!(pbar, ubar, dls, drs, pls, prs, uls, urs;
                   idim::Integer, i1::Integer, i2::Integer, j1::Integer = 1,
                   j2::Integer = 1, gamma::Real, pmin::Real = 1e-20, ipresfree::Integer = 0)
    be = KA.get_backend(pbar)
    T = eltype(pbar)
    nj = j2 - j1 + 1
    _ts_kernel!(be)(pbar, ubar, dls, drs, pls, prs, uls, urs,
                    Int(idim), Int(i1), Int(j1), T(gamma), T(pmin), Int(ipresfree);
                    ndrange = (i2 - i1 + 1, nj))
    KA.synchronize(be)
    return pbar, ubar
end
