# ── Phase 2.3a — intvar: per-variable PPM parabolic reconstruction ────────────
# Port of Enzo's `intvar.F` (the workhorse `inteuler` delegates to for each of
# d,p,u,v,w,ge). Given a 1-D field `q` it produces the monotonized parabola and
# the characteristic-averaged interface states `qla/qra` (from char1/char2) and
# `ql0/qr0` (from c0), plus the intermediates `dq/ql/qr/q6` the caller reuses.
#
# Fortran's per-row scratch arrays become GLOBAL device arrays; each of the
# routine's sequential passes is one per-cell kernel launch (the launch boundary
# is the barrier between passes). All passes are precision-generic on `T`.
#
# Index conventions (column-major idim×jdim, 1-based): geometry coeffs c1..c6 are
# 1-D (function of i only); char1/char2/c0 are 2-D (per row j). q and all temps
# are 2-D. Active region i1..i2, ghosts ≥4 each side (the reconstruction reaches
# i±3 through the dq stencil at i1-2 / i2+2).

const _FT(::Type{T}) where {T} = T(4) / T(3)        # ft = 4/3 (CW84 eq. 1.124)

# pass 1 — monotonized (van-Leer-limited) slope dq, over i1-2 .. i2+2
@kernel function _iv_dq!(dq, @Const(q), @Const(c1), @Const(c2),
                         idim::Int, istart::Int, j1::Int)
    gi, gj = @index(Global, NTuple)
    i = istart + gi - 1
    j = j1 + gj - 1
    idx = (j - 1) * idim + i
    T = eltype(dq)
    @inbounds begin
        qplus = q[idx + 1] - q[idx]
        qmnus = q[idx] - q[idx - 1]
        if qplus * qmnus > zero(T)
            qcent = c1[i] * qplus + c2[i] * qmnus
            qvanl = T(2) * qplus * qmnus / (qmnus + qplus)
            t1 = min(abs(qcent), abs(qvanl), T(2) * abs(qmnus), T(2) * abs(qplus))
            dq[idx] = qcent >= zero(T) ? t1 : -t1
        else
            dq[idx] = zero(T)
        end
    end
end

# pass 2 — interface values ql,qr (eq. 1.6); qr[i] ≡ ql[i+1] computed directly.
# ql over i1-1..i2+2, qr over i1-2..i2+1 — guarded inside one launch.
@kernel function _iv_lr!(ql, qr, @Const(q), @Const(dq),
                         @Const(c3), @Const(c4), @Const(c5), @Const(c6),
                         idim::Int, istart::Int, i1::Int, i2::Int, j1::Int)
    gi, gj = @index(Global, NTuple)
    i = istart + gi - 1
    j = j1 + gj - 1
    base = (j - 1) * idim
    idx = base + i
    @inbounds begin
        if i1 - 1 <= i <= i2 + 2
            ql[idx] = c3[i] * q[idx - 1] + c4[i] * q[idx] + c5[i] * dq[idx - 1] + c6[i] * dq[idx]
        end
        if i1 - 2 <= i <= i2 + 1                  # qr[i] = ql[i+1]
            ip = i + 1
            qr[idx] = c3[ip] * q[idx] + c4[ip] * q[idx + 1] + c5[ip] * dq[idx] + c6[ip] * dq[idx + 1]
        end
    end
end

# pass 3 — optional steepening (density only), in place on ql,qr, over i1-1..i2+1
@kernel function _iv_steepen!(ql, qr, @Const(q), @Const(dq), @Const(steepen),
                              idim::Int, istart::Int, j1::Int)
    gi, gj = @index(Global, NTuple)
    i = istart + gi - 1
    j = j1 + gj - 1
    idx = (j - 1) * idim + i
    T = eltype(ql)
    @inbounds begin
        st = steepen[idx]
        ql[idx] = (one(T) - st) * ql[idx] + st * (q[idx - 1] + T(0.5) * dq[idx - 1])
        qr[idx] = (one(T) - st) * qr[idx] + st * (q[idx + 1] - T(0.5) * dq[idx + 1])
    end
end

# pass 4 — re-monotonize the parabola (eq. 1.10), in place, over i1-1..i2+1
@kernel function _iv_monotonize!(ql, qr, @Const(q), idim::Int, istart::Int, j1::Int)
    gi, gj = @index(Global, NTuple)
    i = istart + gi - 1
    j = j1 + gj - 1
    idx = (j - 1) * idim + i
    T = eltype(ql)
    @inbounds begin
        qi = q[idx]; qli = ql[idx]; qri = qr[idx]
        t1 = (qri - qi) * (qi - qli)
        t2 = qri - qli
        t3 = T(6) * (qi - T(0.5) * (qri + qli))
        if t1 <= zero(T)
            qli = qi; qri = qi
        end
        t22 = t2 * t2
        t23 = t2 * t3
        if t22 < t23
            qli = T(3) * qi - T(2) * qri
        end
        if t22 < -t23
            qri = T(3) * qi - T(2) * qli
        end
        ql[idx] = qli; qr[idx] = qri
    end
end

# pass 5 — optional flattening, in place, over i1-1..i2+1
@kernel function _iv_flatten!(ql, qr, @Const(q), @Const(flatten),
                              idim::Int, istart::Int, j1::Int)
    gi, gj = @index(Global, NTuple)
    i = istart + gi - 1
    j = j1 + gj - 1
    idx = (j - 1) * idim + i
    T = eltype(ql)
    @inbounds begin
        fl = flatten[idx]
        ql[idx] = q[idx] * fl + ql[idx] * (one(T) - fl)
        qr[idx] = q[idx] * fl + qr[idx] * (one(T) - fl)
    end
end

# pass 6 — clamp L/R between neighbouring cell centres (ATHENA), over i1-1..i2+2
@kernel function _iv_checklr!(ql, qr, @Const(q), idim::Int, istart::Int, j1::Int)
    gi, gj = @index(Global, NTuple)
    i = istart + gi - 1
    j = j1 + gj - 1
    idx = (j - 1) * idim + i
    @inbounds begin
        qm = q[idx - 1]; qi = q[idx]; qp = q[idx + 1]
        ql[idx] = min(max(qi, qm), max(min(qi, qm), ql[idx]))
        qr[idx] = min(max(qi, qp), max(min(qi, qp), qr[idx]))
    end
end

# pass 7 — parabola coefficients q6 and dq=qr-ql (eq. 1.12), over i1-1..i2+1
@kernel function _iv_q6dq!(q6, dq, @Const(ql), @Const(qr), @Const(q),
                           idim::Int, istart::Int, j1::Int)
    gi, gj = @index(Global, NTuple)
    i = istart + gi - 1
    j = j1 + gj - 1
    idx = (j - 1) * idim + i
    T = eltype(q6)
    @inbounds begin
        q6[idx] = T(6) * (q[idx] - T(0.5) * (ql[idx] + qr[idx]))
        dq[idx] = qr[idx] - ql[idx]
    end
end

# pass 8 — characteristic-averaged states qla/qra (char1/char2) and ql0/qr0 (c0),
# over i1 .. i2+1
@kernel function _iv_states!(qla, qra, ql0, qr0, @Const(ql), @Const(qr),
                             @Const(dq), @Const(q6), @Const(char1), @Const(char2),
                             @Const(c0), idim::Int, istart::Int, j1::Int)
    gi, gj = @index(Global, NTuple)
    i = istart + gi - 1
    j = j1 + gj - 1
    base = (j - 1) * idim
    idx = base + i
    il = idx - 1
    T = eltype(qla)
    ft = _FT(T)
    @inbounds begin
        h1 = char1[il]; h2 = char2[idx]
        qla[idx] = qr[il] - h1 * (dq[il] - (one(T) - ft * h1) * q6[il])
        qra[idx] = ql[idx] + h2 * (dq[idx] + (one(T) - ft * h2) * q6[idx])
        c0l = c0[il]; c0i = c0[idx]
        ql0[idx] = qr[il] - c0l * (dq[il] - (one(T) - ft * c0l) * q6[il])
        qr0[idx] = ql[idx] - c0i * (dq[idx] + (one(T) + ft * c0i) * q6[idx])
    end
end

"""
    intvar!(out, tmp, q, geom, steepen, flatten; idim, i1, i2, j1, j2, isteep, iflatten)

PPM-reconstruct one field `q`. `out = (qla, qra, ql0, qr0)` and the reusable
intermediates `tmp = (dq, ql, qr, q6)` are caller-provided device arrays (so
`inteuler` can keep p's and u's parabola for the β-correction). `geom` is the
NamedTuple of grid/characteristic coefficients from [`inteuler_geometry`](@ref).
Mirrors `intvar.F` pass-for-pass. No `synchronize` — the caller batches it.
"""
function intvar!(out, tmp, q, geom, steepen, flatten;
                 idim::Integer, i1::Integer, i2::Integer, j1::Integer, j2::Integer,
                 isteep::Integer, iflatten::Integer)
    be = KA.get_backend(q)
    nj = j2 - j1 + 1
    idim, i1, i2, j1 = Int(idim), Int(i1), Int(i2), Int(j1)
    qla, qra, ql0, qr0 = out
    dq, ql, qr, q6 = tmp
    (; c1, c2, c3, c4, c5, c6, char1, char2, c0) = geom

    _iv_dq!(be)(dq, q, c1, c2, idim, i1 - 2, j1; ndrange = (i2 - i1 + 5, nj))
    _iv_lr!(be)(ql, qr, q, dq, c3, c4, c5, c6, idim, i1 - 2, i1, i2, j1;
                ndrange = (i2 - i1 + 5, nj))
    if isteep != 0
        _iv_steepen!(be)(ql, qr, q, dq, steepen, idim, i1 - 1, j1; ndrange = (i2 - i1 + 3, nj))
    end
    _iv_monotonize!(be)(ql, qr, q, idim, i1 - 1, j1; ndrange = (i2 - i1 + 3, nj))
    if iflatten != 0
        _iv_flatten!(be)(ql, qr, q, flatten, idim, i1 - 1, j1; ndrange = (i2 - i1 + 3, nj))
    end
    _iv_checklr!(be)(ql, qr, q, idim, i1 - 1, j1; ndrange = (i2 - i1 + 4, nj))
    _iv_q6dq!(be)(q6, dq, ql, qr, q, idim, i1 - 1, j1; ndrange = (i2 - i1 + 3, nj))
    _iv_states!(be)(qla, qra, ql0, qr0, ql, qr, dq, q6, char1, char2, c0,
                    idim, i1, j1; ndrange = (i2 - i1 + 2, nj))
    return out
end
