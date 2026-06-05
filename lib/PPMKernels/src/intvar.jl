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

# ── FUSED reconstruction ──────────────────────────────────────────────────────
# The 8 passes above are all per-cell-LOCAL stencils chained through the full-grid
# temporaries dq/ql/qr/q6 — on the bandwidth-bound GPU that round-tripping IS the
# PPM hot spot (inteuler = 75% of a sweep, profiled). The fused path inlines the
# whole chain so a thread computes a cell's final parabola in registers, reading
# only `q` (+ coeffs) and writing only the 4 states (+ the parabola for p/u, which
# the β-correction needs). Cost: each thread recomputes its i−1 neighbour's parabola
# (~2× the reconstruction flops) to avoid the cross-pass memory traffic. Same
# formulas as the passes — certified bit-tight vs the Fortran reference (inteuler).

# pass-1 van-Leer-limited slope at one cell (was `_iv_dq!`).
@inline function _iv_slope(qm::T, q0::T, qp::T, c1i::T, c2i::T) where {T}
    qplus = qp - q0; qmnus = q0 - qm
    if qplus * qmnus > zero(T)
        qcent = c1i * qplus + c2i * qmnus
        qvanl = T(2) * qplus * qmnus / (qmnus + qplus)
        t1 = min(abs(qcent), abs(qvanl), T(2) * abs(qmnus), T(2) * abs(qplus))
        return qcent >= zero(T) ? t1 : -t1
    else
        return zero(T)
    end
end

# full single-cell reconstruction (passes 1,2,3,4,5,6,7 inlined): from field `q`
# at cell `ci` (flat index `idx`) → the final monotonized parabola
# (ql, qr, q6, dq=qr−ql). Reads q[idx−2 … idx+2] and coeffs at ci−1 … ci+1.
@inline function _iv_recon_cell(q, c1, c2, c3, c4, c5, c6, steepen, flatten,
                                idx::Int, ci::Int, isteep::Int, iflatten::Int)
    T = eltype(q)
    s_m = _iv_slope(q[idx-2], q[idx-1], q[idx],   c1[ci-1], c2[ci-1])    # slope at ci−1
    s_0 = _iv_slope(q[idx-1], q[idx],   q[idx+1], c1[ci],   c2[ci])      # slope at ci
    s_p = _iv_slope(q[idx],   q[idx+1], q[idx+2], c1[ci+1], c2[ci+1])    # slope at ci+1
    ql = c3[ci] * q[idx-1]     + c4[ci] * q[idx]       + c5[ci] * s_m     + c6[ci] * s_0
    qr = c3[ci+1] * q[idx]     + c4[ci+1] * q[idx+1]   + c5[ci+1] * s_0   + c6[ci+1] * s_p
    if isteep != 0                                   # density steepening (pass 3)
        st = steepen[idx]
        ql = (one(T) - st) * ql + st * (q[idx-1] + T(0.5) * s_m)
        qr = (one(T) - st) * qr + st * (q[idx+1] - T(0.5) * s_p)
    end
    qi = q[idx]                                      # monotonize (pass 4) — order matters
    t1 = (qr - qi) * (qi - ql); t2 = qr - ql; t3 = T(6) * (qi - T(0.5) * (qr + ql))
    if t1 <= zero(T); ql = qi; qr = qi; end
    t22 = t2 * t2; t23 = t2 * t3
    if t22 < t23;  ql = T(3) * qi - T(2) * qr; end
    if t22 < -t23; qr = T(3) * qi - T(2) * ql; end
    if iflatten != 0                                 # flatten (pass 5)
        fl = flatten[idx]
        ql = qi * fl + ql * (one(T) - fl); qr = qi * fl + qr * (one(T) - fl)
    end
    qm = q[idx-1]; qp = q[idx+1]                      # checklr clamp (pass 6)
    ql = min(max(qi, qm), max(min(qi, qm), ql))
    qr = min(max(qi, qp), max(min(qi, qp), qr))
    return (ql, qr, T(6) * (qi - T(0.5) * (ql + qr)), qr - ql)   # ql, qr, q6, dq
end

# pass-8 characteristic averaging from a cell's and its left neighbour's parabola.
@inline function _iv_states4(qrL::T, q6L::T, dqL::T, qlC::T, q6C::T, dqC::T,
                             h1::T, h2::T, c0l::T, c0i::T) where {T}
    ft = _FT(T)
    qla = qrL - h1 * (dqL - (one(T) - ft * h1) * q6L)
    qra = qlC + h2 * (dqC + (one(T) - ft * h2) * q6C)
    ql0 = qrL - c0l * (dqL - (one(T) - ft * c0l) * q6L)
    qr0 = qlC - c0i * (dqC + (one(T) + ft * c0i) * q6C)
    return (qla, qra, ql0, qr0)
end

# fused reconstruction → 4 states only (d, v, w, ge), over i1 .. i2+1.
@kernel function _iv_fused!(qla, qra, ql0, qr0, @Const(q),
                            @Const(c1), @Const(c2), @Const(c3), @Const(c4), @Const(c5), @Const(c6),
                            @Const(char1), @Const(char2), @Const(c0), @Const(steepen), @Const(flatten),
                            idim::Int, istart::Int, j1::Int, isteep::Int, iflatten::Int)
    gi, gj = @index(Global, NTuple)
    i = istart + gi - 1; j = j1 + gj - 1
    idx = (j - 1) * idim + i; il = idx - 1
    @inbounds begin
        (qlL, qrL, q6L, dqL) = _iv_recon_cell(q, c1, c2, c3, c4, c5, c6, steepen, flatten, il, i - 1, isteep, iflatten)
        (qlC, qrC, q6C, dqC) = _iv_recon_cell(q, c1, c2, c3, c4, c5, c6, steepen, flatten, idx, i, isteep, iflatten)
        (a, b, c, d) = _iv_states4(qrL, q6L, dqL, qlC, q6C, dqC, char1[il], char2[idx], c0[il], c0[idx])
        qla[idx] = a; qra[idx] = b; ql0[idx] = c; qr0[idx] = d
    end
end

# fused reconstruction → 4 states AND the parabola (dq,ql,qr,q6) for p & u, whose
# β-correction reads the parabola at i and i−1; so the parabola is emitted over
# i1−1 .. i2+1 (istart=i1−1) and the states only for i ≥ i1.
@kernel function _iv_fused_p!(qla, qra, ql0, qr0, dqo, qlo, qro, q6o, @Const(q),
                              @Const(c1), @Const(c2), @Const(c3), @Const(c4), @Const(c5), @Const(c6),
                              @Const(char1), @Const(char2), @Const(c0), @Const(steepen), @Const(flatten),
                              idim::Int, istart::Int, i1::Int, j1::Int, isteep::Int, iflatten::Int)
    gi, gj = @index(Global, NTuple)
    i = istart + gi - 1; j = j1 + gj - 1
    idx = (j - 1) * idim + i
    @inbounds begin
        (qlC, qrC, q6C, dqC) = _iv_recon_cell(q, c1, c2, c3, c4, c5, c6, steepen, flatten, idx, i, isteep, iflatten)
        dqo[idx] = dqC; qlo[idx] = qlC; qro[idx] = qrC; q6o[idx] = q6C
        if i >= i1
            il = idx - 1
            (qlL, qrL, q6L, dqL) = _iv_recon_cell(q, c1, c2, c3, c4, c5, c6, steepen, flatten, il, i - 1, isteep, iflatten)
            (a, b, c, d) = _iv_states4(qrL, q6L, dqL, qlC, q6C, dqC, char1[il], char2[idx], c0[il], c0[idx])
            qla[idx] = a; qra[idx] = b; ql0[idx] = c; qr0[idx] = d
        end
    end
end

"""
    intvar!(out, tmp, q, geom, steepen, flatten; idim, i1, i2, j1, j2, isteep, iflatten)

PPM-reconstruct one field `q` into `out = (qla, qra, ql0, qr0)`. When
`tmp = (dq, ql, qr, q6)` the monotonized parabola is also emitted (the β-correction
in `inteuler` needs p's and u's parabola); `tmp === nothing` writes only the states.

Backend-dispatched: the **GPU** uses the FUSED single-kernel path (no full-grid
dq/ql/qr/q6 temporaries — the win on the bandwidth-bound device, at the cost of
recomputing each cell's left-neighbour parabola). The **CPU** keeps the 8-pass
path (no recompute — flop-efficient, and it is the parity oracle + the bit-tight
reference). Both are certified against `intvar.F` through the inteuler reference.
No `synchronize` — the caller batches it.
"""
function intvar!(out, tmp, q, geom, steepen, flatten;
                 idim::Integer, i1::Integer, i2::Integer, j1::Integer, j2::Integer,
                 isteep::Integer, iflatten::Integer)
    be = KA.get_backend(q)
    nj = j2 - j1 + 1
    idim, i1, i2, j1 = Int(idim), Int(i1), Int(i2), Int(j1)
    qla, qra, ql0, qr0 = out
    (; c1, c2, c3, c4, c5, c6, char1, char2, c0) = geom
    ist, ifl = Int(isteep), Int(iflatten)

    if be isa CPU
        # flop-efficient 8-pass (no recompute); the CPU is the parity oracle.
        dq, ql, qr, q6 = tmp === nothing ?
            (_zlike(q), _zlike(q), _zlike(q), _zlike(q)) : tmp
        _iv_dq!(be)(dq, q, c1, c2, idim, i1 - 2, j1; ndrange = (i2 - i1 + 5, nj))
        _iv_lr!(be)(ql, qr, q, dq, c3, c4, c5, c6, idim, i1 - 2, i1, i2, j1;
                    ndrange = (i2 - i1 + 5, nj))
        ist != 0 && _iv_steepen!(be)(ql, qr, q, dq, steepen, idim, i1 - 1, j1; ndrange = (i2 - i1 + 3, nj))
        _iv_monotonize!(be)(ql, qr, q, idim, i1 - 1, j1; ndrange = (i2 - i1 + 3, nj))
        ifl != 0 && _iv_flatten!(be)(ql, qr, q, flatten, idim, i1 - 1, j1; ndrange = (i2 - i1 + 3, nj))
        _iv_checklr!(be)(ql, qr, q, idim, i1 - 1, j1; ndrange = (i2 - i1 + 4, nj))
        _iv_q6dq!(be)(q6, dq, ql, qr, q, idim, i1 - 1, j1; ndrange = (i2 - i1 + 3, nj))
        _iv_states!(be)(qla, qra, ql0, qr0, ql, qr, dq, q6, char1, char2, c0,
                        idim, i1, j1; ndrange = (i2 - i1 + 2, nj))
    elseif tmp === nothing
        _iv_fused!(be)(qla, qra, ql0, qr0, q, c1, c2, c3, c4, c5, c6,
                       char1, char2, c0, steepen, flatten, idim, i1, j1, ist, ifl;
                       ndrange = (i2 - i1 + 2, nj))
    else
        dqo, qlo, qro, q6o = tmp
        _iv_fused_p!(be)(qla, qra, ql0, qr0, dqo, qlo, qro, q6o, q, c1, c2, c3, c4, c5, c6,
                         char1, char2, c0, steepen, flatten, idim, i1 - 1, i1, j1, ist, ifl;
                         ndrange = (i2 - i1 + 3, nj))
    end
    return out
end
