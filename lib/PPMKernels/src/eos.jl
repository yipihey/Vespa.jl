# ── Phase 2.1 — EOS: pgas2d / pgas2d_dual ────────────────────────────────────
# Port of Enzo's `pgas2d.F` and `pgas2d_dual.F`: gas pressure on a column-major
# `idim×jdim` slab over the active region `i1..i2 × j1..j2` (1-based inclusive).
# Both are precision-generic `@kernel`s parameterised on the array element type.
#
#   pgas2d       p = (γ-1)·d·(E − ½(u²+v²+w²)), floored at pmin. Purely LOCAL —
#                every cell is independent ⇒ a flat per-cell launch.
#
#   pgas2d_dual  the dual-energy reconciliation. It carries a genuine left-to-
#                right SWEEP dependency: a cell's `demax` reads the *already-
#                updated* total energy of its left neighbour (Fortran mutates
#                `eslice` in the loop). To stay bit-faithful on any device we
#                parallelise over rows (`j`) and run the `i`-sweep sequentially
#                inside each work-item — one thread owns a whole row, so the
#                dependency is preserved on CPU and Metal alike.

export pgas2d!, pgas2d_dual!

# ── pgas2d (local, per-cell) ─────────────────────────────────────────────────
@kernel function _pgas2d_kernel!(pslice, @Const(dslice), @Const(eslice),
                                 @Const(uslice), @Const(vslice), @Const(wslice),
                                 idim::Int, i1::Int, j1::Int, gamma, pmin)
    gi, gj = @index(Global, NTuple)        # 1..ni, 1..nj over the active region
    i = i1 + gi - 1
    j = j1 + gj - 1
    idx = (j - 1) * idim + i
    T = eltype(pslice)
    @inbounds begin
        ke = (uslice[idx] * uslice[idx] + vslice[idx] * vslice[idx] +
              wslice[idx] * wslice[idx]) * T(0.5)
        p = (gamma - one(T)) * dslice[idx] * (eslice[idx] - ke)
        pslice[idx] = ifelse(p < pmin, pmin, p)
    end
end

"""
    pgas2d!(pslice, dslice, eslice, uslice, vslice, wslice;
            idim, i1, i2, j1=1, j2=1, gamma, pmin=1e-20) -> pslice

Fill `pslice` with the ideal-gas pressure `(γ-1)·d·(E − ½‖v‖²)` (floored at
`pmin`) over the active region. All arrays live on the same backend (CPU or
Metal); element type `T = eltype(pslice)` sets the working precision — `gamma`
and `pmin` are converted to `T`. Returns `pslice`.
"""
function pgas2d!(pslice, dslice, eslice, uslice, vslice, wslice;
                 idim::Integer, i1::Integer, i2::Integer,
                 j1::Integer = 1, j2::Integer = 1, gamma::Real, pmin::Real = 1e-20)
    be = KA.get_backend(pslice)
    T = eltype(pslice)
    ni = i2 - i1 + 1
    nj = j2 - j1 + 1
    _pgas2d_kernel!(be)(pslice, dslice, eslice, uslice, vslice, wslice,
                        Int(idim), Int(i1), Int(j1), T(gamma), T(pmin);
                        ndrange = (ni, nj))
    return pslice
end

# ── pgas2d_dual (row-parallel, i-sequential) ─────────────────────────────────
@kernel function _pgas2d_dual_kernel!(eslice, geslice, pslice,
                                      @Const(dslice), @Const(uslice),
                                      @Const(vslice), @Const(wslice),
                                      idim::Int, i1::Int, i2::Int, j1::Int,
                                      eta1, eta2, gamma, pmin)
    gj = @index(Global, Linear)            # 1..nj — one work-item per active row
    j = j1 + gj - 1
    base = (j - 1) * idim
    T = eltype(eslice)
    gm1 = gamma - one(T)
    for i in i1:i2
        idx = base + i
        @inbounds begin
            ke = (uslice[idx] * uslice[idx] + vslice[idx] * vslice[idx] +
                  wslice[idx] * wslice[idx]) * T(0.5)
            ge1 = eslice[idx] - ke
            # nearest-neighbour max of d·E, clamped to the active span; reads the
            # left neighbour's eslice AFTER this same loop already updated it.
            im1 = max(i - 1, i1)
            ip1 = min(i + 1, i2)
            demax = max(dslice[idx] * eslice[idx],
                        dslice[base + im1] * eslice[base + im1],
                        dslice[base + ip1] * eslice[base + ip1])
            if ge1 * dslice[idx] / demax > eta2
                geslice[idx] = ge1          # total-energy gas energy is trustworthy
            end
            ge2 = (ge1 / eslice[idx] > eta1) ? ge1 : geslice[idx]
            ge2 = max(ge2, pmin / (gm1 * dslice[idx]))
            eslice[idx] = eslice[idx] - ge1 + ge2
            pslice[idx] = gm1 * dslice[idx] * ge2
        end
    end
end

"""
    pgas2d_dual!(eslice, geslice, pslice, dslice, uslice, vslice, wslice;
                 idim, i1, i2, j1=1, j2=1, eta1, eta2, gamma, pmin=1e-20)
        -> (eslice, geslice, pslice)

Dual-energy pressure. Reconciles the gas energy `geslice` against the total
energy `eslice` (selection parameters `eta1`/`eta2`), updating BOTH in place,
then forms `pslice`. Faithful to the Fortran left-to-right sweep: rows run in
parallel, the `i`-direction sequentially within each row. Returns the three
updated slices.
"""
function pgas2d_dual!(eslice, geslice, pslice, dslice, uslice, vslice, wslice;
                      idim::Integer, i1::Integer, i2::Integer,
                      j1::Integer = 1, j2::Integer = 1,
                      eta1::Real, eta2::Real, gamma::Real, pmin::Real = 1e-20)
    be = KA.get_backend(eslice)
    T = eltype(eslice)
    nj = j2 - j1 + 1
    _pgas2d_dual_kernel!(be)(eslice, geslice, pslice, dslice, uslice, vslice, wslice,
                             Int(idim), Int(i1), Int(i2), Int(j1),
                             T(eta1), T(eta2), T(gamma), T(pmin); ndrange = nj)
    return eslice, geslice, pslice
end
