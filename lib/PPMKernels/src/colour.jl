# ── Passive colour (species) advection — rides the PPM mass flux ─────────────
# Enzo's PPM advects "colour" fields (here the chemistry species mass densities
# HII, H2I, HDI) with the SAME interface mass flux `df` the hydro update uses, so
# the species are carried conservatively with the gas and a uniform mass fraction
# stays exactly uniform.  This runs INSIDE the 1-D sweep, AFTER the fluxes are
# formed but BEFORE `euler!` overwrites the density — so the upwind mass fraction
# `c/d` is taken from the OLD (pre-update) density, exactly consistent with
# `dnu = dold + (df[i]-df[i+1])`.
#
# Interface convention matches euler!: `df[i]` is the flux through the LEFT face
# of cell i (between cells i-1 and i); `df[i] > 0` ⇒ mass flows i-1 → i, so the
# upwind cell is i-1.  Per colour:
#   colf[i] = df[i] · (c/d)_upwind        (mass-fraction-weighted mass flux)
#   c[i]   += colf[i] - colf[i+1]
# Two passes (flux then update) via a scratch `colf` avoid the in-place race a
# single fused kernel would have (cell i+1 reading cell i's just-written colour).

@inline _COL_tiny(::Type{T}) where {T} = T(1e-20)

@kernel function _colflux_kernel!(colf, @Const(cslice), @Const(dslice), @Const(df),
                                  idim::Int, istart::Int, j1::Int)
    gi, gj = @index(Global, NTuple)
    i = istart + gi - 1
    j = j1 + gj - 1
    idx = (j - 1) * idim + i
    im = idx - 1
    T = eltype(colf)
    @inbounds begin
        f = df[idx]
        frac = f >= zero(T) ? cslice[im] / max(dslice[im], _COL_tiny(T)) :
                              cslice[idx] / max(dslice[idx], _COL_tiny(T))
        colf[idx] = f * frac
    end
end

@kernel function _colupd_kernel!(cslice, @Const(colf), idim::Int, istart::Int, j1::Int)
    gi, gj = @index(Global, NTuple)
    i = istart + gi - 1
    j = j1 + gj - 1
    idx = (j - 1) * idim + i
    ip = idx + 1
    @inbounds cslice[idx] = cslice[idx] + (colf[idx] - colf[ip])
end

"""
    advect_colours!(colours, dslice, df; idim, i1, i2, j1=1, j2=1)

Advect each colour slice in the tuple `colours` with the mass flux `df` and the
OLD density `dslice` (call BEFORE `euler!` updates the density).  Conservative:
`Σ c·dV` is preserved to round-off and a uniform mass fraction is invariant.
"""
function advect_colours!(colours, dslice, df; idim::Integer, i1::Integer, i2::Integer,
                         j1::Integer = 1, j2::Integer = 1)
    isempty(colours) && return colours
    be = KA.get_backend(dslice)
    idim, i1, i2, j1, j2 = Int(idim), Int(i1), Int(i2), Int(j1), Int(j2)
    nj = j2 - j1 + 1
    for c in colours
        colf = _zlike(dslice)
        # fluxes over interfaces i1 .. i2+1 (one more than the cells), then update i1..i2
        _colflux_kernel!(be)(colf, c, dslice, df, idim, i1, j1; ndrange = (i2 - i1 + 2, nj))
        _colupd_kernel!(be)(c, colf, idim, i1, j1; ndrange = (i2 - i1 + 1, nj))
    end
    return colours
end
