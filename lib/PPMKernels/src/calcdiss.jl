# ── Phase 2.2 — calcdiss: PPM diffusion coefficient + slope flattening ────────
# Port of Enzo's `calcdiss.F` (Colella & Woodward 1984 §A) — the optional
# diffusion/flattening pass that feeds `inteuler` (flatten) and `flux_twoshock`
# (diffcoef). It is OFF by default in Enzo (`PPMDiffusionParameter =
# PPMFlatteningParameter = 0`).
#
# SCOPE: the transverse-free 1-D slice regime (`dimy = dimz = 1`). The multi-
# dimensional transverse-velocity diffusion terms (vdiff/wdiff, requiring the
# full 3-D v,w fields and the directional permutation) are deferred to the 3-D
# assembly phase (Phase 4). In the 1-D regime they vanish identically, so this
# port is bit-faithful to the Fortran run with `dimy=dimz=1`. Supported:
#   idiff    ∈ {0, 1}      (0: none; 1: K·max(0,−div u) — u-divergence only in 1-D)
#   iflatten ∈ {0, 1, 3}   (0: none; 1: CW84 A1–A2; 3: CW84 A7–A10, multidim flattener)
# (`iflatten=2` is the untested Lagrangean variant; `idiff=2` is the alternate
# scheme — both out of scope.)
#
# Structure: Fortran's per-row temp buffers (`wflag`, `flattemp`) become GLOBAL
# scratch arrays filled by per-cell kernels — the GPU-faithful expression of the
# two-pass stencil (a per-thread scratch of length idim is infeasible on Metal).
# The `flattemp(i1-2)=flattemp(i1-1)` boundary copies are emulated by clamping
# the neighbour index in the final pass.

export calcdiss!

# CW84 fixed parameters (calcdiss.F `parameter` block); `tiny` = fortran_types.def.
@inline _cd_eps(::Type{T})    where {T} = T(0.33)
@inline _cd_omega1(::Type{T}) where {T} = T(0.75)
@inline _cd_omega2(::Type{T}) where {T} = T(10)
@inline _cd_sigma1(::Type{T}) where {T} = T(0.5)
@inline _cd_sigma2(::Type{T}) where {T} = T(1)
@inline _cd_kappa1(::Type{T}) where {T} = T(2)
@inline _cd_kappa2(::Type{T}) where {T} = T(0.01)
@inline _cd_K(::Type{T})      where {T} = T(0.1)
@inline _cd_tiny(::Type{T})   where {T} = T(1e-20)

# ── wflag (CW84 A1): shock indicator, over i1-2 .. i2+2 ──────────────────────
@kernel function _calcdiss_wflag!(wflag, @Const(p), @Const(u),
                                  idim::Int, istart::Int, j1::Int)
    gi, gj = @index(Global, NTuple)
    i = istart + gi - 1
    j = j1 + gj - 1
    idx = (j - 1) * idim + i
    T = eltype(wflag)
    @inbounds begin
        qb = abs(p[idx + 1] - p[idx - 1]) / min(p[idx + 1], p[idx - 1])
        wflag[idx] = ifelse(qb > _cd_eps(T) && u[idx - 1] > u[idx + 1], one(T), zero(T))
    end
end

# ── diffcoef, idiff=1, transverse-free (over i1 .. i2+1) ─────────────────────
@kernel function _calcdiss_diffcoef1!(diffcoef, @Const(u), idim::Int, istart::Int, j1::Int)
    gi, gj = @index(Global, NTuple)
    i = istart + gi - 1
    j = j1 + gj - 1
    idx = (j - 1) * idim + i
    T = eltype(diffcoef)
    @inbounds diffcoef[idx] = _cd_K(T) * max(zero(T), u[idx - 1] - u[idx])
end

# ── flattemp, iflatten=1 (CW84 A1–A2), over i1-1 .. i2+1 ─────────────────────
@kernel function _calcdiss_flat1_temp!(flattemp, @Const(p), @Const(wflag),
                                       idim::Int, istart::Int, j1::Int)
    gi, gj = @index(Global, NTuple)
    i = istart + gi - 1
    j = j1 + gj - 1
    idx = (j - 1) * idim + i
    T = eltype(flattemp)
    @inbounds begin
        pp2 = p[idx + 2]; pm2 = p[idx - 2]
        denom = pp2 - pm2
        qa = (abs(denom) / min(pp2, pm2) < _cd_eps(T)) ? one(T) :
             (p[idx + 1] - p[idx - 1]) / denom
        ft = min(one(T), (qa - _cd_omega1(T)) * _cd_omega2(T) * wflag[idx])
        flattemp[idx] = max(zero(T), ft)
    end
end

# ── flattemp, iflatten=3 (CW84 A7–A10), over i1-1 .. i2+1 ────────────────────
@kernel function _calcdiss_flat3_temp!(flattemp, @Const(p), @Const(e), @Const(d),
                                       @Const(u), @Const(wflag),
                                       idim::Int, istart::Int, j1::Int, gamma)
    gi, gj = @index(Global, NTuple)
    i = istart + gi - 1
    j = j1 + gj - 1
    base = (j - 1) * idim
    idx = base + i
    T = eltype(flattemp)
    @inbounds begin
        dp1 = p[idx + 1] - p[idx - 1]
        dp2 = p[idx + 2] - p[idx - 2]
        de1 = e[idx + 1] - e[idx - 1]
        de2 = e[idx + 2] - e[idx - 2]
        dpp = dp2 != zero(T) ? dp1 / dp2 : zero(T)
        dee = de2 != zero(T) ? de1 / de2 : zero(T)
        omega_tilde = max(dpp, dee)
        # Fortran SIGN(2,dp1) treats dp1==0 as +; s = -SIGN(1,dp1), s=0 if dp1==0.
        pos = dp1 >= zero(T)
        ism = base + i + (pos ? 2 : -2)          # post-shock zone
        isp = base + i - (pos ? 2 : -2)          # upstream zone
        s   = dp1 > zero(T) ? -one(T) : (dp1 < zero(T) ? one(T) : zero(T))

        pp2 = p[idx + 2]; pm2 = p[idx - 2]
        sigma_tilde = wflag[idx] * abs(dp2) / min(pp2, pm2)
        sigma = max(zero(T), (sigma_tilde - _cd_sigma1(T)) / (sigma_tilde + _cd_sigma2(T)))
        omega = max(zero(T), _cd_omega2(T) * (omega_tilde - _cd_omega1(T)))

        # Lagrangean shock-speed estimate (max(di) = 1/min(d)).
        Z = sqrt((max(pp2, pm2) + (pp2 + pm2) * T(0.5) * (gamma - one(T))) /
                 (one(T) / min(d[idx + 2], d[idx - 2])))
        ZE   = s * Z / d[ism] + u[ism] + _cd_tiny(T)
        cj2s = sqrt(gamma * p[isp] / d[isp])
        kappa_tilde = abs((ZE - u[isp] + s * cj2s) / ZE)
        kappa = max(zero(T), (kappa_tilde - _cd_kappa1(T)) / (kappa_tilde + _cd_kappa2(T)))

        flattemp[idx] = min(kappa, wflag[idx] * omega, wflag[idx] * sigma)
    end
end

# ── final flatten (shared A2/A10 neighbour-max), over i1-1 .. i2+1 ───────────
# Emulates the `flattemp(i1-2)=flattemp(i1-1)` / `(i2+2)=(i2+1)` boundary copies
# by clamping the neighbour index into the computed span [istart, iend].
@kernel function _calcdiss_flat_final!(flatten, @Const(flattemp), @Const(p),
                                       idim::Int, istart::Int, iend::Int, j1::Int)
    gi, gj = @index(Global, NTuple)
    i = istart + gi - 1
    j = j1 + gj - 1
    base = (j - 1) * idim
    idx = base + i
    T = eltype(flatten)
    @inbounds begin
        ip = base + min(i + 1, iend)
        im = base + max(i - 1, istart)
        flatten[idx] = (p[idx + 1] - p[idx - 1] < zero(T)) ?
                       max(flattemp[idx], flattemp[ip]) :
                       max(flattemp[idx], flattemp[im])
    end
end

"""
    calcdiss!(diffcoef, flatten, dslice, eslice, uslice, pslice;
              idim, i1, i2, j1=1, j2=1, gamma, idiff, iflatten) -> (diffcoef, flatten)

Fill `diffcoef` (when `idiff=1`) and `flatten` (when `iflatten∈{1,3}`) over the
active slab; both must be pre-zeroed (cells outside the written ranges stay 0,
matching Fortran). Transverse-free 1-D regime — see file header for scope.
Element type `T = eltype(diffcoef)` sets the working precision.
"""
function calcdiss!(diffcoef, flatten, dslice, eslice, uslice, pslice;
                   idim::Integer, i1::Integer, i2::Integer,
                   j1::Integer = 1, j2::Integer = 1, gamma::Real,
                   idiff::Integer, iflatten::Integer)
    be = KA.get_backend(diffcoef)
    T  = eltype(diffcoef)
    nj = j2 - j1 + 1
    idim, i1, i2, j1 = Int(idim), Int(i1), Int(i2), Int(j1)
    g = T(gamma)

    if idiff == 1
        _calcdiss_diffcoef1!(be)(diffcoef, uslice, idim, i1, j1;
                                 ndrange = (i2 - i1 + 2, nj))
    elseif idiff != 0
        throw(ArgumentError("calcdiss!: only idiff ∈ {0,1} supported (got $idiff)"))
    end

    if iflatten == 1 || iflatten == 3
        wflag    = similar(diffcoef); fill!(wflag, zero(T))
        flattemp = similar(diffcoef); fill!(flattemp, zero(T))
        _calcdiss_wflag!(be)(wflag, pslice, uslice, idim, i1 - 2, j1;
                             ndrange = (i2 - i1 + 5, nj))
        if iflatten == 1
            _calcdiss_flat1_temp!(be)(flattemp, pslice, wflag, idim, i1 - 1, j1;
                                      ndrange = (i2 - i1 + 3, nj))
        else
            _calcdiss_flat3_temp!(be)(flattemp, pslice, eslice, dslice, uslice, wflag,
                                      idim, i1 - 1, j1, g; ndrange = (i2 - i1 + 3, nj))
        end
        _calcdiss_flat_final!(be)(flatten, flattemp, pslice, idim, i1 - 1, i2 + 1, j1;
                                  ndrange = (i2 - i1 + 3, nj))
    elseif iflatten != 0
        throw(ArgumentError("calcdiss!: only iflatten ∈ {0,1,3} supported (got $iflatten)"))
    end

    KA.synchronize(be)
    return diffcoef, flatten
end
