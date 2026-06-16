# tables.jl — log-grid and linear-grid interpolation tables for ChemistryKernels.
#
# LogTable: clamp-floor-lerp on a log(T) grid. Host-side infrastructure; the HyRec
# rate-backend seam. CPU-only for now — GPU paths use analytic RecFast formulas.
#
# FAlphaTable: 1-D Lyα mixing fraction f_α(z). Linear interpolation in z. FA_ZERO
# is the default (f_α ≡ 0) which recovers standard Peebles recombination exactly.

export LogTable, lookup, make_log_table
export FAlphaTable, fa_at, FA_ZERO

# ── LogTable: log-spaced lookup + lerp ───────────────────────────────────────

"""
    LogTable{T}

Read-only lookup table on a log-spaced grid. `lookup(tbl, x)` is a branch-free
clamp + floor + single linear interpolation in log(x). Primary use: HyRec rate
tables for α_e(T_b). Build with `make_log_table`; the RecFast analytic formula
is the current default and does not need this struct at runtime.
"""
struct LogTable{T}
    logx_lo  :: T
    inv_dlog :: T
    data     :: Vector{T}
end

"""
    lookup(tbl::LogTable, x) -> eltype(tbl)

Clamp-floor-lerp in log(x). Returns `tbl.data[1]` below range and `tbl.data[end]`
above range (no extrapolation). Branch-free on the main path.
"""
@inline function lookup(tbl::LogTable{T}, x::Real) where {T}
    N    = length(tbl.data)
    fi   = (log(T(x)) - tbl.logx_lo) * tbl.inv_dlog
    fi   = clamp(fi, zero(T), T(N - 1))
    i    = min(floor(Int, fi), N - 2)    # safe lower index (≥0, ≤N-2)
    frac = fi - T(i)
    @inbounds return tbl.data[i + 1] + frac * (tbl.data[i + 2] - tbl.data[i + 1])
end

"""
    make_log_table(f, x_lo, x_hi, N; dtype=Float64) -> LogTable

Build a `LogTable` by evaluating scalar function `f(x)` on N log-spaced points in
[x_lo, x_hi]. Use to precompute HyRec or other tabulated rate coefficients.
"""
function make_log_table(f, x_lo::Real, x_hi::Real, N::Int;
                        dtype::Type{T} = Float64) where {T}
    logx_lo  = T(log(x_lo))
    inv_dlog = T(N - 1) / T(log(x_hi) - log(x_lo))
    data     = Vector{T}(undef, N)
    for k in 1:N
        x_k     = exp(logx_lo + T(k - 1) / inv_dlog)
        data[k] = T(f(x_k))
    end
    return LogTable{T}(logx_lo, inv_dlog, data)
end

# ── FAlphaTable: Lyα mixing fraction f_α(z) ──────────────────────────────────

"""
    FAlphaTable

1-D table of the Lyα mixing fraction f_α(z) ∈ [0,1], linearly interpolated in z.
`z_nodes` must be sorted ascending. Values outside the table range clamp to the
nearest endpoint.

Physical meaning: f_α = 0 recovers the standard cell-local Peebles recombination.
f_α > 0 mixes in the smoothed neutral density in the Sobolev escape rate R_α, making
the C-factor sensitive to the mean neutral density in the Lyα mean-free-path volume.
The function f_α(z) peaks near the recombination epoch (z ≈ 1100) and goes to 0 at
both high z (short mixing length) and low z (C → 1). Supply from an offline
Monte-Carlo Lyα transport calculation; use `FA_ZERO` to disable mixing entirely.
"""
struct FAlphaTable
    z_nodes :: Vector{Float64}
    fa_vals :: Vector{Float64}
end

"""
    fa_at(tbl::FAlphaTable, z) -> Float64

Linear interpolation of f_α at redshift `z`. Clamps to endpoint values outside range.
"""
@inline function fa_at(tbl::FAlphaTable, z::Real)
    zv = tbl.z_nodes
    fv = tbl.fa_vals
    N  = length(zv)
    N == 0 && return 0.0
    Float64(z) <= zv[1]   && return fv[1]
    Float64(z) >= zv[end] && return fv[end]
    # binary search
    lo = 1; hi = N
    while hi - lo > 1
        mid = (lo + hi) >> 1
        Float64(z) >= zv[mid] ? (lo = mid) : (hi = mid)
    end
    t = (Float64(z) - zv[lo]) / (zv[hi] - zv[lo])
    return fv[lo] + t * (fv[hi] - fv[lo])
end

"""
    FA_ZERO

Default `FAlphaTable` with f_α ≡ 0 everywhere — standard Peebles recombination,
no Lyα mixing. Opt-in: pass a non-trivial table to `solve_chem_mixing!` to enable.
"""
const FA_ZERO = FAlphaTable([0.0, 1.0e5], [0.0, 0.0])
