# uv_background.jl — pluggable metagalactic UV/X-ray background (photoionisation +
# photoheating), e.g. Haardt & Madau (2012, ApJ 746, 125).
#
# The chemistry kernel is self-contained and table-free for its *rates*; an external
# radiation field is, by nature, tabulated INPUT.  We keep that input decoupled: a
# `UVBackground` holds the metagalactic rates vs redshift and `uvb_rates(uvb, z)`
# returns the scalar rates at the current z, which the driver hands to the network
# (helium ionisation equilibrium, H photoionisation) and the energy equation
# (photoheating).  With no UVB the network is exactly the primordial-only model.
#
# Rate convention (the standard primordial-network UVB rates; same symbols as the
# original Enzo/`HM12` machinery):
#   k24 = Γ_HI   [s⁻¹]    HI  + γ → HII  + e
#   k26 = Γ_HeI  [s⁻¹]    HeI + γ → HeII + e
#   k25 = Γ_HeII [s⁻¹]    HeII+ γ → HeIII+ e
#   piHI, piHeI, piHeII [erg s⁻¹]  photoheating (energy deposited per photoionisation
#                                   of that species, ⟨hν−hν_th⟩ × Γ)
#
# Interpolation is log-linear in the rate over z (matching HM12/grackle); the
# background is OFF (all rates 0) above the tabulated z_max (before the sources
# switch on) and clamped to the endpoint below z_min.  Pure & allocation-free.

export UVBackground, uvb_rates, read_uvb_table, read_treecool, fg20_uvb

"""
    UVBackground(lnzp1, k24, k25, k26, piHI, piHeI, piHeII)

Tabulated metagalactic UV/X-ray background.  `lnzp1` is an ASCENDING grid of
ln(1+z); the six rate vectors give, on that grid, the photoionisation rates
`k24=Γ_HI`, `k26=Γ_HeI`, `k25=Γ_HeII` [s⁻¹] and the photoheating rates
`piHI`, `piHeI`, `piHeII` [erg s⁻¹].  Construct directly from arrays, or load a
columnar table with [`read_uvb_table`](@ref) (e.g. an export of the Haardt &
Madau 2012 rates).  See [`uvb_rates`](@ref) for evaluation.
"""
struct UVBackground{V<:AbstractVector}
    lnzp1  :: V    # ln(1+z) grid, ascending
    k24    :: V    # Γ_HI   [s⁻¹]
    k25    :: V    # Γ_HeII [s⁻¹]
    k26    :: V    # Γ_HeI  [s⁻¹]
    piHI   :: V    # HI   photoheating [erg s⁻¹]
    piHeI  :: V    # HeI  photoheating [erg s⁻¹]
    piHeII :: V    # HeII photoheating [erg s⁻¹]
end

# log-linear interpolation of a positive rate vector `r` at ln(1+z) = x on the grid
# `g` (ascending). Returns 0 above the grid (UVB not yet on); endpoint-clamped below.
@inline function _uvb_interp(g::AbstractVector, r::AbstractVector, x::Real)
    R = float(typeof(x))
    n = length(g)
    x <= g[1]   && return R(r[1])
    x >= g[n]   && return zero(R)          # z above z_max ⇒ sources off ⇒ 0
    i = searchsortedfirst(g, x)            # g[i-1] < x ≤ g[i]
    r0 = r[i-1];  r1 = r[i]
    (r0 <= 0 || r1 <= 0) && return R(r0 + (r1 - r0) * (x - g[i-1]) / (g[i] - g[i-1]))
    t = (x - g[i-1]) / (g[i] - g[i-1])     # log-linear in the rate
    return R(exp(log(r0) + t * (log(r1) - log(r0))))
end

"""
    uvb_rates(uvb, z) -> (k24, k25, k26, piHI, piHeI, piHeII)

Photoionisation rates `k24=Γ_HI`, `k25=Γ_HeII`, `k26=Γ_HeI` [s⁻¹] and photoheating
`piHI, piHeI, piHeII` [erg s⁻¹] of the background `uvb` at redshift `z`, log-linearly
interpolated in ln(1+z).  Returns all zeros above the tabulated z_max (background
not yet switched on).  Pure.
"""
@inline function uvb_rates(uvb::UVBackground, z::Real)
    x = log(one(float(typeof(z))) + z)
    g = uvb.lnzp1
    return (_uvb_interp(g, uvb.k24,   x), _uvb_interp(g, uvb.k25,    x),
            _uvb_interp(g, uvb.k26,   x), _uvb_interp(g, uvb.piHI,   x),
            _uvb_interp(g, uvb.piHeI, x), _uvb_interp(g, uvb.piHeII, x))
end

"""
    read_uvb_table(path; T=Float64) -> UVBackground

Read a whitespace/comma-separated table with one header-comment style (`#`) and
seven columns per row:

    z   Γ_HI   Γ_HeII   Γ_HeI   piHI   piHeI   piHeII

(the standard primordial-network UVB ordering: k24, k25, k26, then the three
photoheating rates).  Rows may be in any z order; they are sorted ascending in
ln(1+z).  Use this to load an exported Haardt & Madau (2012) rate table.
"""
function read_uvb_table(path::AbstractString; T::Type = Float64)
    rows = Vector{NTuple{7,T}}()
    for ln in eachline(path)
        s = strip(ln)
        (isempty(s) || startswith(s, '#')) && continue
        v = parse.(T, split(replace(s, ',' => ' ')))
        @assert length(v) >= 7 "UVB table row needs 7 columns (z k24 k25 k26 piHI piHeI piHeII): $ln"
        push!(rows, (v[1], v[2], v[3], v[4], v[5], v[6], v[7]))
    end
    sort!(rows; by = r -> r[1])                      # ascending z
    lnzp1  = T[log(one(T) + r[1]) for r in rows]
    return UVBackground(lnzp1,
                        T[r[2] for r in rows], T[r[3] for r in rows],
                        T[r[4] for r in rows], T[r[5] for r in rows],
                        T[r[6] for r in rows], T[r[7] for r in rows])
end

"""
    read_treecool(path; T=Float64) -> UVBackground

Read a GADGET/GIZMO/Arepo **TREECOOL**-format table (as distributed for FG20, HM12,
FG09, …).  Columns:

    log10(1+z)   Γ_HI   Γ_HeI   Γ_HeII   q̇_HI   q̇_HeI   q̇_HeII

(Γ in s⁻¹, q̇ photoheating in erg s⁻¹).  Note the helium photoionisation columns are
in HeI-then-HeII order here; they are mapped onto the internal `k26=Γ_HeI` /
`k25=Γ_HeII` fields.  First column is log₁₀(1+z), converted to the ln(1+z) grid.
"""
function read_treecool(path::AbstractString; T::Type = Float64)
    rows = Vector{NTuple{7,T}}()
    for ln in eachline(path)
        s = strip(ln)
        (isempty(s) || startswith(s, '#')) && continue
        v = parse.(T, split(s))
        @assert length(v) >= 7 "TREECOOL row needs 7 columns: $ln"
        push!(rows, (v[1], v[2], v[3], v[4], v[5], v[6], v[7]))
    end
    sort!(rows; by = r -> r[1])                          # ascending log10(1+z)
    ln10  = log(T(10))
    lnzp1 = T[r[1] * ln10 for r in rows]                 # log10(1+z) → ln(1+z)
    return UVBackground(lnzp1,
                        T[r[2] for r in rows],            # k24 = Γ_HI
                        T[r[4] for r in rows],            # k25 = Γ_HeII (col 4)
                        T[r[3] for r in rows],            # k26 = Γ_HeI  (col 3)
                        T[r[5] for r in rows],            # piHI   = q̇_HI
                        T[r[6] for r in rows],            # piHeI  = q̇_HeI
                        T[r[7] for r in rows])            # piHeII = q̇_HeII
end

const _UVB_DATA_DIR = joinpath(@__DIR__, "..", "data")

"""
    fg20_uvb(; rescaled=true, T=Float64) -> UVBackground

The Faucher-Giguère (2020, MNRAS 493, 1614; arXiv:1903.08657) metagalactic UV/X-ray
background, loaded from the TREECOOL table shipped under `data/`.  `rescaled=true`
(recommended, default) uses the "effective" fiducial FG20 model with photoheating
rates rescaled ×0.68 (Gaikwad et al. 2020) for a better match to the measured IGM
temperature evolution at z≥2; `rescaled=false` uses the default effective heating
rates.  Photoionisation rates are identical between the two.
"""
function fg20_uvb(; rescaled::Bool = true, T::Type = Float64)
    f = rescaled ? "fg20_treecool_eff_rescaled_heating_rates_068.dat" :
                   "fg20_treecool_eff_default.dat"
    return read_treecool(joinpath(_UVB_DATA_DIR, f); T = T)
end
