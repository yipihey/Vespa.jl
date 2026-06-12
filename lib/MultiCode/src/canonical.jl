# ── the canonical state (ADR-0006 D3) ─────────────────────────────────────────

"""
    CellSet

A code-neutral snapshot of the conserved gas state on a set of cells, in
**common normalized units** (box length 1; ρ/p as the problem spec defines
them).  `units` records the scales the adapter divided out (so the original
code units are recoverable), `meta` carries per-code context the matching
`inject!` needs (handles, level, layout shapes).

Fields are row-aligned: cell `i` is `pos[i,:]`, `vol[i]`, `rho[i]`, `mom[i,:]`,
`etot[i]`.  `mom` is momentum DENSITY (ρu), `etot` total energy DENSITY — the
conservative variables every finite-volume code shares.

`species` optionally carries advected chemical species as MASS densities ρ·x
(n×nspecies), `nothing` when the snapshot is pure hydro.  The v2026 reduced
primordial network uses two columns, HII and H2I; the shared Grackle service
(`GrackleChem`) and the per-code chemistry slots read/write exactly those, so a
shared IC can ride the chemistry across codes that already share hydro state.
"""
struct CellSet
    code::Symbol
    pos::Matrix{Float64}      # n×3 cell centers (normalized)
    vol::Vector{Float64}      # n cell volumes (normalized; box volume sums to 1)
    rho::Vector{Float64}      # mass density
    mom::Matrix{Float64}      # n×3 momentum density ρu
    etot::Vector{Float64}     # total energy density
    units::NamedTuple         # scales divided out: (; length, time, density)
    meta::NamedTuple          # per-code adapter context (opaque to consumers)
    species::Union{Nothing,Matrix{Float64}}  # n×nspecies mass densities ρ·x, or nothing
end

# backward-compatible constructor: pure-hydro CellSet (no species)
CellSet(code, pos, vol, rho, mom, etot, units, meta) =
    CellSet(code, pos, vol, rho, mom, etot, units, meta, nothing)

"Number of advected species columns (0 when the CellSet is pure hydro)."
nspecies(cs::CellSet) = cs.species === nothing ? 0 : size(cs.species, 2)

ncells(cs::CellSet) = length(cs.vol)

"""
    ledger(cs::CellSet) -> (; mass, momentum, energy, volume, cells)

The conservation ledger: the volume-integrated conserved totals.  Every
`extract`/`inject!` round-trip must preserve this to round-off — the D3 gate.
"""
function ledger(cs::CellSet)
    mass = 0.0; en = 0.0; vol = 0.0
    mx = 0.0; my = 0.0; mz = 0.0
    @inbounds for i in eachindex(cs.vol)
        v = cs.vol[i]
        mass += cs.rho[i] * v
        en   += cs.etot[i] * v
        mx   += cs.mom[i, 1] * v
        my   += cs.mom[i, 2] * v
        mz   += cs.mom[i, 3] * v
        vol  += v
    end
    return (mass = mass, momentum = (mx, my, mz), energy = en,
            volume = vol, cells = length(cs.vol))
end

"Relative difference of two ledgers' conserved totals (the round-trip metric)."
function ledger_drift(a, b)
    rel(x, y) = abs(x - y) / max(abs(x), abs(y), eps())
    return max(rel(a.mass, b.mass), rel(a.energy, b.energy),
               maximum(rel.(a.momentum, b.momentum) .* (abs.(a.momentum) .> 1e-13)))
end
