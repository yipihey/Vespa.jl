# ── per-code adapters: extract → CellSet, inject! ← CellSet (ADR-0006 D3) ─────
#
# Each adapter pair converts between one code's native state and the canonical
# CellSet, AT the bridge boundary.  `extract` returns the CellSet; `inject!`
# writes a CellSet (typically a modified copy of an extract from the SAME
# configuration — same mesh, same ordering) back into the live code.  The
# round-trip extract → inject! → extract must be bit-identical; the ledger
# must be preserved to round-off.  That is the D3 gate, tested per code.

# ── Enzo (unigrid root grid; AMR composites are Phase 4) ─────────────────────

const _ENZO_GHOST = 3   # Enzo DEFAULT_GHOST_ZONES; verified by the runtime checks below

# FieldType codes (Enzo typedefs): Density=0, TotalEnergy=1, Velocity1/2/3=4/5/6
_enzo_fi(h, t; grid) = EnzoLib.field_index(h, t; grid = grid)

"""
    enzo_extract(h; grid=0) -> CellSet

Canonicalize an Enzo grid's active region.  Enzo stores SPECIFIC total energy
and velocities; the adapter forms the conserved (ρ, ρu, E) on the way out.
Enzo's Sod/Sedov setups are already in normalized box units, so the unit
scales are 1.
"""
function enzo_extract(h::EnzoLib.Handle; grid::Integer = 0)
    rank = EnzoLib.problem_grid_rank(h, grid)
    dims = EnzoLib.problem_grid_dims(h, grid)                  # incl ghosts; len 3
    l, r = EnzoLib.problem_grid_edge(h, grid)
    nact = [dims[d] > 1 ? dims[d] - 2 * _ENZO_GHOST : 1 for d in 1:3]
    all(>(0), nact) || error("enzo_extract: ghost-zone assumption broke (dims=$dims)")
    dx = [nact[d] > 1 || d <= rank ? (r[d] - l[d]) / max(nact[d], 1) : 1.0 for d in 1:3]
    cellvol = prod(dx[1:rank])

    field(t) = reshape(EnzoLib.problem_get_field(h, _enzo_fi(h, t; grid = grid), grid), dims...)
    act(d) = dims[d] > 1 ? ((_ENZO_GHOST + 1):(dims[d] - _ENZO_GHOST)) : (1:1)
    sl = (act(1), act(2), act(3))

    ρ  = Array(field(0)[sl...])
    te = Array(field(1)[sl...])                                # specific total energy
    types = EnzoLib.problem_field_types(h, grid)
    vel(d) = (4 + d - 1) in types ? Array(field(4 + d - 1)[sl...]) : zeros(size(ρ))
    v = (vel(1), vel(2), vel(3))

    n = length(ρ)
    pos = Matrix{Float64}(undef, n, 3)
    k = 0
    for c in CartesianIndices(ρ)
        k += 1
        for d in 1:3
            pos[k, d] = dims[d] > 1 ? l[d] + (c[d] - 0.5) * dx[d] : 0.5
        end
    end
    rho = vec(ρ)
    mom = hcat(vec(v[1]) .* rho, vec(v[2]) .* rho, vec(v[3]) .* rho)
    etot = vec(te) .* rho                                      # E = ρ·(specific total energy)

    return CellSet(:enzo, pos, fill(cellvol, n), rho, mom, etot,
                   (length = 1.0, time = 1.0, density = 1.0),
                   (handle = h, grid = Int(grid), dims = Tuple(dims), rank = rank,
                    slices = sl, types = types))
end

"""
    enzo_inject!(h, cs::CellSet; grid=cs.meta.grid)

Write a CellSet (from `enzo_extract` of the SAME grid) back into the live
Enzo memory: conserved → Enzo's specific-energy/velocity layout, active region
only (ghosts untouched — Enzo's own SetBoundaryConditions owns those).
"""
function enzo_inject!(h::EnzoLib.Handle, cs::CellSet; grid::Integer = cs.meta.grid)
    cs.code === :enzo || error("enzo_inject!: CellSet came from $(cs.code)")
    dims = collect(cs.meta.dims); sl = cs.meta.slices
    shape = map(length, sl)
    ρ = reshape(cs.rho, shape...)
    te = reshape(cs.etot ./ cs.rho, shape...)
    setf(t, A) = begin
        fi = _enzo_fi(h, t; grid = grid)
        full = reshape(EnzoLib.problem_get_field(h, fi, grid), dims...)
        full[sl...] = A
        EnzoLib.problem_set_field(h, fi, vec(full); grid = grid)
    end
    setf(0, ρ)
    setf(1, te)
    for d in 1:3
        (4 + d - 1) in cs.meta.types || continue
        setf(4 + d - 1, reshape(cs.mom[:, d] ./ cs.rho, shape...))
    end
    return nothing
end

# ── RAMSES (one level of the oct hierarchy) ──────────────────────────────────

# mini-ramses cell centers (pm/rho_fine.f90:281):
#   x_d = (2·ckey_d + bit_d(c) + 0.5) · dx,  bit_d(c) = (c-1) >> (d-1) & 1
# with dx the CELL size at ilevel and x in [0, boxlen) code units.

"""
    ramses_extract(h; lev, boxlen=1.0, lib=:cpu) -> CellSet

Canonicalize one RAMSES level's conserved state (`uold`): vars 1..5 are
(ρ, ρu, ρv, ρw, E) — already the canonical variables.  Positions follow the
mini-ramses ckey convention; normalization divides out `boxlen`.
"""
function ramses_extract(h::RamsesLib.Handle; lev::Integer, boxlen::Real = 1.0,
                        lib::Symbol = :cpu,
                        species::Union{Nothing,AbstractVector{<:Integer}} = nothing)
    nv = RamsesLib.nvar(; lib = lib)
    nv >= 5 || error("ramses_extract: nvar=$nv < 5 — not a HYDRO build (set RAMSES_LIB to bin64h)")
    if species !== nothing
        maximum(species) <= nv ||
            error("ramses_extract: species vars $species exceed nvar=$nv (rebuild with NPSCAL>=$(length(species)))")
    end
    ck, U = RamsesLib.get_hydro_all(h, :uold, lev; lib = lib)  # ck: noct×3, U: noct×8×nv
    noct = size(ck, 1)
    dx = 1.0 / 2^lev                                           # normalized cell size
    n = 8 * noct
    pos = Matrix{Float64}(undef, n, 3)
    rho = Vector{Float64}(undef, n)
    mom = Matrix{Float64}(undef, n, 3)
    etot = Vector{Float64}(undef, n)
    spec = species === nothing ? nothing : Matrix{Float64}(undef, n, length(species))
    @inbounds for i in 1:noct, c in 1:8
        k = (i - 1) * 8 + c
        for d in 1:3
            bit = (c - 1) >> (d - 1) & 1
            pos[k, d] = (2 * ck[i, d] + bit + 0.5) * dx
        end
        rho[k]  = U[i, c, 1]
        mom[k, 1] = U[i, c, 2]; mom[k, 2] = U[i, c, 3]; mom[k, 3] = U[i, c, 4]
        etot[k] = U[i, c, 5]
        if spec !== nothing
            for (j, iv) in enumerate(species); spec[k, j] = U[i, c, iv]; end   # ρ·x
        end
    end
    all(p -> 0.0 <= p <= 1.0, pos) ||
        error("ramses_extract: positions escaped [0,1] — ckey convention drifted")
    return CellSet(:ramses, pos, fill(dx^3, n), rho, mom, etot,
                   (length = float(boxlen), time = float(boxlen), density = 1.0),
                   (handle = h, lev = Int(lev), ckey = ck, noct = noct, lib = lib,
                    species_vars = species === nothing ? Int[] : collect(Int, species)),
                   spec)
end

"""
    ramses_inject!(h, cs::CellSet)

Write a CellSet (from `ramses_extract` of the SAME level/mesh) back into
`uold` via `set_hydro!`, var by var, matched by the stored oct keys.
"""
function ramses_inject!(h::RamsesLib.Handle, cs::CellSet)
    cs.code === :ramses || error("ramses_inject!: CellSet came from $(cs.code)")
    ck = cs.meta.ckey; noct = cs.meta.noct; lev = cs.meta.lev; lib = cs.meta.lib
    asoct(v) = permutedims(reshape(v, 8, noct))                # n → noct×8
    RamsesLib.set_hydro!(h, :uold, 1, lev, ck, asoct(cs.rho); lib = lib)
    for d in 1:3
        RamsesLib.set_hydro!(h, :uold, 1 + d, lev, ck, asoct(cs.mom[:, d]); lib = lib)
    end
    RamsesLib.set_hydro!(h, :uold, 5, lev, ck, asoct(cs.etot); lib = lib)
    if cs.species !== nothing
        sv = get(cs.meta, :species_vars, Int[])
        length(sv) == size(cs.species, 2) ||
            error("ramses_inject!: $(length(sv)) species vars vs $(size(cs.species,2)) species columns")
        for (j, iv) in enumerate(sv)
            RamsesLib.set_hydro!(h, :uold, iv, lev, ck, asoct(cs.species[:, j]); lib = lib)
        end
    end
    return nothing
end

# ── Arepo (the Voronoi cell set — naturally unstructured) ────────────────────

"""
    arepo_extract(h; boxlen) -> CellSet

Canonicalize Arepo's gas cells.  Positions/volumes are normalized by `boxlen`;
the conserved densities are formed from the primitives (ρ, u_thermal, fluid
velocity), which are well-defined regardless of Arepo's internal cell-
integrated bookkeeping.  Arepo's velocities live on the particle record of the
gas cells (the first `numgas` particles).
"""
function arepo_extract(h::ArepoLib.Handle; boxlen::Real, species::Bool = false)
    nfo = ArepoLib.info(h)
    ng = nfo.numgas
    rho = ArepoLib.get_cell_field(h, :rho)
    vol = ArepoLib.get_cell_field(h, :volume)
    uth = ArepoLib.get_cell_field(h, :utherm)
    ctr = ArepoLib.get_cell_field(h, :center)
    vel = ArepoLib.get_particle_field(h, :vel)[1:ng, :]        # gas cells lead P[]
    L = float(boxlen)
    pos = ctr ./ L
    # Normalize volumes by the MEASURED domain volume: Arepo's 1-D examples use
    # a box of 20×1×1 (BoxSize applies to x), and the Voronoi cells tile the
    # whole domain, so Σvol IS the domain volume — assumption-free.
    v̂ = vol ./ sum(vol)
    mom = rho .* vel
    etot = rho .* (uth .+ 0.5 .* vec(sum(abs2, vel; dims = 2)))
    # primitive abundances x → mass densities ρ·x (the canonical species convention)
    spec = species ? Float64.(rho) .* Float64.(ArepoLib.get_cell_field(h, :scalars)) : nothing
    return CellSet(:arepo, pos, v̂, rho, mom, etot,
                   # `volume` = the PHYSICAL domain volume (Arepo's 1-D boxes are
                   # L×1×1, not L³) — exact-deposit geometry needs the true scale
                   (length = L, time = L, density = 1.0, volume = sum(vol)),
                   (handle = h, numgas = ng),
                   spec)
end

"""
    arepo_inject_species!(h, cs::CellSet)

Write a CellSet's species (mass densities ρ·x) back to Arepo's passive-scalar
abundances via `set_cell_field!(h, :scalars, x)`.  No-op when `cs.species` is
`nothing`.
"""
function arepo_inject_species!(h::ArepoLib.Handle, cs::CellSet)
    cs.species === nothing && return nothing
    rho = Float64.(ArepoLib.get_cell_field(h, :rho))
    ArepoLib.set_cell_field!(h, :scalars, cs.species ./ max.(rho, eps()))   # ρ·x → x
    return nothing
end

"""
    arepo_roundtrip_conserved(h) -> Bool

The Arepo D3 round-trip gate on the SETTABLE conserved surface: read the
cell-integrated (:energy, :momentum) conserved fields, write them back
verbatim, and re-read — must be bit-identical.  (Arepo derives ρ/p itself;
density is not directly settable, which the report notes.)
"""
function arepo_roundtrip_conserved(h::ArepoLib.Handle)
    e0 = ArepoLib.get_cell_field(h, :energy)
    m0 = ArepoLib.get_cell_field(h, :momentum)
    ArepoLib.set_cell_field!(h, :energy, e0)
    ArepoLib.set_cell_field!(h, :momentum, m0)
    return ArepoLib.get_cell_field(h, :energy) == e0 &&
           ArepoLib.get_cell_field(h, :momentum) == m0
end
