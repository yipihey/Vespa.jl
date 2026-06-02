"""
    EnzoBackend

A `MeshInterface` backend over a **live Enzo grid**, so EnzoNG's *unchanged*
`Simulation`/driver runs through the seam on Enzo-owned state — the full
seam-level integration (vs the E3 slot, which reused EnzoNG kernels directly).

`EnzoGridMesh` satisfies the seam by delegating the (uniform) geometry/topology to
a `RefMesh.UniformMesh` over the grid's ACTIVE region, and links the live Enzo
handle + field map. Because EnzoNG stores **conserved** `(ρ, ρv, E)` while Enzo
stores `(Density, Velocity1, TotalEnergy_specific)`, the field state cannot be a
zero-copy alias — `sync_from_enzo!`/`sync_to_enzo!` transform between them around
each EnzoNG step. 1D single-grid for now (the demo); ND/AMR is a follow-up.
"""
module EnzoBackend

import MeshInterface as MI
using RefMesh: UniformMesh
import EnzoLib

export EnzoGridMesh, sync_from_enzo!, sync_to_enzo!

# The backend stores only Ints — both the Enzo FieldType field indices and the
# CONSERVED-state role indices (which sv component is density / momentum / energy).
# The role indices are supplied by the caller FROM the EquationSet model
# (`density_index`/`momentum_indices`/`energy_index`), so the variable choice is
# the model's, not hardcoded here — and EnzoBackend stays free of any EnzoNG dep.
struct EnzoGridMesh{N} <: MI.AbstractMeshBackend
    inner::UniformMesh{N,Float64}     # uniform mesh over the active region (seam delegate)
    h::Ptr{Cvoid}                     # live Enzo session/problem handle
    grid::Int                         # grid index in the hierarchy
    nghost::Int                       # ghost zones per side
    di::Int; vi::Int; ei::Int         # Enzo 0-based field indices: Density, Velocity1, TotalEnergy
    cdi::Int                          # conserved index of mass density
    cmom::NTuple{3,Int}               # conserved indices of (x,y,z) momentum
    cei::Int                          # conserved index of total energy
end

"""
    EnzoGridMesh(h; grid=0, nghost=3, domain=((0.0,1.0),),
                 cons_density=1, cons_momentum=(2,3,4), cons_energy=5)

Wrap live Enzo grid `grid` (a 1D hydro grid) as a seam backend. The `cons_*`
role indices say which conserved-state components are density/momentum/energy —
pass them from the `EquationSet` model (defaults are the ideal-hydro layout).
"""
function EnzoGridMesh(h::Ptr{Cvoid}; grid::Integer = 0, nghost::Integer = 3,
                      domain = ((0.0, 1.0),),
                      cons_density::Integer = 1, cons_momentum = (2, 3, 4),
                      cons_energy::Integer = 5)
    T = EnzoLib.problem_grid_size(h, grid)
    N = T - 2 * nghost
    inner = UniformMesh((N,), domain)
    di = EnzoLib.field_index(h, 0; grid = grid)    # Density
    vi = EnzoLib.field_index(h, 4; grid = grid)    # Velocity1 (x)
    ei = EnzoLib.field_index(h, 1; grid = grid)    # TotalEnergy (specific)
    return EnzoGridMesh{1}(inner, h, Int(grid), Int(nghost), di, vi, ei,
                           Int(cons_density), Tuple(Int.(cons_momentum)), Int(cons_energy))
end

# ── seam: delegate everything to the inner uniform mesh ───────────────────────
MI.rank(m::EnzoGridMesh)        = MI.rank(m.inner)
MI.domain(m::EnzoGridMesh)      = MI.domain(m.inner)
MI.n_cells(m::EnzoGridMesh)     = MI.n_cells(m.inner)
MI.max_level(m::EnzoGridMesh)   = MI.max_level(m.inner)
MI.level_of(m::EnzoGridMesh, args...)    = MI.level_of(m.inner, args...)
MI.cell_center(m::EnzoGridMesh, args...) = MI.cell_center(m.inner, args...)
MI.cell_width(m::EnzoGridMesh, args...)  = MI.cell_width(m.inner, args...)
MI.cell_volume(m::EnzoGridMesh, args...) = MI.cell_volume(m.inner, args...)
MI.face_area(m::EnzoGridMesh, args...)   = MI.face_area(m.inner, args...)
MI.neighbor(m::EnzoGridMesh, args...; kw...)        = MI.neighbor(m.inner, args...; kw...)
MI.allocate_fields(m::EnzoGridMesh, args...; kw...) = MI.allocate_fields(m.inner, args...; kw...)
MI.field_view(m::EnzoGridMesh, args...)             = MI.field_view(m.inner, args...)
MI.for_each_cell(f, m::EnzoGridMesh; kw...) = MI.for_each_cell(f, m.inner; kw...)
MI.for_each_face(f, m::EnzoGridMesh; kw...) = MI.for_each_face(f, m.inner; kw...)

# ── field sync: live Enzo grid ↔ EnzoNG conserved views ───────────────────────
# Active cell i (1..N) maps to flat index nghost+i (1-based) of the column-major
# Enzo field; sv is the NTuple of 5 conserved views (ρ, ρvx, ρvy, ρvz, E_density),
# indexed by the inner mesh's CartesianIndex handle.
"Pull the live Enzo grid state into EnzoNG's conserved views `sv` (Enzo → conserved)."
function sync_from_enzo!(sv, m::EnzoGridMesh{1})
    d  = EnzoLib.problem_get_field(m.h, m.di, m.grid)
    vx = EnzoLib.problem_get_field(m.h, m.vi, m.grid)
    es = EnzoLib.problem_get_field(m.h, m.ei, m.grid)     # specific total energy
    N = MI.n_cells(m.inner)
    @inbounds for i in 1:N
        f = m.nghost + i
        ρ = d[f]; u = vx[f]; e = es[f]
        I = CartesianIndex(i)
        sv[m.cdi][I] = ρ
        sv[m.cmom[1]][I] = ρ * u; sv[m.cmom[2]][I] = 0.0; sv[m.cmom[3]][I] = 0.0
        sv[m.cei][I] = ρ * e                              # total energy density = ρ·e_specific
    end
    return nothing
end

"Push EnzoNG's conserved views `sv` back into the live Enzo grid (conserved → Enzo)."
function sync_to_enzo!(m::EnzoGridMesh{1}, sv)
    d  = EnzoLib.problem_get_field(m.h, m.di, m.grid)
    vx = EnzoLib.problem_get_field(m.h, m.vi, m.grid)
    es = EnzoLib.problem_get_field(m.h, m.ei, m.grid)
    N = MI.n_cells(m.inner)
    @inbounds for i in 1:N
        f = m.nghost + i
        I = CartesianIndex(i)
        ρ = sv[m.cdi][I]; mx = sv[m.cmom[1]][I]; E = sv[m.cei][I]
        d[f] = ρ; vx[f] = mx / ρ; es[f] = E / ρ           # back to specific total energy
    end
    EnzoLib.problem_set_field(m.h, m.di, d; grid = m.grid)
    EnzoLib.problem_set_field(m.h, m.vi, vx; grid = m.grid)
    EnzoLib.problem_set_field(m.h, m.ei, es; grid = m.grid)
    return nothing
end

end # module
