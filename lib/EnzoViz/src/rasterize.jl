# Turn a live `Simulation` into plottable arrays, reading only through Vespa's
# public per-cell surface (`for_each_cell`, `cell_center`, `cell_width`,
# `level_of`, `max_level`, `rank`, `domain`, `primitive_at`). No backend internals.
#
# Fields we plot (primitive): density ρ, pressure p, speed |v|. `primitive_at`
# returns (ρ, vx, vy, vz, p).

const FIELD_KEYS = (:density, :pressure, :speed)

@inline function _field_value(W::NTuple{5,Float64}, key::Symbol)
    if key === :density
        return W[1]
    elseif key === :pressure
        return W[5]
    elseif key === :speed
        return sqrt(W[2]^2 + W[3]^2 + W[4]^2)
    else
        error("EnzoViz: unknown field $key (expected one of $FIELD_KEYS)")
    end
end

"""
    raster1d(sim, fields) -> (x, Dict(field => values))

Sort interior cells by x; return the x coordinates and one value vector per
requested field. For 1D problems.
"""
function raster1d(sim, fields)
    b = sim.backend
    pts = Tuple{Float64,NTuple{5,Float64}}[]
    Vespa.for_each_cell(b) do c
        push!(pts, (Vespa.cell_center(b, c)[1], Vespa.primitive_at(sim, c)))
    end
    sort!(pts; by = first)
    x = Float64[p[1] for p in pts]
    cols = Dict{Symbol,Vector{Float64}}()
    for f in fields
        cols[f] = Float64[_field_value(p[2], f) for p in pts]
    end
    return x, cols
end

"""
    LevelGrid

A regular 2D grid covering the whole domain at one refinement level's resolution.
`data[field]` is an `ny × nx` matrix (NaN where no leaf of this level covers the
cell); `xcent`/`ycent` are the pixel-centre coordinates. Row-major `[iy, ix]`
matches Veusz's `SetData2D` (data.shape == (ny, nx)).
"""
struct LevelGrid
    level::Int
    nx::Int
    ny::Int
    xcent::Vector{Float64}
    ycent::Vector{Float64}
    data::Dict{Symbol,Matrix{Float64}}
end

"""
    raster2d_levels(sim, fields) -> Vector{LevelGrid}

For each refinement level present (0..max_level), build a domain-covering regular
grid at that level's resolution and scatter every leaf cell of that level into
its block. Finer levels are sparse (NaN elsewhere) and overlay coarser ones in
the template. Assumes isotropic refinement (square cells), which the HG backend
guarantees.
"""
function raster2d_levels(sim, fields)
    b = sim.backend
    Vespa.rank(b) == 2 || error("raster2d_levels: only rank-2 meshes (got rank $(Vespa.rank(b)))")
    dom = Vespa.domain(b)
    (x0, x1) = dom[1]
    (y0, y1) = dom[2]
    Lx, Ly = x1 - x0, y1 - y0

    # Base resolution = domain span / coarsest (level-0) cell width. We need a
    # level-0 cell width; sample the first level-0 cell.
    w0 = _level0_width(sim)
    nx0 = max(1, round(Int, Lx / w0[1]))
    ny0 = max(1, round(Int, Ly / w0[2]))

    maxlev = Vespa.max_level(b)
    grids = LevelGrid[]
    for lev in 0:maxlev
        r = 2^lev
        nx, ny = nx0 * r, ny0 * r
        dx, dy = Lx / nx, Ly / ny
        xcent = Float64[x0 + (i - 0.5) * dx for i in 1:nx]
        ycent = Float64[y0 + (j - 0.5) * dy for j in 1:ny]
        data = Dict{Symbol,Matrix{Float64}}(f => fill(NaN, ny, nx) for f in fields)
        Vespa.for_each_cell(b) do c
            Vespa.level_of(b, c) == lev || return
            ctr = Vespa.cell_center(b, c)
            ix = clamp(floor(Int, (ctr[1] - x0) / dx) + 1, 1, nx)
            iy = clamp(floor(Int, (ctr[2] - y0) / dy) + 1, 1, ny)
            W = Vespa.primitive_at(sim, c)
            for f in fields
                @inbounds data[f][iy, ix] = _field_value(W, f)
            end
        end
        push!(grids, LevelGrid(lev, nx, ny, xcent, ycent, data))
    end
    return grids
end

# Width of a level-0 cell (the coarsest). Scans for the first leaf whose level is
# the minimum present (0 for a base grid; the base for HG relative levels).
function _level0_width(sim)
    b = sim.backend
    minlev = typemax(Int)
    w = (0.0, 0.0)
    Vespa.for_each_cell(b) do c
        l = Vespa.level_of(b, c)
        if l < minlev
            minlev = l
            wc = Vespa.cell_width(b, c)
            w = (wc[1], wc[2])
        end
    end
    return w
end

"Global min/max of a field over all leaves (for pinning the colour/axis range)."
function field_range(sim, field)
    b = sim.backend
    lo = Inf; hi = -Inf
    Vespa.for_each_cell(b) do c
        v = _field_value(Vespa.primitive_at(sim, c), field)
        v < lo && (lo = v); v > hi && (hi = v)
    end
    return (lo, hi)
end
