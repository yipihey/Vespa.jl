# ── the exchange operators: CellSet ↔ Cartesian grid (ADR-0006 D3/Phase 4) ────
#
# The conservative transfer between an unstructured cell set (Arepo's Voronoi
# mesh, a particle set) and a uniform Cartesian grid (the scratch hierarchy a
# guest physics module runs on).  Two fidelities:
#
#   :ngp — each cell's INTEGRATED conserved content (value·volume) lands in
#          the grid cell containing its center.  Exactly conservative, spatially
#          first-order.
#   :cic — the content is trilinearly spread over the 8 nearest grid cells.
#          Exactly conservative, smoother.
#
# The exact-geometry path (R3D clipping of the actual Voronoi polyhedra — the
# HierarchicalGrids Layer-4 machinery) needs the 3-D cell GEOMETRY, which the
# Arepo bridge does not export yet (only `get_voronoi_2d`); it slots in here as
# a third method when that bridge call lands.  Conservation is asserted, not
# hoped: both methods preserve Σ value·volume to round-off by construction,
# and `deposit_to_grid` verifies it.

"""
    deposit_to_grid(cs::CellSet, n::Integer; method=:cic, periodic=true)
        -> (; rho, mom, etot, vol)

Conservatively deposit a canonical `CellSet` onto an `n³` uniform grid over the
unit box.  Returns conserved DENSITIES on the grid (each grid cell's deposited
content divided by the grid cell volume) plus the deposited volume field — a
diagnostic of coverage (≈ 1 everywhere when the cell set tiles the box).
"""
function deposit_to_grid(cs::CellSet, n::Integer; method::Symbol = :cic, periodic::Bool = true)
    method in (:ngp, :cic) || error("deposit_to_grid: method must be :ngp or :cic")
    dxg = 1.0 / n
    Vg = dxg^3
    mass = zeros(n, n, n)
    momx = zeros(n, n, n); momy = zeros(n, n, n); momz = zeros(n, n, n)
    en = zeros(n, n, n)
    volume = zeros(n, n, n)
    wrap(i) = periodic ? mod(i - 1, n) + 1 : clamp(i, 1, n)

    @inbounds for c in 1:ncells(cs)
        v = cs.vol[c]
        m = cs.rho[c] * v
        px = cs.mom[c, 1] * v; py = cs.mom[c, 2] * v; pz = cs.mom[c, 3] * v
        e = cs.etot[c] * v
        if method === :ngp
            i = wrap(1 + floor(Int, cs.pos[c, 1] / dxg))
            j = wrap(1 + floor(Int, cs.pos[c, 2] / dxg))
            k = wrap(1 + floor(Int, cs.pos[c, 3] / dxg))
            mass[i, j, k] += m; momx[i, j, k] += px; momy[i, j, k] += py
            momz[i, j, k] += pz; en[i, j, k] += e; volume[i, j, k] += v
        else
            # CIC: weights about the cell center relative to grid-cell centers
            fx = cs.pos[c, 1] / dxg - 0.5; fy = cs.pos[c, 2] / dxg - 0.5; fz = cs.pos[c, 3] / dxg - 0.5
            i0 = floor(Int, fx); j0 = floor(Int, fy); k0 = floor(Int, fz)
            wx = fx - i0; wy = fy - j0; wz = fz - k0
            for dk in 0:1, dj in 0:1, di in 0:1
                w = (di == 0 ? 1 - wx : wx) * (dj == 0 ? 1 - wy : wy) * (dk == 0 ? 1 - wz : wz)
                w == 0 && continue
                i = wrap(i0 + di + 1); j = wrap(j0 + dj + 1); k = wrap(k0 + dk + 1)
                mass[i, j, k] += w * m; momx[i, j, k] += w * px; momy[i, j, k] += w * py
                momz[i, j, k] += w * pz; en[i, j, k] += w * e; volume[i, j, k] += w * v
            end
        end
    end

    # conservation is a property, not a hope
    lg = ledger(cs)
    abs(sum(mass) - lg.mass) <= 1e-12 * max(abs(lg.mass), 1.0) ||
        error("deposit_to_grid: mass not conserved ($(sum(mass)) vs $(lg.mass))")
    abs(sum(en) - lg.energy) <= 1e-12 * max(abs(lg.energy), 1.0) ||
        error("deposit_to_grid: energy not conserved")

    return (rho = mass ./ Vg, mom = (momx ./ Vg, momy ./ Vg, momz ./ Vg),
            etot = en ./ Vg, vol = volume ./ Vg)
end

"""
    sample_profile_to_bins(prof, n) -> Vector{Float64}

Bin an x-profile (from `profile_x`) onto `n` uniform bins over [0,1] by
averaging the points in each bin (the 1-D deposit for tube geometries);
empty bins take the nearest filled value.
"""
function sample_profile_to_bins(prof, n::Integer)
    sums = zeros(n); cnts = zeros(Int, n)
    for (x, ρ) in zip(prof.x, prof.rho)
        b = clamp(1 + floor(Int, x * n), 1, n)
        sums[b] += ρ; cnts[b] += 1
    end
    vals = Vector{Float64}(undef, n)
    for b in 1:n
        if cnts[b] > 0
            vals[b] = sums[b] / cnts[b]
        else
            d = findfirst(k -> (b - k >= 1 && cnts[b - k] > 0) || (b + k <= n && cnts[b + k] > 0), 1:n)
            d === nothing && error("sample_profile_to_bins: no filled bins")
            vals[b] = (b - d >= 1 && cnts[b - d] > 0) ? sums[b - d] / cnts[b - d] :
                      sums[b + d] / cnts[b + d]
        end
    end
    return vals
end

"""
    sample_at_points(grid::AbstractArray{<:Real,3}, pos::AbstractMatrix; periodic=true)
        -> Vector

Trilinearly sample a unit-box `n³` grid field at the given `m×3` positions —
the return path of the exchange (e.g. Moray's photo-rates back onto Voronoi
cell centers).
"""
function sample_at_points(grid::AbstractArray{<:Real,3}, pos::AbstractMatrix; periodic::Bool = true)
    n = size(grid, 1)
    size(grid) == (n, n, n) || error("sample_at_points: grid must be cubic")
    dxg = 1.0 / n
    wrap(i) = periodic ? mod(i - 1, n) + 1 : clamp(i, 1, n)
    out = Vector{Float64}(undef, size(pos, 1))
    @inbounds for c in 1:size(pos, 1)
        fx = pos[c, 1] / dxg - 0.5; fy = pos[c, 2] / dxg - 0.5; fz = pos[c, 3] / dxg - 0.5
        i0 = floor(Int, fx); j0 = floor(Int, fy); k0 = floor(Int, fz)
        wx = fx - i0; wy = fy - j0; wz = fz - k0
        s = 0.0
        for dk in 0:1, dj in 0:1, di in 0:1
            w = (di == 0 ? 1 - wx : wx) * (dj == 0 ? 1 - wy : wy) * (dk == 0 ? 1 - wz : wz)
            w == 0 && continue
            s += w * grid[wrap(i0 + di + 1), wrap(j0 + dj + 1), wrap(k0 + dk + 1)]
        end
        out[c] = s
    end
    return out
end
