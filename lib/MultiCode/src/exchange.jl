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
    deposit_exact(cs::CellSet, geometry, n::Integer; periodic=true)
        -> (; rho, mom, etot, vol)

The EXACT exchange: deposit a Voronoi `CellSet` onto an `n³` unit-box grid by
geometric clipping (R3D).  `geometry` is `ArepoLib.get_voronoi_3d(h)` — each
cell's polyhedron as face rings of Delaunay circumcenters, in MESH units
(normalized here by `cs.units.length`).  Every cell is decomposed into
tetrahedra (cell center apex + a fan over each face ring — Voronoi cells are
convex, so any interior apex is valid), and each tetrahedron's intersection
volume with each overlapping grid cell is computed by clipping the grid-cell
box against the tet's four face planes.  The deposited fraction of each cell
is therefore exact to floating point — per-cell volumes reproduce Arepo's own
`SphP.Volume`, the property NGP/CIC cannot give.
"""
function deposit_exact(cs::CellSet, geometry, n::Integer; periodic::Bool = true)
    L = cs.units.length
    # cs.vol is DOMAIN-volume normalized; the clip works in (length L)³ box
    # units — `volscale` converts between them (1 when the domain is the cube)
    volscale = get(cs.units, :volume, L^3) / L^3
    dxg = 1.0 / n
    Vg = dxg^3
    mass = zeros(n, n, n)
    momx = zeros(n, n, n); momy = zeros(n, n, n); momz = zeros(n, n, n)
    en = zeros(n, n, n)
    volume = zeros(n, n, n)
    wrap(i) = periodic ? mod(i - 1, n) + 1 : clamp(i, 1, n)

    # per-cell face lists (a face borders two cells)
    nc = ncells(cs)
    faces = [Int[] for _ in 1:nc]
    for f in eachindex(geometry.nv)
        1 <= geometry.c1[f] <= nc && push!(faces[geometry.c1[f]], f)
        1 <= geometry.c2[f] <= nc && push!(faces[geometry.c2[f]], f)
    end
    offs = vcat(0, cumsum(geometry.nv))           # face f's verts: offs[f]+1 .. offs[f+1]

    V3(a, b, c) = R3D.Vec{3,Float64}((a, b, c))
    clipped_vol = Vector{Float64}(undef, nc)

    for c in 1:nc
        apex = V3(cs.pos[c, 1], cs.pos[c, 2], cs.pos[c, 3])
        vtot = 0.0
        ρ = cs.rho[c]; px = cs.mom[c, 1]; py = cs.mom[c, 2]; pz = cs.mom[c, 3]; e = cs.etot[c]
        for f in faces[c]
            o = offs[f]
            k = geometry.nv[f]
            # OUTWARD direction for THIS cell (the export's normal is side-1's)
            nout = V3(geometry.normals[f, 1], geometry.normals[f, 2], geometry.normals[f, 3])
            geometry.c2[f] == c && (nout = -nout)
            # ring orientation (Newell): traverse so the ring is outward-wound;
            # SIGNED fan tets then sum to the enclosed volume by the divergence
            # theorem — exact even for the wandering rings a degenerate
            # (lattice) Delaunay produces, where unsigned fans overlap.
            ringn = V3(0.0, 0.0, 0.0)
            for m in 1:k
                a = V3(geometry.verts[o + m, 1], geometry.verts[o + m, 2], geometry.verts[o + m, 3])
                b = V3(geometry.verts[o + mod1(m + 1, k), 1], geometry.verts[o + mod1(m + 1, k), 2],
                       geometry.verts[o + mod1(m + 1, k), 3])
                ringn += cross(a, b)
            end
            fwd = dot(ringn, nout) >= 0
            vat(m) = (mm = fwd ? m : k + 1 - m;
                      V3(geometry.verts[o + mm, 1] / L, geometry.verts[o + mm, 2] / L,
                         geometry.verts[o + mm, 3] / L))
            v1 = vat(1)
            for m in 2:(k - 1)
                t1, t2, t3, t4 = apex, v1, vat(m), vat(m + 1)
                vol6 = dot(cross(t2 - t1, t3 - t1), t4 - t1)
                abs(vol6) < 1e-300 && continue                # exactly degenerate
                s = vol6 > 0 ? 1.0 : -1.0                     # SIGNED contribution
                vol6 < 0 && ((t3, t4) = (t4, t3))             # positively oriented tet
                # Clip the TET by each grid cell's AXIS-ALIGNED planes (exactly
                # conditioned), never the cell by the tet's face planes — a
                # degenerate Delaunay (lattice ICs) makes swarms of near-flat
                # sliver tets whose face normals (crosses of nearly parallel
                # edges) are pure noise; a sliver TET polytope, by contrast,
                # simply clips to ≈ zero volume.
                xs = (t1[1], t2[1], t3[1], t4[1]); ys = (t1[2], t2[2], t3[2], t4[2])
                zs = (t1[3], t2[3], t3[3], t4[3])
                ilo = floor(Int, minimum(xs) / dxg); ihi = floor(Int, maximum(xs) / dxg)
                jlo = floor(Int, minimum(ys) / dxg); jhi = floor(Int, maximum(ys) / dxg)
                klo = floor(Int, minimum(zs) / dxg); khi = floor(Int, maximum(zs) / dxg)
                ex = V3(1.0, 0.0, 0.0); ey = V3(0.0, 1.0, 0.0); ez = V3(0.0, 0.0, 1.0)
                for kk in klo:khi, jj in jlo:jhi, ii in ilo:ihi
                    p = R3D.tet(t1, t2, t3, t4)
                    R3D.clip!(p, [R3D.Plane{3,Float64}(ex, -ii * dxg),
                                  R3D.Plane{3,Float64}(-ex, (ii + 1) * dxg),
                                  R3D.Plane{3,Float64}(ey, -jj * dxg),
                                  R3D.Plane{3,Float64}(-ey, (jj + 1) * dxg),
                                  R3D.Plane{3,Float64}(ez, -kk * dxg),
                                  R3D.Plane{3,Float64}(-ez, (kk + 1) * dxg)])
                    v = s * R3D.moments(p, 0)[1]
                    v == 0 && continue
                    gi = wrap(ii + 1); gj = wrap(jj + 1); gk = wrap(kk + 1)
                    mass[gi, gj, gk] += ρ * v
                    momx[gi, gj, gk] += px * v; momy[gi, gj, gk] += py * v
                    momz[gi, gj, gk] += pz * v
                    en[gi, gj, gk] += e * v
                    volume[gi, gj, gk] += v
                    vtot += v
                end
            end
        end
        clipped_vol[c] = vtot
    end

    # the exact-deposit guarantee: each cell's clipped volume IS its volume
    vol_geo = cs.vol .* volscale
    bad = maximum(abs.(clipped_vol .- vol_geo) ./ vol_geo)
    bad < 1e-8 ||
        error("deposit_exact: clipped volumes deviate from cell volumes (max rel $bad)")
    lg = ledger(cs)
    abs(sum(mass) - lg.mass * volscale) <= 1e-10 * max(abs(lg.mass * volscale), 1.0) ||
        error("deposit_exact: mass not conserved")

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
