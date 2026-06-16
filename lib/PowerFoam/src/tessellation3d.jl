# Production-tessellator facing API and debug schema.
#
# The first implementation intentionally wraps the existing local periodic
# halfspace rebuild.  The contract here is the seam that the Delaunay-backed
# KA port must satisfy before it replaces the current rebuild path.

struct TessellationReference3D{G,M1,M2,M3,K,O,D}
    geom::G
    center::M1
    face_center::M2
    face_image_shift::M3
    canonical_face_keys::K
    canonical_face_order::O
    algorithm::Symbol
    backend_residency::Symbol
    metadata::NamedTuple
    delaunay::D
end

struct DelaunayTetrahedra3D
    points::Matrix{Float64}
    original_index::Vector{Int}
    image_shift::Matrix{Int}
    tetras::Vector{NTuple{4,Int}}
    circumcenters::Matrix{Float64}
    counters::TessellationFallbackCounters3D
end

struct DelaunaySoA3D{I<:AbstractVector,R<:AbstractVector}
    point_x::R
    point_y::R
    point_z::R
    original_index::I
    image_sx::I
    image_sy::I
    image_sz::I
    tet_p1::I
    tet_p2::I
    tet_p3::I
    tet_p4::I
    circum_x::R
    circum_y::R
    circum_z::R
    circum_valid::I
end

struct PeriodicPointImages3D{I<:AbstractVector,R<:AbstractVector}
    point_x::R
    point_y::R
    point_z::R
    original_index::I
    image_sx::I
    image_sy::I
    image_sz::I
end

struct DenseCandidatePairs3D{I<:AbstractVector,R<:AbstractVector}
    source::I
    candidate::I
    image_sx::I
    image_sy::I
    image_sz::I
    bin_dx::I
    bin_dy::I
    bin_dz::I
    active::I
    distance2::R
end

struct CandidateStencil3D{I<:AbstractVector,R<:AbstractVector}
    counts::I
    candidate::I
    image_sx::I
    image_sy::I
    image_sz::I
    distance2::R
    max_candidates_per_source::Int
end

struct CandidateTetraPredicates3D{I<:AbstractVector,R<:AbstractVector}
    inside::I
    valid::I
    margin::R
    source_count::Int
    tetra_count::Int
    max_candidates_per_source::Int
end

struct CandidateConflictFaceRows3D{I<:AbstractVector}
    source::I
    slot::I
    tetra::I
    local_face::I
    face_v1::I
    face_v2::I
    face_v3::I
    active::I
    source_count::Int
    tetra_count::Int
    max_candidates_per_source::Int
end

struct CandidateBoundaryFaceRows3D{I<:AbstractVector}
    source::I
    slot::I
    tetra::I
    local_face::I
    face_v1::I
    face_v2::I
    face_v3::I
    boundary::I
    source_count::Int
    tetra_count::Int
    max_candidates_per_source::Int
end

struct CompactBoundaryFaces3D{I<:AbstractVector}
    counts::I
    source::I
    slot::I
    tetra::I
    local_face::I
    face_v1::I
    face_v2::I
    face_v3::I
    max_faces_per_candidate::Int
    source_count::Int
    max_candidates_per_source::Int
end

struct CompactFaceCandidates3D{I<:AbstractVector}
    counts::I
    c1::I
    c2::I
    image_sx::I
    image_sy::I
    image_sz::I
    slot::I
    tetra::I
    local_face::I
    face_v1::I
    face_v2::I
    face_v3::I
    max_faces_per_source::Int
    source_count::Int
end

struct CompactFaceCandidateCSR3D{I<:AbstractVector}
    counts::I
    offsets::I
    max_faces_per_source::Int
    source_count::Int
end

struct SourceOwnedFaceCSR3D{I<:AbstractVector}
    counts::I
    offsets::I
    faces::I
    signs::I
    max_faces_per_source::Int
    source_count::Int
end

struct ReciprocalFaceCandidatePairs3D{I<:AbstractVector}
    active::I
    pair_row::I
    canonical_row::I
    owner::I
    max_faces_per_source::Int
    source_count::Int
end

struct CompactCanonicalFaces3D{I<:AbstractVector}
    source_row::I
    c1::I
    c2::I
    image_sx::I
    image_sy::I
    image_sz::I
    tetra::I
    local_face::I
    face_v1::I
    face_v2::I
    face_v3::I
    source_count::Int
end

struct CompactCanonicalFaceCSR3D{F,C,O<:AbstractVector}
    compact::F
    counts::C
    offsets::O
    max_faces_per_source::Int
    source_count::Int
end

struct TessellationSoA3D{D,G,I<:AbstractVector,R<:AbstractVector}
    delaunay::D
    geom::G
    center_x::R
    center_y::R
    center_z::R
    face_center_x::R
    face_center_y::R
    face_center_z::R
    face_image_sx::I
    face_image_sy::I
    face_image_sz::I
end

const _DELAUNAY_FACE_VERTS3 = ((1, 2, 3), (1, 4, 2), (2, 4, 3), (3, 4, 1))

function _face_image_shift_or_zeros(nf, face_image_shift)
    face_image_shift === nothing || return Matrix{Int}(face_image_shift)
    return zeros(Int, nf, 3)
end

function _owner_vector(default, nf, owner)
    owner === nothing && return fill(default, nf)
    length(owner) == nf || error("owner vector length must match face count")
    return Int.(collect(owner))
end

"""
    canonical_face_keys_3d(geom; face_image_shift=nothing, owner_task=nothing,
                           owner_index=nothing)

Return stable sort keys for a compact 3-D face table.  The key includes the
unordered cell pair, periodic image shift, owner task/index, and original
orientation bit.  It is intentionally richer than the hydro table so CPU/GPU
and AREPO/PowerFoam comparisons can normalize row order without losing update
ownership information.
"""
function canonical_face_keys_3d(geom::ArepoMeshArrays3D;
                                face_image_shift = nothing,
                                owner_task = nothing,
                                owner_index = nothing)
    c1 = Int.(Array(geom.c1))
    c2 = Int.(Array(geom.c2))
    nf = length(c1)
    shifts = _face_image_shift_or_zeros(nf, face_image_shift)
    tasks = _owner_vector(0, nf, owner_task)
    owners = _owner_vector(0, nf, owner_index)
    keys = Vector{NTuple{9,Int}}(undef, nf)
    @inbounds for f in 1:nf
        a = c1[f]
        b = c2[f]
        lo, hi = a <= b ? (a, b) : (b, a)
        orientation = a <= b ? 1 : -1
        keys[f] = (lo, hi, shifts[f, 1], shifts[f, 2], shifts[f, 3],
                   tasks[f], owners[f], orientation, f)
    end
    return keys
end

canonical_face_order_3d(geom::ArepoMeshArrays3D; kwargs...) =
    sortperm(canonical_face_keys_3d(geom; kwargs...))

@kernel function _periodic_point_images_soa_kernel!(
    out_x, out_y, out_z, original_index, image_sx, image_sy, image_sz,
    point_x, point_y, point_z, n::Int, lx, ly, lz)
    row = @index(Global)
    total = 27 * n
    if row <= total
        i = mod(row - 1, n) + 1
        block = (row - 1) ÷ n
        sz = mod(block, 3) - 1
        sy = mod(block ÷ 3, 3) - 1
        sx = (block ÷ 9) - 1
        out_x[row] = point_x[i] + sx * lx
        out_y[row] = point_y[i] + sy * ly
        out_z[row] = point_z[i] + sz * lz
        original_index[row] = i
        image_sx[row] = sx
        image_sy[row] = sy
        image_sz[row] = sz
    end
end

function periodic_point_images_soa_3d(be, point_x::AbstractVector,
                                      point_y::AbstractVector,
                                      point_z::AbstractVector;
                                      domain = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                                      index_type::Type{<:Integer} = Int32)
    n = length(point_x)
    length(point_y) == n && length(point_z) == n ||
        error("point coordinate arrays must have equal length")
    T = eltype(point_x)
    total = 27n
    dom = _domain3(domain)
    lx = T(dom[1][2] - dom[1][1])
    ly = T(dom[2][2] - dom[2][1])
    lz = T(dom[3][2] - dom[3][1])
    out_x = _backend_zeros(be, T, total)
    out_y = _backend_zeros(be, T, total)
    out_z = _backend_zeros(be, T, total)
    original_index = _backend_zeros(be, index_type, total)
    image_sx = _backend_zeros(be, index_type, total)
    image_sy = _backend_zeros(be, index_type, total)
    image_sz = _backend_zeros(be, index_type, total)
    kernel = _periodic_point_images_soa_kernel!(be)
    event = kernel(out_x, out_y, out_z, original_index,
                   image_sx, image_sy, image_sz,
                   point_x, point_y, point_z, Int(n), lx, ly, lz;
                   ndrange = total)
    KA.synchronize(be)
    return PeriodicPointImages3D(out_x, out_y, out_z, original_index,
                                 image_sx, image_sy, image_sz)
end

function periodic_point_images_soa_3d(be, points::AbstractMatrix;
                                      domain = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                                      T::Type{<:AbstractFloat} = Float32,
                                      index_type::Type{<:Integer} = Int32)
    size(points, 2) == 3 || error("points must be n x 3")
    point_x = _backend_copy(be, view(points, :, 1), T)
    point_y = _backend_copy(be, view(points, :, 2), T)
    point_z = _backend_copy(be, view(points, :, 3), T)
    return periodic_point_images_soa_3d(be, point_x, point_y, point_z;
                                        domain, index_type)
end

@kernel function _dense_candidate_pairs_soa_kernel!(
    source, candidate, out_sx, out_sy, out_sz,
    bin_dx, bin_dy, bin_dz, active, distance2,
    src_x, src_y, src_z,
    img_x, img_y, img_z, img_original, img_sx, img_sy, img_sz,
    nsource::Int, nimages::Int, nb::Int, radius::Int,
    xmin, ymin, zmin, inv_dx, inv_dy, inv_dz)
    row = @index(Global)
    total = nsource * nimages
    if row <= total
        i = mod(row - 1, nsource) + 1
        img = (row - 1) ÷ nsource + 1
        j = Int(img_original[img])
        sx = Int(img_sx[img])
        sy = Int(img_sy[img])
        sz = Int(img_sz[img])

        bix = Int(floor((src_x[i] - xmin) * inv_dx)) + 1
        biy = Int(floor((src_y[i] - ymin) * inv_dy)) + 1
        biz = Int(floor((src_z[i] - zmin) * inv_dz)) + 1
        bjx = Int(floor((img_x[img] - xmin) * inv_dx)) + 1
        bjy = Int(floor((img_y[img] - ymin) * inv_dy)) + 1
        bjz = Int(floor((img_z[img] - zmin) * inv_dz)) + 1
        dxbin = bjx - bix
        dybin = bjy - biy
        dzbin = bjz - biz

        dx = img_x[img] - src_x[i]
        dy = img_y[img] - src_y[i]
        dz = img_z[img] - src_z[i]
        d2 = dx * dx + dy * dy + dz * dz
        self_image = i == j && sx == 0 && sy == 0 && sz == 0
        keep = !self_image &&
               abs(dxbin) <= radius &&
               abs(dybin) <= radius &&
               abs(dzbin) <= radius

        source[row] = i
        candidate[row] = j
        out_sx[row] = sx
        out_sy[row] = sy
        out_sz[row] = sz
        bin_dx[row] = dxbin
        bin_dy[row] = dybin
        bin_dz[row] = dzbin
        active[row] = keep ? one(eltype(active)) : zero(eltype(active))
        distance2[row] = d2
    end
end

function dense_candidate_pairs_soa_3d(be, point_x::AbstractVector,
                                      point_y::AbstractVector,
                                      point_z::AbstractVector,
                                      images::PeriodicPointImages3D;
                                      domain = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                                      bins_per_axis::Integer,
                                      search_radius::Integer = 1,
                                      index_type::Type{<:Integer} = Int32)
    n = length(point_x)
    length(point_y) == n && length(point_z) == n ||
        error("point coordinate arrays must have equal length")
    nimages = length(images.point_x)
    nimages % n == 0 || error("image count must be a multiple of source count")
    nb = Int(bins_per_axis)
    nb > 0 || error("bins_per_axis must be positive")
    radius = Int(search_radius)
    radius >= 0 || error("search_radius must be nonnegative")
    T = eltype(point_x)
    total = n * nimages
    dom = _domain3(domain)
    lx = T(dom[1][2] - dom[1][1])
    ly = T(dom[2][2] - dom[2][1])
    lz = T(dom[3][2] - dom[3][1])
    xmin = T(dom[1][1])
    ymin = T(dom[2][1])
    zmin = T(dom[3][1])
    inv_dx = T(nb) / lx
    inv_dy = T(nb) / ly
    inv_dz = T(nb) / lz

    source = _backend_zeros(be, index_type, total)
    candidate = _backend_zeros(be, index_type, total)
    image_sx = _backend_zeros(be, index_type, total)
    image_sy = _backend_zeros(be, index_type, total)
    image_sz = _backend_zeros(be, index_type, total)
    bin_dx = _backend_zeros(be, index_type, total)
    bin_dy = _backend_zeros(be, index_type, total)
    bin_dz = _backend_zeros(be, index_type, total)
    active = _backend_zeros(be, index_type, total)
    distance2 = _backend_zeros(be, T, total)
    kernel = _dense_candidate_pairs_soa_kernel!(be)
    event = kernel(source, candidate, image_sx, image_sy, image_sz,
                   bin_dx, bin_dy, bin_dz, active, distance2,
                   point_x, point_y, point_z,
                   images.point_x, images.point_y, images.point_z,
                   images.original_index, images.image_sx, images.image_sy,
                   images.image_sz,
                   Int(n), Int(nimages), nb, radius,
                   xmin, ymin, zmin, inv_dx, inv_dy, inv_dz;
                   ndrange = total)
    KA.synchronize(be)
    return DenseCandidatePairs3D(source, candidate, image_sx, image_sy,
                                 image_sz, bin_dx, bin_dy, bin_dz, active,
                                 distance2)
end

function dense_candidate_pairs_soa_3d(be, points::AbstractMatrix;
                                      domain = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                                      bins_per_axis::Integer,
                                      search_radius::Integer = 1,
                                      T::Type{<:AbstractFloat} = Float32,
                                      index_type::Type{<:Integer} = Int32)
    size(points, 2) == 3 || error("points must be n x 3")
    point_x = _backend_copy(be, view(points, :, 1), T)
    point_y = _backend_copy(be, view(points, :, 2), T)
    point_z = _backend_copy(be, view(points, :, 3), T)
    images = periodic_point_images_soa_3d(be, point_x, point_y, point_z;
                                          domain, index_type)
    return dense_candidate_pairs_soa_3d(be, point_x, point_y, point_z,
                                        images; domain, bins_per_axis,
                                        search_radius, index_type)
end

@kernel function _pack_candidate_stencil_soa_kernel!(
    counts, out_candidate, out_sx, out_sy, out_sz, out_distance2,
    dense_source, dense_candidate, dense_sx, dense_sy, dense_sz,
    dense_active, dense_distance2,
    nsource::Int, nimages::Int, max_candidates::Int)
    i = @index(Global)
    if i <= nsource
        count = 0
        for img in 1:nimages
            row = (img - 1) * nsource + i
            if dense_active[row] != 0
                count += 1
                if count <= max_candidates
                    out = (i - 1) * max_candidates + count
                    out_candidate[out] = dense_candidate[row]
                    out_sx[out] = dense_sx[row]
                    out_sy[out] = dense_sy[row]
                    out_sz[out] = dense_sz[row]
                    out_distance2[out] = dense_distance2[row]
                end
            end
        end
        counts[i] = count
    end
end

function pack_candidate_stencil_soa_3d(be, dense::DenseCandidatePairs3D,
                                       nsource::Integer;
                                       max_candidates_per_source::Integer)
    nsrc = Int(nsource)
    length(dense.source) % nsrc == 0 ||
        error("dense candidate row count must be divisible by source count")
    nimages = length(dense.source) ÷ nsrc
    maxc = Int(max_candidates_per_source)
    maxc > 0 || error("max_candidates_per_source must be positive")
    I = eltype(dense.source)
    T = eltype(dense.distance2)
    total = nsrc * maxc
    counts = _backend_zeros(be, I, nsrc)
    candidate = _backend_zeros(be, I, total)
    image_sx = _backend_zeros(be, I, total)
    image_sy = _backend_zeros(be, I, total)
    image_sz = _backend_zeros(be, I, total)
    distance2 = _backend_zeros(be, T, total)
    kernel = _pack_candidate_stencil_soa_kernel!(be)
    event = kernel(counts, candidate, image_sx, image_sy, image_sz,
                   distance2,
                   dense.source, dense.candidate, dense.image_sx,
                   dense.image_sy, dense.image_sz, dense.active,
                   dense.distance2,
                   nsrc, nimages, maxc; ndrange = nsrc)
    KA.synchronize(be)
    return CandidateStencil3D(counts, candidate, image_sx, image_sy, image_sz,
                              distance2, maxc)
end

@kernel function _candidate_tetra_predicates_soa_kernel!(
    inside, valid, margin,
    point_x, point_y, point_z,
    counts, stencil_candidate, stencil_sx, stencil_sy, stencil_sz,
    tet_p1, circum_x, circum_y, circum_z, circum_valid,
    delaunay_point_x, delaunay_point_y, delaunay_point_z,
    nsource::Int, max_candidates::Int, ntetra::Int,
    lx, ly, lz, tol)
    row = @index(Global)
    total = nsource * max_candidates * ntetra
    if row <= total
        linear = row - 1
        per_tetra = nsource * max_candidates
        t = linear ÷ per_tetra + 1
        rem = linear - (t - 1) * per_tetra
        slot = rem ÷ nsource + 1
        source = rem - (slot - 1) * nsource + 1
        stencil_idx = (source - 1) * max_candidates + slot

        isvalid = slot <= Int(counts[source]) && circum_valid[t] != 0
        if isvalid
            c = Int(stencil_candidate[stencil_idx])
            sx = stencil_sx[stencil_idx]
            sy = stencil_sy[stencil_idx]
            sz = stencil_sz[stencil_idx]
            px = point_x[c] + sx * lx
            py = point_y[c] + sy * ly
            pz = point_z[c] + sz * lz

            cx = circum_x[t]
            cy = circum_y[t]
            cz = circum_z[t]
            p1 = Int(tet_p1[t])
            dx1 = delaunay_point_x[p1] - cx
            dy1 = delaunay_point_y[p1] - cy
            dz1 = delaunay_point_z[p1] - cz
            r2 = dx1 * dx1 + dy1 * dy1 + dz1 * dz1
            dx = px - cx
            dy = py - cy
            dz = pz - cz
            m = r2 - (dx * dx + dy * dy + dz * dz)
            margin[row] = m
            inside[row] = m >= -tol ? one(eltype(inside)) : zero(eltype(inside))
            valid[row] = one(eltype(valid))
        else
            margin[row] = zero(eltype(margin))
            inside[row] = zero(eltype(inside))
            valid[row] = zero(eltype(valid))
        end
    end
end

function candidate_tetra_predicates_soa_3d(be, point_x::AbstractVector,
                                           point_y::AbstractVector,
                                           point_z::AbstractVector,
                                           stencil::CandidateStencil3D,
                                           delaunay::DelaunaySoA3D;
                                           domain = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                                           tol::Real = 1e-10,
                                           index_type::Type{<:Integer} = eltype(stencil.counts))
    nsource = length(point_x)
    length(point_y) == nsource && length(point_z) == nsource ||
        error("point coordinate arrays must have equal length")
    length(stencil.counts) == nsource ||
        error("candidate stencil source count must match point arrays")
    maxc = Int(stencil.max_candidates_per_source)
    ntetra = length(delaunay.tet_p1)
    T = eltype(point_x)
    total = nsource * maxc * ntetra
    dom = _domain3(domain)
    lx = T(dom[1][2] - dom[1][1])
    ly = T(dom[2][2] - dom[2][1])
    lz = T(dom[3][2] - dom[3][1])
    inside = _backend_zeros(be, index_type, total)
    valid = _backend_zeros(be, index_type, total)
    margin = _backend_zeros(be, T, total)
    kernel = _candidate_tetra_predicates_soa_kernel!(be)
    event = kernel(inside, valid, margin,
                   point_x, point_y, point_z,
                   stencil.counts, stencil.candidate, stencil.image_sx,
                   stencil.image_sy, stencil.image_sz,
                   delaunay.tet_p1, delaunay.circum_x, delaunay.circum_y,
                   delaunay.circum_z, delaunay.circum_valid,
                   delaunay.point_x, delaunay.point_y, delaunay.point_z,
                   Int(nsource), maxc, Int(ntetra), lx, ly, lz, T(tol);
                   ndrange = total)
    KA.synchronize(be)
    return CandidateTetraPredicates3D(inside, valid, margin, Int(nsource),
                                      Int(ntetra), maxc)
end

function candidate_tetra_predicates_soa_3d(be, points::AbstractMatrix,
                                           stencil::CandidateStencil3D,
                                           delaunay::DelaunaySoA3D;
                                           domain = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                                           T::Type{<:AbstractFloat} = Float32,
                                           index_type::Type{<:Integer} = eltype(stencil.counts),
                                           tol::Real = 1e-10)
    size(points, 2) == 3 || error("points must be n x 3")
    point_x = _backend_copy(be, view(points, :, 1), T)
    point_y = _backend_copy(be, view(points, :, 2), T)
    point_z = _backend_copy(be, view(points, :, 3), T)
    return candidate_tetra_predicates_soa_3d(be, point_x, point_y, point_z,
                                             stencil, delaunay; domain, tol,
                                             index_type)
end

@inline function _sort3_ids(a, b, c)
    lo = min(min(a, b), c)
    hi = max(max(a, b), c)
    mid = a + b + c - lo - hi
    return lo, mid, hi
end

@kernel function _candidate_conflict_face_rows_soa_kernel!(
    out_source, out_slot, out_tetra, out_local_face,
    out_v1, out_v2, out_v3, out_active,
    pred_inside, pred_valid,
    tet_p1, tet_p2, tet_p3, tet_p4,
    nsource::Int, max_candidates::Int, ntetra::Int)
    row = @index(Global)
    total = nsource * max_candidates * ntetra * 4
    if row <= total
        linear = row - 1
        predicate_span = nsource * max_candidates
        faces_per_tetra = predicate_span * 4
        t = linear ÷ faces_per_tetra + 1
        rem_t = linear - (t - 1) * faces_per_tetra
        local_face = rem_t ÷ predicate_span + 1
        rem_f = rem_t - (local_face - 1) * predicate_span
        slot = rem_f ÷ nsource + 1
        source = rem_f - (slot - 1) * nsource + 1
        pred_row = (t - 1) * predicate_span + (slot - 1) * nsource + source

        a = tet_p1[t]
        b = tet_p2[t]
        c = tet_p3[t]
        d = tet_p4[t]
        f1 = a
        f2 = b
        f3 = c
        if local_face == 2
            f1 = a; f2 = d; f3 = b
        elseif local_face == 3
            f1 = b; f2 = d; f3 = c
        elseif local_face == 4
            f1 = c; f2 = d; f3 = a
        end
        s1, s2, s3 = _sort3_ids(f1, f2, f3)

        out_source[row] = source
        out_slot[row] = slot
        out_tetra[row] = t
        out_local_face[row] = local_face
        out_v1[row] = s1
        out_v2[row] = s2
        out_v3[row] = s3
        out_active[row] = (pred_valid[pred_row] != 0 && pred_inside[pred_row] != 0) ?
                          one(eltype(out_active)) : zero(eltype(out_active))
    end
end

function candidate_conflict_face_rows_soa_3d(be,
                                             predicates::CandidateTetraPredicates3D,
                                             delaunay::DelaunaySoA3D;
                                             index_type::Type{<:Integer} = eltype(predicates.inside))
    nsource = predicates.source_count
    maxc = predicates.max_candidates_per_source
    ntetra = predicates.tetra_count
    ntetra == length(delaunay.tet_p1) ||
        error("predicate tetra count must match Delaunay tetra count")
    total = nsource * maxc * ntetra * 4
    source = _backend_zeros(be, index_type, total)
    slot = _backend_zeros(be, index_type, total)
    tetra = _backend_zeros(be, index_type, total)
    local_face = _backend_zeros(be, index_type, total)
    face_v1 = _backend_zeros(be, index_type, total)
    face_v2 = _backend_zeros(be, index_type, total)
    face_v3 = _backend_zeros(be, index_type, total)
    active = _backend_zeros(be, index_type, total)
    kernel = _candidate_conflict_face_rows_soa_kernel!(be)
    event = kernel(source, slot, tetra, local_face, face_v1, face_v2,
                   face_v3, active,
                   predicates.inside, predicates.valid,
                   delaunay.tet_p1, delaunay.tet_p2, delaunay.tet_p3,
                   delaunay.tet_p4,
                   Int(nsource), Int(maxc), Int(ntetra); ndrange = total)
    KA.synchronize(be)
    return CandidateConflictFaceRows3D(source, slot, tetra, local_face,
                                       face_v1, face_v2, face_v3, active,
                                       Int(nsource), Int(ntetra), Int(maxc))
end

@kernel function _candidate_boundary_face_rows_soa_kernel!(
    out_boundary,
    source, slot, tetra, local_face, face_v1, face_v2, face_v3, active,
    nsource::Int, max_candidates::Int, ntetra::Int, nrows::Int)
    row = @index(Global)
    if row <= nrows
        keep = active[row] != 0
        if keep
            s = source[row]
            q = slot[row]
            v1 = face_v1[row]
            v2 = face_v2[row]
            v3 = face_v3[row]
            duplicate_count = 0
            predicate_span = nsource * max_candidates
            for ot in 1:ntetra
                for olf in 1:4
                    other = (ot - 1) * predicate_span * 4 +
                            (olf - 1) * predicate_span +
                            (Int(q) - 1) * nsource + Int(s)
                    if active[other] != 0 &&
                   face_v1[other] == v1 &&
                   face_v2[other] == v2 &&
                   face_v3[other] == v3
                        duplicate_count += 1
                    end
                end
            end
            keep = duplicate_count == 1
        end
        out_boundary[row] = keep ? one(eltype(out_boundary)) : zero(eltype(out_boundary))
    end
end

function candidate_boundary_face_rows_soa_3d(be,
                                             conflict::CandidateConflictFaceRows3D;
                                             index_type::Type{<:Integer} = eltype(conflict.active))
    nrows = length(conflict.active)
    boundary = _backend_zeros(be, index_type, nrows)
    kernel = _candidate_boundary_face_rows_soa_kernel!(be)
    event = kernel(boundary,
                   conflict.source, conflict.slot, conflict.tetra,
                   conflict.local_face, conflict.face_v1, conflict.face_v2,
                   conflict.face_v3, conflict.active,
                   Int(conflict.source_count),
                   Int(conflict.max_candidates_per_source),
                   Int(conflict.tetra_count), Int(nrows); ndrange = nrows)
    KA.synchronize(be)
    return CandidateBoundaryFaceRows3D(conflict.source, conflict.slot,
                                       conflict.tetra, conflict.local_face,
                                       conflict.face_v1, conflict.face_v2,
                                       conflict.face_v3, boundary,
                                       conflict.source_count,
                                       conflict.tetra_count,
                                       conflict.max_candidates_per_source)
end

@kernel function _pack_boundary_faces_soa_kernel!(
    counts, out_source, out_slot, out_tetra, out_local_face,
    out_v1, out_v2, out_v3,
    row_source, row_slot, row_tetra, row_local_face,
    row_v1, row_v2, row_v3, row_boundary,
    nsource::Int, max_candidates::Int, ntetra::Int, max_faces::Int)
    idx = @index(Global)
    total_candidates = nsource * max_candidates
    if idx <= total_candidates
        source = mod(idx - 1, nsource) + 1
        slot = (idx - 1) ÷ nsource + 1
        count = 0
        predicate_span = nsource * max_candidates
        for t in 1:ntetra
            for lf in 1:4
                row = (t - 1) * predicate_span * 4 +
                      (lf - 1) * predicate_span +
                      (slot - 1) * nsource + source
                if row_boundary[row] != 0
                    count += 1
                    if count <= max_faces
                        out = (idx - 1) * max_faces + count
                        out_source[out] = row_source[row]
                        out_slot[out] = row_slot[row]
                        out_tetra[out] = row_tetra[row]
                        out_local_face[out] = row_local_face[row]
                        out_v1[out] = row_v1[row]
                        out_v2[out] = row_v2[row]
                        out_v3[out] = row_v3[row]
                    end
                end
            end
        end
        counts[idx] = count
    end
end

function pack_boundary_faces_soa_3d(be, boundary::CandidateBoundaryFaceRows3D;
                                    max_faces_per_candidate::Integer,
                                    index_type::Type{<:Integer} = eltype(boundary.boundary))
    nsource = Int(boundary.source_count)
    maxc = Int(boundary.max_candidates_per_source)
    ntetra = Int(boundary.tetra_count)
    maxf = Int(max_faces_per_candidate)
    maxf > 0 || error("max_faces_per_candidate must be positive")
    total_candidates = nsource * maxc
    total = total_candidates * maxf
    counts = _backend_zeros(be, index_type, total_candidates)
    source = _backend_zeros(be, index_type, total)
    slot = _backend_zeros(be, index_type, total)
    tetra = _backend_zeros(be, index_type, total)
    local_face = _backend_zeros(be, index_type, total)
    face_v1 = _backend_zeros(be, index_type, total)
    face_v2 = _backend_zeros(be, index_type, total)
    face_v3 = _backend_zeros(be, index_type, total)
    kernel = _pack_boundary_faces_soa_kernel!(be)
    event = kernel(counts, source, slot, tetra, local_face,
                   face_v1, face_v2, face_v3,
                   boundary.source, boundary.slot, boundary.tetra,
                   boundary.local_face, boundary.face_v1, boundary.face_v2,
                   boundary.face_v3, boundary.boundary,
                   nsource, maxc, ntetra, maxf;
                   ndrange = total_candidates)
    KA.synchronize(be)
    return CompactBoundaryFaces3D(counts, source, slot, tetra, local_face,
                                  face_v1, face_v2, face_v3, maxf, nsource,
                                  maxc)
end

@kernel function _compact_face_candidates_soa_kernel!(
    counts, out_c1, out_c2, out_sx, out_sy, out_sz, out_slot,
    out_tetra, out_local_face, out_v1, out_v2, out_v3,
    boundary_counts, boundary_slot, boundary_tetra, boundary_local_face,
    boundary_v1, boundary_v2, boundary_v3,
    stencil_candidate, stencil_sx, stencil_sy, stencil_sz,
    nsource::Int, max_candidates::Int, max_faces_per_candidate::Int,
    max_faces_per_source::Int)
    source = @index(Global)
    if source <= nsource
        count = 0
        for slot in 1:max_candidates
            candidate_idx = (source - 1) * max_candidates + slot
            nfaces = Int(boundary_counts[candidate_idx])
            for q in 1:nfaces
                count += 1
                if count <= max_faces_per_source
                    inrow = (candidate_idx - 1) * max_faces_per_candidate + q
                    out = (source - 1) * max_faces_per_source + count
                    out_c1[out] = source
                    out_c2[out] = stencil_candidate[candidate_idx]
                    out_sx[out] = stencil_sx[candidate_idx]
                    out_sy[out] = stencil_sy[candidate_idx]
                    out_sz[out] = stencil_sz[candidate_idx]
                    out_slot[out] = boundary_slot[inrow]
                    out_tetra[out] = boundary_tetra[inrow]
                    out_local_face[out] = boundary_local_face[inrow]
                    out_v1[out] = boundary_v1[inrow]
                    out_v2[out] = boundary_v2[inrow]
                    out_v3[out] = boundary_v3[inrow]
                end
            end
        end
        counts[source] = count
    end
end

function compact_face_candidates_soa_3d(be,
                                        boundary::CompactBoundaryFaces3D,
                                        stencil::CandidateStencil3D;
                                        max_faces_per_source::Integer,
                                        index_type::Type{<:Integer} = eltype(boundary.counts))
    nsource = Int(boundary.source_count)
    maxc = Int(boundary.max_candidates_per_source)
    length(stencil.counts) == nsource ||
        error("candidate stencil source count must match boundary faces")
    stencil.max_candidates_per_source == maxc ||
        error("candidate stencil max-candidate stride must match boundary faces")
    maxf = Int(boundary.max_faces_per_candidate)
    maxfs = Int(max_faces_per_source)
    maxfs > 0 || error("max_faces_per_source must be positive")
    total = nsource * maxfs
    counts = _backend_zeros(be, index_type, nsource)
    c1 = _backend_zeros(be, index_type, total)
    c2 = _backend_zeros(be, index_type, total)
    image_sx = _backend_zeros(be, index_type, total)
    image_sy = _backend_zeros(be, index_type, total)
    image_sz = _backend_zeros(be, index_type, total)
    slot = _backend_zeros(be, index_type, total)
    tetra = _backend_zeros(be, index_type, total)
    local_face = _backend_zeros(be, index_type, total)
    face_v1 = _backend_zeros(be, index_type, total)
    face_v2 = _backend_zeros(be, index_type, total)
    face_v3 = _backend_zeros(be, index_type, total)
    kernel = _compact_face_candidates_soa_kernel!(be)
    event = kernel(counts, c1, c2, image_sx, image_sy, image_sz, slot,
                   tetra, local_face, face_v1, face_v2, face_v3,
                   boundary.counts, boundary.slot, boundary.tetra,
                   boundary.local_face, boundary.face_v1, boundary.face_v2,
                   boundary.face_v3,
                   stencil.candidate, stencil.image_sx, stencil.image_sy,
                   stencil.image_sz,
                   nsource, maxc, maxf, maxfs; ndrange = nsource)
    KA.synchronize(be)
    return CompactFaceCandidates3D(counts, c1, c2, image_sx, image_sy,
                                   image_sz, slot, tetra, local_face, face_v1,
                                   face_v2, face_v3, maxfs, nsource)
end

@kernel function _compact_face_candidate_offsets_soa_kernel!(
    offsets, counts, nsource::Int, max_faces_per_source::Int)
    i = @index(Global)
    if i <= nsource + 1
        offsets[i] = (i - 1) * max_faces_per_source + 1
    end
end

function compact_face_candidate_csr_soa_3d(be,
                                           faces::CompactFaceCandidates3D;
                                           index_type::Type{<:Integer} = eltype(faces.counts))
    nsource = Int(faces.source_count)
    maxfs = Int(faces.max_faces_per_source)
    offsets = _backend_zeros(be, index_type, nsource + 1)
    kernel = _compact_face_candidate_offsets_soa_kernel!(be)
    event = kernel(offsets, faces.counts, nsource, maxfs; ndrange = nsource + 1)
    KA.synchronize(be)
    return CompactFaceCandidateCSR3D(faces.counts, offsets, maxfs, nsource)
end

@kernel function _source_owned_face_csr_soa_kernel!(
    offsets, out_faces, out_signs, counts,
    nsource::Int, max_faces_per_source::Int)
    row = @index(Global)
    total = nsource * max_faces_per_source
    if row <= nsource + 1
        offsets[row] = (row - 1) * max_faces_per_source + 1
    end
    if row <= total
        source = (row - 1) ÷ max_faces_per_source + 1
        local_face_index = row - (source - 1) * max_faces_per_source
        if local_face_index <= Int(counts[source])
            out_faces[row] = row
            out_signs[row] = -one(eltype(out_signs))
        else
            out_faces[row] = row
            out_signs[row] = zero(eltype(out_signs))
        end
    end
end

function source_owned_face_csr_soa_3d(be, faces::CompactFaceCandidates3D;
                                      index_type::Type{<:Integer} = eltype(faces.counts))
    nsource = Int(faces.source_count)
    maxfs = Int(faces.max_faces_per_source)
    total = nsource * maxfs
    offsets = _backend_zeros(be, index_type, nsource + 1)
    out_faces = _backend_zeros(be, index_type, total)
    out_signs = _backend_zeros(be, index_type, total)
    kernel = _source_owned_face_csr_soa_kernel!(be)
    event = kernel(offsets, out_faces, out_signs, faces.counts,
                   nsource, maxfs; ndrange = max(total, nsource + 1))
    KA.synchronize(be)
    return SourceOwnedFaceCSR3D(faces.counts, offsets, out_faces, out_signs,
                                maxfs, nsource)
end

@kernel function _reciprocal_face_candidate_pairs_soa_kernel!(
    out_active, out_pair, out_canonical, out_owner,
    counts, c1, c2, sx, sy, sz, v1, v2, v3,
    nsource::Int, max_faces_per_source::Int)
    row = @index(Global)
    total = nsource * max_faces_per_source
    if row <= total
        source = (row - 1) ÷ max_faces_per_source + 1
        local_face_index = row - (source - 1) * max_faces_per_source
        is_active = local_face_index <= Int(counts[source])
        if is_active
            other = zero(eltype(out_pair))
            ci = Int(c1[row])
            cj = Int(c2[row])
            six = Int(sx[row])
            siy = Int(sy[row])
            siz = Int(sz[row])
            a = Int(v1[row])
            b = Int(v2[row])
            c = Int(v3[row])
            for candidate_row in 1:total
                candidate_source = (candidate_row - 1) ÷ max_faces_per_source + 1
                candidate_local = candidate_row -
                                  (candidate_source - 1) * max_faces_per_source
                if candidate_local <= Int(counts[candidate_source]) &&
                   Int(c1[candidate_row]) == cj &&
                   Int(c2[candidate_row]) == ci &&
                   Int(sx[candidate_row]) == -six &&
                   Int(sy[candidate_row]) == -siy &&
                   Int(sz[candidate_row]) == -siz &&
                   Int(v1[candidate_row]) == a &&
                   Int(v2[candidate_row]) == b &&
                   Int(v3[candidate_row]) == c
                    other = eltype(out_pair)(candidate_row)
                    break
                end
            end
            canonical = other == zero(eltype(out_pair)) || row <= Int(other) ?
                        row : Int(other)
            out_active[row] = one(eltype(out_active))
            out_pair[row] = other
            out_canonical[row] = eltype(out_canonical)(canonical)
            out_owner[row] = row == canonical ? one(eltype(out_owner)) :
                             zero(eltype(out_owner))
        else
            out_active[row] = zero(eltype(out_active))
            out_pair[row] = zero(eltype(out_pair))
            out_canonical[row] = zero(eltype(out_canonical))
            out_owner[row] = zero(eltype(out_owner))
        end
    end
end

function reciprocal_face_candidate_pairs_soa_3d(be,
                                                faces::CompactFaceCandidates3D;
                                                index_type::Type{<:Integer} = eltype(faces.counts))
    nsource = Int(faces.source_count)
    maxfs = Int(faces.max_faces_per_source)
    total = nsource * maxfs
    active = _backend_zeros(be, index_type, total)
    pair_row = _backend_zeros(be, index_type, total)
    canonical_row = _backend_zeros(be, index_type, total)
    owner = _backend_zeros(be, index_type, total)
    kernel = _reciprocal_face_candidate_pairs_soa_kernel!(be)
    event = kernel(active, pair_row, canonical_row, owner,
                   faces.counts, faces.c1, faces.c2, faces.image_sx,
                   faces.image_sy, faces.image_sz, faces.face_v1,
                   faces.face_v2, faces.face_v3, nsource, maxfs;
                   ndrange = total)
    KA.synchronize(be)
    return ReciprocalFaceCandidatePairs3D(active, pair_row, canonical_row,
                                          owner, maxfs, nsource)
end

@kernel function _canonical_face_candidate_csr_soa_kernel!(
    offsets, out_faces, out_signs, counts, pair_row, canonical_row, owner,
    nsource::Int, max_faces_per_source::Int)
    row = @index(Global)
    total = nsource * max_faces_per_source
    if row <= nsource + 1
        offsets[row] = (row - 1) * max_faces_per_source + 1
    end
    if row <= total
        source = (row - 1) ÷ max_faces_per_source + 1
        local_face_index = row - (source - 1) * max_faces_per_source
        if local_face_index <= Int(counts[source])
            out_faces[row] = canonical_row[row]
            if pair_row[row] == zero(eltype(pair_row)) || owner[row] != 0
                out_signs[row] = -one(eltype(out_signs))
            else
                out_signs[row] = one(eltype(out_signs))
            end
        else
            out_faces[row] = row
            out_signs[row] = zero(eltype(out_signs))
        end
    end
end

function canonical_face_candidate_csr_soa_3d(be,
                                             faces::CompactFaceCandidates3D,
                                             pairs::ReciprocalFaceCandidatePairs3D =
                                                 reciprocal_face_candidate_pairs_soa_3d(be, faces);
                                             index_type::Type{<:Integer} = eltype(faces.counts))
    nsource = Int(faces.source_count)
    maxfs = Int(faces.max_faces_per_source)
    pairs.source_count == nsource ||
        error("pair source count must match face candidates")
    pairs.max_faces_per_source == maxfs ||
        error("pair stride must match face candidates")
    total = nsource * maxfs
    offsets = _backend_zeros(be, index_type, nsource + 1)
    out_faces = _backend_zeros(be, index_type, total)
    out_signs = _backend_zeros(be, index_type, total)
    kernel = _canonical_face_candidate_csr_soa_kernel!(be)
    event = kernel(offsets, out_faces, out_signs, faces.counts,
                   pairs.pair_row, pairs.canonical_row, pairs.owner,
                   nsource, maxfs; ndrange = max(total, nsource + 1))
    KA.synchronize(be)
    return SourceOwnedFaceCSR3D(faces.counts, offsets, out_faces, out_signs,
                                maxfs, nsource)
end

@kernel function _compact_face_candidate_mesh_soa_kernel!(
    out_c1, out_c2, out_area, out_nx, out_ny, out_nz, out_vx, out_vy, out_vz,
    face_c1, face_c2, counts,
    default_area, nx, ny, nz, vx, vy, vz,
    nsource::Int, max_faces_per_source::Int)
    row = @index(Global)
    total = nsource * max_faces_per_source
    if row <= total
        source = (row - 1) ÷ max_faces_per_source + 1
        local_face_index = row - (source - 1) * max_faces_per_source
        if local_face_index <= Int(counts[source])
            out_c1[row] = face_c1[row]
            out_c2[row] = face_c2[row]
            out_area[row] = default_area
            out_nx[row] = nx
            out_ny[row] = ny
            out_nz[row] = nz
            out_vx[row] = vx
            out_vy[row] = vy
            out_vz[row] = vz
        else
            out_c1[row] = source
            out_c2[row] = zero(eltype(out_c2))
            out_area[row] = zero(eltype(out_area))
            out_nx[row] = zero(eltype(out_nx))
            out_ny[row] = zero(eltype(out_ny))
            out_nz[row] = zero(eltype(out_nz))
            out_vx[row] = zero(eltype(out_vx))
            out_vy[row] = zero(eltype(out_vy))
            out_vz[row] = zero(eltype(out_vz))
        end
    end
end

function compact_face_candidate_mesh_arrays_3d(be, faces::CompactFaceCandidates3D;
                                               volume = nothing,
                                               default_face_area::Real = 1.0,
                                               default_normal = (1.0, 0.0, 0.0),
                                               default_face_velocity = (0.0, 0.0, 0.0),
                                               T::Type{<:AbstractFloat} = Float64,
                                               index_type::Type{<:Integer} = eltype(faces.counts))
    nsource = Int(faces.source_count)
    maxfs = Int(faces.max_faces_per_source)
    total = nsource * maxfs
    c1 = _backend_zeros(be, index_type, total)
    c2 = _backend_zeros(be, index_type, total)
    area = _backend_zeros(be, T, total)
    nx = _backend_zeros(be, T, total)
    ny = _backend_zeros(be, T, total)
    nz = _backend_zeros(be, T, total)
    vx = _backend_zeros(be, T, total)
    vy = _backend_zeros(be, T, total)
    vz = _backend_zeros(be, T, total)
    kernel = _compact_face_candidate_mesh_soa_kernel!(be)
    event = kernel(c1, c2, area, nx, ny, nz, vx, vy, vz,
                   faces.c1, faces.c2, faces.counts,
                   T(default_face_area), T(default_normal[1]),
                   T(default_normal[2]), T(default_normal[3]),
                   T(default_face_velocity[1]), T(default_face_velocity[2]),
                   T(default_face_velocity[3]),
                   nsource, maxfs; ndrange = total)
    KA.synchronize(be)
    csr = source_owned_face_csr_soa_3d(be, faces; index_type)
    vol = volume === nothing ? _backend_copy(be, fill(T(1 / max(nsource, 1)), nsource), T) :
          _backend_copy(be, volume, T)
    length(vol) == nsource || error("volume length must match source count")
    return ArepoMeshArrays3D(c1, c2, csr.offsets, csr.faces, csr.signs, vol,
                             area, nx, ny, nz, vx, vy, vz)
end

@kernel function _canonical_face_candidate_mesh_soa_kernel!(
    out_c1, out_c2, out_area, out_nx, out_ny, out_nz, out_vx, out_vy, out_vz,
    face_c1, face_c2, counts, owner,
    default_area, nx, ny, nz, vx, vy, vz,
    nsource::Int, max_faces_per_source::Int)
    row = @index(Global)
    total = nsource * max_faces_per_source
    if row <= total
        source = (row - 1) ÷ max_faces_per_source + 1
        local_face_index = row - (source - 1) * max_faces_per_source
        if local_face_index <= Int(counts[source]) && owner[row] != 0
            out_c1[row] = face_c1[row]
            out_c2[row] = face_c2[row]
            out_area[row] = default_area
            out_nx[row] = nx
            out_ny[row] = ny
            out_nz[row] = nz
            out_vx[row] = vx
            out_vy[row] = vy
            out_vz[row] = vz
        else
            out_c1[row] = source
            out_c2[row] = zero(eltype(out_c2))
            out_area[row] = zero(eltype(out_area))
            out_nx[row] = zero(eltype(out_nx))
            out_ny[row] = zero(eltype(out_ny))
            out_nz[row] = zero(eltype(out_nz))
            out_vx[row] = zero(eltype(out_vx))
            out_vy[row] = zero(eltype(out_vy))
            out_vz[row] = zero(eltype(out_vz))
        end
    end
end

function canonical_face_candidate_mesh_arrays_3d(be,
                                                 faces::CompactFaceCandidates3D;
                                                 pairs::Union{Nothing,ReciprocalFaceCandidatePairs3D} = nothing,
                                                 volume = nothing,
                                                 default_face_area::Real = 1.0,
                                                 default_normal = (1.0, 0.0, 0.0),
                                                 default_face_velocity = (0.0, 0.0, 0.0),
                                                 T::Type{<:AbstractFloat} = Float64,
                                                 index_type::Type{<:Integer} = eltype(faces.counts))
    nsource = Int(faces.source_count)
    maxfs = Int(faces.max_faces_per_source)
    total = nsource * maxfs
    pair_data = pairs === nothing ?
                reciprocal_face_candidate_pairs_soa_3d(be, faces; index_type) :
                pairs
    pair_data.source_count == nsource ||
        error("pair source count must match face candidates")
    pair_data.max_faces_per_source == maxfs ||
        error("pair stride must match face candidates")
    c1 = _backend_zeros(be, index_type, total)
    c2 = _backend_zeros(be, index_type, total)
    area = _backend_zeros(be, T, total)
    nx = _backend_zeros(be, T, total)
    ny = _backend_zeros(be, T, total)
    nz = _backend_zeros(be, T, total)
    vx = _backend_zeros(be, T, total)
    vy = _backend_zeros(be, T, total)
    vz = _backend_zeros(be, T, total)
    kernel = _canonical_face_candidate_mesh_soa_kernel!(be)
    event = kernel(c1, c2, area, nx, ny, nz, vx, vy, vz,
                   faces.c1, faces.c2, faces.counts, pair_data.owner,
                   T(default_face_area), T(default_normal[1]),
                   T(default_normal[2]), T(default_normal[3]),
                   T(default_face_velocity[1]), T(default_face_velocity[2]),
                   T(default_face_velocity[3]),
                   nsource, maxfs; ndrange = total)
    KA.synchronize(be)
    csr = canonical_face_candidate_csr_soa_3d(be, faces, pair_data;
                                             index_type)
    vol = volume === nothing ? _backend_copy(be, fill(T(1 / max(nsource, 1)), nsource), T) :
          _backend_copy(be, volume, T)
    length(vol) == nsource || error("volume length must match source count")
    return ArepoMeshArrays3D(c1, c2, csr.offsets, csr.faces, csr.signs, vol,
                             area, nx, ny, nz, vx, vy, vz)
end

function compact_canonical_faces_soa_3d(be,
                                        faces::CompactFaceCandidates3D,
                                        pairs::ReciprocalFaceCandidatePairs3D =
                                            reciprocal_face_candidate_pairs_soa_3d(be, faces);
                                        index_type::Type{<:Integer} = eltype(faces.counts))
    nsource = Int(faces.source_count)
    maxfs = Int(faces.max_faces_per_source)
    pairs.source_count == nsource ||
        error("pair source count must match face candidates")
    pairs.max_faces_per_source == maxfs ||
        error("pair stride must match face candidates")

    h_counts = Int.(Array(faces.counts))
    h_owner = Int.(Array(pairs.owner))
    h_c1 = Int.(Array(faces.c1))
    h_c2 = Int.(Array(faces.c2))
    h_sx = Int.(Array(faces.image_sx))
    h_sy = Int.(Array(faces.image_sy))
    h_sz = Int.(Array(faces.image_sz))
    h_tetra = Int.(Array(faces.tetra))
    h_local_face = Int.(Array(faces.local_face))
    h_v1 = Int.(Array(faces.face_v1))
    h_v2 = Int.(Array(faces.face_v2))
    h_v3 = Int.(Array(faces.face_v3))

    rows = Int[]
    @inbounds for source in 1:nsource
        base = (source - 1) * maxfs
        for q in 1:h_counts[source]
            row = base + q
            h_owner[row] != 0 || continue
            push!(rows, row)
        end
    end

    source_row = index_type.(rows)
    c1 = index_type.(h_c1[rows])
    c2 = index_type.(h_c2[rows])
    sx = index_type.(h_sx[rows])
    sy = index_type.(h_sy[rows])
    sz = index_type.(h_sz[rows])
    tetra = index_type.(h_tetra[rows])
    local_face = index_type.(h_local_face[rows])
    v1 = index_type.(h_v1[rows])
    v2 = index_type.(h_v2[rows])
    v3 = index_type.(h_v3[rows])

    return CompactCanonicalFaces3D(
        _backend_copy(be, source_row, index_type),
        _backend_copy(be, c1, index_type),
        _backend_copy(be, c2, index_type),
        _backend_copy(be, sx, index_type),
        _backend_copy(be, sy, index_type),
        _backend_copy(be, sz, index_type),
        _backend_copy(be, tetra, index_type),
        _backend_copy(be, local_face, index_type),
        _backend_copy(be, v1, index_type),
        _backend_copy(be, v2, index_type),
        _backend_copy(be, v3, index_type),
        nsource)
end

@kernel function _compact_canonical_face_counts_soa_kernel!(
    counts, owner, face_counts, nsource::Int, max_faces_per_source::Int)
    source = @index(Global)
    if source <= nsource
        count = zero(eltype(counts))
        nf = Int(face_counts[source])
        base = (source - 1) * max_faces_per_source
        @inbounds for q in 1:nf
            row = base + q
            owner[row] != 0 && (count += one(eltype(counts)))
        end
        counts[source] = count
    end
end

@kernel function _compact_canonical_face_offsets_soa_kernel!(
    offsets, counts, nsource::Int)
    i = @index(Global)
    if i == 1
        offsets[1] = one(eltype(offsets))
        @inbounds for source in 1:nsource
            offsets[source + 1] = offsets[source] + counts[source]
        end
    end
end

function compact_canonical_face_csr_soa_3d(be,
                                           faces::CompactFaceCandidates3D,
                                           pairs::ReciprocalFaceCandidatePairs3D =
                                               reciprocal_face_candidate_pairs_soa_3d(be, faces);
                                           index_type::Type{<:Integer} = eltype(faces.counts))
    nsource = Int(faces.source_count)
    maxfs = Int(faces.max_faces_per_source)
    pairs.source_count == nsource ||
        error("pair source count must match face candidates")
    pairs.max_faces_per_source == maxfs ||
        error("pair stride must match face candidates")
    compact = compact_canonical_faces_soa_3d(be, faces, pairs; index_type)
    counts = _backend_zeros(be, index_type, nsource)
    offsets = _backend_zeros(be, index_type, nsource + 1)
    if nsource == 0
        offsets[1] = one(index_type)
        return CompactCanonicalFaceCSR3D(compact, counts, offsets, maxfs, nsource)
    end
    count_kernel = _compact_canonical_face_counts_soa_kernel!(be)
    count_kernel(counts, pairs.owner, faces.counts, nsource, maxfs; ndrange = nsource)
    KA.synchronize(be)
    offset_kernel = _compact_canonical_face_offsets_soa_kernel!(be)
    offset_kernel(offsets, counts, nsource; ndrange = 1)
    KA.synchronize(be)
    return CompactCanonicalFaceCSR3D(compact, counts, offsets, maxfs, nsource)
end

function compact_canonical_mesh_arrays_3d(be,
                                          compact::CompactCanonicalFaces3D;
                                          volume = nothing,
                                          face_area = nothing,
                                          normal_x = nothing,
                                          normal_y = nothing,
                                          normal_z = nothing,
                                          face_vx = nothing,
                                          face_vy = nothing,
                                          face_vz = nothing,
                                          default_face_area::Real = 1.0,
                                          default_normal = (1.0, 0.0, 0.0),
                                          default_face_velocity = (0.0, 0.0, 0.0),
                                          T::Type{<:AbstractFloat} = Float64,
                                          index_type::Type{<:Integer} = eltype(compact.c1))
    nsource = Int(compact.source_count)
    c1h = index_type.(Array(compact.c1))
    c2h = index_type.(Array(compact.c2))
    nf = length(c1h)
    offsets, cell_faces, signs = _cell_face_csr(nsource, c1h, c2h, index_type)
    vol = volume === nothing ? fill(T(1 / max(nsource, 1)), nsource) : T.(collect(volume))
    length(vol) == nsource || error("volume length must match source count")
    area = face_area === nothing ? fill(T(default_face_area), nf) : T.(collect(face_area))
    nx = normal_x === nothing ? fill(T(default_normal[1]), nf) : T.(collect(normal_x))
    ny = normal_y === nothing ? fill(T(default_normal[2]), nf) : T.(collect(normal_y))
    nz = normal_z === nothing ? fill(T(default_normal[3]), nf) : T.(collect(normal_z))
    vx = face_vx === nothing ? fill(T(default_face_velocity[1]), nf) : T.(collect(face_vx))
    vy = face_vy === nothing ? fill(T(default_face_velocity[2]), nf) : T.(collect(face_vy))
    vz = face_vz === nothing ? fill(T(default_face_velocity[3]), nf) : T.(collect(face_vz))
    length(area) == nf || error("face_area length must match compact face count")
    length(nx) == nf || error("normal_x length must match compact face count")
    length(ny) == nf || error("normal_y length must match compact face count")
    length(nz) == nf || error("normal_z length must match compact face count")
    length(vx) == nf || error("face_vx length must match compact face count")
    length(vy) == nf || error("face_vy length must match compact face count")
    length(vz) == nf || error("face_vz length must match compact face count")
    return ArepoMeshArrays3D(
        _backend_copy(be, c1h, index_type),
        _backend_copy(be, c2h, index_type),
        _backend_copy(be, offsets, index_type),
        _backend_copy(be, cell_faces, index_type),
        _backend_copy(be, signs, index_type),
        _backend_copy(be, vol, T),
        _backend_copy(be, area, T),
        _backend_copy(be, nx, T),
        _backend_copy(be, ny, T),
        _backend_copy(be, nz, T),
        _backend_copy(be, vx, T),
        _backend_copy(be, vy, T),
        _backend_copy(be, vz, T))
end

function compact_canonical_mesh_arrays_3d(be,
                                          faces::CompactFaceCandidates3D;
                                          pairs::Union{Nothing,ReciprocalFaceCandidatePairs3D} = nothing,
                                          kwargs...)
    pair_data = pairs === nothing ?
                reciprocal_face_candidate_pairs_soa_3d(be, faces) : pairs
    compact = compact_canonical_faces_soa_3d(be, faces, pair_data)
    return compact_canonical_mesh_arrays_3d(be, compact; kwargs...)
end

function delaunay_soa_3d(d::DelaunayTetrahedra3D;
                         T::Type{<:AbstractFloat} = Float64,
                         index_type::Type{<:Integer} = Int32)
    nt = length(d.tetras)
    tet_p1 = Vector{index_type}(undef, nt)
    tet_p2 = Vector{index_type}(undef, nt)
    tet_p3 = Vector{index_type}(undef, nt)
    tet_p4 = Vector{index_type}(undef, nt)
    @inbounds for t in 1:nt
        tet = d.tetras[t]
        tet_p1[t] = index_type(tet[1])
        tet_p2[t] = index_type(tet[2])
        tet_p3[t] = index_type(tet[3])
        tet_p4[t] = index_type(tet[4])
    end
    return DelaunaySoA3D(
        T.(view(d.points, :, 1)),
        T.(view(d.points, :, 2)),
        T.(view(d.points, :, 3)),
        index_type.(d.original_index),
        index_type.(view(d.image_shift, :, 1)),
        index_type.(view(d.image_shift, :, 2)),
        index_type.(view(d.image_shift, :, 3)),
        tet_p1, tet_p2, tet_p3, tet_p4,
        T.(view(d.circumcenters, :, 1)),
        T.(view(d.circumcenters, :, 2)),
        T.(view(d.circumcenters, :, 3)),
        fill(one(index_type), nt),
    )
end

function tessellation_soa_3d(ref::TessellationReference3D;
                             T::Type{<:AbstractFloat} = Float64,
                             index_type::Type{<:Integer} = Int32)
    ref.delaunay isa DelaunayTetrahedra3D ||
        error("TessellationReference3D must carry a DelaunayTetrahedra3D payload; call with return_delaunay=true")
    shifts = _face_image_shift_or_zeros(length(ref.geom.c1), ref.face_image_shift)
    geom = ArepoMeshArrays3D(
        index_type.(Array(ref.geom.c1)),
        index_type.(Array(ref.geom.c2)),
        index_type.(Array(ref.geom.cell_face_offsets)),
        index_type.(Array(ref.geom.cell_faces)),
        index_type.(Array(ref.geom.cell_face_signs)),
        T.(Array(ref.geom.volume)),
        T.(Array(ref.geom.face_area)),
        T.(Array(ref.geom.normal_x)),
        T.(Array(ref.geom.normal_y)),
        T.(Array(ref.geom.normal_z)),
        T.(Array(ref.geom.face_vx)),
        T.(Array(ref.geom.face_vy)),
        T.(Array(ref.geom.face_vz)),
    )
    return TessellationSoA3D(
        delaunay_soa_3d(ref.delaunay; T, index_type),
        geom,
        T.(view(ref.center, :, 1)),
        T.(view(ref.center, :, 2)),
        T.(view(ref.center, :, 3)),
        T.(view(ref.face_center, :, 1)),
        T.(view(ref.face_center, :, 2)),
        T.(view(ref.face_center, :, 3)),
        index_type.(view(shifts, :, 1)),
        index_type.(view(shifts, :, 2)),
        index_type.(view(shifts, :, 3)),
    )
end

function to_backend(be, soa::DelaunaySoA3D; T::Type{<:AbstractFloat} = Float32,
                    index_type::Type{<:Integer} = Int32)
    return DelaunaySoA3D(
        _backend_copy(be, soa.point_x, T),
        _backend_copy(be, soa.point_y, T),
        _backend_copy(be, soa.point_z, T),
        _backend_copy(be, soa.original_index, index_type),
        _backend_copy(be, soa.image_sx, index_type),
        _backend_copy(be, soa.image_sy, index_type),
        _backend_copy(be, soa.image_sz, index_type),
        _backend_copy(be, soa.tet_p1, index_type),
        _backend_copy(be, soa.tet_p2, index_type),
        _backend_copy(be, soa.tet_p3, index_type),
        _backend_copy(be, soa.tet_p4, index_type),
        _backend_copy(be, soa.circum_x, T),
        _backend_copy(be, soa.circum_y, T),
        _backend_copy(be, soa.circum_z, T),
        _backend_copy(be, soa.circum_valid, index_type),
    )
end

function to_backend(be, soa::TessellationSoA3D; T::Type{<:AbstractFloat} = Float32,
                    index_type::Type{<:Integer} = Int32)
    return TessellationSoA3D(
        to_backend(be, soa.delaunay; T, index_type),
        to_backend(be, soa.geom; T, index_type),
        _backend_copy(be, soa.center_x, T),
        _backend_copy(be, soa.center_y, T),
        _backend_copy(be, soa.center_z, T),
        _backend_copy(be, soa.face_center_x, T),
        _backend_copy(be, soa.face_center_y, T),
        _backend_copy(be, soa.face_center_z, T),
        _backend_copy(be, soa.face_image_sx, index_type),
        _backend_copy(be, soa.face_image_sy, index_type),
        _backend_copy(be, soa.face_image_sz, index_type),
    )
end

@kernel function _circumcenters_from_tetra_soa_kernel!(
    out_x, out_y, out_z, out_valid,
    point_x, point_y, point_z,
    tet_p1, tet_p2, tet_p3, tet_p4,
    n::Int, tol)
    t = @index(Global)
    if t <= n
        i1 = Int(tet_p1[t])
        i2 = Int(tet_p2[t])
        i3 = Int(tet_p3[t])
        i4 = Int(tet_p4[t])
        x1 = point_x[i1]; y1 = point_y[i1]; z1 = point_z[i1]
        x2 = point_x[i2]; y2 = point_y[i2]; z2 = point_z[i2]
        x3 = point_x[i3]; y3 = point_y[i3]; z3 = point_z[i3]
        x4 = point_x[i4]; y4 = point_y[i4]; z4 = point_z[i4]

        a11 = 2 * (x2 - x1); a12 = 2 * (y2 - y1); a13 = 2 * (z2 - z1)
        a21 = 2 * (x3 - x1); a22 = 2 * (y3 - y1); a23 = 2 * (z3 - z1)
        a31 = 2 * (x4 - x1); a32 = 2 * (y4 - y1); a33 = 2 * (z4 - z1)
        b1 = x2 * x2 + y2 * y2 + z2 * z2 - x1 * x1 - y1 * y1 - z1 * z1
        b2 = x3 * x3 + y3 * y3 + z3 * z3 - x1 * x1 - y1 * y1 - z1 * z1
        b3 = x4 * x4 + y4 * y4 + z4 * z4 - x1 * x1 - y1 * y1 - z1 * z1

        detA = a11 * (a22 * a33 - a23 * a32) -
               a12 * (a21 * a33 - a23 * a31) +
               a13 * (a21 * a32 - a22 * a31)
        if abs(detA) <= tol
            out_x[t] = zero(eltype(out_x))
            out_y[t] = zero(eltype(out_y))
            out_z[t] = zero(eltype(out_z))
            out_valid[t] = zero(eltype(out_valid))
        else
            detX = b1 * (a22 * a33 - a23 * a32) -
                   a12 * (b2 * a33 - a23 * b3) +
                   a13 * (b2 * a32 - a22 * b3)
            detY = a11 * (b2 * a33 - a23 * b3) -
                   b1 * (a21 * a33 - a23 * a31) +
                   a13 * (a21 * b3 - b2 * a31)
            detZ = a11 * (a22 * b3 - b2 * a32) -
                   a12 * (a21 * b3 - b2 * a31) +
                   b1 * (a21 * a32 - a22 * a31)
            out_x[t] = detX / detA
            out_y[t] = detY / detA
            out_z[t] = detZ / detA
            out_valid[t] = one(eltype(out_valid))
        end
    end
end

function recompute_circumcenters_soa_3d(be, soa::DelaunaySoA3D;
                                        tol::Real = 1e-10,
                                        ndrange = length(soa.tet_p1))
    T = eltype(soa.point_x)
    I = eltype(soa.tet_p1)
    n = Int(length(soa.tet_p1))
    out_x = _backend_zeros(be, T, n)
    out_y = _backend_zeros(be, T, n)
    out_z = _backend_zeros(be, T, n)
    out_valid = _backend_zeros(be, I, n)
    kernel = _circumcenters_from_tetra_soa_kernel!(be)
    event = kernel(out_x, out_y, out_z, out_valid,
                   soa.point_x, soa.point_y, soa.point_z,
                   soa.tet_p1, soa.tet_p2, soa.tet_p3, soa.tet_p4,
                   n, T(tol); ndrange)
    KA.synchronize(be)
    return (; x = out_x, y = out_y, z = out_z, valid = out_valid,
            event)
end

@inline function _orient3d_det(a, b, c, d)
    ax = a[1] - d[1]; ay = a[2] - d[2]; az = a[3] - d[3]
    bx = b[1] - d[1]; by = b[2] - d[2]; bz = b[3] - d[3]
    cx = c[1] - d[1]; cy = c[2] - d[2]; cz = c[3] - d[3]
    return ax * (by * cz - bz * cy) -
           ay * (bx * cz - bz * cx) +
           az * (bx * cy - by * cx)
end

@inline _point_tuple3(points::AbstractMatrix, i::Integer) =
    (points[i, 1], points[i, 2], points[i, 3])

function _oriented_tetra3(points::AbstractMatrix, tet::NTuple{4,Int})
    a, b, c, d = tet
    vol = _orient3d_det(_point_tuple3(points, a), _point_tuple3(points, b),
                        _point_tuple3(points, c), _point_tuple3(points, d))
    vol >= 0 && return tet
    return (a, c, b, d)
end

function _circumsphere3(points::AbstractMatrix, tet::NTuple{4,Int};
                        tol::Float64 = 1e-12)
    p1 = _point_tuple3(points, tet[1])
    p2 = _point_tuple3(points, tet[2])
    p3 = _point_tuple3(points, tet[3])
    p4 = _point_tuple3(points, tet[4])
    A = [
        2 * (p2[1] - p1[1]) 2 * (p2[2] - p1[2]) 2 * (p2[3] - p1[3])
        2 * (p3[1] - p1[1]) 2 * (p3[2] - p1[2]) 2 * (p3[3] - p1[3])
        2 * (p4[1] - p1[1]) 2 * (p4[2] - p1[2]) 2 * (p4[3] - p1[3])
    ]
    b = [
        p2[1]^2 + p2[2]^2 + p2[3]^2 - p1[1]^2 - p1[2]^2 - p1[3]^2,
        p3[1]^2 + p3[2]^2 + p3[3]^2 - p1[1]^2 - p1[2]^2 - p1[3]^2,
        p4[1]^2 + p4[2]^2 + p4[3]^2 - p1[1]^2 - p1[2]^2 - p1[3]^2,
    ]
    if abs(det(A)) <= tol
        return nothing
    end
    c = A \ b
    r2 = (c[1] - p1[1])^2 + (c[2] - p1[2])^2 + (c[3] - p1[3])^2
    return (center = (c[1], c[2], c[3]), radius2 = r2)
end

function _super_tetra_points3(points::AbstractMatrix)
    mins = [minimum(@view points[:, d]) for d in 1:3]
    maxs = [maximum(@view points[:, d]) for d in 1:3]
    cx = 0.5 * (mins[1] + maxs[1])
    cy = 0.5 * (mins[2] + maxs[2])
    cz = 0.5 * (mins[3] + maxs[3])
    span = maximum(maxs .- mins)
    span = span > 0 ? span : 1.0
    s = 64.0 * span
    return [
        cx + s  cy - s  cz - s
        cx - s  cy + s  cz - s
        cx - s  cy - s  cz + s
        cx + s  cy + s  cz + s
    ]
end

function _tet_faces3(tet::NTuple{4,Int})
    return NTuple{3,Int}[
        (tet[1], tet[2], tet[3]),
        (tet[1], tet[4], tet[2]),
        (tet[2], tet[4], tet[3]),
        (tet[3], tet[4], tet[1]),
    ]
end

_sorted_face3(face::NTuple{3,Int}) = Tuple(sort(collect(face)))

function _extended_periodic_points3(points::AbstractMatrix, domain)
    pts = Matrix{Float64}(points)
    n = size(pts, 1)
    dom = _domain3(domain)
    lens = (dom[1][2] - dom[1][1], dom[2][2] - dom[2][1], dom[3][2] - dom[3][1])
    out = Matrix{Float64}(undef, 27n, 3)
    original = Vector{Int}(undef, 27n)
    shifts = Matrix{Int}(undef, 27n, 3)
    row = 1
    for sx in -1:1, sy in -1:1, sz in -1:1
        for i in 1:n
            out[row, 1] = pts[i, 1] + sx * lens[1]
            out[row, 2] = pts[i, 2] + sy * lens[2]
            out[row, 3] = pts[i, 3] + sz * lens[3]
            original[row] = i
            shifts[row, 1] = sx
            shifts[row, 2] = sy
            shifts[row, 3] = sz
            row += 1
        end
    end
    return out, original, shifts
end

function _bowyer_watson_delaunay3(points::AbstractMatrix;
                                  counters = TessellationFallbackCounters3D(),
                                  tol::Float64 = 1e-10)
    base_n = size(points, 1)
    super = _super_tetra_points3(points)
    allpts = vcat(Matrix{Float64}(points), super)
    super_ids = (base_n + 1, base_n + 2, base_n + 3, base_n + 4)
    tetras = NTuple{4,Int}[_oriented_tetra3(allpts, super_ids)]
    for p in 1:base_n
        bad = falses(length(tetras))
        for ti in eachindex(tetras)
            cs = _circumsphere3(allpts, tetras[ti]; tol)
            record_in_sphere_test!(counters; exact = cs === nothing)
            if cs === nothing
                record_degenerate_face!(counters)
                continue
            end
            c = cs.center
            dx = allpts[p, 1] - c[1]
            dy = allpts[p, 2] - c[2]
            dz = allpts[p, 3] - c[3]
            bad[ti] = dx * dx + dy * dy + dz * dz <= cs.radius2 + tol
        end
        any(bad) || begin
            record_topology_retry!(counters)
            continue
        end
        faces = Dict{NTuple{3,Int},NTuple{3,Int}}()
        for ti in eachindex(tetras)
            bad[ti] || continue
            for face in _tet_faces3(tetras[ti])
                key = _sorted_face3(face)
                if haskey(faces, key)
                    delete!(faces, key)
                else
                    faces[key] = face
                end
            end
        end
        keep = NTuple{4,Int}[tetras[ti] for ti in eachindex(tetras) if !bad[ti]]
        for face in values(faces)
            tet = _oriented_tetra3(allpts, (face[1], face[2], face[3], p))
            abs(_orient3d_det(_point_tuple3(allpts, tet[1]), _point_tuple3(allpts, tet[2]),
                              _point_tuple3(allpts, tet[3]), _point_tuple3(allpts, tet[4]))) <= tol &&
                (record_degenerate_face!(counters); continue)
            push!(keep, tet)
        end
        tetras = keep
    end
    super_set = Set(super_ids)
    filter!(tet -> all(v -> !(v in super_set), tet), tetras)
    centers = Matrix{Float64}(undef, length(tetras), 3)
    keep = trues(length(tetras))
    for ti in eachindex(tetras)
        cs = _circumsphere3(allpts, tetras[ti]; tol)
        if cs === nothing
            keep[ti] = false
            record_degenerate_face!(counters)
        else
            centers[ti, 1] = cs.center[1]
            centers[ti, 2] = cs.center[2]
            centers[ti, 3] = cs.center[3]
        end
    end
    if !all(keep)
        tetras = tetras[keep]
        centers = centers[keep, :]
    end
    return tetras, centers, counters
end

function _order_ring_around_edge3(verts::Vector{NTuple{3,Float64}},
                                  p1::NTuple{3,Float64},
                                  p2::NTuple{3,Float64})
    length(verts) <= 3 && return verts
    cx, cy, cz = _centroid3(verts)
    ex = p2[1] - p1[1]
    ey = p2[2] - p1[2]
    ez = p2[3] - p1[3]
    en = sqrt(ex * ex + ey * ey + ez * ez)
    en > 0 || return verts
    ex /= en; ey /= en; ez /= en
    ux, uy, uz = abs(ex) < 0.8 ? (0.0, -ez, ey) : (-ez, 0.0, ex)
    un = sqrt(ux * ux + uy * uy + uz * uz)
    ux /= un; uy /= un; uz /= un
    vx = ey * uz - ez * uy
    vy = ez * ux - ex * uz
    vz = ex * uy - ey * ux
    idx = collect(eachindex(verts))
    sort!(idx; by = q -> begin
        px = verts[q][1] - cx
        py = verts[q][2] - cy
        pz = verts[q][3] - cz
        atan(px * vx + py * vy + pz * vz, px * ux + py * uy + pz * uz)
    end)
    ordered = verts[idx]
    sx, sy, sz = _ring_area_normal3(ordered)
    sx * ex + sy * ey + sz * ez < 0 && reverse!(ordered)
    return ordered
end

function _dedupe_ring_vertices3(verts::Vector{NTuple{3,Float64}}; tol::Float64)
    out = NTuple{3,Float64}[]
    for v in verts
        _push_unique_vertex3!(out, v; tol)
    end
    return out
end

function _delaunay_voronoi_mesh_arrays_3d(points::AbstractMatrix;
                                          domain = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                                          periodic::Bool = true,
                                          T::Type{<:AbstractFloat} = Float64,
                                          index_type::Type{<:Integer} = Int32,
                                          face_velocity = nothing,
                                          cell_velocity = nothing,
                                          min_face_surface_fraction::Real = 1e-8,
                                          tol::Float64 = 1e-9)
    size(points, 2) == 3 || error("points must be n x 3")
    periodic || error("Delaunay reference tessellator currently supports periodic boxes")
    pts = Matrix{Float64}(points)
    n = size(pts, 1)
    ext, original, image_shift = _extended_periodic_points3(pts, domain)
    counters = TessellationFallbackCounters3D()
    tetras, circumcenters, counters = _bowyer_watson_delaunay3(ext;
                                                               counters,
                                                               tol)
    edge_to_tetras = Dict{Tuple{Int,Int},Vector{Int}}()
    for (ti, tet) in pairs(tetras)
        for a in 1:3, b in (a + 1):4
            u, v = tet[a], tet[b]
            key = u < v ? (u, v) : (v, u)
            push!(get!(edge_to_tetras, key, Int[]), ti)
        end
    end
    c1 = Int[]
    c2 = Int[]
    area = Float64[]
    normal_tuples = NTuple{3,Float64}[]
    face_center_tuples = NTuple{3,Float64}[]
    shift_tuples = NTuple{3,Int}[]
    seen = Set{Tuple{Int,Int,Int,Int,Int}}()
    for ((u, v), incident) in edge_to_tetras
        ou = original[u]
        ov = original[v]
        ou == ov && continue
        su = (image_shift[u, 1], image_shift[u, 2], image_shift[u, 3])
        sv = (image_shift[v, 1], image_shift[v, 2], image_shift[v, 3])
        su == (0, 0, 0) || sv == (0, 0, 0) || continue
        i, j = ou < ov ? (ou, ov) : (ov, ou)
        raw_shift = ou < ov ? (sv[1] - su[1], sv[2] - su[2], sv[3] - su[3]) :
                              (su[1] - sv[1], su[2] - sv[2], su[3] - sv[3])
        key = (i, j, raw_shift[1], raw_shift[2], raw_shift[3])
        key in seen && continue
        verts = [_point_tuple3(circumcenters, ti) for ti in incident]
        verts = _dedupe_ring_vertices3(verts; tol = 100tol)
        length(verts) >= 3 || (record_degenerate_face!(counters); continue)
        p_i = _point_tuple3(pts, i)
        p_j = (pts[j, 1] + raw_shift[1], pts[j, 2] + raw_shift[2],
               pts[j, 3] + raw_shift[3])
        verts = _order_ring_around_edge3(verts, p_i, p_j)
        sx, sy, sz = _ring_area_normal3(verts)
        ar = 0.5 * sqrt(sx * sx + sy * sy + sz * sz)
        ar > tol || (record_degenerate_face!(counters); continue)
        fc = _polygon_area_centroid3(verts)
        ex = p_j[1] - p_i[1]
        ey = p_j[2] - p_i[2]
        ez = p_j[3] - p_i[3]
        en = sqrt(ex * ex + ey * ey + ez * ez)
        en > 0 || continue
        push!(c1, i)
        push!(c2, j)
        push!(area, ar)
        push!(normal_tuples, (ex / en, ey / en, ez / en))
        push!(face_center_tuples, fc)
        push!(shift_tuples, raw_shift)
        push!(seen, key)
    end
    nf = length(c1)
    nf > 0 || error("Delaunay reference produced no hydro faces")
    minfrac = Float64(min_face_surface_fraction)
    if minfrac > 0
        surface = zeros(Float64, n)
        for f in eachindex(c1)
            surface[c1[f]] += area[f]
            surface[c2[f]] += area[f]
        end
        keep = findall(f -> area[f] > minfrac * max(surface[c1[f]], surface[c2[f]]),
                       eachindex(c1))
        c1 = c1[keep]; c2 = c2[keep]; area = area[keep]
        normal_tuples = normal_tuples[keep]
        face_center_tuples = face_center_tuples[keep]
        shift_tuples = shift_tuples[keep]
        nf = length(c1)
    end
    volume = zeros(Float64, n)
    center_acc = zeros(Float64, n, 3)
    for f in 1:nf
        i = c1[f]; j = c2[f]
        sx, sy, sz = shift_tuples[f]
        dx = pts[j, 1] + sx - pts[i, 1]
        dy = pts[j, 2] + sy - pts[i, 2]
        dz = pts[j, 3] + sz - pts[i, 3]
        h = 0.5 * sqrt(dx * dx + dy * dy + dz * dz)
        dvol = area[f] * h / 3
        volume[i] += dvol
        volume[j] += dvol
        fc = face_center_tuples[f]
        center_acc[i, 1] += dvol * (0.75 * fc[1] + 0.25 * pts[i, 1])
        center_acc[i, 2] += dvol * (0.75 * fc[2] + 0.25 * pts[i, 2])
        center_acc[i, 3] += dvol * (0.75 * fc[3] + 0.25 * pts[i, 3])
        pjr = (pts[j, 1] + sx, pts[j, 2] + sy, pts[j, 3] + sz)
        center_acc[j, 1] += dvol * (0.75 * fc[1] + 0.25 * pjr[1])
        center_acc[j, 2] += dvol * (0.75 * fc[2] + 0.25 * pjr[2])
        center_acc[j, 3] += dvol * (0.75 * fc[3] + 0.25 * pjr[3])
    end
    dom = _domain3(domain)
    center = Matrix{Float64}(undef, n, 3)
    for i in 1:n
        if volume[i] > tol
            center[i, 1] = center_acc[i, 1] / volume[i]
            center[i, 2] = center_acc[i, 2] / volume[i]
            center[i, 3] = center_acc[i, 3] / volume[i]
        else
            center[i, 1] = pts[i, 1]
            center[i, 2] = pts[i, 2]
            center[i, 3] = pts[i, 3]
        end
        for d in 1:3
            lo, hi = dom[d]
            len = hi - lo
            center[i, d] = lo + mod(center[i, d] - lo, len)
        end
    end
    normals = Matrix{Float64}(undef, nf, 3)
    face_center = Matrix{Float64}(undef, nf, 3)
    face_image_shift = Matrix{Int}(undef, nf, 3)
    for f in 1:nf
        normals[f, 1] = normal_tuples[f][1]
        normals[f, 2] = normal_tuples[f][2]
        normals[f, 3] = normal_tuples[f][3]
        face_center[f, 1] = face_center_tuples[f][1]
        face_center[f, 2] = face_center_tuples[f][2]
        face_center[f, 3] = face_center_tuples[f][3]
        face_image_shift[f, 1] = shift_tuples[f][1]
        face_image_shift[f, 2] = shift_tuples[f][2]
        face_image_shift[f, 3] = shift_tuples[f][3]
    end
    fvx, fvy, fvz = _face_velocity_arrays_3d(c1, c2, face_velocity, cell_velocity, T)
    offsets, faces, signs = _cell_face_csr(n, c1, c2, index_type)
    geom = ArepoMeshArrays3D(index_type.(c1), index_type.(c2), offsets, faces, signs,
                             T.(volume), T.(area), T.(normals[:, 1]),
                             T.(normals[:, 2]), T.(normals[:, 3]), fvx, fvy, fvz)
    delaunay = DelaunayTetrahedra3D(ext, original, image_shift, tetras,
                                    circumcenters, counters)
    return (; geom, volume, center, face_center, face_image_shift,
            generators = pts, domain = dom, delaunay,
            bins_per_axis = nothing, search_radius = nothing)
end

"""
    tessellation_reference_3d(built; algorithm, backend_residency=:host_reference)

Wrap a tessellation builder result in the reference/debug schema used by the
production port gates.  `built` must expose `geom`, `center`, `face_center`,
and optionally `face_image_shift`, as the existing PowerFoam rebuild producers
already do.
"""
function tessellation_reference_3d(built; algorithm::Symbol,
                                   backend_residency::Symbol = :host_reference,
                                   owner_task = nothing,
                                   owner_index = nothing,
                                   metadata = NamedTuple(),
                                   delaunay = nothing)
    geom = built.geom
    nf = length(geom.c1)
    shifts = _face_image_shift_or_zeros(nf, hasproperty(built, :face_image_shift) ?
                                            built.face_image_shift : nothing)
    keys = canonical_face_keys_3d(geom; face_image_shift = shifts,
                                  owner_task, owner_index)
    order = sortperm(keys)
    md = merge((cells = length(geom.volume),
                faces = nf,
                source = :powerfoam,
                has_delaunay = delaunay !== nothing), metadata)
    return TessellationReference3D(geom, Matrix{Float64}(built.center),
                                   Matrix{Float64}(built.face_center), shifts,
                                   keys, order, algorithm, backend_residency,
                                   md, delaunay)
end

"""
    build_arepo_tessellation_3d(points; ...)

Production-facing tessellation API for the AREPO port.  Today this is a
reference wrapper around PowerFoam's local periodic halfspace rebuild, with the
same output contract expected from the future Delaunay/KA implementation.  The
`algorithm` keyword is explicit so gates cannot accidentally mistake this rung
for the completed Delaunay port.
"""
function build_arepo_tessellation_3d(points::AbstractMatrix;
                                     domain = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                                     periodic::Bool = true,
                                     active = nothing,
                                     previous = nothing,
                                     backend = KA.CPU(),
                                     predicates::Symbol = :adaptive,
                                     return_delaunay::Bool = false,
                                     algorithm::Symbol = :local_periodic_halfspace,
                                     T::Type{<:AbstractFloat} = Float64,
                                     index_type::Type{<:Integer} = Int32,
                                     face_velocity = nothing,
                                     cell_velocity = nothing,
                                     bins_per_axis = nothing,
                                     search_radius::Integer = 1,
                                     min_face_surface_fraction::Real = 1e-5,
                                     threaded::Bool = Threads.nthreads() > 1)
    size(points, 2) == 3 || error("points must be n x 3")
    periodic || error("build_arepo_tessellation_3d currently supports only periodic boxes")
    active === nothing || length(active) == size(points, 1) ||
        error("active mask length must match point count")
    previous === nothing || previous isa TessellationReference3D ||
        error("previous must be a TessellationReference3D when provided")
    algorithm in (:local_periodic_halfspace, :arepo_delaunay_reference) ||
        error("algorithm=$algorithm is not implemented yet; use :local_periodic_halfspace or :arepo_delaunay_reference")
    predicates in (:adaptive, :float64, :exact_cpu) ||
        error("unsupported predicate policy: $predicates")

    built = if algorithm == :arepo_delaunay_reference
        _delaunay_voronoi_mesh_arrays_3d(points; domain, periodic, T,
                                         index_type, face_velocity,
                                         cell_velocity,
                                         min_face_surface_fraction)
    else
        local_periodic_voronoi_mesh_arrays_3d(points; domain, T,
                                              index_type,
                                              face_velocity,
                                              cell_velocity,
                                              bins_per_axis,
                                              search_radius,
                                              threaded,
                                              min_face_surface_fraction)
    end
    delaunay_payload = hasproperty(built, :delaunay) ? built.delaunay : nothing
    return tessellation_reference_3d(built; algorithm,
                                     backend_residency = backend isa KA.CPU ?
                                                       :host_reference :
                                                       :host_reference_from_gpu_contract,
                                     metadata = (periodic = periodic,
                                                 predicates = predicates,
                                                 return_delaunay = return_delaunay,
                                                 has_delaunay =
                                                     delaunay_payload !== nothing,
                                                 bins_per_axis = built.bins_per_axis,
                                                 search_radius = built.search_radius,
                                                 min_face_surface_fraction =
                                                     Float64(min_face_surface_fraction)),
                                     delaunay = return_delaunay ? delaunay_payload :
                                                nothing)
end
