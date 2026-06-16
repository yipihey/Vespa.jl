module PowerFoam

using KernelAbstractions
using LinearAlgebra
using Statistics
const KA = KernelAbstractions

export PowerSites2D, PolygonMesh2D, FaceTable2D
export ArepoMeshArrays2D, EulerState2D, PrimitiveState2D, FaceFluxWork2D,
       HydroGradients2D, FaceStates2D
export ArepoMeshArrays3D, EulerState3D, PrimitiveState3D, FaceFluxWork3D
export GradientConnections3D, HydroGradients3D, FaceStates3D
export ArepoRunOptions, ArepoHydroSmokeAssessment, ArepoProblemSpec,
       ArepoRuntimeState3D, arepo_problem_spec, arepo_runtime_state_3d,
       arepo_run_scaffold, classify_ka_hydro_smoke,
       ArepoDirectGravityParticleState, ArepoDirectGravityResult,
       arepo_direct_gravity_accel!, arepo_direct_gravity_accel,
       arepo_direct_gravity_potential_energy, arepo_direct_gravity_oracle,
       arepo_direct_gravity_kick_drift_step, arepo_direct_gravity_runtime_state,
       arepo_pm_gravity_runtime_state,
       arepo_pm_gravity_fixture, periodic_image_sum_accel,
       periodic_background_subtracted_image_oracle,
       momentum_residual, max_abs_accel, periodic_cell_center_residual,
       probe_poissonkernels_monorepo, run_arepo_pm_gravity_preflight,
       ArepoGravitySolverSpec, arepo_gravity_solver_registry,
       arepo_gravity_solver_status,
       ArepoPMGravityWorkspace, ArepoPMGravityResult,
       arepo_pm_gravity_workspace, arepo_pm_gravity!, arepo_pm_gravity,
       arepo_pm_gravity_result_rows,
       ArepoConfigFlags, ArepoParameterSet, ArepoParameterValidation,
       ArepoCosmologyRuntime,
       ArepoCosmologyStepMetadata,
       ArepoRuntimeFeatureSet, arepo_runtime_features,
       read_arepo_param_file, read_arepo_config_flags,
       parse_arepo_param_text, parse_arepo_config_text,
       normalize_arepo_parameters, validate_arepo_parameters,
       arepo_cosmology_runtime, arepo_cosmology_step_metadata,
       arepo_cosmology_expansion_factor, arepo_cosmology_adot_over_a,
       ArepoSnapshotLocator, ArepoSnapshotHeader, ArepoGasSnapshotBlock,
       ArepoSnapshotData, ArepoHydroRuntimePayload, ArepoSnapshotValidation, ArepoSnapshotIOResult,
       arepo_snapshot_hdf5_available, snapshot_available_fields,
       locate_arepo_snapshot, arepo_snapshot_read_preflight,
       arepo_snapshot_hydro_payload, arepo_snapshot_hydro_state,
       arepo_snapshot_hydro_state_2d, arepo_snapshot_hydro_state_3d,
       derive_arepo_snapshot_volume!,
       derive_arepo_snapshot_pressure!, resolve_arepo_snapshot_centers!,
       validate_arepo_snapshot, read_arepo_snapshot, write_arepo_snapshot,
       read_arepo_snapshot_header, read_arepo_gas_snapshot_block
export power_diagram, from_arepo_polygons, arepo_face_table
export polygon_area, polygon_centroid, cell_areas, cell_centroids
export cell_quality, mesh_quality, reconstruction_condition_numbers
export face_velocity_alignment, mesh_loss, refine_patch, refine_patch_points,
       relax_points_velocity_alignment, relax_weights
export arepo_mesh_arrays, to_backend, euler_state_2d, primitive_to_conserved_2d!,
       conserved_to_primitive_2d, primitive_work_2d, conserved_to_primitive_2d!,
       primitive_to_arrays_2d, hydro_work_2d, hydro_gradient_work_2d,
       calculate_gradients_from_mesh_2d!, face_prediction_work_2d,
       predict_face_states_2d!, finite_volume_step_2d!,
       finite_volume_reconstructed_step_2d!,
       moving_mesh_step_2d!, moving_mesh_reconstructed_step_2d!,
       advect_generators_2d, periodic_power_mesh_arrays_2d,
       total_conserved_2d, max_signal_speed_2d,
       PF_NOH2D_DEFAULT_GAMMA, pf_noh2d_run_tag,
       pf_noh2d_uniform_points, pf_noh2d_mesh,
       pf_noh2d_initial_primitives, pf_noh2d_initial_state,
       pf_noh2d_stable_dt, pf_noh2d_radial_bins,
       pf_noh2d_metric_row, pf_noh2d_run,
       PF_SOUNDWAVE2D_DEFAULT_GAMMA,
       pf_soundwave2d_run_tag, pf_soundwave2d_mesh,
       PF_GRESHO2D_DEFAULT_GAMMA,
       pf_gresho2d_run_tag, pf_gresho2d_mesh,
       pf_gresho2d_exact_primitives, pf_gresho2d_initial_state,
       pf_gresho2d_stable_dt, pf_gresho2d_tangential_velocity,
       pf_gresho2d_metric_row,
       pf_gresho2d_profile_rows, pf_gresho2d_run,
       pf_soundwave2d_exact_primitives, pf_soundwave2d_initial_state,
       pf_soundwave2d_stable_dt, pf_soundwave2d_metric_row,
       pf_soundwave2d_profile_rows, pf_soundwave2d_run
export cartesian_periodic_mesh_arrays_3d, arepo_voronoi_mesh_arrays_3d,
       bounded_voronoi_mesh_arrays_3d, periodic_voronoi_mesh_arrays_3d,
       local_periodic_voronoi_mesh_arrays_3d,
       TessellationReference3D, build_arepo_tessellation_3d,
       DelaunayTetrahedra3D, DelaunaySoA3D, PeriodicPointImages3D,
       DenseCandidatePairs3D, CandidateStencil3D, CandidateTetraPredicates3D,
       CandidateConflictFaceRows3D, CandidateBoundaryFaceRows3D,
       CompactBoundaryFaces3D, CompactFaceCandidates3D,
       CompactFaceCandidateCSR3D, SourceOwnedFaceCSR3D,
       ReciprocalFaceCandidatePairs3D, CompactCanonicalFaces3D,
       CompactCanonicalFaceCSR3D,
       TessellationSoA3D,
       delaunay_soa_3d, tessellation_soa_3d,
       periodic_point_images_soa_3d, dense_candidate_pairs_soa_3d,
       pack_candidate_stencil_soa_3d, candidate_tetra_predicates_soa_3d,
       candidate_conflict_face_rows_soa_3d,
       candidate_boundary_face_rows_soa_3d, pack_boundary_faces_soa_3d,
       compact_face_candidates_soa_3d, compact_face_candidate_csr_soa_3d,
       source_owned_face_csr_soa_3d, reciprocal_face_candidate_pairs_soa_3d,
       canonical_face_candidate_csr_soa_3d,
       compact_face_candidate_mesh_arrays_3d,
       canonical_face_candidate_mesh_arrays_3d,
       compact_canonical_faces_soa_3d, compact_canonical_face_csr_soa_3d,
       compact_canonical_mesh_arrays_3d,
       recompute_circumcenters_soa_3d,
       tessellation_reference_3d, canonical_face_keys_3d,
       canonical_face_order_3d,
       TessellationPredicatePolicy3D, TessellationPredicateAdaptive3D,
       TessellationPredicateFloat64Only3D, TessellationPredicateExactCPU3D,
       TessellationPredicateCPUFallback3D, TessellationPointIdentity3D,
       TessellationFaceProvenance3D, TessellationFallbackCounters3D,
       record_in_sphere_test!, record_convex_edge_test!,
       record_in_tetra_test!, record_orient3d_test!,
       record_exact_cpu_fallback!, record_gpu_fallback!,
       record_topology_retry!, record_degenerate_face!,
       record_skipped_infinite_tetra!,
       with_update_targets_3d, face_update_activity_3d,
       advect_generators_3d,
       moving_mesh_step_3d!,
       euler_state_3d, primitive_to_conserved_3d!, conserved_to_primitive_3d,
       primitive_work_3d, conserved_to_primitive_3d!, primitive_to_arrays_3d,
       hydro_work_3d, finite_volume_step_3d!, finite_volume_reconstructed_step_3d!,
       finite_volume_reconstructed_step_activecells_3d!,
       finite_volume_reconstructed_hierarchy_step_3d!,
       arepo_hydro_dt_3d, arepo_timebin_3d, arepo_system_step_3d,
       arepo_next_sync_step_3d,
       arepo_active_cells_3d,
       arepo_hydro_timebins_3d, active_face_table_3d,
       arepo_mesh_velocity_3d, total_conserved_3d,
       max_signal_speed_3d, gradient_connections_3d, hydro_gradient_work_3d,
       gradient_connections_from_mesh_3d, calculate_gradients_3d!,
       calculate_gradients_from_mesh_3d!,
       calculate_gradients_from_mesh_activecells_3d!,
       hydro_gradients_to_arrays,
       face_prediction_work_3d, predict_face_states_3d!, face_states_to_arrays
export write_svg

const Point2 = NTuple{2,Float64}

"""
    PowerSites2D(points; weights=zeros(n), domain=((0, 1), (0, 1)))

Generator set for a 2-D power/Laguerre diagram.  Cell `i` is the set of points
where `|x - p_i|^2 - weights[i]` is minimal, clipped to `domain`.
"""
struct PowerSites2D
    points::Matrix{Float64}             # n x 2
    weights::Vector{Float64}            # n
    domain::NTuple{2,NTuple{2,Float64}}
end

function PowerSites2D(points::AbstractMatrix; weights = nothing,
                      domain = ((0.0, 1.0), (0.0, 1.0)))
    size(points, 2) == 2 || error("PowerSites2D: expected an n x 2 point matrix")
    pts = Matrix{Float64}(points)
    w = weights === nothing ? zeros(size(pts, 1)) : Float64.(collect(weights))
    length(w) == size(pts, 1) || error("PowerSites2D: weights length must match points")
    dom = ((float(domain[1][1]), float(domain[1][2])),
           (float(domain[2][1]), float(domain[2][2])))
    return PowerSites2D(pts, w, dom)
end

"""
    FaceTable2D

AREPO-shaped 2-D face table.  `c1`/`c2` are 1-based cell ids; `c2 == 0` marks a
domain boundary.  `normal[f, :]` points outward from `c1` toward `c2`.
"""
struct FaceTable2D
    c1::Vector{Int}
    c2::Vector{Int}
    area::Vector{Float64}               # edge length in 2-D
    center::Matrix{Float64}             # nf x 2
    normal::Matrix{Float64}             # nf x 2, outward from c1
    v1::Matrix{Float64}                 # nf x 2
    v2::Matrix{Float64}                 # nf x 2
end

"""
    PolygonMesh2D

Bounded 2-D cell complex with one polygon per cell and an AREPO-like face table.
`generators` and `weights` are kept even for imported AREPO polygons so metrics
can compare mesh geometry against the generating point positions.
"""
struct PolygonMesh2D
    cells::Vector{Matrix{Float64}}       # each polygon is m x 2, CCW
    generators::Matrix{Float64}          # n x 2
    weights::Vector{Float64}
    domain::NTuple{2,NTuple{2,Float64}}
    faces::FaceTable2D
    neighbors::Vector{Vector{Int}}
    meta::NamedTuple
end

function _bbox(domain)
    xmin, xmax = domain[1]
    ymin, ymax = domain[2]
    return [xmin ymin;
            xmax ymin;
            xmax ymax;
            xmin ymax]
end

@inline _cross2(ax, ay, bx, by) = ax * by - ay * bx

function polygon_area(poly::AbstractMatrix)
    n = size(poly, 1)
    n < 3 && return 0.0
    s = 0.0
    @inbounds for i in 1:n
        j = i == n ? 1 : i + 1
        s += _cross2(poly[i, 1], poly[i, 2], poly[j, 1], poly[j, 2])
    end
    return 0.5 * s
end

function polygon_centroid(poly::AbstractMatrix)
    n = size(poly, 1)
    n == 0 && return (NaN, NaN)
    a6 = 0.0
    cx = 0.0
    cy = 0.0
    @inbounds for i in 1:n
        j = i == n ? 1 : i + 1
        cr = _cross2(poly[i, 1], poly[i, 2], poly[j, 1], poly[j, 2])
        a6 += cr
        cx += (poly[i, 1] + poly[j, 1]) * cr
        cy += (poly[i, 2] + poly[j, 2]) * cr
    end
    abs(a6) < 1e-300 && return (mean(@view poly[:, 1]), mean(@view poly[:, 2]))
    return (cx / (3a6), cy / (3a6))
end

function _ensure_ccw(poly::Matrix{Float64})
    polygon_area(poly) >= 0 && return poly
    return reverse(poly; dims = 1)
end

function _clip_halfplane(poly::Matrix{Float64}, a::NTuple{2,Float64}, b::Float64;
                         tol::Float64 = 1e-12)
    n = size(poly, 1)
    n == 0 && return poly
    out = Vector{Point2}()
    inside(x, y) = a[1] * x + a[2] * y <= b + tol
    value(x, y) = a[1] * x + a[2] * y - b

    @inbounds for i in 1:n
        j = i == n ? 1 : i + 1
        sx, sy = poly[i, 1], poly[i, 2]
        ex, ey = poly[j, 1], poly[j, 2]
        sins = inside(sx, sy)
        eins = inside(ex, ey)
        if sins && eins
            push!(out, (ex, ey))
        elseif sins && !eins
            vs = value(sx, sy)
            ve = value(ex, ey)
            t = vs / (vs - ve)
            push!(out, (sx + t * (ex - sx), sy + t * (ey - sy)))
        elseif !sins && eins
            vs = value(sx, sy)
            ve = value(ex, ey)
            t = vs / (vs - ve)
            push!(out, (sx + t * (ex - sx), sy + t * (ey - sy)))
            push!(out, (ex, ey))
        end
    end
    return _points_to_matrix(_dedup_ring(out))
end

function _dedup_ring(points::Vector{Point2}; tol::Float64 = 1e-10)
    isempty(points) && return points
    out = Point2[]
    for p in points
        if isempty(out) || hypot(p[1] - out[end][1], p[2] - out[end][2]) > tol
            push!(out, p)
        end
    end
    if length(out) > 1 && hypot(out[1][1] - out[end][1], out[1][2] - out[end][2]) <= tol
        pop!(out)
    end
    return out
end

function _points_to_matrix(points::Vector{Point2})
    poly = Matrix{Float64}(undef, length(points), 2)
    @inbounds for i in eachindex(points)
        poly[i, 1] = points[i][1]
        poly[i, 2] = points[i][2]
    end
    return poly
end

"""
    power_diagram(sites::PowerSites2D; tol=1e-12) -> PolygonMesh2D

Construct a bounded 2-D power diagram by clipping each domain polygon against
every pairwise power bisector.  This is O(n²) and intended as a correctness
prototype for modest AREPO-comparison meshes.
"""
function power_diagram(sites::PowerSites2D; tol::Float64 = 1e-12)
    pts = sites.points
    w = sites.weights
    n = size(pts, 1)
    cells = Vector{Matrix{Float64}}(undef, n)
    @inbounds for i in 1:n
        poly = _bbox(sites.domain)
        pix, piy = pts[i, 1], pts[i, 2]
        for j in 1:n
            i == j && continue
            pjx, pjy = pts[j, 1], pts[j, 2]
            a = (2 * (pjx - pix), 2 * (pjy - piy))
            b = pjx^2 + pjy^2 - pix^2 - piy^2 + w[i] - w[j]
            poly = _clip_halfplane(poly, a, b; tol)
            size(poly, 1) == 0 && break
        end
        cells[i] = _ensure_ccw(poly)
    end
    return _mesh_from_cells(cells, pts, w, sites.domain, (; source = :power_diagram))
end

"""
    from_arepo_polygons(polys; generators=nothing, weights=zeros(n), domain=nothing)

Build the same `PolygonMesh2D`/`FaceTable2D` surface from AREPO's exact
`ArepoLib.get_voronoi_2d(h)` polygon export.  No reconstruction is performed;
the polygons are AREPO's mesh.
"""
function from_arepo_polygons(polys::AbstractVector; generators = nothing,
                             weights = nothing, domain = nothing)
    n = length(polys)
    cells = [_ensure_ccw(Matrix{Float64}(p)) for p in polys]
    gens = if generators === nothing
        g = Matrix{Float64}(undef, n, 2)
        for i in 1:n
            c = polygon_centroid(cells[i])
            g[i, 1] = c[1]
            g[i, 2] = c[2]
        end
        g
    else
        Matrix{Float64}(generators)
    end
    size(gens) == (n, 2) || error("from_arepo_polygons: generators must be n x 2")
    w = weights === nothing ? zeros(n) : Float64.(collect(weights))
    length(w) == n || error("from_arepo_polygons: weights length must match polygons")
    dom = domain === nothing ? _domain_from_cells(cells) :
          ((float(domain[1][1]), float(domain[1][2])),
           (float(domain[2][1]), float(domain[2][2])))
    return _mesh_from_cells(cells, gens, w, dom, (; source = :arepo_polygons))
end

function _domain_from_cells(cells)
    xs = Float64[]
    ys = Float64[]
    for p in cells
        append!(xs, @view p[:, 1])
        append!(ys, @view p[:, 2])
    end
    return ((minimum(xs), maximum(xs)), (minimum(ys), maximum(ys)))
end

function _mesh_from_cells(cells, generators, weights, domain, meta)
    faces, neighbors = _build_faces(cells)
    return PolygonMesh2D(cells, Matrix{Float64}(generators), Float64.(weights),
                         domain, faces, neighbors, meta)
end

function _edge_key(a, b; digits::Int = 11)
    ar = (round(a[1]; digits), round(a[2]; digits))
    br = (round(b[1]; digits), round(b[2]; digits))
    return ar <= br ? (ar, br) : (br, ar)
end

function _build_faces(cells::Vector{Matrix{Float64}})
    pending = Dict{Any,Tuple{Int,Point2,Point2,Point2,Float64,Point2}}()
    c1 = Int[]
    c2 = Int[]
    area = Float64[]
    center = Point2[]
    normal = Point2[]
    v1 = Point2[]
    v2 = Point2[]

    for ci in eachindex(cells)
        poly = cells[ci]
        n = size(poly, 1)
        n < 2 && continue
        @inbounds for k in 1:n
            l = k == n ? 1 : k + 1
            a = (poly[k, 1], poly[k, 2])
            b = (poly[l, 1], poly[l, 2])
            len = hypot(b[1] - a[1], b[2] - a[2])
            len <= 1e-12 && continue
            ctr = ((a[1] + b[1]) / 2, (a[2] + b[2]) / 2)
            # For CCW polygons the interior is left of the edge, so outward is right.
            nrm = ((b[2] - a[2]) / len, -(b[1] - a[1]) / len)
            key = _edge_key(a, b)
            if haskey(pending, key)
                oc, oa, ob, onrm, olen, octr = pop!(pending, key)
                push!(c1, oc); push!(c2, ci); push!(area, 0.5 * (olen + len))
                push!(center, ((octr[1] + ctr[1]) / 2, (octr[2] + ctr[2]) / 2))
                push!(normal, onrm); push!(v1, oa); push!(v2, ob)
            else
                pending[key] = (ci, a, b, nrm, len, ctr)
            end
        end
    end

    for (_, rec) in pending
        oc, oa, ob, onrm, olen, octr = rec
        push!(c1, oc); push!(c2, 0); push!(area, olen)
        push!(center, octr); push!(normal, onrm); push!(v1, oa); push!(v2, ob)
    end

    nf = length(c1)
    centers = Matrix{Float64}(undef, nf, 2)
    normals = Matrix{Float64}(undef, nf, 2)
    vv1 = Matrix{Float64}(undef, nf, 2)
    vv2 = Matrix{Float64}(undef, nf, 2)
    @inbounds for i in 1:nf
        centers[i, 1] = center[i][1]; centers[i, 2] = center[i][2]
        normals[i, 1] = normal[i][1]; normals[i, 2] = normal[i][2]
        vv1[i, 1] = v1[i][1]; vv1[i, 2] = v1[i][2]
        vv2[i, 1] = v2[i][1]; vv2[i, 2] = v2[i][2]
    end

    neighbors = [Int[] for _ in eachindex(cells)]
    for f in eachindex(c1)
        if c2[f] > 0
            push!(neighbors[c1[f]], c2[f])
            push!(neighbors[c2[f]], c1[f])
        end
    end
    return FaceTable2D(c1, c2, area, centers, normals, vv1, vv2), neighbors
end

cell_areas(mesh::PolygonMesh2D) = abs.(polygon_area.(mesh.cells))

function cell_centroids(mesh::PolygonMesh2D)
    c = Matrix{Float64}(undef, length(mesh.cells), 2)
    for i in eachindex(mesh.cells)
        ci = polygon_centroid(mesh.cells[i])
        c[i, 1] = ci[1]
        c[i, 2] = ci[2]
    end
    return c
end

"""
    arepo_face_table(mesh) -> NamedTuple

Return a plain array table with AREPO-like face columns.  In 2-D, `area` is edge
length and `verts[f, :, :]` stores the segment endpoints.
"""
function arepo_face_table(mesh::PolygonMesh2D)
    f = mesh.faces
    verts = Array{Float64}(undef, length(f.c1), 2, 2)
    @inbounds for i in eachindex(f.c1)
        verts[i, 1, 1] = f.v1[i, 1]; verts[i, 1, 2] = f.v1[i, 2]
        verts[i, 2, 1] = f.v2[i, 1]; verts[i, 2, 2] = f.v2[i, 2]
    end
    return (; c1 = copy(f.c1), c2 = copy(f.c2), area = copy(f.area),
            center = copy(f.center), normals = copy(f.normal), verts)
end

"""
    cell_quality(mesh) -> NamedTuple

Per-cell geometry diagnostics for refinement-boundary and regularization tests.
"""
function cell_quality(mesh::PolygonMesh2D)
    areas = cell_areas(mesh)
    cents = cell_centroids(mesh)
    n = length(areas)
    perimeter = zeros(n)
    minface = fill(Inf, n)
    maxface = zeros(n)
    for f in eachindex(mesh.faces.c1)
        a = mesh.faces.area[f]
        i = mesh.faces.c1[f]
        perimeter[i] += a
        minface[i] = min(minface[i], a)
        maxface[i] = max(maxface[i], a)
        j = mesh.faces.c2[f]
        if j > 0
            perimeter[j] += a
            minface[j] = min(minface[j], a)
            maxface[j] = max(maxface[j], a)
        end
    end
    minface[.!isfinite.(minface)] .= 0.0
    radius = sqrt.(areas ./ pi)
    offset = hypot.(mesh.generators[:, 1] .- cents[:, 1],
                    mesh.generators[:, 2] .- cents[:, 2]) ./ max.(radius, eps())
    compactness = 4pi .* areas ./ max.(perimeter .^ 2, eps())
    small_face_ratio = minface ./ max.(perimeter, eps())
    neighbor_count = length.(mesh.neighbors)
    return (; area = areas, centroid = cents, perimeter, radius,
            centroid_offset = offset, compactness, small_face_ratio,
            min_face = minface, max_face = maxface, neighbor_count)
end

function _quantile_or_nan(v, q)
    isempty(v) && return NaN
    return quantile(collect(skipmissing(v)), q)
end

"""
    mesh_quality(mesh) -> NamedTuple

Aggregate metrics meant for power-vs-Voronoi comparisons near refine/derefine
transitions.
"""
function mesh_quality(mesh::PolygonMesh2D)
    q = cell_quality(mesh)
    cond = reconstruction_condition_numbers(mesh)
    interior = findall(!iszero, q.neighbor_count)
    return (; cells = length(mesh.cells),
            faces = length(mesh.faces.c1),
            volume = sum(q.area),
            area_min = minimum(q.area),
            area_max = maximum(q.area),
            area_ratio = maximum(q.area) / max(minimum(q.area), eps()),
            small_face_p01 = _quantile_or_nan(q.small_face_ratio, 0.01),
            compactness_median = median(q.compactness),
            centroid_offset_max = maximum(q.centroid_offset),
            neighbor_count_min = minimum(q.neighbor_count),
            neighbor_count_max = maximum(q.neighbor_count),
            recon_cond_median = median(cond[interior]),
            recon_cond_max = maximum(cond[interior]))
end

"""
    mesh_loss(mesh, target_areas; ...)

Scalar objective for weight relaxation.  The first term matches target cell
areas; optional penalties reject power weights that buy volume control by
creating tiny faces, poor compactness, fragile reconstruction, or face normals
that sit halfway between flow-parallel and flow-perpendicular.
"""
function mesh_loss(mesh::PolygonMesh2D, target_areas::AbstractVector;
                   area_weight::Real = 1.0,
                   small_face_weight::Real = 0.05,
                   small_face_floor::Real = 1e-4,
                   compactness_weight::Real = 0.01,
                   compactness_floor::Real = 0.25,
                   recon_weight::Real = 0.0,
                   recon_cap::Real = 10.0,
                   velocity_alignment_weight::Real = 0.0,
                   velocities = nothing,
                   velocity_field = nothing,
                   velocity_speed_floor::Real = 1e-12)
    target = Float64.(collect(target_areas))
    length(target) == length(mesh.cells) || error("mesh_loss: target_areas length must match mesh cells")
    q = cell_quality(mesh)
    rel_area = (q.area .- target) ./ max.(target, eps())
    area_loss = sqrt(mean(abs2, rel_area))

    sfloor = float(small_face_floor)
    small_face_loss = if sfloor <= 0
        0.0
    else
        small_face_violation = max.(0.0, sfloor .- q.small_face_ratio) ./ sfloor
        max(maximum(small_face_violation), sqrt(mean(abs2, small_face_violation)))
    end

    cfloor = float(compactness_floor)
    compactness_loss = cfloor <= 0 ? 0.0 :
        sqrt(mean(abs2, max.(0.0, cfloor .- q.compactness) ./ cfloor))

    cond = reconstruction_condition_numbers(mesh)
    finite_cond = collect(filter(isfinite, cond))
    rcap = float(recon_cap)
    recon_loss = (isempty(finite_cond) || rcap <= 0) ? 0.0 :
        sqrt(mean(abs2, max.(0.0, finite_cond .- rcap) ./ rcap))

    vweight = float(velocity_alignment_weight)
    velocity_alignment_loss = vweight <= 0 ? 0.0 :
        face_velocity_alignment(mesh; velocities, velocity_field,
                                speed_floor = velocity_speed_floor).loss

    total = area_weight * area_loss +
            small_face_weight * small_face_loss +
            compactness_weight * compactness_loss +
            recon_weight * recon_loss +
            velocity_alignment_weight * velocity_alignment_loss
    return (; total, area = area_loss, small_face = small_face_loss,
            compactness = compactness_loss, reconstruction = recon_loss,
            velocity_alignment = velocity_alignment_loss)
end

"""
    reconstruction_condition_numbers(mesh)

Condition number of each cell's least-squares linear reconstruction matrix using
neighbor centroid offsets.  Large values flag cells where gradients will be
fragile even if the polygon looks visually acceptable.
"""
function reconstruction_condition_numbers(mesh::PolygonMesh2D)
    c = cell_centroids(mesh)
    out = fill(Inf, length(mesh.cells))
    for i in eachindex(mesh.cells)
        nb = mesh.neighbors[i]
        length(nb) < 2 && continue
        A = Matrix{Float64}(undef, length(nb), 2)
        for (r, j) in pairs(nb)
            A[r, 1] = c[j, 1] - c[i, 1]
            A[r, 2] = c[j, 2] - c[i, 2]
        end
        s = svdvals(A)
        out[i] = s[end] <= eps() ? Inf : s[1] / s[end]
    end
    return out
end

function _velocity_matrix(velocities, n::Integer)
    velocities === nothing && return nothing
    if velocities isa AbstractMatrix
        size(velocities) == (n, 2) || error("velocities must be an n x 2 matrix")
        return Matrix{Float64}(velocities)
    end
    length(velocities) == n || error("velocities length must match mesh cells")
    out = Matrix{Float64}(undef, n, 2)
    for i in 1:n
        v = velocities[i]
        out[i, 1] = float(v[1])
        out[i, 2] = float(v[2])
    end
    return out
end

function _face_velocity(mesh::PolygonMesh2D, f::Integer, velocities, velocity_field)
    if velocity_field !== nothing
        x = mesh.faces.center[f, 1]
        y = mesh.faces.center[f, 2]
        v = velocity_field((x, y))
        return (float(v[1]), float(v[2]))
    end
    velocities === nothing && return (0.0, 0.0)
    i = mesh.faces.c1[f]
    j = mesh.faces.c2[f]
    if j > 0
        return (0.5 * (velocities[i, 1] + velocities[j, 1]),
                0.5 * (velocities[i, 2] + velocities[j, 2]))
    else
        return (velocities[i, 1], velocities[i, 2])
    end
end

"""
    face_velocity_alignment(mesh; velocities=nothing, velocity_field=nothing,
                            speed_floor=1e-12)

Measure whether mesh-face normals are either aligned with the flow or tangent to
it.  The angular penalty is `4 μ²(1 - μ²)`, where
`μ = |n̂ ⋅ v̂|`; it is zero for parallel/perpendicular faces and one at 45°.
`velocities` may be an `n x 2` cell-centered velocity matrix, while
`velocity_field((x, y))` can supply a face-centered analytic velocity.
"""
function face_velocity_alignment(mesh::PolygonMesh2D; velocities = nothing,
                                 velocity_field = nothing,
                                 speed_floor::Real = 1e-12,
                                 parallel_threshold::Real = 0.85,
                                 perpendicular_threshold::Real = 0.15)
    velocity_field === nothing || velocities === nothing ||
        error("provide either velocities or velocity_field, not both")
    vmat = _velocity_matrix(velocities, length(mesh.cells))
    sfloor = float(speed_floor)
    pthr = clamp(float(parallel_threshold), 0.0, 1.0)
    othr = clamp(float(perpendicular_threshold), 0.0, 1.0)

    wsum = 0.0
    loss_sum = 0.0
    ambiguous_sum = 0.0
    parallel_sum = 0.0
    perpendicular_sum = 0.0
    skipped = 0
    for f in eachindex(mesh.faces.c1)
        vx, vy = _face_velocity(mesh, f, vmat, velocity_field)
        speed = hypot(vx, vy)
        if speed <= sfloor
            skipped += 1
            continue
        end
        nx = mesh.faces.normal[f, 1]
        ny = mesh.faces.normal[f, 2]
        μ2 = clamp(((nx * vx + ny * vy) / speed)^2, 0.0, 1.0)
        ambiguous = 4 * μ2 * (1 - μ2)
        w = mesh.faces.area[f]
        wsum += w
        loss_sum += w * ambiguous^2
        ambiguous_sum += w * ambiguous
        parallel_sum += μ2 >= pthr ? w : 0.0
        perpendicular_sum += μ2 <= othr ? w : 0.0
    end
    if wsum <= 0
        return (; loss = 0.0, ambiguous_mean = 0.0,
                parallel_fraction = 0.0, perpendicular_fraction = 0.0,
                middle_fraction = 0.0, skipped_faces = skipped)
    end
    aligned = (parallel_sum + perpendicular_sum) / wsum
    return (; loss = sqrt(loss_sum / wsum),
            ambiguous_mean = ambiguous_sum / wsum,
            parallel_fraction = parallel_sum / wsum,
            perpendicular_fraction = perpendicular_sum / wsum,
            middle_fraction = max(0.0, 1.0 - aligned),
            skipped_faces = skipped)
end

function _smooth_weights(w::Vector{Float64}, neighbors; strength::Float64, passes::Integer)
    passes <= 0 || strength <= 0 ? (return copy(w)) : nothing
    strength = clamp(strength, 0.0, 1.0)
    out = copy(w)
    tmp = similar(out)
    for _ in 1:passes
        for i in eachindex(out)
            nb = neighbors[i]
            if isempty(nb)
                tmp[i] = out[i]
            else
                s = 0.0
                for j in nb
                    s += out[j]
                end
                avg = s / length(nb)
                tmp[i] = (1 - strength) * out[i] + strength * avg
            end
        end
        out, tmp = tmp, out
        out .-= mean(out)
    end
    return out
end

function _clamp_points!(pts::Matrix{Float64}, domain)
    xmin, xmax = domain[1]
    ymin, ymax = domain[2]
    @inbounds for i in axes(pts, 1)
        pts[i, 1] = clamp(pts[i, 1], xmin + 1e-10 * (xmax - xmin), xmax - 1e-10 * (xmax - xmin))
        pts[i, 2] = clamp(pts[i, 2], ymin + 1e-10 * (ymax - ymin), ymax - 1e-10 * (ymax - ymin))
    end
    return pts
end

"""
    relax_points_velocity_alignment(points; velocity_field, steps=8, strength=0.15)

Move generators so Voronoi/power face normals prefer either the local flow
direction or its perpendicular.  For each interior face, the generator-pair
separation is rotated toward the nearer of those two directions; faces already
near parallel/perpendicular receive little force, while 45° faces receive the
largest force.  A line search accepts only moves that reduce the face-orientation
score, with an optional displacement penalty to keep the mesh close to the input
layout.

This is the orientation-changing companion to [`relax_weights`](@ref).  Power
weights can shift a face, but only generator motion can rotate its normal.
"""
function relax_points_velocity_alignment(points::AbstractMatrix;
        weights = nothing,
        domain = ((0.0, 1.0), (0.0, 1.0)),
        steps::Integer = 8,
        strength::Real = 0.15,
        velocity_field = nothing,
        velocities = nothing,
        speed_floor::Real = 1e-12,
        displacement_weight::Real = 0.02)
    velocity_field === nothing && velocities === nothing &&
        error("relax_points_velocity_alignment requires velocities or velocity_field")
    velocity_field === nothing || velocities === nothing ||
        error("provide either velocities or velocity_field, not both")
    pts = Matrix{Float64}(points)
    size(pts, 2) == 2 || error("points must be an n x 2 matrix")
    n = size(pts, 1)
    w = weights === nothing ? zeros(n) : Float64.(collect(weights))
    length(w) == n || error("weights length must match points")
    vmat = _velocity_matrix(velocities, n)
    anchor = copy(pts)
    xmin, xmax = domain[1]
    ymin, ymax = domain[2]
    h = sqrt((xmax - xmin) * (ymax - ymin) / n)
    sfloor = float(speed_floor)
    dweight = float(displacement_weight)

    mesh = power_diagram(PowerSites2D(pts; weights = w, domain))
    function score(candidate_mesh, candidate_pts)
        align = face_velocity_alignment(candidate_mesh; velocities = vmat,
                                        velocity_field, speed_floor = sfloor)
        disp = hypot.(candidate_pts[:, 1] .- anchor[:, 1],
                      candidate_pts[:, 2] .- anchor[:, 2])
        return align.loss + dweight * sqrt(mean(abs2, disp ./ max(h, eps()))), align
    end

    history = NamedTuple[]
    for _ in 1:steps
        base_score, base_align = score(mesh, pts)
        Δ = zeros(size(pts))
        counts = zeros(Int, n)
        for f in eachindex(mesh.faces.c1)
            i = mesh.faces.c1[f]
            j = mesh.faces.c2[f]
            j > 0 || continue
            vx, vy = _face_velocity(mesh, f, vmat, velocity_field)
            speed = hypot(vx, vy)
            speed > sfloor || continue
            vhat = (vx / speed, vy / speed)
            tx, ty = -vhat[2], vhat[1]
            dx = pts[j, 1] - pts[i, 1]
            dy = pts[j, 2] - pts[i, 2]
            len = hypot(dx, dy)
            len > 1e-14 || continue
            μ = (dx * vhat[1] + dy * vhat[2]) / len
            μ2 = clamp(μ^2, 0.0, 1.0)
            ambiguous = 4 * μ2 * (1 - μ2)
            if μ2 >= 0.5
                sgn = μ < 0 ? -1.0 : 1.0
                target = (sgn * vhat[1], sgn * vhat[2])
            else
                τ = (dx * tx + dy * ty) / len
                sgn = τ < 0 ? -1.0 : 1.0
                target = (sgn * tx, sgn * ty)
            end
            cx = ambiguous * (len * target[1] - dx)
            cy = ambiguous * (len * target[2] - dy)
            Δ[i, 1] -= 0.5 * cx; Δ[i, 2] -= 0.5 * cy
            Δ[j, 1] += 0.5 * cx; Δ[j, 2] += 0.5 * cy
            counts[i] += 1; counts[j] += 1
        end
        @inbounds for i in 1:n
            if counts[i] > 0
                Δ[i, 1] /= counts[i]
                Δ[i, 2] /= counts[i]
            end
        end

        accepted = false
        trial_strength = float(strength)
        trial_mesh = mesh
        trial_align = base_align
        trial_score = base_score
        for _try in 1:12
            trial_pts = pts .+ trial_strength .* Δ
            _clamp_points!(trial_pts, domain)
            trial_mesh = power_diagram(PowerSites2D(trial_pts; weights = w, domain))
            trial_score, trial_align = score(trial_mesh, trial_pts)
            if trial_score < base_score
                pts = trial_pts
                mesh = trial_mesh
                accepted = true
                break
            end
            trial_strength *= 0.5
        end
        push!(history, (; accepted,
                         gain = accepted ? trial_strength : 0.0,
                         score = accepted ? trial_score : base_score,
                         alignment = accepted ? trial_align : base_align,
                         quality = mesh_quality(mesh)))
        accepted || break
    end
    return (; points = pts, mesh, history)
end


"""
    refine_patch_points(nx, ny; refine_center=(0.5,0.5), refine_radius=0.2)

Create a simple coarse/fine transition point set: a Cartesian base lattice with
four child generators replacing cells whose centers lie inside a circular patch.
"""
function refine_patch_points(nx::Integer, ny::Integer; refine_center = (0.5, 0.5),
                             refine_radius::Real = 0.2, jitter::Real = 0.0)
    return refine_patch(nx, ny; refine_center, refine_radius, jitter).points
end

"""
    refine_patch(nx, ny; refine_center=(0.5,0.5), refine_radius=0.2)

Return `(; points, target_areas, refined)` for a coarse/fine transition test.
Coarse cells target area `dx*dy`; replaced cells contribute four child targets
of area `dx*dy/4`.
"""
function refine_patch(nx::Integer, ny::Integer; refine_center = (0.5, 0.5),
                      refine_radius::Real = 0.2, jitter::Real = 0.0)
    pts = Point2[]
    target = Float64[]
    refined = Bool[]
    dx = 1 / nx
    dy = 1 / ny
    coarse_area = dx * dy
    for j in 1:ny, i in 1:nx
        x = (i - 0.5) * dx
        y = (j - 0.5) * dy
        r = hypot(x - refine_center[1], y - refine_center[2])
        if r <= refine_radius
            for oy in (-0.25, 0.25), ox in (-0.25, 0.25)
                push!(pts, (x + ox * dx + jitter * dx * (rand() - 0.5),
                            y + oy * dy + jitter * dy * (rand() - 0.5)))
                push!(target, coarse_area / 4)
                push!(refined, true)
            end
        else
            push!(pts, (x + jitter * dx * (rand() - 0.5),
                        y + jitter * dy * (rand() - 0.5)))
            push!(target, coarse_area)
            push!(refined, false)
        end
    end
    return (; points = _points_to_matrix(pts), target_areas = target, refined)
end

"""
    relax_weights(points, target_areas; steps=25, gain=0.5, weights=zeros(n))

Small prototype weight-relaxation loop.  Holding generator positions fixed,
increase weights for cells that are too small and decrease them for cells that
are too large.  The mean weight is removed each step because power weights have
a constant gauge freedom.  Candidate updates pass through a line search on
[`mesh_loss`](@ref), so face-quality penalties can stop weights from creating
pathological sliver faces just to hit target areas.
"""
function relax_weights(points::AbstractMatrix, target_areas::AbstractVector;
                       weights = nothing, domain = ((0.0, 1.0), (0.0, 1.0)),
                       steps::Integer = 25, gain::Real = 0.5,
                       area_weight::Real = 1.0,
                       small_face_weight::Real = 0.05,
                       small_face_floor::Real = 1e-4,
                       compactness_weight::Real = 0.01,
                       compactness_floor::Real = 0.25,
                       recon_weight::Real = 0.0,
                       recon_cap::Real = 10.0,
                       velocity_alignment_weight::Real = 0.0,
                       velocities = nothing,
                       velocity_field = nothing,
                       velocity_speed_floor::Real = 1e-12,
                       smooth_strength::Real = 0.0,
                       smooth_passes::Integer = 0)
    pts = Matrix{Float64}(points)
    n = size(pts, 1)
    target = Float64.(collect(target_areas))
    length(target) == n || error("relax_weights: target_areas length must match points")
    w = weights === nothing ? zeros(n) : Float64.(collect(weights))
    vmat = _velocity_matrix(velocities, n)
    history = NamedTuple[]
    mesh = power_diagram(PowerSites2D(pts; weights = w, domain))
    loss(m) = mesh_loss(m, target;
                        area_weight, small_face_weight, small_face_floor,
                        compactness_weight, compactness_floor,
                        recon_weight, recon_cap,
                        velocity_alignment_weight,
                        velocities = vmat,
                        velocity_field,
                        velocity_speed_floor)
    for _ in 1:steps
        area = cell_areas(mesh)
        err = (target .- area) ./ max.(target, eps())
        scale = mean(target)
        base_loss = loss(mesh)
        accepted = false
        trial_gain = float(gain)
        trial_mesh = mesh
        trial_w = copy(w)
        trial_loss = base_loss
        for _try in 1:16
            trial_w .= w .+ trial_gain * scale .* err
            trial_w .-= mean(trial_w)
            if smooth_passes > 0 && smooth_strength > 0
                trial_w .= _smooth_weights(trial_w, mesh.neighbors;
                                           strength = float(smooth_strength),
                                           passes = smooth_passes)
            end
            trial_mesh = power_diagram(PowerSites2D(pts; weights = trial_w, domain))
            trial_area = cell_areas(trial_mesh)
            trial_loss = loss(trial_mesh)
            if minimum(trial_area) > 1e-10 * scale && trial_loss.total < base_loss.total
                w .= trial_w
                mesh = trial_mesh
                accepted = true
                break
            end
            trial_gain *= 0.5
        end
        push!(history, (; max_rel_area_error = maximum(abs.(err)),
                         rms_rel_area_error = sqrt(mean(abs2, err)),
                         accepted, gain = accepted ? trial_gain : 0.0,
                         loss = accepted ? trial_loss : base_loss,
                         quality = mesh_quality(mesh)))
        accepted || break
    end
    return (; mesh, weights = w, history)
end

function _svg_color(t)
    x = clamp(float(t), 0.0, 1.0)
    # blue -> white -> red, useful for signed normalized errors.
    if x < 0.5
        u = 2x
        r = round(Int, 255u)
        g = round(Int, 255u)
        b = 255
    else
        u = 2(x - 0.5)
        r = 255
        g = round(Int, 255(1 - u))
        b = round(Int, 255(1 - u))
    end
    return "rgb($r,$g,$b)"
end

"""
    write_svg(path, mesh; values=nothing, width=900, height=900)

Write an inspectable SVG of a 2-D mesh.  `values`, when supplied, colors cells
with a blue-white-red scale between its 2nd and 98th percentiles.
"""
function write_svg(path::AbstractString, mesh::PolygonMesh2D;
                   values = nothing, width::Integer = 900, height::Integer = 900,
                   stroke::AbstractString = "#222", stroke_width::Real = 0.35)
    vals = values === nothing ? nothing : Float64.(collect(values))
    vals !== nothing && length(vals) != length(mesh.cells) &&
        error("write_svg: values length must match mesh cells")
    xmin, xmax = mesh.domain[1]
    ymin, ymax = mesh.domain[2]
    sx(x) = (x - xmin) / (xmax - xmin) * width
    sy(y) = height - (y - ymin) / (ymax - ymin) * height
    lo = vals === nothing ? 0.0 : quantile(vals, 0.02)
    hi = vals === nothing ? 1.0 : quantile(vals, 0.98)
    open(path, "w") do io
        println(io, """<svg xmlns="http://www.w3.org/2000/svg" width="$width" height="$height" viewBox="0 0 $width $height">""")
        println(io, """<rect x="0" y="0" width="$width" height="$height" fill="white"/>""")
        for (i, poly) in pairs(mesh.cells)
            size(poly, 1) < 3 && continue
            pts = join(("$(sx(poly[k, 1])),$(sy(poly[k, 2]))" for k in axes(poly, 1)), " ")
            fill = vals === nothing ? "#f5f5f5" : _svg_color((vals[i] - lo) / max(hi - lo, eps()))
            println(io, """<polygon points="$pts" fill="$fill" stroke="$stroke" stroke-width="$stroke_width"/>""")
        end
        println(io, "</svg>")
    end
    return path
end

include("hydro2d.jl")
include("arepo_standard_problems.jl")

@inline function _wrap_periodic2(x::Real, lo::Real, len::Real)
    return float(lo) + mod(float(x) - float(lo), float(len))
end

function _cell_face_csr_periodic2(ncells, c1, c2, ::Type{I}) where {I<:Integer}
    counts = zeros(Int, ncells)
    for f in eachindex(c1)
        counts[Int(c1[f])] += 1
        counts[Int(c2[f])] += 1
    end
    offsets = Vector{I}(undef, ncells + 1)
    offsets[1] = one(I)
    for i in 1:ncells
        offsets[i + 1] = offsets[i] + I(counts[i])
    end
    faces = Vector{I}(undef, Int(offsets[end] - one(I)))
    signs = Vector{I}(undef, length(faces))
    cursor = Int.(offsets[1:end-1])
    for f in eachindex(c1)
        i = Int(c1[f])
        p = cursor[i]
        faces[p] = I(f)
        signs[p] = -one(I)
        cursor[i] += 1
        j = Int(c2[f])
        p = cursor[j]
        faces[p] = I(f)
        signs[p] = one(I)
        cursor[j] += 1
    end
    return offsets, faces, signs
end

function _periodic_bins2(points, domain, bins_per_axis)
    xmin, xmax = domain[1]
    ymin, ymax = domain[2]
    lx = xmax - xmin
    ly = ymax - ymin
    if bins_per_axis === nothing
        n = max(1, round(Int, sqrt(size(points, 1))))
        nbx = n
        nby = n
    elseif bins_per_axis isa Integer
        nbx = Int(bins_per_axis)
        nby = Int(bins_per_axis)
    else
        nbx = Int(bins_per_axis[1])
        nby = Int(bins_per_axis[2])
    end
    nbx > 0 && nby > 0 || error("bins_per_axis must be positive")
    bins = [Int[] for _ in 1:(nbx * nby)]
    coords = Matrix{Int}(undef, size(points, 1), 2)
    @inbounds for i in axes(points, 1)
        ix = clamp(floor(Int, (points[i, 1] - xmin) / lx * nbx) + 1, 1, nbx)
        iy = clamp(floor(Int, (points[i, 2] - ymin) / ly * nby) + 1, 1, nby)
        coords[i, 1] = ix
        coords[i, 2] = iy
        push!(bins[ix + nbx * (iy - 1)], i)
    end
    return bins, coords, (nbx, nby)
end

@inline function _wrap_bin_shift(raw::Int, nb::Int)
    shift = fld(raw - 1, nb)
    return mod(raw - 1, nb) + 1, shift
end

function _periodic_candidate_images2(i, bins, coords, nbx::Int, nby::Int,
                                     search_radius::Integer)
    out = Tuple{Int,Int,Int}[]
    ix0 = coords[i, 1]
    iy0 = coords[i, 2]
    r = Int(search_radius)
    for dy in -r:r, dx in -r:r
        ix, sx = _wrap_bin_shift(ix0 + dx, nbx)
        iy, sy = _wrap_bin_shift(iy0 + dy, nby)
        for j in bins[ix + nbx * (iy - 1)]
            push!(out, (j, sx, sy))
        end
    end
    return out
end

function _periodic_power_cell2(points, weights, i, domain; tol::Float64,
                               candidate_images = nothing)
    xmin, xmax = domain[1]
    ymin, ymax = domain[2]
    lx = xmax - xmin
    ly = ymax - ymin
    pix = points[i, 1]
    piy = points[i, 2]
    wi = weights[i]
    poly = [pix - lx piy - ly;
            pix + lx piy - ly;
            pix + lx piy + ly;
            pix - lx piy + ly]
    planes = NamedTuple[]
    images = candidate_images === nothing ?
             ((j, sx, sy) for j in axes(points, 1), sx in -1:1, sy in -1:1) :
             candidate_images
    for item in images
        j, sx, sy = item
        j == i && sx == 0 && sy == 0 && continue
        pjx = points[j, 1] + sx * lx
        pjy = points[j, 2] + sy * ly
        a = (2 * (pjx - pix), 2 * (pjy - piy))
        b = pjx^2 + pjy^2 - pix^2 - piy^2 + wi - weights[j]
        push!(planes, (; a, b, neighbor = j, sx, sy))
        poly = _clip_halfplane(poly, a, b; tol)
        size(poly, 1) == 0 && break
    end
    poly = _ensure_ccw(poly)
    return poly, planes
end

"""
    periodic_power_mesh_arrays_2d(points; domain=((0,1),(0,1)), ...)

Build a periodic 2-D power/Voronoi face table directly on the torus.  Cells are
represented by a local unwrapped polygon around each generator, so boundary
cells can have true cross-boundary neighbors without forcing split torus cells
into `PolygonMesh2D`.
"""
function periodic_power_mesh_arrays_2d(points::AbstractMatrix;
                                       weights = nothing,
                                       domain = ((0.0, 1.0), (0.0, 1.0)),
                                       T::Type{<:AbstractFloat} = Float64,
                                       index_type::Type{<:Integer} = Int32,
                                       face_velocity = nothing,
                                       cell_velocity = nothing,
                                       bins_per_axis = nothing,
                                       search_radius::Integer = 1,
                                       tol::Float64 = 1e-10)
    size(points, 2) == 2 || error("points must be n x 2")
    pts = Matrix{Float64}(points)
    n = size(pts, 1)
    w = weights === nothing ? zeros(n) : Float64.(collect(weights))
    length(w) == n || error("weights length must match point count")
    dom = ((float(domain[1][1]), float(domain[1][2])),
           (float(domain[2][1]), float(domain[2][2])))
    xmin, xmax = dom[1]
    ymin, ymax = dom[2]
    lx = xmax - xmin
    ly = ymax - ymin
    for i in 1:n
        pts[i, 1] = _wrap_periodic2(pts[i, 1], xmin, lx)
        pts[i, 2] = _wrap_periodic2(pts[i, 2], ymin, ly)
    end
    local_bins = bins_per_axis === nothing ? nothing :
                 _periodic_bins2(pts, dom, bins_per_axis)

    volume = Vector{Float64}(undef, n)
    center = Matrix{Float64}(undef, n, 2)
    c1 = Int[]
    c2 = Int[]
    area = Float64[]
    normal = NTuple{2,Float64}[]
    face_center = NTuple{2,Float64}[]
    seen = Set{Tuple{Int,Int,Int,Int}}()

    for i in 1:n
        candidate_images = if local_bins === nothing
            nothing
        else
            bins, coords, nb = local_bins
            _periodic_candidate_images2(i, bins, coords, nb[1], nb[2],
                                        search_radius)
        end
        poly, planes = _periodic_power_cell2(pts, w, i, dom; tol,
                                             candidate_images)
        volume[i] = abs(polygon_area(poly))
        ci = polygon_centroid(poly)
        center[i, 1] = _wrap_periodic2(ci[1], xmin, lx)
        center[i, 2] = _wrap_periodic2(ci[2], ymin, ly)
        m = size(poly, 1)
        m < 2 && continue
        for e in 1:m
            ep = e == m ? 1 : e + 1
            ax = poly[e, 1]; ay = poly[e, 2]
            bx = poly[ep, 1]; by = poly[ep, 2]
            len = hypot(bx - ax, by - ay)
            len <= 1e-12 && continue
            mx = 0.5 * (ax + bx)
            my = 0.5 * (ay + by)
            best = 0
            best_abs = Inf
            for p in eachindex(planes)
                v = abs(planes[p].a[1] * mx + planes[p].a[2] * my - planes[p].b)
                if v < best_abs
                    best_abs = v
                    best = p
                end
            end
            best == 0 && continue
            plane = planes[best]
            j = plane.neighbor
            j == i && continue
            sx = plane.sx
            sy = plane.sy
            key = i < j ? (i, j, sx, sy) : (j, i, -sx, -sy)
            key in seen && continue
            push!(seen, key)
            nn = hypot(plane.a[1], plane.a[2])
            nn > 0 || continue
            nx = plane.a[1] / nn
            ny = plane.a[2] / nn
            if i < j
                push!(c1, i); push!(c2, j)
                push!(normal, (nx, ny))
            else
                push!(c1, j); push!(c2, i)
                push!(normal, (-nx, -ny))
            end
            push!(area, len)
            push!(face_center, (_wrap_periodic2(mx, xmin, lx),
                                _wrap_periodic2(my, ymin, ly)))
        end
    end

    nf = length(c1)
    fc = Matrix{Float64}(undef, nf, 2)
    nm = Matrix{Float64}(undef, nf, 2)
    @inbounds for f in 1:nf
        fc[f, 1] = face_center[f][1]
        fc[f, 2] = face_center[f][2]
        nm[f, 1] = normal[f][1]
        nm[f, 2] = normal[f][2]
    end
    fvx = zeros(T, nf)
    fvy = zeros(T, nf)
    if face_velocity !== nothing
        size(face_velocity) == (nf, 2) || error("face_velocity must be nf x 2")
        fvx .= T.(@view face_velocity[:, 1])
        fvy .= T.(@view face_velocity[:, 2])
    elseif cell_velocity !== nothing
        cv = _velocity_matrix(cell_velocity, n)
        @inbounds for f in 1:nf
            i = c1[f]; j = c2[f]
            fvx[f] = T(0.5 * (cv[i, 1] + cv[j, 1]))
            fvy[f] = T(0.5 * (cv[i, 2] + cv[j, 2]))
        end
    end
    offsets, faces, signs = _cell_face_csr_periodic2(n, c1, c2, index_type)
    geom = ArepoMeshArrays2D(index_type.(c1), index_type.(c2), offsets, faces,
                             signs, T.(volume), T.(area), T.(nm[:, 1]),
                             T.(nm[:, 2]), fvx, fvy)
    return (; geom, volume, center, face_center = fc,
            generators = pts, weights = w, domain = dom,
            bins_per_axis = local_bins === nothing ? nothing : local_bins[3],
            search_radius = local_bins === nothing ? nothing : Int(search_radius))
end

include("arepo_runtime_scaffold.jl")
include("arepo_gravity_scaffold.jl")
include("arepo_pm_gravity.jl")
include("arepo_gravity_runtime.jl")
include("arepo_io_parameters.jl")
include("arepo_cosmology_coefficients.jl")
include("hydro3d.jl")
include("arepo_io_snapshots.jl")
include("arepo_io_runtime.jl")
include("tessellation3d_semantics.jl")
include("tessellation3d.jl")

end # module
