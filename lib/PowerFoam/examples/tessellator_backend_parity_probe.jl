using KernelAbstractions
using PowerFoam
using Printf

const KA = KernelAbstractions
const DOMAIN = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))

sample_points() = Float64[
    0.15 0.20 0.25
    0.72 0.18 0.31
    0.28 0.76 0.22
    0.64 0.67 0.79
    0.41 0.47 0.63
]

function try_metal_backend()
    metal_path = Base.find_package("Metal")
    metal_path === nothing &&
        return (available = false,
                reason = "Package Metal is not available in the active project environment.")
    try
        Base.eval(Main, :(import Metal))
        metal = getfield(Main, :Metal)
        return (available = true, backend = metal.MetalBackend(),
                reason = "Loaded Metal from $(metal_path).")
    catch err
        return (available = false,
                reason = sprint(showerror, err))
    end
end

hostsum(x) = sum(Array(x))
hostmax(x) = maximum(Array(x))

function run_probe(be, points)
    ref = build_arepo_tessellation_3d(points;
                                      domain = DOMAIN,
                                      algorithm = :arepo_delaunay_reference,
                                      return_delaunay = true,
                                      min_face_surface_fraction = 0.0,
                                      backend = be)
    images = periodic_point_images_soa_3d(be, points; domain = DOMAIN, T = Float32)
    dense = dense_candidate_pairs_soa_3d(be, points;
                                         domain = DOMAIN,
                                         bins_per_axis = 2,
                                         search_radius = 1,
                                         T = Float32)
    stencil = pack_candidate_stencil_soa_3d(be, dense, size(points, 1);
                                            max_candidates_per_source = 64)
    delaunay = delaunay_soa_3d(ref.delaunay; T = Float32)
    delaunay_be = be isa KA.CPU ? delaunay : to_backend(be, delaunay; T = Float32)
    circum = recompute_circumcenters_soa_3d(be, delaunay_be)
    predicates = candidate_tetra_predicates_soa_3d(be, points, stencil, delaunay_be;
                                                   domain = DOMAIN,
                                                   T = Float32)
    conflict = candidate_conflict_face_rows_soa_3d(be, predicates, delaunay_be)
    boundary = candidate_boundary_face_rows_soa_3d(be, conflict)
    tess_soa = tessellation_soa_3d(ref; T = Float32)
    tess_soa_be = be isa KA.CPU ? tess_soa : to_backend(be, tess_soa; T = Float32)
    return (
        backend = string(nameof(typeof(be))),
        faces = length(ref.geom.c1),
        tetras = length(ref.delaunay.tetras),
        image_rows = length(images.point_x),
        dense_active = Int(hostsum(dense.active)),
        max_candidates = Int(hostmax(stencil.counts)),
        valid_circumcenters = Int(hostsum(circum.valid)),
        valid_predicates = Int(hostsum(predicates.valid)),
        active_conflict_rows = Int(hostsum(conflict.active)),
        boundary_rows = Int(hostsum(boundary.boundary)),
        soa_faces = length(Array(tess_soa_be.geom.c1)),
    )
end

function print_summary(label, summary)
    println("[$label]")
    for key in (:backend, :faces, :tetras, :image_rows, :dense_active,
                :max_candidates, :valid_circumcenters, :valid_predicates,
                :active_conflict_rows, :boundary_rows, :soa_faces)
        println(@sprintf("  %-20s %s", string(key) * ":", getproperty(summary, key)))
    end
end

function main()
    points = sample_points()
    println("PowerFoam tessellator backend parity probe")
    println("points=$(size(points, 1)) domain=$(DOMAIN)")
    println()

    cpu_summary = run_probe(KA.CPU(), points)
    print_summary("cpu", cpu_summary)
    println()

    metal = try_metal_backend()
    if !metal.available
        println("[metal]")
        println("  status: skipped")
        println("  reason: $(metal.reason)")
        return
    end

    println("[metal]")
    println("  status: available")
    println("  reason: $(metal.reason)")
    metal_summary = run_probe(metal.backend, points)
    print_summary("metal-run", metal_summary)
end

main()
