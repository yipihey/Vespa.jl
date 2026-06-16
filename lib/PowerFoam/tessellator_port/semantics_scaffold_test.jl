using Test
using PowerFoam

@testset "tessellation semantics scaffold" begin
    @test TessellationPredicateAdaptive3D != TessellationPredicateExactCPU3D
    @test TessellationPredicateFloat64Only3D != TessellationPredicateCPUFallback3D

    point = TessellationPointIdentity3D(12, 3; owner_task = 7, owner_index = 9,
                                        timebin = 4, image_flags = 0x11,
                                        image_shift = (1, -1, 0))
    @test point.original_index == 12
    @test point.active_index == 3
    @test point.owner_task == 7
    @test point.owner_index == 9
    @test point.timebin == 4
    @test point.image_flags == 0x11
    @test point.image_shift == (1, -1, 0)

    face = TessellationFaceProvenance3D(5, 2, 7; owner_task = 8, owner_index = 4,
                                         image_shift = (0, 0, -1), orientation = -1,
                                         duplicate = true)
    @test face.face_index == 5
    @test face.c1 == 2
    @test face.c2 == 7
    @test face.owner_task == 8
    @test face.owner_index == 4
    @test face.image_shift == (0, 0, -1)
    @test face.orientation == -1
    @test face.duplicate

    counters = TessellationFallbackCounters3D()
    record_in_sphere_test!(counters)
    record_in_sphere_test!(counters; exact = true)
    record_convex_edge_test!(counters; exact = true)
    record_in_tetra_test!(counters)
    record_orient3d_test!(counters; exact = true)
    record_exact_cpu_fallback!(counters)
    record_gpu_fallback!(counters)
    record_topology_retry!(counters)
    record_degenerate_face!(counters)
    record_skipped_infinite_tetra!(counters)

    @test counters.count_in_sphere_tests == 2
    @test counters.count_in_sphere_tests_exact == 1
    @test counters.count_convex_edge_test == 1
    @test counters.count_convex_edge_test_exact == 1
    @test counters.count_in_tetra == 1
    @test counters.count_in_tetra_exact == 0
    @test counters.orient3d_tests == 1
    @test counters.orient3d_tests_exact == 1
    @test counters.exact_cpu_fallbacks == 1
    @test counters.gpu_fallbacks == 1
    @test counters.topology_retries == 1
    @test counters.degenerate_faces == 1
    @test counters.skipped_infinite_tetra == 1

    PowerFoam.reset!(counters)
    @test all(getfield(counters, name) == 0 for name in fieldnames(typeof(counters)))
end
