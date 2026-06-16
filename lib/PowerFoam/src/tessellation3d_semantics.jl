# Semantic scaffolding for the AREPO 3-D tessellator port.
#
# Include this after `using PowerFoam` so the mesh/hydro types already live in
# the namespace.  The file is intentionally standalone and avoids new deps.

@enum TessellationPredicatePolicy3D::UInt8 begin
    TessellationPredicateAdaptive3D = 0
    TessellationPredicateFloat64Only3D = 1
    TessellationPredicateExactCPU3D = 2
    TessellationPredicateCPUFallback3D = 3
end

"""
    TessellationPointIdentity3D

AREPO-style generator and image identity for a single point row.  The fields
mirror the source-side point bookkeeping that must survive Delaunay/Voronoi
rebuilds:

- `original_index`: generator identity before any compaction.
- `active_index`: current local row used by the rebuild.
- `owner_task` / `owner_index`: ownership metadata for exported or ghost rows.
- `timebin`: timestep bucket carried through rebuilds.
- `image_flags`: periodic or image-state flags.
- `image_shift`: nearest-image offset used to reconstruct the periodic copy.
"""
struct TessellationPointIdentity3D
    original_index::Int
    active_index::Int
    owner_task::Int
    owner_index::Int
    timebin::Int
    image_flags::UInt32
    image_shift::NTuple{3,Int}
end

TessellationPointIdentity3D(original_index::Integer, active_index::Integer;
                            owner_task::Integer = 0,
                            owner_index::Integer = 0,
                            timebin::Integer = 0,
                            image_flags::Unsigned = 0x00000000,
                            image_shift = (0, 0, 0)) =
    TessellationPointIdentity3D(Int(original_index), Int(active_index),
                                Int(owner_task), Int(owner_index), Int(timebin),
                                UInt32(image_flags),
                                (Int(image_shift[1]), Int(image_shift[2]),
                                 Int(image_shift[3])))

"""
    TessellationFaceProvenance3D

Compact face-side provenance for the production tessellator port.  The face
itself is still represented by the hydro geometry arrays, while this record
keeps the AREPO semantics that can be lost by canonical sorting:

- `face_index`: original face row before normalization.
- `c1` / `c2`: endpoint cell ids as exported by the tessellator.
- `owner_task` / `owner_index`: flux ownership after periodic or ghost mapping.
- `image_shift`: the periodic image used to build the face.
- `orientation`: orientation bit after canonicalization.
- `duplicate`: whether this row is a duplicate or mirrored export.
"""
struct TessellationFaceProvenance3D
    face_index::Int
    c1::Int
    c2::Int
    owner_task::Int
    owner_index::Int
    image_shift::NTuple{3,Int}
    orientation::Int8
    duplicate::Bool
end

function TessellationFaceProvenance3D(face_index::Integer, c1::Integer, c2::Integer;
                                      owner_task::Integer = 0,
                                      owner_index::Integer = 0,
                                      image_shift = (0, 0, 0),
                                      orientation::Integer = 1,
                                      duplicate::Bool = false)
    return TessellationFaceProvenance3D(Int(face_index), Int(c1), Int(c2),
                                        Int(owner_task), Int(owner_index),
                                        (Int(image_shift[1]), Int(image_shift[2]),
                                         Int(image_shift[3])),
                                        Int8(orientation), duplicate)
end

"""
    TessellationFallbackCounters3D

Mutable counter block for predicate and topology fallbacks.  The field names
follow the AREPO gate vocabulary so the later CPU reference path can expose the
same measurements:

- `count_in_sphere_tests` / `count_in_sphere_tests_exact`
- `count_convex_edge_test` / `count_convex_edge_test_exact`
- `count_in_tetra` / `count_in_tetra_exact`

The remaining fields track explicit retries and fallback exits that the port
needs to surface when an adaptive predicate or topology step cannot stay on the
fast path.
"""
mutable struct TessellationFallbackCounters3D
    count_in_sphere_tests::Int
    count_in_sphere_tests_exact::Int
    count_convex_edge_test::Int
    count_convex_edge_test_exact::Int
    count_in_tetra::Int
    count_in_tetra_exact::Int
    orient3d_tests::Int
    orient3d_tests_exact::Int
    exact_cpu_fallbacks::Int
    gpu_fallbacks::Int
    topology_retries::Int
    degenerate_faces::Int
    skipped_infinite_tetra::Int
end

TessellationFallbackCounters3D() =
    TessellationFallbackCounters3D(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

function reset!(c::TessellationFallbackCounters3D)
    c.count_in_sphere_tests = 0
    c.count_in_sphere_tests_exact = 0
    c.count_convex_edge_test = 0
    c.count_convex_edge_test_exact = 0
    c.count_in_tetra = 0
    c.count_in_tetra_exact = 0
    c.orient3d_tests = 0
    c.orient3d_tests_exact = 0
    c.exact_cpu_fallbacks = 0
    c.gpu_fallbacks = 0
    c.topology_retries = 0
    c.degenerate_faces = 0
    c.skipped_infinite_tetra = 0
    return c
end

@inline function record_in_sphere_test!(c::TessellationFallbackCounters3D;
                                        exact::Bool = false)
    c.count_in_sphere_tests += 1
    exact && (c.count_in_sphere_tests_exact += 1)
    return c
end

@inline function record_convex_edge_test!(c::TessellationFallbackCounters3D;
                                          exact::Bool = false)
    c.count_convex_edge_test += 1
    exact && (c.count_convex_edge_test_exact += 1)
    return c
end

@inline function record_in_tetra_test!(c::TessellationFallbackCounters3D;
                                       exact::Bool = false)
    c.count_in_tetra += 1
    exact && (c.count_in_tetra_exact += 1)
    return c
end

@inline function record_orient3d_test!(c::TessellationFallbackCounters3D;
                                       exact::Bool = false)
    c.orient3d_tests += 1
    exact && (c.orient3d_tests_exact += 1)
    return c
end

@inline record_exact_cpu_fallback!(c::TessellationFallbackCounters3D) =
    (c.exact_cpu_fallbacks += 1; c)

@inline record_gpu_fallback!(c::TessellationFallbackCounters3D) =
    (c.gpu_fallbacks += 1; c)

@inline record_topology_retry!(c::TessellationFallbackCounters3D) =
    (c.topology_retries += 1; c)

@inline record_degenerate_face!(c::TessellationFallbackCounters3D) =
    (c.degenerate_faces += 1; c)

@inline record_skipped_infinite_tetra!(c::TessellationFallbackCounters3D) =
    (c.skipped_infinite_tetra += 1; c)
