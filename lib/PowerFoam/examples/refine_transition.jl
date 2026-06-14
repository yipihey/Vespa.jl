#!/usr/bin/env julia

using Printf
using PowerFoam

function show_quality(label, mesh, target)
    q = mesh_quality(mesh)
    loss = mesh_loss(mesh, target)
    area = cell_areas(mesh)
    rel = abs.(area .- target) ./ target
    println(label)
    @printf("  cells                 %d\n", q.cells)
    @printf("  faces                 %d\n", q.faces)
    @printf("  total area            %.16f\n", q.volume)
    @printf("  max target-area err   %.6e\n", maximum(rel))
    @printf("  rms target-area err   %.6e\n", sqrt(sum(abs2, rel) / length(rel)))
    @printf("  cell area ratio       %.6e\n", q.area_ratio)
    @printf("  small-face p01        %.6e\n", q.small_face_p01)
    @printf("  compactness median    %.6e\n", q.compactness_median)
    @printf("  centroid offset max   %.6e\n", q.centroid_offset_max)
    @printf("  recon cond median     %.6e\n", q.recon_cond_median)
    @printf("  recon cond max        %.6e\n", q.recon_cond_max)
    @printf("  loss total            %.6e\n", loss.total)
    @printf("  loss area             %.6e\n", loss.area)
    @printf("  loss small-face       %.6e\n", loss.small_face)
    @printf("  loss compactness      %.6e\n", loss.compactness)
end

patch = refine_patch(16, 16; refine_radius = 0.18)
vor = power_diagram(PowerSites2D(patch.points))
area_only = relax_weights(patch.points, patch.target_areas; steps = 30, gain = 0.35,
                          small_face_weight = 0.0, compactness_weight = 0.0).mesh
face_aware = relax_weights(patch.points, patch.target_areas; steps = 30, gain = 0.35,
                           small_face_weight = 0.03, small_face_floor = 1e-4,
                           compactness_weight = 0.02,
                           smooth_strength = 0.5, smooth_passes = 1).mesh

show_quality("ordinary Voronoi", vor, patch.target_areas)
println()
show_quality("area-only weighted power", area_only, patch.target_areas)
println()
show_quality("smoothed face-aware weighted power", face_aware, patch.target_areas)
