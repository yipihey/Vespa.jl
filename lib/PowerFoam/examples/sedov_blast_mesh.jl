#!/usr/bin/env julia

using Printf
using PowerFoam

const OUTDIR = joinpath(@__DIR__, "out")
const SCALE = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 1.0
const STEPS = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : (SCALE > 1 ? 14 : 35)
const TAG = SCALE == 1 ? "" : "_$(replace(string(SCALE), "." => "p"))x"

function jittered_lattice(n; jitter = 0.18)
    pts = Matrix{Float64}(undef, n * n, 2)
    q = 1
    for j in 1:n, i in 1:n
        x = (i - 0.5) / n
        y = (j - 0.5) / n
        # Deterministic quasi-jitter; enough to avoid perfect lattice degeneracy.
        dx = jitter / n * sin(12.9898i + 78.233j)
        dy = jitter / n * sin(39.3467i + 11.135j)
        pts[q, 1] = clamp(x + dx, 1e-6, 1 - 1e-6)
        pts[q, 2] = clamp(y + dy, 1e-6, 1 - 1e-6)
        q += 1
    end
    return pts
end

function sedov_targets(points; center = (0.5, 0.5), shock_radius = 0.33,
                       shell_width = 0.045, core_width = 0.10)
    r = hypot.(points[:, 1] .- center[1], points[:, 2] .- center[2])
    shell = exp.(-0.5 .* ((r .- shock_radius) ./ shell_width) .^ 2)
    core = exp.(-0.5 .* (r ./ core_width) .^ 2)
    resolution_density = 1.0 .+ 7.0 .* shell .+ 2.0 .* core
    target = 1.0 ./ resolution_density
    target ./= sum(target)
    return target
end

function summarize(label, mesh, target)
    q = mesh_quality(mesh)
    area = cell_areas(mesh)
    rel = abs.(area .- target) ./ target
    @printf("%-32s rms_area=%9.4e max_area=%9.4e small_p01=%9.4e cond_max=%7.3f faces=%d\n",
            label, sqrt(sum(abs2, rel) / length(rel)), maximum(rel),
            q.small_face_p01, q.recon_cond_max, q.faces)
end

mkpath(OUTDIR)

points = jittered_lattice(round(Int, 24 * SCALE))
target = sedov_targets(points)

vor = power_diagram(PowerSites2D(points))
area_only = relax_weights(points, target; steps = STEPS, gain = 0.30,
                          small_face_weight = 0.0, compactness_weight = 0.0).mesh
face_aware = relax_weights(points, target; steps = STEPS, gain = 0.30,
                           small_face_weight = 0.03, small_face_floor = 1e-4,
                           compactness_weight = 0.02,
                           smooth_strength = 0.45, smooth_passes = 1).mesh

println("Sedov-Taylor blast-wave target mesh prototype")
println("cells: $(size(points, 1))")
println("scale: $SCALE, relaxation steps: $STEPS")
summarize("ordinary Voronoi", vor, target)
summarize("area-only weighted power", area_only, target)
summarize("smoothed face-aware power", face_aware, target)

relerr(mesh) = (cell_areas(mesh) .- target) ./ target
write_svg(joinpath(OUTDIR, "sedov_voronoi$(TAG).svg"), vor; values = relerr(vor))
write_svg(joinpath(OUTDIR, "sedov_area_only$(TAG).svg"), area_only; values = relerr(area_only))
write_svg(joinpath(OUTDIR, "sedov_face_aware$(TAG).svg"), face_aware; values = relerr(face_aware))

println("wrote SVGs to $OUTDIR")
