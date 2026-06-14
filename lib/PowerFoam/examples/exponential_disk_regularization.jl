#!/usr/bin/env julia

using Printf
using PowerFoam

const OUTDIR = joinpath(@__DIR__, "out")
const SCALE = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 1.0
const STEPS = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : (SCALE > 1 ? 16 : 40)
const TAG = SCALE == 1 ? "" : "_$(replace(string(SCALE), "." => "p"))x"

function disk_points(; nrings = 18, rd = 0.28, rmax = 0.96)
    pts = Tuple{Float64,Float64}[]
    push!(pts, (0.0, 0.0))
    for k in 1:nrings
        r = rmax * k / nrings
        sigma = exp(-r / rd)
        # More azimuthal samples in the dense inner disk, fewer outside.
        nphi = max(10, round(Int, 2pi * r * nrings * sqrt(sigma) / 0.55))
        phase = 0.37k
        for m in 0:(nphi - 1)
            phi = 2pi * (m + phase) / nphi
            push!(pts, (r * cos(phi), r * sin(phi)))
        end
    end
    mat = Matrix{Float64}(undef, length(pts), 2)
    for (i, p) in pairs(pts)
        mat[i, 1] = p[1]
        mat[i, 2] = p[2]
    end
    return mat
end

function exponential_targets(points; rd = 0.28, floor = 0.05)
    r = hypot.(points[:, 1], points[:, 2])
    sigma = floor .+ exp.(-r ./ rd)
    target = 1.0 ./ sigma
    target ./= sum(target)
    target .*= 4.0                         # domain is [-1, 1]^2
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

points = disk_points(nrings = round(Int, 18 * SCALE))
target = exponential_targets(points)
domain = ((-1.0, 1.0), (-1.0, 1.0))

vor = power_diagram(PowerSites2D(points; domain))
area_only = relax_weights(points, target; domain, steps = STEPS, gain = 0.28,
                          small_face_weight = 0.0, compactness_weight = 0.0).mesh
face_aware = relax_weights(points, target; domain, steps = STEPS, gain = 0.28,
                           small_face_weight = 0.01, small_face_floor = 1e-4,
                           compactness_weight = 0.01).mesh

println("Exponential disk mesh regularization prototype")
println("cells: $(size(points, 1))")
println("scale: $SCALE, relaxation steps: $STEPS")
summarize("ordinary Voronoi", vor, target)
summarize("area-only weighted power", area_only, target)
summarize("face-aware weighted power", face_aware, target)

relerr(mesh) = (cell_areas(mesh) .- target) ./ target
write_svg(joinpath(OUTDIR, "exponential_disk_voronoi$(TAG).svg"), vor; values = relerr(vor))
write_svg(joinpath(OUTDIR, "exponential_disk_area_only$(TAG).svg"), area_only; values = relerr(area_only))
write_svg(joinpath(OUTDIR, "exponential_disk_face_aware$(TAG).svg"), face_aware; values = relerr(face_aware))

println("wrote SVGs to $OUTDIR")
