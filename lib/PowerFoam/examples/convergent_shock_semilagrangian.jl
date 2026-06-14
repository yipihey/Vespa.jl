#!/usr/bin/env julia

using Printf
using PowerFoam

const OUTDIR = joinpath(@__DIR__, "out")
const SCALE = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 1.0
const N = round(Int, 32 * SCALE)
const STEPS = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : (SCALE > 1 ? 14 : 28)
const TAG = SCALE == 1 ? "" : "_$(replace(string(SCALE), "." => "p"))x"

function jittered_lattice(n; jitter = 0.12)
    pts = Matrix{Float64}(undef, n * n, 2)
    q = 1
    for j in 1:n, i in 1:n
        x = (i - 0.5) / n
        y = (j - 0.5) / n
        dx = jitter / n * sin(12.9898i + 78.233j)
        dy = jitter / n * sin(39.3467i + 11.135j)
        pts[q, 1] = clamp(x + dx, 1e-6, 1 - 1e-6)
        pts[q, 2] = clamp(y + dy, 1e-6, 1 - 1e-6)
        q += 1
    end
    return pts
end

function radial_infall(points; center = (0.5, 0.5), shock_radius = 0.25,
                       shock_width = 0.035, displacement = 0.18,
                       lagrangian_fraction = 1.0)
    out = similar(points)
    for i in axes(points, 1)
        dx = points[i, 1] - center[1]
        dy = points[i, 2] - center[2]
        r = hypot(dx, dy)
        if r < 1e-12
            out[i, 1] = points[i, 1]
            out[i, 2] = points[i, 2]
            continue
        end
        # Preshock gas falls inward; postshock gas is almost stalled.  Full
        # Lagrangian motion therefore piles points at the shock and dilutes the
        # still-infalling upstream region.
        preshock = 0.5 * (1 + tanh((r - shock_radius) / shock_width))
        dr = lagrangian_fraction * displacement * preshock
        rnew = max(0.015, r - dr)
        out[i, 1] = center[1] + rnew * dx / r
        out[i, 2] = center[2] + rnew * dy / r
    end
    return out
end

function shock_targets(points; center = (0.5, 0.5), shock_radius = 0.25,
                       shell_width = 0.028, preshock_width = 0.16)
    r = hypot.(points[:, 1] .- center[1], points[:, 2] .- center[2])
    shock = exp.(-0.5 .* ((r .- shock_radius) ./ shell_width) .^ 2)
    upstream = @. (r > shock_radius) * exp(-max(r - shock_radius, 0.0) / preshock_width)
    core = exp.(-0.5 .* (r ./ 0.10) .^ 2)
    resolution_density = 1.0 .+ 9.0 .* shock .+ 2.0 .* upstream .+ 1.5 .* core
    target = 1.0 ./ resolution_density
    target ./= sum(target)
    return target
end

function displacement_stats(points, initial)
    d = hypot.(points[:, 1] .- initial[:, 1], points[:, 2] .- initial[:, 2])
    return (; mean = sum(d) / length(d), max = maximum(d))
end

function summarize(label, mesh, target, initial)
    q = mesh_quality(mesh)
    area = cell_areas(mesh)
    rel = abs.(area .- target) ./ target
    disp = displacement_stats(mesh.generators, initial)
    @printf("%-32s rms_area=%9.4e max_area=%9.4e small_p01=%9.4e cond_max=%7.3f mean_disp=%7.4f max_disp=%7.4f faces=%d\n",
            label, sqrt(sum(abs2, rel) / length(rel)), maximum(rel),
            q.small_face_p01, q.recon_cond_max, disp.mean, disp.max, q.faces)
end

mkpath(OUTDIR)

initial = jittered_lattice(N)
target = shock_targets(initial)

eulerian_points = initial
lagrangian_points = radial_infall(initial; lagrangian_fraction = 1.0)
semi_points = radial_infall(initial; lagrangian_fraction = 0.35)

eulerian = power_diagram(PowerSites2D(eulerian_points))
lagrangian = power_diagram(PowerSites2D(lagrangian_points))
semi = relax_weights(semi_points, target; steps = STEPS, gain = 0.32,
                     small_face_weight = 0.04, small_face_floor = 1e-4,
                     compactness_weight = 0.02,
                     smooth_strength = 0.45, smooth_passes = 1).mesh

println("Convergent-flow central-shock semi-Lagrangian mesh prototype")
println("cells: $(size(initial, 1))")
println("scale: $SCALE, relaxation steps: $STEPS")
summarize("Eulerian fixed Voronoi", eulerian, target, initial)
summarize("fully Lagrangian Voronoi", lagrangian, target, initial)
summarize("semi-Lagrangian power mesh", semi, target, initial)

relerr(mesh) = (cell_areas(mesh) .- target) ./ target
write_svg(joinpath(OUTDIR, "convergent_shock_eulerian$(TAG).svg"), eulerian; values = relerr(eulerian))
write_svg(joinpath(OUTDIR, "convergent_shock_lagrangian$(TAG).svg"), lagrangian; values = relerr(lagrangian))
write_svg(joinpath(OUTDIR, "convergent_shock_power$(TAG).svg"), semi; values = relerr(semi))

println("wrote SVGs to $OUTDIR")
