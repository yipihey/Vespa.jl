#!/usr/bin/env julia

using DelimitedFiles
using Printf
using PowerFoam

const ROOT = @__DIR__
const OUTDIR = joinpath(ROOT, "out")
const N = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 64
const TIME_MAX = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 2.0
const SHELL_WIDTH = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 0.18
const DISPLACEMENT = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 0.26
const AREA_MODE = length(ARGS) >= 5 ? Symbol(ARGS[5]) : :exact
const ALIGN_STRENGTH = length(ARGS) >= 6 ? parse(Float64, ARGS[6]) : 0.0
const ALIGN_STEPS = length(ARGS) >= 7 ? parse(Int, ARGS[7]) : 6

const BOX = 6.0
const CENTER = (0.5 * BOX, 0.5 * BOX)
const GAMMA = 5.0 / 3.0
const RHO0 = 1.0
const VR0 = -1.0
const PRESSURE0 = 1.0e-4
const SHOCK_RADIUS = TIME_MAX / 3.0

function jittered_lattice(n; jitter = 0.08)
    pts = Matrix{Float64}(undef, n * n, 2)
    q = 1
    for j in 1:n, i in 1:n
        x = (i - 0.5) * BOX / n
        y = (j - 0.5) * BOX / n
        dx = jitter * BOX / n * sin(12.9898i + 78.233j)
        dy = jitter * BOX / n * sin(39.3467i + 11.135j)
        pts[q, 1] = clamp(x + dx, 1e-8, BOX - 1e-8)
        pts[q, 2] = clamp(y + dy, 1e-8, BOX - 1e-8)
        q += 1
    end
    return pts
end

function noh_powerfoam_points(points; center = CENTER, shock_radius = SHOCK_RADIUS,
                              shell_width = SHELL_WIDTH, displacement = DISPLACEMENT)
    out = similar(points)
    for i in axes(points, 1)
        dx = points[i, 1] - center[1]
        dy = points[i, 2] - center[2]
        r = hypot(dx, dy)
        if r < 1e-12
            out[i, :] .= points[i, :]
            continue
        end

        shell = tanh((r - shock_radius) / shell_width) *
                exp(-0.5 * ((r - shock_radius) / (3.0 * shell_width))^2)
        dr = -displacement * shell
        rnew = max(r + dr, 0.025 * BOX)
        out[i, 1] = center[1] + rnew * dx / r
        out[i, 2] = center[2] + rnew * dy / r
    end
    return clamp.(out, 1e-8, BOX - 1e-8)
end

function radial_velocity(points; center = CENTER)
    vx = zeros(size(points, 1))
    vy = zeros(size(points, 1))
    for i in axes(points, 1)
        dx = points[i, 1] - center[1]
        dy = points[i, 2] - center[2]
        r = hypot(dx, dy)
        if r > 1e-12
            vx[i] = VR0 * dx / r
            vy[i] = VR0 * dy / r
        end
    end
    return vx, vy
end

function radial_velocity_at(xy; center = CENTER)
    dx = xy[1] - center[1]
    dy = xy[2] - center[2]
    r = hypot(dx, dy)
    r <= 1e-12 && return (0.0, 0.0)
    return (VR0 * dx / r, VR0 * dy / r)
end

function write_case(label, points, areas = nothing)
    mesh = nothing
    if areas === nothing
        mesh = power_diagram(PowerSites2D(points; domain = ((0.0, BOX), (0.0, BOX))))
        areas = cell_areas(mesh)
    end

    vx, vy = radial_velocity(points)
    rho = fill(RHO0, size(points, 1))
    pressure = fill(PRESSURE0, size(points, 1))
    mass = rho .* areas
    uthermal = pressure ./ ((GAMMA - 1.0) .* rho)
    ids = collect(1:size(points, 1))

    table = hcat(ids, points[:, 1], points[:, 2], areas, mass,
                 vx, vy, uthermal, pressure, rho)
    path = joinpath(OUTDIR, "$(label)_ic.csv")
    writedlm(path, table, ',')

    cond = mesh === nothing ? NaN : mesh_quality(mesh).recon_cond_max
    @printf("%-10s cells=%d area_min=%.8e area_max=%.8e mass=%.8e cond_max=%s\n",
            label, size(points, 1), minimum(areas), maximum(areas), sum(mass),
            isnan(cond) ? "n/a" : @sprintf("%.3f", cond))
    return (; label, path, mesh, areas)
end

mkpath(OUTDIR)

standard = jittered_lattice(N)
powerfoam_base = noh_powerfoam_points(standard)
powerfoam = if ALIGN_STRENGTH > 0
    before = power_diagram(PowerSites2D(powerfoam_base; domain = ((0.0, BOX), (0.0, BOX))))
    before_align = face_velocity_alignment(before; velocity_field = radial_velocity_at)
    aligned = relax_points_velocity_alignment(powerfoam_base;
        domain = ((0.0, BOX), (0.0, BOX)),
        velocity_field = radial_velocity_at,
        steps = ALIGN_STEPS,
        strength = ALIGN_STRENGTH,
        displacement_weight = 0.01)
    after_align = face_velocity_alignment(aligned.mesh; velocity_field = radial_velocity_at)
    @printf("flow alignment: loss %.5f -> %.5f, middle_fraction %.5f -> %.5f, accepted_steps=%d\n",
            before_align.loss, after_align.loss,
            before_align.middle_fraction, after_align.middle_fraction,
            count(h -> h.accepted, aligned.history))
    aligned.points
else
    powerfoam_base
end

base_area = fill(BOX * BOX / (N * N), N * N)
cases = if AREA_MODE == :exact
    [write_case("standard", standard), write_case("powerfoam", powerfoam)]
elseif AREA_MODE == :uniform_mass
    [write_case("standard", standard, base_area), write_case("powerfoam", powerfoam, base_area)]
else
    error("unknown area mode '$AREA_MODE'; use exact or uniform_mass")
end

metadata = joinpath(OUTDIR, "metadata.txt")
open(metadata, "w") do io
    println(io, "problem,noh_2d")
    println(io, "n,$N")
    println(io, "cells,$(N * N)")
    println(io, "box,$BOX")
    println(io, "gamma,$GAMMA")
    println(io, "rho0,$RHO0")
    println(io, "vr0,$VR0")
    println(io, "pressure0,$PRESSURE0")
    println(io, "time_max,$TIME_MAX")
    println(io, "shock_radius,$SHOCK_RADIUS")
    println(io, "shell_width,$SHELL_WIDTH")
    println(io, "displacement,$DISPLACEMENT")
    println(io, "area_mode,$AREA_MODE")
    println(io, "align_strength,$ALIGN_STRENGTH")
    println(io, "align_steps,$ALIGN_STEPS")
    for c in cases
        println(io, "$(c.label)_table,$(basename(c.path))")
    end
end

println("wrote Noh IC tables to $OUTDIR")
