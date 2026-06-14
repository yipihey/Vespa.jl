#!/usr/bin/env julia

using DelimitedFiles
using Printf
using PowerFoam

const ROOT = @__DIR__
const OUTDIR = joinpath(ROOT, "out")
const N = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 64
const SHOCK_RADIUS = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 0.235
const SHELL_WIDTH = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 0.040
const DISPLACEMENT = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 0.075
const LAGRANGIAN_FRACTION = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : 0.45
const AREA_MODE = length(ARGS) >= 6 ? Symbol(ARGS[6]) : :exact
const BOX = 1.0
const GAMMA = 5.0 / 3.0
const RHO0 = 1.0
const PRESSURE_FLOOR = 1e-5
const ENERGY = 1.0
const BOMB_RADIUS = 2.5 / N

function jittered_lattice(n; jitter = 0.10)
    pts = Matrix{Float64}(undef, n * n, 2)
    q = 1
    for j in 1:n, i in 1:n
        x = (i - 0.5) / n
        y = (j - 0.5) / n
        dx = jitter / n * sin(12.9898i + 78.233j)
        dy = jitter / n * sin(39.3467i + 11.135j)
        pts[q, 1] = mod(x + dx, BOX)
        pts[q, 2] = mod(y + dy, BOX)
        q += 1
    end
    return pts
end

function semi_lagrangian_sedov(points; center = (0.5, 0.5),
                               shock_radius = SHOCK_RADIUS,
                               shell_width = SHELL_WIDTH,
                               displacement = DISPLACEMENT,
                               lagrangian_fraction = LAGRANGIAN_FRACTION)
    out = similar(points)
    for i in axes(points, 1)
        dx = points[i, 1] - center[1]
        dy = points[i, 2] - center[2]
        r = hypot(dx, dy)
        if r < 1e-12
            out[i, :] .= points[i, :]
            continue
        end

        # Pull points mildly toward the expected shock.  This is deliberately
        # weaker than full Lagrangian motion: we want a resolved shell without
        # evacuating the upstream mesh.
        toward_shell = tanh((r - shock_radius) / shell_width)
        dr = -lagrangian_fraction * displacement * toward_shell *
             exp(-0.5 * ((r - shock_radius) / 0.18)^2)
        rnew = clamp(r + dr, 0.015, 0.70)
        out[i, 1] = center[1] + rnew * dx / r
        out[i, 2] = center[2] + rnew * dy / r
    end
    return clamp.(out, 1e-8, 1 - 1e-8)
end

function radial_map_radius(r; shock_radius = SHOCK_RADIUS,
                           shell_width = SHELL_WIDTH,
                           displacement = DISPLACEMENT,
                           lagrangian_fraction = LAGRANGIAN_FRACTION)
    toward_shell = tanh((r - shock_radius) / shell_width)
    dr = -lagrangian_fraction * displacement * toward_shell *
         exp(-0.5 * ((r - shock_radius) / 0.18)^2)
    return clamp(r + dr, 0.015, 0.70)
end

function semi_jacobian_areas(points; center = (0.5, 0.5))
    dx0 = 1.0 / N
    base = dx0 * dx0
    areas = similar(points[:, 1])
    for i in axes(points, 1)
        x = points[i, 1] - center[1]
        y = points[i, 2] - center[2]
        r = hypot(x, y)
        if r < 1e-8
            areas[i] = base
            continue
        end
        eps = max(1e-5, 1e-3 * r)
        rp = radial_map_radius(r + eps)
        rm = radial_map_radius(max(r - eps, 1e-8))
        rnew = radial_map_radius(r)
        drdr = (rp - rm) / ((r + eps) - max(r - eps, 1e-8))
        jac = max(0.2, min(5.0, (rnew / r) * drdr))
        areas[i] = base * jac
    end
    areas .*= 1.0 / sum(areas)
    return areas
end

function sedov_state(points, areas; center = (0.5, 0.5))
    ncell = size(points, 1)
    rho = fill(RHO0, ncell)
    vx = zeros(ncell)
    vy = zeros(ncell)
    pressure = fill(PRESSURE_FLOOR, ncell)

    r = hypot.(points[:, 1] .- center[1], points[:, 2] .- center[2])
    inside = findall(<=(BOMB_RADIUS), r)
    isempty(inside) && error("Bomb radius $BOMB_RADIUS selected no cells")

    # Inject total energy as thermal energy over the selected cell masses.
    bomb_area = sum(areas[inside])
    pressure[inside] .= (GAMMA - 1.0) * ENERGY / bomb_area

    mass = rho .* areas
    uthermal = pressure ./ ((GAMMA - 1.0) .* rho)
    return (; rho, vx, vy, pressure, mass, uthermal, bomb_area, ncells_bomb = length(inside))
end

function write_case(label, points, areas = nothing)
    mesh = nothing
    if areas === nothing
        mesh = power_diagram(PowerSites2D(points))
        areas = cell_areas(mesh)
    end
    state = sedov_state(points, areas)
    ids = collect(1:size(points, 1))

    table = hcat(ids, points[:, 1], points[:, 2], areas, state.mass,
                 state.vx, state.vy, state.uthermal, state.pressure, state.rho)
    path = joinpath(OUTDIR, "$(label)_ic.csv")
    writedlm(path, table, ',')

    cond = mesh === nothing ? NaN : mesh_quality(mesh).recon_cond_max
    @printf("%-10s cells=%d area_min=%.8e area_max=%.8e bomb_cells=%d bomb_area=%.8e cond_max=%s\n",
            label, size(points, 1), minimum(areas), maximum(areas),
            state.ncells_bomb, state.bomb_area,
            isnan(cond) ? "n/a" : @sprintf("%.3f", cond))
    return (; label, path, mesh, areas, state)
end

mkpath(OUTDIR)

standard = jittered_lattice(N)
semi = semi_lagrangian_sedov(standard)

cases = if AREA_MODE == :exact
    [write_case("standard", standard), write_case("semi", semi)]
elseif AREA_MODE == :fast_jacobian
    base = fill(1.0 / (N * N), N * N)
    [write_case("standard", standard, base), write_case("semi", semi, semi_jacobian_areas(standard))]
else
    error("unknown area mode '$AREA_MODE'; use exact or fast_jacobian")
end

metadata = joinpath(OUTDIR, "metadata.txt")
open(metadata, "w") do io
    println(io, "n,$N")
    println(io, "cells,$(N * N)")
    println(io, "box,$BOX")
    println(io, "gamma,$GAMMA")
    println(io, "rho0,$RHO0")
    println(io, "pressure_floor,$PRESSURE_FLOOR")
    println(io, "energy,$ENERGY")
    println(io, "bomb_radius,$BOMB_RADIUS")
    println(io, "shock_radius,$SHOCK_RADIUS")
    println(io, "shell_width,$SHELL_WIDTH")
    println(io, "displacement,$DISPLACEMENT")
    println(io, "lagrangian_fraction,$LAGRANGIAN_FRACTION")
    println(io, "area_mode,$AREA_MODE")
    for c in cases
        println(io, "$(c.label)_table,$(basename(c.path))")
    end
end

println("wrote IC tables to $OUTDIR")
