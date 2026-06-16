using Printf
using Statistics
using KernelAbstractions
using PowerFoam

const GAMMA = 5 / 3
const OUTBASE = joinpath(@__DIR__, "out", "powerfoam_kh2d_compare_gate")
const AREPO_REF_DEFAULT = joinpath(@__DIR__, "out", "arepo_kh2d_original_gate",
                                   "N32_t0p1_drat1", "runs", "arepo_hll",
                                   "analysis", "kh_metrics.csv")
const AREPO_FIELD_REF_DEFAULT = joinpath(@__DIR__, "out", "arepo_kh2d_original_gate",
                                         "N32_t0p1_drat1", "runs", "arepo_hll",
                                         "analysis", "final_fields.csv")

parse_arg(i, default, T) = length(ARGS) >= i ? parse(T, ARGS[i]) : default
parse_arg(i, default, ::Type{String}) = length(ARGS) >= i ? ARGS[i] : default

const NX = parse_arg(1, 32, Int)
const TFINAL = parse_arg(2, 0.1, Float64)
const DRAT = parse_arg(3, 1.0, Float64)
const CFL = parse_arg(4, 0.18, Float64)
const RIEMANN = Symbol(lowercase(parse_arg(5, "hll", String)))
const MOVING_N = parse_arg(6, min(NX, 16), Int)
const MOVING_TFINAL = parse_arg(7, min(TFINAL, 0.01), Float64)
const RUN_TAG = replace(@sprintf("N%d_t%.4g_drat%.3g_%s", NX, TFINAL, DRAT, RIEMANN), "." => "p")
const OUTDIR = joinpath(OUTBASE, RUN_TAG)
const AREPO_REF = get(ENV, "POWERFOAM_KH_AREPO_REF", AREPO_REF_DEFAULT)
const AREPO_FIELD_REF = get(ENV, "POWERFOAM_KH_AREPO_FIELDS", AREPO_FIELD_REF_DEFAULT)
const FIELD_RTOL = parse(Float64, get(ENV, "POWERFOAM_KH_FIELD_RTOL", "1e-10"))
const FIELD_ATOL = parse(Float64, get(ENV, "POWERFOAM_KH_FIELD_ATOL", "1e-12"))
const REQUIRE_FIELD_PARITY = lowercase(get(ENV, "POWERFOAM_REQUIRE_KH_FIELD_PARITY", "false")) in
                             ("1", "true", "yes", "on")

@inline cell_id(i, j, nx, ny) = i + nx * (j - 1)
@inline wrap1(i, n) = i > n ? 1 : i

function cell_face_csr(ncells, c1, c2, ::Type{I}) where {I<:Integer}
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

function cartesian_periodic_mesh_arrays_2d(nx, ny; T = Float64, index_type = Int32)
    nc = nx * ny
    nf = 2nc
    c1 = Vector{index_type}(undef, nf)
    c2 = Vector{index_type}(undef, nf)
    normal_x = zeros(T, nf)
    normal_y = zeros(T, nf)
    area = Vector{T}(undef, nf)
    dx = one(T) / T(nx)
    dy = T(2) / T(ny)
    f = 1
    for j in 1:ny, i in 1:nx
        id = cell_id(i, j, nx, ny)
        c1[f] = index_type(id)
        c2[f] = index_type(cell_id(wrap1(i + 1, nx), j, nx, ny))
        normal_x[f] = one(T)
        area[f] = dy
        f += 1
        c1[f] = index_type(id)
        c2[f] = index_type(cell_id(i, wrap1(j + 1, ny), nx, ny))
        normal_y[f] = one(T)
        area[f] = dx
        f += 1
    end
    offsets, faces, signs = cell_face_csr(nc, c1, c2, index_type)
    volume = fill(dx * dy, nc)
    return ArepoMeshArrays2D(c1, c2, offsets, faces, signs, volume, area,
                             normal_x, normal_y, zeros(T, nf), zeros(T, nf))
end

function grid_points_2d(nx, ny)
    pts = Matrix{Float64}(undef, nx * ny, 2)
    q = 1
    for j in 1:ny, i in 1:nx
        pts[q, 1] = (i - 0.5) / nx
        pts[q, 2] = (j - 0.5) * 2 / ny
        q += 1
    end
    return pts
end

function kh_arepo_ids(nx, ny)
    ids = Vector{Int}(undef, nx * ny)
    q = 1
    for j in 1:ny, i in 1:nx
        ids[q] = j + ny * (i - 1)
        q += 1
    end
    return ids
end

function kh_primitives(nx, ny; drat)
    nc = nx * ny
    rho = Vector{Float64}(undef, nc)
    vx = Vector{Float64}(undef, nc)
    vy = Vector{Float64}(undef, nc)
    pressure = fill(10.0, nc)
    dye = Vector{Float64}(undef, nc)
    a_shear = 0.05
    sigma = 0.2
    z1 = 0.5
    z2 = 1.5
    q = 1
    for j in 1:ny, i in 1:nx
        x = (i - 0.5) / nx
        y = (j - 0.5) * 2 / ny
        t1 = tanh((y - z1) / a_shear)
        t2 = tanh((y - z2) / a_shear)
        rho[q] = 1.0 + drat * 0.5 * (t1 - t2)
        vx[q] = t1 - t2 - 1.0
        vy[q] = 0.01 * sin(2pi * x) *
                (exp(-((y - z1)^2) / sigma^2) + exp(-((y - z2)^2) / sigma^2))
        dye[q] = 0.5 * (t2 - t1 + 2.0)
        q += 1
    end
    return (; rho, vx, vy, pressure, dye)
end

function initial_state(mesh, nx, ny; drat)
    p = kh_primitives(nx, ny; drat)
    D = copy(p.rho)
    Mx = p.rho .* p.vx
    My = p.rho .* p.vy
    E = p.pressure ./ (GAMMA - 1) .+ 0.5 .* p.rho .* (p.vx .* p.vx .+ p.vy .* p.vy)
    return EulerState2D(D, Mx, My, E)
end

function moving_voronoi_initial(nx, ny; drat)
    pts = grid_points_2d(nx, ny)
    built = periodic_power_mesh_arrays_2d(pts; domain = ((0.0, 1.0), (0.0, 2.0)),
                                          T = Float64,
                                          bins_per_axis = (nx, ny),
                                          search_radius = 1)
    p = kh_primitives(nx, ny; drat)
    D = copy(p.rho)
    Mx = p.rho .* p.vx
    My = p.rho .* p.vy
    E = p.pressure ./ (GAMMA - 1) .+
        0.5 .* p.rho .* (p.vx .* p.vx .+ p.vy .* p.vy)
    state = EulerState2D(D, Mx, My, E)
    return pts, built, state
end

function stable_dt(state, mesh; cfl)
    dx = sqrt(minimum(Array(mesh.volume)))
    return cfl * dx / max_signal_speed_2d(state; gamma = GAMMA)
end

function grid_arrays(state, nx, ny)
    prim = conserved_to_primitive_2d(state; gamma = GAMMA)
    rho = reshape(prim.rho, nx, ny)
    vx = reshape(prim.vx, nx, ny)
    vy = reshape(prim.vy, nx, ny)
    pressure = reshape(prim.pressure, nx, ny)
    return rho, vx, vy, pressure
end

function vorticity_z(vx, vy, dx, dy)
    nx, ny = size(vx)
    out = similar(vx)
    for j in 1:ny, i in 1:nx
        ip = wrap1(i + 1, nx)
        im = i == 1 ? nx : i - 1
        jp = wrap1(j + 1, ny)
        jm = j == 1 ? ny : j - 1
        out[i, j] = (vy[ip, j] - vy[im, j]) / (2dx) -
                    (vx[i, jp] - vx[i, jm]) / (2dy)
    end
    return out
end

function symmetry_error(a)
    nx, ny = size(a)
    s = 0.0
    nrm = 0.0
    shift = nx ÷ 2
    for j in 1:ny, i in 1:nx
        ir = wrap1(i + shift, nx)
        jr = ny - j + 1
        d = a[i, j] - a[ir, jr]
        s += d * d
        nrm += a[i, j] * a[i, j]
    end
    return sqrt(s / max(nrm, eps(Float64)))
end

function diagnostics(label, state, mesh, time, step, nx, ny)
    prim = conserved_to_primitive_2d(state; gamma = GAMMA)
    volume = Array(mesh.volume)
    rho, vx, vy, pressure = grid_arrays(state, nx, ny)
    vort = vorticity_z(vx, vy, 1 / nx, 2 / ny)
    mixed = sum(volume[(prim.rho .> 1.1) .& (prim.rho .< 1.9)]) / 2
    vertical_ke = sum(0.5 .* prim.rho .* prim.vy .* prim.vy .* volume) / 2
    pressure_wiggle = maximum(abs.(prim.pressure .- 10.0)) / 10.0
    return (; label, step, t = time,
            mixed_area = mixed,
            vertical_ke,
            enstrophy = mean(vort .* vort),
            pressure_wiggle,
            symmetry_error = symmetry_error(rho),
            rho_min = minimum(prim.rho),
            rho_max = maximum(prim.rho),
            p_min = minimum(prim.pressure),
            p_max = maximum(prim.pressure))
end

function run_moving_reconstructed(; nx, tfinal, cfl, riemann, drat)
    ny = 2nx
    domain = ((0.0, 1.0), (0.0, 2.0))
    points, built, state = moving_voronoi_initial(nx, ny; drat)
    ids = kh_arepo_ids(nx, ny)
    geom = built.geom
    rows = [diagnostics("powerfoam_moving_reconstructed_periodic", state, geom,
                        0.0, 0, nx, ny)]
    t = 0.0
    step = 0
    rejected_steps = 0
    while t < tfinal - 1e-14
        dt = min(stable_dt(state, geom; cfl), tfinal - t)
        accepted = false
        trial_dt = dt
        saved = (D = copy(state.D), Mx = copy(state.Mx), My = copy(state.My),
                 E = copy(state.E), points = copy(points), geom = geom)
        trial_new_points = points
        trial_new = built
        for _ in 1:10
            state.D .= saved.D
            state.Mx .= saved.Mx
            state.My .= saved.My
            state.E .= saved.E
            points = copy(saved.points)
            geom = saved.geom
            prim = conserved_to_primitive_2d(state; gamma = GAMMA)
            vmesh = hcat(prim.vx, prim.vy)
            old = periodic_power_mesh_arrays_2d(points; domain, T = Float64,
                                                cell_velocity = vmesh,
                                                bins_per_axis = (nx, ny),
                                                search_radius = 1)
            trial_new_points = advect_generators_2d(points, vmesh, trial_dt,
                                                    domain; boundary = :periodic)
            trial_new = periodic_power_mesh_arrays_2d(trial_new_points; domain,
                                                      T = Float64,
                                                      bins_per_axis = (nx, ny),
                                                      search_radius = 1)

            primwork = primitive_work_2d(state)
            conserved_to_primitive_2d!(primwork, state; gamma = GAMMA)
            gradients = hydro_gradient_work_2d(primwork.rho)
            calculate_gradients_from_mesh_2d!(gradients, old.geom, primwork,
                                              old.center, old.face_center;
                                              box_size = 1.0, box_size_y = 2.0,
                                              gamma = GAMMA)
            work = hydro_work_2d(state, old.geom)
            states = face_prediction_work_2d(old.geom)
            finite_volume_reconstructed_step_2d!(state, old.geom, gradients,
                                                 primwork, old.center,
                                                 old.face_center;
                                                 dt = trial_dt, gamma = GAMMA,
                                                 riemann, work, states,
                                                 new_volume = trial_new.geom.volume,
                                                 box_size = 1.0,
                                                 box_size_y = 2.0)
            trial_diag = diagnostics("powerfoam_moving_reconstructed_periodic",
                                     state, trial_new.geom, t + trial_dt,
                                     step + 1, nx, ny)
            if trial_diag.rho_min > 0 && trial_diag.p_min > 0 &&
               isfinite(trial_diag.enstrophy)
                accepted = true
                break
            end
            rejected_steps += 1
            trial_dt *= 0.5
        end
        accepted || error("PowerFoam moving KH produced non-positive state after timestep retries")
        points = trial_new_points
        geom = trial_new.geom
        t += dt
        if trial_dt != dt
            t -= dt - trial_dt
        end
        step += 1
        push!(rows, diagnostics("powerfoam_moving_reconstructed_periodic", state,
                                geom, t, step, nx, ny))
        rows[end].rho_min > 0 && rows[end].p_min > 0 ||
            error("PowerFoam moving KH produced non-positive state")
    end
    return (; rows, rejected_steps,
            state = EulerState2D(copy(state.D), copy(state.Mx),
                                 copy(state.My), copy(state.E)),
            geom,
            points = copy(points),
            ids)
end

function read_arepo_final(path)
    isfile(path) || return nothing
    lines = readlines(path)
    length(lines) >= 2 || return nothing
    header = split(lines[1], ',')
    vals = split(lines[end], ',')
    d = Dict(Symbol(k) => parse(Float64, v) for (k, v) in zip(header, vals))
    return (; (k => d[k] for k in keys(d))...)
end

function final_field_rows(label, state, mesh, nx, ny, time; points = nothing,
                          ids = nothing)
    prim = conserved_to_primitive_2d(state; gamma = GAMMA)
    volume = Array(mesh.volume)
    rows = NamedTuple[]
    for id in eachindex(volume)
        if points === nothing
            i = ((id - 1) % nx) + 1
            j = ((id - 1) ÷ nx) + 1
            x = (i - 0.5) / nx
            y = (j - 0.5) * 2 / ny
        else
            x = points[id, 1]
            y = points[id, 2]
        end
        rho = prim.rho[id]
        vx = prim.vx[id]
        vy = prim.vy[id]
        pressure = prim.pressure[id]
        mass = state.D[id] * volume[id]
        mx = state.Mx[id] * volume[id]
        my = state.My[id] * volume[id]
        energy_density = state.E[id]
        energy = energy_density * volume[id]
        out_id = ids === nothing ? id : ids[id]
        push!(rows, (; label, id = out_id, t = time, x, y, volume = volume[id],
                     rho, vx, vy, pressure, mass, mx, my,
                     energy_density, energy))
    end
    return rows
end

function write_field_csv(path, rowsets)
    open(path, "w") do io
        println(io, "label,id,t,x,y,volume,rho,vx,vy,pressure,mass,mx,my,energy_density,energy")
        for rows in rowsets, r in rows
            @printf(io, "%s,%d,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g\n",
                    r.label, r.id, r.t, r.x, r.y, r.volume, r.rho, r.vx,
                    r.vy, r.pressure, r.mass, r.mx, r.my, r.energy_density,
                    r.energy)
        end
    end
end

function read_field_csv(path)
    isfile(path) || return NamedTuple[]
    lines = readlines(path)
    length(lines) >= 2 || return NamedTuple[]
    header = Symbol.(split(lines[1], ','))
    rows = NamedTuple[]
    for line in lines[2:end]
        isempty(strip(line)) && continue
        vals = split(line, ',')
        parsed = Dict{Symbol,Any}()
        for (k, v) in zip(header, vals)
            parsed[k] = k === :label ? v :
                        k === :id ? parse(Int, v) :
                        parse(Float64, v)
        end
        push!(rows, (; (k => parsed[k] for k in header)...))
    end
    return rows
end

function _field_stats(candidate, reference, field)
    ref_by_id = Dict(r.id => r for r in reference)
    diffs = Float64[]
    refs = Float64[]
    for c in candidate
        r = get(ref_by_id, c.id, nothing)
        r === nothing && continue
        cv = getproperty(c, field)
        rv = getproperty(r, field)
        push!(diffs, abs(cv - rv))
        push!(refs, abs(rv))
    end
    isempty(diffs) && return (; cells = 0, l1 = NaN, linf = NaN,
                              rel_l1 = NaN, rel_linf = NaN)
    l1 = mean(diffs)
    linf = maximum(diffs)
    denom = max(maximum(refs), eps(Float64))
    return (; cells = length(diffs), l1, linf,
            rel_l1 = l1 / denom, rel_linf = linf / denom)
end

function compare_field_rows(candidate_label, candidate, reference;
                            reference_label = "arepo_hll")
    fields = (:volume, :rho, :vx, :vy, :pressure, :mass, :mx, :my,
              :energy_density, :energy)
    rows = NamedTuple[]
    for field in fields
        s = _field_stats(candidate, reference, field)
        push!(rows, (; reference = reference_label,
                     candidate = candidate_label,
                     field = String(field),
                     cells = s.cells,
                     l1 = s.l1,
                     linf = s.linf,
                     rel_l1 = s.rel_l1,
                     rel_linf = s.rel_linf))
    end
    return rows
end

function field_compare_ok(row)
    row.cells > 0 && isfinite(row.linf) &&
        (row.linf <= FIELD_ATOL || row.rel_linf <= FIELD_RTOL)
end

csvquote(v) = "\"" * replace(string(v), "\"" => "\"\"") * "\""

function write_field_compare_csv(path, rows)
    open(path, "w") do io
        println(io, "reference,candidate,field,cells,l1,linf,rel_l1,rel_linf,atol,rtol,status")
        for r in rows
            vals = (r.reference, r.candidate, r.field, r.cells,
                    r.l1, r.linf, r.rel_l1, r.rel_linf, FIELD_ATOL,
                    FIELD_RTOL, field_compare_ok(r) ? "passed" : "failed")
            println(io, join((csvquote(v) for v in vals), ","))
        end
    end
end

function write_csv(path, rows)
    open(path, "w") do io
        println(io, "label,step,t,mixed_area,vertical_ke,enstrophy,pressure_wiggle,symmetry_error,rho_min,rho_max,p_min,p_max")
        for r in rows
            @printf(io, "%s,%d,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g\n",
                    r.label, r.step, r.t, r.mixed_area, r.vertical_ke,
                    r.enstrophy, r.pressure_wiggle, r.symmetry_error,
                    r.rho_min, r.rho_max, r.p_min, r.p_max)
        end
    end
end

function metric_diff(pf, arepo, key)
    arepo === nothing && return NaN
    return getproperty(pf, key) - getproperty(arepo, key)
end

function write_report(path, rows, moving_rows, arepo_final, metrics_csv,
                      field_csv, field_compare_csv, field_compare_rows,
                      moving_rejected_steps = 0)
    final = rows[end]
    moving_final = isempty(moving_rows) ? nothing : moving_rows[end]
    moving_ok = moving_final === nothing ||
                (moving_final.rho_min > 0 && moving_final.p_min > 0 &&
                 isfinite(moving_final.enstrophy))
    field_ok = !isempty(field_compare_rows) && all(field_compare_ok, field_compare_rows)
    status = final.rho_min > 0 && final.p_min > 0 && isfinite(final.enstrophy) &&
             moving_ok && (!REQUIRE_FIELD_PARITY || field_ok) ? "passed" : "failed"
    open(path, "w") do io
        println(io, "# PowerFoam 2-D KH Comparison Gate")
        println(io)
        println(io, "This gate runs the Lecoanet-style 2-D Kelvin-Helmholtz IC with")
        println(io, "PowerFoam's 2-D face-table hydro on a periodic Cartesian mesh and")
        println(io, "compares the same scalar diagnostics to the original AREPO HLL")
        println(io, "reference generated by `arepo_kh2d_original_gate.jl`.")
        println(io)
        @printf(io, "- status: %s\n", status)
        @printf(io, "- PowerFoam mesh: fixed periodic Cartesian\n")
        @printf(io, "- nx x ny: %d x %d\n", NX, 2 * NX)
        @printf(io, "- t_final: %.12g\n", TFINAL)
        @printf(io, "- moving reconstructed mesh: periodic Voronoi, %d x %d, t_final %.12g\n",
                MOVING_N, 2 * MOVING_N, MOVING_TFINAL)
        @printf(io, "- moving reconstructed rejected trial steps: %d\n",
                moving_rejected_steps)
        @printf(io, "- CFL: %.12g\n", CFL)
        @printf(io, "- solver: `%s`\n", RIEMANN)
        @printf(io, "- metrics: `%s`\n", relpath(metrics_csv, dirname(path)))
        @printf(io, "- final fields: `%s`\n", relpath(field_csv, dirname(path)))
        @printf(io, "- field comparison: `%s`\n",
                relpath(field_compare_csv, dirname(path)))
        @printf(io, "- AREPO reference: `%s`\n", AREPO_REF)
        @printf(io, "- AREPO field reference: `%s`\n", AREPO_FIELD_REF)
        @printf(io, "- field tolerance: abs %.12g or rel %.12g\n",
                FIELD_ATOL, FIELD_RTOL)
        @printf(io, "- require field parity: `%s`\n", REQUIRE_FIELD_PARITY)
        @printf(io, "- field parity status: %s\n",
                field_ok ? "passed" : "not available or failed")
        println(io)
        println(io, "## Final Metrics")
        println(io)
        println(io, "| run | t | mixed area | vertical KE | enstrophy | pressure wiggle | symmetry error | rho min | rho max | p min | p max |")
        println(io, "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        @printf(io, "| powerfoam_fixed_periodic | %.12g | %.12g | %.12g | %.12g | %.12g | %.12g | %.12g | %.12g | %.12g | %.12g |\n",
                final.t, final.mixed_area, final.vertical_ke, final.enstrophy,
                final.pressure_wiggle, final.symmetry_error,
                final.rho_min, final.rho_max, final.p_min, final.p_max)
        if moving_final !== nothing
            @printf(io, "| powerfoam_moving_reconstructed_periodic | %.12g | %.12g | %.12g | %.12g | %.12g | %.12g | %.12g | %.12g | %.12g | %.12g |\n",
                    moving_final.t, moving_final.mixed_area, moving_final.vertical_ke,
                    moving_final.enstrophy, moving_final.pressure_wiggle,
                    moving_final.symmetry_error, moving_final.rho_min,
                    moving_final.rho_max, moving_final.p_min, moving_final.p_max)
        end
        if arepo_final !== nothing
            @printf(io, "| arepo_hll_reference | %.12g | %.12g | %.12g | %.12g | %.12g | %.12g | %.12g | %.12g | %.12g | %.12g |\n",
                    arepo_final.t, arepo_final.mixed_area, arepo_final.vertical_ke,
                    arepo_final.enstrophy, arepo_final.pressure_wiggle,
                    arepo_final.symmetry_error, arepo_final.rho_min,
                    arepo_final.rho_max, arepo_final.p_min, arepo_final.p_max)
        end
        println(io)
        println(io, "## Difference From Original AREPO HLL")
        println(io)
        println(io, "The fixed-periodic row is the direct scalar diagnostic comparison.")
        if MOVING_N == NX && MOVING_TFINAL == TFINAL
            println(io, "The moving reconstructed row uses the same resolution and final")
            println(io, "time, with adaptive rejected-trial retries recorded above.")
        else
            println(io, "The moving reconstructed row is a smaller/shorter periodic-Voronoi")
            println(io, "smoke rung, so its differences are reported for scale and")
            println(io, "positivity rather than final parity.")
        end
        println(io)
        println(io, "| metric | fixed periodic - AREPO | moving reconstructed - AREPO |")
        println(io, "| --- | ---: | ---: |")
        for key in (:mixed_area, :vertical_ke, :enstrophy, :pressure_wiggle, :symmetry_error,
                    :rho_min, :rho_max, :p_min, :p_max)
            moving_diff = moving_final === nothing ? NaN :
                          metric_diff(moving_final, arepo_final, key)
            @printf(io, "| %s | %.12g | %.12g |\n", String(key),
                    metric_diff(final, arepo_final, key), moving_diff)
        end
        println(io)
        println(io, "## Final-Field Difference From Original AREPO HLL")
        println(io)
        if isempty(field_compare_rows)
            println(io, "No compatible AREPO final-field CSV was available. Re-run")
            println(io, "`arepo_kh2d_original_gate.jl` to generate")
            println(io, "`analysis/final_fields.csv`, or set")
            println(io, "`POWERFOAM_KH_AREPO_FIELDS` to a reference with the same")
            println(io, "cell count as the PowerFoam row being compared.")
        else
            println(io, "Rows compare normalized cell-id fields against the original")
            println(io, "AREPO HLL `final_fields.csv` artifact.")
            println(io)
            println(io, "| reference | candidate | field | cells | L1 | Linf | relative L1 | relative Linf | status |")
            println(io, "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |")
            for r in field_compare_rows
                @printf(io, "| %s | %s | %s | %d | %.12g | %.12g | %.12g | %.12g | %s |\n",
                        r.reference, r.candidate, r.field, r.cells, r.l1,
                        r.linf, r.rel_l1, r.rel_linf,
                        field_compare_ok(r) ? "passed" : "failed")
            end
        end
        println(io)
        println(io, "## Scope")
        println(io)
        println(io, "This gate now has two PowerFoam KH rungs. The fixed periodic")
        println(io, "Cartesian row validates the conservative periodic face-table path")
        println(io, "against the original-code KH diagnostics. The moving reconstructed")
        println(io, "row exercises PowerFoam's 2-D periodic Voronoi ALE reconstruction")
        println(io, "and rebuild path with true cross-boundary faces. The next parity")
        if MOVING_N == NX && MOVING_TFINAL == TFINAL
            println(io, "upgrade is to promote the final-field table above from diagnostic")
            println(io, "evidence into explicit tolerance gates for solver choices.")
        else
            println(io, "upgrade is to raise this moving row to the direct `N32`, `t=0.1`")
            println(io, "comparison once the small periodic smoke rung is stable.")
        end
    end
    return status
end

function main()
    mkpath(OUTDIR)
    nx = NX
    ny = 2NX
    mesh = cartesian_periodic_mesh_arrays_2d(nx, ny; T = Float64)
    state = initial_state(mesh, nx, ny; drat = DRAT)
    rows = [diagnostics("powerfoam_fixed_periodic", state, mesh, 0.0, 0, nx, ny)]
    t = 0.0
    step = 0
    while t < TFINAL - 1e-14
        dt = min(stable_dt(state, mesh; cfl = CFL), TFINAL - t)
        finite_volume_step_2d!(state, mesh; dt, gamma = GAMMA, riemann = RIEMANN)
        t += dt
        step += 1
        push!(rows, diagnostics("powerfoam_fixed_periodic", state, mesh, t, step, nx, ny))
        rows[end].rho_min > 0 && rows[end].p_min > 0 ||
            error("PowerFoam KH produced non-positive state")
    end
    metrics_csv = joinpath(OUTDIR, "powerfoam_kh_metrics.csv")
    moving = run_moving_reconstructed(; nx = MOVING_N,
                                      tfinal = MOVING_TFINAL,
                                      cfl = CFL,
                                      riemann = RIEMANN,
                                      drat = DRAT)
    moving_rows = moving.rows
    write_csv(metrics_csv, vcat(rows, moving_rows))
    field_csv = joinpath(OUTDIR, "powerfoam_final_fields.csv")
    fixed_fields = final_field_rows("powerfoam_fixed_periodic", state, mesh,
                                    nx, ny, t; ids = kh_arepo_ids(nx, ny))
    moving_fields = final_field_rows("powerfoam_moving_reconstructed_periodic",
                                     moving.state, moving.geom, MOVING_N,
                                     2 * MOVING_N, moving_rows[end].t;
                                     points = moving.points, ids = moving.ids)
    write_field_csv(field_csv, (fixed_fields, moving_fields))
    arepo_final = read_arepo_final(AREPO_REF)
    arepo_fields = read_field_csv(AREPO_FIELD_REF)
    field_compare_rows = NamedTuple[]
    if !isempty(arepo_fields)
        if length(fixed_fields) == length(arepo_fields)
            append!(field_compare_rows,
                    compare_field_rows("powerfoam_fixed_periodic",
                                       fixed_fields, arepo_fields))
        end
        if length(moving_fields) == length(arepo_fields)
            append!(field_compare_rows,
                    compare_field_rows("powerfoam_moving_reconstructed_periodic",
                                       moving_fields, arepo_fields))
        end
    end
    field_compare_csv = joinpath(OUTDIR, "field_compare.csv")
    write_field_compare_csv(field_compare_csv, field_compare_rows)
    report = joinpath(OUTDIR, "README.md")
    status = write_report(report, rows, moving_rows, arepo_final, metrics_csv,
                          field_csv, field_compare_csv, field_compare_rows,
                          moving.rejected_steps)
    @printf("wrote %s\n", report)
    @printf("wrote %s\n", metrics_csv)
    @printf("wrote %s\n", field_csv)
    @printf("wrote %s\n", field_compare_csv)
    final = rows[end]
    moving_final = moving_rows[end]
    @printf("powerfoam KH %s: fixed_steps=%d fixed_t=%.6g moving_steps=%d moving_t=%.6g fixed_mixed=%.6g moving_mixed=%.6g rho=[%.6g, %.6g]\n",
            status, final.step, final.t, moving_final.step, moving_final.t,
            final.mixed_area, moving_final.mixed_area, final.rho_min, final.rho_max)
    status == "passed" || exit(1)
end

main()
