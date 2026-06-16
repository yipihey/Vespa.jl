const AREPOLIB_DIR = get(ENV, "AREPO_LIB_JL", "/Users/tabel/Projects/Arepo.jl/lib/ArepoLib")
if isdir(AREPOLIB_DIR) && !(AREPOLIB_DIR in LOAD_PATH)
    push!(LOAD_PATH, AREPOLIB_DIR)
end

using Printf
using Statistics
using LinearAlgebra
using PowerFoam
using ArepoLib

const GAMMA = 5 / 3
const AREPO_DIR = get(ENV, "AREPO_DIR", "/Users/tabel/Projects/arepo")
const EXAMPLE = joinpath(AREPO_DIR, "examples", "noh_3d")
const OUTBASE = joinpath(@__DIR__, "out", "arepo_noh3d_smoke_gate")

parse_arg(i, default, T) = length(ARGS) >= i ? parse(T, ARGS[i]) : default
parse_arg(i, default, ::Type{String}) = length(ARGS) >= i ? ARGS[i] : default

const N_STEPS = parse_arg(1, 1, Int)
const RIEMANN = Symbol(lowercase(parse_arg(2, "hll", String)))
const NBINS = parse_arg(3, 48, Int)
const OUTDIR = joinpath(OUTBASE, @sprintf("stock_n30_n%d_%s", N_STEPS, RIEMANN))
const FIELD_RTOL = parse(Float64, get(ENV, "POWERFOAM_NOH_FIELD_RTOL", "1e-10"))
const FIELD_ATOL = parse(Float64, get(ENV, "POWERFOAM_NOH_FIELD_ATOL", "1e-12"))
const REQUIRE_FIELD_PARITY = lowercase(get(ENV, "POWERFOAM_REQUIRE_NOH_FIELD_PARITY", "false")) in
                             ("1", "true", "yes", "on")

struct NohSnapshot
    label::String
    time::Float64
    ng::Int
    faces::Int
    box::Float64
    geo::Any
    mass::Vector{Float64}
    volume::Vector{Float64}
    rho::Vector{Float64}
    pressure::Vector{Float64}
    energy::Vector{Float64}
    momentum::Matrix{Float64}
    center::Matrix{Float64}
    pos::Matrix{Float64}
    vel::Matrix{Float64}
    velvertex::Matrix{Float64}
end

function python_cmd()
    for exe in (get(ENV, "AREPO_PYTHON", ""), joinpath(AREPO_DIR, ".venv", "bin", "python"),
                "python3", "python")
        isempty(exe) && continue
        try
            run(pipeline(Cmd([exe, "-c", "import h5py, numpy"]);
                         stdout = devnull, stderr = devnull))
            return exe
        catch
        end
    end
    error("no Python with h5py/numpy found; set AREPO_PYTHON")
end

function set_param(text, key, value)
    line = @sprintf("%-38s %s", key, value)
    pattern = Regex("(?m)^" * key * "\\s+.*\$")
    return occursin(pattern, text) ? replace(text, pattern => line) : text * "\n" * line * "\n"
end

function normalize_param_for_linked_arepo(text; riemann::Symbol)
    lines = split(text, '\n'; keepempty = true)
    keep = String[]
    for line in lines
        if occursin(r"^SofteningComovingType[2-5]\s", line) ||
           occursin(r"^SofteningMaxPhysType[2-5]\s", line)
            continue
        end
        push!(keep, line)
    end
    text = join(keep, "\n")
    text = replace(text,
                   r"(?m)^SofteningTypeOfPartType1\s+.*$" => "SofteningTypeOfPartType1              1",
                   r"(?m)^SofteningTypeOfPartType2\s+.*$" => "SofteningTypeOfPartType2              1",
                   r"(?m)^SofteningTypeOfPartType3\s+.*$" => "SofteningTypeOfPartType3              1",
                   r"(?m)^SofteningTypeOfPartType4\s+.*$" => "SofteningTypeOfPartType4              1",
                   r"(?m)^SofteningTypeOfPartType5\s+.*$" => "SofteningTypeOfPartType5              1")
    text = set_param(text, "TimeOfFirstSnapshot", "2.0")
    text = set_param(text, "TimeBetSnapshot", "2.0")
    text = set_param(text, "HydroRiemannSolver", uppercase(String(riemann)))
    text = set_param(text, "MinimumComovingHydroSoftening", "0.001")
    text = set_param(text, "AdaptiveHydroSofteningSpacing", "1.2")
    return text
end

function stage_arepo_case(; riemann::Symbol)
    isdir(EXAMPLE) || error("AREPO Noh example not found at $EXAMPLE")
    dir = mktempdir()
    cp(joinpath(EXAMPLE, "param.txt"), joinpath(dir, "param.txt"))
    param = joinpath(dir, "param.txt")
    write(param, normalize_param_for_linked_arepo(read(param, String); riemann))
    mkpath(joinpath(dir, "output"))
    py = python_cmd()
    run(pipeline(`$py $(joinpath(EXAMPLE, "create.py")) $dir`; stdout = devnull))
    isfile(joinpath(dir, "IC.hdf5")) || error("AREPO create.py produced no IC.hdf5")
    return dir
end

function cell_centers_or_positions(center, pos)
    if all(iszero, center)
        return pos
    end
    return center
end

function collect_snapshot(h, label)
    info = ArepoLib.info(h)
    ng = info.numgas
    geo = ArepoLib.get_voronoi_3d(h)
    mass = ArepoLib.get_particle_field(h, :mass)[1:ng]
    pos = ArepoLib.get_particle_field(h, :pos)[1:ng, :]
    volume = ArepoLib.get_cell_field(h, :volume)
    rho = ArepoLib.get_cell_field(h, :rho)
    pressure = ArepoLib.get_cell_field(h, :pressure)
    energy = ArepoLib.get_cell_field(h, :energy)
    momentum = ArepoLib.get_cell_field(h, :momentum)
    velvertex = ArepoLib.get_cell_field(h, :velvertex)
    vel = similar(momentum)
    @inbounds for k in 1:3, i in eachindex(mass)
        vel[i, k] = momentum[i, k] / mass[i]
    end
    center = ArepoLib.get_cell_field(h, :center)
    return NohSnapshot(label, info.time, ng, length(geo.nv), info.boxsize, geo,
                       mass, volume, rho, pressure, energy, momentum,
                       cell_centers_or_positions(center, pos), pos, vel,
                       velvertex)
end

function primitive_diagnostics(s::NohSnapshot)
    v2 = s.vel[:, 1] .^ 2 .+ s.vel[:, 2] .^ 2 .+ s.vel[:, 3] .^ 2
    cs2 = GAMMA .* s.pressure ./ s.rho
    return (; label = s.label,
            time = s.time,
            cells = s.ng,
            faces = s.faces,
            volume = sum(s.volume),
            mass = sum(s.mass),
            rho_min = minimum(s.rho),
            rho_max = maximum(s.rho),
            pressure_min = minimum(s.pressure),
            pressure_max = maximum(s.pressure),
            energy = sum(s.energy),
            vrms = sqrt(mean(v2)),
            mach_rms = sqrt(mean(v2 ./ cs2)))
end

function powerfoam_roundtrip_diagnostics(s::NohSnapshot)
    geom = arepo_voronoi_mesh_arrays_3d(s.geo, s.volume; T = Float64)
    state = euler_state_3d(geom; rho = s.rho,
                           vx = s.vel[:, 1], vy = s.vel[:, 2], vz = s.vel[:, 3],
                           pressure = s.pressure, gamma = GAMMA, T = Float64)
    totals = total_conserved_3d(state, geom)
    return (; label = s.label,
            mass_gap = abs(totals.mass - sum(s.mass)),
            mx_gap = abs(totals.mx - sum(@view s.momentum[:, 1])),
            my_gap = abs(totals.my - sum(@view s.momentum[:, 2])),
            mz_gap = abs(totals.mz - sum(@view s.momentum[:, 3])),
            energy_gap = abs(totals.energy - sum(s.energy)))
end

function powerfoam_fixed_geometry_step(s::NohSnapshot, dt; riemann::Symbol)
    geom = arepo_voronoi_mesh_arrays_3d(s.geo, s.volume; T = Float64)
    state = euler_state_3d(geom; rho = s.rho,
                           vx = s.vel[:, 1], vy = s.vel[:, 2], vz = s.vel[:, 3],
                           pressure = s.pressure, gamma = GAMMA, T = Float64)
    finite_volume_step_3d!(state, geom; dt, gamma = GAMMA, riemann)
    prim = conserved_to_primitive_3d(state; gamma = GAMMA)
    volume = Array(geom.volume)
    rho = Array(state.D)
    mx = Array(state.Mx)
    my = Array(state.My)
    mz = Array(state.Mz)
    energy_density = Array(state.E)
    momentum = hcat(mx .* volume, my .* volume, mz .* volume)
    mass = rho .* volume
    energy = energy_density .* volume
    vel = hcat(Array(prim.vx), Array(prim.vy), Array(prim.vz))
    return NohSnapshot("powerfoam_fixed_step1", s.time + dt, s.ng, s.faces, s.box, s.geo,
                       mass, volume, Array(prim.rho), Array(prim.pressure), energy,
                       momentum, s.center, s.pos, vel, copy(s.velvertex))
end

function wrap_points3(points, box)
    pts = Matrix{Float64}(points)
    @inbounds for i in axes(pts, 1), k in 1:3
        pts[i, k] = mod(pts[i, k], box)
    end
    return pts
end

function stock_lattice_points3(ng, box)
    n = round(Int, cbrt(ng))
    n^3 == ng || return nothing
    pts = Matrix{Float64}(undef, ng, 3)
    q = 1
    dx = box / n
    @inbounds for k in 1:n, j in 1:n, i in 1:n
        pts[q, 1] = (i - 0.5) * dx
        pts[q, 2] = (j - 0.5) * dx
        pts[q, 3] = (k - 0.5) * dx
        q += 1
    end
    return pts
end

function noh_radial_velocity(points, box)
    v = similar(points)
    center = box / 2
    @inbounds for i in axes(points, 1)
        dx = points[i, 1] - center
        dy = points[i, 2] - center
        dz = points[i, 3] - center
        dx -= box * round(dx / box)
        dy -= box * round(dy / box)
        dz -= box * round(dz / box)
        r = sqrt(dx * dx + dy * dy + dz * dz)
        if r > 0
            v[i, 1] = -dx / r
            v[i, 2] = -dy / r
            v[i, 3] = -dz / r
        else
            v[i, 1] = 0
            v[i, 2] = 0
            v[i, 3] = 0
        end
    end
    return v
end

function powerfoam_moving_mesh_step(s::NohSnapshot, dt; riemann::Symbol)
    domain = ((0.0, s.box), (0.0, s.box), (0.0, s.box))
    points = something(stock_lattice_points3(s.ng, s.box), wrap_points3(s.pos, s.box))
    vmesh = noh_radial_velocity(points, s.box)
    old = local_periodic_voronoi_mesh_arrays_3d(points; domain, T = Float64,
                                                cell_velocity = vmesh,
                                                bins_per_axis = round(Int, cbrt(s.ng)),
                                                search_radius = 1)
    state = euler_state_3d(old.geom; rho = s.rho,
                           vx = vmesh[:, 1], vy = vmesh[:, 2], vz = vmesh[:, 3],
                           pressure = s.pressure, gamma = GAMMA, T = Float64)
    moved = moving_mesh_step_3d!(state, points; dt, gamma = GAMMA,
                                 mesh_velocity = vmesh, domain,
                                 boundary = :periodic, rebuild = :local,
                                 local_bins_per_axis = old.bins_per_axis,
                                 local_search_radius = 1, riemann, T = Float64)
    prim = conserved_to_primitive_3d(state; gamma = GAMMA)
    volume = Array(moved.geom.volume)
    rho = Array(state.D)
    mx = Array(state.Mx)
    my = Array(state.My)
    mz = Array(state.Mz)
    energy_density = Array(state.E)
    momentum = hcat(mx .* volume, my .* volume, mz .* volume)
    mass = rho .* volume
    energy = energy_density .* volume
    vel = hcat(Array(prim.vx), Array(prim.vy), Array(prim.vz))
    return NohSnapshot("powerfoam_moving_step1", s.time + dt, s.ng,
                       length(Array(moved.geom.c1)), s.box, s.geo,
                       mass, volume, Array(prim.rho), Array(prim.pressure),
                       energy, momentum, moved.center, moved.points, vel, vmesh)
end

function _total_from_volume(state::EulerState3D, volume)
    D = Array(state.D)
    Mx = Array(state.Mx)
    My = Array(state.My)
    Mz = Array(state.Mz)
    E = Array(state.E)
    v = Array(volume)
    return (; mass = sum(D .* v),
            mx = sum(Mx .* v),
            my = sum(My .* v),
            mz = sum(Mz .* v),
            energy = sum(E .* v))
end

function _radial_hierarchy_bins(points, box)
    center = box / 2
    radii = Vector{Float64}(undef, size(points, 1))
    @inbounds for i in eachindex(radii)
        dx = points[i, 1] - center
        dy = points[i, 2] - center
        dz = points[i, 3] - center
        dx -= box * round(dx / box)
        dy -= box * round(dy / box)
        dz -= box * round(dz / box)
        radii[i] = sqrt(dx * dx + dy * dy + dz * dz)
    end
    split_radius = median(radii)
    bins = fill(1, length(radii))
    bins[radii .<= split_radius] .= 0
    return bins
end

function _active_list_mismatch(active, active_list)
    listed = falses(length(active))
    for i in active_list
        1 <= i <= length(active) && (listed[i] = true)
    end
    return count(listed .!= active)
end

function powerfoam_moving_hierarchy_probe(s::NohSnapshot, dt; riemann::Symbol,
                                          arepo_bins = nothing,
                                          use_snapshot_points::Bool = false)
    domain = ((0.0, s.box), (0.0, s.box), (0.0, s.box))
    traced = arepo_bins !== nothing
    points = traced || use_snapshot_points ?
             wrap_points3(s.pos, s.box) :
             something(stock_lattice_points3(s.ng, s.box), wrap_points3(s.pos, s.box))
    vmesh = traced ? Matrix{Float64}(s.velvertex) : noh_radial_velocity(points, s.box)
    nb = round(Int, cbrt(s.ng))
    bins = traced ? Int.(arepo_bins.bins) : _radial_hierarchy_bins(points, s.box)
    ti_current = traced ? Int(arepo_bins.ti_current) : 1
    timebase_interval = traced ? Float64(arepo_bins.timebase_interval) : dt
    ti_step = arepo_next_sync_step_3d(bins, ti_current)
    step_dt = timebase_interval * ti_step
    old = local_periodic_voronoi_mesh_arrays_3d(points; domain, T = Float64,
                                                cell_velocity = vmesh,
                                                bins_per_axis = nb,
                                                search_radius = 1)
    new_points = advect_generators_3d(points, vmesh, step_dt, domain;
                                      boundary = :periodic)
    new = local_periodic_voronoi_mesh_arrays_3d(new_points; domain, T = Float64,
                                                bins_per_axis = old.bins_per_axis,
                                                search_radius = old.search_radius)
    state = euler_state_3d(old.geom; rho = s.rho,
                           vx = s.vel[:, 1], vy = s.vel[:, 2], vz = s.vel[:, 3],
                           pressure = s.pressure, gamma = GAMMA, T = Float64)
    prim = primitive_work_3d(state)
    conserved_to_primitive_3d!(prim, state; gamma = GAMMA)
    gradients = hydro_gradient_work_3d(prim.rho)
    calculate_gradients_from_mesh_3d!(gradients, old.geom, prim,
                                      old.center, old.face_center;
                                      box_size = s.box, gamma = GAMMA)

    before = _total_from_volume(state, old.geom.volume)
    result = finite_volume_reconstructed_hierarchy_step_3d!(
        state, old.geom, gradients, prim,
        collect(view(old.center, :, 1)), collect(view(old.center, :, 2)),
        collect(view(old.center, :, 3)),
        collect(view(old.face_center, :, 1)), collect(view(old.face_center, :, 2)),
        collect(view(old.face_center, :, 3)), bins;
        ti_current, timebase_interval,
        gamma = GAMMA, riemann, new_volume = new.geom.volume,
        box_size = s.box)
    active = result.active
    active_mismatches = traced ?
                        count(active .!= Bool.(arepo_bins.active)) :
                        0
    active_list_mismatches = traced ?
                             _active_list_mismatch(active, Int.(arepo_bins.active_list)) :
                             0
    volume_after = Array(old.geom.volume)
    volume_after[active] .= Array(new.geom.volume)[active]
    after = _total_from_volume(state, volume_after)
    prim_after = conserved_to_primitive_3d(state; gamma = GAMMA)
    return (; label = traced ? "powerfoam_moving_hierarchy_trace_probe" :
                              "powerfoam_moving_hierarchy_probe",
            source = traced ? "arepo_timebins" : "radial_split",
            cells = s.ng,
            faces = length(Array(old.geom.c1)),
            active_count = count(active),
            active_fraction = count(active) / length(active),
            active_mismatches,
            active_list_mismatches,
            ti_current,
            ti_next = result.ti_next,
            ti_step = result.ti_step,
            dt = result.dt,
            mass_gap = abs(after.mass - before.mass),
            mx_gap = abs(after.mx - before.mx),
            my_gap = abs(after.my - before.my),
            mz_gap = abs(after.mz - before.mz),
            energy_gap = abs(after.energy - before.energy),
            rho_min = minimum(Array(prim_after.rho)),
            pressure_min = minimum(Array(prim_after.pressure)))
end

function maybe_hydro_timebins(h)
    isdefined(ArepoLib, :get_hydro_timebins) || return nothing
    try
        return ArepoLib.get_hydro_timebins(h)
    catch err
        @warn "AREPO hydro timebin trace unavailable" exception=(err, catch_backtrace())
        return nothing
    end
end

function _estimate_radial_center(pos, vel, box)
    center = fill(box / 2, 3)
    for _ in 1:6
        a = zeros(Float64, 3, 3)
        b = zeros(Float64, 3)
        @inbounds for i in axes(pos, 1)
            p = [pos[i, 1], pos[i, 2], pos[i, 3]]
            for k in 1:3
                p[k] -= box * round((p[k] - center[k]) / box)
            end
            v = [vel[i, 1], vel[i, 2], vel[i, 3]]
            vn = norm(v)
            vn == 0 && continue
            n = v / vn
            proj = Matrix{Float64}(I, 3, 3) - n * n'
            a .+= proj
            b .+= proj * p
        end
        center .= a \ b
    end
    return mod.(center, box)
end

function radial_bins(s::NohSnapshot; nbins = NBINS)
    radii = similar(s.rho)
    coords = s.pos
    center = _estimate_radial_center(coords, s.vel, s.box)
    @inbounds for i in eachindex(s.rho)
        dx = coords[i, 1] - center[1]
        dy = coords[i, 2] - center[2]
        dz = coords[i, 3] - center[3]
        dx -= s.box * round(dx / s.box)
        dy -= s.box * round(dy / s.box)
        dz -= s.box * round(dz / s.box)
        r = sqrt(dx * dx + dy * dy + dz * dz)
        radii[i] = r
    end
    rmax = maximum(radii)
    edges = range(0, rmax; length = nbins + 1)
    rows = Vector{NamedTuple}()
    for b in 1:nbins
        lo = edges[b]
        hi = edges[b + 1]
        idx = findall(r -> (b == nbins ? lo <= r <= hi : lo <= r < hi), radii)
        if isempty(idx)
            push!(rows, (; label = s.label, bin = b, r_min = lo, r_max = hi,
                         r_mid = 0.5 * (lo + hi), count = 0,
                         rho = NaN, rho_std = NaN, pressure = NaN))
        else
            push!(rows, (; label = s.label, bin = b, r_min = lo, r_max = hi,
                         r_mid = mean(radii[idx]), count = length(idx),
                         rho = mean(s.rho[idx]), rho_std = std(s.rho[idx]),
                         pressure = mean(s.pressure[idx])))
        end
    end
    return rows
end

csvquote(v) = "\"" * replace(string(v), "\"" => "\"\"") * "\""

function write_bins_csv(path, snapshots)
    open(path, "w") do io
        println(io, "label,bin,r_min,r_max,r_mid,count,density,density_std,pressure")
        for s in snapshots, row in radial_bins(s)
            vals = (row.label, row.bin, row.r_min, row.r_max, row.r_mid, row.count,
                    row.rho, row.rho_std, row.pressure)
            println(io, join((csvquote(v) for v in vals), ","))
        end
    end
end

function profile_compare(reference::NohSnapshot, candidate::NohSnapshot)
    rrows = radial_bins(reference)
    crows = radial_bins(candidate)
    n = min(length(rrows), length(crows))
    rho_diffs = Float64[]
    p_diffs = Float64[]
    for i in 1:n
        r = rrows[i]
        c = crows[i]
        if isfinite(r.rho) && isfinite(c.rho)
            push!(rho_diffs, abs(c.rho - r.rho))
        end
        if isfinite(r.pressure) && isfinite(c.pressure)
            push!(p_diffs, abs(c.pressure - r.pressure))
        end
    end
    return (; reference = reference.label,
            candidate = candidate.label,
            bins = n,
            rho_l1 = isempty(rho_diffs) ? NaN : mean(rho_diffs),
            rho_linf = isempty(rho_diffs) ? NaN : maximum(rho_diffs),
            pressure_l1 = isempty(p_diffs) ? NaN : mean(p_diffs),
            pressure_linf = isempty(p_diffs) ? NaN : maximum(p_diffs))
end

function _field_stats(reference, candidate)
    n = min(length(reference), length(candidate))
    n == 0 && return (; n = 0, l1 = NaN, linf = NaN, rel_l1 = NaN, rel_linf = NaN)
    diffs = abs.(view(candidate, 1:n) .- view(reference, 1:n))
    refscale = maximum(abs.(view(reference, 1:n)))
    linf = maximum(diffs)
    l1 = mean(diffs)
    denom = max(refscale, eps(Float64))
    return (; n, l1, linf, rel_l1 = l1 / denom, rel_linf = linf / denom)
end

function field_compare(reference::NohSnapshot, candidate::NohSnapshot)
    fields = [
        (:rho, reference.rho, candidate.rho),
        (:pressure, reference.pressure, candidate.pressure),
        (:volume, reference.volume, candidate.volume),
        (:mass, reference.mass, candidate.mass),
        (:mx, view(reference.momentum, :, 1), view(candidate.momentum, :, 1)),
        (:my, view(reference.momentum, :, 2), view(candidate.momentum, :, 2)),
        (:mz, view(reference.momentum, :, 3), view(candidate.momentum, :, 3)),
        (:energy, reference.energy, candidate.energy),
        (:vx, view(reference.vel, :, 1), view(candidate.vel, :, 1)),
        (:vy, view(reference.vel, :, 2), view(candidate.vel, :, 2)),
        (:vz, view(reference.vel, :, 3), view(candidate.vel, :, 3)),
    ]
    rows = NamedTuple[]
    for (name, ref, cand) in fields
        stats = _field_stats(ref, cand)
        push!(rows, (; reference = reference.label,
                     candidate = candidate.label,
                     field = String(name),
                     cells = stats.n,
                     l1 = stats.l1,
                     linf = stats.linf,
                     rel_l1 = stats.rel_l1,
                     rel_linf = stats.rel_linf))
    end
    return rows
end

function write_field_compare_csv(path, rows)
    open(path, "w") do io
        println(io, "reference,candidate,field,cells,l1,linf,rel_l1,rel_linf,atol,rtol,status")
        for r in rows
            ok = field_compare_ok(r)
            vals = (r.reference, r.candidate, r.field, r.cells,
                    r.l1, r.linf, r.rel_l1, r.rel_linf,
                    FIELD_ATOL, FIELD_RTOL, ok ? "passed" : "failed")
            println(io, join((csvquote(v) for v in vals), ","))
        end
    end
end

function field_compare_ok(row)
    row.cells > 0 && isfinite(row.linf) &&
        (row.linf <= FIELD_ATOL || row.rel_linf <= FIELD_RTOL)
end

function write_report(path, staged_dir, snapshots, diag_rows, pf_rows,
                      profile_rows, field_rows, hierarchy_rows, step_statuses)
    volume_target = first(snapshots).box ^ 3
    initial_volume_ok = abs(first(diag_rows).volume - volume_target) <= 5e-10 * volume_target
    moving_diag = findfirst(d -> d.label == "powerfoam_moving_step1", diag_rows)
    moving_volume_gap = moving_diag === nothing ? NaN :
                        abs(diag_rows[moving_diag].volume - volume_target)
    moving_volume_ok = moving_diag !== nothing &&
                       moving_volume_gap <= 2e-2 * volume_target
    smoke_rows = filter(d -> d.label != "powerfoam_moving_step1", diag_rows)
    rows_ok = all(d -> d.rho_min > 0 && d.pressure_min > 0 &&
                       isfinite(d.mass) && isfinite(d.volume) &&
                       abs(d.volume - volume_target) <= 2e-2 * volume_target,
                  smoke_rows)
    hierarchy_ok = all(h -> h.active_count > 0 && h.active_count < h.cells &&
                            h.active_mismatches == 0 &&
                            h.active_list_mismatches == 0 &&
                            h.rho_min > 0 && h.pressure_min > 0 &&
                            isfinite(h.mass_gap) && isfinite(h.energy_gap),
                       hierarchy_rows)
    field_ok = isempty(field_rows) || all(field_compare_ok, field_rows)
    status = (initial_volume_ok && rows_ok && hierarchy_ok &&
              (!REQUIRE_FIELD_PARITY || field_ok)) ? "passed" : "failed"
    open(path, "w") do io
        println(io, "# AREPO 3-D Noh Smoke Gate")
        println(io)
        println(io, "This gate stages the stock AREPO `noh_3d` problem, initializes it")
        println(io, "through the linked AREPO library, advances a small bounded number")
        println(io, "of native sync steps, and exports radial diagnostics. It is an")
        println(io, "executable standard-problem smoke/import gate, not yet a final")
        println(io, "PowerFoam-vs-AREPO radial profile parity claim.")
        println(io)
        @printf(io, "- status: %s\n", status)
        @printf(io, "- AREPO library: `%s`\n", ArepoLib.libpath())
        @printf(io, "- staged case: `%s`\n", staged_dir)
        @printf(io, "- stock grid: 30^3\n")
        @printf(io, "- requested native steps: %d\n", N_STEPS)
        @printf(io, "- native step statuses: `%s`\n", join(string.(step_statuses), ", "))
        @printf(io, "- solver: `%s`\n", uppercase(String(RIEMANN)))
        @printf(io, "- radial bins: `%s`\n", relpath(joinpath(OUTDIR, "radial_bins.csv"), dirname(path)))
        @printf(io, "- field comparison: `%s`\n",
                relpath(joinpath(OUTDIR, "field_compare.csv"), dirname(path)))
        @printf(io, "- field tolerance: abs %.12g or rel %.12g\n",
                FIELD_ATOL, FIELD_RTOL)
        @printf(io, "- require field parity: `%s`\n", REQUIRE_FIELD_PARITY)
        @printf(io, "- field parity status: %s\n", field_ok ? "passed" : "failed")
        if moving_diag !== nothing
            @printf(io, "- moving rebuild diagnostic: %s, volume gap %.12g\n",
                    moving_volume_ok ? "passed" : "failed", moving_volume_gap)
        end
        println(io)
        println(io, "## Native Diagnostics")
        println(io)
        println(io, "| snapshot | time | cells | faces | volume | mass | rho min | rho max | pressure min | pressure max | energy | rms Mach |")
        println(io, "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for d in diag_rows
            @printf(io, "| %s | %.12g | %d | %d | %.12g | %.12g | %.12g | %.12g | %.12g | %.12g | %.12g | %.12g |\n",
                    d.label, d.time, d.cells, d.faces, d.volume, d.mass,
                    d.rho_min, d.rho_max, d.pressure_min, d.pressure_max,
                    d.energy, d.mach_rms)
        end
        println(io)
        println(io, "## PowerFoam Import Round Trip")
        println(io)
        println(io, "These gaps compare PowerFoam totals rebuilt from live AREPO primitives")
        println(io, "and volumes against AREPO's conserved arrays for the same snapshot.")
        println(io)
        println(io, "| snapshot | mass gap | mx gap | my gap | mz gap | energy gap |")
        println(io, "| --- | ---: | ---: | ---: | ---: | ---: |")
        for p in pf_rows
            @printf(io, "| %s | %.12g | %.12g | %.12g | %.12g | %.12g |\n",
                    p.label, p.mass_gap, p.mx_gap, p.my_gap, p.mz_gap, p.energy_gap)
        end
        println(io)
        println(io, "## PowerFoam Profile Comparison")
        println(io)
        println(io, "The PowerFoam rows compare first-order HLL/LLF one-step updates")
        println(io, "against AREPO's native one-step result. `powerfoam_fixed_step1`")
        println(io, "reuses AREPO's initial mesh. `powerfoam_moving_step1` is a")
        println(io, "diagnostic attempt to rebuild PowerFoam's local periodic 3-D")
        println(io, "Voronoi mesh from the canonical stock Noh lattice when available,")
        println(io, "then advects generators with the analytic Noh velocity. When the")
        println(io, "moving rebuild diagnostic above passes volume closure, this row is")
        println(io, "a certified moving full-sync rebuild comparison, not yet a partial")
        println(io, "active-cell hierarchy or final-field parity result.")
        println(io)
        println(io, "| reference | candidate | bins | density L1 | density Linf | pressure L1 | pressure Linf |")
        println(io, "| --- | --- | ---: | ---: | ---: | ---: | ---: |")
        for p in profile_rows
            @printf(io, "| %s | %s | %d | %.12g | %.12g | %.12g | %.12g |\n",
                    p.reference, p.candidate, p.bins, p.rho_l1, p.rho_linf,
                    p.pressure_l1, p.pressure_linf)
        end
        println(io)
        println(io, "## PowerFoam Final-Field Comparison")
        println(io)
        println(io, "These rows compare cell-indexed primitive and conserved fields")
        println(io, "against AREPO's native first step. They are the direct final-field")
        println(io, "tolerance inputs for the Noh parity rung; the gate still reports")
        println(io, "them as diagnostics because the fixed, moving, and traced hierarchy")
        println(io, "updates intentionally exercise different pieces of the path.")
        println(io)
        println(io, "| reference | candidate | field | cells | L1 | Linf | relative L1 | relative Linf | status |")
        println(io, "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |")
        for r in field_rows
            @printf(io, "| %s | %s | %s | %d | %.12g | %.12g | %.12g | %.12g | %s |\n",
                    r.reference, r.candidate, r.field, r.cells, r.l1, r.linf,
                    r.rel_l1, r.rel_linf,
                    field_compare_ok(r) ? "passed" : "failed")
        end
        println(io)
        println(io, "## PowerFoam Hierarchical Timestep Probe")
        println(io)
        println(io, "This probe runs the reconstructed active-cell hierarchy update on")
        println(io, "Noh local-periodic moving-face geometry. When AREPO exposes live")
        println(io, "hydro timebins, the active set comes from `P[].TimeBinHydro` and")
        println(io, "`TimeBinsHydro.ActiveParticleList`; otherwise the row falls back")
        println(io, "to a deterministic radial split. This certifies the hierarchy hot")
        println(io, "path for Noh-like moving geometry, not yet final field parity.")
        println(io)
        println(io, "| label | source | cells | faces | active cells | active fraction | active mismatches | active-list mismatches | ti current | ti next | dt | mass gap | mx gap | my gap | mz gap | energy gap | rho min | pressure min |")
        println(io, "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for h in hierarchy_rows
            @printf(io, "| %s | %s | %d | %d | %d | %.12g | %d | %d | %d | %d | %.12g | %.12g | %.12g | %.12g | %.12g | %.12g | %.12g | %.12g |\n",
                    h.label, h.source, h.cells, h.faces, h.active_count,
                    h.active_fraction, h.active_mismatches,
                    h.active_list_mismatches, h.ti_current, h.ti_next, h.dt,
                    h.mass_gap, h.mx_gap, h.my_gap, h.mz_gap, h.energy_gap,
                    h.rho_min, h.pressure_min)
        end
        println(io)
        println(io, "## Next Gate")
        println(io)
        if moving_volume_ok
            println(io, "Tighten the field tolerances only after the fixed/moving rows")
            println(io, "are promoted from diagnostic comparisons to the traced native")
            println(io, "hierarchy update path.")
        else
            println(io, "Fix local periodic 3-D candidate coverage for converging Noh")
            println(io, "generator motion before certifying the moving-mesh/hierarchical")
            println(io, "PowerFoam Noh parity rung.")
        end
    end
    return status
end

function main()
    mkpath(OUTDIR)
    staged_dir = stage_arepo_case(; riemann = RIEMANN)
    h = cd(() -> ArepoLib.init("param.txt"), staged_dir)
    snapshots = NohSnapshot[]
    timebin_traces = Any[]
    step_statuses = Symbol[]
    try
        push!(snapshots, collect_snapshot(h, "initial"))
        push!(timebin_traces, maybe_hydro_timebins(h))
        for step in 1:N_STEPS
            status = ArepoLib.run_step!(h)
            push!(step_statuses, status)
            push!(snapshots, collect_snapshot(h, "step$(step)"))
            push!(timebin_traces, maybe_hydro_timebins(h))
            status === :continue || break
        end
        if length(snapshots) >= 2
            dt_pf = snapshots[2].time - snapshots[1].time
            push!(snapshots, powerfoam_fixed_geometry_step(snapshots[1], dt_pf;
                                                           riemann = RIEMANN))
            push!(snapshots, powerfoam_moving_mesh_step(snapshots[1], dt_pf;
                                                        riemann = RIEMANN))
        end
        hierarchy_trace = length(timebin_traces) >= 2 ? timebin_traces[2] : nothing
        hierarchy_source_snapshot = hierarchy_trace === nothing ? snapshots[1] : snapshots[2]
        hierarchy_rows = length(snapshots) >= 2 ?
                         [powerfoam_moving_hierarchy_probe(hierarchy_source_snapshot,
                                                           snapshots[2].time -
                                                           snapshots[1].time;
                                                           riemann = RIEMANN,
                                                           arepo_bins = hierarchy_trace,
                                                           use_snapshot_points = hierarchy_trace !== nothing)] :
                         NamedTuple[]
        diag_rows = primitive_diagnostics.(snapshots)
        pf_rows = powerfoam_roundtrip_diagnostics.(snapshots)
        profile_rows = length(snapshots) >= 4 ?
                       [profile_compare(snapshots[2], snapshots[3]),
                        profile_compare(snapshots[2], snapshots[4])] :
                       length(snapshots) >= 3 ?
                       [profile_compare(snapshots[2], snapshots[3])] :
                       NamedTuple[]
        field_rows = NamedTuple[]
        if length(snapshots) >= 4
            append!(field_rows, field_compare(snapshots[2], snapshots[3]))
            append!(field_rows, field_compare(snapshots[2], snapshots[4]))
        elseif length(snapshots) >= 3
            append!(field_rows, field_compare(snapshots[2], snapshots[3]))
        end
        bins_csv = joinpath(OUTDIR, "radial_bins.csv")
        fields_csv = joinpath(OUTDIR, "field_compare.csv")
        report = joinpath(OUTDIR, "README.md")
        write_bins_csv(bins_csv, snapshots)
        write_field_compare_csv(fields_csv, field_rows)
        status = write_report(report, staged_dir, snapshots, diag_rows, pf_rows,
                              profile_rows, field_rows, hierarchy_rows,
                              step_statuses)
        @printf("wrote %s\n", report)
        @printf("wrote %s\n", bins_csv)
        @printf("wrote %s\n", fields_csv)
        @printf("noh3d smoke %s: snapshots=%d final_time=%.12g\n",
                status, length(snapshots), snapshots[end].time)
        status == "passed" || exit(1)
    finally
        ArepoLib.finalize(h)
    end
end

main()
