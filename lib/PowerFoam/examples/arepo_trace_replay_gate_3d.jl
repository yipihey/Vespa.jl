using Printf
using Statistics
using KernelAbstractions

include(joinpath(@__DIR__, "arepo_geometry_gate_3d.jl"))

const REPLAY_OUTBASE = joinpath(@__DIR__, "out", "arepo_trace_replay_gate_3d")
const REPLAY_OUTDIR = joinpath(REPLAY_OUTBASE, replace(RUN_TAG, "." => "p"))
const REPLAY_TOL = 1e-10
const REPLAY_GEOMETRY = Symbol(lowercase(get(ENV, "POWERFOAM_REPLAY_GEOMETRY", "trace")))
const REPLAY_NATIVE_RADIUS = parse(Int, get(ENV, "POWERFOAM_NATIVE_TRACE_RADIUS", "1"))
const REPLAY_NATIVE_MIN_FACE_SURFACE_FRACTION =
    parse(Float64, get(ENV, "POWERFOAM_NATIVE_TRACE_MIN_FACE_SURFACE_FRACTION", "1e-5"))
const REPLAY_FACE_VELOCITY =
    Symbol(lowercase(get(ENV, "POWERFOAM_REPLAY_FACE_VELOCITY", "trace")))
const REPLAY_UPDATE_TARGETS =
    Symbol(lowercase(get(ENV, "POWERFOAM_REPLAY_UPDATE_TARGETS", "trace")))
const REPLAY_ROWS = Symbol(lowercase(get(ENV, "POWERFOAM_REPLAY_ROWS", "trace")))
const REPLAY_NATIVE_DT_SOURCE =
    Symbol(lowercase(get(ENV, "POWERFOAM_REPLAY_NATIVE_DT_SOURCE", "trace_cells")))

function _replay_bridge_available()
    return isdefined(Main, :ArepoLib) &&
           isdefined(ArepoLib, :get_hydro_face_traces_3d) &&
           isdefined(ArepoLib, :get_hydro_preflux_states_3d)
end

function _conserved_after_arepo_step(h, ng)
    ids = ArepoLib.get_particle_ids(h)[1:ng]
    mass = ArepoLib.get_particle_field(h, :mass)[1:ng]
    momentum = ArepoLib.get_cell_field(h, :momentum)
    energy = ArepoLib.get_cell_field(h, :energy)
    conserved = hcat(mass, momentum[:, 1], momentum[:, 2], momentum[:, 3], energy)
    volume = ArepoLib.get_cell_field(h, :volume)
    primitive = hcat(ArepoLib.get_cell_field(h, :rho),
                     ArepoLib.get_particle_field(h, :vel)[1:ng, :],
                     ArepoLib.get_cell_field(h, :pressure))
    return (; ids, conserved, volume, primitive)
end

function _apply_trace_pass_conserved(snapshot, trace, pass_index)
    out = copy(snapshot.conserved)
    pass = _replay_pass_geometry(snapshot, trace, pass_index)
    idx = pass.idx
    side1 = pass.geom_update.c1
    side2 = pass.geom_update.c2
    for (row, i) in pairs(idx)
        fac = 0.5 * trace.face_dt[i] * trace.area[i]
        c1 = side1[row]
        c2 = side2[row]
        if c1 > 0
            @views out[c1, :] .-= fac .* trace.flux_lab[i, :]
        end
        if c2 > 0
            @views out[c2, :] .+= fac .* trace.flux_lab[i, :]
        end
    end
    return out
end

function _primitive_from_conserved(conserved, volume; gamma = GAMMA)
    mass = conserved[:, 1]
    rho = mass ./ volume
    vx = conserved[:, 2] ./ mass
    vy = conserved[:, 3] ./ mass
    vz = conserved[:, 4] ./ mass
    etot_density = conserved[:, 5] ./ volume
    kinetic_density = 0.5 .* rho .* (vx .* vx .+ vy .* vy .+ vz .* vz)
    pressure = (gamma - 1) .* (etot_density .- kinetic_density)
    return hcat(rho, vx, vy, vz, pressure)
end

function _packed_face_states(mat)
    nf = size(mat, 1)
    out = Vector{Float64}(undef, 5nf)
    @inbounds for k in 1:5, f in 1:nf
        out[(k - 1) * nf + f] = mat[f, k]
    end
    return out
end

function _state_from_conserved(conserved, volume)
    return EulerState3D(conserved[:, 1] ./ volume,
                        conserved[:, 2] ./ volume,
                        conserved[:, 3] ./ volume,
                        conserved[:, 4] ./ volume,
                        conserved[:, 5] ./ volume)
end

function _state_to_conserved(state, volume)
    return hcat(Array(state.D) .* volume,
                Array(state.Mx) .* volume,
                Array(state.My) .* volume,
                Array(state.Mz) .* volume,
                Array(state.E) .* volume)
end

function _primitive_state_from_snapshot(snapshot)
    return PrimitiveState3D(snapshot.primitive[:, 1],
                            snapshot.primitive[:, 2],
                            snapshot.primitive[:, 3],
                            snapshot.primitive[:, 4],
                            snapshot.primitive[:, 5])
end

function _gradients_from_snapshot(snapshot, be; T = Float64)
    hasproperty(snapshot, :gradients) ||
        error("pre-flux snapshot lacks gradients; rebuild AREPO/ArepoLib bridge")
    return Base.invokelatest(getfield(Main, :hydro_gradients_from_arepo),
                             snapshot.gradients, be; T)
end

function _periodic_point_owner(pos, point; box = 1.0, tol = 1e-9)
    best_i = 0
    best_d2 = Inf
    @inbounds for i in axes(pos, 1)
        dx = point[1] - pos[i, 1]
        dy = point[2] - pos[i, 2]
        dz = point[3] - pos[i, 3]
        dx -= round(dx / box) * box
        dy -= round(dy / box) * box
        dz -= round(dz / box) * box
        d2 = dx * dx + dy * dy + dz * dz
        if d2 < best_d2
            best_d2 = d2
            best_i = i
        end
    end
    best_d2 <= tol * tol ||
        error("could not map traced face endpoint to a local periodic generator; closest squared distance=$best_d2")
    return best_i
end

function _native_update_targets_from_trace_points(snapshot, trace, idx)
    hasproperty(trace, :point_l) && hasproperty(trace, :point_r) ||
        error("trace lacks point_l/point_r; rebuild AREPO/ArepoLib bridge")
    nf = length(idx)
    update_c1 = Vector{Int32}(undef, nf)
    update_c2 = Vector{Int32}(undef, nf)
    @inbounds for row in 1:nf
        i = idx[row]
        l = _periodic_point_owner(snapshot.pos,
                                  (trace.point_l[i, 1],
                                   trace.point_l[i, 2],
                                   trace.point_l[i, 3]))
        r = _periodic_point_owner(snapshot.pos,
                                  (trace.point_r[i, 1],
                                   trace.point_r[i, 2],
                                   trace.point_r[i, 3]))
        if l == r
            update_c1[row] = Int32(0)
            update_c2[row] = Int32(0)
        else
            update_c1[row] = Int32(l)
            update_c2[row] = Int32(r)
        end
    end
    return update_c1, update_c2
end

function _select_update_targets(snapshot, trace, idx)
    trace_c1 = Int32.(trace.update_c1[idx])
    trace_c2 = Int32.(trace.update_c2[idx])
    if REPLAY_UPDATE_TARGETS == :trace
        return trace_c1, trace_c2, 0
    elseif REPLAY_UPDATE_TARGETS == :native
        update_c1, update_c2 =
            _native_update_targets_from_trace_points(snapshot, trace, idx)
        mismatches = count(i -> update_c1[i] != trace_c1[i] ||
                                update_c2[i] != trace_c2[i],
                           eachindex(update_c1))
        return update_c1, update_c2, mismatches
    elseif REPLAY_UPDATE_TARGETS == :native_mesh
        return trace_c1, trace_c2, 0
    else
        error("unsupported POWERFOAM_REPLAY_UPDATE_TARGETS=$(REPLAY_UPDATE_TARGETS); use trace, native, or native_mesh")
    end
end

function _trace_cell_dt_for_pass(snapshot, trace, pass_index)
    dt_host = zeros(Float64, length(snapshot.ids))
    idx = findall(i -> trace.active[i] && trace.pass_index[i] == pass_index,
                  eachindex(trace.c1))
    @inbounds for i in idx
        trace.update_c1[i] > 0 && (dt_host[trace.update_c1[i]] = trace.state_dt_l[i])
        trace.update_c2[i] > 0 && (dt_host[trace.update_c2[i]] = trace.state_dt_r[i])
    end
    return dt_host
end

function _trace_pass_geometry(snapshot, trace, pass_index; T = Float64)
    idx = findall(i -> trace.active[i] && trace.pass_index[i] == pass_index,
                  eachindex(trace.c1))
    isempty(idx) && error("pass $pass_index has no active trace rows")
    hasproperty(trace, :update_c1) && hasproperty(trace, :update_c2) ||
        error("trace lacks update_c1/update_c2; rebuild AREPO/ArepoLib bridge")
    face_dt = trace.face_dt[idx]
    dt_min, dt_max = extrema(face_dt)
    isapprox(dt_min, dt_max; atol = 0, rtol = 1e-14) ||
        error("PowerFoam trace replay currently requires uniform face_dt per pass; got $dt_min..$dt_max")
    geom_c1 = Int32.(trace.c1[idx])
    geom_c2 = Int32.(trace.c2[idx])
    update_c1, update_c2, update_target_mismatches =
        _select_update_targets(snapshot, trace, idx)
    base_offsets, base_faces, base_signs =
        PowerFoam._cell_face_csr(length(snapshot.ids), geom_c1, geom_c2, Int32)
    geom_base = ArepoMeshArrays3D(geom_c1, geom_c2,
                                  base_offsets, base_faces, base_signs,
                                  T.(snapshot.volume),
                                  T.(trace.area[idx]),
                                  T.(trace.normal[idx, 1]),
                                  T.(trace.normal[idx, 2]),
                                  T.(trace.normal[idx, 3]),
                                  T.(trace.vel_face[idx, 1]),
                                  T.(trace.vel_face[idx, 2]),
                                  T.(trace.vel_face[idx, 3]))
    geom_update = with_update_targets_3d(geom_base, update_c1, update_c2)
    offsets, faces, signs = PowerFoam._cell_face_csr(length(snapshot.ids),
                                                     update_c1, update_c2,
                                                     Int32)
    geom_predict = ArepoMeshArrays3D(update_c1, update_c2,
                                     offsets, faces, signs,
                                     T.(snapshot.volume),
                                     T.(trace.area[idx]),
                                     T.(trace.normal[idx, 1]),
                                     T.(trace.normal[idx, 2]),
                                     T.(trace.normal[idx, 3]),
                                     T.(trace.vel_face[idx, 1]),
                                     T.(trace.vel_face[idx, 2]),
                                     T.(trace.vel_face[idx, 3]))
    geometry_stats = (; pass_index,
                      max_face_velocity_diff = NaN,
                      max_local_face_velocity_diff = NaN,
                      max_one_sided_face_velocity_diff = NaN,
                      update_target_mismatches)
    return (; idx, dt = dt_min, geom_update, geom_predict,
            face_center = trace.face_center[idx, :],
            flux_activity_c2 = face_update_activity_3d(update_c1, update_c2),
            geometry_stats)
end

function _native_row_set(geom)
    rows = Dict{Tuple{Int,Int},Int}()
    c1 = Int.(geom.c1)
    c2 = Int.(geom.c2)
    @inbounds for f in eachindex(c1)
        c1[f] > 0 && c2[f] > 0 || continue
        key = c1[f] <= c2[f] ? (c1[f], c2[f]) : (c2[f], c1[f])
        rows[key] = f
    end
    return rows
end

@inline function _nearest_periodic_image(x, ref; box = 1.0)
    return x - round((x - ref) / box) * box
end

function _oriented_image_delta(pos, i::Int, j::Int, n)
    best = (0.0, 0.0, 0.0)
    best_nn = 0.0
    best_score = -Inf
    @inbounds for sz in -1:1, sy in -1:1, sx in -1:1
        dx = pos[j, 1] + sx - pos[i, 1]
        dy = pos[j, 2] + sy - pos[i, 2]
        dz = pos[j, 3] + sz - pos[i, 3]
        nn = sqrt(dx * dx + dy * dy + dz * dz)
        nn > 0 || continue
        score = (dx * n[1] + dy * n[2] + dz * n[3]) / nn
        if score > best_score
            best = (dx, dy, dz)
            best_nn = nn
            best_score = score
        end
    end
    return best, best_nn
end

function _arepo_face_velocity(pos, velvertex, i::Int, j::Int, n, fc)
    d, nn = _oriented_image_delta(pos, i, j, n)
    p1 = (pos[i, 1], pos[i, 2], pos[i, 3])
    p2 = (p1[1] + d[1], p1[2] + d[2], p1[3] + d[3])
    v1 = (velvertex[i, 1], velvertex[i, 2], velvertex[i, 3])
    v2 = (velvertex[j, 1], velvertex[j, 2], velvertex[j, 3])
    wx = 0.5 * (v1[1] + v2[1])
    wy = 0.5 * (v1[2] + v2[2])
    wz = 0.5 * (v1[3] + v2[3])
    cx = fc[1] - 0.5 * (p1[1] + p2[1])
    cy = fc[2] - 0.5 * (p1[2] + p2[2])
    cz = fc[3] - 0.5 * (p1[3] + p2[3])
    facv = (cx * (v1[1] - v2[1]) + cy * (v1[2] - v2[2]) +
            cz * (v1[3] - v2[3])) / nn
    cc = sqrt(cx * cx + cy * cy + cz * cz)
    if cc > 0.9 * nn
        facv *= (0.9 * nn) / cc
    end
    return (wx + facv * n[1], wy + facv * n[2], wz + facv * n[3])
end

function _arepo_face_velocity_from_points(p1, p2, v1, v2, n, fc)
    wx = 0.5 * (v1[1] + v2[1])
    wy = 0.5 * (v1[2] + v2[2])
    wz = 0.5 * (v1[3] + v2[3])
    dx = p2[1] - p1[1]
    dy = p2[2] - p1[2]
    dz = p2[3] - p1[3]
    nn = sqrt(dx * dx + dy * dy + dz * dz)
    cx = fc[1] - 0.5 * (p1[1] + p2[1])
    cy = fc[2] - 0.5 * (p1[2] + p2[2])
    cz = fc[3] - 0.5 * (p1[3] + p2[3])
    facv = (cx * (v1[1] - v2[1]) + cy * (v1[2] - v2[2]) +
            cz * (v1[3] - v2[3])) / nn
    cc = sqrt(cx * cx + cy * cy + cz * cz)
    if cc > 0.9 * nn
        facv *= (0.9 * nn) / cc
    end
    return (wx + facv * n[1], wy + facv * n[2], wz + facv * n[3])
end

function _arepo_face_velocity_from_shift(pos, velvertex, i::Int, j::Int, shift,
                                         n, fc)
    p1 = (pos[i, 1], pos[i, 2], pos[i, 3])
    p2 = (pos[j, 1] + shift[1], pos[j, 2] + shift[2],
          pos[j, 3] + shift[3])
    v1 = (velvertex[i, 1], velvertex[i, 2], velvertex[i, 3])
    v2 = (velvertex[j, 1], velvertex[j, 2], velvertex[j, 3])
    return _arepo_face_velocity_from_points(p1, p2, v1, v2, n, fc)
end

function _native_pass_geometry(snapshot, trace, pass_index; T = Float64)
    hasproperty(snapshot, :pos) ||
        error("pre-flux snapshot lacks generator positions; rebuild AREPO/ArepoLib bridge")
    trace_pass = _trace_pass_geometry(snapshot, trace, pass_index; T)
    native = local_periodic_voronoi_mesh_arrays_3d(snapshot.pos;
        T, bins_per_axis = N, search_radius = REPLAY_NATIVE_RADIUS,
        min_face_surface_fraction = REPLAY_NATIVE_MIN_FACE_SURFACE_FRACTION,
        threaded = false)
    native_rows = _native_row_set(native.geom)
    idx = trace_pass.idx
    trace_update_c1 = Int32.(trace.update_c1[idx])
    trace_update_c2 = Int32.(trace.update_c2[idx])
    point_update_c1, point_update_c2 =
        _native_update_targets_from_trace_points(snapshot, trace, idx)
    if REPLAY_UPDATE_TARGETS == :trace
        update_c1 = copy(trace_update_c1)
        update_c2 = copy(trace_update_c2)
    elseif REPLAY_UPDATE_TARGETS == :native
        update_c1 = copy(point_update_c1)
        update_c2 = copy(point_update_c2)
    elseif REPLAY_UPDATE_TARGETS == :native_mesh
        update_c1 = similar(trace_update_c1)
        update_c2 = similar(trace_update_c2)
    else
        error("unsupported POWERFOAM_REPLAY_UPDATE_TARGETS=$(REPLAY_UPDATE_TARGETS); use trace, native, or native_mesh")
    end
    nf = length(idx)
    area = Vector{T}(undef, nf)
    nx = Vector{T}(undef, nf)
    ny = Vector{T}(undef, nf)
    nz = Vector{T}(undef, nf)
    face_center = Matrix{T}(undef, nf, 3)
    native_vel = Matrix{T}(undef, nf, 3)
    max_face_velocity_diff = 0.0
    max_local_face_velocity_diff = 0.0
    max_one_sided_face_velocity_diff = 0.0
    @inbounds for row in 1:nf
        a = Int(point_update_c1[row])
        b = Int(point_update_c2[row])
        key = a <= b ? (a, b) : (b, a)
        nf_native = get(native_rows, key, 0)
        nf_native > 0 || error("native rebuild is missing trace pair $key in pass $pass_index")
        area[row] = T(native.geom.face_area[nf_native])
        nnx = Float64(native.geom.normal_x[nf_native])
        nny = Float64(native.geom.normal_y[nf_native])
        nnz = Float64(native.geom.normal_z[nf_native])
        dot_trace = nnx * trace.normal[idx[row], 1] +
                    nny * trace.normal[idx[row], 2] +
                    nnz * trace.normal[idx[row], 3]
        flipped = dot_trace < 0
        if flipped
            nnx = -nnx
            nny = -nny
            nnz = -nnz
        end
        if REPLAY_UPDATE_TARGETS == :native_mesh
            if flipped
                update_c1[row] = Int32(native.geom.c2[nf_native])
                update_c2[row] = Int32(native.geom.c1[nf_native])
            else
                update_c1[row] = Int32(native.geom.c1[nf_native])
                update_c2[row] = Int32(native.geom.c2[nf_native])
            end
        end
        nx[row] = T(nnx)
        ny[row] = T(nny)
        nz[row] = T(nnz)
        face_center[row, 1] = T(_nearest_periodic_image(
            native.face_center[nf_native, 1], trace.face_center[idx[row], 1]))
        face_center[row, 2] = T(_nearest_periodic_image(
            native.face_center[nf_native, 2], trace.face_center[idx[row], 2]))
        face_center[row, 3] = T(_nearest_periodic_image(
            native.face_center[nf_native, 3], trace.face_center[idx[row], 3]))
        if hasproperty(trace, :point_l) && hasproperty(trace, :point_r) &&
           hasproperty(trace, :velvertex_l) && hasproperty(trace, :velvertex_r)
            nv = _arepo_face_velocity_from_points(
                (trace.point_l[idx[row], 1], trace.point_l[idx[row], 2],
                 trace.point_l[idx[row], 3]),
                (trace.point_r[idx[row], 1], trace.point_r[idx[row], 2],
                 trace.point_r[idx[row], 3]),
                (trace.velvertex_l[idx[row], 1], trace.velvertex_l[idx[row], 2],
                 trace.velvertex_l[idx[row], 3]),
                (trace.velvertex_r[idx[row], 1], trace.velvertex_r[idx[row], 2],
                 trace.velvertex_r[idx[row], 3]),
                (nnx, nny, nnz),
                (Float64(face_center[row, 1]), Float64(face_center[row, 2]),
                 Float64(face_center[row, 3])))
        else
            gi = trace.c1[idx[row]] > 0 ? Int(trace.c1[idx[row]]) : a
            gj = trace.c2[idx[row]] > 0 ? Int(trace.c2[idx[row]]) : b
            nv = _arepo_face_velocity(snapshot.pos, snapshot.velvertex, gi, gj,
                                      (nnx, nny, nnz),
                                      (Float64(face_center[row, 1]),
                                       Float64(face_center[row, 2]),
                                       Float64(face_center[row, 3])))
        end
        native_vel[row, 1] = T(nv[1])
        native_vel[row, 2] = T(nv[2])
        native_vel[row, 3] = T(nv[3])
        dv = sqrt((nv[1] - trace.vel_face[idx[row], 1])^2 +
                  (nv[2] - trace.vel_face[idx[row], 2])^2 +
                  (nv[3] - trace.vel_face[idx[row], 3])^2)
        max_face_velocity_diff = max(max_face_velocity_diff, dv)
        if trace.c1[idx[row]] > 0 && trace.c2[idx[row]] > 0
            max_local_face_velocity_diff =
                max(max_local_face_velocity_diff, dv)
        else
            max_one_sided_face_velocity_diff =
                max(max_one_sided_face_velocity_diff, dv)
        end
    end
    update_target_mismatches = count(i -> update_c1[i] != trace_update_c1[i] ||
                                          update_c2[i] != trace_update_c2[i],
                                     eachindex(update_c1))
    offsets, faces, signs = PowerFoam._cell_face_csr(length(snapshot.ids),
                                                     update_c1, update_c2,
                                                     Int32)
    if REPLAY_FACE_VELOCITY == :trace
        fvx = T.(trace.vel_face[idx, 1])
        fvy = T.(trace.vel_face[idx, 2])
        fvz = T.(trace.vel_face[idx, 3])
    elseif REPLAY_FACE_VELOCITY == :native
        fvx = native_vel[:, 1]
        fvy = native_vel[:, 2]
        fvz = native_vel[:, 3]
    else
        error("unsupported POWERFOAM_REPLAY_FACE_VELOCITY=$(REPLAY_FACE_VELOCITY); use trace or native")
    end
    geom = ArepoMeshArrays3D(update_c1, update_c2, offsets, faces, signs,
                             T.(snapshot.volume), area, nx, ny, nz,
                             fvx, fvy, fvz)
    return (; idx, dt = trace_pass.dt, geom_update = geom, geom_predict = geom,
            face_center,
            flux_activity_c2 = face_update_activity_3d(update_c1, update_c2),
            geometry_stats = (; pass_index, max_face_velocity_diff,
                              max_local_face_velocity_diff,
                              max_one_sided_face_velocity_diff,
                              update_target_mismatches))
end

function _native_full_pass_geometry(snapshot, trace, pass_index; T = Float64)
    hasproperty(snapshot, :pos) ||
        error("pre-flux snapshot lacks generator positions; rebuild AREPO/ArepoLib bridge")
    idx = findall(i -> trace.active[i] && trace.pass_index[i] == pass_index,
                  eachindex(trace.c1))
    isempty(idx) && error("pass $pass_index has no active trace rows")
    face_dt = trace.face_dt[idx]
    dt_min, dt_max = extrema(face_dt)
    isapprox(dt_min, dt_max; atol = 0, rtol = 1e-14) ||
        error("native row replay currently requires uniform face_dt per pass; got $dt_min..$dt_max")
    native = local_periodic_voronoi_mesh_arrays_3d(snapshot.pos;
        T, bins_per_axis = N, search_radius = REPLAY_NATIVE_RADIUS,
        min_face_surface_fraction = REPLAY_NATIVE_MIN_FACE_SURFACE_FRACTION,
        threaded = false)
    nf = length(native.geom.c1)
    fvx = Vector{T}(undef, nf)
    fvy = Vector{T}(undef, nf)
    fvz = Vector{T}(undef, nf)
    @inbounds for f in 1:nf
        i = Int(native.geom.c1[f])
        j = Int(native.geom.c2[f])
        n = (Float64(native.geom.normal_x[f]),
             Float64(native.geom.normal_y[f]),
             Float64(native.geom.normal_z[f]))
        fc = (Float64(native.face_center[f, 1]),
              Float64(native.face_center[f, 2]),
              Float64(native.face_center[f, 3]))
        shift = (Float64(native.face_image_shift[f, 1]),
                 Float64(native.face_image_shift[f, 2]),
                 Float64(native.face_image_shift[f, 3]))
        nv = _arepo_face_velocity_from_shift(snapshot.pos, snapshot.velvertex,
                                             i, j, shift, n, fc)
        fvx[f] = T(nv[1])
        fvy[f] = T(nv[2])
        fvz[f] = T(nv[3])
    end
    geom = ArepoMeshArrays3D(native.geom.c1, native.geom.c2,
                             native.geom.cell_face_offsets,
                             native.geom.cell_faces,
                             native.geom.cell_face_signs,
                             T.(snapshot.volume), native.geom.face_area,
                             native.geom.normal_x, native.geom.normal_y,
                             native.geom.normal_z, fvx, fvy, fvz)
    return (; idx = nothing, dt = dt_min, geom_update = geom,
            geom_predict = geom, face_center = T.(native.face_center),
            flux_activity_c2 = face_update_activity_3d(geom.c1, geom.c2),
            geometry_stats = (; pass_index,
                              max_face_velocity_diff = NaN,
                              max_local_face_velocity_diff = NaN,
                              max_one_sided_face_velocity_diff = NaN,
                              update_target_mismatches = 0))
end

function _replay_pass_geometry(snapshot, trace, pass_index; T = Float64)
    REPLAY_ROWS == :native &&
        return _native_full_pass_geometry(snapshot, trace, pass_index; T)
    REPLAY_ROWS == :trace ||
        error("unsupported POWERFOAM_REPLAY_ROWS=$(REPLAY_ROWS); use trace or native")
    REPLAY_GEOMETRY == :trace &&
        return _trace_pass_geometry(snapshot, trace, pass_index; T)
    REPLAY_GEOMETRY == :native &&
        return _native_pass_geometry(snapshot, trace, pass_index; T)
    error("unsupported POWERFOAM_REPLAY_GEOMETRY=$(REPLAY_GEOMETRY); use trace or native")
end

function _powerfoam_trace_kernel_pass(snapshot, trace, pass_index, target_volume;
                                      riemann = RIEMANN)
    pass = _replay_pass_geometry(snapshot, trace, pass_index)
    idx = pass.idx
    geom = pass.geom_update
    states = FaceStates3D(_packed_face_states(trace.state_face_l[idx, :]),
                          _packed_face_states(trace.state_face_r[idx, :]))
    state = _state_from_conserved(snapshot.conserved, snapshot.volume)
    work = hydro_work_3d(state, geom)
    be = KernelAbstractions.CPU()
    nf = length(idx)
    PowerFoam._face_flux_from_predicted_3d_k!(be)(
        work.FD, work.FMx, work.FMy, work.FMz, work.FE,
        states.left, states.right, pass.flux_activity_c2, geom.face_area,
        geom.normal_x, geom.normal_y, geom.normal_z,
        geom.face_vx, geom.face_vy, geom.face_vz,
        Float64(GAMMA), PowerFoam._solver_code(riemann), 1e-12;
        ndrange = nf)
    PowerFoam._cell_update_3d_k!(be)(
        state.D, state.Mx, state.My, state.Mz, state.E,
        work.FD, work.FMx, work.FMy, work.FMz, work.FE,
        geom.volume, target_volume, geom.cell_face_offsets,
        geom.cell_faces, geom.cell_face_signs, Float64(0.5 * pass.dt);
        ndrange = length(snapshot.ids))
    KernelAbstractions.synchronize(be)
    return _state_to_conserved(state, target_volume)
end

function _powerfoam_predictor_kernel_pass(snapshot, trace, pass_index, target_volume;
                                          riemann = RIEMANN)
    pass = _replay_pass_geometry(snapshot, trace, pass_index)
    state = _state_from_conserved(snapshot.conserved, snapshot.volume)
    geom = pass.geom_predict
    be = KernelAbstractions.CPU()
    states = face_prediction_work_3d(geom)
    prim = _primitive_state_from_snapshot(snapshot)
    gradients = _gradients_from_snapshot(snapshot, be; T = Float64)
    dt_host = zeros(Float64, length(snapshot.ids))
    if pass.idx === nothing
        if REPLAY_NATIVE_DT_SOURCE == :face_dt
            fill!(dt_host, pass.dt)
        elseif REPLAY_NATIVE_DT_SOURCE == :trace_cells
            dt_host .= _trace_cell_dt_for_pass(snapshot, trace, pass_index)
        elseif REPLAY_NATIVE_DT_SOURCE == :snapshot_time
            hasproperty(snapshot, :time) &&
                hasproperty(snapshot, :time_last_prim_update) ||
                error("POWERFOAM_REPLAY_NATIVE_DT_SOURCE=snapshot_time requires preflux snapshot timing fields")
            dt_host .= snapshot.time .- snapshot.time_last_prim_update
        else
            error("unsupported POWERFOAM_REPLAY_NATIVE_DT_SOURCE=$(REPLAY_NATIVE_DT_SOURCE); use trace_cells, snapshot_time, or face_dt")
        end
    else
        @inbounds for (row, i) in pairs(pass.idx)
            c1 = Int(geom.c1[row])
            c2 = Int(geom.c2[row])
            c1 > 0 && (dt_host[c1] = trace.state_dt_l[i])
            c2 > 0 && (dt_host[c2] = trace.state_dt_r[i])
        end
    end
    predict_face_states_3d!(states, geom, gradients, prim, snapshot.center,
                            pass.face_center; dt_extrapolation = dt_host,
                            box_size = 1.0, gamma = GAMMA)
    work = hydro_work_3d(state, geom)
    nf = length(geom.c1)
    PowerFoam._face_flux_from_predicted_3d_k!(be)(
        work.FD, work.FMx, work.FMy, work.FMz, work.FE,
        states.left, states.right, pass.flux_activity_c2, geom.face_area,
        geom.normal_x, geom.normal_y, geom.normal_z,
        geom.face_vx, geom.face_vy, geom.face_vz,
        Float64(GAMMA), PowerFoam._solver_code(riemann), 1e-12;
        ndrange = nf)
    PowerFoam._cell_update_3d_k!(be)(
        state.D, state.Mx, state.My, state.Mz, state.E,
        work.FD, work.FMx, work.FMy, work.FMz, work.FE,
        geom.volume, target_volume, geom.cell_face_offsets,
        geom.cell_faces, geom.cell_face_signs, Float64(0.5 * pass.dt);
        ndrange = length(snapshot.ids))
    KernelAbstractions.synchronize(be)
    return _state_to_conserved(state, target_volume)
end

function _aligned_maxdiff(ids_a, a, ids_b, b)
    pos = Dict{Int64,Int}()
    for i in eachindex(ids_a)
        pos[Int64(ids_a[i])] = i
    end
    diffs = zeros(Float64, size(a, 2))
    for j in eachindex(ids_b)
        i = pos[Int64(ids_b[j])]
        @views diffs .= max.(diffs, abs.(a[i, :] .- b[j, :]))
    end
    return Tuple(diffs)
end

function _align_by_id(ids_source, ids_target, values_target)
    pos = Dict{Int64,Int}()
    for i in eachindex(ids_target)
        pos[Int64(ids_target[i])] = i
    end
    if values_target isa AbstractVector
        out = similar(values_target, length(ids_source))
        for i in eachindex(ids_source)
            out[i] = values_target[pos[Int64(ids_source[i])]]
        end
        return out
    end
    out = similar(values_target, length(ids_source), size(values_target, 2))
    for i in eachindex(ids_source)
        @views out[i, :] .= values_target[pos[Int64(ids_source[i])], :]
    end
    return out
end

function _id_set_delta(ids_a, ids_b)
    a = Set(Int64.(ids_a))
    b = Set(Int64.(ids_b))
    return length(setdiff(a, b)) + length(setdiff(b, a))
end

function _trace_pass_stats(trace, pass_index)
    idx = findall(i -> trace.active[i] && trace.pass_index[i] == pass_index,
                  eachindex(trace.c1))
    local_local = count(i -> trace.c1[i] > 0 && trace.c2[i] > 0, idx)
    one_sided = count(i -> xor(trace.c1[i] > 0, trace.c2[i] > 0), idx)
    if hasproperty(trace, :update_c1) && hasproperty(trace, :update_c2)
        update_one_sided = count(i -> xor(trace.update_c1[i] > 0,
                                         trace.update_c2[i] > 0), idx)
        update_none = count(i -> trace.update_c1[i] <= 0 &&
                                 trace.update_c2[i] <= 0, idx)
    else
        update_one_sided = one_sided
        update_none = 0
    end
    return (; pass_index, active = length(idx), local_local, one_sided,
            update_one_sided, update_none,
            face_dt_min = isempty(idx) ? NaN : minimum(trace.face_dt[idx]),
            face_dt_max = isempty(idx) ? NaN : maximum(trace.face_dt[idx]))
end

function _write_replay_report(path; status, dt_arepo = NaN, step_status = nothing,
                              trace = nothing, snapshots = nothing,
                              pass_stats = nothing, pass_gaps = nothing,
                              final_gap = nothing, pf_pass_gaps = nothing,
                              pf_final_gap = nothing, pred_pass_gaps = nothing,
                              pred_final_gap = nothing,
                              geometry_stats = nothing)
    open(path, "w") do io
        println(io, "# AREPO Trace Replay Gate")
        println(io)
        println(io, "This gate replays AREPO's exported lab-frame face fluxes")
        println(io, "from the per-pass pre-flux cell snapshots captured inside")
        println(io, "`compute_interface_fluxes()`.  It is a diagnostic bridge")
        println(io, "between per-face predictor/flux parity and full one-step")
        println(io, "final-field parity.")
        println(io)
        @printf(io, "- AREPO library: `%s`\n", ArepoLib.libpath())
        @printf(io, "- N: %d^3\n", N)
        @printf(io, "- Riemann solver: %s\n", RIEMANN)
        @printf(io, "- replay rows: %s\n", string(REPLAY_ROWS))
        @printf(io, "- replay geometry: %s\n", string(REPLAY_GEOMETRY))
        @printf(io, "- replay update targets: %s\n",
                string(REPLAY_UPDATE_TARGETS))
        @printf(io, "- replay native dt source: %s\n",
                string(REPLAY_NATIVE_DT_SOURCE))
        if REPLAY_GEOMETRY == :native
            @printf(io, "- native search radius: %d\n", REPLAY_NATIVE_RADIUS)
            @printf(io, "- native min face/surface fraction: %.12g\n",
                    REPLAY_NATIVE_MIN_FACE_SURFACE_FRACTION)
            @printf(io, "- replay face velocity: %s\n",
                    string(REPLAY_FACE_VELOCITY))
        end
        @printf(io, "- AREPO step status: %s\n", string(step_status))
        @printf(io, "- AREPO step dt: %.12g\n", dt_arepo)
        @printf(io, "- status: %s\n", status)
        println(io)
        if trace === nothing || snapshots === nothing
            println(io, "The active ArepoLib bridge does not expose both")
            println(io, "`get_hydro_face_traces_3d` and")
            println(io, "`get_hydro_preflux_states_3d`.")
            return
        end
        @printf(io, "- trace rows: %d\n", length(trace.c1))
        @printf(io, "- pre-flux snapshots: %d\n", length(snapshots))
        println(io)
        println(io, "## Face Coverage")
        println(io)
        println(io, "| pass | active faces | local-local | one-sided geom | one-sided update | no local update | face_dt min | face_dt max |")
        println(io, "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for s in pass_stats
            @printf(io, "| %d | %d | %d | %d | %d | %d | %.12g | %.12g |\n",
                    s.pass_index, s.active, s.local_local, s.one_sided,
                    s.update_one_sided, s.update_none, s.face_dt_min,
                    s.face_dt_max)
        end
        if geometry_stats !== nothing && !isempty(geometry_stats)
            println(io)
            println(io, "## Native Geometry Diagnostics")
            println(io)
            println(io, "| pass | max face velocity diff | local-local max | one-sided geom max | update-target mismatches |")
            println(io, "| ---: | ---: | ---: | ---: | ---: |")
            for s in geometry_stats
                @printf(io, "| %d | %.12g | %.12g | %.12g | %d |\n", s.pass_index,
                        s.max_face_velocity_diff,
                        s.max_local_face_velocity_diff,
                        s.max_one_sided_face_velocity_diff,
                        s.update_target_mismatches)
            end
        end
        println(io)
        println(io, "## Replay Max Absolute Difference")
        println(io)
        println(io, "| comparison | mass | mx | my | mz | energy | id set delta |")
        println(io, "| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for g in pass_gaps
            @printf(io, "| %s | %.12g | %.12g | %.12g | %.12g | %.12g | %d |\n",
                    g.label, g.conserved..., g.id_delta)
        end
        if final_gap !== nothing
            @printf(io, "| %s | %.12g | %.12g | %.12g | %.12g | %.12g | %d |\n",
                    final_gap.label, final_gap.conserved..., final_gap.id_delta)
        end
        if pf_pass_gaps !== nothing || pf_final_gap !== nothing
            println(io)
            println(io, "## PowerFoam Kernel Replay")
            println(io)
            println(io, "This section runs PowerFoam's KA face-flux kernel and")
            println(io, "cell-update kernel using AREPO's traced face states and")
            println(io, "geometric face endpoints with update-target CSR/activity.")
            println(io)
            println(io, "| comparison | mass | mx | my | mz | energy | id set delta |")
            println(io, "| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
            for g in pf_pass_gaps
                @printf(io, "| %s | %.12g | %.12g | %.12g | %.12g | %.12g | %d |\n",
                        g.label, g.conserved..., g.id_delta)
            end
            if pf_final_gap !== nothing
                @printf(io, "| %s | %.12g | %.12g | %.12g | %.12g | %.12g | %d |\n",
                        pf_final_gap.label, pf_final_gap.conserved...,
                        pf_final_gap.id_delta)
            end
        end
        if pred_pass_gaps !== nothing || pred_final_gap !== nothing
            println(io)
            println(io, "## PowerFoam Predictor Replay")
            println(io)
            println(io, "This section recomputes PowerFoam face states, moving-face")
            println(io, "fluxes, and cell updates from AREPO's per-pass pre-flux")
            println(io, "snapshots, gradients, and face geometry. It does not consume")
            println(io, "AREPO's traced face states or traced fluxes.")
            println(io)
            println(io, "| comparison | mass | mx | my | mz | energy | id set delta |")
            println(io, "| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
            for g in pred_pass_gaps
                @printf(io, "| %s | %.12g | %.12g | %.12g | %.12g | %.12g | %d |\n",
                        g.label, g.conserved..., g.id_delta)
            end
            if pred_final_gap !== nothing
                @printf(io, "| %s | %.12g | %.12g | %.12g | %.12g | %.12g | %d |\n",
                        pred_final_gap.label, pred_final_gap.conserved...,
                        pred_final_gap.id_delta)
            end
        end
        println(io)
        println(io, "## Primitive Difference")
        println(io)
        println(io, "| comparison | rho | vx | vy | vz | pressure |")
        println(io, "| --- | ---: | ---: | ---: | ---: | ---: |")
        for g in pass_gaps
            @printf(io, "| %s | %.12g | %.12g | %.12g | %.12g | %.12g |\n",
                    g.label, g.primitive...)
        end
        if final_gap !== nothing
            @printf(io, "| %s | %.12g | %.12g | %.12g | %.12g | %.12g |\n",
                    final_gap.label, final_gap.primitive...)
        end
    end
end

function main_replay()
    mkpath(REPLAY_OUTDIR)
    report = joinpath(REPLAY_OUTDIR, "README.md")
    if !_replay_bridge_available()
        _write_replay_report(report; status = "skipped: missing trace/pre-flux bridge")
        @printf("wrote %s\n", report)
        @printf("skipped: trace or pre-flux bridge is unavailable\n")
        return
    end
    dir = stage_arepo_case(N; riemann = RIEMANN)
    exported = arepo_initial_export(dir)
    try
        t0 = ArepoLib.sim_time(exported.h)
        step_status = ArepoLib.run_step!(exported.h)
        dt_arepo = ArepoLib.sim_time(exported.h) - t0
        trace = ArepoLib.get_hydro_face_traces_3d(exported.h)
        snapshots = ArepoLib.get_hydro_preflux_states_3d(exported.h)
        final = _conserved_after_arepo_step(exported.h, exported.ng)
        pass_stats = [_trace_pass_stats(trace, s.pass_index) for s in snapshots]
        geometry_stats = (REPLAY_GEOMETRY == :native || REPLAY_ROWS == :native) ?
                         [(_replay_pass_geometry(s, trace, s.pass_index)).geometry_stats
                          for s in snapshots] :
                         nothing
        pass_gaps = NamedTuple[]
        pf_pass_gaps = NamedTuple[]
        pred_pass_gaps = NamedTuple[]
        post_by_pass = Dict{Int,Matrix{Float64}}()
        pf_post_by_pass = Dict{Int,Matrix{Float64}}()
        pred_post_by_pass = Dict{Int,Matrix{Float64}}()
        for (i, s) in pairs(snapshots)
            if i < length(snapshots)
                next_s = snapshots[i + 1]
                target_volume = _align_by_id(s.ids, next_s.ids, next_s.volume)
                pred_post = _powerfoam_predictor_kernel_pass(s, trace,
                                                             s.pass_index,
                                                             target_volume)
                pred_post_by_pass[s.pass_index] = pred_post
                if REPLAY_ROWS == :trace
                    post = _apply_trace_pass_conserved(s, trace, s.pass_index)
                    post_by_pass[s.pass_index] = post
                    pf_post = _powerfoam_trace_kernel_pass(s, trace, s.pass_index,
                                                           target_volume)
                    pf_post_by_pass[s.pass_index] = pf_post
                    prim_post = _primitive_from_conserved(post, target_volume)
                    push!(pass_gaps,
                          (; label = @sprintf("pass %d replay vs pass %d pre-flux",
                                               s.pass_index, next_s.pass_index),
                             conserved = _aligned_maxdiff(s.ids, post, next_s.ids,
                                                           next_s.conserved),
                             primitive = _aligned_maxdiff(s.ids, prim_post,
                                                          next_s.ids,
                                                          next_s.primitive),
                             id_delta = _id_set_delta(s.ids, next_s.ids)))
                    push!(pf_pass_gaps,
                          (; label = @sprintf("PF pass %d kernel replay vs pass %d pre-flux",
                                               s.pass_index, next_s.pass_index),
                             conserved = _aligned_maxdiff(s.ids, pf_post,
                                                           next_s.ids,
                                                           next_s.conserved),
                             id_delta = _id_set_delta(s.ids, next_s.ids)))
                end
                push!(pred_pass_gaps,
                      (; label = @sprintf("PF pass %d predictor replay vs pass %d pre-flux",
                                           s.pass_index, next_s.pass_index),
                         conserved = _aligned_maxdiff(s.ids, pred_post,
                                                       next_s.ids,
                                                       next_s.conserved),
                         id_delta = _id_set_delta(s.ids, next_s.ids)))
            end
        end
        last_s = snapshots[end]
        final_volume = _align_by_id(last_s.ids, final.ids, final.volume)
        pred_last_post = _powerfoam_predictor_kernel_pass(last_s, trace,
                                                         last_s.pass_index,
                                                         final_volume)
        if REPLAY_ROWS == :trace
            last_post = post_by_pass[last_s.pass_index]
            pf_last_post = _powerfoam_trace_kernel_pass(last_s, trace,
                                                       last_s.pass_index,
                                                       final_volume)
            last_prim = _primitive_from_conserved(last_post, final_volume)
            final_gap = (; label = @sprintf("pass %d replay vs final AREPO",
                                            last_s.pass_index),
                         conserved = _aligned_maxdiff(last_s.ids, last_post,
                                                       final.ids, final.conserved),
                         primitive = _aligned_maxdiff(last_s.ids, last_prim,
                                                      final.ids, final.primitive),
                         id_delta = _id_set_delta(last_s.ids, final.ids))
            pf_final_gap = (; label = @sprintf("PF pass %d kernel replay vs final AREPO",
                                               last_s.pass_index),
                            conserved = _aligned_maxdiff(last_s.ids,
                                                         pf_last_post,
                                                         final.ids,
                                                         final.conserved),
                            id_delta = _id_set_delta(last_s.ids, final.ids))
        else
            final_gap = nothing
            pf_final_gap = nothing
            pf_pass_gaps = nothing
        end
        pred_final_gap = (; label = @sprintf("PF pass %d predictor replay vs final AREPO",
                                             last_s.pass_index),
                          conserved = _aligned_maxdiff(last_s.ids,
                                                       pred_last_post,
                                                       final.ids,
                                                      final.conserved),
                          id_delta = _id_set_delta(last_s.ids, final.ids))
        max_gap = final_gap === nothing ? 0.0 :
                  maximum(vcat([collect(g.conserved) for g in pass_gaps]...,
                                collect(final_gap.conserved)))
        max_pf_gap = pf_final_gap === nothing ? 0.0 :
                     maximum(vcat([collect(g.conserved) for g in pf_pass_gaps]...,
                                   collect(pf_final_gap.conserved)))
        max_pred_gap = maximum(vcat([collect(g.conserved) for g in pred_pass_gaps]...,
                                    collect(pred_final_gap.conserved)))
        update_mismatches = geometry_stats === nothing ? 0 :
                            sum(s.update_target_mismatches for s in geometry_stats)
        status = max(max_gap, max_pf_gap, max_pred_gap) <= REPLAY_TOL &&
                 update_mismatches == 0 ? "passed" : "diagnostic"
        _write_replay_report(report; status, dt_arepo, step_status, trace,
                             snapshots, pass_stats, pass_gaps, final_gap,
                             pf_pass_gaps, pf_final_gap, pred_pass_gaps,
                             pred_final_gap, geometry_stats)
        @printf("wrote %s\n", report)
        @printf("trace replay %s: snapshots=%d rows=%d max conserved gap=%.6g pf=%.6g pred=%.6g update_mismatch=%d\n",
                status, length(snapshots), length(trace.c1), max_gap,
                max_pf_gap, max_pred_gap, update_mismatches)
    finally
        ArepoLib.finalize(exported.h)
    end
end

main_replay()
