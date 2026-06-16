using Printf
using LinearAlgebra
using KernelAbstractions
using PowerFoam

trace_arg(i, default, T) = length(ARGS) >= i ? parse(T, ARGS[i]) : default

const TRACE_OUTBASE = joinpath(@__DIR__, "out", "arepo_face_trace_gate_3d")
const TRACE_N = trace_arg(1, 12, Int)
const TRACE_DT = trace_arg(2, 0.001, Float64)
const TRACE_RIEMANN = Symbol(length(ARGS) >= 3 ? ARGS[3] : "hll")
const TRACE_NSTEPS = trace_arg(4, 1, Int)
const TRACE_PASS_INDEX = trace_arg(5, parse(Int, get(ENV, "POWERFOAM_TRACE_PASS", "1")), Int)
const TRACE_GAMMA = 5 / 3
const TRACE_TOL = 1e-6
const TRACE_ALL_PASSES = get(ENV, "POWERFOAM_TRACE_ALL_PASSES", "true") == "true"
const TRACE_RUN_TAG_BASE = TRACE_NSTEPS == 1 ?
                           @sprintf("N%d_dt%.3g_%s", TRACE_N, TRACE_DT, TRACE_RIEMANN) :
                           @sprintf("N%d_dt%.3g_n%d_%s", TRACE_N, TRACE_DT,
                                    TRACE_NSTEPS, TRACE_RIEMANN)
const TRACE_RUN_TAG = TRACE_ALL_PASSES ? @sprintf("%s_allpasses", TRACE_RUN_TAG_BASE) :
                      TRACE_PASS_INDEX == 1 ? TRACE_RUN_TAG_BASE :
                      @sprintf("%s_pass%d", TRACE_RUN_TAG_BASE, TRACE_PASS_INDEX)
const TRACE_OUTDIR = joinpath(TRACE_OUTBASE, replace(TRACE_RUN_TAG, "." => "p"))
const TRACE_AREPOLIB_IMPORT_ERROR = Ref{Any}(nothing)

try
    @eval import ArepoLib
catch err
    TRACE_AREPOLIB_IMPORT_ERROR[] = err
end

function _trace_bridge_available()
    return isdefined(Main, :ArepoLib) &&
           isdefined(ArepoLib, :get_hydro_face_traces_3d)
end

function _preflux_bridge_available()
    return isdefined(Main, :ArepoLib) &&
           isdefined(ArepoLib, :get_hydro_preflux_states_3d)
end

function _timebin_bridge_available()
    return isdefined(Main, :ArepoLib) &&
           isdefined(ArepoLib, :get_hydro_timebins)
end

function _trace_arepo_libpath()
    if isdefined(Main, :ArepoLib)
        return ArepoLib.libpath()
    end
    return "unavailable: ArepoLib package is not in the active Julia environment"
end

function _matrix_from_packed(v, nf)
    a = Array(v)
    return hcat(a[1:nf], a[(nf + 1):(2nf)], a[(2nf + 1):(3nf)],
                a[(3nf + 1):(4nf)], a[(4nf + 1):(5nf)])
end

function _maxdiff_by_column(a, b)
    return ntuple(i -> maximum(abs.(a[:, i] .- b[:, i])), size(a, 2))
end

function _trace_local_solver_flux(trace, idx; gamma = TRACE_GAMMA,
                                  riemann = TRACE_RIEMANN)
    out = zeros(Float64, length(idx), 5)
    gm1 = gamma - 1
    for (row, i) in pairs(idx)
        ρl, ux_l, uy_l, uz_l, pl = trace.state_face_l[i, :]
        ρr, ux_r, uy_r, uz_r, pr = trace.state_face_r[i, :]
        n = trace.normal[i, :]
        m = trace.tangent_m[i, :]
        p = trace.tangent_p[i, :]
        vl = (dot((ux_l, uy_l, uz_l), n),
              dot((ux_l, uy_l, uz_l), m),
              dot((ux_l, uy_l, uz_l), p))
        vr = (dot((ux_r, uy_r, uz_r), n),
              dot((ux_r, uy_r, uz_r), m),
              dot((ux_r, uy_r, uz_r), p))
        El = pl / gm1 + 0.5 * ρl * (vl[1]^2 + vl[2]^2 + vl[3]^2)
        Er = pr / gm1 + 0.5 * ρr * (vr[1]^2 + vr[2]^2 + vr[3]^2)
        FL = (ρl * vl[1],
              ρl * vl[1]^2 + pl,
              ρl * vl[1] * vl[2],
              ρl * vl[1] * vl[3],
              (El + pl) * vl[1])
        FR = (ρr * vr[1],
              ρr * vr[1]^2 + pr,
              ρr * vr[1] * vr[2],
              ρr * vr[1] * vr[3],
              (Er + pr) * vr[1])
        cl = sqrt(gamma * pl / ρl)
        cr = sqrt(gamma * pr / ρr)
        sl = min(vl[1] - cl, vr[1] - cr)
        sr = max(vl[1] + cl, vr[1] + cr)
        UL = (ρl, ρl * vl[1], ρl * vl[2], ρl * vl[3], El)
        UR = (ρr, ρr * vr[1], ρr * vr[2], ρr * vr[3], Er)
        if riemann == :llf
            a = max(abs(vl[1]) + cl, abs(vr[1]) + cr)
            f = ntuple(k -> 0.5 * (FL[k] + FR[k]) - 0.5 * a * (UR[k] - UL[k]), 5)
        elseif sl >= 0
            f = FL
        elseif sr <= 0
            f = FR
        else
            denom = sr - sl
            f = ntuple(k -> (sr * FL[k] - sl * FR[k] +
                             sr * sl * (UR[k] - UL[k])) / denom, 5)
        end
        for k in 1:5
            out[row, k] = f[k]
        end
    end
    return out
end

function _active_indices(trace)
    if hasproperty(trace, :active)
        idx = findall(Bool.(trace.active))
        return isempty(idx) ? collect(eachindex(trace.c1)) : idx
    end
    return collect(eachindex(trace.c1))
end

function _trace_indices(trace; pass_index = 1)
    idx_all = _active_indices(trace)
    idx_pass = hasproperty(trace, :pass_index) ?
               [i for i in idx_all if trace.pass_index[i] == pass_index] :
               idx_all
    return [i for i in idx_pass if trace.c1[i] > 0 && trace.c2[i] > 0]
end

function _trace_update_indices(trace; pass_index = 1)
    idx_all = _active_indices(trace)
    idx_pass = hasproperty(trace, :pass_index) ?
               [i for i in idx_all if trace.pass_index[i] == pass_index] :
               idx_all
    if hasproperty(trace, :update_c1) && hasproperty(trace, :update_c2)
        return [i for i in idx_pass if trace.update_c1[i] > 0 && trace.update_c2[i] > 0]
    end
    return [i for i in idx_pass if trace.c1[i] > 0 && trace.c2[i] > 0]
end

function _trace_geom_from_arepo_trace(exported, trace, be; pass_index = 1)
    idx = _trace_indices(trace; pass_index)
    c1 = Int32.(trace.c1[idx])
    c2 = Int32.(trace.c2[idx])
    offsets, faces, signs = PowerFoam._cell_face_csr(exported.ng, c1, c2, Int32)
    geom_host = ArepoMeshArrays3D(c1, c2, offsets, faces, signs,
                                  Float64.(exported.vol),
                                  Float64.(trace.area[idx]),
                                  Float64.(trace.normal[idx, 1]),
                                  Float64.(trace.normal[idx, 2]),
                                  Float64.(trace.normal[idx, 3]),
                                  Float64.(trace.vel_face[idx, 1]),
                                  Float64.(trace.vel_face[idx, 2]),
                                  Float64.(trace.vel_face[idx, 3]))
    return to_backend(be, geom_host; T = Float32), trace.face_center[idx, :], idx
end

function _state_from_preflux_snapshot(snapshot)
    return EulerState3D(snapshot.conserved[:, 1] ./ snapshot.volume,
                        snapshot.conserved[:, 2] ./ snapshot.volume,
                        snapshot.conserved[:, 3] ./ snapshot.volume,
                        snapshot.conserved[:, 4] ./ snapshot.volume,
                        snapshot.conserved[:, 5] ./ snapshot.volume)
end

function _primitive_from_snapshot(snapshot)
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

function _trace_geom_from_snapshot(snapshot, trace, be; pass_index = 1,
                                   T = Float64)
    idx = _trace_update_indices(trace; pass_index)
    geom_c1 = Int32.(trace.c1[idx])
    geom_c2 = Int32.(trace.c2[idx])
    update_c1 = hasproperty(trace, :update_c1) ? Int32.(trace.update_c1[idx]) : geom_c1
    update_c2 = hasproperty(trace, :update_c2) ? Int32.(trace.update_c2[idx]) : geom_c2
    offsets, faces, signs = PowerFoam._cell_face_csr(length(snapshot.ids), update_c1,
                                                     update_c2, Int32)
    geom_host = ArepoMeshArrays3D(update_c1, update_c2, offsets, faces, signs,
                                  T.(snapshot.volume),
                                  T.(trace.area[idx]),
                                  T.(trace.normal[idx, 1]),
                                  T.(trace.normal[idx, 2]),
                                  T.(trace.normal[idx, 3]),
                                  T.(trace.vel_face[idx, 1]),
                                  T.(trace.vel_face[idx, 2]),
                                  T.(trace.vel_face[idx, 3]))
    return to_backend(be, geom_host; T), trace.face_center[idx, :], idx,
           (; geom_c1, geom_c2, update_c1, update_c2)
end

function _powerfoam_face_trace_snapshot(snapshot, trace, be; pass_index,
                                        riemann = TRACE_RIEMANN)
    T = Float64
    state = _state_from_preflux_snapshot(snapshot)
    geom, face_center, trace_idx, endpoints =
        _trace_geom_from_snapshot(snapshot, trace, be; pass_index, T)
    bstate = to_backend(be, state; T)
    prim = _primitive_from_snapshot(snapshot)
    bprim = PrimitiveState3D(PowerFoam._backend_copy(be, prim.rho, T),
                             PowerFoam._backend_copy(be, prim.vx, T),
                             PowerFoam._backend_copy(be, prim.vy, T),
                             PowerFoam._backend_copy(be, prim.vz, T),
                             PowerFoam._backend_copy(be, prim.pressure, T))
    gradients = _gradients_from_snapshot(snapshot, be; T)
    states = face_prediction_work_3d(geom)
    nf = length(geom.c1)
    dt_host = zeros(T, length(snapshot.ids))
    for i in trace_idx
        if hasproperty(trace, :update_c1) && hasproperty(trace, :state_dt_l)
            trace.update_c1[i] > 0 && (dt_host[trace.update_c1[i]] = T(trace.state_dt_l[i]))
            trace.update_c2[i] > 0 && (dt_host[trace.update_c2[i]] = T(trace.state_dt_r[i]))
        elseif hasproperty(trace, :state_dt_l)
            trace.c1[i] > 0 && (dt_host[trace.c1[i]] = T(trace.state_dt_l[i]))
            trace.c2[i] > 0 && (dt_host[trace.c2[i]] = T(trace.state_dt_r[i]))
        end
    end
    half_dt = PowerFoam._backend_copy(be, dt_host, T)
    predict_face_states_3d!(states, geom, gradients, bprim, snapshot.center,
                            face_center; dt_extrapolation = half_dt,
                            box_size = 1.0, gamma = TRACE_GAMMA)
    work = hydro_work_3d(bstate, geom)
    flux_activity_c2 = face_update_activity_3d(endpoints.update_c1, endpoints.update_c2)
    PowerFoam._face_flux_from_predicted_3d_k!(be)(
        work.FD, work.FMx, work.FMy, work.FMz, work.FE,
        states.left, states.right, PowerFoam._backend_copy(be, flux_activity_c2, Int32),
        geom.face_area, geom.normal_x, geom.normal_y, geom.normal_z,
        geom.face_vx, geom.face_vy, geom.face_vz,
        T(TRACE_GAMMA), PowerFoam._solver_code(riemann), T(1e-12);
        ndrange = nf)
    KernelAbstractions.synchronize(be)
    return (;
        c1 = Int.(endpoints.update_c1),
        c2 = Int.(endpoints.update_c2),
        geom_c1 = Int.(endpoints.geom_c1),
        geom_c2 = Int.(endpoints.geom_c2),
        trace_idx,
        left = _matrix_from_packed(states.left, nf),
        right = _matrix_from_packed(states.right, nf),
        flux_area = hcat(Array(work.FD), Array(work.FMx), Array(work.FMy),
                         Array(work.FMz), Array(work.FE)))
end

function _powerfoam_face_trace(exported, trace, be; riemann = TRACE_RIEMANN)
    _, state, _, _ = Base.invokelatest(getfield(Main, :make_state_and_geom), exported, be)
    geom, face_center, trace_idx = _trace_geom_from_arepo_trace(exported, trace, be;
                                                                pass_index = TRACE_PASS_INDEX)
    prim = primitive_work_3d(state)
    conserved_to_primitive_3d!(prim, state; gamma = TRACE_GAMMA)
    gradients = Base.invokelatest(getfield(Main, :hydro_gradients_from_arepo),
                                  exported.cgrad, be; T = Float32)
    states = face_prediction_work_3d(geom)
    nf = length(geom.c1)
    half_dt = if hasproperty(trace, :state_dt_l) && hasproperty(trace, :state_dt_r)
        dt_host = fill(Float32(0), exported.ng)
        for i in trace_idx
            trace.c1[i] > 0 && (dt_host[trace.c1[i]] = Float32(trace.state_dt_l[i]))
            trace.c2[i] > 0 && (dt_host[trace.c2[i]] = Float32(trace.state_dt_r[i]))
        end
        PowerFoam._backend_copy(be, dt_host, Float32)
    elseif _timebin_bridge_available() && hasproperty(trace, :cell_half_dt)
        PowerFoam._backend_copy(be, Array(trace.cell_half_dt), Float32)
    else
        hydro_dt = arepo_hydro_dt_3d(exported.vol, exported.pressure, exported.rho;
                                     gamma = TRACE_GAMMA, courant = 0.3,
                                     max_dt = 0.05, min_dt = 1e-6)
        PowerFoam._backend_copy(be, fill(Float32(0.5 * minimum(arepo_system_step_3d(hydro_dt))), exported.ng),
                                Float32)
    end
    predict_face_states_3d!(states, geom, gradients, prim, exported.center,
                            face_center; dt_extrapolation = half_dt,
                            box_size = exported.box, gamma = TRACE_GAMMA)
    work = hydro_work_3d(state, geom)
    PowerFoam._face_flux_from_predicted_3d_k!(be)(
        work.FD, work.FMx, work.FMy, work.FMz, work.FE,
        states.left, states.right, geom.c2, geom.face_area,
        geom.normal_x, geom.normal_y, geom.normal_z,
        geom.face_vx, geom.face_vy, geom.face_vz,
        Float32(TRACE_GAMMA), PowerFoam._solver_code(riemann), Float32(1e-12);
        ndrange = nf)
    KernelAbstractions.synchronize(be)
    return (;
        c1 = Int.(Array(geom.c1)),
        c2 = Int.(Array(geom.c2)),
        left = _matrix_from_packed(states.left, nf),
        right = _matrix_from_packed(states.right, nf),
        flux_area = hcat(Array(work.FD), Array(work.FMx), Array(work.FMy),
                         Array(work.FMz), Array(work.FE)))
end

function _face_lookup(pf)
    lookup = Dict{Tuple{Int,Int},Int}()
    for f in eachindex(pf.c1)
        lookup[(pf.c1[f], pf.c2[f])] = f
    end
    return lookup
end

function _trace_stats(exported, trace, pf; pass_index = 1)
    idx = _trace_indices(trace; pass_index)
    if length(pf.c1) == length(idx) &&
       all(pf.c1[k] == trace.c1[idx[k]] && pf.c2[k] == trace.c2[idx[k]]
           for k in eachindex(idx))
        pf_idx = collect(eachindex(idx))
        trace_idx = idx
        missing = 0
        reversed = 0
    else
        lookup = _face_lookup(pf)
        pf_idx = Int[]
        trace_idx = Int[]
        missing = 0
        reversed = 0
        for i in idx
            key = (trace.c1[i], trace.c2[i])
            if haskey(lookup, key)
                push!(pf_idx, lookup[key])
                push!(trace_idx, i)
            elseif haskey(lookup, (trace.c2[i], trace.c1[i]))
                reversed += 1
                missing += 1
            else
                missing += 1
            end
        end
    end
    arepo_flux_area = trace.flux_lab[trace_idx, :] .* trace.area[trace_idx]
    local_calc = hasproperty(trace, :flux_local) ?
                 _trace_local_solver_flux(trace, trace_idx) :
                 zeros(Float64, 0, 5)
    local_flux = hasproperty(trace, :flux_local) ?
                 _maxdiff_by_column(local_calc, trace.flux_local[trace_idx, :]) :
                 ntuple(_ -> NaN, 5)
    local_calc_max = hasproperty(trace, :flux_local) ?
                     ntuple(i -> maximum(abs.(local_calc[:, i])), 5) :
                     ntuple(_ -> NaN, 5)
    local_trace_max = hasproperty(trace, :flux_local) ?
                      ntuple(i -> maximum(abs.(trace.flux_local[trace_idx, i])), 5) :
                      ntuple(_ -> NaN, 5)
    return (;
        pass_index,
        faces_arepo = length(trace.c1),
        faces_powerfoam = length(pf.c1),
        active_faces = length(idx),
        matched_faces = length(trace_idx),
        missing_faces = missing,
        reversed_faces = reversed,
        c1_mismatches = missing,
        c2_mismatches = missing,
        left = isempty(trace_idx) ? ntuple(_ -> Inf, 5) :
               _maxdiff_by_column(pf.left[pf_idx, :], trace.state_face_l[trace_idx, :]),
        right = isempty(trace_idx) ? ntuple(_ -> Inf, 5) :
                _maxdiff_by_column(pf.right[pf_idx, :], trace.state_face_r[trace_idx, :]),
        flux = isempty(trace_idx) ? ntuple(_ -> Inf, 5) :
               _maxdiff_by_column(pf.flux_area[pf_idx, :], arepo_flux_area),
        local_flux,
        local_calc_max,
        local_trace_max)
end

function _trace_stats_for_indices(trace, pf, idx; pass_index = 1)
    pf_idx = collect(eachindex(idx))
    trace_idx = idx
    missing = 0
    reversed = 0
    geom_one_sided = count(i -> xor(trace.c1[i] > 0, trace.c2[i] > 0), idx)
    update_one_sided = hasproperty(trace, :update_c1) ?
                       count(i -> xor(trace.update_c1[i] > 0, trace.update_c2[i] > 0), idx) :
                       geom_one_sided
    arepo_flux_area = trace.flux_lab[trace_idx, :] .* trace.area[trace_idx]
    local_calc = hasproperty(trace, :flux_local) ?
                 _trace_local_solver_flux(trace, trace_idx) :
                 zeros(Float64, 0, 5)
    local_flux = hasproperty(trace, :flux_local) ?
                 _maxdiff_by_column(local_calc, trace.flux_local[trace_idx, :]) :
                 ntuple(_ -> NaN, 5)
    local_calc_max = hasproperty(trace, :flux_local) ?
                     ntuple(i -> maximum(abs.(local_calc[:, i])), 5) :
                     ntuple(_ -> NaN, 5)
    local_trace_max = hasproperty(trace, :flux_local) ?
                      ntuple(i -> maximum(abs.(trace.flux_local[trace_idx, i])), 5) :
                      ntuple(_ -> NaN, 5)
    return (;
        pass_index,
        faces_arepo = length(trace.c1),
        faces_powerfoam = length(pf.c1),
        active_faces = length(idx),
        matched_faces = length(trace_idx),
        missing_faces = missing,
        reversed_faces = reversed,
        c1_mismatches = missing,
        c2_mismatches = missing,
        geom_one_sided,
        update_one_sided,
        left = isempty(trace_idx) ? ntuple(_ -> Inf, 5) :
               _maxdiff_by_column(pf.left[pf_idx, :], trace.state_face_l[trace_idx, :]),
        right = isempty(trace_idx) ? ntuple(_ -> Inf, 5) :
                _maxdiff_by_column(pf.right[pf_idx, :], trace.state_face_r[trace_idx, :]),
        flux = isempty(trace_idx) ? ntuple(_ -> Inf, 5) :
               _maxdiff_by_column(pf.flux_area[pf_idx, :], arepo_flux_area),
        local_flux,
        local_calc_max,
        local_trace_max)
end

function _all_pass_trace_stats(snapshots, trace, be)
    stats = NamedTuple[]
    for snapshot in snapshots
        pf = _powerfoam_face_trace_snapshot(snapshot, trace, be;
                                           pass_index = snapshot.pass_index)
        push!(stats, _trace_stats_for_indices(trace, pf, pf.trace_idx;
                                              pass_index = snapshot.pass_index))
    end
    return stats
end

function _max_tuple(stats, field)
    vals = [maximum(getproperty(s, field)) for s in stats]
    return isempty(vals) ? Inf : maximum(vals)
end

function _write_trace_report(path; status, stats = nothing, timebins = nothing)
    open(path, "w") do io
        println(io, "# AREPO Face Trace Gate")
        println(io)
        println(io, "This gate compares PowerFoam's 3-D reconstructed predictor and")
        println(io, "moving-face flux against AREPO's live per-face hydro trace export.")
        println(io)
        @printf(io, "- AREPO library: `%s`\n", _trace_arepo_libpath())
        @printf(io, "- N: %d^3\n", TRACE_N)
        @printf(io, "- Riemann solver: %s\n", TRACE_RIEMANN)
        @printf(io, "- status: %s\n", status)
        println(io)
        if stats === nothing
            if TRACE_AREPOLIB_IMPORT_ERROR[] !== nothing
                println(io, "The active Julia environment does not expose")
                println(io, "`ArepoLib`, so the face-trace bridge could not be queried.")
            else
                println(io, "The AREPO bridge does not yet expose")
                println(io, "`get_hydro_face_traces_3d`.")
            end
            println(io)
            println(io, "See")
            println(io, "`lib/PowerFoam/external_patches/arepo_bridge_face_trace_contract.md`.")
            return
        end
        stats_list = stats isa AbstractVector ? stats : [stats]
        println(io, "## Face Trace Parity")
        println(io)
        println(io, "| pass | AREPO faces | PF faces | active compared | matched | missing | one-sided geom | one-sided update |")
        println(io, "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for s in stats_list
            @printf(io, "| %d | %d | %d | %d | %d | %d | %d | %d |\n",
                    s.pass_index, s.faces_arepo, s.faces_powerfoam,
                    s.active_faces, s.matched_faces, s.missing_faces,
                    hasproperty(s, :geom_one_sided) ? s.geom_one_sided : 0,
                    hasproperty(s, :update_one_sided) ? s.update_one_sided : 0)
        end
        println(io)
        println(io, "## Max Absolute Difference By Column")
        println(io)
        println(io, "| pass | field | rho/mass | vx/mx | vy/my | vz/mz | pressure/energy |")
        println(io, "| ---: | --- | ---: | ---: | ---: | ---: | ---: |")
        for s in stats_list
            @printf(io, "| %d | left face state | %.12g | %.12g | %.12g | %.12g | %.12g |\n",
                    s.pass_index, s.left...)
            @printf(io, "| %d | right face state | %.12g | %.12g | %.12g | %.12g | %.12g |\n",
                    s.pass_index, s.right...)
            @printf(io, "| %d | lab flux times area | %.12g | %.12g | %.12g | %.12g | %.12g |\n",
                    s.pass_index, s.flux...)
            @printf(io, "| %d | traced local %s flux | %.12g | %.12g | %.12g | %.12g | %.12g |\n",
                    s.pass_index, TRACE_RIEMANN, s.local_flux...)
            @printf(io, "| %d | local %s calc max | %.12g | %.12g | %.12g | %.12g | %.12g |\n",
                    s.pass_index, TRACE_RIEMANN, s.local_calc_max...)
            @printf(io, "| %d | local %s trace max | %.12g | %.12g | %.12g | %.12g | %.12g |\n",
                    s.pass_index, TRACE_RIEMANN, s.local_trace_max...)
        end
        if timebins !== nothing
            println(io)
            println(io, "## Timebin Export")
            println(io)
            @printf(io, "- ti_current: %d\n", timebins.ti_current)
            @printf(io, "- timebase interval: %.12g\n", timebins.timebase_interval)
            @printf(io, "- active cells: %d\n", count(timebins.active))
            @printf(io, "- min/max hydro bin: %d / %d\n",
                    minimum(timebins.bins), maximum(timebins.bins))
        end
    end
end

function main_trace()
    mkpath(TRACE_OUTDIR)
    report = joinpath(TRACE_OUTDIR, "README.md")
    if !_trace_bridge_available()
        _write_trace_report(report; status = "skipped: missing AREPO face-trace bridge")
        @printf("wrote %s\n", report)
        @printf("skipped: ArepoLib.get_hydro_face_traces_3d is not available\n")
        return
    end
    include(joinpath(@__DIR__, "arepo_geometry_gate_3d.jl"))
    dir = Base.invokelatest(getfield(Main, :stage_arepo_case), TRACE_N;
                            riemann = TRACE_RIEMANN)
    exported = Base.invokelatest(getfield(Main, :arepo_initial_export), dir)
    try
        status_step = ArepoLib.run_step!(exported.h)
        trace = ArepoLib.get_hydro_face_traces_3d(exported.h)
        timebins = _timebin_bridge_available() ? ArepoLib.get_hydro_timebins(exported.h) : nothing
        if TRACE_ALL_PASSES
            _preflux_bridge_available() ||
                error("POWERFOAM_TRACE_ALL_PASSES requires ArepoLib.get_hydro_preflux_states_3d")
            snapshots = ArepoLib.get_hydro_preflux_states_3d(exported.h)
            stats = _all_pass_trace_stats(snapshots, trace, KernelAbstractions.CPU())
            status = all(s -> s.missing_faces == 0, stats) &&
                     _max_tuple(stats, :left) <= TRACE_TOL &&
                     _max_tuple(stats, :right) <= TRACE_TOL &&
                     _max_tuple(stats, :flux) <= TRACE_TOL ? "passed" : "failed"
        else
            pf = _powerfoam_face_trace(exported, trace, KernelAbstractions.CPU())
            stats = _trace_stats(exported, trace, pf; pass_index = TRACE_PASS_INDEX)
            status = stats.missing_faces == 0 &&
                     maximum(stats.left) <= TRACE_TOL &&
                     maximum(stats.right) <= TRACE_TOL &&
                     maximum(stats.flux) <= TRACE_TOL ? "passed" : "failed"
        end
        _write_trace_report(report; status, stats, timebins)
        @printf("wrote %s\n", report)
        @printf("AREPO step status=%s\n", status_step)
        if stats isa AbstractVector
            @printf("face trace %s: passes=%d matched=%d missing=%d maxleft=%.6g maxright=%.6g maxflux=%.6g\n",
                    status, length(stats), sum(s.matched_faces for s in stats),
                    sum(s.missing_faces for s in stats), _max_tuple(stats, :left),
                    _max_tuple(stats, :right), _max_tuple(stats, :flux))
        else
            @printf("face trace %s: matched=%d missing=%d left=%s right=%s flux=%s\n",
                    status, stats.matched_faces, stats.missing_faces,
                    stats.left, stats.right, stats.flux)
        end
    finally
        ArepoLib.finalize(exported.h)
    end
end

main_trace()
