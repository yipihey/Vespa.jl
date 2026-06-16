using Printf
using Statistics

include(joinpath(@__DIR__, "arepo_geometry_gate_3d.jl"))

const NATIVE_TRACE_OUTBASE = joinpath(@__DIR__, "out", "arepo_native_rebuild_trace_gate_3d")
const NATIVE_TRACE_ALGORITHM_NAME =
    get(ENV, "POWERFOAM_TESSELLATOR_ALGORITHM", "local_periodic_halfspace")
const NATIVE_TRACE_ALGORITHM = Symbol(NATIVE_TRACE_ALGORITHM_NAME)
const NATIVE_TRACE_TAG = NATIVE_TRACE_ALGORITHM == :local_periodic_halfspace ?
                         replace(RUN_TAG, "." => "p") :
                         string(replace(RUN_TAG, "." => "p"), "_",
                                NATIVE_TRACE_ALGORITHM_NAME)
const NATIVE_TRACE_OUTDIR = joinpath(NATIVE_TRACE_OUTBASE, NATIVE_TRACE_TAG)
const NATIVE_TRACE_RADIUS = parse(Int, get(ENV, "POWERFOAM_NATIVE_TRACE_RADIUS", "1"))
const NATIVE_TRACE_MIN_FACE_SURFACE_FRACTION =
    parse(Float64, get(ENV, "POWERFOAM_NATIVE_TRACE_MIN_FACE_SURFACE_FRACTION", "1e-5"))

function _native_trace_bridge_available()
    return isdefined(Main, :ArepoLib) &&
           isdefined(ArepoLib, :get_hydro_face_traces_3d) &&
           isdefined(ArepoLib, :get_hydro_preflux_states_3d)
end

function _pair_key(a, b)
    return a <= b ? (a, b) : (b, a)
end

function _row_set_from_trace(trace, pass_index)
    idx = findall(i -> trace.active[i] && trace.pass_index[i] == pass_index &&
                       hasproperty(trace, :update_c1) &&
                       trace.update_c1[i] > 0 && trace.update_c2[i] > 0,
                  eachindex(trace.c1))
    rows = Dict{Tuple{Int,Int},Int}()
    duplicates = 0
    for i in idx
        key = _pair_key(trace.update_c1[i], trace.update_c2[i])
        haskey(rows, key) && (duplicates += 1)
        rows[key] = i
    end
    return rows, duplicates, length(idx)
end

function _row_set_from_native(geom)
    rows = Dict{Tuple{Int,Int},Int}()
    duplicates = 0
    c1 = Int.(geom.c1)
    c2 = Int.(geom.c2)
    for f in eachindex(c1)
        c1[f] > 0 && c2[f] > 0 || continue
        key = _pair_key(c1[f], c2[f])
        haskey(rows, key) && (duplicates += 1)
        rows[key] = f
    end
    return rows, duplicates
end

function _surface_area_from_native(geom)
    n = length(geom.volume)
    surface = zeros(Float64, n)
    c1 = Int.(geom.c1)
    c2 = Int.(geom.c2)
    @inbounds for f in eachindex(c1)
        a = Float64(geom.face_area[f])
        c1[f] > 0 && (surface[c1[f]] += a)
        c2[f] > 0 && c2[f] <= n && (surface[c2[f]] += a)
    end
    return surface
end

function _native_extra_rows(native_rows, trace_rows, native)
    surface = _surface_area_from_native(native.geom)
    extra = setdiff(keys(native_rows), keys(trace_rows))
    rows = NamedTuple[]
    @inbounds for key in sort(collect(extra))
        f = native_rows[key]
        a = Float64(native.geom.face_area[f])
        surf = max(surface[key[1]], surface[key[2]])
        cut = 1e-5 * surf
        push!(rows, (; pair = key, face = f, area = a,
                     surface_max = surf, arepo_cut = cut,
                     below_arepo_cut = !(a > cut),
                     center = (native.face_center[f, 1],
                               native.face_center[f, 2],
                               native.face_center[f, 3])))
    end
    return rows
end

@inline function _nearest_periodic_delta(x, ref; box = 1.0)
    return x - ref - round((x - ref) / box) * box
end

function _face_diffs(snapshot, trace, native, pass_index)
    trace_rows, trace_dups, trace_active = _row_set_from_trace(trace, pass_index)
    native_rows, native_dups = _row_set_from_native(native.geom)
    common = intersect(keys(trace_rows), keys(native_rows))
    missing = setdiff(keys(trace_rows), keys(native_rows))
    extra = setdiff(keys(native_rows), keys(trace_rows))
    extra_rows = _native_extra_rows(native_rows, trace_rows, native)
    max_area = 0.0
    max_normal = 0.0
    max_center = 0.0
    extra_area_sum = 0.0
    extra_area_max = 0.0
    extra_self = 0
    extra_below_arepo_cut = count(r -> r.below_arepo_cut, extra_rows)
    @inbounds for key in common
        ti = trace_rows[key]
        nf = native_rows[key]
        area_diff = abs(native.geom.face_area[nf] - trace.area[ti])
        max_area = max(max_area, area_diff)
        n_native = (native.geom.normal_x[nf], native.geom.normal_y[nf],
                    native.geom.normal_z[nf])
        n_trace = (trace.normal[ti, 1], trace.normal[ti, 2], trace.normal[ti, 3])
        same = sqrt(sum((n_native[k] - n_trace[k])^2 for k in 1:3))
        flip = sqrt(sum((n_native[k] + n_trace[k])^2 for k in 1:3))
        max_normal = max(max_normal, min(same, flip))
        center_diff = sqrt(sum(abs2(_nearest_periodic_delta(native.face_center[nf, k],
                                                            trace.face_center[ti, k]))
                               for k in 1:3))
        max_center = max(max_center, center_diff)
    end
    @inbounds for key in extra
        nf = native_rows[key]
        a = native.geom.face_area[nf]
        extra_area_sum += a
        extra_area_max = max(extra_area_max, a)
        key[1] == key[2] && (extra_self += 1)
    end
    vol_diff = maximum(abs.(native.geom.volume .- snapshot.volume))
    return (; pass_index,
            algorithm = hasproperty(native, :reference) ? native.reference.algorithm : :unknown,
            canonical_faces = hasproperty(native, :reference) ?
                              length(native.reference.canonical_face_order) :
                              length(native.geom.c1),
            trace_active,
            trace_pairs = length(trace_rows),
            native_pairs = length(native_rows),
            matched = length(common),
            missing = length(missing),
            extra = length(extra),
            trace_duplicates = trace_dups,
            native_duplicates = native_dups,
            max_area_diff = max_area,
            max_normal_diff = max_normal,
            max_center_diff = max_center,
            extra_area_sum,
            extra_area_max,
            extra_self,
            extra_below_arepo_cut,
            extra_rows,
            max_volume_diff = vol_diff)
end

function _write_native_trace_report(path; status, step_status = nothing,
                                    stats = nothing)
    open(path, "w") do io
        println(io, "# AREPO Native Rebuild Trace Gate")
        println(io)
        println(io, "This gate rebuilds PowerFoam's native local periodic 3-D")
        println(io, "Voronoi table from AREPO pre-flux generator positions and")
        println(io, "compares the resulting face pairs, face geometry, and cell")
        println(io, "volumes against AREPO's traced update-target face table.")
        println(io)
        @printf(io, "- AREPO library: `%s`\n", _native_trace_bridge_available() ? ArepoLib.libpath() : "unavailable")
        @printf(io, "- N: %d^3\n", N)
        @printf(io, "- Riemann solver: %s\n", RIEMANN)
        @printf(io, "- native search radius: %d\n", NATIVE_TRACE_RADIUS)
        @printf(io, "- native min face/surface fraction: %.12g\n",
                NATIVE_TRACE_MIN_FACE_SURFACE_FRACTION)
        @printf(io, "- AREPO step status: %s\n", string(step_status))
        @printf(io, "- status: %s\n", status)
        println(io)
        if stats === nothing
            println(io, "The active bridge does not expose both face traces and")
            println(io, "pre-flux snapshots.")
            return
        end
        println(io, "| pass | trace active | trace pairs | native pairs | matched | missing | extra | extra self | extra below AREPO cut | extra area sum | extra area max | max area diff | max normal diff | max center diff | max volume diff |")
        println(io, "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for s in stats
            @printf(io, "| %d | %d | %d | %d | %d | %d | %d | %d | %d | %.12g | %.12g | %.12g | %.12g | %.12g | %.12g |\n",
                    s.pass_index, s.trace_active, s.trace_pairs,
                    s.native_pairs, s.matched, s.missing, s.extra,
                    s.extra_self, s.extra_below_arepo_cut,
                    s.extra_area_sum, s.extra_area_max, s.max_area_diff,
                    s.max_normal_diff, s.max_center_diff, s.max_volume_diff)
        end
        println(io)
        println(io, "## Tessellator API")
        println(io)
        println(io, "| pass | algorithm | canonical faces |")
        println(io, "| ---: | --- | ---: |")
        for s in stats
            @printf(io, "| %d | `%s` | %d |\n",
                    s.pass_index, s.algorithm, s.canonical_faces)
        end
        for s in stats
            isempty(s.extra_rows) && continue
            println(io)
            @printf(io, "## Pass %d Extra Native Faces\n\n", s.pass_index)
            println(io, "| pair | face | area | AREPO cut | area/cut | face center |")
            println(io, "| --- | ---: | ---: | ---: | ---: | --- |")
            for r in s.extra_rows
                ratio = r.arepo_cut > 0 ? r.area / r.arepo_cut : Inf
                @printf(io, "| (%d,%d) | %d | %.12g | %.12g | %.12g | (%.12g, %.12g, %.12g) |\n",
                        r.pair[1], r.pair[2], r.face, r.area, r.arepo_cut,
                        ratio, r.center[1], r.center[2], r.center[3])
            end
        end
    end
end

function main_native_trace()
    mkpath(NATIVE_TRACE_OUTDIR)
    report = joinpath(NATIVE_TRACE_OUTDIR, "README.md")
    if !_native_trace_bridge_available()
        _write_native_trace_report(report; status = "skipped: missing trace/pre-flux bridge")
        @printf("wrote %s\n", report)
        @printf("skipped: trace or pre-flux bridge is unavailable\n")
        return
    end
    dir = stage_arepo_case(N; riemann = RIEMANN)
    exported = arepo_initial_export(dir)
    try
        step_status = ArepoLib.run_step!(exported.h)
        trace = ArepoLib.get_hydro_face_traces_3d(exported.h)
        snapshots = ArepoLib.get_hydro_preflux_states_3d(exported.h)
        stats = NamedTuple[]
        for snapshot in snapshots
            hasproperty(snapshot, :pos) ||
                error("pre-flux snapshot lacks generator positions; rebuild AREPO/ArepoLib bridge")
            native_ref = build_arepo_tessellation_3d(snapshot.pos;
                                                     algorithm = NATIVE_TRACE_ALGORITHM,
                                                     return_delaunay = NATIVE_TRACE_ALGORITHM == :arepo_delaunay_reference,
                                                     T = Float64,
                                                     bins_per_axis = N,
                                                     search_radius = NATIVE_TRACE_RADIUS,
                                                     min_face_surface_fraction = NATIVE_TRACE_MIN_FACE_SURFACE_FRACTION,
                                                     threaded = false)
            native = (; geom = native_ref.geom,
                      face_center = native_ref.face_center,
                      face_image_shift = native_ref.face_image_shift,
                      reference = native_ref)
            push!(stats, _face_diffs(snapshot, trace, native, snapshot.pass_index))
        end
        status = all(s -> s.missing == 0 && s.extra == 0 &&
                          s.max_area_diff <= 1e-10 &&
                          s.max_volume_diff <= 1e-10, stats) ?
                 "passed" : "diagnostic"
        _write_native_trace_report(report; status, step_status, stats)
        @printf("wrote %s\n", report)
        @printf("native rebuild trace %s: passes=%d missing=%d extra=%d maxvol=%.6g\n",
                status, length(stats), sum(s.missing for s in stats),
                sum(s.extra for s in stats), maximum(s.max_volume_diff for s in stats))
    finally
        ArepoLib.finalize(exported.h)
    end
end

main_native_trace()
