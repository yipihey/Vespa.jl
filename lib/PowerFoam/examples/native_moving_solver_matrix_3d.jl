const _REQUEST_METAL = lowercase(get(ENV, "POWERFOAM_BACKEND", "metal")) == "metal"
const _METAL_IMPORT_ERROR = Ref{Any}(nothing)
if _REQUEST_METAL
    try
        @eval using Metal
    catch err
        _METAL_IMPORT_ERROR[] = err
    end
end

using KernelAbstractions
using LinearAlgebra
using PowerFoam
using Printf
using Random
using Statistics

const GAMMA = 5 / 3
const OUTBASE = joinpath(@__DIR__, "out", "native_moving_solver_matrix_3d")

parse_arg(i, default, T) = length(ARGS) >= i ? parse(T, ARGS[i]) : default
parse_list_arg(i, default, T) =
    split(length(ARGS) >= i ? ARGS[i] : default, ",") .|> x -> parse(T, strip(x))

const N = parse_arg(1, 12, Int)
const DT = parse_arg(2, 0.001, Float64)
const STEP_COUNTS = parse_list_arg(3, "8", Int)
const SOLVERS = Symbol.(split(length(ARGS) >= 4 ? ARGS[4] : "hll,llf", ","))
const SEARCH_RADIUS = parse_arg(5, 1, Int)
const ORDER = Symbol(length(ARGS) >= 6 ? ARGS[6] : "first")
ORDER in (:first, :reconstruct) || error("ORDER must be first or reconstruct")
const PERF_WARMUP = lowercase(get(ENV, "POWERFOAM_PERF_WARMUP", "true")) in
                    ("1", "true", "yes")
const DIAGNOSTICS = Symbol(lowercase(get(ENV, "POWERFOAM_DIAGNOSTICS", "final")))
DIAGNOSTICS in (:final, :step) ||
    error("POWERFOAM_DIAGNOSTICS must be final or step")
const SYNC_TIMING = lowercase(get(ENV, "POWERFOAM_SYNC_TIMING", "false")) in
                    ("1", "true", "yes")
const MESH_PROFILE = lowercase(get(ENV, "POWERFOAM_MESH_PROFILE", "false")) in
                     ("1", "true", "yes")
const REBUILD_MODE = Symbol(lowercase(get(ENV, "POWERFOAM_REBUILD", "exact")))
REBUILD_MODE in (:exact, :gpu_fixed, :gpu_local, :gpu_compact) ||
    error("POWERFOAM_REBUILD must be exact, gpu_fixed, gpu_local, or gpu_compact")
const ACTIVE_CELL_MODE = Symbol(lowercase(get(ENV, "POWERFOAM_ACTIVE_CELLS", "off")))
ACTIVE_CELL_MODE in (:off, :gradients, :all) ||
    error("POWERFOAM_ACTIVE_CELLS must be off, gradients, or all")
const COMPACT_SCAN_CHUNK = parse(Int, get(ENV, "POWERFOAM_COMPACT_SCAN_CHUNK", "256"))
COMPACT_SCAN_CHUNK > 0 || error("POWERFOAM_COMPACT_SCAN_CHUNK must be positive")
const DEFAULT_COMPACT_CELL_SCAN_MODE = :parallel
const COMPACT_CELL_SCAN_MODE =
    Symbol(lowercase(get(ENV, "POWERFOAM_COMPACT_CELL_SCAN_MODE",
                         String(DEFAULT_COMPACT_CELL_SCAN_MODE))))
COMPACT_CELL_SCAN_MODE in (:chunked, :parallel) ||
    error("POWERFOAM_COMPACT_CELL_SCAN_MODE must be chunked or parallel")
const FACE_CLIP_WORKGROUP = parse(Int, get(ENV, "POWERFOAM_FACE_CLIP_WORKGROUP", "16"))
FACE_CLIP_WORKGROUP > 0 || error("POWERFOAM_FACE_CLIP_WORKGROUP must be positive")
const CLIP_SELF_IMAGES = lowercase(get(ENV, "POWERFOAM_CLIP_SELF_IMAGES", "true")) in
                         ("1", "true", "yes")
const PLANE_CULL = lowercase(get(ENV, "POWERFOAM_PLANE_CULL", "true")) in
                   ("1", "true", "yes")
const CANDIDATE_TIER = Symbol(lowercase(get(ENV, "POWERFOAM_CANDIDATE_TIER", "full")))
CANDIDATE_TIER in (:full, :axial, :axis_edge) ||
    error("POWERFOAM_CANDIDATE_TIER must be full, axial, or axis_edge")
const CANDIDATE_TIER_CODE = CANDIDATE_TIER == :full ? Int32(0) :
                            CANDIDATE_TIER == :axial ? Int32(1) : Int32(2)
const MESH_WORK_STATS = lowercase(get(ENV, "POWERFOAM_MESH_WORK_STATS", "false")) in
                        ("1", "true", "yes")
const DIRTY_MOTION_THRESHOLD = parse(Float64, get(ENV, "POWERFOAM_DIRTY_MOTION_THRESHOLD",
                                                  string(0.05 / N)))
DIRTY_MOTION_THRESHOLD >= 0 || error("POWERFOAM_DIRTY_MOTION_THRESHOLD must be non-negative")
const STEP_TAG = length(STEP_COUNTS) == 1 ? @sprintf("n%d", only(STEP_COUNTS)) :
                 "n" * join(STEP_COUNTS, "-")
const ACTIVE_TAG = ACTIVE_CELL_MODE == :off ? "" : "_active-$(ACTIVE_CELL_MODE)"
const CELL_SCAN_TAG = COMPACT_CELL_SCAN_MODE == DEFAULT_COMPACT_CELL_SCAN_MODE ?
                      "" : "_$(COMPACT_CELL_SCAN_MODE)-cellscan"
const SELF_CLIP_TAG = CLIP_SELF_IMAGES ? "" : "_no-selfclip"
const PLANE_CULL_TAG = PLANE_CULL ? "" : "_nocull"
const TIER_TAG = CANDIDATE_TIER == :full ? "" : "_$(CANDIDATE_TIER)-candidates"
const SYNC_TAG = SYNC_TIMING ? "_sync-timing" : ""
const PROFILE_TAG = MESH_PROFILE ? "_mesh-profile" : ""
const WORK_STATS_TAG = MESH_WORK_STATS ? "_workstats" : ""
const RUN_TAG = replace(@sprintf("N%d_dt%.3g_%s_r%d_%s_%s_%s%s", N, DT, STEP_TAG,
                                 SEARCH_RADIUS, ORDER, REBUILD_MODE,
                                 join(String.(SOLVERS), "-"),
                                 string(ACTIVE_TAG, CELL_SCAN_TAG,
                                        SELF_CLIP_TAG, PLANE_CULL_TAG,
                                        TIER_TAG, SYNC_TAG, PROFILE_TAG,
                                        WORK_STATS_TAG)),
                        "." => "p")
const OUTDIR = joinpath(OUTBASE, RUN_TAG)

mutable struct TimingAccumulator
    primitive::Float64
    old_mesh::Float64
    advect::Float64
    new_mesh::Float64
    connections::Float64
    staging::Float64
    gradients::Float64
    hydro::Float64
    finalize::Float64
end

TimingAccumulator() = TimingAccumulator((0.0 for _ in 1:9)...)

function add_timing!(timing::TimingAccumulator, field::Symbol, t0::UInt64)
    setfield!(timing, field, getfield(timing, field) + (time_ns() - t0) * 1e-9)
end

function record_timing!(be, timing::TimingAccumulator, field::Symbol, t0::UInt64)
    SYNC_TIMING && KernelAbstractions.synchronize(be)
    return add_timing!(timing, field, t0)
end

timing_rows(t::TimingAccumulator, nsteps) = (
    (:primitive, t.primitive / nsteps),
    (:old_mesh, t.old_mesh / nsteps),
    (:advect, t.advect / nsteps),
    (:new_mesh, t.new_mesh / nsteps),
    (:connections, t.connections / nsteps),
    (:staging, t.staging / nsteps),
    (:gradients, t.gradients / nsteps),
    (:hydro, t.hydro / nsteps),
    (:finalize, t.finalize / nsteps),
)

mutable struct MeshTimingAccumulator
    face_clip::Float64
    volumes::Float64
    active_cells::Float64
    face_scan::Float64
    face_pack::Float64
    cell_scan::Float64
    csr_fill::Float64
end

MeshTimingAccumulator() = MeshTimingAccumulator((0.0 for _ in 1:7)...)

function add_mesh_timing!(timing::MeshTimingAccumulator, field::Symbol, t0::UInt64)
    setfield!(timing, field, getfield(timing, field) + (time_ns() - t0) * 1e-9)
end

function record_mesh_timing!(be, timing, field::Symbol, t0::UInt64)
    MESH_PROFILE || return nothing
    timing === nothing && return nothing
    KernelAbstractions.synchronize(be)
    return add_mesh_timing!(timing, field, t0)
end

mesh_timing_rows(t::MeshTimingAccumulator, nsteps) = (
    (:face_clip, t.face_clip / nsteps),
    (:volumes, t.volumes / nsteps),
    (:active_cells, t.active_cells / nsteps),
    (:face_scan, t.face_scan / nsteps),
    (:face_pack, t.face_pack / nsteps),
    (:cell_scan, t.cell_scan / nsteps),
    (:csr_fill, t.csr_fill / nsteps),
)

mutable struct MeshWorkAccumulator
    refreshes::Int
    candidate_faces::Int
    dirty_cells::Int
    dirty_faces::Int
    active_faces::Int
    clip_planes::Int
    clip_inside::Int
    clip_empty::Int
    clip_clipped::Int
    tier_rejected::Int
end

MeshWorkAccumulator() = MeshWorkAccumulator((0 for _ in 1:10)...)

mesh_work_rows(w::MeshWorkAccumulator) = (
    (:refreshes, w.refreshes),
    (:candidate_faces, w.candidate_faces),
    (:dirty_cells, w.dirty_cells),
    (:dirty_faces, w.dirty_faces),
    (:active_faces, w.active_faces),
    (:clip_planes, w.clip_planes),
    (:clip_inside, w.clip_inside),
    (:clip_empty, w.clip_empty),
    (:clip_clipped, w.clip_clipped),
    (:tier_rejected, w.tier_rejected),
)

@kernel function _advect_points_backend3_k!(px, py, pz, @Const(vx), @Const(vy),
                                            @Const(vz), dt, box)
    i = @index(Global, Linear)
    @inbounds begin
        x = px[i] + dt * vx[i]
        y = py[i] + dt * vy[i]
        z = pz[i] + dt * vz[i]
        px[i] = x - floor(x / box) * box
        py[i] = y - floor(y / box) * box
        pz[i] = z - floor(z / box) * box
    end
end

@kernel function _advect_points_to_backend3_k!(nx, ny, nz, @Const(px), @Const(py),
                                               @Const(pz), @Const(vx), @Const(vy),
                                               @Const(vz), dt, box)
    i = @index(Global, Linear)
    @inbounds begin
        x = px[i] + dt * vx[i]
        y = py[i] + dt * vy[i]
        z = pz[i] + dt * vz[i]
        nx[i] = x - floor(x / box) * box
        ny[i] = y - floor(y / box) * box
        nz[i] = z - floor(z / box) * box
    end
end

@kernel function _refresh_fixed_topology_faces3_k!(
    face_center_x, face_center_y, face_center_z,
    normal_x, normal_y, normal_z,
    face_vx, face_vy, face_vz,
    @Const(c1), @Const(c2), @Const(px), @Const(py), @Const(pz),
    @Const(vx), @Const(vy), @Const(vz), box)
    f = @index(Global, Linear)
    T = eltype(face_center_x)
    @inbounds begin
        i = Int(c1[f])
        j = Int(c2[f])
        if j > 0
            dx = px[j] - px[i]
            dy = py[j] - py[i]
            dz = pz[j] - pz[i]
            half = T(0.5) * box
            dx = dx < -half ? dx + box : dx > half ? dx - box : dx
            dy = dy < -half ? dy + box : dy > half ? dy - box : dy
            dz = dz < -half ? dz + box : dz > half ? dz - box : dz
            dist = sqrt(dx * dx + dy * dy + dz * dz)
            invd = one(T) / dist
            normal_x[f] = dx * invd
            normal_y[f] = dy * invd
            normal_z[f] = dz * invd
            x = px[i] + T(0.5) * dx
            y = py[i] + T(0.5) * dy
            z = pz[i] + T(0.5) * dz
            face_center_x[f] = x - floor(x / box) * box
            face_center_y[f] = y - floor(y / box) * box
            face_center_z[f] = z - floor(z / box) * box
            face_vx[f] = T(0.5) * (vx[i] + vx[j])
            face_vy[f] = T(0.5) * (vy[i] + vy[j])
            face_vz[f] = T(0.5) * (vz[i] + vz[j])
        end
    end
end

@kernel function _refresh_compact_face_velocities3_k!(
    face_vx, face_vy, face_vz, @Const(c1), @Const(c2),
    @Const(vx), @Const(vy), @Const(vz))
    f = @index(Global, Linear)
    T = eltype(face_vx)
    @inbounds begin
        i = Int(c1[f])
        j = Int(c2[f])
        if i > 0 && j > 0
            face_vx[f] = T(0.5) * (vx[i] + vx[j])
            face_vy[f] = T(0.5) * (vy[i] + vy[j])
            face_vz[f] = T(0.5) * (vz[i] + vz[j])
        elseif i > 0
            face_vx[f] = vx[i]
            face_vy[f] = vy[i]
            face_vz[f] = vz[i]
        else
            face_vx[f] = zero(T)
            face_vy[f] = zero(T)
            face_vz[f] = zero(T)
        end
    end
end

@inline function _wrap_index_shift3(raw, n)
    if raw < 1
        return raw + n, -1
    elseif raw > n
        return raw - n, 1
    else
        return raw, 0
    end
end

@inline _cell_id_kernel3(ix, iy, iz, n) = ix + n * (iy - 1) + n * n * (iz - 1)
@inline _periodic_delta_local3(dx, box) = dx - round(dx / box) * box

@inline function _cell_ijk_kernel3(id, n)
    q = id - 1
    ix = q % n + 1
    iy = (q ÷ n) % n + 1
    iz = q ÷ (n * n) + 1
    return ix, iy, iz
end

@inline function _neighbor_id_shift3(id, dx, dy, dz, n)
    ix, iy, iz = _cell_ijk_kernel3(id, n)
    jx, sx = _wrap_index_shift3(ix + dx, n)
    jy, sy = _wrap_index_shift3(iy + dy, n)
    jz, sz = _wrap_index_shift3(iz + dz, n)
    return _cell_id_kernel3(jx, jy, jz, n), sx, sy, sz
end

@inline function _plane_for_neighbor3(i, dx, dy, dz, n, px, py, pz, box)
    j, sx, sy, sz = _neighbor_id_shift3(i, dx, dy, dz, n)
    pix = px[i]; piy = py[i]; piz = pz[i]
    pjx = px[j] + sx * box
    pjy = py[j] + sy * box
    pjz = pz[j] + sz * box
    a1 = 2 * (pjx - pix)
    a2 = 2 * (pjy - piy)
    a3 = 2 * (pjz - piz)
    b = pjx * pjx + pjy * pjy + pjz * pjz - pix * pix - piy * piy - piz * piz
    return a1, a2, a3, b
end

@inline function _plane_for_self_image3(i, q, px, py, pz, box)
    pix = px[i]; piy = py[i]; piz = pz[i]
    if q == 1
        return one(pix), zero(pix), zero(pix), pix + 0.5f0 * box
    elseif q == 2
        return -one(pix), zero(pix), zero(pix), -(pix - 0.5f0 * box)
    elseif q == 3
        return zero(pix), one(pix), zero(pix), piy + 0.5f0 * box
    elseif q == 4
        return zero(pix), -one(pix), zero(pix), -(piy - 0.5f0 * box)
    elseif q == 5
        return zero(pix), zero(pix), one(pix), piz + 0.5f0 * box
    else
        return zero(pix), zero(pix), -one(pix), -(piz - 0.5f0 * box)
    end
end

@inline function _clip_polygon_plane3!(xout, yout, xin, yin, nv, ca, cb, cc)
    nv == 0 && return 0
    nout = 0
    px0 = xin[nv]
    py0 = yin[nv]
    sp = ca * px0 + cb * py0 - cc
    pin = sp <= 1.0f-5
    @inbounds for q in 1:nv
        cx0 = xin[q]
        cy0 = yin[q]
        sc = ca * cx0 + cb * cy0 - cc
        cin = sc <= 1.0f-5
        if cin != pin
            den = sp - sc
            t = abs(den) > 1.0f-12 ? sp / den : 0.5f0
            nout += 1
            if nout <= 64
                xout[nout] = px0 + t * (cx0 - px0)
                yout[nout] = py0 + t * (cy0 - py0)
            end
        end
        if cin
            nout += 1
            if nout <= 64
                xout[nout] = cx0
                yout[nout] = cy0
            end
        end
        px0 = cx0
        py0 = cy0
        sp = sc
        pin = cin
    end
    return min(nout, 64)
end

@inline function _clip_polygon_plane_lane3!(xout, yout, xin, yin, nv,
                                            ca, cb, cc, lane)
    nv == 0 && return 0
    nout = 0
    px0 = xin[nv, lane]
    py0 = yin[nv, lane]
    sp = ca * px0 + cb * py0 - cc
    pin = sp <= 1.0f-5
    @inbounds for q in 1:nv
        cx0 = xin[q, lane]
        cy0 = yin[q, lane]
        sc = ca * cx0 + cb * cy0 - cc
        cin = sc <= 1.0f-5
        if cin != pin
            den = sp - sc
            t = abs(den) > 1.0f-12 ? sp / den : 0.5f0
            nout += 1
            if nout <= 64
                xout[nout, lane] = px0 + t * (cx0 - px0)
                yout[nout, lane] = py0 + t * (cy0 - py0)
            end
        end
        if cin
            nout += 1
            if nout <= 64
                xout[nout, lane] = cx0
                yout[nout, lane] = cy0
            end
        end
        px0 = cx0
        py0 = cy0
        sp = sc
        pin = cin
    end
    return min(nout, 64)
end

@inline function _clip_or_cull_polygon_plane_lane3!(xout, yout, xin, yin, nv,
                                                    ca, cb, cc, lane,
                                                    plane_cull)
    if plane_cull
        any_inside = false
        any_outside = false
        @inbounds for q in 1:nv
            s = ca * xin[q, lane] + cb * yin[q, lane] - cc
            inside = s <= 1.0f-5
            any_inside |= inside
            any_outside |= !inside
        end
        !any_outside && return nv, Int32(1)
        !any_inside && return 0, Int32(2)
    end
    return _clip_polygon_plane_lane3!(xout, yout, xin, yin, nv,
                                      ca, cb, cc, lane), Int32(3)
end

@inline function _candidate_tier_allowed3(i, j, sx, sy, sz, n, tier)
    tier == 0 && return true
    ix, iy, iz = _cell_ijk_kernel3(i, n)
    jx, jy, jz = _cell_ijk_kernel3(j, n)
    dx = jx + Int(sx) * n - ix
    dy = jy + Int(sy) * n - iy
    dz = jz + Int(sz) * n - iz
    manhattan = abs(dx) + abs(dy) + abs(dz)
    tier == 1 && return manhattan <= 1
    return manhattan <= 2
end

@kernel function _refresh_local_candidate_faces3_k!(
    face_area, face_center_x, face_center_y, face_center_z,
    normal_x, normal_y, normal_z, face_vx, face_vy, face_vz,
    clip_stats,
    @Const(c1), @Const(c2), @Const(shift_x), @Const(shift_y), @Const(shift_z),
    @Const(px), @Const(py), @Const(pz), @Const(vx), @Const(vy), @Const(vz),
    ngrid, box, clip_self_images, plane_cull, candidate_tier)
    f = @index(Global, Linear)
    lane = @index(Local, Linear)
    @uniform lanes = prod(@groupsize())
    T = eltype(face_area)
    x1 = @localmem eltype(face_area) (64, lanes)
    y1 = @localmem eltype(face_area) (64, lanes)
    x2 = @localmem eltype(face_area) (64, lanes)
    y2 = @localmem eltype(face_area) (64, lanes)
    @inbounds begin
        i = Int(c1[f])
        j = Int(c2[f])
        sx = shift_x[f]
        sy = shift_y[f]
        sz = shift_z[f]
        nclip = zero(eltype(clip_stats))
        ninside = zero(eltype(clip_stats))
        nempty = zero(eltype(clip_stats))
        nclipped = zero(eltype(clip_stats))
        stat_base = (f - 1) * 5
        rejected = !_candidate_tier_allowed3(i, j, sx, sy, sz, Int(ngrid),
                                             Int(candidate_tier))
        if rejected
            face_area[f] = zero(T)
            normal_x[f] = zero(T)
            normal_y[f] = zero(T)
            normal_z[f] = zero(T)
            face_center_x[f] = zero(T)
            face_center_y[f] = zero(T)
            face_center_z[f] = zero(T)
            face_vx[f] = zero(T)
            face_vy[f] = zero(T)
            face_vz[f] = zero(T)
            clip_stats[stat_base + 1] = nclip
            clip_stats[stat_base + 2] = ninside
            clip_stats[stat_base + 3] = nempty
            clip_stats[stat_base + 4] = nclipped
            clip_stats[stat_base + 5] = one(eltype(clip_stats))
        else
            clip_stats[stat_base + 5] = zero(eltype(clip_stats))
            pix = px[i]; piy = py[i]; piz = pz[i]
        pjx = px[j] + T(sx) * box
        pjy = py[j] + T(sy) * box
        pjz = pz[j] + T(sz) * box
        dx = pjx - pix
        dy = pjy - piy
        dz = pjz - piz
        dist = sqrt(dx * dx + dy * dy + dz * dz)
        valid = dist > zero(T)
        dist = valid ? dist : one(T)
        nx = dx / dist
        ny = dy / dist
        nz = dz / dist
        ox = T(0.5) * (pix + pjx)
        oy = T(0.5) * (piy + pjy)
        oz = T(0.5) * (piz + pjz)
        ux = abs(nx) < T(0.8) ? zero(T) : -nz
        uy = abs(nx) < T(0.8) ? -nz : zero(T)
        uz = abs(nx) < T(0.8) ? ny : nx
        un = sqrt(ux * ux + uy * uy + uz * uz)
        ux /= un; uy /= un; uz /= un
        vx1 = ny * uz - nz * uy
        vy1 = nz * ux - nx * uz
        vz1 = nx * uy - ny * ux

        h = box
        x1[1, lane] = -h; y1[1, lane] = -h
        x1[2, lane] =  h; y1[2, lane] = -h
        x1[3, lane] =  h; y1[3, lane] =  h
        x1[4, lane] = -h; y1[4, lane] =  h
        nv = 4
        src_is_1 = true

        if clip_self_images
            for selfp in 1:6
                nv > 0 || continue
                a1, a2, a3, b = _plane_for_self_image3(i, selfp, px, py, pz, box)
                ca = a1 * ux + a2 * uy + a3 * uz
                cb = a1 * vx1 + a2 * vy1 + a3 * vz1
                cc = b - (a1 * ox + a2 * oy + a3 * oz)
                action = Int32(0)
                if src_is_1
                    nv, action = _clip_or_cull_polygon_plane_lane3!(
                        x2, y2, x1, y1, nv, ca, cb, cc, lane, plane_cull)
                else
                    nv, action = _clip_or_cull_polygon_plane_lane3!(
                        x1, y1, x2, y2, nv, ca, cb, cc, lane, plane_cull)
                end
                nclip += one(eltype(clip_stats))
                ninside += action == 1 ? one(eltype(clip_stats)) : zero(eltype(clip_stats))
                nempty += action == 2 ? one(eltype(clip_stats)) : zero(eltype(clip_stats))
                nclipped += action == 3 ? one(eltype(clip_stats)) : zero(eltype(clip_stats))
                src_is_1 = action == 3 ? !src_is_1 : src_is_1
            end
        end
        for ozoff in -1:1, oyoff in -1:1, oxoff in -1:1
            nv > 0 || continue
            oxoff == 0 && oyoff == 0 && ozoff == 0 && continue
            a1, a2, a3, b = _plane_for_neighbor3(i, oxoff, oyoff, ozoff,
                                                  Int(ngrid), px, py, pz, box)
            ca = a1 * ux + a2 * uy + a3 * uz
            cb = a1 * vx1 + a2 * vy1 + a3 * vz1
            cc = b - (a1 * ox + a2 * oy + a3 * oz)
            action = Int32(0)
            if src_is_1
                nv, action = _clip_or_cull_polygon_plane_lane3!(
                    x2, y2, x1, y1, nv, ca, cb, cc, lane, plane_cull)
            else
                nv, action = _clip_or_cull_polygon_plane_lane3!(
                    x1, y1, x2, y2, nv, ca, cb, cc, lane, plane_cull)
            end
            nclip += one(eltype(clip_stats))
            ninside += action == 1 ? one(eltype(clip_stats)) : zero(eltype(clip_stats))
            nempty += action == 2 ? one(eltype(clip_stats)) : zero(eltype(clip_stats))
            nclipped += action == 3 ? one(eltype(clip_stats)) : zero(eltype(clip_stats))
            src_is_1 = action == 3 ? !src_is_1 : src_is_1
        end

        valid = valid && nv >= 3
        sx2 = zero(T); su = zero(T); sv = zero(T)
        if src_is_1
            for q in 1:nv
                r = q == nv ? 1 : q + 1
                sx2 += x1[q, lane] * y1[r, lane] -
                       y1[q, lane] * x1[r, lane]
                su += x1[q, lane]
                sv += y1[q, lane]
            end
        else
            for q in 1:nv
                r = q == nv ? 1 : q + 1
                sx2 += x2[q, lane] * y2[r, lane] -
                       y2[q, lane] * x2[r, lane]
                su += x2[q, lane]
                sv += y2[q, lane]
            end
        end
        area = T(0.5) * abs(sx2)
        valid = valid && area > T(1.0f-8)
        denom = T(max(nv, 1))
        cu = su / denom
        cv = sv / denom
        fx = ox + cu * ux + cv * vx1
        fy = oy + cu * uy + cv * vy1
        fz = oz + cu * uz + cv * vz1
        face_area[f] = valid ? area : zero(T)
        normal_x[f] = nx
        normal_y[f] = ny
        normal_z[f] = nz
        face_center_x[f] = fx - floor(fx / box) * box
        face_center_y[f] = fy - floor(fy / box) * box
        face_center_z[f] = fz - floor(fz / box) * box
        face_vx[f] = T(0.5) * (vx[i] + vx[j])
        face_vy[f] = T(0.5) * (vy[i] + vy[j])
        face_vz[f] = T(0.5) * (vz[i] + vz[j])
            clip_stats[stat_base + 1] = nclip
            clip_stats[stat_base + 2] = ninside
            clip_stats[stat_base + 3] = nempty
            clip_stats[stat_base + 4] = nclipped
        end
    end
end

@kernel function _update_local_candidate_volumes3_k!(
    volume, @Const(offsets), @Const(cell_faces), @Const(cell_signs),
    @Const(face_area), @Const(normal_x), @Const(normal_y), @Const(normal_z),
    @Const(face_center_x), @Const(face_center_y), @Const(face_center_z),
    @Const(px), @Const(py), @Const(pz), box)
    i = @index(Global, Linear)
    T = eltype(volume)
    acc = zero(T)
    @inbounds begin
        cx = px[i]
        cy = py[i]
        cz = pz[i]
        for p in Int(offsets[i]):(Int(offsets[i + 1]) - 1)
            f = Int(cell_faces[p])
            area = face_area[f]
            area > zero(T) || continue
            s = -T(cell_signs[p])
            dx = _periodic_delta_local3(face_center_x[f] - cx, box)
            dy = _periodic_delta_local3(face_center_y[f] - cy, box)
            dz = _periodic_delta_local3(face_center_z[f] - cz, box)
            acc += area * s *
                   (dx * normal_x[f] + dy * normal_y[f] + dz * normal_z[f])
        end
        volume[i] = max(acc / T(3), T(1.0f-12))
    end
end

@kernel function _mark_motion_dirty_cells3_k!(dirty, @Const(px0), @Const(py0),
                                              @Const(pz0), @Const(px1),
                                              @Const(py1), @Const(pz1),
                                              box, threshold2)
    i = @index(Global, Linear)
    T = eltype(px0)
    @inbounds begin
        dx = _periodic_delta_local3(px1[i] - px0[i], box)
        dy = _periodic_delta_local3(py1[i] - py0[i], box)
        dz = _periodic_delta_local3(pz1[i] - pz0[i], box)
        dirty[i] = dx * dx + dy * dy + dz * dz > T(threshold2) ? Int32(1) : Int32(0)
    end
end

@kernel function _mark_dirty_candidate_faces3_k!(dirty_faces, @Const(c1), @Const(c2),
                                                 @Const(dirty_cells))
    f = @index(Global, Linear)
    @inbounds begin
        dirty_faces[f] = (dirty_cells[Int(c1[f])] != 0 ||
                          dirty_cells[Int(c2[f])] != 0) ? Int32(1) : Int32(0)
    end
end

@kernel function _compact_active_cell_faces_fixed3_k!(
    active_counts, active_faces, active_signs,
    @Const(offsets), @Const(cell_faces), @Const(cell_signs),
    @Const(face_area), active_stride)
    i = @index(Global, Linear)
    T = eltype(face_area)
    stride = Int(active_stride)
    count = 0
    @inbounds begin
        base = (i - 1) * stride
        for p in Int(offsets[i]):(Int(offsets[i + 1]) - 1)
            f = Int(cell_faces[p])
            if face_area[f] > zero(T) && count < stride
                count += 1
                active_faces[base + count] = cell_faces[p]
                active_signs[base + count] = cell_signs[p]
            end
        end
        active_counts[i] = count
        for q in (count + 1):stride
            active_faces[base + q] = zero(eltype(active_faces))
            active_signs[base + q] = zero(eltype(active_signs))
        end
    end
end

@kernel function _scan_active_faces_serial3_k!(face_prefix, active_total,
                                               @Const(face_area))
    T = eltype(face_area)
    count = zero(eltype(face_prefix))
    @inbounds begin
        for f in 1:Int(length(face_area))
            if face_area[f] > zero(T)
                count += one(eltype(face_prefix))
                face_prefix[f] = count
            else
                face_prefix[f] = zero(eltype(face_prefix))
            end
        end
        active_total[1] = count
    end
end

@kernel function _scan_active_faces_chunks3_k!(face_prefix, block_counts,
                                               @Const(face_area), chunk)
    b = @index(Global, Linear)
    T = eltype(face_area)
    I = eltype(face_prefix)
    first = (b - 1) * Int(chunk) + 1
    last = min(b * Int(chunk), Int(length(face_area)))
    count = zero(I)
    @inbounds begin
        for f in first:last
            if face_area[f] > zero(T)
                count += one(I)
                face_prefix[f] = count
            else
                face_prefix[f] = zero(I)
            end
        end
        block_counts[b] = count
    end
end

@kernel function _scan_counts_serial3_k!(block_offsets, total,
                                         @Const(block_counts))
    I = eltype(block_offsets)
    acc = zero(I)
    @inbounds begin
        for b in 1:Int(length(block_counts))
            block_offsets[b] = acc
            acc += block_counts[b]
        end
        total[1] = acc
    end
end

@kernel function _compact_faces3_k!(
    compact_c1, compact_c2, compact_area,
    compact_nx, compact_ny, compact_nz,
    compact_fvx, compact_fvy, compact_fvz,
    compact_fcx, compact_fcy, compact_fcz,
    face_prefix, @Const(active_total), @Const(block_offsets),
    @Const(c1), @Const(c2), @Const(face_area),
    @Const(normal_x), @Const(normal_y), @Const(normal_z),
    @Const(face_vx), @Const(face_vy), @Const(face_vz),
    @Const(face_center_x), @Const(face_center_y), @Const(face_center_z),
    chunk)
    f = @index(Global, Linear)
    T = eltype(compact_area)
    I = eltype(compact_c1)
    @inbounds begin
        cf_local = face_prefix[f]
        if cf_local > zero(eltype(face_prefix))
            b = (f - 1) ÷ Int(chunk) + 1
            cf = cf_local + block_offsets[b]
            face_prefix[f] = cf
            cfi = Int(cf)
            compact_c1[cfi] = c1[f]
            compact_c2[cfi] = c2[f]
            compact_area[cfi] = face_area[f]
            compact_nx[cfi] = normal_x[f]
            compact_ny[cfi] = normal_y[f]
            compact_nz[cfi] = normal_z[f]
            compact_fvx[cfi] = face_vx[f]
            compact_fvy[cfi] = face_vy[f]
            compact_fvz[cfi] = face_vz[f]
            compact_fcx[cfi] = face_center_x[f]
            compact_fcy[cfi] = face_center_y[f]
            compact_fcz[cfi] = face_center_z[f]
        end
        if f > Int(active_total[1])
            compact_c1[f] = one(I)
            compact_c2[f] = one(I)
            compact_area[f] = zero(T)
            compact_nx[f] = zero(T)
            compact_ny[f] = zero(T)
            compact_nz[f] = zero(T)
            compact_fvx[f] = zero(T)
            compact_fvy[f] = zero(T)
            compact_fvz[f] = zero(T)
            compact_fcx[f] = zero(T)
            compact_fcy[f] = zero(T)
            compact_fcz[f] = zero(T)
        end
    end
end

@kernel function _count_scan_compact_cell_chunks3_k!(
    cell_offsets, cell_counts, block_counts,
    @Const(offsets), @Const(cell_faces), @Const(face_prefix), chunk)
    b = @index(Global, Linear)
    I = eltype(cell_offsets)
    first = (b - 1) * Int(chunk) + 1
    last = min(b * Int(chunk), Int(length(cell_counts)))
    acc = zero(I)
    @inbounds begin
        for i in first:last
            count = zero(I)
            for p in Int(offsets[i]):(Int(offsets[i + 1]) - 1)
                f = Int(cell_faces[p])
                if face_prefix[f] > zero(eltype(face_prefix))
                    count += one(I)
                end
            end
            cell_counts[i] = count
            cell_offsets[i] = acc
            acc += count
        end
        block_counts[b] = acc
    end
end

@kernel function _count_compact_cell_faces3_k!(
    cell_counts, @Const(offsets), @Const(cell_faces), @Const(face_prefix))
    i = @index(Global, Linear)
    I = eltype(cell_counts)
    count = zero(I)
    @inbounds begin
        for p in Int(offsets[i]):(Int(offsets[i + 1]) - 1)
            f = Int(cell_faces[p])
            if face_prefix[f] > zero(eltype(face_prefix))
                count += one(I)
            end
        end
        cell_counts[i] = count
    end
end

@kernel function _scan_compact_cell_count_chunks3_k!(
    cell_offsets, block_counts, @Const(cell_counts), chunk)
    b = @index(Global, Linear)
    I = eltype(cell_offsets)
    first = (b - 1) * Int(chunk) + 1
    last = min(b * Int(chunk), Int(length(cell_counts)))
    acc = zero(I)
    @inbounds begin
        for i in first:last
            cell_offsets[i] = acc
            acc += cell_counts[i]
        end
        block_counts[b] = acc
    end
end

@kernel function _finalize_offsets_fill_compact_csr3_k!(
    compact_faces, compact_signs, compact_offsets, @Const(cell_counts),
    @Const(block_offsets),
    @Const(candidate_offsets), @Const(candidate_faces), @Const(candidate_signs),
    @Const(face_prefix), chunk)
    i = @index(Global, Linear)
    I = eltype(compact_faces)
    @inbounds begin
        b = (i - 1) ÷ Int(chunk) + 1
        offset = compact_offsets[i] + block_offsets[b] + one(eltype(compact_offsets))
        compact_offsets[i] = offset
        if i == Int(length(cell_counts))
            compact_offsets[i + 1] = offset + cell_counts[i]
        end
        cursor = Int(offset)
        for p in Int(candidate_offsets[i]):(Int(candidate_offsets[i + 1]) - 1)
            f = Int(candidate_faces[p])
            cf = face_prefix[f]
            if cf > zero(eltype(face_prefix))
                compact_faces[cursor] = I(cf)
                compact_signs[cursor] = candidate_signs[p]
                cursor += 1
            end
        end
    end
end

function maybe_metal_backend()
    _REQUEST_METAL || return nothing
    if _METAL_IMPORT_ERROR[] !== nothing
        @warn "Metal unavailable; running CPU-only native matrix" err = _METAL_IMPORT_ERROR[]
        return nothing
    end
    return Metal.MetalBackend()
end

function grid_points_3d(n)
    pts = Matrix{Float64}(undef, n^3, 3)
    q = 1
    for k in 1:n, j in 1:n, i in 1:n
        pts[q, 1] = (i - 0.5) / n
        pts[q, 2] = (j - 0.5) / n
        pts[q, 3] = (k - 0.5) / n
        q += 1
    end
    return pts
end

const FORWARD_OFFSETS_3D = Tuple{Int,Int,Int}[
    (dx, dy, dz) for dz in -1:1 for dy in -1:1 for dx in -1:1
    if !(dx == 0 && dy == 0 && dz == 0) &&
       (dz > 0 || (dz == 0 && dy > 0) || (dz == 0 && dy == 0 && dx > 0))
]
const LOCAL_ACTIVE_STRIDE_3D = 26

function cell_face_csr_local_3d(ncells, c1, c2, ::Type{I}) where {I<:Integer}
    counts = zeros(Int, ncells)
    @inbounds for f in eachindex(c1)
        counts[Int(c1[f])] += 1
        counts[Int(c2[f])] += 1
    end
    offsets = Vector{I}(undef, ncells + 1)
    offsets[1] = one(I)
    @inbounds for i in 1:ncells
        offsets[i + 1] = offsets[i] + I(counts[i])
    end
    faces = Vector{I}(undef, Int(offsets[end] - one(I)))
    signs = Vector{I}(undef, length(faces))
    cursor = Int.(offsets[1:end-1])
    @inbounds for f in eachindex(c1)
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

function candidate_neighbor_3d(id, dx, dy, dz, n)
    q = id - 1
    ix = mod(q, n) + 1
    iy = mod(div(q, n), n) + 1
    iz = div(q, n * n) + 1
    rawx = ix + dx
    rawy = iy + dy
    rawz = iz + dz
    jx = rawx < 1 ? rawx + n : rawx > n ? rawx - n : rawx
    jy = rawy < 1 ? rawy + n : rawy > n ? rawy - n : rawy
    jz = rawz < 1 ? rawz + n : rawz > n ? rawz - n : rawz
    sx = rawx < 1 ? -1 : rawx > n ? 1 : 0
    sy = rawy < 1 ? -1 : rawy > n ? 1 : 0
    sz = rawz < 1 ? -1 : rawz > n ? 1 : 0
    return jx + n * (jy - 1) + n * n * (jz - 1), sx, sy, sz
end

function local_candidate_mesh_arrays_3d(n; T::Type{<:AbstractFloat} = Float32,
                                        index_type::Type{<:Integer} = Int32)
    nc = n^3
    nf = length(FORWARD_OFFSETS_3D) * nc
    c1 = Vector{index_type}(undef, nf)
    c2 = Vector{index_type}(undef, nf)
    shift_x = Vector{index_type}(undef, nf)
    shift_y = Vector{index_type}(undef, nf)
    shift_z = Vector{index_type}(undef, nf)
    f = 1
    @inbounds for i in 1:nc
        for (dx, dy, dz) in FORWARD_OFFSETS_3D
            j, sx, sy, sz = candidate_neighbor_3d(i, dx, dy, dz, n)
            c1[f] = index_type(i)
            c2[f] = index_type(j)
            shift_x[f] = index_type(sx)
            shift_y[f] = index_type(sy)
            shift_z[f] = index_type(sz)
            f += 1
        end
    end
    offsets, faces, signs = cell_face_csr_local_3d(nc, c1, c2, index_type)
    geom = ArepoMeshArrays3D(c1, c2, offsets, faces, signs,
                             fill(T(1 / nc), nc), zeros(T, nf),
                             zeros(T, nf), zeros(T, nf), zeros(T, nf),
                             zeros(T, nf), zeros(T, nf), zeros(T, nf))
    face_center = zeros(Float64, nf, 3)
    center = grid_points_3d(n)
    return (; geom, center, face_center, shift_x, shift_y, shift_z)
end

function compact_mesh_backend_3d(be, ncells::Integer, nfaces::Integer;
                                 T::Type{<:AbstractFloat} = Float32,
                                 index_type::Type{<:Integer} = Int32)
    ninc = 2 * nfaces
    return ArepoMeshArrays3D(
        PowerFoam._backend_zeros(be, index_type, nfaces),
        PowerFoam._backend_zeros(be, index_type, nfaces),
        PowerFoam._backend_zeros(be, index_type, ncells + 1),
        PowerFoam._backend_zeros(be, index_type, ninc),
        PowerFoam._backend_zeros(be, index_type, ninc),
        PowerFoam._backend_zeros(be, T, ncells),
        PowerFoam._backend_zeros(be, T, nfaces),
        PowerFoam._backend_zeros(be, T, nfaces),
        PowerFoam._backend_zeros(be, T, nfaces),
        PowerFoam._backend_zeros(be, T, nfaces),
        PowerFoam._backend_zeros(be, T, nfaces),
        PowerFoam._backend_zeros(be, T, nfaces),
        PowerFoam._backend_zeros(be, T, nfaces),
    )
end

function solenoidal_modes_3d(points; mach = 0.3, gamma = GAMMA,
                             seed = 271, kmin = 2, kmax = 3)
    rng = MersenneTwister(seed)
    ncell = size(points, 1)
    vx = zeros(Float64, ncell)
    vy = zeros(Float64, ncell)
    vz = zeros(Float64, ncell)
    for kx in -kmax:kmax, ky in -kmax:kmax, kz in -kmax:kmax
        kx == 0 && ky == 0 && kz == 0 && continue
        kmag = sqrt(kx * kx + ky * ky + kz * kz)
        (kmin <= kmag <= kmax) || continue
        phase = 2pi * rand(rng)
        ax, ay, az = randn(rng), randn(rng), randn(rng)
        dotak = ax * kx + ay * ky + az * kz
        ax -= dotak * kx / (kmag * kmag)
        ay -= dotak * ky / (kmag * kmag)
        az -= dotak * kz / (kmag * kmag)
        amp = randn(rng) / (kmag * kmag)
        @inbounds for i in 1:ncell
            arg = 2pi * (kx * points[i, 1] + ky * points[i, 2] +
                         kz * points[i, 3]) + phase
            s = amp * cos(arg)
            vx[i] += ax * s
            vy[i] += ay * s
            vz[i] += az * s
        end
    end
    vx .-= mean(vx)
    vy .-= mean(vy)
    vz .-= mean(vz)
    vrms = sqrt(mean(vx .* vx .+ vy .* vy .+ vz .* vz))
    isfinite(vrms) && vrms > 0 || error("turbulence initializer produced zero velocity")
    cs = sqrt(gamma * (1 / gamma))
    scale = 0.3 * cs / vrms
    return vx .* scale, vy .* scale, vz .* scale
end

function initial_case(be)
    points = grid_points_3d(N)
    vx, vy, vz = solenoidal_modes_3d(points)
    vmesh = hcat(vx, vy, vz)
    mesh = local_periodic_voronoi_mesh_arrays_3d(points; T = Float32,
                                                 bins_per_axis = N,
                                                 search_radius = SEARCH_RADIUS,
                                                 cell_velocity = vmesh)
    state = euler_state_3d(mesh.geom; rho = 1.0, vx, vy, vz,
                           pressure = 1 / GAMMA, gamma = GAMMA, T = Float32)
    return points, to_backend(be, mesh.geom; T = Float32),
           to_backend(be, state; T = Float32), mesh
end

function diagnostics(label, state, geom, points, step, time)
    prim = conserved_to_primitive_3d(state; gamma = GAMMA)
    totals = total_conserved_3d(state, geom)
    v2 = prim.vx .* prim.vx .+ prim.vy .* prim.vy .+ prim.vz .* prim.vz
    cs2 = GAMMA .* prim.pressure ./ prim.rho
    counts = diff(Int.(Array(geom.cell_face_offsets)))
    areas = Array(geom.face_area)
    return (; label, step, time, cells = length(prim.rho), faces = length(geom.c1),
            active_faces = count(>(0), areas),
            face_count_min = minimum(counts), face_count_max = maximum(counts),
            volume_sum = sum(Array(geom.volume)), mass = totals.mass,
            mx = totals.mx, my = totals.my, mz = totals.mz, energy = totals.energy,
            vrms = sqrt(mean(v2)), mach_rms = sqrt(mean(v2 ./ cs2)),
            density_rms = std(prim.rho) / mean(prim.rho),
            rho_min = minimum(prim.rho), rho_max = maximum(prim.rho),
            pmin = minimum(prim.pressure),
            point_disp_rms = sqrt(mean(sum((points .- grid_points_3d(N)) .^ 2; dims = 2)[:])))
end

function run_case(label, be; solver, nsteps)
    points, geom, state, mesh = initial_case(be)
    workspace = ORDER == :reconstruct ?
                reconstructed_workspace(state, mesh, points, be) : nothing
    rows = [diagnostics(label, state, geom, points, 0, 0.0)]
    for step in 1:nsteps
        moved = ORDER == :first ?
            moving_mesh_step_3d!(state, points; dt = DT, gamma = GAMMA,
                                 boundary = :periodic, rebuild = :local,
                                 local_bins_per_axis = N,
                                 local_search_radius = SEARCH_RADIUS,
                                 riemann = solver, T = Float32) :
            reconstructed_native_step!(state, points, be, workspace; solver)
        points = moved.points
        geom = moved.geom
        if DIAGNOSTICS == :step
            push!(rows, diagnostics(label, state, geom, points, step, step * DT))
            prim = conserved_to_primitive_3d(state; gamma = GAMMA)
            minimum(prim.rho) > 0 || error("$label/$solver produced non-positive density")
            minimum(prim.pressure) > 0 || error("$label/$solver produced non-positive pressure")
        end
    end
    if DIAGNOSTICS == :final
        if ORDER == :reconstruct &&
           REBUILD_MODE in (:gpu_fixed, :gpu_local, :gpu_compact)
            sync_fixed_workspace_to_host!(be, workspace)
            points = workspace.points_host
            geom = active_workspace_geom(workspace)
        end
        push!(rows, diagnostics(label, state, geom, points, nsteps, nsteps * DT))
        prim = conserved_to_primitive_3d(state; gamma = GAMMA)
        minimum(prim.rho) > 0 || error("$label/$solver produced non-positive density")
        minimum(prim.pressure) > 0 || error("$label/$solver produced non-positive pressure")
    end
    timing = workspace === nothing ? TimingAccumulator() : workspace.timing
    mesh_timing = workspace === nothing ? MeshTimingAccumulator() : workspace.mesh_timing
    mesh_work = workspace === nothing ? MeshWorkAccumulator() : workspace.mesh_work
    return rows, state, geom, points, timing, mesh_timing, mesh_work
end

function reconstructed_workspace(state, mesh)
    error("reconstructed_workspace requires points and backend")
end

function reconstructed_workspace(state, mesh, points, be)
    n = length(state.D)
    candidate = REBUILD_MODE in (:gpu_local, :gpu_compact) ?
                local_candidate_mesh_arrays_3d(N; T = Float32) :
                nothing
    use_active_cells = candidate !== nothing && ACTIVE_CELL_MODE != :off
    host_geom = candidate === nothing ? mesh.geom : candidate.geom
    host_face_center = candidate === nothing ? mesh.face_center : candidate.face_center
    nf = length(host_geom.c1)
    nface_blocks = cld(nf, COMPACT_SCAN_CHUNK)
    ncell_blocks = cld(n, COMPACT_SCAN_CHUNK)
    fixed_geom = to_backend(be, host_geom; T = Float32)
    compact_geom = REBUILD_MODE == :gpu_compact ?
                   compact_mesh_backend_3d(be, n, nf; T = Float32) :
                   nothing
    compact_new_geom = REBUILD_MODE == :gpu_compact ?
                       compact_mesh_backend_3d(be, n, nf; T = Float32) :
                       nothing
    return (; prim = primitive_work_3d(state),
            prim_host = (; rho = Vector{Float32}(undef, n),
                         vx = Vector{Float32}(undef, n),
                         vy = Vector{Float32}(undef, n),
                         vz = Vector{Float32}(undef, n),
                         pressure = Vector{Float32}(undef, n)),
            vmesh = Matrix{Float64}(undef, n, 3),
            gradients = hydro_gradient_work_3d(state.D),
            current_mesh = Ref{Any}(mesh),
            old_geom_backend = Ref{Any}(nothing),
            new_volume_backend = Ref{Any}(nothing),
            fixed_geom_backend = Ref{Any}(fixed_geom),
            compact_geom_backend = Ref{Any}(compact_geom),
            compact_new_geom_backend = Ref{Any}(compact_new_geom),
            compact_reuse_ready = Ref(false),
            compact_face_prefix_backend = compact_geom === nothing ? nothing :
                                          PowerFoam._backend_zeros(be, Int32, nf),
            compact_face_block_counts_backend = compact_geom === nothing ? nothing :
                                                PowerFoam._backend_zeros(
                                                    be, Int32, nface_blocks),
            compact_face_block_offsets_backend = compact_geom === nothing ? nothing :
                                                 PowerFoam._backend_zeros(
                                                     be, Int32, nface_blocks),
            compact_active_total_backend = compact_geom === nothing ? nothing :
                                           PowerFoam._backend_zeros(be, Int32, 1),
            compact_cell_counts_backend = compact_geom === nothing ? nothing :
                                          PowerFoam._backend_zeros(be, Int32, n),
            compact_cell_block_counts_backend = compact_geom === nothing ? nothing :
                                                PowerFoam._backend_zeros(
                                                    be, Int32, ncell_blocks),
            compact_cell_block_offsets_backend = compact_geom === nothing ? nothing :
                                                 PowerFoam._backend_zeros(
                                                     be, Int32, ncell_blocks),
            clip_stats_backend = candidate === nothing ? nothing :
                                 PowerFoam._backend_zeros(be, Int32, 5 * nf),
            dirty_cells_backend = candidate === nothing ? nothing :
                                  PowerFoam._backend_zeros(be, Int32, n),
            dirty_faces_backend = candidate === nothing ? nothing :
                                  PowerFoam._backend_zeros(be, Int32, nf),
            hydro_work = hydro_work_3d(state, fixed_geom),
            face_states = face_prediction_work_3d(fixed_geom),
            half_dt = PowerFoam._backend_copy(be, fill(Float32(0.5 * DT), n), Float32),
            shift_x_backend = candidate === nothing ? nothing :
                              PowerFoam._backend_copy(be, candidate.shift_x, Int32),
            shift_y_backend = candidate === nothing ? nothing :
                              PowerFoam._backend_copy(be, candidate.shift_y, Int32),
            shift_z_backend = candidate === nothing ? nothing :
                              PowerFoam._backend_copy(be, candidate.shift_z, Int32),
            active_stride = LOCAL_ACTIVE_STRIDE_3D,
            active_counts_backend = use_active_cells ?
                                    PowerFoam._backend_zeros(be, Int32, n) :
                                    nothing,
            active_cell_faces_backend = use_active_cells ?
                                        PowerFoam._backend_zeros(
                                            be, Int32, LOCAL_ACTIVE_STRIDE_3D * n) :
                                        nothing,
            active_cell_signs_backend = use_active_cells ?
                                        PowerFoam._backend_zeros(
                                            be, Int32, LOCAL_ACTIVE_STRIDE_3D * n) :
                                        nothing,
            point_x_backend = Ref{Any}(PowerFoam._backend_copy(be, collect(view(points, :, 1)), Float32)),
            point_y_backend = Ref{Any}(PowerFoam._backend_copy(be, collect(view(points, :, 2)), Float32)),
            point_z_backend = Ref{Any}(PowerFoam._backend_copy(be, collect(view(points, :, 3)), Float32)),
            next_point_x_backend = Ref{Any}(REBUILD_MODE == :gpu_compact ?
                                   PowerFoam._backend_copy(be, collect(view(points, :, 1)), Float32) :
                                   nothing),
            next_point_y_backend = Ref{Any}(REBUILD_MODE == :gpu_compact ?
                                   PowerFoam._backend_copy(be, collect(view(points, :, 2)), Float32) :
                                   nothing),
            next_point_z_backend = Ref{Any}(REBUILD_MODE == :gpu_compact ?
                                   PowerFoam._backend_copy(be, collect(view(points, :, 3)), Float32) :
                                   nothing),
            face_center_x_backend = Ref{Any}(PowerFoam._backend_copy(be, collect(view(host_face_center, :, 1)), Float32)),
            face_center_y_backend = Ref{Any}(PowerFoam._backend_copy(be, collect(view(host_face_center, :, 2)), Float32)),
            face_center_z_backend = Ref{Any}(PowerFoam._backend_copy(be, collect(view(host_face_center, :, 3)), Float32)),
            compact_face_center_x_backend = Ref{Any}(compact_geom === nothing ? nothing :
                                            PowerFoam._backend_zeros(be, Float32, nf)),
            compact_face_center_y_backend = Ref{Any}(compact_geom === nothing ? nothing :
                                            PowerFoam._backend_zeros(be, Float32, nf)),
            compact_face_center_z_backend = Ref{Any}(compact_geom === nothing ? nothing :
                                            PowerFoam._backend_zeros(be, Float32, nf)),
            compact_new_face_center_x_backend = Ref{Any}(compact_geom === nothing ? nothing :
                                                PowerFoam._backend_zeros(be, Float32, nf)),
            compact_new_face_center_y_backend = Ref{Any}(compact_geom === nothing ? nothing :
                                                PowerFoam._backend_zeros(be, Float32, nf)),
            compact_new_face_center_z_backend = Ref{Any}(compact_geom === nothing ? nothing :
                                                PowerFoam._backend_zeros(be, Float32, nf)),
            points_host = Matrix{Float64}(points),
            center_host = Matrix{Float64}(points),
            face_center_host = Matrix{Float64}(host_face_center),
            _point_copy_x = Vector{Float32}(undef, n),
            _point_copy_y = Vector{Float32}(undef, n),
            _point_copy_z = Vector{Float32}(undef, n),
            _face_copy_x = Vector{Float32}(undef, nf),
            _face_copy_y = Vector{Float32}(undef, nf),
            _face_copy_z = Vector{Float32}(undef, nf),
            _clip_stats_copy = Vector{Int32}(undef, 5 * nf),
            _dirty_cells_copy = Vector{Int32}(undef, n),
            _dirty_faces_copy = Vector{Int32}(undef, nf),
            timing = TimingAccumulator(),
            mesh_timing = MeshTimingAccumulator(),
            mesh_work = MeshWorkAccumulator())
end

function mesh_with_cell_velocity(mesh, cell_velocity)
    geom = mesh.geom
    nf = length(geom.c1)
    fvx = similar(geom.face_vx, nf)
    fvy = similar(geom.face_vy, nf)
    fvz = similar(geom.face_vz, nf)
    @inbounds for f in 1:nf
        i = Int(geom.c1[f])
        j = Int(geom.c2[f])
        if j > 0
            fvx[f] = eltype(fvx)(0.5 * (cell_velocity[i, 1] + cell_velocity[j, 1]))
            fvy[f] = eltype(fvy)(0.5 * (cell_velocity[i, 2] + cell_velocity[j, 2]))
            fvz[f] = eltype(fvz)(0.5 * (cell_velocity[i, 3] + cell_velocity[j, 3]))
        else
            fvx[f] = eltype(fvx)(cell_velocity[i, 1])
            fvy[f] = eltype(fvy)(cell_velocity[i, 2])
            fvz[f] = eltype(fvz)(cell_velocity[i, 3])
        end
    end
    moved_geom = ArepoMeshArrays3D(geom.c1, geom.c2, geom.cell_face_offsets,
                                   geom.cell_faces, geom.cell_face_signs,
                                   geom.volume, geom.face_area, geom.normal_x,
                                   geom.normal_y, geom.normal_z, fvx, fvy, fvz)
    return (; geom = moved_geom, volume = mesh.volume, center = mesh.center,
            face_center = mesh.face_center, generators = mesh.generators,
            domain = mesh.domain)
end

function same_shape(dst::ArepoMeshArrays3D, src::ArepoMeshArrays3D)
    return length(dst.c1) == length(src.c1) &&
           length(dst.cell_faces) == length(src.cell_faces) &&
           length(dst.volume) == length(src.volume)
end

function copy_mesh_to_backend!(dst::ArepoMeshArrays3D, src::ArepoMeshArrays3D)
    copyto!(dst.c1, src.c1); copyto!(dst.c2, src.c2)
    copyto!(dst.cell_face_offsets, src.cell_face_offsets)
    copyto!(dst.cell_faces, src.cell_faces)
    copyto!(dst.cell_face_signs, src.cell_face_signs)
    copyto!(dst.volume, src.volume); copyto!(dst.face_area, src.face_area)
    copyto!(dst.normal_x, src.normal_x); copyto!(dst.normal_y, src.normal_y)
    copyto!(dst.normal_z, src.normal_z)
    copyto!(dst.face_vx, src.face_vx); copyto!(dst.face_vy, src.face_vy)
    copyto!(dst.face_vz, src.face_vz)
    return dst
end

function cached_mesh_to_backend!(slot::Base.RefValue{Any}, be, mesh::ArepoMeshArrays3D)
    cached = slot[]
    if cached isa ArepoMeshArrays3D && same_shape(cached, mesh)
        return copy_mesh_to_backend!(cached, mesh)
    end
    staged = to_backend(be, mesh; T = Float32)
    slot[] = staged
    return staged
end

function cached_vector_to_backend!(slot::Base.RefValue{Any}, be, values)
    cached = slot[]
    if cached isa AbstractVector && length(cached) == length(values)
        copyto!(cached, values)
        return cached
    end
    staged = PowerFoam._backend_copy(be, values, Float32)
    slot[] = staged
    return staged
end

function copy_primitive_to_host!(host, prim)
    copyto!(host.rho, prim.rho)
    copyto!(host.vx, prim.vx)
    copyto!(host.vy, prim.vy)
    copyto!(host.vz, prim.vz)
    copyto!(host.pressure, prim.pressure)
    return host
end

function fill_mesh_velocity!(vmesh, prim_host)
    @inbounds for i in eachindex(prim_host.vx)
        vmesh[i, 1] = prim_host.vx[i]
        vmesh[i, 2] = prim_host.vy[i]
        vmesh[i, 3] = prim_host.vz[i]
    end
    return vmesh
end

function copy_backend_points_to_host!(points, center, scratch_x, scratch_y, scratch_z,
                                      px, py, pz)
    copyto!(scratch_x, px)
    copyto!(scratch_y, py)
    copyto!(scratch_z, pz)
    @inbounds for i in eachindex(scratch_x)
        x = Float64(scratch_x[i])
        y = Float64(scratch_y[i])
        z = Float64(scratch_z[i])
        points[i, 1] = x
        points[i, 2] = y
        points[i, 3] = z
        center[i, 1] = x
        center[i, 2] = y
        center[i, 3] = z
    end
    return points
end

function copy_backend_face_centers_to_host!(face_center, scratch_x, scratch_y, scratch_z,
                                            fcx, fcy, fcz)
    copyto!(scratch_x, fcx)
    copyto!(scratch_y, fcy)
    copyto!(scratch_z, fcz)
    @inbounds for f in eachindex(scratch_x)
        face_center[f, 1] = Float64(scratch_x[f])
        face_center[f, 2] = Float64(scratch_y[f])
        face_center[f, 3] = Float64(scratch_z[f])
    end
    return face_center
end

active_workspace_geom(workspace) =
    REBUILD_MODE == :gpu_compact ? workspace.compact_geom_backend[] :
    workspace.fixed_geom_backend[]

active_face_center_x(workspace) =
    REBUILD_MODE == :gpu_compact ? workspace.compact_face_center_x_backend[] :
    workspace.face_center_x_backend[]

active_face_center_y(workspace) =
    REBUILD_MODE == :gpu_compact ? workspace.compact_face_center_y_backend[] :
    workspace.face_center_y_backend[]

active_face_center_z(workspace) =
    REBUILD_MODE == :gpu_compact ? workspace.compact_face_center_z_backend[] :
    workspace.face_center_z_backend[]

function mark_motion_dirty!(be, candidate_geom, workspace)
    candidate_geom === nothing && return nothing
    _mark_motion_dirty_cells3_k!(be)(
        workspace.dirty_cells_backend,
        workspace.point_x_backend[], workspace.point_y_backend[],
        workspace.point_z_backend[],
        workspace.next_point_x_backend[], workspace.next_point_y_backend[],
        workspace.next_point_z_backend[],
        Float32(1.0), Float32(DIRTY_MOTION_THRESHOLD^2);
        ndrange = length(workspace.dirty_cells_backend))
    _mark_dirty_candidate_faces3_k!(be)(
        workspace.dirty_faces_backend, candidate_geom.c1, candidate_geom.c2,
        workspace.dirty_cells_backend;
        ndrange = length(candidate_geom.c1))
    return nothing
end

function record_mesh_work!(workspace, candidate_geom)
    MESH_WORK_STATS || return nothing
    candidate_geom === nothing && return nothing
    copyto!(workspace._clip_stats_copy, workspace.clip_stats_backend)
    copyto!(workspace._dirty_cells_copy, workspace.dirty_cells_backend)
    copyto!(workspace._dirty_faces_copy, workspace.dirty_faces_backend)
    areas = Array(candidate_geom.face_area)
    work = workspace.mesh_work
    work.refreshes += 1
    work.candidate_faces += length(candidate_geom.c1)
    work.dirty_cells += sum(workspace._dirty_cells_copy)
    work.dirty_faces += sum(workspace._dirty_faces_copy)
    work.active_faces += count(>(0), areas)
    @inbounds for f in eachindex(candidate_geom.c1)
        base = 5 * (f - 1)
        work.clip_planes += workspace._clip_stats_copy[base + 1]
        work.clip_inside += workspace._clip_stats_copy[base + 2]
        work.clip_empty += workspace._clip_stats_copy[base + 3]
        work.clip_clipped += workspace._clip_stats_copy[base + 4]
        work.tier_rejected += workspace._clip_stats_copy[base + 5]
    end
    return nothing
end

function refresh_fixed_topology_faces!(be, geom, workspace, prim)
    _refresh_fixed_topology_faces3_k!(be)(
        workspace.face_center_x_backend[], workspace.face_center_y_backend[],
        workspace.face_center_z_backend[],
        geom.normal_x, geom.normal_y, geom.normal_z,
        geom.face_vx, geom.face_vy, geom.face_vz,
        geom.c1, geom.c2,
        workspace.point_x_backend[], workspace.point_y_backend[],
        workspace.point_z_backend[],
        prim.vx, prim.vy, prim.vz, Float32(1.0);
        ndrange = length(geom.c1))
    return geom
end

function refresh_local_candidate_faces!(be, geom, workspace, prim,
                                        px = workspace.point_x_backend[],
                                        py = workspace.point_y_backend[],
                                        pz = workspace.point_z_backend[];
                                        volume_dst = geom.volume,
                                        mesh_timing = nothing)
    t0 = time_ns()
    _refresh_local_candidate_faces3_k!(be, FACE_CLIP_WORKGROUP)(
        geom.face_area,
        workspace.face_center_x_backend[], workspace.face_center_y_backend[],
        workspace.face_center_z_backend[],
        geom.normal_x, geom.normal_y, geom.normal_z,
        geom.face_vx, geom.face_vy, geom.face_vz,
        workspace.clip_stats_backend,
        geom.c1, geom.c2,
        workspace.shift_x_backend, workspace.shift_y_backend,
        workspace.shift_z_backend,
        px, py, pz,
        prim.vx, prim.vy, prim.vz, Int32(N), Float32(1.0), CLIP_SELF_IMAGES,
        PLANE_CULL, CANDIDATE_TIER_CODE;
        ndrange = length(geom.c1))
    record_mesh_timing!(be, mesh_timing, :face_clip, t0)
    t0 = time_ns()
    _update_local_candidate_volumes3_k!(be)(
        volume_dst, geom.cell_face_offsets, geom.cell_faces,
        geom.cell_face_signs, geom.face_area,
        geom.normal_x, geom.normal_y, geom.normal_z,
        workspace.face_center_x_backend[], workspace.face_center_y_backend[],
        workspace.face_center_z_backend[],
        px, py, pz, Float32(1.0);
        ndrange = length(geom.volume))
    record_mesh_timing!(be, mesh_timing, :volumes, t0)
    if ACTIVE_CELL_MODE != :off
        t0 = time_ns()
        _compact_active_cell_faces_fixed3_k!(be)(
            workspace.active_counts_backend,
            workspace.active_cell_faces_backend,
            workspace.active_cell_signs_backend,
            geom.cell_face_offsets, geom.cell_faces, geom.cell_face_signs,
            geom.face_area, Int32(workspace.active_stride);
            ndrange = length(geom.volume))
        record_mesh_timing!(be, mesh_timing, :active_cells, t0)
    end
    return geom
end

function compact_local_candidate_faces!(be, candidate_geom, workspace,
                                        compact = workspace.compact_geom_backend[],
                                        compact_fcx = workspace.compact_face_center_x_backend[],
                                        compact_fcy = workspace.compact_face_center_y_backend[],
                                        compact_fcz = workspace.compact_face_center_z_backend[];
                                        mesh_timing = nothing)
    nf = length(candidate_geom.c1)
    nc = length(candidate_geom.volume)
    nface_blocks = length(workspace.compact_face_block_counts_backend)
    ncell_blocks = length(workspace.compact_cell_block_counts_backend)
    t0 = time_ns()
    _scan_active_faces_chunks3_k!(be)(
        workspace.compact_face_prefix_backend,
        workspace.compact_face_block_counts_backend,
        candidate_geom.face_area, Int32(COMPACT_SCAN_CHUNK);
        ndrange = nface_blocks)
    _scan_counts_serial3_k!(be)(
        workspace.compact_face_block_offsets_backend,
        workspace.compact_active_total_backend,
        workspace.compact_face_block_counts_backend;
        ndrange = 1)
    record_mesh_timing!(be, mesh_timing, :face_scan, t0)
    t0 = time_ns()
    _compact_faces3_k!(be)(
        compact.c1, compact.c2, compact.face_area,
        compact.normal_x, compact.normal_y, compact.normal_z,
        compact.face_vx, compact.face_vy, compact.face_vz,
        compact_fcx, compact_fcy, compact_fcz,
        workspace.compact_face_prefix_backend,
        workspace.compact_active_total_backend,
        workspace.compact_face_block_offsets_backend,
        candidate_geom.c1, candidate_geom.c2, candidate_geom.face_area,
        candidate_geom.normal_x, candidate_geom.normal_y, candidate_geom.normal_z,
        candidate_geom.face_vx, candidate_geom.face_vy, candidate_geom.face_vz,
        workspace.face_center_x_backend[],
        workspace.face_center_y_backend[],
        workspace.face_center_z_backend[],
        Int32(COMPACT_SCAN_CHUNK);
        ndrange = nf)
    record_mesh_timing!(be, mesh_timing, :face_pack, t0)
    t0 = time_ns()
    if COMPACT_CELL_SCAN_MODE == :parallel
        _count_compact_cell_faces3_k!(be)(
            workspace.compact_cell_counts_backend,
            candidate_geom.cell_face_offsets, candidate_geom.cell_faces,
            workspace.compact_face_prefix_backend;
            ndrange = nc)
        _scan_compact_cell_count_chunks3_k!(be)(
            compact.cell_face_offsets,
            workspace.compact_cell_block_counts_backend,
            workspace.compact_cell_counts_backend,
            Int32(COMPACT_SCAN_CHUNK);
            ndrange = ncell_blocks)
    else
        _count_scan_compact_cell_chunks3_k!(be)(
            compact.cell_face_offsets,
            workspace.compact_cell_counts_backend,
            workspace.compact_cell_block_counts_backend,
            candidate_geom.cell_face_offsets, candidate_geom.cell_faces,
            workspace.compact_face_prefix_backend,
            Int32(COMPACT_SCAN_CHUNK);
            ndrange = ncell_blocks)
    end
    _scan_counts_serial3_k!(be)(
        workspace.compact_cell_block_offsets_backend,
        workspace.compact_active_total_backend,
        workspace.compact_cell_block_counts_backend;
        ndrange = 1)
    record_mesh_timing!(be, mesh_timing, :cell_scan, t0)
    t0 = time_ns()
    _finalize_offsets_fill_compact_csr3_k!(be)(
        compact.cell_faces, compact.cell_face_signs,
        compact.cell_face_offsets,
        workspace.compact_cell_counts_backend,
        workspace.compact_cell_block_offsets_backend,
        candidate_geom.cell_face_offsets, candidate_geom.cell_faces,
        candidate_geom.cell_face_signs,
        workspace.compact_face_prefix_backend,
        Int32(COMPACT_SCAN_CHUNK);
        ndrange = nc)
    record_mesh_timing!(be, mesh_timing, :csr_fill, t0)
    return compact
end

function refresh_gpu_rebuild_faces!(be, geom, workspace, prim)
    if REBUILD_MODE == :gpu_compact
        compact = workspace.compact_geom_backend[]
        candidate = refresh_local_candidate_faces!(be, geom, workspace, prim;
                                                   volume_dst = compact.volume)
        return compact_local_candidate_faces!(be, candidate, workspace, compact)
    elseif REBUILD_MODE == :gpu_local
        candidate = refresh_local_candidate_faces!(be, geom, workspace, prim)
        return candidate
    else
        return refresh_fixed_topology_faces!(be, geom, workspace, prim)
    end
end

function sync_fixed_workspace_to_host!(be, workspace)
    KernelAbstractions.synchronize(be)
    copy_backend_points_to_host!(workspace.points_host, workspace.center_host,
                                 workspace._point_copy_x, workspace._point_copy_y,
                                 workspace._point_copy_z,
                                 workspace.point_x_backend[],
                                 workspace.point_y_backend[],
                                 workspace.point_z_backend[])
    copy_backend_face_centers_to_host!(workspace.face_center_host,
                                       workspace._face_copy_x,
                                       workspace._face_copy_y,
                                       workspace._face_copy_z,
                                       active_face_center_x(workspace),
                                       active_face_center_y(workspace),
                                       active_face_center_z(workspace))
    return workspace
end

function reconstructed_native_step!(state, points, be, workspace; solver)
    if REBUILD_MODE == :gpu_compact
        return reconstructed_native_step_gpu_compact!(state, points, be, workspace;
                                                      solver)
    elseif REBUILD_MODE in (:gpu_fixed, :gpu_local)
        return reconstructed_native_step_gpu_fixed!(state, points, be, workspace;
                                                    solver)
    end
    timing = workspace.timing
    t0 = time_ns()
    prim = workspace.prim
    conserved_to_primitive_3d!(prim, state; gamma = GAMMA)
    prim_host = copy_primitive_to_host!(workspace.prim_host, prim)
    vmesh = fill_mesh_velocity!(workspace.vmesh, prim_host)
    add_timing!(timing, :primitive, t0)
    t0 = time_ns()
    old = mesh_with_cell_velocity(workspace.current_mesh[], vmesh)
    add_timing!(timing, :old_mesh, t0)
    t0 = time_ns()
    new_points = advect_generators_3d(points, vmesh, DT; boundary = :periodic)
    add_timing!(timing, :advect, t0)
    t0 = time_ns()
    new = local_periodic_voronoi_mesh_arrays_3d(new_points; T = Float32,
                                                bins_per_axis = N,
                                                search_radius = SEARCH_RADIUS)
    workspace.current_mesh[] = new
    add_timing!(timing, :new_mesh, t0)
    t0 = time_ns()
    old_geom = cached_mesh_to_backend!(workspace.old_geom_backend, be, old.geom)
    new_volume = cached_vector_to_backend!(workspace.new_volume_backend, be,
                                           new.geom.volume)
    add_timing!(timing, :staging, t0)
    t0 = time_ns()
    gradients = workspace.gradients
    calculate_gradients_from_mesh_3d!(gradients, old_geom, prim,
                                      old.center, old.face_center;
                                      box_size = 1.0, gamma = GAMMA,
                                      synchronize = false)
    add_timing!(timing, :gradients, t0)
    t0 = time_ns()
    finite_volume_reconstructed_step_3d!(state, old_geom, gradients,
                                         prim, old.center, old.face_center;
                                         dt = DT, gamma = GAMMA,
                                         riemann = solver,
                                         new_volume,
                                         box_size = 1.0)
    add_timing!(timing, :hydro, t0)
    t0 = time_ns()
    moved = (; points = new_points,
             geom = new.geom,
             state,
             mesh_velocity = vmesh,
             center = new.center,
             face_center = new.face_center)
    add_timing!(timing, :finalize, t0)
    return moved
end

function reconstructed_native_step_gpu_compact!(state, points, be, workspace; solver)
    timing = workspace.timing
    candidate_geom = workspace.fixed_geom_backend[]

    t0 = time_ns()
    prim = workspace.prim
    conserved_to_primitive_3d!(prim, state; gamma = GAMMA, synchronize = false)
    record_timing!(be, timing, :primitive, t0)

    t0 = time_ns()
    old_geom = workspace.compact_geom_backend[]
    if workspace.compact_reuse_ready[]
        _refresh_compact_face_velocities3_k!(be)(
            old_geom.face_vx, old_geom.face_vy, old_geom.face_vz,
            old_geom.c1, old_geom.c2,
            prim.vx, prim.vy, prim.vz;
            ndrange = length(old_geom.c1))
    else
        refresh_local_candidate_faces!(
            be, candidate_geom, workspace, prim;
            volume_dst = workspace.compact_geom_backend[].volume)
        old_geom = compact_local_candidate_faces!(
            be, candidate_geom, workspace,
            workspace.compact_geom_backend[],
            workspace.compact_face_center_x_backend[],
            workspace.compact_face_center_y_backend[],
            workspace.compact_face_center_z_backend[])
    end
    record_timing!(be, timing, :old_mesh, t0)

    t0 = time_ns()
    gradients = workspace.gradients
    calculate_gradients_from_mesh_3d!(
        gradients, old_geom, prim,
        workspace.point_x_backend[], workspace.point_y_backend[],
        workspace.point_z_backend[],
        workspace.compact_face_center_x_backend[],
        workspace.compact_face_center_y_backend[],
        workspace.compact_face_center_z_backend[];
        box_size = 1.0, gamma = GAMMA,
        synchronize = false)
    record_timing!(be, timing, :gradients, t0)

    t0 = time_ns()
    _advect_points_to_backend3_k!(be)(
        workspace.next_point_x_backend[], workspace.next_point_y_backend[],
        workspace.next_point_z_backend[],
        workspace.point_x_backend[], workspace.point_y_backend[],
        workspace.point_z_backend[],
        prim.vx, prim.vy, prim.vz, Float32(DT), Float32(1.0);
        ndrange = length(state.D))
    record_timing!(be, timing, :advect, t0)

    MESH_WORK_STATS && mark_motion_dirty!(be, candidate_geom, workspace)
    t0 = time_ns()
    refresh_local_candidate_faces!(
        be, candidate_geom, workspace, prim,
        workspace.next_point_x_backend[], workspace.next_point_y_backend[],
        workspace.next_point_z_backend[];
        volume_dst = workspace.compact_new_geom_backend[].volume,
        mesh_timing = workspace.mesh_timing)
    record_mesh_work!(workspace, candidate_geom)
    new_geom = compact_local_candidate_faces!(
        be, candidate_geom, workspace,
        workspace.compact_new_geom_backend[],
        workspace.compact_new_face_center_x_backend[],
        workspace.compact_new_face_center_y_backend[],
        workspace.compact_new_face_center_z_backend[];
        mesh_timing = workspace.mesh_timing)
    record_timing!(be, timing, :new_mesh, t0)

    t0 = time_ns()
    finite_volume_reconstructed_step_3d!(
        state, old_geom, gradients, prim,
        workspace.point_x_backend[], workspace.point_y_backend[],
        workspace.point_z_backend[],
        workspace.compact_face_center_x_backend[],
        workspace.compact_face_center_y_backend[],
        workspace.compact_face_center_z_backend[];
        dt = DT, gamma = GAMMA,
        riemann = solver,
        dt_extrapolation = workspace.half_dt,
        work = workspace.hydro_work,
        states = workspace.face_states,
        new_volume = new_geom.volume,
        box_size = 1.0,
        synchronize = false)
    record_timing!(be, timing, :hydro, t0)

    t0 = time_ns()
    workspace.point_x_backend[], workspace.next_point_x_backend[] =
        workspace.next_point_x_backend[], workspace.point_x_backend[]
    workspace.point_y_backend[], workspace.next_point_y_backend[] =
        workspace.next_point_y_backend[], workspace.point_y_backend[]
    workspace.point_z_backend[], workspace.next_point_z_backend[] =
        workspace.next_point_z_backend[], workspace.point_z_backend[]
    workspace.compact_geom_backend[], workspace.compact_new_geom_backend[] =
        workspace.compact_new_geom_backend[], workspace.compact_geom_backend[]
    workspace.compact_face_center_x_backend[], workspace.compact_new_face_center_x_backend[] =
        workspace.compact_new_face_center_x_backend[], workspace.compact_face_center_x_backend[]
    workspace.compact_face_center_y_backend[], workspace.compact_new_face_center_y_backend[] =
        workspace.compact_new_face_center_y_backend[], workspace.compact_face_center_y_backend[]
    workspace.compact_face_center_z_backend[], workspace.compact_new_face_center_z_backend[] =
        workspace.compact_new_face_center_z_backend[], workspace.compact_face_center_z_backend[]
    workspace.compact_reuse_ready[] = true
    DIAGNOSTICS == :step && sync_fixed_workspace_to_host!(be, workspace)
    record_timing!(be, timing, :finalize, t0)

    return (; points = workspace.points_host,
            geom = workspace.compact_geom_backend[],
            state,
            mesh_velocity = nothing,
            center = workspace.center_host,
            face_center = workspace.face_center_host)
end

function reconstructed_native_step_gpu_fixed!(state, points, be, workspace; solver)
    timing = workspace.timing
    candidate_geom = workspace.fixed_geom_backend[]
    t0 = time_ns()
    prim = workspace.prim
    conserved_to_primitive_3d!(prim, state; gamma = GAMMA, synchronize = false)
    add_timing!(timing, :primitive, t0)

    t0 = time_ns()
    geom = refresh_gpu_rebuild_faces!(be, candidate_geom, workspace, prim)
    add_timing!(timing, :old_mesh, t0)

    t0 = time_ns()
    gradients = workspace.gradients
    if REBUILD_MODE == :gpu_local && ACTIVE_CELL_MODE != :off
        calculate_gradients_from_mesh_activecells_3d!(
            gradients, geom, prim,
            workspace.point_x_backend[], workspace.point_y_backend[],
            workspace.point_z_backend[],
            active_face_center_x(workspace),
            active_face_center_y(workspace),
            active_face_center_z(workspace),
            workspace.active_counts_backend,
            workspace.active_cell_faces_backend,
            workspace.active_cell_signs_backend;
            active_stride = workspace.active_stride,
            box_size = 1.0, gamma = GAMMA,
            synchronize = false)
    else
        calculate_gradients_from_mesh_3d!(gradients, geom, prim,
                                          workspace.point_x_backend[],
                                          workspace.point_y_backend[],
                                          workspace.point_z_backend[],
                                          active_face_center_x(workspace),
                                          active_face_center_y(workspace),
                                          active_face_center_z(workspace);
                                          box_size = 1.0, gamma = GAMMA,
                                          synchronize = false)
    end
    add_timing!(timing, :gradients, t0)

    t0 = time_ns()
    if REBUILD_MODE == :gpu_local && ACTIVE_CELL_MODE == :all
        finite_volume_reconstructed_step_activecells_3d!(
            state, geom, gradients, prim,
            workspace.point_x_backend[], workspace.point_y_backend[],
            workspace.point_z_backend[],
            active_face_center_x(workspace),
            active_face_center_y(workspace),
            active_face_center_z(workspace),
            workspace.active_counts_backend,
            workspace.active_cell_faces_backend,
            workspace.active_cell_signs_backend;
            active_stride = workspace.active_stride,
            dt = DT, gamma = GAMMA,
            riemann = solver,
            dt_extrapolation = workspace.half_dt,
            work = workspace.hydro_work,
            states = workspace.face_states,
            new_volume = geom.volume,
            box_size = 1.0,
            synchronize = false)
    else
        finite_volume_reconstructed_step_3d!(state, geom, gradients,
                                             prim,
                                             workspace.point_x_backend[],
                                             workspace.point_y_backend[],
                                             workspace.point_z_backend[],
                                             active_face_center_x(workspace),
                                             active_face_center_y(workspace),
                                             active_face_center_z(workspace);
                                             dt = DT, gamma = GAMMA,
                                             riemann = solver,
                                             dt_extrapolation = workspace.half_dt,
                                             work = workspace.hydro_work,
                                             states = workspace.face_states,
                                             new_volume = geom.volume,
                                             box_size = 1.0,
                                             synchronize = false)
    end
    add_timing!(timing, :hydro, t0)

    t0 = time_ns()
    _advect_points_backend3_k!(be)(
        workspace.point_x_backend[], workspace.point_y_backend[],
        workspace.point_z_backend[], prim.vx, prim.vy, prim.vz,
        Float32(DT), Float32(1.0);
        ndrange = length(state.D))
    add_timing!(timing, :advect, t0)

    t0 = time_ns()
    geom = refresh_gpu_rebuild_faces!(be, candidate_geom, workspace, prim)
    add_timing!(timing, :new_mesh, t0)

    t0 = time_ns()
    DIAGNOSTICS == :step && sync_fixed_workspace_to_host!(be, workspace)
    add_timing!(timing, :finalize, t0)
    return (; points = workspace.points_host,
            geom,
            state,
            mesh_velocity = nothing,
            center = workspace.center_host,
            face_center = workspace.face_center_host)
end

max_abs_diff(a, b) = maximum(abs.(Array(a) .- Array(b)))
compare_states(a, b) = (; D = max_abs_diff(a.D, b.D),
                         Mx = max_abs_diff(a.Mx, b.Mx),
                         My = max_abs_diff(a.My, b.My),
                         Mz = max_abs_diff(a.Mz, b.Mz),
                         E = max_abs_diff(a.E, b.E))

function write_csv(path, summaries)
    open(path, "w") do io
        println(io, "nsteps,solver,backend,elapsed_s,step_s,speedup_vs_cpu,step,time,faces,active_faces,volume_sum,vrms,mach_rms,density_rms,rho_min,rho_max,pmin,mass,energy,dmass,denergy,point_disp_rms")
        for s in summaries
            init = s.cpu_rows[1]
            for (backend, row, elapsed, speedup) in
                (("CPU Float32", s.cpu_rows[end], s.cpu_elapsed, 1.0),
                 ("Metal Float32", s.gpu_rows[end], s.gpu_elapsed,
                  s.gpu_elapsed > 0 ? s.cpu_elapsed / s.gpu_elapsed : NaN))
                @printf(io, "%d,%s,%s,%.9g,%.9g,%.9g,%d,%.9g,%d,%d,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g\n",
                        s.nsteps, s.solver, backend, elapsed, elapsed / s.nsteps,
                        speedup, row.step, row.time,
                        row.faces, row.active_faces, row.volume_sum, row.vrms, row.mach_rms,
                        row.density_rms, row.rho_min, row.rho_max, row.pmin,
                        row.mass, row.energy, row.mass - init.mass,
                        row.energy - init.energy, row.point_disp_rms)
            end
        end
    end
end

baseline_for(summaries, nsteps) = first(filter(s -> s.nsteps == nsteps, summaries))
summary_for(summaries, nsteps, solver) =
    first(filter(s -> s.nsteps == nsteps && s.solver == solver, summaries))

function write_report(path, summaries, gpu_enabled)
    open(path, "w") do io
        println(io, "# Native 3-D moving-mesh GPU solver matrix")
        println(io)
        println(io, "This gate removes AREPO from the hydro/rebuild loop. It initializes")
        println(io, "a periodic subsonic turbulence box, rebuilds the 3-D Voronoi face")
        println(io, "table in Julia every step with the local periodic stencil producer,")
        println(io, "and runs the finite-volume update on CPU and Metal.")
        println(io)
        @printf(io, "- N: %d^3 cells\n", N)
        @printf(io, "- dt: %.8g\n", DT)
        @printf(io, "- step counts: `%s`\n", join(STEP_COUNTS, "`, `"))
        @printf(io, "- solvers: `%s`\n", join(String.(SOLVERS), "`, `"))
        @printf(io, "- local search radius: %d bin(s)\n", SEARCH_RADIUS)
        @printf(io, "- hydro order: `%s`\n", ORDER)
        @printf(io, "- rebuild mode: `%s`\n", REBUILD_MODE)
        @printf(io, "- active-cell traversal: `%s`\n", ACTIVE_CELL_MODE)
        @printf(io, "- compact scan chunk: `%d`\n", COMPACT_SCAN_CHUNK)
        @printf(io, "- compact cell scan mode: `%s`\n", COMPACT_CELL_SCAN_MODE)
        @printf(io, "- face clip workgroup: `%d`\n", FACE_CLIP_WORKGROUP)
        @printf(io, "- clip self-image planes: `%s`\n", CLIP_SELF_IMAGES)
        @printf(io, "- plane culling: `%s`\n", PLANE_CULL)
        @printf(io, "- candidate tier: `%s`\n", CANDIDATE_TIER)
        @printf(io, "- mesh work stats: `%s`\n", MESH_WORK_STATS)
        @printf(io, "- dirty motion threshold: `%.8g`\n", DIRTY_MOTION_THRESHOLD)
        @printf(io, "- synchronized phase timing: `%s`\n", SYNC_TIMING)
        @printf(io, "- new-mesh subprofile: `%s`\n", MESH_PROFILE)
        @printf(io, "- timing warmup: `%s`\n", PERF_WARMUP)
        @printf(io, "- diagnostics: `%s`\n", DIAGNOSTICS)
        @printf(io, "- Metal storage: `%s`\n", get(ENV, "POWERFOAM_METAL_STORAGE", "shared"))
        println(io)
        println(io, "## End-to-end timing")
        println(io)
        if REBUILD_MODE in (:gpu_fixed, :gpu_local, :gpu_compact)
            if REBUILD_MODE == :gpu_compact
                println(io, "Times include one initial host mesh build, backend fixed-capacity")
                println(io, "local-candidate halfspace face clipping, device-side compact")
                println(io, "face/CSR rebuild, generator advection,")
            elseif REBUILD_MODE == :gpu_local
                println(io, "Times include one initial host mesh build, backend fixed-capacity")
                println(io, "local-candidate halfspace face clipping, generator advection,")
            else
                println(io, "Times include one initial host mesh build, backend fixed-topology")
                println(io, "generator advection and face geometry refresh, final-diagnostics")
            end
            println(io, "host mirror copies when requested, and the CPU or Metal hydro kernels.")
            println(io, "They exclude the optional warmup run.")
        else
            println(io, "Times include the host-side native mesh rebuild, host/device staging,")
            println(io, "and the CPU or Metal hydro kernels. They exclude the optional warmup run.")
        end
        if DIAGNOSTICS == :final
            println(io, "Diagnostics and positivity checks are collected at the initial and final states.")
        else
            println(io, "Diagnostics and positivity checks are collected every step.")
        end
        println(io)
        println(io, "| steps | solver | CPU s | Metal s | CPU step s | Metal step s | CPU/Metal speedup |")
        println(io, "| ---: | --- | ---: | ---: | ---: | ---: | ---: |")
        for s in summaries
            speedup = s.gpu_elapsed > 0 ? s.cpu_elapsed / s.gpu_elapsed : NaN
            @printf(io, "| %d | %s | %.6g | %.6g | %.6g | %.6g | %.4g |\n",
                    s.nsteps, s.solver, s.cpu_elapsed, s.gpu_elapsed,
                    s.cpu_elapsed / s.nsteps, s.gpu_elapsed / s.nsteps, speedup)
        end
        println(io)
        println(io, "## Metal final-state comparison")
        println(io)
        println(io, "| steps | solver | faces | active_faces | volume_sum | vrms | mach_rms | density_rms | rho_min | rho_max | pmin | mass drift | energy drift | point_disp_rms |")
        println(io, "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for s in summaries
            init = s.gpu_rows[1]
            r = s.gpu_rows[end]
            @printf(io, "| %d | %s | %d | %d | %.9g | %.8g | %.8g | %.8g | %.8g | %.8g | %.8g | %.9g | %.9g | %.8g |\n",
                    s.nsteps, s.solver, r.faces, r.active_faces, r.volume_sum, r.vrms,
                    r.mach_rms, r.density_rms, r.rho_min, r.rho_max, r.pmin,
                    r.mass - init.mass, r.energy - init.energy, r.point_disp_rms)
        end
        if ORDER == :reconstruct
            println(io)
            println(io, "## Reconstructed Step Timing Breakdown")
            println(io)
            println(io, "Per-step seconds accumulated inside the reconstructed moving-mesh step.")
            println(io)
            println(io, "| steps | solver | backend | primitive | old_mesh | advect | new_mesh | connections | staging | gradients | hydro | finalize |")
            println(io, "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
            for s in summaries
                for (backend, timing) in (("CPU", s.cpu_timing), ("Metal", s.gpu_timing))
                    parts = Dict(timing_rows(timing, s.nsteps))
                    @printf(io, "| %d | %s | %s | %.6g | %.6g | %.6g | %.6g | %.6g | %.6g | %.6g | %.6g | %.6g |\n",
                            s.nsteps, s.solver, backend,
                            parts[:primitive], parts[:old_mesh], parts[:advect],
                            parts[:new_mesh], parts[:connections], parts[:staging],
                            parts[:gradients], parts[:hydro], parts[:finalize])
                end
            end
        end
        if ORDER == :reconstruct && REBUILD_MODE == :gpu_compact && MESH_WORK_STATS
            println(io)
            println(io, "## Compact New-Mesh Work Counts")
            println(io)
            println(io, "Counts are accumulated for newly advected compact rebuilds only.")
            println(io)
            println(io, "| steps | solver | backend | refreshes | candidate_faces | dirty_cells | dirty_faces | active_faces | planes/face | inside_frac | empty_frac | clipped_frac | tier_rejected |")
            println(io, "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
            for s in summaries
                for (backend, work) in (("CPU", s.cpu_mesh_work),
                                        ("Metal", s.gpu_mesh_work))
                    planes = max(work.clip_planes, 1)
                    candidates = max(work.candidate_faces, 1)
                    @printf(io, "| %d | %s | %s | %d | %d | %d | %d | %d | %.6g | %.6g | %.6g | %.6g | %d |\n",
                            s.nsteps, s.solver, backend, work.refreshes,
                            work.candidate_faces, work.dirty_cells,
                            work.dirty_faces, work.active_faces,
                            work.clip_planes / candidates,
                            work.clip_inside / planes,
                            work.clip_empty / planes,
                            work.clip_clipped / planes,
                            work.tier_rejected)
                end
            end
        end
        if ORDER == :reconstruct && REBUILD_MODE == :gpu_compact && MESH_PROFILE
            println(io)
            println(io, "## Compact New-Mesh Subphase Timing")
            println(io)
            println(io, "Per-step synchronized seconds inside the newly advected compact rebuild.")
            println(io)
            println(io, "| steps | solver | backend | face_clip | volumes | active_cells | face_scan | face_pack | cell_scan | csr_fill |")
            println(io, "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
            for s in summaries
                for (backend, timing) in (("CPU", s.cpu_mesh_timing),
                                          ("Metal", s.gpu_mesh_timing))
                    parts = Dict(mesh_timing_rows(timing, s.nsteps))
                    @printf(io, "| %d | %s | %s | %.6g | %.6g | %.6g | %.6g | %.6g | %.6g | %.6g |\n",
                            s.nsteps, s.solver, backend,
                            parts[:face_clip], parts[:volumes],
                            parts[:active_cells], parts[:face_scan],
                            parts[:face_pack], parts[:cell_scan],
                            parts[:csr_fill])
                end
            end
        end
        println(io)
        println(io, "## Metal deltas relative to first solver at each step count")
        println(io)
        println(io, "| steps | solver | dvrms | ddensity_rms | dpmin |")
        println(io, "| ---: | --- | ---: | ---: | ---: |")
        for s in summaries
            base = baseline_for(summaries, s.nsteps).gpu_rows[end]
            r = s.gpu_rows[end]
            @printf(io, "| %d | %s | %.9g | %.9g | %.9g |\n",
                    s.nsteps, s.solver, r.vrms - base.vrms,
                    r.density_rms - base.density_rms, r.pmin - base.pmin)
        end
        println(io)
        println(io, "## Observed trend")
        println(io)
        if length(SOLVERS) > 1
            first_steps = minimum(STEP_COUNTS)
            last_steps = maximum(STEP_COUNTS)
            for solver in SOLVERS[2:end]
                early = summary_for(summaries, first_steps, solver).gpu_rows[end]
                early_base = baseline_for(summaries, first_steps).gpu_rows[end]
                late = summary_for(summaries, last_steps, solver).gpu_rows[end]
                late_base = baseline_for(summaries, last_steps).gpu_rows[end]
                @printf(io, "- `%s` relative to `%s`: dvrms moves from %.9g at %d steps to %.9g at %d steps; ddensity_rms moves from %.9g to %.9g; dpmin moves from %.9g to %.9g.\n",
                        solver, first(SOLVERS),
                        early.vrms - early_base.vrms, first_steps,
                        late.vrms - late_base.vrms, last_steps,
                        early.density_rms - early_base.density_rms,
                        late.density_rms - late_base.density_rms,
                        early.pmin - early_base.pmin,
                        late.pmin - late_base.pmin)
            end
        else
            println(io, "Only one solver was requested, so no cross-solver trend is available.")
        end
        println(io)
        println(io, "## CPU/Metal final field differences")
        println(io)
        if gpu_enabled
            println(io, "| steps | solver | D | Mx | My | Mz | E |")
            println(io, "| ---: | --- | ---: | ---: | ---: | ---: | ---: |")
            for s in summaries
                d = s.diffs
                @printf(io, "| %d | %s | %.9g | %.9g | %.9g | %.9g | %.9g |\n",
                        s.nsteps, s.solver, d.D, d.Mx, d.My, d.Mz, d.E)
            end
        else
            println(io, "Metal was not available in this Julia environment.")
        end
        println(io)
        println(io, "## Interpretation boundary")
        println(io)
        if REBUILD_MODE == :gpu_compact
            println(io, "This is the first compact topology-changing GPU local rebuild")
            println(io, "baseline. It starts from the same fixed-capacity local candidate")
            println(io, "graph as `gpu_local`, clips candidate bisectors on the backend,")
            println(io, "then builds a compact active-face table and compact cell-face CSR")
            println(io, "on the device. The compact path uses chunked hierarchical scans")
            println(io, "without atomics: parallel chunk-local scans, small device scans")
            println(io, "over chunk totals, and parallel face/CSR fill. Compact volumes")
            println(io, "are written directly into the target compact geometry, stale")
            println(io, "tail clearing is fused into face compaction, cell incident")
            println(io, "counts use per-cell parallel counting plus chunk-local row scans,")
            println(io, "and final row offsets are fused with CSR row fill. The local candidate")
            println(io, "clipper uses lane-local scratch with a tuned workgroup")
            println(io, "and culls planes that leave the current polygon wholly inside")
            println(io, "or outside. Optional work counters expose dirty-cell, dirty-face,")
            println(io, "plane-cull, and candidate-tier rates so topology coherence,")
            println(io, "lattice-near stencils, incremental CSR, and hierarchical")
            println(io, "active-cell rebuilds can be judged by operation counts. It keeps")
            println(io, "separate old and new compact geometry buffers: gradients and fluxes use")
            println(io, "the old face table, while the conservative update writes into")
            println(io, "the newly advected cell volumes. After each hydro step, the")
            println(io, "new compact geometry rotates into the old slot for the next")
            println(io, "step; the next step refreshes only compact face velocities")
            println(io, "before rebuilding the newly advected geometry. That layout is")
            println(io, "the GPU-resident geometry cadence needed by hierarchical")
            println(io, "timestepping.")
        elseif REBUILD_MODE == :gpu_local
            println(io, "This is the first topology-changing GPU local rebuild rung,")
            println(io, "not yet the final compacted Voronoi tessellator. It keeps")
            println(io, "a fixed-capacity local candidate graph, clips each candidate")
            println(io, "bisector against the 3-D local periodic stencil on the backend,")
            println(io, "and represents inactive faces with zero area. Volumes are")
            println(io, "accumulated from active faces on the backend in fixed candidate")
            println(io, "storage. A fixed-stride device active-face traversal is available")
            println(io, "behind `POWERFOAM_ACTIVE_CELLS=gradients` or `all`, but the")
            println(io, "default remains the faster CSR traversal until that path beats")
            println(io, "Metal's current indirect-loop cost. Compact CSR rebuild and a")
            println(io, "dual old/new geometry path are the next pieces needed for full")
            println(io, "parity with the exact host local halfspace rebuild.")
        elseif REBUILD_MODE == :gpu_fixed
            println(io, "This is the first GPU-resident rebuild rung, not the final")
            println(io, "topology-changing Voronoi tessellator. It keeps the initial")
            println(io, "CSR connectivity, face areas, and cell volumes fixed, while")
            println(io, "the backend advects generators and refreshes face centers,")
            println(io, "normals, and mesh velocities. It is useful for exposing the")
            println(io, "remaining host/device overhead and testing the hydro path under")
            println(io, "GPU-updated moving geometry.")
        else
            println(io, "This is the native dynamic 3-D rebuild loop at the current")
            println(io, "laptop validation scale. With `ORDER=reconstruct`, it computes")
            println(io, "limited gradients from the native face table and uses the predictor")
            println(io, "before updating into the newly rebuilt volumes. The rebuild is a")
            println(io, "local periodic halfspace clipper for near-lattice turbulence meshes,")
            println(io, "not the final arbitrary point-set Delaunay tessellator.")
        end
    end
end

function main()
    mkpath(OUTDIR)
    cpu_be = KernelAbstractions.CPU()
    gpu_be = maybe_metal_backend()
    summaries = NamedTuple[]
    if PERF_WARMUP
        warmup_steps = REBUILD_MODE == :gpu_compact ? 2 : 1
        @printf("warming native moving mesh kernels: solver=%s order=%s rebuild=%s\n",
                first(SOLVERS), ORDER, REBUILD_MODE)
        run_case("CPU warmup", cpu_be; solver = first(SOLVERS),
                 nsteps = warmup_steps)
        gpu_be === nothing || run_case("Metal warmup", gpu_be;
                                       solver = first(SOLVERS),
                                       nsteps = warmup_steps)
    end
    for nsteps in STEP_COUNTS, solver in SOLVERS
        @printf("native moving mesh: solver=%s steps=%d\n", solver, nsteps)
        cpu_elapsed = @elapsed begin
            cpu_rows, cpu_state, _, _, cpu_timing, cpu_mesh_timing, cpu_mesh_work =
                run_case("CPU", cpu_be; solver, nsteps)
        end
        gpu_rows = NamedTuple[]
        diffs = (; D = NaN, Mx = NaN, My = NaN, Mz = NaN, E = NaN)
        gpu_elapsed = NaN
        gpu_timing = TimingAccumulator()
        gpu_mesh_timing = MeshTimingAccumulator()
        gpu_mesh_work = MeshWorkAccumulator()
        if gpu_be !== nothing
            gpu_elapsed = @elapsed begin
                gpu_rows, gpu_state, _, _, gpu_timing, gpu_mesh_timing, gpu_mesh_work =
                    run_case("Metal", gpu_be; solver, nsteps)
            end
            diffs = compare_states(cpu_state, gpu_state)
        else
            gpu_rows = cpu_rows
            gpu_elapsed = cpu_elapsed
            gpu_timing = cpu_timing
            gpu_mesh_timing = cpu_mesh_timing
            gpu_mesh_work = cpu_mesh_work
        end
        push!(summaries, (; nsteps, solver, cpu_rows, gpu_rows, diffs,
                          cpu_elapsed, gpu_elapsed, cpu_timing, gpu_timing,
                          cpu_mesh_timing, gpu_mesh_timing,
                          cpu_mesh_work, gpu_mesh_work))
        r = gpu_rows[end]
        @printf("%s n=%d Metal: faces=%d vrms=%.6g density_rms=%.6g pmin=%.6g speedup=%.4g\n",
                solver, nsteps, r.faces, r.vrms, r.density_rms, r.pmin,
                gpu_elapsed > 0 ? cpu_elapsed / gpu_elapsed : NaN)
    end
    csv = joinpath(OUTDIR, "solver_summary.csv")
    report = joinpath(OUTDIR, "README.md")
    write_csv(csv, summaries)
    write_report(report, summaries, gpu_be !== nothing)
    @printf("wrote %s\n", csv)
    @printf("wrote %s\n", report)
end

main()
