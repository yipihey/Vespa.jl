# transfer.jl — batched rectangle-job kernels: one launch executes a whole
# RectJobTable (thousands of rectangles) for one field array.  A thread maps to
# (job, in-rectangle cell) by binary-searching the prefix `cellstart`; all index
# math is Int32 (host builders already reduced UInt128 topology to local offsets).
#
# Kernels are precision-generic (`eltype(dst)`) and work for f32 (validation
# phases), f16 (production; per-block scale conversion is layered on in the f16
# phase) and UInt16 species (raw copy/PC paths).

# narrowing store: plain convert for floats; round for u16 species codes (the
# θ-interpolation of log₂ codes is a geometric interpolation of fractions).
@inline _narrow(::Type{T}, v::Float32) where {T} = T(v)
@inline _narrow(::Type{Float16}, v::Float32) = Float16(clamp(v, -65504.0f0, 65504.0f0))
@inline _narrow(::Type{UInt16}, v::Float32) = UInt16(unsafe_trunc(Int32, v + 0.5f0))

# largest j (1-based) with cellstart[j] ≤ t0 (0-based cell id); cellstart[1] = 0.
@inline function _job_of(cellstart, njobs::Int32, t0::Int32)
    lo = Int32(1); hi = njobs
    while lo < hi
        mid = (lo + hi + Int32(1)) >> 0x01
        @inbounds if cellstart[mid] <= t0
            lo = mid
        else
            hi = mid - Int32(1)
        end
    end
    return lo
end

@inline function _job_decode(jobs, cellstart, njobs::Int32, t0::Int32)
    j = _job_of(cellstart, njobs, t0)
    b = (j - Int32(1)) * Int32(_JOB)
    @inbounds begin
        r0 = t0 - cellstart[j]
        e1 = jobs[b+9]; e2 = jobs[b+10]
        ri = r0 % e1; rq = r0 ÷ e1
        rj = rq % e2; rk = rq ÷ e2
        return b, ri, rj, rk
    end
end

# ── same-level copy (sibling ghost fill; also regrid overlap copies) ─────────
@kernel function _rect_copy_k!(dst, @Const(src), @Const(jobs), @Const(cellstart),
                               njobs::Int32, nd::Int32, stride::Int32)
    t = @index(Global)
    t0 = Int32(t) - Int32(1)
    b, ri, rj, rk = _job_decode(jobs, cellstart, njobs, t0)
    @inbounds begin
        di = jobs[b+3] + ri; dj = jobs[b+4] + rj; dk = jobs[b+5] + rk
        si = jobs[b+6] + ri; sj = jobs[b+7] + rj; sk = jobs[b+8] + rk
        dbase = (jobs[b+1] - Int32(1)) * stride
        sbase = (jobs[b+2] - Int32(1)) * stride
        dst[dbase + (dk * nd + dj) * nd + di + Int32(1)] =
            src[sbase + (sk * nd + sj) * nd + si + Int32(1)]
    end
end

# ── coarse→fine prolongation (piecewise-constant; θ-interpolated coarse R/O) ──
# Fine cell r maps to parent local src_off + ((r + parity) >> 1).  The coarse
# state is read as (1−θ)·R + θ·O — θ = 0 (Phase 1/2) or ½ (second fine substep
# under 2:1 subcycling).  Pass srcO = srcR when no O state exists yet.
@kernel function _rect_prolong_pc_k!(dst, @Const(srcR), @Const(srcO), θ::Float32,
                                     @Const(jobs), @Const(cellstart),
                                     njobs::Int32, nd::Int32, stride::Int32)
    t = @index(Global)
    t0 = Int32(t) - Int32(1)
    b, ri, rj, rk = _job_decode(jobs, cellstart, njobs, t0)
    @inbounds begin
        di = jobs[b+3] + ri; dj = jobs[b+4] + rj; dk = jobs[b+5] + rk
        si = jobs[b+6] + ((ri + jobs[b+12]) >> 0x01)
        sj = jobs[b+7] + ((rj + jobs[b+13]) >> 0x01)
        sk = jobs[b+8] + ((rk + jobs[b+14]) >> 0x01)
        dbase = (jobs[b+1] - Int32(1)) * stride
        sbase = (jobs[b+2] - Int32(1)) * stride
        sidx  = sbase + (sk * nd + sj) * nd + si + Int32(1)
        vR = Float32(srcR[sidx]); vO = Float32(srcO[sidx])
        dst[dbase + (dk * nd + dj) * nd + di + Int32(1)] =
            _narrow(eltype(dst), (1.0f0 - θ) * vR + θ * vO)
    end
end

# ── fine→coarse restriction (conservative 2³ mean) ───────────────────────────
@kernel function _rect_restrict_k!(dst, @Const(src), @Const(jobs), @Const(cellstart),
                                   njobs::Int32, nd::Int32, stride::Int32)
    t = @index(Global)
    t0 = Int32(t) - Int32(1)
    b, ri, rj, rk = _job_decode(jobs, cellstart, njobs, t0)   # r in COARSE cells
    @inbounds begin
        di = jobs[b+3] + ri; dj = jobs[b+4] + rj; dk = jobs[b+5] + rk
        si = jobs[b+6] + Int32(2) * ri
        sj = jobs[b+7] + Int32(2) * rj
        sk = jobs[b+8] + Int32(2) * rk
        dbase = (jobs[b+1] - Int32(1)) * stride
        sbase = (jobs[b+2] - Int32(1)) * stride
        acc = 0.0f0
        for ck in Int32(0):Int32(1), cj in Int32(0):Int32(1), ci in Int32(0):Int32(1)
            acc += Float32(src[sbase + ((sk + ck) * nd + (sj + cj)) * nd + (si + ci) + Int32(1)])
        end
        dst[dbase + (dk * nd + dj) * nd + di + Int32(1)] = eltype(dst)(acc * 0.125f0)
    end
end

# ── per-block-scaled variants (f16 phase): values cross blocks as PHYSICAL f32,
#    encoded with the destination block's power-of-two scale (exact rescaling). ──
@kernel function _rect_copy_sc_k!(dst, @Const(src), @Const(scd), @Const(scs),
                                  @Const(jobs), @Const(cellstart),
                                  njobs::Int32, nd::Int32, stride::Int32)
    t = @index(Global)
    t0 = Int32(t) - Int32(1)
    b, ri, rj, rk = _job_decode(jobs, cellstart, njobs, t0)
    @inbounds begin
        di = jobs[b+3] + ri; dj = jobs[b+4] + rj; dk = jobs[b+5] + rk
        si = jobs[b+6] + ri; sj = jobs[b+7] + rj; sk = jobs[b+8] + rk
        ratio = scs[jobs[b+2]] / scd[jobs[b+1]]
        dbase = (jobs[b+1] - Int32(1)) * stride
        sbase = (jobs[b+2] - Int32(1)) * stride
        dst[dbase + (dk * nd + dj) * nd + di + Int32(1)] =
            _narrow(eltype(dst), Float32(src[sbase + (sk * nd + sj) * nd + si + Int32(1)]) * ratio)
    end
end

@kernel function _rect_prolong_sc_k!(dst, @Const(srcR), @Const(srcO), θ::Float32,
                                     @Const(scd), @Const(scs),
                                     @Const(jobs), @Const(cellstart),
                                     njobs::Int32, nd::Int32, stride::Int32)
    t = @index(Global)
    t0 = Int32(t) - Int32(1)
    b, ri, rj, rk = _job_decode(jobs, cellstart, njobs, t0)
    @inbounds begin
        di = jobs[b+3] + ri; dj = jobs[b+4] + rj; dk = jobs[b+5] + rk
        si = jobs[b+6] + ((ri + jobs[b+12]) >> 0x01)
        sj = jobs[b+7] + ((rj + jobs[b+13]) >> 0x01)
        sk = jobs[b+8] + ((rk + jobs[b+14]) >> 0x01)
        dbase = (jobs[b+1] - Int32(1)) * stride
        sbase = (jobs[b+2] - Int32(1)) * stride
        sidx  = sbase + (sk * nd + sj) * nd + si + Int32(1)
        phys = ((1.0f0 - θ) * Float32(srcR[sidx]) + θ * Float32(srcO[sidx])) * scs[jobs[b+2]]
        dst[dbase + (dk * nd + dj) * nd + di + Int32(1)] =
            _narrow(eltype(dst), phys / scd[jobs[b+1]])
    end
end

@kernel function _rect_restrict_sc_k!(dst, @Const(src), @Const(scd), @Const(scs),
                                      @Const(jobs), @Const(cellstart),
                                      njobs::Int32, nd::Int32, stride::Int32)
    t = @index(Global)
    t0 = Int32(t) - Int32(1)
    b, ri, rj, rk = _job_decode(jobs, cellstart, njobs, t0)
    @inbounds begin
        di = jobs[b+3] + ri; dj = jobs[b+4] + rj; dk = jobs[b+5] + rk
        si = jobs[b+6] + Int32(2) * ri
        sj = jobs[b+7] + Int32(2) * rj
        sk = jobs[b+8] + Int32(2) * rk
        dbase = (jobs[b+1] - Int32(1)) * stride
        sbase = (jobs[b+2] - Int32(1)) * stride
        acc = 0.0f0
        for ck in Int32(0):Int32(1), cj in Int32(0):Int32(1), ci in Int32(0):Int32(1)
            acc += Float32(src[sbase + ((sk + ck) * nd + (sj + cj)) * nd + (si + ci) + Int32(1)])
        end
        phys = acc * 0.125f0 * scs[jobs[b+2]]
        dst[dbase + (dk * nd + dj) * nd + di + Int32(1)] =
            _narrow(eltype(dst), phys / scd[jobs[b+1]])
    end
end

# ── host orchestration ────────────────────────────────────────────────────────
function _run_copy!(be, tab::RectJobTable, dst, src, nd::Int, stride::Int,
                    scd = nothing, scs = nothing)
    tab.total == 0 && return nothing
    if scd === nothing
        _rect_copy_k!(be)(dst, src, tab.jobs, tab.cellstart, Int32(tab.njobs),
                          Int32(nd), Int32(stride); ndrange = tab.total)
    else
        _rect_copy_sc_k!(be)(dst, src, scd, scs, tab.jobs, tab.cellstart,
                             Int32(tab.njobs), Int32(nd), Int32(stride);
                             ndrange = tab.total)
    end
    return nothing
end

function _run_prolong!(be, tab::RectJobTable, dst, srcR, srcO, θ::Float32,
                       nd::Int, stride::Int, scd = nothing, scs = nothing)
    tab.total == 0 && return nothing
    if scd === nothing
        _rect_prolong_pc_k!(be)(dst, srcR, srcO, θ, tab.jobs, tab.cellstart,
                                Int32(tab.njobs), Int32(nd), Int32(stride);
                                ndrange = tab.total)
    else
        _rect_prolong_sc_k!(be)(dst, srcR, srcO, θ, scd, scs, tab.jobs,
                                tab.cellstart, Int32(tab.njobs), Int32(nd),
                                Int32(stride); ndrange = tab.total)
    end
    return nothing
end

function _run_restrict!(be, tab::RectJobTable, dst, src, nd::Int, stride::Int,
                        scd = nothing, scs = nothing)
    tab.total == 0 && return nothing
    if scd === nothing
        _rect_restrict_k!(be)(dst, src, tab.jobs, tab.cellstart, Int32(tab.njobs),
                              Int32(nd), Int32(stride); ndrange = tab.total)
    else
        _rect_restrict_sc_k!(be)(dst, src, scd, scs, tab.jobs, tab.cellstart,
                                 Int32(tab.njobs), Int32(nd), Int32(stride);
                                 ndrange = tab.total)
    end
    return nothing
end

"""
    build_level_tables!(hier, l)

(Re)build + upload the data-motion tables of AMR level `l`: `:sib` (same-level
ghost copies), and for l ≥ 1 `:pro` (ghost prolongation from l−1) and `:res`
(restriction into l−1).  Call after any topology change; tables never survive a
regrid or pool growth.
"""
function build_level_tables!(hier::AMRHierarchy, l::Int)
    lev = hier.levels[l + 1]
    lev.tabs[:sib] = to_device_table(lev.be, build_sibling_jobs(lev))
    sync_block_geometry!(lev)
    if l >= 1
        plev = hier.levels[l]
        lev.tabs[:pro] = to_device_table(lev.be, build_prolong_jobs(lev, plev))
        lev.tabs[:res] = to_device_table(lev.be, build_restrict_jobs(lev, plev))
    end
    return nothing
end

"""
    fill_ghosts!(hier, l; θ = 0, buf = :R)

Fill the ghost shells of every level-`l` block in buffer `buf` (∈ {:R, :O}):
prolong the WHOLE shell from the parent level ((1−θ)·R + θ·O of the parent —
θ = 0: parent's step-start state; θ = 1: parent's stage-1/end state; θ = ½:
midpoint under 2:1 subcycling), then overwrite the parts covered by same-level
siblings with fine data.  Species travel with the :R pass only (they are not
advected in the validation phases).
"""
function fill_ghosts!(hier::AMRHierarchy, l::Int; θ::Real = 0, buf::Symbol = :R)
    lev = hier.levels[l + 1]
    dst = buf === :R ? gasfields(lev) : gasfields_o(lev)
    if l >= 1
        plev = hier.levels[l]
        pro  = lev.tabs[:pro]::RectJobTable
        for (fd, fR, fO, scd, scs) in zip(dst, gasfields(plev), gasfields_o(plev),
                                          classes(lev), classes(plev))
            _run_prolong!(lev.be, pro, fd, fR, fO, Float32(θ), lev.nd, lev.stride,
                          scd, scs)
        end
        if buf === :R
            for (sd, sR, sO) in zip(lev.sp, plev.sp, plev.spo)
                _run_prolong!(lev.be, pro, sd, sR, sO, Float32(θ), lev.nd, lev.stride)
            end
        end
    end
    sib = lev.tabs[:sib]::RectJobTable
    for (fd, sc) in zip(dst, classes(lev))
        _run_copy!(lev.be, sib, fd, fd, lev.nd, lev.stride, sc, sc)
    end
    if buf === :R
        for sd in lev.sp
            _run_copy!(lev.be, sib, sd, sd, lev.nd, lev.stride)
        end
    end
    return nothing
end

"""
    restrict_level!(hier, l)

Conservatively restrict level-`l` (≥1) active data into the covered cells of
level l−1 (2³ mean of the R buffers into the parent's R buffers).
"""
function restrict_level!(hier::AMRHierarchy, l::Int)
    @assert l >= 1
    lev = hier.levels[l + 1]; plev = hier.levels[l]
    res = lev.tabs[:res]::RectJobTable
    for (fc, ff, scd, scs) in zip(gasfields(plev), gasfields(lev),
                                  classes(plev), classes(lev))
        _run_restrict!(lev.be, res, fc, ff, lev.nd, lev.stride, scd, scs)
    end
    # species restriction needs the u16 codec (decode→mean→encode) — f16 phase.
    return nothing
end
