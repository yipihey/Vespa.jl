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

# species restriction: mass-weighted fraction mean over the 2³ fine cells.  The
# fine block's density scale cancels in the ratio (one src slot per job).
@kernel function _rect_restrict_sp_k!(spc, @Const(spf), @Const(Df),
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
        num = 0.0f0; den = 0.0f0
        for ck in Int32(0):Int32(1), cj in Int32(0):Int32(1), ci in Int32(0):Int32(1)
            sidx = sbase + ((sk + ck) * nd + (sj + cj)) * nd + (si + ci) + Int32(1)
            ρ̂ = Float32(Df[sidx])
            num += ρ̂ * ChemistryKernels.decode_log2sp(Float32, spf[sidx])
            den += ρ̂
        end
        spc[dbase + (dk * nd + dj) * nd + di + Int32(1)] =
            ChemistryKernels.encode_log2sp(num / max(den, 1.0f-30))
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

"Interleave the low 21 bits of the block-lattice coords (locality sort key)."
function _morton(m::BlockMeta, B::Int)
    k = ntuple(d -> UInt64(m.origin[d] ÷ UInt128(B)) & 0x1fffff, 3)
    z = UInt64(0)
    for b in 0:20
        z |= ((k[1] >> b) & 1) << (3b) | ((k[2] >> b) & 1) << (3b + 1) |
             ((k[3] >> b) & 1) << (3b + 2)
    end
    return z
end

# Slot compaction: gather old slot `perm[i]`'s full stored block (stride elements,
# ghosts included) into new slot `i` of `dst`.  Pure data movement — dst must be
# a different array (the O twin / a scratch pool).
@kernel function _permute_slots_k!(dst, @Const(src), @Const(perm), stride::Int32)
    t = @index(Global)
    t0 = Int(t) - 1
    i = t0 ÷ Int(stride) + 1               # new slot (1-based)
    r = t0 % Int(stride)
    @inbounds dst[t0 + 1] = src[(Int(perm[i]) - 1) * Int(stride) + r + 1]
end

function _permute_slots!(be, dst, src, perm_d, n::Int, stride::Int)
    n == 0 && return nothing
    _permute_slots_k!(be)(dst, src, perm_d, Int32(stride); ndrange = n * stride)
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
    # NOTE: Morton-ordering the LAUNCH list alone regressed throughput 10× —
    # slots don't move, so spatially-adjacent launch order scatters pool
    # accesses that allocation order kept contiguous.  _morton stays for the
    # future slot-COMPACTION pass (physically reordering block data at regrid).
    lev.tabs[:sib] = to_device_table(lev.be, build_sibling_jobs(lev))
    sync_block_geometry!(lev)
    delete!(lev.tabs, :pkey); delete!(lev.tabs, :pslot)   # particle lookup: lazy rebuild
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
    # NOTE: fused 6-field variants (_rect_copy6_k!/_rect_prolong6_k!) regressed
    # GPU throughput ~10× — runtime tuple indexing defeats coalescing.  Kept in
    # the file for a future revisit (compile-time unrolled variant); the
    # per-field launch train stands.
    if l >= 1
        plev = hier.levels[l]
        pro  = lev.tabs[:pro]::RectJobTable
        for (fd, fR, fO, scd, scs) in zip(dst, gasfields(plev), gasfields_o(plev),
                                          classes(lev), classes(plev))
            _run_prolong!(lev.be, pro, fd, fR, fO, Float32(θ), lev.nd, lev.stride,
                          scd, scs)
        end
        spdst = buf === :R ? lev.sp : lev.spo
        for (sd, sR, sO) in zip(spdst, plev.sp, plev.spo)
            _run_prolong!(lev.be, pro, sd, sR, sO, Float32(θ), lev.nd, lev.stride)
        end
    end
    sib = lev.tabs[:sib]::RectJobTable
    for (fd, sc) in zip(dst, classes(lev))
        _run_copy!(lev.be, sib, fd, fd, lev.nd, lev.stride, sc, sc)
    end
    for sd in (buf === :R ? lev.sp : lev.spo)
        _run_copy!(lev.be, sib, sd, sd, lev.nd, lev.stride)
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
    for (sc, sf) in zip(plev.sp, lev.sp)                # mass-weighted X mean
        res.total == 0 && continue
        _rect_restrict_sp_k!(lev.be)(sc, sf, lev.D, res.jobs, res.cellstart,
                                     Int32(res.njobs), Int32(lev.nd),
                                     Int32(lev.stride); ndrange = res.total)
    end
    return nothing
end

# ── trilinear prolongation (gravity Dirichlet BCs) ────────────────────────────
# Fine value = trilinear interpolation of the parent field at the fine cell
# CENTER: per axis the center sits at parent fraction ¼ or ¾ (weights ¾/¼ toward
# the containing parent cell and its ± neighbour, direction from fine parity).
# Non-conservative — used for POTENTIAL boundary values only.  Parent ±1 taps may
# reach the parent's ghost ring, which must be current (parent solved first).
@kernel function _rect_prolong_tl_k!(dst, @Const(src), @Const(jobs), @Const(cellstart),
                                     njobs::Int32, nd::Int32, stride::Int32)
    t = @index(Global)
    t0 = Int32(t) - Int32(1)
    b, ri, rj, rk = _job_decode(jobs, cellstart, njobs, t0)
    @inbounds begin
        di = jobs[b+3] + ri; dj = jobs[b+4] + rj; dk = jobs[b+5] + rk
        pi_ = jobs[b+6] + ((ri + jobs[b+12]) >> 0x01)
        pj_ = jobs[b+7] + ((rj + jobs[b+13]) >> 0x01)
        pk_ = jobs[b+8] + ((rk + jobs[b+14]) >> 0x01)
        d1 = ((ri + jobs[b+12]) & Int32(1)) == Int32(0) ? Int32(-1) : Int32(1)
        d2 = ((rj + jobs[b+13]) & Int32(1)) == Int32(0) ? Int32(-1) : Int32(1)
        d3 = ((rk + jobs[b+14]) & Int32(1)) == Int32(0) ? Int32(-1) : Int32(1)
        sbase = (jobs[b+2] - Int32(1)) * stride
        g(a, bq, cq) = Float32(src[sbase + ((pk_ + cq * d3) * nd + (pj_ + bq * d2)) * nd +
                                   (pi_ + a * d1) + Int32(1)])
        w(q) = q == Int32(0) ? 0.75f0 : 0.25f0
        acc = 0.0f0
        for cq in Int32(0):Int32(1), bq in Int32(0):Int32(1), aq in Int32(0):Int32(1)
            acc += w(aq) * w(bq) * w(cq) * g(aq, bq, cq)
        end
        dst[(jobs[b+1] - Int32(1)) * stride + (dk * nd + dj) * nd + di + Int32(1)] = acc
    end
end

# ── fused 6-field variants: one launch moves ALL gas fields through a table
# (the per-field launch train was the latency floor for small deep levels).
# Field index = t0 ÷ total; homogeneous-tuple indexing compiles to a select
# chain on the GPU.  Scale classes ride along per field.
@kernel function _rect_copy6_k!(F, SC, @Const(jobs), @Const(cellstart),
                                njobs::Int32, total::Int32, nd::Int32, stride::Int32)
    t = @index(Global)
    t0 = Int32(t) - Int32(1)
    fi = t0 ÷ total + Int32(1)
    b, ri, rj, rk = _job_decode(jobs, cellstart, njobs, t0 % total)
    @inbounds begin
        A = F[fi]; sc = SC[fi]
        di = jobs[b+3] + ri; dj = jobs[b+4] + rj; dk = jobs[b+5] + rk
        si = jobs[b+6] + ri; sj = jobs[b+7] + rj; sk = jobs[b+8] + rk
        ratio = sc[jobs[b+2]] / sc[jobs[b+1]]
        A[(jobs[b+1] - Int32(1)) * stride + (dk * nd + dj) * nd + di + Int32(1)] =
            _narrow(eltype(A), Float32(A[(jobs[b+2] - Int32(1)) * stride +
                                         (sk * nd + sj) * nd + si + Int32(1)]) * ratio)
    end
end

@kernel function _rect_prolong6_k!(F, FR, FO, SC, SCP, θ::Float32,
                                   @Const(jobs), @Const(cellstart),
                                   njobs::Int32, total::Int32, nd::Int32, stride::Int32)
    t = @index(Global)
    t0 = Int32(t) - Int32(1)
    fi = t0 ÷ total + Int32(1)
    b, ri, rj, rk = _job_decode(jobs, cellstart, njobs, t0 % total)
    @inbounds begin
        A = F[fi]; AR = FR[fi]; AO = FO[fi]; scd = SC[fi]; scs = SCP[fi]
        di = jobs[b+3] + ri; dj = jobs[b+4] + rj; dk = jobs[b+5] + rk
        si = jobs[b+6] + ((ri + jobs[b+12]) >> 0x01)
        sj = jobs[b+7] + ((rj + jobs[b+13]) >> 0x01)
        sk = jobs[b+8] + ((rk + jobs[b+14]) >> 0x01)
        sidx = (jobs[b+2] - Int32(1)) * stride + (sk * nd + sj) * nd + si + Int32(1)
        phys = ((1.0f0 - θ) * Float32(AR[sidx]) + θ * Float32(AO[sidx])) * scs[jobs[b+2]]
        A[(jobs[b+1] - Int32(1)) * stride + (dk * nd + dj) * nd + di + Int32(1)] =
            _narrow(eltype(A), phys / scd[jobs[b+1]])
    end
end
