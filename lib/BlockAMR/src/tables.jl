# tables.jl — Int32 rectangle-job tables: ALL cross-block data motion (same-level
# ghost copies, coarse→fine prolongation, fine→coarse restriction) is described by
# host-built tables of small rectangle pairs; batched kernels (transfer.jl) execute
# one table in one launch.  Every UInt128 reduces to an Int32 local offset here —
# kernels never see wide integers.
#
# Job encoding (16 × Int32 per job, padded for alignment):
#   [1] dst_slot  [2] src_slot
#   [3..5]  dst_off (0-based, in the dst block's STORED frame, incl. ghost frame)
#   [6..8]  src_off (0-based, stored frame of src)
#   [9..11] ext     (rectangle extent, dst-resolution cells)
#   [12..14] parity (prolong only: fine-origin oddness per axis)
#   [15..16] pad
# `cellstart` is the 0-based exclusive prefix sum of prod(ext) — kernels binary-
# search it to map a global thread id to (job, in-rectangle index).

const _JOB = 16

struct RectJobTable{I<:AbstractVector{Int32}}
    njobs     :: Int
    total     :: Int
    jobs      :: I          # _JOB · njobs
    cellstart :: I          # njobs + 1, 0-based exclusive prefix
end

RectJobTable() = RectJobTable(0, 0, Int32[], Int32[0])

"Upload a host-built table to backend `be` (no-op payloads for empty tables)."
to_device_table(be, t::RectJobTable) =
    RectJobTable(t.njobs, t.total, to_device(be, t.jobs, Int32),
                 to_device(be, t.cellstart, Int32))

# host-side accumulation
mutable struct _JobAcc
    jobs :: Vector{Int32}
    cs   :: Vector{Int32}
    tot  :: Int
end
_JobAcc() = _JobAcc(Int32[], Int32[0], 0)

function _push_job!(a::_JobAcc, dst::Integer, src::Integer,
                    doff::NTuple{3,<:Integer}, soff::NTuple{3,<:Integer},
                    ext::NTuple{3,<:Integer},
                    par::NTuple{3,<:Integer} = (0, 0, 0))
    append!(a.jobs, Int32[dst, src, doff..., soff..., ext..., par..., 0, 0])
    a.tot += prod(Int.(ext))
    push!(a.cs, Int32(a.tot))
    return nothing
end

_finish(a::_JobAcc) = RectJobTable(length(a.cs) - 1, a.tot, a.jobs, a.cs)

# ── same-level sibling ghost copies ──────────────────────────────────────────
"""
    build_sibling_jobs(lev) -> RectJobTable (host)

For every live block: rectangles where its ghost shell is covered by another
live block's ACTIVE region (or its own, through the periodic wrap).  Emitted in
dst/src stored-frame offsets; the identity image (src == dst, zero shift) is the
block's own interior and is skipped.  Same-level blocks are disjoint, so every
emitted destination cell is a ghost cell.
"""
function build_sibling_jobs(lev::Level)
    acc = _JobAcc(); B = lev.B; ng = lev.ng
    for s in lev.live
        m = lev.meta[s]
        lo  = ntuple(d -> Int128(m.origin[d]) - ng, 3)     # stored-frame global lo
        cands = lattice_neighbors(lev, m.origin, B)        # O(1) hits, lattice-aligned
        for c in cands
            cm = lev.meta[c]
            imgs = ntuple(d -> axis_images(lo[d], lev.nd, Int128(cm.origin[d]), B,
                                           Int128(lev.P[d])), 3)
            for i1 in imgs[1], i2 in imgs[2], i3 in imgs[3]
                c == s && i1[3] == 0 && i2[3] == 0 && i3[3] == 0 && continue
                glo  = (i1[1], i2[1], i3[1])
                ext  = (i1[2], i2[2], i3[2])
                doff = ntuple(d -> Int(glo[d] - lo[d]), 3)
                soff = ntuple(d -> Int(glo[d] - (i1, i2, i3)[d][3] -
                                       (Int128(cm.origin[d]) - ng)), 3)
                _push_job!(acc, s, c, doff, soff, ext)
            end
        end
    end
    return _finish(acc)
end

# ── coarse→fine prolongation rectangles ──────────────────────────────────────
# Decompose a fine block's target region (ghost shell, or full interior for new
# blocks) into rectangles, each sourced from exactly ONE coarse block image (the
# nesting invariant guarantees coverage).  Fine cell g → parent cell fld(g,2);
# in-kernel: parent_local = src_off + ((r + parity) >> 1).
"""
    build_prolong_jobs(lev, plev; interior=false) -> RectJobTable

Ghost-shell (default) or full-interior (`interior=true`, for freshly created
blocks) prolongation jobs for every live block of `lev` (level l) sourced from
`plev` (level l−1).  Errors if the nesting invariant is violated.
"""
function build_prolong_jobs(lev::Level, plev::Level; interior::Bool = false,
                            only_slots = nothing)
    acc = _JobAcc(); B = lev.B; ng = lev.ng
    slots = only_slots === nothing ? lev.live : only_slots
    for s in slots
        m = lev.meta[s]
        lo = ntuple(d -> Int128(m.origin[d]) - ng, 3)
        rects = if interior
            [(ntuple(d -> Int128(m.origin[d]), 3), ntuple(_ -> B, 3))]
        else
            _ghost_shell_rects(m.origin, B, ng)
        end
        for (rlo, rext) in rects
            _emit_prolong!(acc, s, rlo, rext, lo, plev)
        end
    end
    return _finish(acc)
end

"The 26 ghost-shell rectangles of a block (3×3×3 segment lattice minus the core)."
function _ghost_shell_rects(origin::Origin, B::Int, ng::Int)
    seg(d) = ((Int128(origin[d]) - ng, ng), (Int128(origin[d]), B),
              (Int128(origin[d]) + B, ng))
    out = Tuple{NTuple{3,Int128},NTuple{3,Int}}[]
    s1, s2, s3 = seg(1), seg(2), seg(3)
    for k in 1:3, j in 1:3, i in 1:3
        (i == 2 && j == 2 && k == 2) && continue
        lo  = (s1[i][1], s2[j][1], s3[k][1])
        ext = (s1[i][2], s2[j][2], s3[k][2])
        push!(out, (lo, ext))
    end
    return out
end

function _emit_prolong!(acc::_JobAcc, s::Int32, rlo::NTuple{3,Int128},
                        rext::NTuple{3,Int}, dlo::NTuple{3,Int128}, plev::Level)
    # parent-cell footprint of the fine rectangle
    plo  = ntuple(d -> fld(rlo[d], 2), 3)
    phi  = ntuple(d -> cld(rlo[d] + rext[d], 2), 3)
    plen = ntuple(d -> Int(phi[d] - plo[d]), 3)
    covered = 0
    for ps in overlapping_blocks(plev, plo, plen)
        pm = plev.meta[ps]
        imgs = ntuple(d -> axis_images(plo[d], plen[d], Int128(pm.origin[d]),
                                       plev.B, Int128(plev.P[d])), 3)
        for i1 in imgs[1], i2 in imgs[2], i3 in imgs[3]
            img = (i1, i2, i3)
            # fine cells whose parent lies in this image, clipped to the rectangle
            flo = ntuple(d -> max(rlo[d], 2 * img[d][1]), 3)
            fhi = ntuple(d -> min(rlo[d] + rext[d], 2 * (img[d][1] + img[d][2])), 3)
            all(fhi .> flo) || continue
            ext  = ntuple(d -> Int(fhi[d] - flo[d]), 3)
            doff = ntuple(d -> Int(flo[d] - dlo[d]), 3)
            # src stored-frame offset of parent cell fld(flo,2) (shift back the image)
            soff = ntuple(d -> Int(fld(flo[d], 2) - img[d][3] -
                                   (Int128(pm.origin[d]) - plev.ng)), 3)
            par  = ntuple(d -> Int(mod(flo[d], 2)), 3)
            _push_job!(acc, s, ps, doff, soff, ext, par)
            covered += prod(ext)
        end
    end
    covered == prod(rext) ||
        error("prolongation under-covered (nesting violation): block slot $s rect $rlo/$rext")
    return nothing
end

# ── fine→coarse restriction ──────────────────────────────────────────────────
"""
    build_restrict_jobs(lev, plev) -> RectJobTable

One job per live fine block: its whole active region restricted (2³ average)
into the covered cells of its parent (dst_off = ng .+ offset; ext in COARSE
cells; kernel reads fine cells at src_off + 2r + {0,1}³).
"""
function build_restrict_jobs(lev::Level, plev::Level)
    acc = _JobAcc()
    Bh = lev.B ÷ 2
    for s in lev.live
        m = lev.meta[s]
        doff = ntuple(d -> plev.ng + Int(m.offset[d]), 3)
        _push_job!(acc, m.parent, s, doff, ntuple(_ -> lev.ng, 3),
                   ntuple(_ -> Bh, 3))
    end
    return _finish(acc)
end
