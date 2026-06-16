# field_ops — generic KA grid operations for the resident cosmology loop:
#   copy_field!           the baryon time-centering copy (BaryonField → OldBaryonField)
#   fill_periodic_ghosts! the root-grid periodic ghost-zone wrap (set_boundary)
# Both are device-agnostic (CPU / Metal / any KA backend) and any element type.

# ── field copy (baryon time-centering) ───────────────────────────────────────
@kernel function _copy_kernel!(dst, @Const(src))
    I = @index(Global, Cartesian)
    @inbounds dst[I] = src[I]
end

"""
    copy_field!(dst, src) -> dst

Copy `src` into `dst` (same shape, any rank/eltype) on whatever backend the
arrays live on — the device equivalent of Enzo's per-cycle `CopyBaryonFieldToOld`
time-centering memcpy. Call once per field (cheap; queue-ordered with no host
stall).
"""
function copy_field!(dst::AbstractArray, src::AbstractArray)
    size(dst) == size(src) || throw(DimensionMismatch("copy_field!: $(size(dst)) vs $(size(src))"))
    be = KA.get_backend(dst)
    _copy_kernel!(be)(dst, src; ndrange = size(dst))
    return dst
end

# ── periodic ghost-zone fill (root-grid set_boundary) ─────────────────────────
# Each cell maps to its periodic image in the active region [ng+1, ng+N] along
# every axis independently; active cells map to themselves (identity). One pass
# fills all ghosts — faces, edges AND corners — reading only active data, so it is
# safe in place (ghost writes never feed another ghost read). Requires ng ≤ N.
@kernel function _ghost_kernel!(field, ng::Int, n1::Int, n2::Int, n3::Int)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        si = i <= ng ? i + n1 : (i > ng + n1 ? i - n1 : i)
        sj = j <= ng ? j + n2 : (j > ng + n2 ? j - n2 : j)
        sk = k <= ng ? k + n3 : (k > ng + n3 ? k - n3 : k)
        field[i, j, k] = field[si, sj, sk]
    end
end

"""
    fill_periodic_ghosts!(field; ng) -> field

Fill the `ng`-deep ghost zones of the 3-D `field` by periodic wrap from the
active interior `[ng+1 : end-ng]` along each axis — Enzo's periodic
`SetExternalBoundary` for the root grid. Handles faces, edges and corners in one
launch and is safe in place. The active extent per axis must be ≥ `ng`.
"""
function fill_periodic_ghosts!(field::AbstractArray{<:Any,3}; ng::Integer)
    d1, d2, d3 = size(field)
    n1 = d1 - 2ng; n2 = d2 - 2ng; n3 = d3 - 2ng
    (n1 ≥ ng && n2 ≥ ng && n3 ≥ ng) ||
        error("fill_periodic_ghosts!: active extent < ng (ng=$ng, dims=$(size(field)))")
    be = KA.get_backend(field)
    _ghost_kernel!(be)(field, Int(ng), n1, n2, n3; ndrange = size(field))
    return field
end
