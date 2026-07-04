# topology.jl — exact-integer block topology.
#
# A level-l block's position is defined ONLY by integers: its global origin is a
# `UInt128` triple in level-l cell units (0-based, always reduced mod the level
# period P_l = nbase·2^l), derived from its parent's origin plus an integer offset
# in whole parent cells.  `UInt128` is exact far beyond 60 levels (needs
# log2(nbase)+l ≤ 127 bits); float coordinates appear NOWHERE in topology.
# Kernels never see UInt128 — the table builders (tables.jl) reduce everything to
# small Int32 rectangle jobs on the host.
#
# Periodic wrap is handled two ways, both exact:
#   * O(1) wrapped-interval overlap test (`overlap1`) for tilemap queries;
#   * shift-image enumeration (`axis_images`, s ∈ {−P,0,+P}) when actual
#     intersection rectangles are needed — this also yields the periodic
#     self-neighbor case (a block supplying its own ghosts through the wrap).

"Global block origin: level-l cell units, 0-based, reduced mod the level period."
const Origin = NTuple{3,UInt128}

# ── block flags ───────────────────────────────────────────────────────────────
const FLAG_ALIVE   = 0x01
const FLAG_NEW     = 0x02   # created this regrid (needs interior prolongation)
const FLAG_RESCALE = 0x04   # per-block f16 scale window drifted; rescale due

"""
    BlockMeta

Host metadata for one block.  `origin` is the exact global position (level cells);
`parent` is the slot in the parent level's pool (0 for level-0 blocks); `offset`
is the integer placement in whole parent active cells, o ∈ 0:B−B÷2 per axis, with
the invariant `origin == mod.(2 .* (parent_origin .+ offset), P_l)`.
"""
mutable struct BlockMeta
    origin :: Origin
    parent :: Int32
    offset :: NTuple{3,Int16}
    flags  :: UInt8
end
BlockMeta() = BlockMeta((UInt128(0), UInt128(0), UInt128(0)), Int32(0),
                        (Int16(0), Int16(0), Int16(0)), 0x00)

isalive(m::BlockMeta) = (m.flags & FLAG_ALIVE) != 0

# ── period / origin arithmetic ────────────────────────────────────────────────
"Level-l period (cells per axis): P_l = nbase·2^l, as UInt128."
level_period(nbase::NTuple{3,Int}, l::Int)::Origin =
    ntuple(d -> UInt128(nbase[d]) << l, 3)

"Child origin (level-l cells) from a level-(l−1) parent origin + integer offset."
child_origin(porigin::Origin, off::NTuple{3,Int16}, P::Origin)::Origin =
    ntuple(d -> (2 * (porigin[d] + UInt128(Int(off[d])))) % P[d], 3)

"Wrap a signed coordinate into [0, P)."
wrapc(x::Int128, P::UInt128)::UInt128 = UInt128(mod(x, Int128(P)))

# ── wrapped-interval predicates (exact, O(1)) ─────────────────────────────────
# [a,a+la) and [b,b+lb) mod P overlap  ⟺  mod(b−a,P) < la  or  mod(a−b,P) < lb.
# (Also correct when la+lb > P: the intervals then always overlap and the test
# always fires.)  All lengths are small Ints; coordinates UInt128.
@inline function overlap1(a::UInt128, la::Integer, b::UInt128, lb::Integer, P::UInt128)
    d1 = mod(Int128(b) - Int128(a), Int128(P))
    d2 = mod(Int128(a) - Int128(b), Int128(P))
    return d1 < la || d2 < lb
end

"""
    axis_images(a0, alen, b0, blen, P) -> Vector{(lo, len, shift)}

All non-empty UNWRAPPED intersections of the (possibly negative-origin) target
interval `[a0, a0+alen)` with the periodic images `[b0+s, b0+blen+s)`,
s ∈ {−P, 0, +P}.  Valid while `alen ≤ 2P` (a stored block never spans more than
two periods; asserted by the caller).  Returns Int128 lows, Int lengths.
"""
function axis_images(a0::Int128, alen::Integer, b0::Int128, blen::Integer, P::Int128)
    out = Tuple{Int128,Int,Int128}[]
    for s in (-P, Int128(0), P)
        lo = max(a0, b0 + s)
        hi = min(a0 + Int128(alen), b0 + Int128(blen) + s)
        hi > lo && push!(out, (lo, Int(hi - lo), s))
    end
    return out
end

# ── box subtraction (nesting validator support) ───────────────────────────────
# Boxes are (lo::NTuple{3,Int128}, hi::NTuple{3,Int128}) half-open, in an
# UNWRAPPED local frame.  subtract_box returns A ∖ B as ≤6 disjoint boxes.
const IBox = Tuple{NTuple{3,Int128},NTuple{3,Int128}}

function subtract_box(A::IBox, B::IBox)
    (alo, ahi) = A; (blo, bhi) = B
    # no intersection → A untouched
    for d in 1:3
        (bhi[d] <= alo[d] || blo[d] >= ahi[d]) && return IBox[A]
    end
    out = IBox[]
    lo = collect(alo); hi = collect(ahi)
    for d in 1:3
        if blo[d] > lo[d]                    # slab below B along d
            nlo = ntuple(i -> Int128(lo[i]), 3)
            nhi = ntuple(i -> i == d ? Int128(blo[d]) : Int128(hi[i]), 3)
            push!(out, (nlo, nhi)); lo[d] = blo[d]
        end
        if bhi[d] < hi[d]                    # slab above B along d
            nlo = ntuple(i -> i == d ? Int128(bhi[d]) : Int128(lo[i]), 3)
            nhi = ntuple(i -> Int128(hi[i]), 3)
            push!(out, (nlo, nhi)); hi[d] = bhi[d]
        end
    end
    return out                               # remaining core is inside B → dropped
end
