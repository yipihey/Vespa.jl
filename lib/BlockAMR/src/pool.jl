# pool.jl — the per-level block pool and the hierarchy.
#
# Each Level owns one flat device array PER VARIABLE, slot-strided: block `s`'s
# field occupies elements (s−1)·stride+1 : s·stride, reshaped (nd,nd,nd) with
# ghosts inline (nd = B+2ng).  R→O double buffering (`D` … `Ge` vs `Do` … `Geo`):
# the hydro reads R and writes O; the intact R state is what makes recompute-reflux
# exact and coarse-ghost time interpolation free.  Slots are recycled through a
# free-list; growth reallocates (all device job tables are rebuilt after any
# regrid, never held across growth).
#
# Per-block f32 scale tables (Dsc/Ssc/Esc) live here from day one but stay 1.0
# until the f16 phase — the f32 validation phases run through the same code path.

"Aligned slot stride (elements): (B+2ng)³ rounded up to a multiple of 64."
_slot_stride(nd::Int) = ((nd^3 + 63) ÷ 64) * 64

mutable struct Level{V<:AbstractVector,U<:AbstractVector{UInt16},
                     F<:AbstractVector{Float32},I<:AbstractVector{Int32}}
    l        :: Int
    B        :: Int
    ng       :: Int
    nd       :: Int                      # B + 2ng
    stride   :: Int                      # aligned nd³
    P        :: Origin                   # level period (cells/axis)
    th       :: Int                      # tilemap tile size = B ÷ 2 (level cells)
    cap      :: Int
    nsp      :: Int
    be       :: Any
    # ── host topology ──
    meta     :: Vector{BlockMeta}
    live     :: Vector{Int32}            # dense launch-order slot list
    freelist :: Vector{Int32}
    tilemap  :: Dict{Origin,Vector{Int32}}   # tile coord → slots whose grown bbox overlaps
    byorigin :: Dict{Origin,Int32}
    # ── device state: R buffers + O ping-pong twins ──
    D::V;  S1::V;  S2::V;  S3::V;  Tau::V;  Ge::V
    Do::V; S1o::V; S2o::V; S3o::V; Tauo::V; Geo::V
    sp       :: Vector{U}
    spo      :: Vector{U}
    # ── per-block power-of-two f32 scales (identity until the f16 phase) ──
    Dsc::F; Ssc::F; Esc::F
    # ── gravity: level potential + rhs + particle-deposit pools (f32, ghosts inline) ──
    phi::F; rhs::F; dm::F
    live_d   :: I                        # device copy of `live`
    tabs     :: Dict{Symbol,Any}         # device RectJobTables; rebuilt at regrid only
end

# typed table accessors: `tabs` is Dict{Symbol,Any}, so bare lookups make every
# downstream kernel launch a dynamic dispatch (~1 ms/launch measured on the
# gravity solve path).  The Level type parameters pin the concrete types.
@inline _tabt(lev::Level{V,U,F,I}, k::Symbol) where {V,U,F,I} =
    lev.tabs[k]::RectJobTable{I}
@inline _tabi(lev::Level{V,U,F,I}, k::Symbol) where {V,U,F,I} = lev.tabs[k]::I
@inline _tabf(lev::Level{V,U,F,I}, k::Symbol) where {V,U,F,I} = lev.tabs[k]::F

gasfields(lev::Level)   = (lev.D, lev.S1, lev.S2, lev.S3, lev.Tau, lev.Ge)
gasfields_o(lev::Level) = (lev.Do, lev.S1o, lev.S2o, lev.S3o, lev.Tauo, lev.Geo)
"Per-field scale-class vectors, field order (D,S1,S2,S3,Tau,Ge)."
classes(lev::Level)     = (lev.Dsc, lev.Ssc, lev.Ssc, lev.Ssc, lev.Esc, lev.Esc)

"Swap the R and O gas/species buffers (after a level substep)."
function swap_buffers!(lev::Level)
    lev.D, lev.Do = lev.Do, lev.D;   lev.S1, lev.S1o = lev.S1o, lev.S1
    lev.S2, lev.S2o = lev.S2o, lev.S2; lev.S3, lev.S3o = lev.S3o, lev.S3
    lev.Tau, lev.Tauo = lev.Tauo, lev.Tau; lev.Ge, lev.Geo = lev.Geo, lev.Ge
    lev.sp, lev.spo = lev.spo, lev.sp
    return nothing
end

function Level(; l::Int, B::Int, ng::Int = 2, nbase::NTuple{3,Int}, be,
                 T::Type = Float32, nsp::Int = 0, cap0::Int = 8)
    @assert B % 2 == 0 && B >= 2ng "B must be even and ≥ 2ng"
    @assert all(nbase .% B .== 0) "nbase must be divisible by B"
    nd = B + 2ng; stride = _slot_stride(nd)
    P  = level_period(nbase, l); th = B ÷ 2
    zed() = device_zeros(be, T, (cap0 * stride,))
    zu()  = device_zeros(be, UInt16, (cap0 * stride,))
    ones32(n) = to_device(be, ones(Float32, n), Float32)
    lev = Level{typeof(zed()),typeof(zu()),typeof(ones32(1)),typeof(to_device(be, Int32[], Int32))}(
        l, B, ng, nd, stride, P, th, cap0, nsp, be,
        [BlockMeta() for _ in 1:cap0], Int32[], Int32[],
        Dict{Origin,Vector{Int32}}(), Dict{Origin,Int32}(),
        zed(), zed(), zed(), zed(), zed(), zed(),
        zed(), zed(), zed(), zed(), zed(), zed(),
        [zu() for _ in 1:nsp], [zu() for _ in 1:nsp],
        ones32(cap0), ones32(cap0), ones32(cap0),
        device_zeros(be, Float32, (cap0 * stride,)),
        device_zeros(be, Float32, (cap0 * stride,)),
        device_zeros(be, Float32, (0,)),     # dm: lazy — only levels that deposit pay for it
        to_device(be, Int32[], Int32), Dict{Symbol,Any}())
    # out-of-core: pools are born host-preferred so allocation + checkpoint
    # restore land in host RAM (not device) — prefetch_level! pulls the working
    # set to the GPU per advance.  No-op unless memory_mode() === :managed.
    advise_level_host!(lev)
    return lev
end

# ── slot management ───────────────────────────────────────────────────────────
"Grow every device array of `lev` to `newcap` slots, preserving existing data."
function _grow!(lev::Level, newcap::Int)
    T = eltype(lev.D); n_old = lev.cap * lev.stride; n_new = newcap * lev.stride
    # finalize the old array NOW: at production scale (20+ GB of pools) waiting
    # for GC to reclaim each superseded field is what turns a 1 GB grow into an
    # OOM (stream-ordered, so the pending copy completes first).
    grow(a) = (b = device_zeros(lev.be, eltype(a), (n_new,));
               copyto!(view(b, 1:n_old), view(a, 1:n_old)); finalize(a); b)
    lev.D  = grow(lev.D);  lev.S1 = grow(lev.S1); lev.S2 = grow(lev.S2)
    lev.S3 = grow(lev.S3); lev.Tau = grow(lev.Tau); lev.Ge = grow(lev.Ge)
    lev.Do = grow(lev.Do); lev.S1o = grow(lev.S1o); lev.S2o = grow(lev.S2o)
    lev.S3o = grow(lev.S3o); lev.Tauo = grow(lev.Tauo); lev.Geo = grow(lev.Geo)
    lev.sp  = [grow(a) for a in lev.sp]
    lev.spo = [grow(a) for a in lev.spo]
    growsc(a) = (b = to_device(lev.be, ones(Float32, newcap), Float32);
                 copyto!(view(b, 1:lev.cap), view(a, 1:lev.cap)); b)
    lev.Dsc = growsc(lev.Dsc); lev.Ssc = growsc(lev.Ssc); lev.Esc = growsc(lev.Esc)
    lev.phi = grow(lev.phi); lev.rhs = grow(lev.rhs)
    length(lev.dm) > 0 && (lev.dm = grow(lev.dm))      # dm is lazy (deposit levels only)
    append!(lev.meta, [BlockMeta() for _ in 1:(newcap - lev.cap)])
    lev.cap = newcap
    advise_level_host!(lev)          # out-of-core: grown pools stay host-preferred
    return nothing
end

"""
    reserve!(lev, cap)

Pre-grow a level's pools to `cap` slots in ONE allocation. Restoring a large
checkpoint otherwise triggers ~log₁.₂₅(N) incremental `_grow!`s, each copying
the growing pool — and under managed oversubscription those unified→unified
copies thrash the pager to a standstill. Pre-sizing to the known final count
makes the restore a single clean allocation per level.
"""
reserve!(lev::Level, cap::Int) = (cap > lev.cap && _grow!(lev, cap); lev)

function alloc_slot!(lev::Level)::Int32
    isempty(lev.freelist) || return pop!(lev.freelist)
    s = Int32(length(lev.live) + length(lev.freelist) + 1)
    # 1.25× growth: 2× doubling at production caps (~65k slots × ~8k stride)
    # overshoots by tens of GB and tipped the 256³ lmax=6 run into OOM.
    s > lev.cap && _grow!(lev, max(cld(5 * lev.cap, 4), Int(s)))
    return s
end

"Refresh the device copy of the live-slot list (after any live-list change)."
sync_live!(lev::Level) = (lev.live_d = to_device(lev.be, lev.live, Int32); nothing)

# ── tilemap registration ─────────────────────────────────────────────────────
"Tiles (wrapped) covered by the ghost-grown bbox [origin−ng, origin+B+ng)."
function _covered_tiles(f, lev::Level, origin::Origin)
    nt = ntuple(d -> Int(lev.P[d] ÷ lev.th), 3)          # tiles per axis (P divisible by th)
    rng = ntuple(3) do d
        lo = Int128(origin[d]) - lev.ng
        hi = Int128(origin[d]) + lev.B + lev.ng - 1
        fld(lo, lev.th):fld(hi, lev.th)
    end
    for tk in rng[3], tj in rng[2], ti in rng[1]
        f((wrapc(Int128(ti), UInt128(nt[1])), wrapc(Int128(tj), UInt128(nt[2])),
           wrapc(Int128(tk), UInt128(nt[3]))))
    end
end

function _register!(lev::Level, s::Int32)
    m = lev.meta[s]
    _covered_tiles(lev, m.origin) do key
        push!(get!(Vector{Int32}, lev.tilemap, key), s)
    end
    lev.byorigin[m.origin] = s
    return nothing
end

function _unregister!(lev::Level, s::Int32)
    m = lev.meta[s]
    _covered_tiles(lev, m.origin) do key
        v = get(lev.tilemap, key, nothing)
        v === nothing && return
        deleteat!(v, findall(==(s), v))
        isempty(v) && delete!(lev.tilemap, key)
    end
    delete!(lev.byorigin, m.origin)
    return nothing
end

"""
    lattice_neighbors(lev, origin, B) -> Vector{Int32}

Live slots of the ≤26 B-lattice neighbours of the block at `origin` (all block
origins are lattice-aligned — the clusterer guarantee — so neighbour lookup is
26 O(1) `byorigin` hits instead of a tilemap walk).  Includes the block itself
in the periodic self-wrap sense the sibling builder expects (caller filters).
"""
function lattice_neighbors(lev::Level, origin::Origin, B::Int)
    out = Int32[]
    for dk in -1:1, dj in -1:1, di in -1:1
        o = (wrapc(Int128(origin[1]) + di * B, lev.P[1]),
             wrapc(Int128(origin[2]) + dj * B, lev.P[2]),
             wrapc(Int128(origin[3]) + dk * B, lev.P[3]))
        s = get(lev.byorigin, o, Int32(0))
        s != 0 && !(s in out) && push!(out, s)
    end
    return out
end

"O(1) owner of the block-lattice cell containing level cell `q` (0 if none)."
lattice_owner(lev::Level, q::NTuple{3,UInt128}, B::Int) =
    get(lev.byorigin, ntuple(d -> (q[d] ÷ UInt128(B)) * UInt128(B), 3), Int32(0))

"""
    overlapping_blocks(lev, lo, len) -> Vector{Int32}

Live slots whose ACTIVE region [origin, origin+B) overlaps the (wrapped) query
box `[lo, lo+len)` given in level-`lev.l` cells (`lo` signed Int128, may be
negative or beyond the period).  Exact; periodic.
"""
function overlapping_blocks(lev::Level, lo::NTuple{3,Int128}, len::NTuple{3,Int})
    nt  = ntuple(d -> Int(lev.P[d] ÷ lev.th), 3)
    out = Int32[]; seen = Set{Int32}()
    rng = ntuple(d -> fld(lo[d], lev.th):fld(lo[d] + len[d] - 1, lev.th), 3)
    low = ntuple(d -> wrapc(lo[d], lev.P[d]), 3)
    for tk in rng[3], tj in rng[2], ti in rng[1]
        key = (wrapc(Int128(ti), UInt128(nt[1])), wrapc(Int128(tj), UInt128(nt[2])),
               wrapc(Int128(tk), UInt128(nt[3])))
        for s in get(lev.tilemap, key, Int32[])
            s in seen && continue
            m = lev.meta[s]
            ok = true
            for d in 1:3
                overlap1(low[d], len[d], m.origin[d], lev.B, lev.P[d]) || (ok = false; break)
            end
            ok && (push!(out, s); push!(seen, s))
        end
    end
    return out
end

# ── the hierarchy ─────────────────────────────────────────────────────────────
mutable struct AMRHierarchy{L<:Level}
    levels :: Vector{L}                  # levels[l+1] = AMR level l
    B      :: Int
    ng     :: Int
    nbase  :: NTuple{3,Int}
    box    :: Float64
    Lcap   :: Int
    be     :: Any
    besym  :: Symbol
    gamma  :: Float64
    cfl    :: Float64
    λ      :: Float32
    nstep  :: Vector{Int64}              # per-level substep counters (exact time)
    scheme :: Symbol                     # hydro integrator: :rk2 (oracle) | :ctu (perf)
end

level_dx(hier::AMRHierarchy, l::Int) = hier.box / (hier.nbase[1] * exp2(l))
lmax(hier::AMRHierarchy) = findlast(lv -> !isempty(lv.live), hier.levels) - 1

function AMRHierarchy(; nbase::NTuple{3,Int}, box::Real = 1.0, B::Int = 16,
                        ng::Int = 2, backend::Symbol = :cpu, T::Type = Float32,
                        nsp::Int = 0, Lcap::Int = 60, gamma::Real = 5/3,
                        cfl::Real = 0.4, cap0::Int = 8, scheme::Symbol = :rk2)
    @assert scheme in (:rk2, :ctu) "scheme must be :rk2 or :ctu"
    be = BlockAMR.backend(backend)
    lev0 = Level(; l = 0, B, ng, nbase, be, T, nsp, cap0)
    AMRHierarchy{typeof(lev0)}([lev0], B, ng, nbase, Float64(box), Lcap, be, backend,
                               Float64(gamma), Float64(cfl), 1.0f0, Int64[0], scheme)
end

"Extend `hier.levels` (and `nstep`) so AMR level `l` exists."
function ensure_level!(hier::AMRHierarchy, l::Int)
    @assert l <= hier.Lcap "level $l exceeds Lcap=$(hier.Lcap)"
    while length(hier.levels) < l + 1
        ln = length(hier.levels)             # next AMR level index
        push!(hier.levels, Level(; l = ln, B = hier.B, ng = hier.ng,
                                   nbase = hier.nbase, be = hier.be,
                                   T = eltype(hier.levels[1].D),
                                   nsp = hier.levels[1].nsp, cap0 = 8))
        push!(hier.nstep, 0)
    end
    return hier.levels[l + 1]
end

"Tile all of level 0 with (nbase÷B)³ base blocks (standalone runs; topgrid shadow later)."
function init_base_level!(hier::AMRHierarchy)
    lev = hier.levels[1]
    nb  = hier.nbase .÷ hier.B
    for k in 0:nb[3]-1, j in 0:nb[2]-1, i in 0:nb[1]-1
        org = (UInt128(i * hier.B), UInt128(j * hier.B), UInt128(k * hier.B))
        s = alloc_slot!(lev)
        lev.meta[s] = BlockMeta(org, Int32(0), (Int16(0), Int16(0), Int16(0)), FLAG_ALIVE)
        push!(lev.live, s); _register!(lev, s)
    end
    sync_live!(lev)
    return lev
end

"""
    add_block!(hier, l, parent_slot, offset) -> slot

Create a level-`l` (≥1) block placed at integer `offset` (whole parent active
cells, each component in `0:B−B÷2`) inside parent block `parent_slot` of level
`l−1`.  The child covers B÷2 parent cells and B³ own cells.
"""
function add_block!(hier::AMRHierarchy, l::Int, parent_slot::Integer,
                    offset::NTuple{3,<:Integer})
    @assert l >= 1 "level-0 blocks come from init_base_level!/the topgrid shadow"
    lev  = ensure_level!(hier, l)
    plev = hier.levels[l]
    Bh   = hier.B - hier.B ÷ 2
    all(0 .<= offset .<= Bh) || error("offset $offset outside 0:$Bh")
    pm  = plev.meta[Int32(parent_slot)]
    @assert isalive(pm) "parent slot $parent_slot is not alive"
    off = ntuple(d -> Int16(offset[d]), 3)
    org = child_origin(pm.origin, off, lev.P)
    haskey(lev.byorigin, org) && error("block at origin $org already exists on level $l")
    s = alloc_slot!(lev)
    lev.meta[s] = BlockMeta(org, Int32(parent_slot), off, FLAG_ALIVE | FLAG_NEW)
    push!(lev.live, s); _register!(lev, s); sync_live!(lev)
    return s
end

function remove_block!(hier::AMRHierarchy, l::Int, slot::Integer)
    lev = hier.levels[l + 1]
    s = Int32(slot)
    @assert isalive(lev.meta[s])
    _unregister!(lev, s)
    lev.meta[s].flags = 0x00
    deleteat!(lev.live, findall(==(s), lev.live))
    push!(lev.freelist, s); sync_live!(lev)
    return nothing
end

# ── views + nesting validator ─────────────────────────────────────────────────
"3-D view of one block's field (`:D`, `:S1`, …, `:Ge`, `:Do`, …) — host or device."
function blockview(lev::Level, slot::Integer, f::Symbol)
    a = getfield(lev, f)
    r = (Int(slot) - 1) * lev.stride
    reshape(view(a, r+1:r+lev.nd^3), lev.nd, lev.nd, lev.nd)
end

"""
    check_nesting(hier) -> Bool

Validate the proper-nesting invariant: every level-l (≥1) block's active region,
grown by 1 parent cell per side, is covered by the union of level-(l−1) active
regions.  (⇒ fine ghosts always prolong from l−1 only; no interface level jumps.)
"""
function check_nesting(hier::AMRHierarchy; verbose::Bool = false)
    for l in 1:length(hier.levels)-1
        lev = hier.levels[l + 1]; plev = hier.levels[l]
        for s in lev.live
            m = lev.meta[s]
            lo = ntuple(d -> fld(Int128(m.origin[d]), 2) - 1, 3)
            hi = ntuple(d -> cld(Int128(m.origin[d]) + lev.B, 2) + 1, 3)
            boxes = IBox[(lo, hi)]
            plen  = ntuple(d -> Int(hi[d] - lo[d]), 3)
            for ps in overlapping_blocks(plev, lo, plen)
                pm = plev.meta[ps]
                # subtract every periodic image of the parent's active region
                imgs = ntuple(d -> axis_images(lo[d], plen[d], Int128(pm.origin[d]),
                                               plev.B, Int128(plev.P[d])), 3)
                for i1 in imgs[1], i2 in imgs[2], i3 in imgs[3]
                    blo = (i1[1], i2[1], i3[1])
                    bhi = (i1[1] + i1[2], i2[1] + i2[2], i3[1] + i3[2])
                    boxes = reduce(vcat, (subtract_box(bx, (blo, bhi)) for bx in boxes);
                                   init = IBox[])
                end
                isempty(boxes) && break
            end
            if !isempty(boxes)
                verbose && @warn "nesting violation" level=l slot=s origin=m.origin uncovered=boxes
                return false
            end
        end
    end
    return true
end

# ── recycled-slot hygiene ─────────────────────────────────────────────────────
@kernel function _zero_slots_k!(A, @Const(slots), stride::Int32)
    t = @index(Global)
    t0 = Int32(t) - Int32(1)
    b = t0 ÷ stride; c = t0 % stride
    @inbounds A[(slots[b + Int32(1)] - Int32(1)) * stride + c + Int32(1)] =
        zero(eltype(A))
end

"""
    zero_slots!(lev, slots)

Zero the FULL slot storage (ghosts included) of `slots` in every pool of `lev`.
A recycled freelist slot carries a DEAD block's f16 data; between block creation
and the deferred table rebuild there are readers that see it raw — the news
prolongation stencil reads one cell into the PARENT's ghost ring (a new parent's
ring is recycled garbage when parent and child levels rebuild in the same
regrid), and the children's Dirichlet φ refill reads parent φ ghosts likewise.
"""
function zero_slots!(lev::Level, slots::Vector{Int32})
    isempty(slots) && return nothing
    sd = to_device(lev.be, slots, Int32)
    n = length(slots) * lev.stride
    for A in (lev.D, lev.S1, lev.S2, lev.S3, lev.Tau, lev.Ge,
              lev.Do, lev.S1o, lev.S2o, lev.S3o, lev.Tauo, lev.Geo,
              lev.phi, lev.rhs, lev.dm)
        length(A) == 0 && continue
        _zero_slots_k!(lev.be)(A, sd, Int32(lev.stride); ndrange = n)
    end
    for A in lev.sp
        _zero_slots_k!(lev.be)(A, sd, Int32(lev.stride); ndrange = n)
    end
    for A in lev.spo
        _zero_slots_k!(lev.be)(A, sd, Int32(lev.stride); ndrange = n)
    end
    return nothing
end

# ── memory accounting (out-of-core Phase 0 instrumentation) ───────────────────
"Resident bytes of one level's field pools, broken out by class."
function level_bytes(lev::Level)
    b(a) = length(a) * sizeof(eltype(a))
    gas  = sum(b, gasfields(lev)) + sum(b, gasfields_o(lev))     # 12 f16 pools
    spb  = sum(b, lev.sp; init = 0) + sum(b, lev.spo; init = 0)  # 2·nsp u16 pools
    grav = b(lev.phi) + b(lev.rhs) + b(lev.dm)                   # f32 phi/rhs/dm
    sc   = b(lev.Dsc) + b(lev.Ssc) + b(lev.Esc) + b(lev.live_d)  # per-block, tiny
    tabs = sum((t isa RectJobTable ? b(t.jobs) + b(t.cellstart) : 0
                for t in values(lev.tabs)); init = 0)
    return (gas = gas, species = spb, gravity = grav, scales = sc, tables = tabs,
            total = gas + spb + grav + sc + tabs)
end

"""
    memory_report(hier; io = stdout)

Print resident pool bytes per level (gas / species / gravity / tables) and the
grand total — the Phase-0 lens on where the GPU's memory goes at each redshift.
Returns the grand total in bytes.
"""
function memory_report(hier::AMRHierarchy; io = stdout)
    GB = 1 / 2^30
    tot = (gas = 0, species = 0, gravity = 0, scales = 0, tables = 0, total = 0)
    println(io, "── BlockAMR memory report (mode=$(memory_mode())) ──")
    @printf(io, "%3s %8s %8s | %8s %8s %8s %8s | %9s\n",
            "L", "blocks", "cap", "gas", "species", "gravity", "tables", "total GB")
    for l in 0:length(hier.levels)-1
        lev = hier.levels[l + 1]
        isempty(lev.live) && lev.cap <= 8 && continue
        m = level_bytes(lev)
        tot = map(+, tot, m)
        @printf(io, "%3d %8d %8d | %7.2fG %7.2fG %7.2fG %7.2fG | %9.3f\n",
                l, length(lev.live), lev.cap, m.gas * GB, m.species * GB,
                m.gravity * GB, m.tables * GB, m.total * GB)
    end
    @printf(io, "%3s %8s %8s | %7.2fG %7.2fG %7.2fG %7.2fG | %9.3f\n",
            "Σ", "", "", tot.gas * GB, tot.species * GB, tot.gravity * GB,
            tot.tables * GB, tot.total * GB)
    flush(io)
    return tot.total
end

# ── managed-memory staging (out-of-core Phase 0.5) ────────────────────────────
# All field pools of a level, in one tuple (dm/sp/spo may be empty).
_field_pools(lev::Level) = (lev.D, lev.S1, lev.S2, lev.S3, lev.Tau, lev.Ge,
                            lev.Do, lev.S1o, lev.S2o, lev.S3o, lev.Tauo, lev.Geo,
                            lev.phi, lev.rhs, lev.dm, lev.sp..., lev.spo...)

"Prefetch a level's field pools + device tables onto the GPU (managed mode only)."
function prefetch_level!(lev::Level)
    memory_mode() === :managed || return lev
    for a in _field_pools(lev); prefetch_device!(a); end
    for t in values(lev.tabs)
        t isa RectJobTable || continue
        prefetch_device!(t.jobs); prefetch_device!(t.cellstart)
    end
    prefetch_device!(lev.live_d)
    prefetch_device!(lev.Dsc); prefetch_device!(lev.Ssc); prefetch_device!(lev.Esc)
    return lev
end

"Mark a level's field pools as host-resident by default (managed mode only)."
function advise_level_host!(lev::Level)
    memory_mode() === :managed || return lev
    for a in _field_pools(lev); advise_host!(a); end
    return lev
end

"Advise every level's pools host-resident — the default home under oversubscription."
advise_all_host!(hier::AMRHierarchy) =
    (foreach(advise_level_host!, hier.levels); hier)
