# particles.jl — DM particles in the block hierarchy.
#
# Particles remain ONE global device SoA (positions f32 box-normalized [0,1),
# velocities f32, equal mass — the patch_cicass convention); no per-block
# particle lists to maintain under regrid.  Two directions of coupling:
#
#   * DEPOSIT: per level, CIC clouds at the LEVEL's resolution land in the
#     owner block's `dm` pool (ghost ring included — a cloud never reaches past
#     it), then a reverse-sibling pass folds ghost mass into the true owners.
#     The level Poisson solve adds `dm` to its source.
#   * GATHER: each particle takes its acceleration from the FINEST level whose
#     block union covers it — trilinear interpolation of cell-centred central
#     differences of that level's φ pool (the ±1 taps live in the ghost ring;
#     ng = 2 is exactly deep enough).
#
# Owner lookup on device: block origins are multiples of B (lattice clustering),
# so a level's blocks live on a global (nbase·2^l/B)³ lattice; keys pack the
# lattice coords into 3×21 bits of an Int64 and a per-particle binary search
# over the sorted key table finds the slot.  This (and f32 positions) bounds the
# particle-coupled depth at l ≤ 21 − log2(nbase/B) (nbase=512, B=16: l ≤ 16;
# f32 positions give out at a similar depth) — deeper levels see DM only through
# their Dirichlet boundaries and rho_ext, which is the physically sensible
# treatment of topgrid-mass particles far below their native resolution.

# ── particle streaming (out-of-core SoA) ─────────────────────────────────────
# The global particle SoA (positions/velocities/mass/accel, ~7×4 B/particle) is
# the last device-resident chunk that caps the run at device memory.  To use the
# 2 TB host as the limit we keep the SoA host/managed and TILE every
# particle-length launch: no single kernel touches more than `particle_tile()`
# particles, so its working set fits; the managed driver auto-evicts cold tiles
# under pressure → device residency is bounded to ~one tile, not the full count.
# Tiling is per-particle-independent for kick/drift/gather (bit-identical) and
# accumulates identically (to fp round-off) for the atomic CIC deposit.
const _PTILE = Ref(0)                       # 0 ⇒ streaming off (one whole-array launch)
"Set the particle streaming tile size (0 disables tiling — one whole-array launch)."
set_particle_tile!(n::Integer) = (_PTILE[] = Int(n); nothing)
particle_tile() = _PTILE[]

"""
    ptile_ranges(np) -> iterable of contiguous UnitRanges

Particle tiles of the current `particle_tile()` size covering `1:np`
(0 or ≥ np ⇒ a single whole-array range).  Every particle-length launch — here
and in the driver's topgrid deposit — iterates these to bound its working set.
"""
function ptile_ranges(np::Integer)
    t = _PTILE[]; np = Int(np)
    (t <= 0 || t >= np) && return (1:np,)
    return (i:min(i + t - 1, np) for i in 1:t:np)
end

"Build the per-level device block-lookup (sorted packed lattice keys + slots)."
function build_block_lookup!(hier::AMRHierarchy, l::Int)
    lev = hier.levels[l + 1]
    nlat = ntuple(d -> (UInt128(hier.nbase[d]) << l) ÷ UInt128(hier.B), 3)
    all(n -> n <= UInt128(1) << 21, nlat) ||
        error("level $l block lattice exceeds the 21-bit particle-lookup key")
    keys_ = Vector{Int64}(undef, length(lev.live))
    slots = Vector{Int32}(undef, length(lev.live))
    for (i, s) in enumerate(lev.live)
        m = lev.meta[s]
        k = ntuple(d -> Int64(m.origin[d] ÷ UInt128(hier.B)), 3)
        keys_[i] = k[1] | (k[2] << 21) | (k[3] << 42)
        slots[i] = s
    end
    p = sortperm(keys_)
    lev.tabs[:pkey]  = to_device(lev.be, keys_[p], Int64)
    lev.tabs[:pslot] = to_device(lev.be, slots[p], Int32)
    return nothing
end

@inline function _lookup_slot(pkey, pslot, nkey::Int32, key::Int64)
    lo = Int32(1); hi = nkey
    while lo <= hi
        mid = (lo + hi) >> 0x01
        @inbounds km = pkey[mid]
        if km == key
            @inbounds return pslot[mid]
        elseif km < key
            lo = mid + Int32(1)
        else
            hi = mid - Int32(1)
        end
    end
    return Int32(0)
end

# per-particle CIC deposit into the owner block's dm pool (ghosts included)
@kernel function _pdeposit_k!(dm, @Const(px), @Const(py), @Const(pz),
                              @Const(pkey), @Const(pslot), nkey::Int32,
                              mass::Float32, N::Float32, B::Int32, ng::Int32,
                              nd::Int32, stride::Int32)
    p = @index(Global)
    @inbounds begin
        # containing cell (owner) and CIC cloud on cell centers
        gx = px[p] * N; gy = py[p] * N; gz = pz[p] * N
        cx = unsafe_trunc(Int32, gx); cy = unsafe_trunc(Int32, gy)
        cz = unsafe_trunc(Int32, gz)
        key = Int64(cx ÷ B) | (Int64(cy ÷ B) << 21) | (Int64(cz ÷ B) << 42)
        slot = _lookup_slot(pkey, pslot, nkey, key)
        if slot != Int32(0)
            base = (slot - Int32(1)) * stride
            hx = gx - 0.5f0; hy = gy - 0.5f0; hz = gz - 0.5f0
            i0 = unsafe_trunc(Int32, floor(hx)); fx = hx - Float32(i0)
            j0 = unsafe_trunc(Int32, floor(hy)); fy = hy - Float32(j0)
            k0 = unsafe_trunc(Int32, floor(hz)); fz = hz - Float32(k0)
            # local coords of the low corner (may sit in the ghost ring)
            li = i0 - (cx ÷ B) * B + ng
            lj = j0 - (cy ÷ B) * B + ng
            lk = k0 - (cz ÷ B) * B + ng
            for c3 in Int32(0):Int32(1), c2 in Int32(0):Int32(1), c1 in Int32(0):Int32(1)
                w = (c1 == 0 ? 1.0f0 - fx : fx) * (c2 == 0 ? 1.0f0 - fy : fy) *
                    (c3 == 0 ? 1.0f0 - fz : fz)
                idx = base + _lidx(li + c1, lj + c2, lk + c3, nd)
                KA.@atomic dm[idx] += mass * w
            end
        end
    end
end

# per-particle-MASS variant (multi-mass DM, e.g. a MUSIC/CAMB zoom IC): pm[p] is
# the particle's mass in code units, scaled to a density by invdx3 = 1/dx_l³.
# Identical CIC registration to _pdeposit_k! otherwise.
@kernel function _pdeposit_mm_k!(dm, @Const(px), @Const(py), @Const(pz), @Const(pm),
                                 @Const(pkey), @Const(pslot), nkey::Int32,
                                 invdx3::Float32, N::Float32, B::Int32, ng::Int32,
                                 nd::Int32, stride::Int32)
    p = @index(Global)
    @inbounds begin
        gx = px[p] * N; gy = py[p] * N; gz = pz[p] * N
        cx = unsafe_trunc(Int32, gx); cy = unsafe_trunc(Int32, gy); cz = unsafe_trunc(Int32, gz)
        key = Int64(cx ÷ B) | (Int64(cy ÷ B) << 21) | (Int64(cz ÷ B) << 42)
        slot = _lookup_slot(pkey, pslot, nkey, key)
        if slot != Int32(0)
            m = pm[p] * invdx3
            base = (slot - Int32(1)) * stride
            hx = gx - 0.5f0; hy = gy - 0.5f0; hz = gz - 0.5f0
            i0 = unsafe_trunc(Int32, floor(hx)); fx = hx - Float32(i0)
            j0 = unsafe_trunc(Int32, floor(hy)); fy = hy - Float32(j0)
            k0 = unsafe_trunc(Int32, floor(hz)); fz = hz - Float32(k0)
            li = i0 - (cx ÷ B) * B + ng; lj = j0 - (cy ÷ B) * B + ng; lk = k0 - (cz ÷ B) * B + ng
            for c3 in Int32(0):Int32(1), c2 in Int32(0):Int32(1), c1 in Int32(0):Int32(1)
                w = (c1 == 0 ? 1.0f0 - fx : fx) * (c2 == 0 ? 1.0f0 - fy : fy) *
                    (c3 == 0 ? 1.0f0 - fz : fz)
                idx = base + _lidx(li + c1, lj + c2, lk + c3, nd)
                KA.@atomic dm[idx] += m * w
            end
        end
    end
end

# fold ghost-ring deposits into the sibling owners (REVERSE of the ghost fill:
# read my ghost cells, atomically add into the neighbour's active cells)
@kernel function _rect_accum_rev_k!(F, @Const(jobs), @Const(cellstart),
                                    njobs::Int32, nd::Int32, stride::Int32)
    t = @index(Global)
    t0 = Int32(t) - Int32(1)
    b, ri, rj, rk = _job_decode(jobs, cellstart, njobs, t0)
    @inbounds begin
        di = jobs[b+3] + ri; dj = jobs[b+4] + rj; dk = jobs[b+5] + rk   # ghost (source now)
        si = jobs[b+6] + ri; sj = jobs[b+7] + rj; sk = jobs[b+8] + rk   # active (target now)
        gidx = (jobs[b+1] - Int32(1)) * stride + (dk * nd + dj) * nd + di + Int32(1)
        aidx = (jobs[b+2] - Int32(1)) * stride + (sk * nd + sj) * nd + si + Int32(1)
        v = F[gidx]
        v != 0.0f0 && (KA.@atomic F[aidx] += v)
    end
end

"""
    deposit_particles_level!(hier, l, parts; mass_code) -> nothing

CIC-deposit the global particle SoA into level `l`'s `dm` pool at level-l
resolution.  `mass_code` = particle mass in code density·volume units — the
deposited field is a DENSITY: mass_code/dx_l³ per unit CIC weight.  Pass a scalar
`Real` for equal-mass DM, or a per-particle device `AbstractVector` for MULTI-MASS
DM (a zoom IC).  Ghost-ring deposits are folded into sibling owners; pool zeroed first.
"""
function deposit_particles_level!(hier::AMRHierarchy, l::Int, parts; mass_code)
    lev = hier.levels[l + 1]
    isempty(lev.live) && return nothing
    haskey(lev.tabs, :pkey) || build_block_lookup!(hier, l)
    # dm is lazy: first deposit on this level allocates it (levels that never
    # deposit — l > the driver's LDM cap — never pay the cap·stride f32 pool)
    length(lev.dm) < lev.cap * lev.stride &&
        (lev.dm = device_zeros(lev.be, Float32, (lev.cap * lev.stride,)))
    fill!(lev.dm, 0.0f0)
    N = Float32(hier.nbase[1] * exp2(l))
    nkey = Int32(length(lev.live))
    for r in ptile_ranges(length(parts.px))               # stream: tile the deposit
        cnt = length(r)
        pxr = view(parts.px, r); pyr = view(parts.py, r); pzr = view(parts.pz, r)
        if mass_code isa AbstractVector                   # multi-mass DM (per particle)
            invdx3 = Float32(1.0 / level_dx(hier, l)^3)
            _pdeposit_mm_k!(lev.be)(lev.dm, pxr, pyr, pzr, view(mass_code, r),
                                    lev.tabs[:pkey], lev.tabs[:pslot], nkey, invdx3, N,
                                    Int32(hier.B), Int32(lev.ng), Int32(lev.nd),
                                    Int32(lev.stride); ndrange = cnt)
        else
            ρ1 = Float32(mass_code / level_dx(hier, l)^3)
            _pdeposit_k!(lev.be)(lev.dm, pxr, pyr, pzr,
                                 lev.tabs[:pkey], lev.tabs[:pslot], nkey, ρ1, N,
                                 Int32(hier.B), Int32(lev.ng), Int32(lev.nd),
                                 Int32(lev.stride); ndrange = cnt)
        end
    end
    sib = _tabt(lev, :sib)
    if sib.total > 0
        _rect_accum_rev_k!(lev.be)(lev.dm, sib.jobs, sib.cellstart,
                                   Int32(sib.njobs), Int32(lev.nd),
                                   Int32(lev.stride); ndrange = sib.total)
    end
    return nothing
end

# gather a = −∇φ at particle positions from ONE level's φ pool; particles whose
# containing cell has no block here keep `done = 0` for the next (coarser) level.
@kernel function _pgather_k!(ax, ay, az, done,
                             @Const(px), @Const(py), @Const(pz), @Const(phi),
                             @Const(pkey), @Const(pslot), nkey::Int32,
                             N::Float32, inv2h::Float32, B::Int32, ng::Int32,
                             nd::Int32, stride::Int32)
    p = @index(Global)
    @inbounds if done[p] == UInt8(0)
        gx = px[p] * N; gy = py[p] * N; gz = pz[p] * N
        cx = unsafe_trunc(Int32, gx); cy = unsafe_trunc(Int32, gy)
        cz = unsafe_trunc(Int32, gz)
        key = Int64(cx ÷ B) | (Int64(cy ÷ B) << 21) | (Int64(cz ÷ B) << 42)
        slot = _lookup_slot(pkey, pslot, nkey, key)
        if slot != Int32(0)
            base = (slot - Int32(1)) * stride
            hx = gx - 0.5f0; hy = gy - 0.5f0; hz = gz - 0.5f0
            i0 = unsafe_trunc(Int32, floor(hx)); fx = hx - Float32(i0)
            j0 = unsafe_trunc(Int32, floor(hy)); fy = hy - Float32(j0)
            k0 = unsafe_trunc(Int32, floor(hz)); fz = hz - Float32(k0)
            li = i0 - (cx ÷ B) * B + ng; lj = j0 - (cy ÷ B) * B + ng
            lk = k0 - (cz ÷ B) * B + ng
            aax = 0.0f0; aay = 0.0f0; aaz = 0.0f0
            for c3 in Int32(0):Int32(1), c2 in Int32(0):Int32(1), c1 in Int32(0):Int32(1)
                w = (c1 == 0 ? 1.0f0 - fx : fx) * (c2 == 0 ? 1.0f0 - fy : fy) *
                    (c3 == 0 ? 1.0f0 - fz : fz)
                idx = base + _lidx(li + c1, lj + c2, lk + c3, nd)
                aax -= w * (Float32(phi[idx + Int32(1)]) - Float32(phi[idx - Int32(1)])) * inv2h
                aay -= w * (Float32(phi[idx + nd])       - Float32(phi[idx - nd]))       * inv2h
                aaz -= w * (Float32(phi[idx + nd*nd])    - Float32(phi[idx - nd*nd]))    * inv2h
            end
            ax[p] = aax; ay[p] = aay; az[p] = aaz
            done[p] = UInt8(1)
        end
    end
end

"""
    gather_accel_particles!(hier, parts, ax, ay, az; lmax_dm = …) -> nothing

Fill `ax,ay,az` (device f32, particle-length) with −∇φ at each particle from
the FINEST level (≤ `lmax_dm`) whose block union covers it, walking fine →
coarse (level 0 covers everything in standalone hierarchies).  φ pools must be
solved and ghost-consistent.
"""
function gather_accel_particles!(hier::AMRHierarchy, parts, ax, ay, az;
                                 lmax_dm::Int = length(hier.levels) - 1)
    np = length(parts.px)
    done = device_zeros(hier.be, UInt8, (np,))            # np bytes — cheap, stays resident
    for l in min(lmax_dm, length(hier.levels) - 1):-1:0
        lev = hier.levels[l + 1]
        isempty(lev.live) && continue
        haskey(lev.tabs, :pkey) || build_block_lookup!(hier, l)
        stream_grav() && prefetch_grav!(lev)   # optional: bulk-pull this level's φ (else driver faults it in)
        N = Float32(hier.nbase[1] * exp2(l))
        inv2h = Float32(1.0 / (2.0 * level_dx(hier, l)))
        nkey = Int32(length(lev.live))
        # level-outer / tile-inner: this level's φ stays resident while every
        # particle tile gathers from it, then evicts — good φ locality under
        # streaming; the particle SoA + accel stream a tile at a time.
        for r in ptile_ranges(np)
            _pgather_k!(lev.be)(view(ax, r), view(ay, r), view(az, r), view(done, r),
                                view(parts.px, r), view(parts.py, r), view(parts.pz, r),
                                lev.phi, lev.tabs[:pkey], lev.tabs[:pslot], nkey, N, inv2h,
                                Int32(hier.B), Int32(lev.ng), Int32(lev.nd),
                                Int32(lev.stride); ndrange = length(r))
        end
        stream_grav() && evict_grav!(lev)
    end
    return nothing
end

# ── KDK push of the global SoA ────────────────────────────────────────────────
@kernel function _pkick_k!(vx, vy, vz, @Const(ax), @Const(ay), @Const(az), dt::Float32)
    p = @index(Global)
    @inbounds begin
        vx[p] += ax[p] * dt; vy[p] += ay[p] * dt; vz[p] += az[p] * dt
    end
end

@kernel function _pdrift_k!(px, py, pz, @Const(vx), @Const(vy), @Const(vz),
                            dtinvbox::Float32)
    p = @index(Global)
    @inbounds begin
        px[p] = mod(px[p] + vx[p] * dtinvbox, 1.0f0)
        py[p] = mod(py[p] + vy[p] * dtinvbox, 1.0f0)
        pz[p] = mod(pz[p] + vz[p] * dtinvbox, 1.0f0)
    end
end

"v += a·dt (device particle SoA + accel arrays); streamed a tile at a time."
function particles_kick!(hier, parts, ax, ay, az, dt)
    dtf = Float32(dt)
    for r in ptile_ranges(length(parts.px))
        _pkick_k!(hier.be)(view(parts.vx, r), view(parts.vy, r), view(parts.vz, r),
                           view(ax, r), view(ay, r), view(az, r), dtf; ndrange = length(r))
    end
    return nothing
end

"x += v·dt/box (positions box-normalized, periodic); streamed a tile at a time."
function particles_drift!(hier, parts, dt)
    c = Float32(dt / hier.box)
    for r in ptile_ranges(length(parts.px))
        _pdrift_k!(hier.be)(view(parts.px, r), view(parts.py, r), view(parts.pz, r),
                            view(parts.vx, r), view(parts.vy, r), view(parts.vz, r),
                            c; ndrange = length(r))
    end
    return nothing
end
