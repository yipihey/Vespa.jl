# Particle (dark-matter) self-gravity on the composite mesh (Phase 2 of the native
# re-platform). Dark matter is the dominant mass in cosmology, so the Poisson source
# is ρ_gas + ρ_DM. Particles carry mass + a comoving position/velocity; each solve
# step they are CIC-deposited to a per-leaf density field that is ADDED to the gas
# density in the existing composite Poisson RHS (`src/gravity.jl`), the same
# across-level CG solve gravity already uses. The gravitational acceleration is then
# interpolated back to the particles and they are pushed (KDK) — see `particle_push`.
#
# This file is the CPU correctness path (mirroring `gravity.jl`'s plain-Julia CG);
# the KA/GPU deposit+interp+push (reusing PoissonKernels) is the P2.3 follow-up, to
# be parity-tested against this. Deposit/interpolation here are CELL-CENTERED CIC on
# a UNIFORM (single-level) leaf mesh; AMR point-location (tree descent) is P2.1d.

"""
    ParticleSet{T}

Dark-matter particles as struct-of-arrays: comoving positions `px,py,pz`,
velocities `vx,vy,vz`, and per-particle mass `m`. `rho_p` is a PLAIN, ordinal-indexed
density vector (NOT a backend field view) holding the cloud-deposited particle
density for the Poisson source — entry `i` is the density of the `i`-th leaf in
`for_each_cell` order (= the cached locator's ordinal = the Poisson struct ordinal),
so the deposit writes a plain `Vector` (allocation-free) and the RHS reads it by a
`for_each_cell` counter. Attach with `enable_particles!`; index by leaf handle via
[`particle_density`](@ref).
"""
mutable struct ParticleSet{Tf}
    px::Vector{Float64}; py::Vector{Float64}; pz::Vector{Float64}   # POSITIONS: f64 (until integer-based)
    vx::Vector{Tf}; vy::Vector{Tf}; vz::Vector{Tf}                  # velocities: field precision (f32 ok)
    m::Vector{Tf}                                                  # masses: field precision
    rho_p::Vector{Tf}                                             # ordinal-indexed deposited density (Tf)
    loc::Any                                                       # cached locator (rebuilt on regrid)
    dev::Any                                                       # cached DEVICE locator arrays (KA push; regrid-invalidated)
    dev_pos::Any                                                   # push-scoped device (dpx,dpy,dpz); deposit reads it instead of ps.px
end

"""
    enable_particles!(sim; px, py, pz, vx=0, vy=0, vz=0, m) -> ParticleSet

Attach a dark-matter `ParticleSet` to `sim` (positions/velocities/masses as
length-N vectors; scalar velocity args broadcast to zero by default). Allocates the
deposit-density store on the backend and sets `sim.particles`, so `solve_poisson!`
includes the particle density in the gravitational source.
"""
function enable_particles!(sim::Simulation; px, py, pz,
                           vx = nothing, vy = nothing, vz = nothing, m)
    T = _Tf(sim)
    n = length(px)
    cv(a) = a === nothing ? zeros(T, n) : convert(Vector{T}, a)   # velocities/mass at field precision
    f64(a) = convert(Vector{Float64}, a)                          # positions stay f64
    # rho_p is a plain ordinal-indexed density vector, sized lazily in the deposit.
    ps = ParticleSet{T}(f64(px), f64(py), f64(pz), cv(vx), cv(vy), cv(vz), convert(Vector{T}, m),
                        T[], nothing, nothing, nothing)
    sim.particles = ps
    return ps
end

# The cell-index locator is expensive to build (walks all leaves + a Dict), so it
# is cached on the ParticleSet and reused while the mesh is unchanged. Call
# `invalidate_particle_locator!` after a regrid (P2.1d wires this into `regrid!`).
function _locator(sim::Simulation, ps::ParticleSet)
    ps.loc === nothing && (ps.loc = _overlap_locator(sim.backend))
    return ps.loc
end
invalidate_particle_locator!(ps::ParticleSet) = (ps.loc = nothing; ps.dev = nothing; nothing)

# ── cloud-overlap locator (uniform AND AMR) ──────────────────────────────────
# A particle's "cloud" is a cube of side = its home leaf's width, centered on the
# particle. Its mass is partitioned among leaves in proportion to the cloud↔leaf
# volume overlap. Because leaves tile space with no gaps, Σ overlaps = cloud volume
# ⇒ mass is conserved EXACTLY on any leaf set (uniform or refined). On a uniform
# mesh the overlap weights are exactly the trilinear CIC weights, so this reduces
# to standard CIC. Using the SAME overlap weights to interpolate g back keeps the
# scheme momentum-conserving. Periodic wrap is handled per axis on the circle.
#
# A coarse bucket grid (bucket width = the coarsest leaf width) indexes leaves by
# their center for the candidate query: a cloud (≤ coarsest width) touches ≤ 2^N
# buckets, and every leaf lives in exactly one bucket, so candidates are found in
# O(1) with no double counting. Backend-agnostic (only cell_center/width/volume).
# FLAT, ordinal-indexed locator: leaf boxes in `blo`/`bhi`/`vol` (column = leaf
# ordinal, in `for_each_cell` order — the SAME ordinal the Poisson struct uses), and
# a bucket→leaf CSR (`bptr`/`bleaf`). No `Dict`, no per-leaf struct, so the host
# overlap loops are allocation-free, and every array uploads directly to the GPU for
# the device deposit/interp kernels.
struct FlatLocator{N,H}
    lo::NTuple{N,Float64}
    L::NTuple{N,Float64}                        # period length per axis
    bw::NTuple{N,Float64}                        # bucket width (coarsest leaf width)
    nbk::NTuple{N,Int}                           # bucket dims
    nbucket::Int
    handles::Vector{H}                           # ordinal → leaf handle
    blo::Matrix{Float64}                         # (N, n) leaf box lo
    bhi::Matrix{Float64}                         # (N, n) leaf box hi
    vol::Vector{Float64}                         # (n,) leaf volume
    bptr::Vector{Int32}                          # bucket CSR rowptr (nbucket+1, 1-based)
    bleaf::Vector{Int32}                         # leaf ordinals grouped by bucket
    ordmap::Dict{H,Int}                          # leaf handle → ordinal (for particle_density by handle)
end

function _overlap_locator(b)
    N = rank(b)
    dom = domain(b)
    lo = ntuple(d -> Float64(dom[d][1]), Val(N))
    hi = ntuple(d -> Float64(dom[d][2]), Val(N))
    L = ntuple(d -> hi[d] - lo[d], Val(N))
    handles = Any[]
    maxw = Ref(ntuple(_ -> 0.0, Val(N)))
    for_each_cell(b) do c
        push!(handles, c)
        w = cell_width(b, c)
        maxw[] = ntuple(d -> max(maxw[][d], Float64(w[d])), Val(N))
        return nothing
    end
    H = isempty(handles) ? Any : typeof(handles[1])
    handlesH = convert(Vector{H}, handles)
    n = length(handlesH)
    bw = maxw[]
    nbk = ntuple(d -> max(1, round(Int, L[d] / bw[d])), Val(N))
    nbucket = prod(nbk)
    blo = Matrix{Float64}(undef, N, n); bhi = Matrix{Float64}(undef, N, n)
    vol = Vector{Float64}(undef, n); homebk = Vector{Int}(undef, n)
    @inbounds for i in 1:n
        h = handlesH[i]; ctr = cell_center(b, h); w = cell_width(b, h)
        bi = 1; stride = 1
        for d in 1:N
            blo[d, i] = Float64(ctr[d]) - 0.5 * Float64(w[d])
            bhi[d, i] = Float64(ctr[d]) + 0.5 * Float64(w[d])
            bk = clamp(floor(Int, (Float64(ctr[d]) - lo[d]) / bw[d]), 0, nbk[d] - 1)
            bi += bk * stride; stride *= nbk[d]
        end
        vol[i] = Float64(cell_volume(b, h)); homebk[i] = bi
    end
    # bucket CSR (counting sort by home bucket)
    bptr = zeros(Int32, nbucket + 1)
    @inbounds for i in 1:n
        bptr[homebk[i] + 1] += 1
    end
    bptr[1] = 1
    @inbounds for k in 1:nbucket
        bptr[k + 1] += bptr[k]
    end
    bleaf = Vector{Int32}(undef, n); cur = copy(bptr)
    @inbounds for i in 1:n
        k = homebk[i]; bleaf[cur[k]] = Int32(i); cur[k] += 1
    end
    ordmap = Dict{H,Int}()                                       # handle → ordinal (particle_density)
    sizehint!(ordmap, n)
    @inbounds for i in 1:n
        ordmap[handlesH[i]] = i
    end
    return FlatLocator{N,H}(lo, L, bw, nbk, nbucket, handlesH, blo, bhi, vol, bptr, bleaf, ordmap)
end

# 1-D overlap length of cloud interval [a0,a1] with leaf interval [b0,b1] on the
# circle of circumference Lp (cloud and leaf are each ≤ Lp, so k∈{−1,0,1} suffices).
@inline function _ov1d(a0, a1, b0, b1, Lp)
    s = 0.0
    @inbounds for k in -1:1
        lo = max(a0 + k * Lp, b0); hi = min(a1 + k * Lp, b1)
        hi > lo && (s += hi - lo)
    end
    return s
end

# Home-leaf ORDINAL for point x (periodic), searched in x's bucket. 0 if none.
# Allocation-free: scalar bucket index, no tuple/closure.
@inline function _home_ord(loc::FlatLocator{N}, x) where {N}
    bi = 1; stride = 1
    @inbounds for d in 1:N
        bk = clamp(floor(Int, (x[d] - loc.lo[d]) / loc.bw[d]), 0, loc.nbk[d] - 1)
        bi += bk * stride; stride *= loc.nbk[d]
    end
    @inbounds for k in loc.bptr[bi]:(loc.bptr[bi + 1] - 1)
        ord = loc.bleaf[k]; inside = true
        for d in 1:N
            δ = mod(x[d] - loc.blo[d, ord], loc.L[d])
            (δ >= 0 && δ < loc.bhi[d, ord] - loc.blo[d, ord]) || (inside = false; break)
        end
        inside && return Int(ord)
    end
    return loc.bptr[bi] < loc.bptr[bi + 1] ? Int(loc.bleaf[loc.bptr[bi]]) : 0   # fallback
end

# N-D overlap fraction of the cloud [clo,chi] (volume cvol) with leaf `ord`.
@inline function _cloud_leaf_frac(loc::FlatLocator{N}, clo, chi, cvol, ord) where {N}
    ov = 1.0
    @inbounds for d in 1:N
        ov *= _ov1d(clo[d], chi[d], loc.blo[d, ord], loc.bhi[d, ord], loc.L[d])
        ov == 0.0 && return 0.0
    end
    return ov / cvol
end

# Number of candidate buckets the cloud spans, and a scalar decode of the t-th one
# (0-based t) into a 1-based bucket index — both allocation-free (plain loops, no
# closures), so the deposit/interp/CSR loops over them don't allocate.
@inline function _cloud_nbuckets(loc::FlatLocator{N}, clo, chi) where {N}
    nc = 1
    @inbounds for d in 1:N
        k0 = floor(Int, (clo[d] - loc.lo[d]) / loc.bw[d])
        k1 = floor(Int, (chi[d] - loc.lo[d]) / loc.bw[d])
        nc *= (k1 - k0 + 1)
    end
    return nc
end
@inline function _cloud_bucket(loc::FlatLocator{N}, clo, chi, t) where {N}
    bi = 1; stride = 1; r = t
    @inbounds for d in 1:N
        k0 = floor(Int, (clo[d] - loc.lo[d]) / loc.bw[d])
        k1 = floor(Int, (chi[d] - loc.lo[d]) / loc.bw[d])
        span = k1 - k0 + 1
        off = r % span; r ÷= span
        bk = mod(k0 + off, loc.nbk[d])
        bi += bk * stride; stride *= loc.nbk[d]
    end
    return bi
end

# ── deposit: particle mass → per-leaf density via cloud overlap ──────────────
# `_locator` returns an `Any`-typed cached struct, so the hot loop runs behind a
# function barrier (`_deposit_overlap!`) specialized on the concrete `FlatLocator`
# — without it every `loc.*` field access is type-unstable and allocates.
# Overloaded by `VespaKAParticlesExt`: runs the cloud-overlap deposit as a KA kernel
# (atomic scatter to a device `rho_p`), then copies back to the host `ps.rho_p`. The
# per-particle overlap geometry (the host bottleneck at scale — 308M allocs / 5 s at
# 64³) moves to the device; only the O(n_leaf) host RHS read remains. `nothing` ⇒ no ext.
function _deposit_particles_ka! end

function deposit_particle_density!(sim::Simulation, ps::ParticleSet)
    grav = sim.grav
    if grav !== nothing && grav.ka !== nothing            # GPU Poisson path ⇒ device deposit
        return _deposit_particles_ka!(sim, ps, grav.ka)
    end
    loc = _locator(sim, ps)
    _deposit_overlap!(ps, loc)                            # function barrier (concrete ps + loc)
    return nothing
end

# 3-D explicit (no closures, no Any indexing) — the particle path is always 3-D.
# Deposits into the PLAIN ordinal `ps.rho_p` (sized lazily to the leaf count); the
# overlap loop writes `rho_p[ord]` directly — no backend field view, so it is
# allocation-free (the host analogue of the device deposit). The barrier is
# specialized on `ParticleSet{T}` + `FlatLocator{3}` so every field access is typed.
function _deposit_overlap!(ps::ParticleSet{T}, loc::FlatLocator{3}) where {T}
    rho_p = ps.rho_p
    resize!(rho_p, length(loc.vol)); fill!(rho_p, zero(T))   # ordinal-indexed, for_each_cell order
    px = ps.px; py = ps.py; pz = ps.pz; mass = ps.m
    @inbounds for p in eachindex(mass)
        x = (Float64(px[p]), Float64(py[p]), Float64(pz[p]))
        home = _home_ord(loc, x)
        home == 0 && continue
        s = (loc.bhi[1, home] - loc.blo[1, home], loc.bhi[2, home] - loc.blo[2, home],
             loc.bhi[3, home] - loc.blo[3, home])
        clo = (x[1] - 0.5s[1], x[2] - 0.5s[2], x[3] - 0.5s[3])
        chi = (x[1] + 0.5s[1], x[2] + 0.5s[2], x[3] + 0.5s[3])
        cvol = s[1] * s[2] * s[3]; mp = Float64(mass[p])
        for t in 0:(_cloud_nbuckets(loc, clo, chi) - 1)
            bi = _cloud_bucket(loc, clo, chi, t)
            for k in loc.bptr[bi]:(loc.bptr[bi + 1] - 1)
                ord = loc.bleaf[k]
                frac = _cloud_leaf_frac(loc, clo, chi, cvol, ord)
                frac > 0.0 && (rho_p[ord] += T(mp * frac / loc.vol[ord]))
            end
        end
    end
    return nothing
end

"""
    particle_density(sim, cell) -> Tf

Deposited dark-matter density at a leaf `cell` (by handle), read from the
ordinal-indexed `rho_p` via the cached locator's `ordmap`. The locator must be
valid (it is during regrid's indicator pass, before `ps.loc` is invalidated, and
after any `deposit_particle_density!`). Use this — not a backend field view — as the
refinement indicator on DM overdensity.
"""
@inline particle_density(sim::Simulation, cell) =
    (ps = sim.particles; ps.rho_p[ps.loc.ordmap[cell]])

# Total deposited particle mass Σ ρ_p·V (the conservation check / diagnostic).
# rho_p is ordinal-indexed (for_each_cell order), aligned with the locator's vol.
function particle_deposited_mass(sim::Simulation, ps::ParticleSet)
    loc = _locator(sim, ps)
    s = 0.0
    @inbounds for i in eachindex(ps.rho_p)
        s += Float64(ps.rho_p[i]) * loc.vol[i]
    end
    return s
end

# ── interpolate g = −∇φ to the particles (CIC, SAME kernel as the deposit) ────
# Gathering with the same cell-centered CIC stencil used for the deposit, combined
# with the antisymmetric central-difference −∇φ (`gravity_accel`), makes the
# total particle force Σ m_p g_p vanish to round-off — momentum conservation, the
# defining property of a particle-mesh scheme. Returns per-particle (ax,ay,az).
function interp_gravity_to_particles!(sim::Simulation, ps::ParticleSet, grav::GravityField)
    T = _Tf(sim)
    loc = _locator(sim, ps)
    np = length(ps.m)
    ax = zeros(T, np); ay = zeros(T, np); az = zeros(T, np)
    _interp_overlap!(ax, ay, az, loc, ps.px, ps.py, ps.pz, sim, grav, T)
    return ax, ay, az
end

function _interp_overlap!(ax, ay, az, loc::FlatLocator{N}, px, py, pz, sim, grav, ::Type{T}) where {N,T}
    pos = (px, py, pz)
    @inbounds for p in eachindex(ax)
        x = ntuple(d -> Float64(pos[d][p]), Val(N))
        home = _home_ord(loc, x)
        home == 0 && continue
        s = ntuple(d -> loc.bhi[d, home] - loc.blo[d, home], Val(N))
        clo = ntuple(d -> x[d] - 0.5 * s[d], Val(N))
        chi = ntuple(d -> x[d] + 0.5 * s[d], Val(N))
        cvol = prod(s)
        gx = 0.0; gy = 0.0; gz = 0.0
        for t in 0:(_cloud_nbuckets(loc, clo, chi) - 1)
            bi = _cloud_bucket(loc, clo, chi, t)
            for k in loc.bptr[bi]:(loc.bptr[bi + 1] - 1)
                ord = loc.bleaf[k]
                frac = _cloud_leaf_frac(loc, clo, chi, cvol, ord)
                frac == 0.0 && continue
                gc = gravity_accel(sim, grav, loc.handles[ord])
                gx += frac * Float64(gc[1]); gy += frac * Float64(gc[2]); gz += frac * Float64(gc[3])
            end
        end
        ax[p] = T(gx); ay[p] = T(gy); az[p] = T(gz)
    end
    return nothing
end

# Total particle momentum Σ m v per axis (the conservation diagnostic / gate).
function particle_momentum(ps::ParticleSet)
    return (sum(ps.m .* ps.vx), sum(ps.m .* ps.vy), sum(ps.m .* ps.vz))
end

# ── KDK push: half-kick → drift → (re-solve) → half-kick ─────────────────────
# Leapfrog (kick-drift-kick) with the gravitational acceleration interpolated from
# the potential. `solve_poisson!` must have run for the CURRENT positions before
# the call (the driver does this). Non-cosmological here (a≡1); the comoving
# expansion factors (`particle_kick!`/`particle_drift!` coefficients) are Phase 3.
# Positions wrap periodically into the domain.
# Overloaded by `VespaKAParticlesExt` (load KernelAbstractions): runs the KDK
# kinematics (interp-gather + kick + drift) on the KA backend `ka`. The cloud
# weights and per-leaf g are still built on the host (`_particle_csr`/`_leaf_gravity`);
# moving those to the device is a perf follow-up — the result is parity-exact.
function _push_particles_ka! end

# Comoving KDK coefficients at the n+½ expansion state (a, ȧ). The acceleration
# enters as g/a, the drift as dt/a, and the peculiar velocity feels the
# semi-implicit Hubble drag with c = ½(ȧ/a)·h — exactly Enzo's particle scheme and
# consistent with the gas (`apply_expansion_terms!`). `cosmo === nothing` gives
# (a,ȧ)=(1,0): drift dt, kick v += g·h — the ordinary non-cosmological push.
@inline function _cosmo_ah(sim::Simulation, dt::Real)
    sim.cosmo === nothing && return (1.0, 0.0)
    return expansion_at(sim.cosmo, sim.cosmo.t_initial + sim.t + 0.5 * dt)
end

function push_particles!(sim::Simulation, dt::Real; ka = nothing)
    ka === nothing || return _push_particles_ka!(sim, dt, ka)
    ps = sim.particles
    grav = sim.grav
    grav === nothing && return nothing
    N = rank(sim.backend)
    dom = domain(sim.backend)
    # positions + time/expansion stay f64 (drift acts on f64 positions); only the
    # velocity STORAGE is field precision (f32). The kick coefficients are f64 and
    # promote the f32 velocity/accel, storing the result back at f32.
    box = ntuple(d -> Float64(dom[d][2] - dom[d][1]), N)
    lo = ntuple(d -> Float64(dom[d][1]), N)
    a, dadt = _cosmo_ah(sim, dt)
    h = 0.5 * dt; inv_a = 1.0 / a; cc = 0.5 * (dadt / a) * (0.5 * dt); drift_dt = dt / a

    ax, ay, az = interp_gravity_to_particles!(sim, ps, grav)
    _kick_comoving!(ps.vx, ax, h, inv_a, cc)             # semi-implicit half-kick
    _kick_comoving!(ps.vy, ay, h, inv_a, cc)
    _kick_comoving!(ps.vz, az, h, inv_a, cc)
    _drift_wrap!(ps.px, ps.vx, drift_dt, lo[1], box[1])  # comoving drift + wrap
    N >= 2 && _drift_wrap!(ps.py, ps.vy, drift_dt, lo[2], box[2])
    N >= 3 && _drift_wrap!(ps.pz, ps.vz, drift_dt, lo[3], box[3])

    solve_poisson!(sim, grav)                            # φ at the drifted positions
    ax, ay, az = interp_gravity_to_particles!(sim, ps, grav)
    _kick_comoving!(ps.vx, ax, h, inv_a, cc)             # second half-kick
    _kick_comoving!(ps.vy, ay, h, inv_a, cc)
    _kick_comoving!(ps.vz, az, h, inv_a, cc)
    return nothing
end

# v ← ((1−c)·v + (g/a)·h) / (1+c)   (Enzo VELOCITY_METHOD3 half-kick; c=0,a=1 ⇒ v+=g·h)
@inline function _kick_comoving!(v, g, h, inv_a, c)
    f = 1 / (1 + c)
    @inbounds for p in eachindex(v)
        v[p] = ((1 - c) * v[p] + g[p] * inv_a * h) * f
    end
    return nothing
end

@inline function _drift_wrap!(x, v, dt, lo, box)
    @inbounds for p in eachindex(x)
        xp = x[p] + dt * v[p]
        xp = lo + mod(xp - lo, box)                      # periodic wrap into [lo, lo+box)
        x[p] = xp
    end
    return nothing
end

# ── host geometry the KA particle push consumes (flattened to leaf ordinals) ──
# Per-leaf gravitational acceleration in a stable ordinal order, plus the
# handle→ordinal map (so `_particle_csr` indexes the SAME ordering).
function _leaf_gravity(sim::Simulation, grav::GravityField)
    b = sim.backend
    handles = Any[]; ordinal = IdDict{Any,Int}()
    for_each_cell(b) do c
        push!(handles, c); ordinal[c] = length(handles)
        return nothing
    end
    n = length(handles)
    gx = zeros(Float64, n); gy = zeros(Float64, n); gz = zeros(Float64, n)
    @inbounds for k in 1:n
        g = gravity_accel(sim, grav, handles[k])
        gx[k] = Float64(g[1]); gy[k] = Float64(g[2]); gz[k] = Float64(g[3])
    end
    return gx, gy, gz, ordinal
end

# Per-particle cloud contributions as CSR (pptr/pleaf/pwgt): `pleaf` are leaf
# ORDINALS in the flat-locator / Poisson `for_each_cell` order (so the device interp
# gathers from the ordinal-indexed g directly), `pwgt` the overlap fractions (Σ per
# particle = 1). Same cloud-overlap as `deposit_particle_density!`. Allocation-free
# per-particle work (only the output arrays grow, amortized).
function _particle_csr(sim::Simulation, ps::ParticleSet)
    loc = _locator(sim, ps)
    np = length(ps.m)
    pptr = Vector{Int32}(undef, np + 1); pptr[1] = 1
    pleaf = Int32[]; pwgt = Float64[]
    _particle_csr!(pptr, pleaf, pwgt, loc, ps.px, ps.py, ps.pz)
    return pptr, pleaf, pwgt
end

function _particle_csr!(pptr, pleaf, pwgt, loc::FlatLocator{N}, px, py, pz) where {N}
    pos = (px, py, pz)
    @inbounds for p in eachindex(px)
        x = ntuple(d -> Float64(pos[d][p]), Val(N))
        home = _home_ord(loc, x)
        if home == 0
            pptr[p + 1] = pptr[p]; continue
        end
        s = ntuple(d -> loc.bhi[d, home] - loc.blo[d, home], Val(N))
        clo = ntuple(d -> x[d] - 0.5 * s[d], Val(N))
        chi = ntuple(d -> x[d] + 0.5 * s[d], Val(N))
        cvol = prod(s); cnt = 0
        for t in 0:(_cloud_nbuckets(loc, clo, chi) - 1)
            bi = _cloud_bucket(loc, clo, chi, t)
            for k in loc.bptr[bi]:(loc.bptr[bi + 1] - 1)
                ord = loc.bleaf[k]
                frac = _cloud_leaf_frac(loc, clo, chi, cvol, ord)
                frac > 0.0 && (push!(pleaf, ord); push!(pwgt, frac); cnt += 1)
            end
        end
        pptr[p + 1] = pptr[p] + cnt
    end
    return nothing
end
