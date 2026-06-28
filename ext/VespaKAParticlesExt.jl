# KernelAbstractions particle push (the GPU/CPU device path for
# `push_particles!(sim, dt; ka=backend)`) AND the device cloud-overlap deposit/interp
# (STEP 2). The cloud-overlap GEOMETRY — home-leaf location + cloud↔leaf volume
# overlap — now runs on the device for BOTH the density deposit (atomic scatter to a
# device `rho_p`) and the g-interpolation (gather from the device g). This removes the
# host per-particle overlap loops (`deposit_particle_density!` / `_particle_csr`) that
# were the 64³ bottleneck (308M allocs / ~5 s). The locator's flat ordinal arrays
# (`blo/bhi/vol/bptr/bleaf` + `lo/L/bw/nbk`) upload once per mesh (cached on `ps.dev`,
# regrid-invalidated). The device math mirrors the CPU `_home_ord`/`_ov1d`/cloud loops
# exactly (same f64 ops, same bucket/leaf iteration order), so the gather interp is
# parity-exact with the CPU push; the atomic-scatter deposit matches to round-off.

module VespaKAParticlesExt

using Vespa
using MeshInterface: rank, domain
using KernelAbstractions
const KA = KernelAbstractions

_to_dev(be, a) = (d = KA.allocate(be, eltype(a), size(a)); copyto!(d, a); d)

# ── device cloud-overlap geometry (mirrors src/particles.jl, 3-D explicit) ────
# Home-leaf ORDINAL for point x (periodic), searched in x's bucket; 0 if none.
@inline function _ka_home_ord(x, blo, bhi, lo, L, bw, nbk, bptr, bleaf)
    bi = 1; stride = 1
    @inbounds for d in 1:3
        bk = clamp(floor(Int, (x[d] - lo[d]) / bw[d]), 0, nbk[d] - 1)
        bi += bk * stride; stride *= nbk[d]
    end
    @inbounds for k in bptr[bi]:(bptr[bi + 1] - 1)
        ord = bleaf[k]
        d1 = mod(x[1] - blo[1, ord], L[1])
        d2 = mod(x[2] - blo[2, ord], L[2])
        d3 = mod(x[3] - blo[3, ord], L[3])
        ((d1 < bhi[1, ord] - blo[1, ord]) & (d2 < bhi[2, ord] - blo[2, ord]) &
         (d3 < bhi[3, ord] - blo[3, ord])) && return Int(ord)
    end
    return bptr[bi] < bptr[bi + 1] ? Int(bleaf[bptr[bi]]) : 0     # fallback
end

# 1-D periodic overlap length of cloud [a0,a1] with leaf [b0,b1] (circumference Lp).
@inline function _ka_ov1d(a0, a1, b0, b1, Lp)
    s = 0.0
    for k in -1:1
        lo = max(a0 + k * Lp, b0); hi = min(a1 + k * Lp, b1)
        hi > lo && (s += hi - lo)
    end
    return s
end

# ── deposit: particle mass → device `rho_p` (atomic scatter, cloud overlap) ───
@kernel function _deposit_kernel!(rho_p, @Const(px), @Const(py), @Const(pz), @Const(mass),
                                  @Const(blo), @Const(bhi), @Const(vol), @Const(bptr), @Const(bleaf),
                                  lo, L, bw, nbk)
    p = @index(Global)
    @inbounds begin
        x = (px[p], py[p], pz[p])
        home = _ka_home_ord(x, blo, bhi, lo, L, bw, nbk, bptr, bleaf)
        if home != 0
            s1 = bhi[1, home] - blo[1, home]; s2 = bhi[2, home] - blo[2, home]; s3 = bhi[3, home] - blo[3, home]
            clo = (x[1] - 0.5s1, x[2] - 0.5s2, x[3] - 0.5s3)
            chi = (x[1] + 0.5s1, x[2] + 0.5s2, x[3] + 0.5s3)
            cvol = s1 * s2 * s3; mp = Float64(mass[p])
            k01 = floor(Int, (clo[1] - lo[1]) / bw[1]); n1 = floor(Int, (chi[1] - lo[1]) / bw[1]) - k01 + 1
            k02 = floor(Int, (clo[2] - lo[2]) / bw[2]); n2 = floor(Int, (chi[2] - lo[2]) / bw[2]) - k02 + 1
            k03 = floor(Int, (clo[3] - lo[3]) / bw[3]); n3 = floor(Int, (chi[3] - lo[3]) / bw[3]) - k03 + 1
            for t in 0:(n1 * n2 * n3 - 1)
                o1 = t % n1; r = t ÷ n1; o2 = r % n2; o3 = (r ÷ n2) % n3
                bk1 = mod(k01 + o1, nbk[1]); bk2 = mod(k02 + o2, nbk[2]); bk3 = mod(k03 + o3, nbk[3])
                bi = 1 + bk1 + bk2 * nbk[1] + bk3 * nbk[1] * nbk[2]
                for kk in bptr[bi]:(bptr[bi + 1] - 1)
                    ord = bleaf[kk]
                    ov1 = _ka_ov1d(clo[1], chi[1], blo[1, ord], bhi[1, ord], L[1]); ov1 == 0.0 && continue
                    ov2 = _ka_ov1d(clo[2], chi[2], blo[2, ord], bhi[2, ord], L[2]); ov2 == 0.0 && continue
                    ov3 = _ka_ov1d(clo[3], chi[3], blo[3, ord], bhi[3, ord], L[3]); ov3 == 0.0 && continue
                    val = mp * (ov1 * ov2 * ov3 / cvol) / vol[ord]
                    KA.@atomic rho_p[ord] += oftype(rho_p[ord], val)
                end
            end
        end
    end
end

# ── interp: gather device g to particles (same cloud overlap, no atomics) ─────
@kernel function _interp_overlap_kernel!(ax, ay, az, @Const(px), @Const(py), @Const(pz),
                                         @Const(blo), @Const(bhi), @Const(bptr), @Const(bleaf),
                                         lo, L, bw, nbk, @Const(gx), @Const(gy), @Const(gz))
    p = @index(Global)
    @inbounds begin
        x = (px[p], py[p], pz[p])
        home = _ka_home_ord(x, blo, bhi, lo, L, bw, nbk, bptr, bleaf)
        sx = 0.0; sy = 0.0; sz = 0.0
        if home != 0
            s1 = bhi[1, home] - blo[1, home]; s2 = bhi[2, home] - blo[2, home]; s3 = bhi[3, home] - blo[3, home]
            clo = (x[1] - 0.5s1, x[2] - 0.5s2, x[3] - 0.5s3)
            chi = (x[1] + 0.5s1, x[2] + 0.5s2, x[3] + 0.5s3)
            cvol = s1 * s2 * s3
            k01 = floor(Int, (clo[1] - lo[1]) / bw[1]); n1 = floor(Int, (chi[1] - lo[1]) / bw[1]) - k01 + 1
            k02 = floor(Int, (clo[2] - lo[2]) / bw[2]); n2 = floor(Int, (chi[2] - lo[2]) / bw[2]) - k02 + 1
            k03 = floor(Int, (clo[3] - lo[3]) / bw[3]); n3 = floor(Int, (chi[3] - lo[3]) / bw[3]) - k03 + 1
            for t in 0:(n1 * n2 * n3 - 1)
                o1 = t % n1; r = t ÷ n1; o2 = r % n2; o3 = (r ÷ n2) % n3
                bk1 = mod(k01 + o1, nbk[1]); bk2 = mod(k02 + o2, nbk[2]); bk3 = mod(k03 + o3, nbk[3])
                bi = 1 + bk1 + bk2 * nbk[1] + bk3 * nbk[1] * nbk[2]
                for kk in bptr[bi]:(bptr[bi + 1] - 1)
                    ord = bleaf[kk]
                    ov1 = _ka_ov1d(clo[1], chi[1], blo[1, ord], bhi[1, ord], L[1]); ov1 == 0.0 && continue
                    ov2 = _ka_ov1d(clo[2], chi[2], blo[2, ord], bhi[2, ord], L[2]); ov2 == 0.0 && continue
                    ov3 = _ka_ov1d(clo[3], chi[3], blo[3, ord], bhi[3, ord], L[3]); ov3 == 0.0 && continue
                    frac = ov1 * ov2 * ov3 / cvol
                    sx += frac * gx[ord]; sy += frac * gy[ord]; sz += frac * gz[ord]
                end
            end
        end
        ax[p] = sx; ay[p] = sy; az[p] = sz
    end
end

# Device locator arrays + scratch, cached on `ps.dev` and rebuilt on regrid (when
# the host locator object — hence its `objectid` — changes; `invalidate_particle_locator!`
# also clears `ps.dev`). `lo/L/bw/nbk` are isbits tuples passed as kernel args.
function _dev_locator(ps, loc, be, ::Type{T}) where {T}
    if ps.dev !== nothing && ps.dev.id === objectid(loc)
        return ps.dev
    end
    dl = (id = objectid(loc),
          blo = _to_dev(be, loc.blo), bhi = _to_dev(be, loc.bhi), vol = _to_dev(be, loc.vol),
          bptr = _to_dev(be, loc.bptr), bleaf = _to_dev(be, loc.bleaf),
          lo = loc.lo, L = loc.L, bw = loc.bw, nbk = loc.nbk,
          rho_p = KA.allocate(be, T, (length(loc.vol),)))
    ps.dev = dl
    return dl
end

# Device cloud-overlap deposit: atomic-scatter the particle mass to `dl.rho_p`, then
# copy back to the host `ps.rho_p` (the host RHS assembly + diagnostics read it, in
# the same for_each_cell/ordinal order). Builds/uses the cached host + device locator.
function Vespa._deposit_particles_ka!(sim, ps, be)
    loc = Vespa._locator(sim, ps)
    T = eltype(ps.rho_p)
    dl = _dev_locator(ps, loc, be, T)
    n = length(loc.vol)
    resize!(ps.rho_p, n)
    fill!(dl.rho_p, zero(T))
    # Reuse the push's device positions when present (no host→device re-upload);
    # otherwise (initial/standalone solve) upload from the host ps.px.
    if ps.dev_pos === nothing
        dpx = _to_dev(be, ps.px); dpy = _to_dev(be, ps.py); dpz = _to_dev(be, ps.pz)
    else
        dpx, dpy, dpz = ps.dev_pos
    end
    dm = _to_dev(be, ps.m)
    _deposit_kernel!(be)(dl.rho_p, dpx, dpy, dpz, dm,
                         dl.blo, dl.bhi, dl.vol, dl.bptr, dl.bleaf,
                         dl.lo, dl.L, dl.bw, dl.nbk; ndrange = length(ps.m))
    KA.synchronize(be)
    copyto!(ps.rho_p, dl.rho_p)                   # → host (ordinal-indexed; direct device→host, no temp)
    return nothing
end

# Device interp: g = −∇φ on the device (from the cached φ via `_device_leaf_gravity!`,
# or host-built + uploaded when the Poisson ran on the CPU). The per-particle cloud
# overlap is recomputed in the kernel (no host `_particle_csr`); g ordinals match the
# locator ordinals (both for_each_cell order), so the gather indexes the device g.
# `dpx/dpy/dpz` are the DEVICE position arrays the push already holds — passed in so the
# interp doesn't re-upload positions (the push does it once at the top, not per interp).
function _dev_interp(sim, ps, grav, be, np, dpx, dpy, dpz)
    if grav.ka_cache !== nothing
        gx, gy, gz = Vespa._device_leaf_gravity!(grav, be)   # device g from grav.ka_cache.phi
    else
        gxh, gyh, gzh, _ = Vespa._leaf_gravity(sim, grav)    # CPU Poisson fallback: build + upload
        gx = _to_dev(be, gxh); gy = _to_dev(be, gyh); gz = _to_dev(be, gzh)
    end
    loc = Vespa._locator(sim, ps)
    dl = _dev_locator(ps, loc, be, eltype(ps.rho_p))
    ax = KA.allocate(be, Float64, (np,)); ay = KA.allocate(be, Float64, (np,))
    az = KA.allocate(be, Float64, (np,))
    _interp_overlap_kernel!(be)(ax, ay, az, dpx, dpy, dpz,
                                dl.blo, dl.bhi, dl.bptr, dl.bleaf,
                                dl.lo, dl.L, dl.bw, dl.nbk, gx, gy, gz; ndrange = np)
    KA.synchronize(be)
    return ax, ay, az
end

# v ← ((1−c)v + (g/a)·h)/(1+c)   (comoving semi-implicit half-kick; c=0,a=1 ⇒ v+=g·h)
_kick!(dv, g, h, inv_a, c) = (@. dv = ((1 - c) * dv + g * inv_a * h) / (1 + c))

function Vespa._push_particles_ka!(sim, dt, be)
    ps = sim.particles; grav = sim.grav
    grav === nothing && return nothing
    N = rank(sim.backend); dom = domain(sim.backend)
    box = ntuple(d -> Float64(dom[d][2] - dom[d][1]), N)
    lo = ntuple(d -> Float64(dom[d][1]), N)
    np = length(ps.m); h = 0.5 * dt
    a, dadt = Vespa._cosmo_ah(sim, dt)              # comoving factors (1,0 if non-cosmological)
    inv_a = 1.0 / a; cc = 0.5 * (dadt / a) * (0.5 * dt); drift_dt = dt / a

    dvx = _to_dev(be, ps.vx); dvy = _to_dev(be, ps.vy); dvz = _to_dev(be, ps.vz)   # velocities: field precision
    dpx = _to_dev(be, ps.px); dpy = _to_dev(be, ps.py); dpz = _to_dev(be, ps.pz)   # positions: f64 (uploaded ONCE)

    # Positions stay device-resident for the whole step: publish them so the re-solve's
    # deposit reads the device arrays directly (no mid-push download + re-upload). Cleared
    # in `finally` so a later standalone deposit falls back to the host ps.px.
    ps.dev_pos = (dpx, dpy, dpz)
    try
        # half-kick (φ already solved for the current positions); interps reuse the device
        # position arrays — no per-interp re-upload.
        ax, ay, az = _dev_interp(sim, ps, grav, be, np, dpx, dpy, dpz)
        _kick!(dvx, ax, h, inv_a, cc); _kick!(dvy, ay, h, inv_a, cc); _kick!(dvz, az, h, inv_a, cc)

        # comoving drift + periodic wrap on device (dpx mutated in place; ps.dev_pos tracks it)
        @. dpx = lo[1] + mod(dpx + drift_dt * dvx - lo[1], box[1])
        N >= 2 && (@. dpy = lo[2] + mod(dpy + drift_dt * dvy - lo[2], box[2]))
        N >= 3 && (@. dpz = lo[3] + mod(dpz + drift_dt * dvz - lo[3], box[3]))

        # A CPU Poisson re-solve deposits on the HOST (reads ps.px), so it needs the drifted
        # positions on the host first. The device Poisson path reads ps.dev_pos — no download.
        if grav.ka === nothing
            copyto!(ps.px, dpx); N >= 2 && copyto!(ps.py, dpy); N >= 3 && copyto!(ps.pz, dpz)
        end
        Vespa.solve_poisson!(sim, grav)             # φ at the drifted positions; device deposit uses ps.dev_pos

        ax, ay, az = _dev_interp(sim, ps, grav, be, np, dpx, dpy, dpz) # second half-kick
        _kick!(dvx, ax, h, inv_a, cc); _kick!(dvy, ay, h, inv_a, cc); _kick!(dvz, az, h, inv_a, cc)
    finally
        ps.dev_pos = nothing
    end

    # Land final positions + velocities on the host ONCE (host is the source of truth
    # between steps: next push re-uploads, regrid/diagnostics read these). Direct
    # `copyto!` device→host — no allocating `Array(...)` temporary.
    copyto!(ps.px, dpx); N >= 2 && copyto!(ps.py, dpy); N >= 3 && copyto!(ps.pz, dpz)
    copyto!(ps.vx, dvx); copyto!(ps.vy, dvy); copyto!(ps.vz, dvz)
    return nothing
end

end # module
