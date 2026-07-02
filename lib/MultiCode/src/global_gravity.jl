# global_gravity.jl — the GLOBAL top-grid gravity solve that couples all patches.
#
# Hydro+chem are per-patch on the GPU (patchgrid.jl); gravity is fundamentally
# global (Poisson couples all mass to all potential), so it is solved ONCE over the
# whole ncell³ domain on the CPU with a (threaded) FFTW transform — the "parallel
# CPU FFT for the top grid" — then the acceleration is scattered back to the
# patches as a ghosted block each applies as a KDK kick (patch_step!).
#
# Per cycle:  gather gas density → mean-subtract → fft_poisson_root! (CPU FFTW) →
#   difference φ (periodic-padded) → comp_accel! → scatter accel octants (periodic
#   gather, incl. ghosts) → return per-patch (ax,ay,az) for patch_step!.
#
# Particles (global periodic CIC into the same source) are a planned extension; the
# gas-density source already exercises the full global coupling and is sufficient
# for the decomposition validation (REF np=1 ≡ DECOMP np=8).

"""
    assemble_global_density!(ρg, pg; particles=nothing, dt=0.0, a=1.0,
                             particle_density=nothing, particle_host=nothing) -> ρg

Gather the patches' interior gas density into the global `ncell³` host array `ρg`,
ADD the dark-matter particles via a global periodic CIC deposit (when `particles`
is given), and mean-subtract (the periodic gravity source is the overdensity).

`particles` is a NamedTuple `(px,py,pz, vx,vy,vz, mass)` of device vectors with
GLOBAL box-normalized positions in `[0,1)` — `PoissonKernels.cic_deposit!` then
wraps periodically over the whole `ncell` grid (no per-patch edge handling).  `dt`
and `a` set the half-drift registration `disp = 0.5·dt/a` (Enzo GMF convention),
matching the undecomposed deposit exactly.
"""
function assemble_global_density!(ρg::Array{Tg,3}, pg::PatchGrid;
                                  particles=nothing, dt::Real=0.0, a::Real=1.0,
                                  meandens::Real=1.0, particle_density=nothing,
                                  particle_host=nothing) where {Tg<:AbstractFloat}
    size(ρg) == pg.ncell || error("assemble_global_density!: ρg size $(size(ρg)) ≠ ncell $(pg.ncell)")
    all(==(pg.ncell[1]), pg.ncell) || error("assemble_global_density!: cubic ncell required for CIC")
    fill!(ρg, zero(Tg))
    li, lj, lk = _interior(pg)
    for p in pg.patches                                   # baryon gas density
        gi, gj, gk = _octant(pg, p)
        h = _r3(PPMKernels.to_host(p.D), pg.nd)
        @views ρg[gi, gj, gk] .= Tg.(h[li, lj, lk])
    end
    if particles !== nothing                              # DM: global periodic CIC
        n = prod(pg.ncell)
        ρp = particle_density === nothing ? PPMKernels.device_zeros(pg.backend, pg.T, (n,)) : particle_density
        length(ρp) == n || error("assemble_global_density!: particle_density length $(length(ρp)) != $n")
        PoissonKernels.cic_deposit!(ρp, particles.px, particles.py, particles.pz,
                                    particles.vx, particles.vy, particles.vz, particles.mass;
                                    N=pg.ncell[1], disp=0.5*dt/a, shift=-0.5)
        ρph = particle_host === nothing ? PPMKernels.to_host(ρp) : particle_host
        if particle_host !== nothing
            length(ρph) == n || error("assemble_global_density!: particle_host length $(length(ρph)) != $n")
            copyto!(ρph, ρp)
        end
        @inbounds for i in eachindex(ρg)
            ρg[i] += Tg(ρph[i])
        end
    end
    # the periodic gravity source is the overdensity δ = ρ − mean.  In a periodic
    # cosmological box the mean is FIXED by Ωb,Ω0 (mass conservation), so we subtract the
    # KNOWN constant `meandens` (=1 in code units: gas mean fb + DM mean 1−fb) — no sum
    # over the field, hence no √N·ε reduction error and no need for f64.
    ρg .-= Tg(meandens)
    return ρg
end

"""
    solve_global_poisson!(φg, ρg; G=1.0, a=1.0, boxsize=1.0, greens=:spectral, solver=:fftw) -> φg

Top-grid Poisson solve on the global `ncell³` host arrays.  Two host backends:

  * `solver=:fftw` (default) — the threaded CPU **FFTW** path
    (`PoissonKernels.fft_poisson_root!`); set `fft_set_num_threads!(n)` once first.
  * `solver=:ka` — the **KernelAbstractions** radix-2 FFT run on the CPU backend,
    parallelized over `Threads.nthreads()` host threads (no FFTW dependency; the
    same kernel that runs the device FFT, with the transposes threaded). Needs a
    cubic power-of-two grid.  Start Julia with `-t N` for the parallel transform.

Both use the continuum −1/k² Green's function (`:spectral`), so they agree to
round-off.  `:ka` keeps the whole gravity path in the KA programming model.
"""
function solve_global_poisson!(φg::Array{Tg,3}, ρg::Array{Tg,3};
                               G::Real=1.0, a::Real=1.0, boxsize::Real=1.0,
                               greens::Symbol=:spectral, solver::Symbol=:fftw) where {Tg<:AbstractFloat}
    if solver === :ka
        PoissonKernels.fft_poisson_root_gpu!(φg, ρg; G=G, a=a, boxsize=boxsize)   # CPU() backend ⇒ host threads
    else
        PoissonKernels.fft_poisson_root!(φg, ρg; G=G, a=a, boxsize=boxsize, greens=greens)
    end
    return φg
end

# A patch's ghosted block maps to periodically-wrapped contiguous runs of the global accel
# along each axis: local 1:nd ↔ global starting at (o-ng) mod nc, length nd, wrapping mod nc.
# Tiled into contiguous (local, global) segments and copied with vectorized broadcasts. Handles
# nd > nc (e.g. np=1, where the ghosted block ncell+2ng exceeds the global grid → MULTIPLE wraps;
# the old ≤2-segment form ran the global index out of bounds, yielding NaN ghosts on CUDA).
function _wrap_segments(o::Int, ng::Int, nd::Int, nc::Int)
    segs = Tuple{UnitRange{Int},UnitRange{Int}}[]
    l = 1; g = mod(o - ng, nc)                 # local 1-based cursor, 0-based global cursor
    while l <= nd
        run = min(nc - g, nd - l + 1)          # cells until the wrap point or the block end
        push!(segs, (l:(l+run-1), (g+1):(g+run)))
        l += run; g = (g + run) % nc
    end
    return segs
end

# fill a patch's nd³ ghosted accel block from the global `src` by the segment copies.
# `dst::Array{Tt}` is a TYPE-PARAMETER barrier so the `Tt`-conversion specializes (a
# bare `pg.T(x)` per element is a dynamic dispatch — the real cost of the old gather).
function _scatter_block!(dst::Array{Tt,3}, src::Array{Ts,3}, xs, ys, zs) where {Tt,Ts<:AbstractFloat}
    @inbounds for (zl, zg) in zs, (yl, yg) in ys, (xl, xg) in xs
        @views dst[xl, yl, zl] .= Tt.(src[xg, yg, zg])
    end
    return dst
end

"""
    patch_accel(pg, φg; dx) -> Vector{A}

Scatter the global potential `φg` (host `ncell³`) into a ghosted `nd³` **potential** block
per patch (periodic index wrap, incl. ghosts), uploaded to the device.  No acceleration is
formed or stored — `patch_step!` central-differences these φ blocks on the fly
(`grav_kick_from_potential!`).  Returns per-patch φ device vectors (length nd³).
"""
function patch_accel(pg::PatchGrid, φg::Array{Tg,3}; dx::Real) where {Tg<:AbstractFloat}
    ng = pg.ng; nc = pg.ncell; nd = pg.nd; npatch = length(pg.patches)
    hb = Vector{Array{pg.T,3}}(undef, npatch)
    Threads.@threads for pi in 1:npatch                  # host scatter of φ (one block/patch)
        p = pg.patches[pi]; o = p.idx .* pg.pdim
        φh = Array{pg.T,3}(undef, nd)
        _scatter_block!(φh, φg, _wrap_segments(o[1],ng,nd[1],nc[1]),
                                 _wrap_segments(o[2],ng,nd[2],nc[2]),
                                 _wrap_segments(o[3],ng,nd[3],nc[3]))
        hb[pi] = φh
    end
    blocks = Vector{typeof(pg.patches[1].D)}(undef, npatch)
    for pi in 1:npatch                                   # uploads serial on the main thread
        blocks[pi] = PPMKernels.to_device(pg.backend, vec(hb[pi]), pg.T)
    end
    return blocks
end

"""
    patch_accel_gpu(pg, φ; dx) -> Vector{A}

Device version of [`patch_accel`](@ref): gather a ghosted `nd³` **potential** block per
patch from the device φ (`gather_periodic_block!`) — no acceleration formed or stored.
`patch_step!` central-differences these φ blocks in the kick (`grav_kick_from_potential!`).
"""
function patch_accel_gpu(pg::PatchGrid, φ; dx::Real)
    be = pg.backend; T = pg.T; ng = pg.ng; nc = pg.ncell; nd = pg.nd; n = prod(nd)
    φd = φ isa Array ? PPMKernels.to_device(be, φ, T) : φ
    blocks = Vector{typeof(pg.patches[1].D)}(undef, length(pg.patches))
    for (pi, p) in enumerate(pg.patches)
        b = PPMKernels.device_zeros(be, T, (n,))
        PoissonKernels.gather_periodic_block!(b, φd, p.idx .* pg.pdim, ng, nd, nc)
        blocks[pi] = b
    end
    return blocks
end

"""
    assemble_global_density_gpu!(ρd, pg; particles, dt, a) -> ρd

Device version of [`assemble_global_density!`](@ref): gather the patches' interior gas
density (device→device) + periodic DM CIC (device), then subtract the KNOWN cosmological
mean `meandens` (=1 in code units) — all on the GPU in `pg.T` (Float32). Because the mean
is fixed by Ωb,Ω0 (mass conservation in a periodic box), there is NO reduction over the
field, so f32 carries the overdensity fine (no √N·ε mean error) — no f64 needed.
"""
function assemble_global_density_gpu!(ρd, pg::PatchGrid; particles=nothing, dt::Real=0.0,
                                      a::Real=1.0, meandens::Real=1.0, depbuf=nothing,
                                      timing=nothing)
    be = pg.backend; T = pg.T; nc = pg.ncell; Ntot = prod(nc)
    all(==(nc[1]), nc) || error("assemble_global_density_gpu!: cubic ncell required")
    detail = timing !== nothing
    syncd() = detail ? PPMKernels.KA.synchronize(be) : nothing
    tgas = time()
    fill!(ρd, zero(T))
    li, lj, lk = _interior(pg)
    for p in pg.patches                                   # gas octants, device→device
        gi, gj, gk = _octant(pg, p)
        D3 = reshape(p.D, pg.nd)
        @views ρd[gi, gj, gk] .= D3[li, lj, lk]
    end
    syncd(); gas_t = time() - tgas
    dep_t = 0.0
    if particles !== nothing                              # DM periodic CIC on device
        tdep = time()
        # DETERMINISTIC integer deposit (quantized weights + integer atomics, order-
        # independent) so the gravity — and hence checkpoint/restart and run-to-run — is
        # bit-reproducible.  The float `cic_deposit!`'s atomicAdd is order-dependent: the
        # ~2e-5 run-to-run noise AND the restart divergence both traced to it. repic-style.
        # Int32 (not Int64): per-cell Σ ≈ ρ_dm·2²³ ≲ 2.5e8 ≪ 2³¹ for these z≥20 runs (mild
        # clustering), and T.(Int) → f32 is identical for both widths (24-bit exact ≪ 2.5e8),
        # so it's BIT-IDENTICAL to the Int64 deposit but ~1.7× faster (Int32 atomics on Ampere).
        # `depbuf` (when given, the KA-FFT path passes its Stockham scratch reinterpreted as Int32):
        # reuse it as the deposit accumulator — no separate N³·4 B alloc at the grid ceiling.  The FFT
        # runs AFTER this and overwrites it, so the borrow is safe.  Else allocate a fresh Int32 grid.
        borrowed = depbuf !== nothing
        ρpi = borrowed ? (fill!(depbuf, Int32(0)); reshape(depbuf, Ntot)) :
                         PPMKernels.device_zeros(be, Int32, (Ntot,))
        PoissonKernels.cic_deposit_det!(ρpi, particles.px, particles.py, particles.pz,
                                        particles.vx, particles.vy, particles.vz, particles.mass;
                                        N=nc[1], disp=0.5*dt/a, shift=-0.5, qbits=23)
        # Fuse the Int32→f32 rescale into the accumulate — no full N³ `T.(ρpi)` copy (which, even
        # once freed, keeps the CUDA.jl pool reservation high and starves cuFFT's out-of-pool work
        # area at the grid ceiling).  reshape(ρpi) is a view; Int32*Float32 promotes per-element.
        ρd .+= reshape(ρpi, nc) .* T(2.0^-23)
        # Release the freshly-allocated deposit buffer before the FFT.  On Metal the producer kernels
        # and the following MPSGraph command are separate command buffers, so make the release
        # stream-safe; otherwise the 1024³ Int32 accumulator can linger/retire late and inflate the
        # next FFT's live set by a full 4 GiB.
        if !borrowed
            occursin("Metal", string(typeof(be))) && PPMKernels.KA.synchronize(be)
            let m = parentmodule(typeof(ρpi))
                isdefined(m, :unsafe_free!) && m.unsafe_free!(ρpi)
            end
        end
        syncd(); dep_t = time() - tdep
    end
    tmean = time()
    ρd .-= T(meandens)                                    # known constant — no reduction, no f64
    syncd(); mean_t = time() - tmean
    timing !== nothing && (timing[] = (gas=gas_t, deposit=dep_t, mean=mean_t))
    return ρd
end

# assemble one patch field's interior into a global device grid (column-major, ncell³)
function _assemble_field_gpu!(dst, pg::PatchGrid, getf)
    li, lj, lk = _interior(pg)
    for p in pg.patches
        gi, gj, gk = _octant(pg, p)
        f3 = reshape(getf(p), pg.nd)
        @views dst[gi, gj, gk] .= f3[li, lj, lk]
    end
    return dst
end

@kernel function _normalize_delta_k!(a, meanv::Float32)
    i = @index(Global)
    @inbounds a[i] = a[i] / meanv - 1f0
end

@kernel function _momentum_to_velocity_k!(mom, @Const(rho), scalev::Float32, floorv::Float32)
    i = @index(Global)
    @inbounds begin
        r = Float32(rho[i])
        mom[i] = r > floorv ? Float32(mom[i]) / r * scalev : 0f0
    end
end

@kernel function _cic_deposit_weighted_unif_k!(ρ, @Const(px), @Const(py), @Const(pz), @Const(w),
                                               N::Int, shift, mass)
    p = @index(Global)
    @inbounds begin
        z = zero(px[p]); one_ = oneunit(px[p])
        gx = mod(px[p] + z, one_) * N + shift
        gy = mod(py[p] + z, one_) * N + shift
        gz = mod(pz[p] + z, one_) * N + shift
        fi = floor(gx); i0 = unsafe_trunc(Int, fi); fx = gx - fi
        fj = floor(gy); j0 = unsafe_trunc(Int, fj); fy = gy - fj
        fk = floor(gz); k0 = unsafe_trunc(Int, fk); fz = gz - fk
        ia = mod(i0, N); ib = mod(i0 + 1, N); wxa = one_ - fx; wxb = fx
        ja = mod(j0, N); jb = mod(j0 + 1, N); wya = one_ - fy; wyb = fy
        ka = mod(k0, N); kb = mod(k0 + 1, N); wza = one_ - fz; wzb = fz
        m = mass * (w[p] + zero(w[p]))
        Nj = N; Nk = N * N
        @atomic ρ[ia + Nj*ja + Nk*ka + 1] += m*wxa*wya*wza
        @atomic ρ[ib + Nj*ja + Nk*ka + 1] += m*wxb*wya*wza
        @atomic ρ[ia + Nj*jb + Nk*ka + 1] += m*wxa*wyb*wza
        @atomic ρ[ib + Nj*jb + Nk*ka + 1] += m*wxb*wyb*wza
        @atomic ρ[ia + Nj*ja + Nk*kb + 1] += m*wxa*wya*wzb
        @atomic ρ[ib + Nj*ja + Nk*kb + 1] += m*wxb*wya*wzb
        @atomic ρ[ia + Nj*jb + Nk*kb + 1] += m*wxa*wyb*wzb
        @atomic ρ[ib + Nj*jb + Nk*kb + 1] += m*wxb*wyb*wzb
    end
end

function _normalize_delta!(be, a, meanv)
    _normalize_delta_k!(be)(a, Float32(meanv); ndrange=length(a))
    PoissonKernels.KA.synchronize(be)
    return a
end

function _momentum_to_velocity!(be, mom, rho, scalev, floorv)
    _momentum_to_velocity_k!(be)(mom, rho, Float32(scalev), Float32(floorv); ndrange=length(mom))
    PoissonKernels.KA.synchronize(be)
    return mom
end

function _deposit_weighted_unif!(be, ρ, parts, w; N::Integer)
    fill!(ρ, zero(eltype(ρ)))
    _cic_deposit_weighted_unif_k!(be)(ρ, parts.px, parts.py, parts.pz, w,
        Int(N), -0.5f0, eltype(ρ)(parts.mass); ndrange=length(parts.px))
    PoissonKernels.KA.synchronize(be)
    return reshape(ρ, (N, N, N))
end

function _free_device_buffer!(a)
    a === nothing && return nothing
    m = parentmodule(typeof(a))
    isdefined(m, :unsafe_free!) && getfield(m, :unsafe_free!)(a)
    return nothing
end

"""
    patch_power_spectra(pg, parts; box, nmu=4, nbins=0, axis=1, scale_v=1.0, velocity=true)
        -> (; k, gas_delta, dm_delta, gas_vel, Nmodes)

On-device anisotropic P(k,μ) of the gas overdensity, DM overdensity, and (optionally) the
gas velocity — measured straight from the resident GPU patch fields + particle SoA, NO host
transfer and NO full-grid dump.  μ=|k_axis|/|k| (the v_bc stream is along `axis`).  `box` is
the sim length unit (P is in box³); `scale_v` multiplies the code velocity (e.g. u.v/1e5 for
km/s).  Each `P` is `(nbins, nmu)`.  Uses the device FFT [`PoissonKernels.power_spectrum_aniso_gpu`]
and the deterministic CIC deposit for the DM.  Cubic power-of-two `ncell` required.
"""
function patch_power_spectra(pg::PatchGrid, parts; box::Real, nmu::Integer=4, nbins::Integer=0,
                             axis::Integer=1, scale_v::Real=1.0, velocity::Bool=true)
    be = pg.backend; T = pg.T; nc = pg.ncell; Ntot = prod(nc); sv = T(scale_v)
    pk(f) = PoissonKernels.power_spectrum_aniso_gpu(f; boxsize=box, nmu=nmu, nbins=nbins, axis=axis)
    # deposit per-particle weight `w` (mass, or mass·v for momentum) to a float grid.
    # Metal does not support the Int64 deterministic atomic path used on CUDA here, so
    # use the f32 atomic deposit for diagnostic P(k) tables on Metal.
    metal = pg.besym === :metal
    function dep!(buf, w)
        if metal
            PoissonKernels.cic_deposit!(buf, parts.px, parts.py, parts.pz,
                parts.vx, parts.vy, parts.vz, w; N=nc[1], disp=0, shift=-0.5)
            return reshape(buf, nc)
        else
            PoissonKernels.cic_deposit_det!(buf, parts.px, parts.py, parts.pz,
                parts.vx, parts.vy, parts.vz, w; N=nc[1], disp=0, shift=-0.5, qbits=23)
            return reshape(T.(buf) .* T(2.0^-23), nc)
        end
    end
    # ── gas: overdensity + mass-weighted velocity from the resident patch fields ──
    g = PPMKernels.device_zeros(be, T, nc); _assemble_field_gpu!(g, pg, p->p.D)
    μg = T(Float64(sum(g))/Ntot)
    Pvgas = nothing
    if velocity
        Psum = nothing; Nm = nothing; kk = nothing
        mom = PPMKernels.device_zeros(be, T, nc)
        for pick in (p->p.S1, p->p.S2, p->p.S3)
            fill!(mom, zero(T))
            _assemble_field_gpu!(mom, pg, pick)
            _momentum_to_velocity!(be, mom, g, sv, zero(T))
            Pv = pk(mom)
            Psum = Psum === nothing ? copy(Pv.P) : (Psum .+ Pv.P)
            Nm === nothing && (Nm = Pv.Nmodes; kk = Pv.k)
        end
        Pvgas = (k=kk, P=Psum, Nmodes=Nm)
        _free_device_buffer!(mom); mom = nothing
    end
    _normalize_delta!(be, g, μg)
    Pgas = pk(g)
    _free_device_buffer!(g); g = nothing
    # ── DM: overdensity + mass-weighted velocity via the deterministic CIC deposit ──
    ρpi = PPMKernels.device_zeros(be, metal ? T : Int64, (Ntot,))
    ρd  = dep!(ρpi, parts.mass)
    μd = T(Float64(sum(ρd))/Ntot)
    Pvdm = nothing
    if velocity
        flr = T(1f-12) * μd                                      # ignore (near-)empty cells
        Psum = nothing; Nm = nothing; kk = nothing
        pc = PPMKernels.device_zeros(be, metal ? T : Int64, (Ntot,))
        for vp in (parts.vx, parts.vy, parts.vz)
            if metal
                vgrid = _deposit_weighted_unif!(be, pc, parts, vp; N=nc[1])  # Σ m·v_c·W
            else
                vgrid = dep!(pc, parts.mass .* vp)
            end
            _momentum_to_velocity!(be, vgrid, ρd, sv, flr)       # mass-weighted v_c (× scale_v)
            Pv = pk(vgrid)
            Psum = Psum === nothing ? copy(Pv.P) : (Psum .+ Pv.P)
            Nm === nothing && (Nm = Pv.Nmodes; kk = Pv.k)
        end
        Pvdm = (k=kk, P=Psum, Nmodes=Nm)
        _free_device_buffer!(pc); pc = nothing
    end
    _normalize_delta!(be, ρd, μd)
    Pdm = pk(ρd)
    _free_device_buffer!(ρpi)
    pg.besym === :metal && PoissonKernels.clear_fft_scratch!(pg.backend)
    return (; k=Pgas.k, gas_delta=Pgas.P, dm_delta=Pdm.P,
            gas_vel=(Pvgas===nothing ? nothing : Pvgas.P),
            dm_vel =(Pvdm ===nothing ? nothing : Pvdm.P), Nmodes=Pgas.Nmodes)
end

"""
    particle_accel_field_gpu(pg, φd; ng2=pg.ng) -> (φpad, leftedge, cellsize)

Device version of [`particle_accel_field`](@ref): pad the device potential `φd`
periodically to `(ncell+2·ng2)³` for the particle interp.  No acceleration formed —
`interp_force_from_potential!` central-differences this padded φ at the CIC cells.
"""
function particle_accel_field_gpu(pg::PatchGrid, φd; ng2::Int=pg.ng)
    be = pg.backend; T = pg.T; nc = pg.ncell
    d = PPMKernels.device_zeros(be, T, nc .+ 2ng2)
    @views d[ng2+1:ng2+nc[1], ng2+1:ng2+nc[2], ng2+1:ng2+nc[3]] .= φd
    PoissonKernels.fill_periodic_ghosts!(d; ng=ng2)
    return d, -ng2/nc[1], 1.0/nc[1]
end

"""
    global_gravity_gpu(pg; G, a, boxsize, particles, dt, ρd, φd, ng2) -> (; gas, phi, le, cs)

FULL on-GPU top-grid gravity: device density assemble → device Poisson solve
(native rFFT or optional in-tree KA FFT) → per-patch potential blocks
(`patch_accel_gpu`) + padded or global potential for particles.
No accel fields stored: `g = −∇φ` is differenced on demand in the gas kick and
particle interp.  No host round-trip, no FFTW.
"""
function global_gravity_gpu(pg::PatchGrid; G::Real=1.0, a::Real=1.0, boxsize::Real=1.0,
                            particles=nothing, dt::Real=0.0, meandens::Real=1.0,
                            ρd=nothing, φd=nothing, ng2::Int=max(pg.ng, 2),
                            global_push::Bool=false)
    # ng2 = particle-potential halo; must stay >=2 for the CIC force interp INDEPENDENT of the gas
    # ghost depth pg.ng (the FVGK dedup sets pg.ng=0, but particles still need the padded potential).
    be = pg.backend; nc = pg.ncell
    ρd === nothing && (ρd = PPMKernels.device_zeros(be, pg.T, nc))
    φd === nothing && (φd = PPMKernels.device_zeros(be, pg.T, nc))     # may alias ρd (in-place solve)
    # CIC_GRAV_KAFFT=1: solve with the in-tree KA FFT.  On CUDA this avoids cuFFT's
    # out-of-pool work area; on Metal it gives a pure-KA alternative to the MPSGraph rFFT.
    # Default = the REAL FFT (`fft_poisson_rfft_ka!`, ~2× faster + 8 B/cell); CIC_GRAV_KAFFT_C2C=1 forces
    # the c2c Stockham path (12 B/cell) — whose N³ Stockham scratch the deposit can borrow (poisson_scratch_i32).
    # Needs a 2·3·5-smooth cubic grid (rfft additionally needs even n with n/2 smooth; else it self-falls-back).
    kafft = get(ENV, "CIC_GRAV_KAFFT", "0") == "1"
    rfft  = kafft && get(ENV, "CIC_GRAV_KAFFT_C2C", "0") != "1"
    depbuf = (kafft && !rfft) ? PoissonKernels.poisson_scratch_i32(be, pg.T, nc) : nothing  # only c2c has an N³ buffer to lend
    detail = get(ENV, "CIC_GRAV_DETAIL", "0") == "1"
    syncd() = detail ? PPMKernels.KA.synchronize(be) : nothing
    asm_ref = Ref{Any}(nothing)
    tasm = time()
    assemble_global_density_gpu!(ρd, pg; particles=particles, dt=dt, a=a,
                                 meandens=meandens, depbuf=depbuf,
                                 timing=detail ? asm_ref : nothing)
    syncd(); asm_t = time() - tasm
    tfft = time()
    if kafft
        rfft ? PoissonKernels.fft_poisson_rfft_ka!(φd, ρd; G=G, a=a, boxsize=boxsize) :
               PoissonKernels.fft_poisson_root_gpu!(φd, ρd; G=G, a=a, boxsize=boxsize)
    else
        PoissonKernels.fft_poisson_rfft!(φd, ρd; G=G, a=a, boxsize=boxsize)   # native rfft; φd may === ρd
    end
    syncd(); fft_t = time() - tfft
    # dedup: the gas kick reads the GLOBAL φ directly (grav_kick_from_global_potential!), so the
    # per-patch ghosted φ-block copy is unnecessary — pass φd itself as the "accel".
    tpacc = time()
    gas = pg.dedup ? φd : patch_accel_gpu(pg, φd; dx=pg.dx)
    syncd(); pacc_t = time() - tpacc
    mk_timing(pfield_t) = detail ? (;
        assemble = asm_t,
        gas = asm_ref[] === nothing ? 0.0 : asm_ref[].gas,
        deposit = asm_ref[] === nothing ? 0.0 : asm_ref[].deposit,
        mean = asm_ref[] === nothing ? 0.0 : asm_ref[].mean,
        fft = fft_t,
        patch_accel = pacc_t,
        particle_field = pfield_t,
    ) : nothing
    if global_push
        # particles read the GLOBAL φ (periodic wrap) — no padded (ncell+2ng2)³ copy, no fill work.
        timing = mk_timing(0.0)
        return detail ? (gas=gas, phi=φd, le=0.0, cs=1.0/nc[1], nc=nc, timing=timing) :
                        (gas=gas, phi=φd, le=0.0, cs=1.0/nc[1], nc=nc)
    end
    tpfld = time()
    φpad, le, cs = particle_accel_field_gpu(pg, φd; ng2=ng2)
    syncd(); pfld_t = time() - tpfld
    timing = mk_timing(pfld_t)
    return detail ? (gas=gas, phi=φpad, le=le, cs=cs, nc=nothing, timing=timing) :
                    (gas=gas, phi=φpad, le=le, cs=cs, nc=nothing)
end

"""
    particle_accel_field(pg, φg; ng2=pg.ng) -> (gx, gy, gz, leftedge, cellsize)

Global ghosted acceleration field for the particle push: difference φ (periodic-
padded) into accel on the full `ncell³`, then re-pad periodically to `(ncell+2·ng2)³`
device arrays so `PoissonKernels.interp_accel_to_particles!` (which reads `g[i±1]`)
has valid ghosts for particles anywhere in `[0,1)`.  `leftedge = -ng2/ncell`,
`cellsize = 1/ncell` (box-normalized code coords).
"""
function particle_accel_field(pg::PatchGrid, φg::Array{Tg,3}; ng2::Int=pg.ng) where {Tg<:AbstractFloat}
    nc = pg.ncell
    a = zeros(pg.T, nc .+ 2ng2)                          # padded POTENTIAL (no accel formed)
    @views a[ng2+1:ng2+nc[1], ng2+1:ng2+nc[2], ng2+1:ng2+nc[3]] .= pg.T.(φg)
    PoissonKernels.fill_periodic_ghosts!(a; ng=ng2)
    return PPMKernels.to_device(pg.backend, a, pg.T), -ng2/nc[1], 1.0/nc[1]
end

"""
    global_gravity_accel(pg; G=1.0, a=1.0, boxsize=1.0, greens=:spectral,
                         ρg=nothing, φg=nothing) -> Vector{NTuple{3,A}}

Convenience: one full top-grid gravity solve — gather gas density, FFT-solve,
return per-patch accelerations.  Pass scratch `ρg`/`φg` (host `ncell³`) to avoid
reallocating across cycles.
"""
function global_gravity_accel(pg::PatchGrid; G::Real=1.0, a::Real=1.0, boxsize::Real=1.0,
                              greens::Symbol=:spectral, particles=nothing, dt::Real=0.0,
                              ρg::Union{Nothing,Array{<:AbstractFloat,3}}=nothing,
                              φg::Union{Nothing,Array{<:AbstractFloat,3}}=nothing)
    ρg === nothing && (ρg = zeros(Float64, pg.ncell))
    φg === nothing && (φg = zeros(Float64, pg.ncell))
    assemble_global_density!(ρg, pg; particles=particles, dt=dt, a=a)
    solve_global_poisson!(φg, ρg; G=G, a=a, boxsize=boxsize, greens=greens)
    return patch_accel(pg, φg; dx=pg.dx)
end
