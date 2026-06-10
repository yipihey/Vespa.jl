# ── the first cross-code slot: PPMKernels hydro inside RAMSES (ADR-0006 D4) ───
#
# RAMSES is the HOST: it owns the mesh, the clock, and the state (`uold`).  The
# GUEST is PPMKernels' KernelAbstractions MUSCL-Hancock sweep — the kernel
# family certified bit-tight against live Enzo (CPU f64/f32 + Metal f32).  The
# slot is the ADR-0004 sync→step→sync-back sequence:
#
#   uold (octs) ──raster──▶ ghosted Cartesian conserved fields
#                                │ muscl_hancock_step_3d! (dt from the host)
#   uold (octs) ◀──deraster── updated interior
#
# replacing the host's own `godunov_fine!`-centered update for that step.  The
# raster is exact (a uniform RAMSES level IS a Cartesian grid; the oct ckey is
# the cell address), so the only physics difference is the scheme itself —
# which is what the certification measures.

"""
    ramses_raster(h; lev, ng=2, lib=:cpu)
        -> (; D, S1, S2, S3, Tau, dims, ng, ck, n1d)

Raster one uniform RAMSES level's `uold` into ghosted flat Cartesian conserved
fields (PPMKernels layout, linear index `i + nx*(j-1) + nx*ny*(k-1)`).  RAMSES
vars 1..5 = (ρ, ρu, ρv, ρw, E) are already the canonical conserved set —
the raster is pure addressing, no physics.
"""
function ramses_raster(h::RamsesLib.Handle; lev::Integer, ng::Integer = 2, lib::Symbol = :cpu)
    ck, U = RamsesLib.get_hydro_all(h, :uold, lev; lib = lib)   # noct×3, noct×8×nvar
    noct = size(ck, 1)
    n1d = 2^lev
    8 * noct == n1d^3 || error("ramses_raster: level $lev is not the full uniform grid " *
                               "($(8noct) cells ≠ $(n1d^3)) — AMR composites are Phase 4")
    nx = n1d + 2ng
    flat() = zeros(Float64, nx^3)
    D = flat(); S1 = flat(); S2 = flat(); S3 = flat(); Tau = flat()
    @inbounds for o in 1:noct, c in 1:8
        i = 2 * ck[o, 1] + ((c - 1) & 1) + 1 + ng           # 1-based ghosted indices
        j = 2 * ck[o, 2] + ((c - 1) >> 1 & 1) + 1 + ng
        k = 2 * ck[o, 3] + ((c - 1) >> 2 & 1) + 1 + ng
        g = i + nx * (j - 1) + nx * nx * (k - 1)
        D[g]   = U[o, c, 1]
        S1[g]  = U[o, c, 2]
        S2[g]  = U[o, c, 3]
        S3[g]  = U[o, c, 4]
        Tau[g] = U[o, c, 5]
    end
    return (D = D, S1 = S1, S2 = S2, S3 = S3, Tau = Tau,
            dims = (nx, nx, nx), ng = Int(ng), ck = ck, n1d = n1d)
end

"Write the rastered interior back into the level's `uold` (inverse addressing)."
function ramses_deraster!(h::RamsesLib.Handle, r; lev::Integer, lib::Symbol = :cpu)
    ck = r.ck; noct = size(ck, 1); ng = r.ng; nx = r.dims[1]
    vals = [Matrix{Float64}(undef, noct, 8) for _ in 1:5]
    flds = (r.D, r.S1, r.S2, r.S3, r.Tau)
    @inbounds for o in 1:noct, c in 1:8
        i = 2 * ck[o, 1] + ((c - 1) & 1) + 1 + ng
        j = 2 * ck[o, 2] + ((c - 1) >> 1 & 1) + 1 + ng
        k = 2 * ck[o, 3] + ((c - 1) >> 2 & 1) + 1 + ng
        g = i + nx * (j - 1) + nx * nx * (k - 1)
        for v in 1:5
            vals[v][o, c] = flds[v][g]
        end
    end
    for v in 1:5
        RamsesLib.set_hydro!(h, :uold, v, lev, ck, vals[v]; lib = lib)
    end
    return nothing
end

"""
    ramses_ppmk_hydro_step!(h; lev, dt, gamma, boxlen, ng=2,
                            recon=:plm, riemann=:hllc, lib=:cpu, device=:cpu)

THE cross-code slot step: advance one RAMSES level by `dt` (code units, set by
the HOST's own CFL machinery) using PPMKernels' `muscl_hancock_step_3d!`
instead of `godunov_fine!`.  Periodic ghost fill, conserved in/out, host state
mutated in place — the next host step (or the comparison harness) sees exactly
the layout it expects.

`device = :metal` runs the sweep on the GPU in Float32 (Apple GPUs have no
f64): the raster stages onto the device, the certified Metal kernels advance
it, and the result lands back in the host's f64 `uold` — RAMSES's mesh, Enzo's
kernels, Apple's GPU, one statement.
"""
function ramses_ppmk_hydro_step!(h::RamsesLib.Handle; lev::Integer, dt::Real, gamma::Real,
                                 boxlen::Real, ng::Integer = 2,
                                 recon::Symbol = :plm, riemann::Symbol = :hllc,
                                 lib::Symbol = :cpu, device::Symbol = :cpu)
    r = ramses_raster(h; lev = lev, ng = ng, lib = lib)
    dx = boxlen / r.n1d
    if device === :cpu
        bc!(fields...) = PPMKernels.fill_periodic!(r.dims, r.ng, fields...)
        PPMKernels.muscl_hancock_step_3d!(r.D, r.S1, r.S2, r.S3, r.Tau, r.dims, r.ng;
                                          dt = dt, gamma = gamma, dx = dx,
                                          recon = recon, riemann = riemann, bc! = bc!)
    else
        be = PPMKernels.backend(device)
        T = Float32
        D = PPMKernels.to_device(be, r.D, T)
        S1 = PPMKernels.to_device(be, r.S1, T)
        S2 = PPMKernels.to_device(be, r.S2, T)
        S3 = PPMKernels.to_device(be, r.S3, T)
        Tau = PPMKernels.to_device(be, r.Tau, T)
        dbc!(fields...) = PPMKernels.fill_periodic!(r.dims, r.ng, fields...)
        PPMKernels.muscl_hancock_step_3d!(D, S1, S2, S3, Tau, r.dims, r.ng;
                                          dt = T(dt), gamma = T(gamma), dx = T(dx),
                                          recon = recon, riemann = riemann, bc! = dbc!)
        r.D .= Float64.(PPMKernels.to_host(D))
        r.S1 .= Float64.(PPMKernels.to_host(S1))
        r.S2 .= Float64.(PPMKernels.to_host(S2))
        r.S3 .= Float64.(PPMKernels.to_host(S3))
        r.Tau .= Float64.(PPMKernels.to_host(Tau))
    end
    ramses_deraster!(h, r; lev = lev, lib = lib)
    return nothing
end

# ── the guest slot under AMR: the COMPOSITE raster (ADR-0006 Phase 6) ─────────
#
# Correctness-first composite coupling (the AMReX-composite philosophy): the
# whole RAMSES hierarchy [levmin..levmax] is rastered onto the UNIFORM finest
# grid (leaf cells inject into their 2^(levmax-ℓ)³ footprints), the guest
# advances that composite, and the result is written back to EVERY level
# (fine leaves directly; coarser cells as the conservative average of their
# footprint — the restriction the native upload would do).  Conservation is
# exact by construction; the per-level optimization (raster each level with
# coarse-interpolated ghosts + flux registers) is the future fast path.

"Per-level leaf masks: cell (o,c) at level ℓ is refined iff its coordinate is a level-(ℓ+1) oct ckey."
function _ramses_leafmask(cks::Vector{Matrix{Int32}})
    nlev = length(cks)
    masks = Vector{Matrix{Bool}}(undef, nlev)
    for l in 1:nlev
        noct = size(cks[l], 1)
        m = trues(noct, 8)
        if l < nlev
            children = Set{NTuple{3,Int32}}((cks[l + 1][o, 1], cks[l + 1][o, 2], cks[l + 1][o, 3])
                                            for o in 1:size(cks[l + 1], 1))
            @inbounds for o in 1:noct, c in 1:8
                key = (Int32(2 * cks[l][o, 1] + ((c - 1) & 1)),
                       Int32(2 * cks[l][o, 2] + ((c - 1) >> 1 & 1)),
                       Int32(2 * cks[l][o, 3] + ((c - 1) >> 2 & 1)))
                key in children && (m[o, c] = false)
            end
        end
        masks[l] = m
    end
    return masks
end

"""
    ramses_composite_raster(h; levmin, levmax, ng=2, lib=:cpu)

Raster the hierarchy onto the uniform level-`levmax` grid (ghosted, PPMKernels
layout): every LEAF cell fills its footprint by injection.  Returns the
conserved fields plus the per-level keys/masks `ramses_composite_deraster!`
needs.
"""
function ramses_composite_raster(h::RamsesLib.Handle; levmin::Integer, levmax::Integer,
                                 ng::Integer = 2, lib::Symbol = :cpu)
    nlev = levmax - levmin + 1
    cks = Vector{Matrix{Int32}}(undef, nlev)
    Us = Vector{Array{Float64,3}}(undef, nlev)
    for (li, l) in enumerate(levmin:levmax)
        cks[li], Us[li] = RamsesLib.get_hydro_all(h, :uold, l; lib = lib)
    end
    masks = _ramses_leafmask(cks)
    n1d = 2^levmax
    nx = n1d + 2ng
    flat() = zeros(Float64, nx^3)
    D = flat(); S1 = flat(); S2 = flat(); S3 = flat(); Tau = flat()
    fields = (D, S1, S2, S3, Tau)
    covered = 0
    for (li, l) in enumerate(levmin:levmax)
        scale = 2^(levmax - l)
        @inbounds for o in 1:size(cks[li], 1), c in 1:8
            masks[li][o, c] || continue
            i0 = (2 * cks[li][o, 1] + ((c - 1) & 1)) * scale
            j0 = (2 * cks[li][o, 2] + ((c - 1) >> 1 & 1)) * scale
            k0 = (2 * cks[li][o, 3] + ((c - 1) >> 2 & 1)) * scale
            for dk in 1:scale, dj in 1:scale, di in 1:scale
                g = (i0 + di + ng) + nx * (j0 + dj + ng - 1) + nx * nx * (k0 + dk + ng - 1)
                for (v, f) in enumerate(fields)
                    f[g] = Us[li][o, c, v]
                end
            end
            covered += scale^3
        end
    end
    covered == n1d^3 ||
        error("ramses_composite_raster: leaves cover $covered of $(n1d^3) fine cells")
    return (D = D, S1 = S1, S2 = S2, S3 = S3, Tau = Tau,
            dims = (nx, nx, nx), ng = Int(ng), n1d = n1d,
            cks = cks, levmin = Int(levmin), levmax = Int(levmax))
end

"Write the composite interior back to EVERY level (fine direct, coarse = footprint average)."
function ramses_composite_deraster!(h::RamsesLib.Handle, r; lib::Symbol = :cpu)
    nx = r.dims[1]; ng = r.ng
    fields = (r.D, r.S1, r.S2, r.S3, r.Tau)
    for (li, l) in enumerate(r.levmin:r.levmax)
        ck = r.cks[li]
        noct = size(ck, 1)
        scale = 2^(r.levmax - l)
        vals = [Matrix{Float64}(undef, noct, 8) for _ in 1:5]
        @inbounds for o in 1:noct, c in 1:8
            i0 = (2 * ck[o, 1] + ((c - 1) & 1)) * scale
            j0 = (2 * ck[o, 2] + ((c - 1) >> 1 & 1)) * scale
            k0 = (2 * ck[o, 3] + ((c - 1) >> 2 & 1)) * scale
            for v in 1:5
                s = 0.0
                f = fields[v]
                for dk in 1:scale, dj in 1:scale, di in 1:scale
                    s += f[(i0 + di + ng) + nx * (j0 + dj + ng - 1) + nx * nx * (k0 + dk + ng - 1)]
                end
                vals[v][o, c] = s / scale^3                # conservative restriction
            end
        end
        for v in 1:5
            RamsesLib.set_hydro!(h, :uold, v, l, ck, vals[v]; lib = lib)
        end
    end
    return nothing
end

"""
    ramses_ppmk_hydro_step_amr!(h; levmin, levmax, gamma, boxlen, dt=nothing,
                                courant=0.4, recon=:plm, riemann=:hllc,
                                device=:cpu) -> dt_used

THE guest slot on a LIVE MULTI-LEVEL hierarchy: composite raster → one
PPMKernels step at the finest resolution → conservative write-back to every
level.  The host keeps owning refinement (call its flag/refine between steps)
and sees a hierarchy whose levels are mutually consistent (coarse = restricted
fine) — exactly the state its own upload machinery would produce.

With `dt = nothing` the GUEST computes its own CFL from the composite
(`courant·dx/max wavespeed`) — the scheme-consistent bound, and the safe
choice: RAMSES's `newdt_fine` returns 0 on a level whose time state the guest
manages.  Pass `dt_max` to cap (e.g. to land on a snapshot).  Returns the dt
actually taken.
"""
function ramses_ppmk_hydro_step_amr!(h::RamsesLib.Handle; levmin::Integer, levmax::Integer,
                                     gamma::Real, boxlen::Real, dt::Union{Nothing,Real} = nothing,
                                     dt_max::Real = Inf, courant::Real = 0.4, ng::Integer = 2,
                                     recon::Symbol = :plm, riemann::Symbol = :hllc,
                                     lib::Symbol = :cpu, device::Symbol = :cpu)
    r = ramses_composite_raster(h; levmin = levmin, levmax = levmax, ng = ng, lib = lib)
    dx = boxlen / r.n1d
    if dt === nothing
        # ghosts are zero straight out of the raster — fill them first or the
        # wavespeed scan divides by ρ = 0
        PPMKernels.fill_periodic!(r.dims, r.ng, r.D, r.S1, r.S2, r.S3, r.Tau)
        scratch = similar(r.D)
        smax = PPMKernels.max_wavespeed(scratch, r.D, r.S1, r.S2, r.S3, r.Tau; gamma = gamma)
        (isfinite(smax) && smax > 0) || error("ramses_ppmk_hydro_step_amr!: bad wavespeed $smax")
        dt = min(courant * dx / smax, dt_max)
    end
    if device === :cpu
        bc!(fields...) = PPMKernels.fill_periodic!(r.dims, r.ng, fields...)
        PPMKernels.muscl_hancock_step_3d!(r.D, r.S1, r.S2, r.S3, r.Tau, r.dims, r.ng;
                                          dt = dt, gamma = gamma, dx = dx,
                                          recon = recon, riemann = riemann, bc! = bc!)
    else
        be = PPMKernels.backend(device)
        T = Float32
        D = PPMKernels.to_device(be, r.D, T); S1 = PPMKernels.to_device(be, r.S1, T)
        S2 = PPMKernels.to_device(be, r.S2, T); S3 = PPMKernels.to_device(be, r.S3, T)
        Tau = PPMKernels.to_device(be, r.Tau, T)
        dbc!(fields...) = PPMKernels.fill_periodic!(r.dims, r.ng, fields...)
        PPMKernels.muscl_hancock_step_3d!(D, S1, S2, S3, Tau, r.dims, r.ng;
                                          dt = T(dt), gamma = T(gamma), dx = T(dx),
                                          recon = recon, riemann = riemann, bc! = dbc!)
        r.D .= Float64.(PPMKernels.to_host(D)); r.S1 .= Float64.(PPMKernels.to_host(S1))
        r.S2 .= Float64.(PPMKernels.to_host(S2)); r.S3 .= Float64.(PPMKernels.to_host(S3))
        r.Tau .= Float64.(PPMKernels.to_host(Tau))
    end
    ramses_composite_deraster!(h, r; lib = lib)
    return Float64(dt)
end

"""
    run_ramses_sod_guest(spec=SodSpec(); level=6, recon=:plm, riemann=:hllc)
        -> (; cs, t, profile, diag)

The per-run certification driver: the SAME generated Sod setup as
[`run_ramses_sod`](@ref) (double domain, host CFL via `newdt_fine!`), but with
every hydro step taken by the PPMKernels guest slot instead of
`godunov_fine!`.  Output shape is identical to the native runner, so the same
gates (ledger, profile, L1 vs exact) apply directly.
"""
function run_ramses_sod_guest(spec::SodSpec = SodSpec(); level::Integer = 6,
                              recon::Symbol = :plm, riemann::Symbol = :hllc,
                              lib::Symbol = :cpu, device::Symbol = :cpu)
    RamsesLib.available() || error("RAMSES library not found (set RAMSES_LIB to the bin64h hydro build)")
    spec.t <= 0.28 || error("t̂ > 0.28 lets the periodic seam waves reach the window")
    dir = mktempdir()
    write(joinpath(dir, "sod3d.nml"), ramses_sod_namelist(spec; level = level))
    return cd(dir) do
        h = RamsesLib.init("sod3d.nml"; lib = lib)
        lev = RamsesLib.info(h; lib = lib).levelmin
        t = 0.0; n = 0
        while t < spec.t * (1 - 1e-12) && n < 100_000
            RamsesLib.newdt_fine!(h, lev; lib = lib)            # the HOST's CFL
            dt = min(RamsesLib.get_dt(h, lev; lib = lib).dtnew, spec.t - t)
            RamsesLib.set_dt!(h, lev, dt; lib = lib)
            ramses_ppmk_hydro_step!(h; lev = lev, dt = dt, gamma = spec.gamma,
                                    boxlen = 2.0, recon = recon, riemann = riemann,
                                    lib = lib, device = device)
            t += dt; n += 1
        end
        cs = ramses_extract(h; lev = lev, boxlen = 2.0, lib = lib)
        full = profile_x(cs)
        xw = 2.0 .* full.x .- 0.5
        keep = findall(x -> 0.0 <= x <= 1.0, xw)
        profile = (x = xw[keep], rho = full.rho[keep], u = full.u[keep],
                   scatter = full.scatter)
        return (cs = cs, t = t, profile = profile,
                diag = (steps = n, level = lev, recon = recon, riemann = riemann), handle = h,
                free = () -> RamsesLib.finalize(h; lib = lib))
    end
end
