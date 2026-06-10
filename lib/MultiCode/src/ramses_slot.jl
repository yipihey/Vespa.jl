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

# ── device residency: raster once, run resident, deraster once ────────────────

"""
    _guest_resident_run!(h, t_end; lev, gamma, boxlen, courant=0.4,
                         recon=:plm, riemann=:hllc, device=:cpu, lib=:cpu)
        -> (t, steps)

Advance a uniform RAMSES level to `t_end` with the guest state RESIDENT on the
compute device: one raster in, all steps on-device (the guest owns its CFL via
`max_wavespeed`; PPMKernels' `with_pool` recycles the per-sweep scratch), one
deraster out.  This removes the per-step host↔device round-trip that dominates
the non-resident guest's wall-clock.
"""
function _guest_resident_run!(h::RamsesLib.Handle, t_end::Real; lev::Integer, gamma::Real,
                              boxlen::Real, courant::Real = 0.4, ng::Integer = 2,
                              recon::Symbol = :plm, riemann::Symbol = :hllc,
                              device::Symbol = :cpu, lib::Symbol = :cpu)
    r = ramses_raster(h; lev = lev, ng = ng, lib = lib)
    dx = boxlen / r.n1d
    be = PPMKernels.backend(device)
    T = device === :cpu ? Float64 : Float32
    D = PPMKernels.to_device(be, r.D, T); S1 = PPMKernels.to_device(be, r.S1, T)
    S2 = PPMKernels.to_device(be, r.S2, T); S3 = PPMKernels.to_device(be, r.S3, T)
    Tau = PPMKernels.to_device(be, r.Tau, T)
    bc!(fields...) = PPMKernels.fill_periodic!(r.dims, r.ng, fields...)
    scratch = similar(D)
    t = 0.0; steps = 0
    PPMKernels.with_pool() do
        while t < t_end * (1 - 1e-12)
            steps < 100_000 || error("_guest_resident_run!: did not reach t_end (t=$t)")
            bc!(D, S1, S2, S3, Tau)
            smax = PPMKernels.max_wavespeed(scratch, D, S1, S2, S3, Tau; gamma = T(gamma))
            (isfinite(smax) && smax > 0) || error("_guest_resident_run!: bad wavespeed $smax")
            dt = min(courant * dx / Float64(smax), t_end - t)
            PPMKernels.muscl_hancock_step_3d!(D, S1, S2, S3, Tau, r.dims, r.ng;
                                              dt = T(dt), gamma = T(gamma), dx = T(dx),
                                              recon = recon, riemann = riemann, bc! = bc!)
            t += dt; steps += 1
        end
    end
    r.D .= Float64.(PPMKernels.to_host(D)); r.S1 .= Float64.(PPMKernels.to_host(S1))
    r.S2 .= Float64.(PPMKernels.to_host(S2)); r.S3 .= Float64.(PPMKernels.to_host(S3))
    r.Tau .= Float64.(PPMKernels.to_host(Tau))
    ramses_deraster!(h, r; lev = lev, lib = lib)
    return (t, steps)
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

# ── the per-level AMR fast path (ADR-0006 "Next"): ghosts + flux registers ────
#
# The composite raster advances the WHOLE domain at the finest resolution; the
# fast path advances each level on its own bounding-box raster — the coarse
# level at coarse cost, the fine level only over the refined region — and
# restores the composite update's exact conservation with FLUX REGISTERS:
#
#   1. raster every level (bbox + in-level mask); fill every non-level cell
#      (ghost band included) top-down by parent injection — the
#      coarse-interpolated ghosts;
#   2. one global dt (the finest CFL across levels); advance each level with
#      the guest RECORDING per-axis face fluxes (`fluxrec`); a child level's
#      bc! re-injects the frozen time-t parent state into every non-level
#      cell before each directional sweep;
#   3. reflux: every leaf cell facing a refined cell replaces its own face
#      flux with the area mean of the 4 child-face fluxes (ΔU = ±dt/dx·(F−F̄)),
#      after which the composite flux telescopes — every physical face is
#      crossed by exactly one flux — and conservation is exact by construction;
#   4. restrict refined cells bottom-up (coarse = average of its children)
#      and write every level back.
#
# 2:1 grading (RAMSES enforces it) guarantees the child cells on a coarse-fine
# face are leaves, so one register level per level pair suffices.

"Per-level bbox raster: fields over the level's bounding box (+ng ghosts) with an in-level mask."
function _lr_raster(h::RamsesLib.Handle, lev::Integer; ng::Integer = 2, lib::Symbol = :cpu)
    ck, U = RamsesLib.get_hydro_all(h, :uold, lev; lib = lib)
    noct = size(ck, 1)
    noct > 0 || return nothing
    n1d = 2^lev
    lo = (typemax(Int), typemax(Int), typemax(Int))
    hi = (typemin(Int), typemin(Int), typemin(Int))
    @inbounds for o in 1:noct
        b = (2 * Int(ck[o, 1]), 2 * Int(ck[o, 2]), 2 * Int(ck[o, 3]))
        lo = min.(lo, b); hi = max.(hi, b .+ 1)
    end
    nloc = hi .- lo .+ 1
    dims = nloc .+ 2ng
    nx, ny, nz = dims
    flat() = zeros(Float64, nx * ny * nz)
    D = flat(); S1 = flat(); S2 = flat(); S3 = flat(); Tau = flat()
    covered = falses(dims...); refined = falses(dims...)
    @inbounds for o in 1:noct, c in 1:8
        i = 2 * Int(ck[o, 1]) + ((c - 1) & 1) - lo[1] + ng + 1
        j = 2 * Int(ck[o, 2]) + ((c - 1) >> 1 & 1) - lo[2] + ng + 1
        k = 2 * Int(ck[o, 3]) + ((c - 1) >> 2 & 1) - lo[3] + ng + 1
        g = i + nx * (j - 1) + nx * ny * (k - 1)
        D[g] = U[o, c, 1]; S1[g] = U[o, c, 2]; S2[g] = U[o, c, 3]
        S3[g] = U[o, c, 4]; Tau[g] = U[o, c, 5]
        covered[i, j, k] = true
    end
    return (D = D, S1 = S1, S2 = S2, S3 = S3, Tau = Tau,
            dims = dims, ng = Int(ng), lev = Int(lev), n1d = n1d,
            off = lo, nloc = nloc, ck = ck, covered = covered, refined = refined)
end

_lr_fields(r) = (r.D, r.S1, r.S2, r.S3, r.Tau)

"Mark the parent raster's refined cells (a child oct's ckey IS its parent cell's coordinate)."
function _lr_mark_refined!(p, ck_child::Matrix{Int32})
    ng = p.ng
    @inbounds for o in 1:size(ck_child, 1)
        i = Int(ck_child[o, 1]) - p.off[1] + ng + 1
        j = Int(ck_child[o, 2]) - p.off[2] + ng + 1
        k = Int(ck_child[o, 3]) - p.off[3] + ng + 1
        p.covered[i, j, k] || error("_lr_mark_refined!: child oct over an uncovered parent cell")
        p.refined[i, j, k] = true
    end
    return nothing
end

"""
Precompute (child linear, parent linear) injection pairs for every NON-level
cell of the child raster (ghost band included): the coarse ghost fill, applied
once at raster time and re-applied (frozen at time t) before every sweep.
"""
function _lr_fill_pairs(c, p)
    pairs = NTuple{2,Int}[]
    ngc = c.ng; nxc, nyc, nzc = c.dims
    ngp = p.ng; nxp, nyp, _ = p.dims
    for k in 1:nzc, j in 1:nyc, i in 1:nxc
        c.covered[i, j, k] && continue
        gx = mod(c.off[1] + (i - ngc - 1), c.n1d)     # periodic global child coords
        gy = mod(c.off[2] + (j - ngc - 1), c.n1d)
        gz = mod(c.off[3] + (k - ngc - 1), c.n1d)
        pi = (gx >> 1) - p.off[1] + ngp + 1
        pj = (gy >> 1) - p.off[2] + ngp + 1
        pk = (gz >> 1) - p.off[3] + ngp + 1
        (ngp + 1 <= pi <= ngp + p.nloc[1] && ngp + 1 <= pj <= ngp + p.nloc[2] &&
         ngp + 1 <= pk <= ngp + p.nloc[3]) ||
            error("per-level fast path: level-$(c.lev) ghost ($gx,$gy,$gz) maps outside " *
                  "the level-$(p.lev) raster — nesting too tight for ng=$ngc")
        push!(pairs, (i + nxc * (j - 1) + nxc * nyc * (k - 1),
                      pi + nxp * (pj - 1) + nxp * nyp * (pk - 1)))
    end
    return pairs
end

"Inject parent values into the child's non-level cells (the precomputed pairs)."
function _lr_inject!(cf, pf, pairs::Vector{NTuple{2,Int}})
    @inbounds for (gc, gp) in pairs
        cf[1][gc] = pf[1][gp]; cf[2][gc] = pf[2][gp]; cf[3][gc] = pf[3][gp]
        cf[4][gc] = pf[4][gp]; cf[5][gc] = pf[5][gp]
    end
    return nothing
end

# face-axis child-local index with a periodic wrap into the recorded flux range
@inline function _lr_clocal_wrap(cg::Int, off::Int, ng::Int, nact::Int, n1d::Int)
    cl = cg - off + ng + 1
    cl < ng + 1 && (cl += n1d)
    cl > ng + nact + 1 && (cl -= n1d)
    return cl
end

"""
The flux register: for every leaf parent cell with a refined face neighbor,
replace the parent's recorded face flux with the area mean of the 4 child-face
fluxes.  `frec[axis][v][g]` is the flux through cell `g`'s −axis face (the
guest's `fluxrec` convention), so the correction for the face on side `s` of
the cell is `ΔU = s · dt/dx_parent · (F_parent − F̄_child)`.
"""
function _lr_reflux!(p, frec_p, c, frec_c, dtdx_p::Float64)
    ng = p.ng; nx, ny, _ = p.dims
    ngc = c.ng; nxc, nyc, _ = c.dims
    pf = _lr_fields(p)
    @inbounds for k in ng+1:ng+p.nloc[3], j in ng+1:ng+p.nloc[2], i in ng+1:ng+p.nloc[1]
        (p.covered[i, j, k] && !p.refined[i, j, k]) || continue
        gP = (p.off[1] + (i - ng - 1), p.off[2] + (j - ng - 1), p.off[3] + (k - ng - 1))
        for axis in 1:3, side in (-1, 1)
            gN = ntuple(d -> d == axis ? mod(gP[d] + side, p.n1d) : gP[d], 3)
            nix = gN[1] - p.off[1] + ng + 1
            njy = gN[2] - p.off[2] + ng + 1
            nkz = gN[3] - p.off[3] + ng + 1
            (1 <= nix <= nx && 1 <= njy <= ny && 1 <= nkz <= p.dims[3]) || continue
            p.refined[nix, njy, nkz] || continue
            # the +axis-side parent cell owns the face (the flux lives at its −axis face)
            gplus = side == 1 ? gN : gP
            gp_lin = (side == 1 ? nix : i) + nx * ((side == 1 ? njy : j) - 1) +
                     nx * ny * ((side == 1 ? nkz : k) - 1)
            t1 = axis % 3 + 1; t2 = t1 % 3 + 1
            # the 4 child flux carriers: child cells whose −axis face IS the coarse face
            ca = _lr_clocal_wrap(2 * gplus[axis], c.off[axis], ngc, c.nloc[axis], c.n1d)
            (ngc + 1 <= ca <= ngc + c.nloc[axis] + 1) ||
                error("_lr_reflux!: face carrier outside the child raster's recorded range")
            P_lin = i + nx * (j - 1) + nx * ny * (k - 1)
            carriers = ntuple(4) do q
                b1 = (q - 1) & 1; b2 = (q - 1) >> 1
                ct1 = 2 * gP[t1] + b1 - c.off[t1] + ngc + 1
                ct2 = 2 * gP[t2] + b2 - c.off[t2] + ngc + 1
                cidx = ntuple(d -> d == axis ? ca : (d == t1 ? ct1 : ct2), 3)
                cidx[1] + nxc * (cidx[2] - 1) + nxc * nyc * (cidx[3] - 1)
            end
            for v in 1:5
                Fp = frec_p[axis][v][gp_lin]
                Fbar = 0.25 * (frec_c[axis][v][carriers[1]] + frec_c[axis][v][carriers[2]] +
                               frec_c[axis][v][carriers[3]] + frec_c[axis][v][carriers[4]])
                pf[v][P_lin] += side * dtdx_p * (Fp - Fbar)
            end
        end
    end
    return nothing
end

"Restriction: every refined parent cell becomes the average of its 8 children."
function _lr_restrict!(p, c)
    ng = p.ng; nx, ny, _ = p.dims
    ngc = c.ng; nxc, nyc, _ = c.dims
    pf = _lr_fields(p); cf = _lr_fields(c)
    @inbounds for k in ng+1:ng+p.nloc[3], j in ng+1:ng+p.nloc[2], i in ng+1:ng+p.nloc[1]
        p.refined[i, j, k] || continue
        b = (2 * (p.off[1] + (i - ng - 1)) - c.off[1] + ngc + 1,
             2 * (p.off[2] + (j - ng - 1)) - c.off[2] + ngc + 1,
             2 * (p.off[3] + (k - ng - 1)) - c.off[3] + ngc + 1)
        P_lin = i + nx * (j - 1) + nx * ny * (k - 1)
        for v in 1:5
            s = 0.0
            for dk in 0:1, dj in 0:1, di in 0:1
                s += cf[v][(b[1] + di) + nxc * (b[2] + dj - 1) + nxc * nyc * (b[3] + dk - 1)]
            end
            pf[v][P_lin] = s / 8
        end
    end
    return nothing
end

"Write a per-level raster's in-level cells back into `uold`."
function _lr_deraster!(h::RamsesLib.Handle, r; lib::Symbol = :cpu)
    ck = r.ck; noct = size(ck, 1); ng = r.ng; nx, ny, _ = r.dims
    vals = [Matrix{Float64}(undef, noct, 8) for _ in 1:5]
    flds = _lr_fields(r)
    @inbounds for o in 1:noct, c in 1:8
        i = 2 * Int(ck[o, 1]) + ((c - 1) & 1) - r.off[1] + ng + 1
        j = 2 * Int(ck[o, 2]) + ((c - 1) >> 1 & 1) - r.off[2] + ng + 1
        k = 2 * Int(ck[o, 3]) + ((c - 1) >> 2 & 1) - r.off[3] + ng + 1
        g = i + nx * (j - 1) + nx * ny * (k - 1)
        for v in 1:5
            vals[v][o, c] = flds[v][g]
        end
    end
    for v in 1:5
        RamsesLib.set_hydro!(h, :uold, v, r.lev, ck, vals[v]; lib = lib)
    end
    return nothing
end

"""
    ramses_ppmk_hydro_step_amr_fast!(h; levmin, levmax, gamma, boxlen,
                                     dt=nothing, dt_max=Inf, courant=0.4,
                                     recon=:plm, riemann=:hllc) -> dt_used

The PER-LEVEL fast path of [`ramses_ppmk_hydro_step_amr!`](@ref): each level
advances on its own bounding-box raster (the coarse level at coarse cost, the
fine level only over the refined region) with coarse-injected ghosts, and the
flux registers restore exact conservation at every coarse-fine face.  One
global dt (the finest CFL) — per-level subcycling is the next optimization.
Same contract as the composite step: the host keeps owning refinement, and
every level it sees afterwards is mutually consistent (coarse ≡ restricted
fine).  CPU only (the registers and masks live host-side).
"""
function ramses_ppmk_hydro_step_amr_fast!(h::RamsesLib.Handle; levmin::Integer, levmax::Integer,
                                          gamma::Real, boxlen::Real,
                                          dt::Union{Nothing,Real} = nothing,
                                          dt_max::Real = Inf, courant::Real = 0.4,
                                          ng::Integer = 2, recon::Symbol = :plm,
                                          riemann::Symbol = :hllc, lib::Symbol = :cpu)
    rs = NamedTuple[]
    for l in levmin:levmax
        r = _lr_raster(h, l; ng = ng, lib = lib)
        r === nothing && break                    # no octs ⇒ nothing finer either
        push!(rs, r)
    end
    nlev = length(rs)
    nlev > 0 || error("ramses_ppmk_hydro_step_amr_fast!: empty hierarchy at level $levmin")
    # refined masks + top-down coarse-injection fill of every non-level cell
    pairs = Vector{Vector{NTuple{2,Int}}}(undef, nlev)
    for li in 2:nlev
        _lr_mark_refined!(rs[li-1], rs[li].ck)
        pairs[li] = _lr_fill_pairs(rs[li], rs[li-1])
        _lr_inject!(_lr_fields(rs[li]), _lr_fields(rs[li-1]), pairs[li])
    end
    PPMKernels.fill_periodic!(rs[1].dims, rs[1].ng, _lr_fields(rs[1])...)
    if dt === nothing
        dt = dt_max
        for r in rs
            scratch = similar(r.D)
            smax = PPMKernels.max_wavespeed(scratch, r.D, r.S1, r.S2, r.S3, r.Tau; gamma = gamma)
            (isfinite(smax) && smax > 0) ||
                error("ramses_ppmk_hydro_step_amr_fast!: bad wavespeed $smax on level $(r.lev)")
            dt = min(dt, courant * (boxlen / r.n1d) / smax)
        end
    end
    # frozen time-t parent states: the ghost source for every child sweep
    frozen = [ntuple(v -> copy(_lr_fields(rs[li])[v]), 5) for li in 1:nlev-1]
    frecs = Vector{Any}(undef, nlev)
    for (li, r) in enumerate(rs)
        frec = ntuple(_ -> ntuple(_ -> zeros(Float64, length(r.D)), 5), 3)
        bc! = li == 1 ?
            ((flds...) -> PPMKernels.fill_periodic!(r.dims, r.ng, flds...)) :
            (let pr = frozen[li-1], pp = pairs[li]
                 (flds...) -> _lr_inject!(flds, pr, pp)
             end)
        PPMKernels.muscl_hancock_step_3d!(r.D, r.S1, r.S2, r.S3, r.Tau, r.dims, r.ng;
                                          dt = dt, gamma = gamma, dx = boxlen / r.n1d,
                                          recon = recon, riemann = riemann,
                                          bc! = bc!, fluxrec = frec)
        frecs[li] = frec
    end
    for li in 1:nlev-1
        _lr_reflux!(rs[li], frecs[li], rs[li+1], frecs[li+1], Float64(dt) / (boxlen / rs[li].n1d))
    end
    for li in nlev-1:-1:1
        _lr_restrict!(rs[li], rs[li+1])
    end
    for r in rs
        _lr_deraster!(h, r; lib = lib)
    end
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
