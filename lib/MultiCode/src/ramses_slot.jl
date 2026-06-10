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
