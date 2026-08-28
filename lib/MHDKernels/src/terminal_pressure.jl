export terminal_pressure_ct_step!

@inline function _terminal_pressure_velocity_at(rho, fx, fy, fz, i, j, k, N,
                                                inv2dx, gamma_drag,
                                                pressure_coeff, smallr, vcap)
    c = _tct_lin3(i, j, k, N)
    cxp = _tct_lin3(i + 1, j, k, N)
    cxm = _tct_lin3(i - 1, j, k, N)
    cyp = _tct_lin3(i, j + 1, k, N)
    cym = _tct_lin3(i, j - 1, k, N)
    czp = _tct_lin3(i, j, k + 1, N)
    czm = _tct_lin3(i, j, k - 1, N)
    r = max(rho[c], smallr)
    gx = pressure_coeff * (rho[cxp] - rho[cxm]) * inv2dx
    gy = pressure_coeff * (rho[cyp] - rho[cym]) * inv2dx
    gz = pressure_coeff * (rho[czp] - rho[czm]) * inv2dx
    invrg = one(gamma_drag) / (r * max(gamma_drag, eps(gamma_drag)))
    vx = (fx[c] - gx) * invrg
    vy = (fy[c] - gy) * invrg
    vz = (fz[c] - gz) * invrg
    v2 = vx * vx + vy * vy + vz * vz
    if v2 > vcap * vcap
        factor = vcap / sqrt(v2)
        vx *= factor
        vy *= factor
        vz *= factor
    end
    return vx, vy, vz
end

@inline function _terminal_magnetic_velocity_at(rho, fx, fy, fz, c,
                                                gamma_drag, smallr, vcap)
    r = max(rho[c], smallr)
    invrg = one(gamma_drag) / (r * max(gamma_drag, eps(gamma_drag)))
    vx = fx[c] * invrg
    vy = fy[c] * invrg
    vz = fz[c] * invrg
    v2 = vx * vx + vy * vy + vz * vz
    if v2 > vcap * vcap
        factor = vcap / sqrt(v2)
        vx *= factor
        vy *= factor
        vz *= factor
    end
    return vx, vy, vz
end

@kernel function _terminal_pressure_source_k!(source, @Const(rho),
                                              @Const(fx), @Const(fy), @Const(fz),
                                              N::Int, inv2dx, gamma_drag, smallr, vcap)
    c = @index(Global)
    @inbounds begin
        T = eltype(source)
        i = (c - 1) % N + 1
        j = ((c - 1) ÷ N) % N + 1
        k = (c - 1) ÷ (N * N) + 1
        cxp = _tct_lin3(i + 1, j, k, N); cxm = _tct_lin3(i - 1, j, k, N)
        cyp = _tct_lin3(i, j + 1, k, N); cym = _tct_lin3(i, j - 1, k, N)
        czp = _tct_lin3(i, j, k + 1, N); czm = _tct_lin3(i, j, k - 1, N)
        vx, vy, vz = _terminal_magnetic_velocity_at(rho, fx, fy, fz, c,
                                                     gamma_drag, smallr, vcap)
        vxp, _, _ = _terminal_magnetic_velocity_at(rho, fx, fy, fz, cxp,
                                                    gamma_drag, smallr, vcap)
        vxm, _, _ = _terminal_magnetic_velocity_at(rho, fx, fy, fz, cxm,
                                                    gamma_drag, smallr, vcap)
        _, vyp, _ = _terminal_magnetic_velocity_at(rho, fx, fy, fz, cyp,
                                                    gamma_drag, smallr, vcap)
        _, vym, _ = _terminal_magnetic_velocity_at(rho, fx, fy, fz, cym,
                                                    gamma_drag, smallr, vcap)
        _, _, vzp = _terminal_magnetic_velocity_at(rho, fx, fy, fz, czp,
                                                    gamma_drag, smallr, vcap)
        _, _, vzm = _terminal_magnetic_velocity_at(rho, fx, fy, fz, czm,
                                                    gamma_drag, smallr, vcap)
        vxfp = T(0.5) * (vx + vxp); vxfm = T(0.5) * (vxm + vx)
        vyfp = T(0.5) * (vy + vyp); vyfm = T(0.5) * (vym + vy)
        vzfp = T(0.5) * (vz + vzp); vzfm = T(0.5) * (vzm + vz)
        flux_xp = vxfp * max(ifelse(vxfp >= zero(T), rho[c], rho[cxp]), smallr)
        flux_xm = vxfm * max(ifelse(vxfm >= zero(T), rho[cxm], rho[c]), smallr)
        flux_yp = vyfp * max(ifelse(vyfp >= zero(T), rho[c], rho[cyp]), smallr)
        flux_ym = vyfm * max(ifelse(vyfm >= zero(T), rho[cym], rho[c]), smallr)
        flux_zp = vzfp * max(ifelse(vzfp >= zero(T), rho[c], rho[czp]), smallr)
        flux_zm = vzfm * max(ifelse(vzfm >= zero(T), rho[czm], rho[c]), smallr)
        source[c] = -((flux_xp - flux_xm) + (flux_yp - flux_ym) +
                      (flux_zp - flux_zm)) * (T(2) * inv2dx)
    end
end

@kernel function _terminal_stash_eint_force_k!(eint, mx, my, mz,
                                                @Const(rho), @Const(E),
                                                @Const(Bx), @Const(By), @Const(Bz),
                                                @Const(fx), @Const(fy), @Const(fz),
                                                smallr, smallE)
    c = @index(Global)
    @inbounds begin
        T = eltype(E)
        r = max(rho[c], smallr)
        ek = T(0.5) * (mx[c]^2 + my[c]^2 + mz[c]^2) / r
        eb = T(0.5) * (Bx[c]^2 + By[c]^2 + Bz[c]^2)
        eint[c] = max(E[c] - ek - eb, smallE)
        mx[c] = fx[c]
        my[c] = fy[c]
        mz[c] = fz[c]
    end
end

@kernel function _terminal_pressure_hydro_k!(orho, omx, omy, omz, oE,
                                             @Const(rho),
                                             @Const(Bx), @Const(By), @Const(Bz),
                                             @Const(eint), @Const(rho_new),
                                             @Const(fx), @Const(fy), @Const(fz),
                                             N::Int, inv2dx, gamma_drag,
                                             pressure_coeff, gamma_gas,
                                             smallr, smallE, vcap)
    c = @index(Global)
    @inbounds begin
        T = eltype(eint)
        i = (c - 1) % N + 1
        j = ((c - 1) ÷ N) % N + 1
        k = (c - 1) ÷ (N * N) + 1
        r0 = max(rho[c], smallr)
        nr = max(rho_new[c], smallr)
        vx, vy, vz = _terminal_pressure_velocity_at(rho_new, fx, fy, fz,
                                                     i, j, k, N, inv2dx,
                                                     gamma_drag, pressure_coeff,
                                                     smallr, vcap)
        nmx = nr * vx; nmy = nr * vy; nmz = nr * vz
        ek = T(0.5) * (nmx^2 + nmy^2 + nmz^2) / nr
        eb = T(0.5) * (Bx[c]^2 + By[c]^2 + Bz[c]^2)
        orho[c] = nr
        omx[c] = nmx; omy[c] = nmy; omz[c] = nmz
        oE[c] = max(eint[c] * (nr / r0)^gamma_gas + ek + eb, smallE)
    end
end

"""
    terminal_pressure_ct_step!(solve_exponential!, s, dt; gamma_drag, ...)

Advance the overdamped density/pressure response and divergence-preserving CT
induction. `solve_exponential!(dst, state, source, coeff_cells2, dt)` supplies the
backend-specific real-FFT exponential pressure solve. Stencil inputs remain in
distinct buffers throughout the hydro finalize; the state is updated by rotating
existing scratch arrays, with no full-grid allocation or copy.
"""
function terminal_pressure_ct_step!(solve_exponential!, s::MHDState{T}, dt::Real;
                                    gamma_drag::Real, pressure_coeff::Real=1,
                                    induction_cfl::Real=0.4,
                                    induction_dissipation::Real=0.05,
                                    induction_dissipation_order::Int=4,
                                    vcap::Real=Inf,
                                    max_induction_subcycles::Int=256) where {T}
    isfinite(gamma_drag) && gamma_drag > 0 ||
        error("terminal pressure step requires finite positive gamma_drag")
    pressure_coeff >= 0 || error("pressure_coeff must be non-negative")
    N = Int(s.dims[1])
    all(==(N), s.dims) || error("terminal pressure CT currently requires a cubic grid")
    inv2dx = T(1) / (T(2) * s.dx)
    source = s.scratch[1]
    fx, fy, fz = s.scratch[2], s.scratch[3], s.scratch[4]
    eint = s.scratch[5]

    force_s = @elapsed begin
        _tct_magnetic_force_k!(s.be)(fx, fy, fz, s.U[6], s.U[7], s.U[8],
                                     N, inv2dx; ndrange=ncells(s))
        _terminal_pressure_source_k!(s.be)(source, s.U[1], fx, fy, fz,
                                            N, inv2dx, T(gamma_drag), s.smallr,
                                            T(vcap); ndrange=ncells(s))
        KA.synchronize(s.be)
    end

    coeff_cells2 = Float64(dt) * Float64(pressure_coeff) /
                   Float64(gamma_drag) / Float64(s.dx)^2
    pressure_fft_s = @elapsed begin
        solve_exponential!(reshape(eint, N, N, N), reshape(s.U[1], N, N, N),
                           reshape(source, N, N, N), coeff_cells2, dt)
        copyto!(source, eint)
        KA.synchronize(s.be)
    end

    old_u = s.U
    old_scratch = s.scratch
    finalize_s = @elapsed begin
        _terminal_stash_eint_force_k!(s.be)(eint, old_u[2], old_u[3], old_u[4],
                                             old_u[1], old_u[5], old_u[6], old_u[7], old_u[8],
                                             fx, fy, fz, s.smallr, T(1e-30);
                                             ndrange=ncells(s))
        _terminal_pressure_hydro_k!(s.be)(old_scratch[6], old_scratch[7],
                                           old_scratch[8], old_scratch[9], old_u[5],
                                           old_u[1], old_u[6], old_u[7], old_u[8],
                                           eint, source, old_u[2], old_u[3], old_u[4],
                                           N, inv2dx, T(gamma_drag), T(pressure_coeff),
                                           s.γ, s.smallr, T(1e-30), T(vcap);
                                           ndrange=ncells(s))
        KA.synchronize(s.be)
    end
    s.U = (old_scratch[6], old_scratch[7], old_scratch[8], old_scratch[9],
           old_u[5], old_u[6], old_u[7], old_u[8], old_u[9])
    s.scratch = (old_u[1], old_u[2], old_u[3], old_u[4],
                 old_scratch[1], old_scratch[2], old_scratch[3],
                 old_scratch[4], old_scratch[5])

    result = nothing
    induction_s = @elapsed begin
        result = terminal_ct_induction!(s, dt;
            gamma_drag=gamma_drag, pressure_coeff=pressure_coeff,
            cfl=induction_cfl, dissipation=induction_dissipation,
            dissipation_order=induction_dissipation_order, vcap=vcap,
            max_subcycles=max_induction_subcycles)
    end
    return (nsub=result[1], vmax=result[2], eta=result[3], force_s, pressure_fft_s,
            finalize_s, induction_s, projection_s=0.0)
end
