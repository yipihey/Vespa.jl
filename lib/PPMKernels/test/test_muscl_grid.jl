# 3-D unsplit SSP-RK2 MUSCL driver (Enzo HydroMethod=3) validation.
#
# The 1-D flux line (`muscl_flux_line!`) is already certified bit-tight vs the live
# Enzo hydro_rk solver (test_muscl.jl). This file certifies the *driver* — the
# unsplit flux-divergence assembly, the per-axis transpose + momentum rotation, and
# the SSP-RK2 coefficients — by three independent checks:
#
#   1. Per-axis reduction. On a state that varies only along one axis, the other two
#      sweeps are exact no-ops, so a full 3-D step must equal a transparent 1-D RK2
#      built from the SAME certified flux line. Run for x, y AND z (the y/z cases
#      exercise the cyclic momentum rotation against the trusted 1-D unit).
#   2. Conservation — mass, all three momenta, and total energy to round-off.
#   3. with_pool ≡ no-pool, bitwise (the pool only changes storage).
#
# Needs no Enzo dylib (pure Julia); the f32 GPU layers self-skip without Metal.

# ── transparent 1-D SSP-RK2 reference (S1 = normal momentum) ──────────────────
# Uses the certified CPU muscl_flux_line!; this IS the trusted unit, so a
# bit-identical match validates the driver's divergence/rotation/RK2 assembly.
function _rk2_line_ref(D, S1, S2, S3, Tau; na, ng, dt, dx, gamma, theta)
    active = na - 2ng; nfi = active + 1; dtdx = dt / dx
    Lop(D, S1, S2, S3, Tau) = begin
        rho = copy(D); vx = S1 ./ D; vy = S2 ./ D; vz = S3 ./ D
        eint = Tau ./ D .- 0.5 .* (vx .^ 2 .+ vy .^ 2 .+ vz .^ 2)
        fd = zeros(nfi); fs1 = zeros(nfi); fs2 = zeros(nfi); fs3 = zeros(nfi); fe = zeros(nfi)
        PPMKernels.muscl_flux_line!(fd, fs1, fs2, fs3, fe, rho, eint, vx, vy, vz;
                                    ncells = na, nghost = ng, gamma = gamma, theta = theta)
        dD = zeros(na); dS1 = zeros(na); dS2 = zeros(na); dS3 = zeros(na); dE = zeros(na)
        for m in 1:active
            c = ng + m
            dD[c]  = -(fd[m+1]  - fd[m])  * dtdx
            dS1[c] = -(fs1[m+1] - fs1[m]) * dtdx
            dS2[c] = -(fs2[m+1] - fs2[m]) * dtdx
            dS3[c] = -(fs3[m+1] - fs3[m]) * dtdx
            dE[c]  = -(fe[m+1]  - fe[m])  * dtdx
        end
        (dD, dS1, dS2, dS3, dE)
    end
    U0 = (copy(D), copy(S1), copy(S2), copy(S3), copy(Tau))
    d1 = Lop(U0...)
    U1 = ntuple(i -> U0[i] .+ d1[i], 5)
    d2 = Lop(U1...)
    return ntuple(i -> 0.5 .* (U0[i] .+ U1[i] .+ d2[i]), 5)
end

@testset "MUSCL 3-D driver (Enzo HydroMethod=3 / HD_RK)" begin
    ng = 3
    gamma, theta = 1.4, 1.5
    dt, dx = 0.02, 1.0

    # ── 1. per-axis reduction: 3-D step ≡ 1-D RK2 on an axis-only-varying state ──
    # cyclic momentum role for an axis: (normal, t1, t2) = which (S1,S2,S3) slots
    momroles(axis) = axis == 1 ? (1, 2, 3) : axis == 2 ? (2, 3, 1) : (3, 1, 2)

    @testset "reduces to 1-D RK2 along axis $(("x","y","z")[axis])" for axis in (1, 2, 3)
        na = 22
        # transverse dims ≥ 2·ng+1 (valid active region); state depends ONLY on the
        # swept-axis coordinate, so the two transverse sweeps are exact no-ops.
        odims = axis == 1 ? (na, 8, 7) : axis == 2 ? (7, na, 8) : (8, 7, na)
        nx, ny, nz = odims; N = nx * ny * nz
        idx3(i, j, k) = i + nx * (j - 1) + nx * ny * (k - 1)

        # 1-D profiles along the swept axis (a smooth + shock-ish structure)
        rho1 = zeros(na); pr1 = zeros(na); vn1 = zeros(na); vt1 = zeros(na); vt2 = zeros(na)
        for m in 1:na
            t = tanh((m - 11.0) * 0.8)
            rho1[m] = 0.6 - 0.35 * t
            pr1[m]  = 0.6 - 0.45 * t
            vn1[m]  = 0.15 - 0.10 * t          # normal velocity (varies)
            vt1[m]  = 0.05                       # transverse (uniform → no cross-sweep)
            vt2[m]  = -0.03
        end
        eint1 = pr1 ./ ((gamma - 1) .* rho1)
        etot1 = eint1 .+ 0.5 .* (vn1 .^ 2 .+ vt1 .^ 2 .+ vt2 .^ 2)

        # 1-D conserved line in (normal, t1, t2) momentum order for the reference
        Dl  = copy(rho1); Snl = rho1 .* vn1; St1l = rho1 .* vt1; St2l = rho1 .* vt2; Tl = rho1 .* etot1
        ref = _rk2_line_ref(Dl, Snl, St1l, St2l, Tl; na = na, ng = ng, dt = dt, dx = dx,
                            gamma = gamma, theta = theta)
        refmap = Dict(:D => ref[1], :Sn => ref[2], :St1 => ref[3], :St2 => ref[4], :Tau => ref[5])

        # broadcast the 1-D line over the whole 3-D grid (constant in transverse dirs)
        coord(i, j, k) = (i, j, k)[axis]
        mn, mt1, mt2 = momroles(axis)              # which (S1,S2,S3) carry (normal,t1,t2)
        D0 = zeros(N); S0 = (zeros(N), zeros(N), zeros(N)); Tau0 = zeros(N)
        for k in 1:nz, j in 1:ny, i in 1:nx
            q = idx3(i, j, k); m = coord(i, j, k)
            D0[q] = Dl[m]; Tau0[q] = Tl[m]
            S0[mn][q] = Snl[m]; S0[mt1][q] = St1l[m]; S0[mt2][q] = St2l[m]
        end

        run_step(name, ::Type{T}, sym::Symbol) where {T} = begin
            be = PPMKernels.backend(name)
            dev(a) = PPMKernels.to_device(be, a, T)
            D = dev(D0); S1 = dev(S0[1]); S2 = dev(S0[2]); S3 = dev(S0[3]); Tau = dev(Tau0)
            PPMKernels.muscl_step_3d!(D, S1, S2, S3, Tau, odims, ng;
                                      dt = dt, gamma = gamma, theta = theta, dx = dx)
            slot = (; D = D, S1 = S1, S2 = S2, S3 = S3, Tau = Tau)
            field = sym === :D ? slot.D : sym === :Tau ? slot.Tau :
                    sym === :Sn ? slot[(:S1, :S2, :S3)[mn]] :
                    sym === :St1 ? slot[(:S1, :S2, :S3)[mt1]] : slot[(:S1, :S2, :S3)[mt2]]
            h = PPMKernels.to_host(field)
            # the state is transverse-uniform, so any fixed transverse index gives
            # the pencil along the swept axis; pull it out at transverse (4,4).
            [h[axis == 1 ? idx3(mm, 4, 4) : axis == 2 ? idx3(4, mm, 4) : idx3(4, 4, mm)]
             for mm in 1:na]
        end

        for sym in (:D, :Sn, :St1, :St2, :Tau)
            r = refmap[sym]
            layerA!("muscl3d.ax$axis.$sym", run_step(:cpu, Float64, sym), r)
            layerB!("muscl3d.ax$axis.$sym", (nm, T) -> run_step(nm, T, sym))
        end
    end

    # ── 2. conservation: mass / momentum / energy to round-off ──────────────────
    @testset "conservation (mass, momentum, energy)" begin
        m = 26; cdims = (m, m, m); N = m^3
        ci3(i, j, k) = i + m * (j - 1) + m * m * (k - 1)
        rho = zeros(N); vx = zeros(N); vy = zeros(N); vz = zeros(N); etot = zeros(N)
        for k in 1:m, j in 1:m, i in 1:m
            x = (i - 13.5) / 2.2; y = (j - 13.5) / 2.2; z = (k - 13.5) / 2.2
            r2 = x^2 + y^2 + z^2; b = exp(-r2)                 # tightly contained blob
            q = ci3(i, j, k)
            rho[q] = 1.0 + 0.4 * b
            pr = 0.6 + 0.3 * b
            vx[q] = 0.10 * x * b; vy[q] = 0.08 * y * b; vz[q] = -0.06 * z * b
            eint = pr / ((gamma - 1) * rho[q])
            etot[q] = eint + 0.5 * (vx[q]^2 + vy[q]^2 + vz[q]^2)
        end
        be = PPMKernels.backend(:cpu)
        dev(a) = PPMKernels.to_device(be, a, Float64)
        D = dev(rho); S1 = similar(D); S2 = similar(D); S3 = similar(D); Tau = similar(D)
        PPMKernels.prim_to_cons!(D, S1, S2, S3, Tau, dev(rho), dev(vx), dev(vy), dev(vz), dev(etot))
        tot(f) = PPMKernels.total_field(f, cdims, ng, dx)
        m0 = tot(D); px0 = tot(S1); py0 = tot(S2); pz0 = tot(S3); e0 = tot(Tau)
        for s in 1:3
            PPMKernels.muscl_step_3d!(D, S1, S2, S3, Tau, cdims, ng;
                                      dt = dt, gamma = gamma, theta = theta, dx = dx)
        end
        @test abs(tot(D)   - m0)  / abs(m0) < 1e-12
        @test abs(tot(S1)  - px0)           < 1e-11
        @test abs(tot(S2)  - py0)           < 1e-11
        @test abs(tot(S3)  - pz0)           < 1e-11
        @test abs(tot(Tau) - e0)  / abs(e0) < 1e-12
    end

    # ── 3. with_pool ≡ no-pool, bitwise ─────────────────────────────────────────
    @testset "with_pool ≡ no pool (bitwise)" begin
        m = 16; cdims = (m, m, m); N = m^3
        ci3(i, j, k) = i + m * (j - 1) + m * m * (k - 1)
        rho = zeros(N); vx = zeros(N); vy = zeros(N); vz = zeros(N); etot = zeros(N)
        for k in 1:m, j in 1:m, i in 1:m
            x = (i - 8.5) / 2; y = (j - 8.5) / 2; z = (k - 8.5) / 2
            b = exp(-(x^2 + y^2 + z^2)); q = ci3(i, j, k)
            rho[q] = 1.0 + 0.3 * b; pr = 0.6 + 0.2 * b
            vx[q] = 0.1 * x * b; vy[q] = 0.05 * y * b; vz[q] = -0.04 * z * b
            etot[q] = pr / ((gamma - 1) * rho[q]) + 0.5 * (vx[q]^2 + vy[q]^2 + vz[q]^2)
        end
        be = PPMKernels.backend(:cpu)
        run3(pooled::Bool) = begin
            dev(a) = PPMKernels.to_device(be, a, Float64)
            D = dev(rho); S1 = similar(D); S2 = similar(D); S3 = similar(D); Tau = similar(D)
            PPMKernels.prim_to_cons!(D, S1, S2, S3, Tau, dev(rho), dev(vx), dev(vy), dev(vz), dev(etot))
            go() = for s in 1:2
                PPMKernels.muscl_step_3d!(D, S1, S2, S3, Tau, cdims, ng;
                                          dt = dt, gamma = gamma, theta = theta, dx = dx)
            end
            pooled ? PPMKernels.with_pool(go) : go()
            map(PPMKernels.to_host, (D, S1, S2, S3, Tau))
        end
        ref = run3(false); got = run3(true)
        for (r, g) in zip(ref, got)
            @test g == r
        end
    end
end

# uniform-grid PPM coefficients of length n (for recon=:ppm references)
_ppm_coeffs(n) = (fill(0.5, n), fill(0.5, n), fill(0.5, n), fill(0.5, n),
                  fill(1 / 6, n), fill(-1 / 6, n))

# ── transparent 1-D Hancock sweep reference (S1 = normal momentum) ────────────
function _hancock_line_ref(D, Sn, St1, St2, Tau; na, ng, dt, dx, gamma, theta,
                           small_rho = 1e-10, recon = :plm)
    active = na - 2ng; nfi = active + 1; dtdx = dt / dx; cpred = dt / (2dx)
    rho = copy(D); vx = Sn ./ D; vy = St1 ./ D; vz = St2 ./ D
    eint = Tau ./ D .- 0.5 .* (vx .^ 2 .+ vy .^ 2 .+ vz .^ 2)
    fd = zeros(nfi); fs1 = zeros(nfi); fs2 = zeros(nfi); fs3 = zeros(nfi); fe = zeros(nfi)
    PPMKernels.muscl_hancock_flux_line!(fd, fs1, fs2, fs3, fe, rho, eint, vx, vy, vz;
                                        ncells = na, nghost = ng, gamma = gamma, theta = theta,
                                        cpred = cpred, small_rho = small_rho,
                                        recon = recon, coeffs = recon === :ppm ? _ppm_coeffs(na) : nothing)
    out = (copy(D), copy(Sn), copy(St1), copy(St2), copy(Tau))
    flux = (fd, fs1, fs2, fs3, fe)
    for f in 1:5, mm in 1:active
        out[f][ng+mm] -= (flux[f][mm+1] - flux[f][mm]) * dtdx
    end
    return out
end

@testset "MUSCL-Hancock 3-D driver (dim-split, 3 sweeps)" begin
    ng = 3; gamma, theta = 1.4, 1.5; dt, dx = 0.02, 1.0

    # ── A. cpred=0 Hancock flux line ≡ bare reconstruction+HLL (muscl_flux_line!) ─
    @testset "cpred=0 reduces to the bare flux line" begin
        ncells = 24; active = ncells - 2 * ng
        rho = zeros(ncells); eint = zeros(ncells); vx = zeros(ncells); vy = zeros(ncells); vz = zeros(ncells)
        for i in 1:ncells
            t = tanh((i - 12.5) * 0.9)
            rho[i] = 0.5625 - 0.4375t; pr = 0.55 - 0.45t
            vx[i] = 0.2 - 0.15t; vy[i] = 0.1; vz[i] = -0.05
            eint[i] = pr / ((gamma - 1) * rho[i])
        end
        z() = zeros(active + 1)
        b1 = (z(), z(), z(), z(), z()); b2 = (z(), z(), z(), z(), z())
        PPMKernels.muscl_flux_line!(b1..., rho, eint, vx, vy, vz;
                                    ncells = ncells, nghost = ng, gamma = gamma, theta = theta)
        PPMKernels.muscl_hancock_flux_line!(b2..., rho, eint, vx, vy, vz;
                                            ncells = ncells, nghost = ng, gamma = gamma,
                                            theta = theta, cpred = 0.0)
        for f in 1:5
            @check("hancock.cpred0[$f]", b2[f], b1[f], RTOL_A)
        end
    end

    # ── B. per-axis reduction: 3-D Hancock step ≡ 1-D Hancock sweep along the axis ─
    momroles(axis) = axis == 1 ? (1, 2, 3) : axis == 2 ? (2, 3, 1) : (3, 1, 2)
    @testset "reduces to 1-D Hancock along axis $(("x","y","z")[axis])" for axis in (1, 2, 3)
        na = 22
        odims = axis == 1 ? (na, 8, 7) : axis == 2 ? (7, na, 8) : (8, 7, na)
        nx, ny, nz = odims; N = nx * ny * nz
        idx3(i, j, k) = i + nx * (j - 1) + nx * ny * (k - 1)
        rho1 = zeros(na); pr1 = zeros(na); vn1 = zeros(na); vt1 = zeros(na); vt2 = zeros(na)
        for m in 1:na
            t = tanh((m - 11.0) * 0.8)
            rho1[m] = 0.6 - 0.35t; pr1[m] = 0.6 - 0.45t; vn1[m] = 0.15 - 0.10t
            vt1[m] = 0.05; vt2[m] = -0.03
        end
        eint1 = pr1 ./ ((gamma - 1) .* rho1)
        etot1 = eint1 .+ 0.5 .* (vn1 .^ 2 .+ vt1 .^ 2 .+ vt2 .^ 2)
        Dl = copy(rho1); Snl = rho1 .* vn1; St1l = rho1 .* vt1; St2l = rho1 .* vt2; Tl = rho1 .* etot1
        ref = _hancock_line_ref(Dl, Snl, St1l, St2l, Tl; na = na, ng = ng, dt = dt, dx = dx,
                                gamma = gamma, theta = theta)
        refmap = Dict(:D => ref[1], :Sn => ref[2], :St1 => ref[3], :St2 => ref[4], :Tau => ref[5])

        coord(i, j, k) = (i, j, k)[axis]
        mn, mt1, mt2 = momroles(axis)
        D0 = zeros(N); S0 = (zeros(N), zeros(N), zeros(N)); Tau0 = zeros(N)
        for k in 1:nz, j in 1:ny, i in 1:nx
            q = idx3(i, j, k); m = coord(i, j, k)
            D0[q] = Dl[m]; Tau0[q] = Tl[m]
            S0[mn][q] = Snl[m]; S0[mt1][q] = St1l[m]; S0[mt2][q] = St2l[m]
        end
        run_step(name, ::Type{T}, sym::Symbol) where {T} = begin
            be = PPMKernels.backend(name)
            dev(a) = PPMKernels.to_device(be, a, T)
            D = dev(D0); S1 = dev(S0[1]); S2 = dev(S0[2]); S3 = dev(S0[3]); Tau = dev(Tau0)
            PPMKernels.muscl_hancock_step_3d!(D, S1, S2, S3, Tau, odims, ng;
                                              dt = dt, gamma = gamma, theta = theta, dx = dx)
            slot = (; D = D, S1 = S1, S2 = S2, S3 = S3, Tau = Tau)
            field = sym === :D ? slot.D : sym === :Tau ? slot.Tau :
                    sym === :Sn ? slot[(:S1, :S2, :S3)[mn]] :
                    sym === :St1 ? slot[(:S1, :S2, :S3)[mt1]] : slot[(:S1, :S2, :S3)[mt2]]
            h = PPMKernels.to_host(field)
            [h[axis == 1 ? idx3(mm, 4, 4) : axis == 2 ? idx3(4, mm, 4) : idx3(4, 4, mm)] for mm in 1:na]
        end
        for sym in (:D, :Sn, :St1, :St2, :Tau)
            r = refmap[sym]
            layerA!("hancock3d.ax$axis.$sym", run_step(:cpu, Float64, sym), r)
            layerB!("hancock3d.ax$axis.$sym", (nm, T) -> run_step(nm, T, sym))
        end
    end

    # ── C. conservation (mass, momentum, energy) ────────────────────────────────
    @testset "conservation (mass, momentum, energy)" begin
        m = 26; cdims = (m, m, m); N = m^3
        ci3(i, j, k) = i + m * (j - 1) + m * m * (k - 1)
        rho = zeros(N); vx = zeros(N); vy = zeros(N); vz = zeros(N); etot = zeros(N)
        for k in 1:m, j in 1:m, i in 1:m
            x = (i - 13.5) / 2.2; y = (j - 13.5) / 2.2; z = (k - 13.5) / 2.2
            b = exp(-(x^2 + y^2 + z^2)); q = ci3(i, j, k)
            rho[q] = 1.0 + 0.4b; pr = 0.6 + 0.3b
            vx[q] = 0.1x * b; vy[q] = 0.08y * b; vz[q] = -0.06z * b
            etot[q] = pr / ((gamma - 1) * rho[q]) + 0.5 * (vx[q]^2 + vy[q]^2 + vz[q]^2)
        end
        be = PPMKernels.backend(:cpu); dev(a) = PPMKernels.to_device(be, a, Float64)
        D = dev(rho); S1 = similar(D); S2 = similar(D); S3 = similar(D); Tau = similar(D)
        PPMKernels.prim_to_cons!(D, S1, S2, S3, Tau, dev(rho), dev(vx), dev(vy), dev(vz), dev(etot))
        tot(f) = PPMKernels.total_field(f, cdims, ng, dx)
        m0 = tot(D); px0 = tot(S1); py0 = tot(S2); pz0 = tot(S3); e0 = tot(Tau)
        for s in 1:3
            PPMKernels.muscl_hancock_step_3d!(D, S1, S2, S3, Tau, cdims, ng;
                                              dt = dt, gamma = gamma, theta = theta, dx = dx,
                                              order = isodd(s) ? (1, 2, 3) : (3, 2, 1))
        end
        @test abs(tot(D) - m0) / abs(m0) < 1e-12
        @test abs(tot(S1) - px0) < 1e-11
        @test abs(tot(S2) - py0) < 1e-11
        @test abs(tot(S3) - pz0) < 1e-11
        @test abs(tot(Tau) - e0) / abs(e0) < 1e-12
    end
end

@testset "MUSCL-Hancock 3-D driver — PPM reconstruction" begin
    ng = 3; gamma, theta = 1.4, 1.5; dt, dx = 0.02, 1.0
    momroles(axis) = axis == 1 ? (1, 2, 3) : axis == 2 ? (2, 3, 1) : (3, 1, 2)

    # ── A. per-axis reduction: 3-D PPM step ≡ 1-D PPM-Hancock sweep along the axis ─
    @testset "reduces to 1-D PPM-Hancock along axis $(("x","y","z")[axis])" for axis in (1, 2, 3)
        na = 22
        odims = axis == 1 ? (na, 8, 7) : axis == 2 ? (7, na, 8) : (8, 7, na)
        nx, ny, nz = odims; N = nx * ny * nz
        idx3(i, j, k) = i + nx * (j - 1) + nx * ny * (k - 1)
        rho1 = zeros(na); pr1 = zeros(na); vn1 = zeros(na); vt1 = zeros(na); vt2 = zeros(na)
        for m in 1:na
            t = tanh((m - 11.0) * 0.8)
            rho1[m] = 0.6 - 0.35t; pr1[m] = 0.6 - 0.45t; vn1[m] = 0.15 - 0.10t
            vt1[m] = 0.05; vt2[m] = -0.03
        end
        eint1 = pr1 ./ ((gamma - 1) .* rho1)
        etot1 = eint1 .+ 0.5 .* (vn1 .^ 2 .+ vt1 .^ 2 .+ vt2 .^ 2)
        Dl = copy(rho1); Snl = rho1 .* vn1; St1l = rho1 .* vt1; St2l = rho1 .* vt2; Tl = rho1 .* etot1
        ref = _hancock_line_ref(Dl, Snl, St1l, St2l, Tl; na = na, ng = ng, dt = dt, dx = dx,
                                gamma = gamma, theta = theta, recon = :ppm)
        refmap = Dict(:D => ref[1], :Sn => ref[2], :St1 => ref[3], :St2 => ref[4], :Tau => ref[5])

        coord(i, j, k) = (i, j, k)[axis]
        mn, mt1, mt2 = momroles(axis)
        D0 = zeros(N); S0 = (zeros(N), zeros(N), zeros(N)); Tau0 = zeros(N)
        for k in 1:nz, j in 1:ny, i in 1:nx
            q = idx3(i, j, k); m = coord(i, j, k)
            D0[q] = Dl[m]; Tau0[q] = Tl[m]
            S0[mn][q] = Snl[m]; S0[mt1][q] = St1l[m]; S0[mt2][q] = St2l[m]
        end
        run_step(name, ::Type{T}, sym::Symbol) where {T} = begin
            be = PPMKernels.backend(name)
            dev(a) = PPMKernels.to_device(be, a, T)
            D = dev(D0); S1 = dev(S0[1]); S2 = dev(S0[2]); S3 = dev(S0[3]); Tau = dev(Tau0)
            PPMKernels.muscl_hancock_step_3d!(D, S1, S2, S3, Tau, odims, ng;
                                              dt = dt, gamma = gamma, dx = dx, recon = :ppm)
            slot = (; D = D, S1 = S1, S2 = S2, S3 = S3, Tau = Tau)
            field = sym === :D ? slot.D : sym === :Tau ? slot.Tau :
                    sym === :Sn ? slot[(:S1, :S2, :S3)[mn]] :
                    sym === :St1 ? slot[(:S1, :S2, :S3)[mt1]] : slot[(:S1, :S2, :S3)[mt2]]
            h = PPMKernels.to_host(field)
            [h[axis == 1 ? idx3(mm, 4, 4) : axis == 2 ? idx3(4, mm, 4) : idx3(4, 4, mm)] for mm in 1:na]
        end
        for sym in (:D, :Sn, :St1, :St2, :Tau)
            r = refmap[sym]
            layerA!("ppm3d.ax$axis.$sym", run_step(:cpu, Float64, sym), r)
            layerB!("ppm3d.ax$axis.$sym", (nm, T) -> run_step(nm, T, sym))
        end
    end

    # ── B. conservation under PPM steps (Strang-alternated) ─────────────────────
    # ── C. PPM ≠ PLM (genuinely a different, sharper reconstruction) ────────────
    @testset "conservation + PPM differs from PLM" begin
        m = 26; cdims = (m, m, m); N = m^3
        ci3(i, j, k) = i + m * (j - 1) + m * m * (k - 1)
        rho = zeros(N); vx = zeros(N); vy = zeros(N); vz = zeros(N); etot = zeros(N)
        for k in 1:m, j in 1:m, i in 1:m
            x = (i - 13.5) / 2.2; y = (j - 13.5) / 2.2; z = (k - 13.5) / 2.2
            b = exp(-(x^2 + y^2 + z^2)); q = ci3(i, j, k)
            rho[q] = 1.0 + 0.4b; pr = 0.6 + 0.3b
            vx[q] = 0.1x * b; vy[q] = 0.08y * b; vz[q] = -0.06z * b
            etot[q] = pr / ((gamma - 1) * rho[q]) + 0.5 * (vx[q]^2 + vy[q]^2 + vz[q]^2)
        end
        be = PPMKernels.backend(:cpu); dev(a) = PPMKernels.to_device(be, a, Float64)
        stage() = begin
            D = dev(rho); S1 = similar(D); S2 = similar(D); S3 = similar(D); Tau = similar(D)
            PPMKernels.prim_to_cons!(D, S1, S2, S3, Tau, dev(rho), dev(vx), dev(vy), dev(vz), dev(etot))
            (D, S1, S2, S3, Tau)
        end
        tot(f) = PPMKernels.total_field(f, cdims, ng, dx)
        st = stage(); m0 = tot(st[1]); e0 = tot(st[5]); p0 = (tot(st[2]), tot(st[3]), tot(st[4]))
        for s in 1:3
            PPMKernels.muscl_hancock_step_3d!(st..., cdims, ng; dt = dt, gamma = gamma, dx = dx,
                                              order = isodd(s) ? (1, 2, 3) : (3, 2, 1), recon = :ppm)
        end
        @test abs(tot(st[1]) - m0) / abs(m0) < 1e-12          # mass
        @test abs(tot(st[2]) - p0[1]) < 1e-11                 # momentum x
        @test abs(tot(st[3]) - p0[2]) < 1e-11
        @test abs(tot(st[4]) - p0[3]) < 1e-11
        @test abs(tot(st[5]) - e0) / abs(e0) < 1e-12          # total energy

        # PPM and PLM are different schemes ⇒ a single step must differ, but agree
        # to O(Δx²) (closeness, not equality) on this smooth-ish blob.
        plm = stage(); ppm = stage()
        PPMKernels.muscl_hancock_step_3d!(plm..., cdims, ng; dt = dt, gamma = gamma, dx = dx, recon = :plm)
        PPMKernels.muscl_hancock_step_3d!(ppm..., cdims, ng; dt = dt, gamma = gamma, dx = dx, recon = :ppm)
        d_plm = PPMKernels.to_host(plm[1]); d_ppm = PPMKernels.to_host(ppm[1])
        @test d_plm != d_ppm                                  # genuinely different reconstruction
        @test maximum(abs.(d_plm .- d_ppm)) < 0.05            # but close (same physics class)
    end
end

@testset "Dual-energy formalism (Enzo hydro_rk style)" begin
    ng = 3; gamma = 1.4; dx = 1.0; dt = 0.02; m = 26; cdims = (m, m, m); N = m^3
    ci3(i, j, k) = i + m * (j - 1) + m * m * (k - 1)
    rho = zeros(N); vx = zeros(N); vy = zeros(N); vz = zeros(N); etot = zeros(N); eint = zeros(N)
    for k in 1:m, j in 1:m, i in 1:m
        x = (i - 13.5) / 2.2; y = (j - 13.5) / 2.2; z = (k - 13.5) / 2.2
        b = exp(-(x^2 + y^2 + z^2)); q = ci3(i, j, k)
        rho[q] = 1.0 + 0.4b; pr = 0.6 + 0.3b
        vx[q] = 0.1x * b; vy[q] = 0.08y * b; vz[q] = -0.06z * b
        ei = pr / ((gamma - 1) * rho[q]); eint[q] = ei
        etot[q] = ei + 0.5 * (vx[q]^2 + vy[q]^2 + vz[q]^2)
    end
    mkstate(be, ::Type{T} = Float64) where {T} = begin
        dev(a) = PPMKernels.to_device(be, a, T)
        D = dev(rho); S1 = similar(D); S2 = similar(D); S3 = similar(D); Tau = similar(D)
        PPMKernels.prim_to_cons!(D, S1, S2, S3, Tau, dev(rho), dev(vx), dev(vy), dev(vz), dev(etot))
        (D, S1, S2, S3, Tau)
    end
    tot(f) = PPMKernels.total_field(f, cdims, ng, dx)

    # (a) both drivers, dual on: mass conserved, Ge>0, AND for this SUBSONIC flow the
    #     density is bit-identical to dual-off (the η₁ selection picks eint_tot, so
    #     the DEF changes nothing where it shouldn't).
    @testset "$name: dual conserves + ≡ non-dual on subsonic" for (name, isrk2) in (("hancock", false), ("rk2", true))
        be = PPMKernels.backend(:cpu); dev(a) = PPMKernels.to_device(be, a, Float64)
        A = mkstate(be); Ge = dev(rho .* eint); B = mkstate(be); m0 = tot(A[1])
        step!(st, ge) = isrk2 ?
            PPMKernels.muscl_step_3d!(st..., cdims, ng; dt = dt, gamma = gamma, dx = dx, ge = ge) :
            PPMKernels.muscl_hancock_step_3d!(st..., cdims, ng; dt = dt, gamma = gamma, dx = dx, ge = ge)
        PPMKernels.with_pool() do
            for _ in 1:3
                step!(A, Ge); step!(B, nothing)
            end
        end
        @test abs(tot(A[1]) - m0) / abs(m0) < 1e-11               # mass conserved
        @test minimum(PPMKernels.to_host(Ge)) > 0                 # gas energy positive
        @test PPMKernels.to_host(A[1]) == PPMKernels.to_host(B[1])  # subsonic ⇒ DEF inert (bit-identical)
    end

    # (b) metal-f32 ≡ cpu-f32 with dual on (GPU parity of the whole dual path).
    if metal_ready()
        run_d(nm) = begin
            be = PPMKernels.backend(nm); dev(a) = PPMKernels.to_device(be, a, Float32)
            st = mkstate(be, Float32); Ge = dev(Float32.(rho .* eint))
            PPMKernels.with_pool() do
                for s in 1:2
                    PPMKernels.muscl_hancock_step_3d!(st..., cdims, ng; dt = dt, gamma = gamma, dx = dx,
                                                      order = isodd(s) ? (1, 2, 3) : (3, 2, 1), ge = Ge, recon = :ppm)
                end
            end
            PPMKernels.to_host(st[5])
        end
        @check("dual.metal≡cpu", run_d(:metal), run_d(:cpu), RTOL_B)
    end
end

# Flux RECORDING for the AMR reflux (`fluxrec=` on muscl_hancock_step_3d!): the
# recorded grid-frame interface fluxes must EXACTLY reproduce the conservative
# update they drove — i.e. ΔU[c] = −Σ_a (F_a[c+ê_a] − F_a[c])·(dt/dx) per cell, per
# conserved field, to round-off. This is what lets the :julia-under-AMR slot fill
# Enzo's flux registers with PPMKernels' own fluxes (ADR-0003 part B).
@testset "flux recording reproduces the conservative update (reflux enabler)" begin
    ng = 3; n = 16; nb = n + 2ng; dims = (nb, nb, nb); N = nb^3
    gamma = 1.4; dx = 1.0 / n; dt = 0.01
    nx, ny, nz = dims; idx(i, j, k) = i + nx * (j - 1) + nx * ny * (k - 1)
    rho = zeros(N); vx = zeros(N); vy = zeros(N); vz = zeros(N); etot = zeros(N)
    for k in 1:nz, j in 1:ny, i in 1:nx
        x = (i - ng - 0.5) / n; y = (j - ng - 0.5) / n; z = (k - ng - 0.5) / n; q = idx(i, j, k)
        rho[q] = 1.0 + 0.3 * sinpi(2x) * cospi(2y); pr = 0.6 + 0.2 * cospi(2z)
        vx[q] = 0.2 * sinpi(2y); vy[q] = 0.15 * cospi(2z); vz[q] = -0.1 * sinpi(2x)
        etot[q] = pr / ((gamma - 1) * rho[q]) + 0.5 * (vx[q]^2 + vy[q]^2 + vz[q]^2)
    end
    ea = (1, nx, nx * ny)                       # column-major neighbour stride per axis
    # frec[axis][field][c] = flux through the −axis face of cell c (6 fields in GRID
    # order D,S1,S2,S3,E,Ge). The +ê_a neighbour holds the +axis face of the same cell.
    @testset "$rc reconstruction" for rc in (:plm, :ppm)
        D = copy(rho); S1 = rho .* vx; S2 = rho .* vy; S3 = rho .* vz; Tau = rho .* etot
        U0 = (copy(D), copy(S1), copy(S2), copy(S3), copy(Tau))
        frec = ntuple(_ -> ntuple(_ -> zeros(N), 6), 3)
        PPMKernels.muscl_hancock_step_3d!(D, S1, S2, S3, Tau, dims, ng;
                                          dt = dt, gamma = gamma, dx = dx, recon = rc, fluxrec = frec)
        U1 = (D, S1, S2, S3, Tau); dtdx = dt / dx
        for fld in 1:5
            maxres = 0.0
            for k in ng+1:nz-ng, j in ng+1:ny-ng, i in ng+1:nx-ng
                c = idx(i, j, k)
                div = 0.0
                for a in 1:3
                    div += (frec[a][fld][c+ea[a]] - frec[a][fld][c]) * dtdx
                end
                maxres = max(maxres, abs((U1[fld][c] - U0[fld][c]) + div))
            end
            @test maxres < 1e-13                # round-off ⇒ recorded fluxes ARE the update
        end
    end

    # the default (fluxrec=nothing) path is inert: stepping with/without recording
    # gives a bit-identical state (recording is a pure read-out, no feedback).
    @testset "fluxrec=nothing is bit-identical" begin
        s0 = (copy(rho), rho .* vx, rho .* vy, rho .* vz, rho .* etot)
        s1 = (copy(rho), rho .* vx, rho .* vy, rho .* vz, rho .* etot)
        PPMKernels.muscl_hancock_step_3d!(s0..., dims, ng; dt = dt, gamma = gamma, dx = dx)
        frec = ntuple(_ -> ntuple(_ -> zeros(N), 6), 3)
        PPMKernels.muscl_hancock_step_3d!(s1..., dims, ng; dt = dt, gamma = gamma, dx = dx, fluxrec = frec)
        @test all(s0[f] == s1[f] for f in 1:5)
    end
end
