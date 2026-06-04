# Phase 4 — 3-D directional-split PPM vs the live Fortran (pencil-wise).
#
# There is no 3-D Fortran reference, so each axis sweep is certified against
# EnzoLib.ppm_sweep_1d_full! applied to every 1-D pencil along that axis (with the
# cyclic velocity permutation). The composed Strang step is then validated by an
# exact tie-back: an x-only-varying initial state must make ppm_step_3d! reduce
# to the certified 1-D sweep (the y/z sweeps must be exact no-ops — which also
# exercises the transpose round-trip and the per-step pressure recompute).

using EnzoLib

@testset "Phase 4 — 3-D directional-split PPM" begin
    if !EnzoLib.available()
        @test_skip "EnzoLib pilot library not built — skipping Fortran-reference 3-D layers"
    else
        nx, ny, nz = 13, 12, 11
        ng         = 4
        dims       = (nx, ny, nz)
        N          = nx * ny * nz
        gamma, dx, dt = 1.4, 1.0, 0.04
        idx3(i, j, k) = i + nx * (j - 1) + nx * ny * (k - 1)

        # smooth centred blob (quiescent edges); pressure is the exact EOS
        d0 = zeros(N); e0 = zeros(N); ge0 = zeros(N)
        vx0 = zeros(N); vy0 = zeros(N); vz0 = zeros(N); p0 = zeros(N); gr0 = zeros(N)
        for k in 1:nz, j in 1:ny, i in 1:nx
            x = (i - 7) / 4; y = (j - 6.5) / 4; z = (k - 6) / 4
            r2 = x^2 + y^2 + z^2; b = exp(-r2)
            q = idx3(i, j, k)
            d0[q] = 1.0 + 0.30 * b
            pr    = 0.60 + 0.20 * b
            vx0[q] = 0.10 * x * b; vy0[q] = 0.08 * y * b; vz0[q] = 0.06 * z * b
            ge0[q] = pr / ((gamma - 1) * d0[q])
            e0[q]  = ge0[q] + 0.5 * (vx0[q]^2 + vy0[q]^2 + vz0[q]^2)
            p0[q]  = (gamma - 1) * d0[q] * (e0[q] - 0.5 * (vx0[q]^2 + vy0[q]^2 + vz0[q]^2))
            gr0[q] = 0.05 * sinpi((i + j + k) / 12)
        end

        # ── reference: ppm_sweep_1d_full! on every pencil along `axis` ──────────
        pencil_lines(axis) = axis == 1 ? [( [idx3(i, j, k) for i in 1:nx] ) for k in 1:nz for j in 1:ny] :
                             axis == 2 ? [( [idx3(i, j, k) for j in 1:ny] ) for k in 1:nz for i in 1:nx] :
                                         [( [idx3(i, j, k) for k in 1:nz] ) for j in 1:ny for i in 1:nx]
        # velocity arrays in (normal, t1, t2) cyclic order for an axis
        vroles(axis, vx, vy, vz) = axis == 1 ? (vx, vy, vz) : axis == 2 ? (vy, vz, vx) : (vz, vx, vy)

        function ref_sweep_axis(axis, fl)
            d = copy(d0); e = copy(e0); ge = copy(ge0)
            vx = copy(vx0); vy = copy(vy0); vz = copy(vz0)
            na = dims[axis]; i1, i2 = ng + 1, na - ng
            vn, vt1, vt2 = vroles(axis, vx, vy, vz)
            for line in pencil_lines(axis)
                pd = d[line]; pe = e[line]; pge = ge[line]
                pu = vn[line]; pv = vt1[line]; pw = vt2[line]
                pp = p0[line]; pg = gr0[line]
                EnzoLib.ppm_sweep_1d_full!(pd, pe, pu, pv, pw, pp;
                    i1 = i1, i2 = i2, dx = dx, dt = dt, gamma = gamma, geslice = pge, grslice = pg,
                    gravity = fl.gravity, idual = fl.idual, eta2 = fl.eta2, isteep = fl.isteep,
                    iflatten = fl.iflatten, idiff = fl.idiff)
                d[line] = pd; e[line] = pe; ge[line] = pge
                vn[line] = pu; vt1[line] = pv; vt2[line] = pw
            end
            return (; dslice = d, eslice = e, geslice = ge, vx = vx, vy = vy, vz = vz)
        end

        flagsets = (
            ("plain", (gravity=0, idual=0, isteep=0, iflatten=0, idiff=0, eta2=0.0)),
            ("full",  (gravity=1, idual=1, isteep=1, iflatten=3, idiff=1, eta2=0.1)),
        )

        for axis in (1, 2, 3), (ftag, fl) in flagsets
            (axis != 1 && ftag == "full") && continue          # full config on x is enough
            @testset "sweep_axis $(("x","y","z")[axis]) [$ftag]" begin
                ref = ref_sweep_axis(axis, fl)
                sfields = fl.idual == 1 ? (:dslice, :eslice, :geslice, :vx, :vy, :vz) :
                                          (:dslice, :eslice, :vx, :vy, :vz)

                run_ax(name, ::Type{T}, field::Symbol) where {T} = begin
                    be = PPMKernels.backend(name)
                    dev(a) = PPMKernels.to_device(be, a, T)
                    d, e, ge = dev(d0), dev(e0), dev(ge0)
                    vx, vy, vz = dev(vx0), dev(vy0), dev(vz0)
                    PPMKernels.sweep_axis!(d, e, ge, vx, vy, vz, dev(p0), dev(gr0), dims, ng, axis;
                        dt = dt, gamma = gamma, dx = dx, gravity = fl.gravity, idual = fl.idual,
                        isteep = fl.isteep, iflatten = fl.iflatten, idiff = fl.idiff, eta2 = fl.eta2)
                    slot = (; dslice = d, eslice = e, geslice = ge, vx = vx, vy = vy, vz = vz)
                    PPMKernels.to_host(slot[field])
                end

                for field in sfields
                    reff = ref[field]
                    layerA!("grid.ax$axis.$ftag.$field", run_ax(:cpu, Float64, field), reff)
                    layerB!("grid.ax$axis.$ftag.$field", (nm, T) -> run_ax(nm, T, field))
                    layerC!("grid.ax$axis.$ftag.$field", (nm, T) -> run_ax(nm, T, field), reff)
                end
            end
        end

        # ── composed Strang step: x-only-varying IC ⇒ step ≡ 1-D Fortran sweep ──
        @testset "ppm_step_3d! reduces to 1-D on an x-only state" begin
            # rebuild an IC that varies only in x, with vy=vz=0 (y/z sweeps must no-op)
            d1 = zeros(N); e1 = zeros(N); ge1 = zeros(N)
            vx1 = zeros(N); gr1 = zeros(N)
            xprof_d = Float64[]; xprof_e = Float64[]; xprof_u = Float64[]; xprof_ge = Float64[]; xprof_gr = Float64[]
            for i in 1:nx
                t = tanh((i - 7) * 0.7)
                dd = 0.6 - 0.35 * t; pr = 0.6 - 0.45 * t; uu = 0.15 - 0.1 * t
                gg = pr / ((gamma - 1) * dd)
                push!(xprof_d, dd); push!(xprof_e, gg + 0.5 * uu^2); push!(xprof_u, uu)
                push!(xprof_ge, gg); push!(xprof_gr, 0.05 * sinpi((i - 1) / 7))
            end
            for k in 1:nz, j in 1:ny, i in 1:nx
                q = idx3(i, j, k)
                d1[q] = xprof_d[i]; e1[q] = xprof_e[i]; ge1[q] = xprof_ge[i]
                vx1[q] = xprof_u[i]; gr1[q] = xprof_gr[i]
            end

            # 1-D Fortran reference on the single x-pencil (dual+gravity on)
            rd = copy(xprof_d); re = copy(xprof_e); ru = copy(xprof_u)
            rv = zeros(nx); rw = zeros(nx); rge = copy(xprof_ge)
            rp = [(gamma - 1) * rd[i] * (re[i] - 0.5 * ru[i]^2) for i in 1:nx]
            EnzoLib.ppm_sweep_1d_full!(rd, re, ru, rv, rw, rp; i1 = ng + 1, i2 = nx - ng, dx = dx,
                dt = dt, gamma = gamma, geslice = rge, grslice = xprof_gr, gravity = 1, idual = 1, eta2 = 0.1)

            run_step(name, ::Type{T}, field::Symbol) where {T} = begin
                be = PPMKernels.backend(name)
                dev(a) = PPMKernels.to_device(be, a, T)
                zerg() = PPMKernels.device_zeros(be, T, (N,))
                d, e, ge = dev(d1), dev(e1), dev(ge1)
                vx, vy, vz = dev(vx1), zerg(), zerg()
                # gravity is x-only here (gry=grz=0) ⇒ the y/z sweeps stay exact no-ops
                PPMKernels.ppm_step_3d!(d, e, ge, vx, vy, vz, dev(gr1), zerg(), zerg(), dims, ng;
                    dt = dt, gamma = gamma, order = (1, 2, 3), gravity = 1, idual = 1, eta2 = 0.1)
                slot = (; dslice = d, eslice = e, vx = vx); PPMKernels.to_host(slot[field])
            end

            # interior cell (i, j0, k0) of the 3-D result must equal the 1-D pencil
            j0, k0 = 6, 6
            for (field, rprof) in ((:dslice, rd), (:eslice, re), (:vx, ru))
                got3 = run_step(:cpu, Float64, field)
                got = [got3[idx3(i, j0, k0)] for i in 1:nx]
                layerA!("grid.step.$field", got, rprof)
                # GPU consistency (Layer B analogue): cpu-f32 ≡ metal-f32 across the slab
                layerB!("grid.step.$field", (nm, T) -> run_step(nm, T, field))
            end
        end

        # ── conservation: directional sweeps are flux-conservative ──────────────
        # On a larger grid with a tightly-contained blob, the active-region
        # boundary sits in quiescent gas, so the net boundary flux (the only thing
        # that can change interior mass) is negligible and mass is conserved.
        @testset "mass conservation under a full step" begin
            mx = my = mz = 24
            cdims = (mx, my, mz); cN = mx * my * mz
            ci3(i, j, k) = i + mx * (j - 1) + mx * my * (k - 1)
            cd = zeros(cN); ce = zeros(cN); cge = zeros(cN)
            cvx = zeros(cN); cvy = zeros(cN); cvz = zeros(cN)
            for k in 1:mz, j in 1:my, i in 1:mx
                x = (i - 12.5) / 2; y = (j - 12.5) / 2; z = (k - 12.5) / 2   # width 2, centred
                r2 = x^2 + y^2 + z^2; b = exp(-r2)
                q = ci3(i, j, k)
                cd[q] = 1.0 + 0.30 * b
                pr = 0.60 + 0.20 * b
                cvx[q] = 0.10 * x * b; cvy[q] = 0.08 * y * b; cvz[q] = 0.06 * z * b
                cge[q] = pr / ((gamma - 1) * cd[q])
                ce[q] = cge[q] + 0.5 * (cvx[q]^2 + cvy[q]^2 + cvz[q]^2)
            end
            be = PPMKernels.backend(:cpu)
            dev(a) = PPMKernels.to_device(be, a, Float64)
            zc() = PPMKernels.device_zeros(be, Float64, (cN,))
            d, e, ge = dev(cd), dev(ce), dev(cge)
            vx, vy, vz = dev(cvx), dev(cvy), dev(cvz)
            m_before = PPMKernels.total_mass(d, cdims, ng, dx)
            PPMKernels.ppm_step_3d!(d, e, ge, vx, vy, vz, zc(), zc(), zc(), cdims, ng;
                                    dt = dt, gamma = gamma, order = (1, 2, 3))
            m_after = PPMKernels.total_mass(d, cdims, ng, dx)
            @test abs(m_after - m_before) / m_before < 1e-9
        end
    end
end
