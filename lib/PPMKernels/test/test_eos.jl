# Phase 2.1 — EOS (pgas2d / pgas2d_dual) certified against the live Fortran.
#
# The Fortran reference is EnzoLib's `ccall` binding into the same legacy kernel
# the production solver uses; the KA port is diffed against it on the three-layer
# ladder (A: cpu-f64 vs Fortran, B: cpu-f32 ≡ metal-f32, C: metal-f32 vs Fortran).
#
# Two datasets, by design:
#   • MODERATE  — O(1) energies/velocities, no total-vs-internal cancellation.
#     Used for plain `pgas2d`, whose single-precision path has no protection and
#     so demands well-conditioned inputs to certify Layer C.
#   • STRESS    — a cold, fast band where internal ≪ kinetic energy, AND a gas
#     energy `geslice` deliberately DRIFTED off the total-energy-implied value.
#     This fires BOTH dual-energy branches (the gas-energy substitution through
#     cancellation, and the `geslice` overwrite), genuinely mutates `eslice` in
#     the left-to-right sweep, and shows the dual formalism staying f32-accurate
#     where plain EOS cannot.

using EnzoLib

@testset "Phase 2.1 — EOS pgas2d / pgas2d_dual" begin
    if !EnzoLib.available()
        @test_skip "EnzoLib pilot library not built — skipping Fortran-reference EOS layers"
    else
        EnzoLib.check_precision()                          # (8-byte baryons, 4-byte ints)

        # ── slab geometry: column-major idim×jdim, active i1..i2 over all rows ──
        idim, jdim = 16, 3
        i1, i2     = 4, 13
        j1, j2     = 1, jdim
        n          = idim * jdim
        gamma, pmin = 1.4, 1e-20
        eta1, eta2  = 1e-3, 1e-1
        lin(i, j)   = (j - 1) * idim + i                   # 1-based column-major index

        # ── MODERATE dataset (well-conditioned for plain f32 EOS) ──────────────
        dM = zeros(n); eM = zeros(n); uM = zeros(n); vM = zeros(n); wM = zeros(n)
        for j in 1:jdim, i in 1:idim
            k = lin(i, j)
            dM[k] = 1.0 + 0.10 * i + 0.05 * j
            uM[k] = 0.30 + 0.05 * i
            vM[k] = 0.10 + 0.01 * j
            wM[k] = 0.20
            gint  = 0.50 + 0.02 * i                        # internal ~ O(1): no cancellation
            eM[k] = gint + 0.5 * (uM[k]^2 + vM[k]^2 + wM[k]^2)
        end

        # ── STRESS dataset (cold/fast band + drifted gas energy) ───────────────
        dS = zeros(n); eS = zeros(n); geS = zeros(n)
        uS = zeros(n); vS = zeros(n); wS = zeros(n)
        for j in 1:jdim, i in 1:idim
            k    = lin(i, j)
            dS[k] = 1.0 + 0.10 * i + 0.05 * j
            fast  = 7 <= i <= 10
            uS[k] = fast ? 20.0 : 0.30
            vS[k] = 0.10 + 0.01 * j
            wS[k] = 0.20
            gint  = fast ? 1.0e-3 : 0.50                   # tiny internal energy in the fast band
            ke    = 0.5 * (uS[k]^2 + vS[k]^2 + wS[k]^2)
            eS[k]  = gint + ke
            geS[k] = gint * 1.07                           # DRIFT ⇒ reconciliation must move eslice
        end

        # ── pgas2d (local EOS) on the moderate data ────────────────────────────
        @testset "pgas2d" begin
            ref = EnzoLib.pgas2d(dM, eM, uM, vM, wM;
                                 idim = idim, jdim = jdim, i1 = i1, i2 = i2,
                                 j1 = j1, j2 = j2, gamma = gamma, pmin = pmin)

            run_p(name, ::Type{T}) where {T} = begin
                be = PPMKernels.backend(name)
                ps = PPMKernels.device_zeros(be, T, (n,))
                d  = PPMKernels.to_device(be, dM, T); e = PPMKernels.to_device(be, eM, T)
                u  = PPMKernels.to_device(be, uM, T); v = PPMKernels.to_device(be, vM, T)
                w  = PPMKernels.to_device(be, wM, T)
                PPMKernels.pgas2d!(ps, d, e, u, v, w; idim = idim, i1 = i1, i2 = i2,
                                   j1 = j1, j2 = j2, gamma = gamma, pmin = pmin)
                PPMKernels.to_host(ps)
            end

            layerA!("pgas2d.p", run_p(:cpu, Float64), ref)
            layerB!("pgas2d.p", run_p)
            layerC!("pgas2d.p", run_p, ref)
        end

        # ── pgas2d pressure floor: pmin high enough to clamp every active cell ──
        @testset "pgas2d (pmin floor)" begin
            pfloor = 1.0e6                                  # ≫ any physical p here
            ref = EnzoLib.pgas2d(dM, eM, uM, vM, wM;
                                 idim = idim, jdim = jdim, i1 = i1, i2 = i2,
                                 j1 = j1, j2 = j2, gamma = gamma, pmin = pfloor)
            @test all(ref[lin(i, j)] == pfloor for j in j1:j2, i in i1:i2)   # floor really fires

            run_p(name, ::Type{T}) where {T} = begin
                be = PPMKernels.backend(name)
                ps = PPMKernels.device_zeros(be, T, (n,))
                d  = PPMKernels.to_device(be, dM, T); e = PPMKernels.to_device(be, eM, T)
                u  = PPMKernels.to_device(be, uM, T); v = PPMKernels.to_device(be, vM, T)
                w  = PPMKernels.to_device(be, wM, T)
                PPMKernels.pgas2d!(ps, d, e, u, v, w; idim = idim, i1 = i1, i2 = i2,
                                   j1 = j1, j2 = j2, gamma = gamma, pmin = pfloor)
                PPMKernels.to_host(ps)
            end
            layerA!("pgas2d.floor", run_p(:cpu, Float64), ref)
            layerC!("pgas2d.floor", run_p, ref)
        end

        # ── pgas2d_dual (sweep-dependent, three outputs) on the stress data ─────
        @testset "pgas2d_dual" begin
            ref_e, ref_ge, ref_p =
                EnzoLib.pgas2d_dual(dS, eS, geS, uS, vS, wS;
                                    eta1 = eta1, eta2 = eta2, idim = idim, jdim = jdim,
                                    i1 = i1, i2 = i2, j1 = j1, j2 = j2,
                                    gamma = gamma, pmin = pmin)
            # non-vacuity: the reconciliation actually MOVED total energy (the
            # gas-energy substitution fired in the cold band) and rewrote geslice
            # (the overwrite branch fired in the subsonic band).
            @test ref_e  != eS
            @test ref_ge != geS

            run_dual(name, ::Type{T}, which::Symbol) where {T} = begin
                be = PPMKernels.backend(name)
                es = PPMKernels.to_device(be, eS, T); g = PPMKernels.to_device(be, geS, T)
                ps = PPMKernels.device_zeros(be, T, (n,))
                d  = PPMKernels.to_device(be, dS, T)
                u  = PPMKernels.to_device(be, uS, T); v = PPMKernels.to_device(be, vS, T)
                w  = PPMKernels.to_device(be, wS, T)
                PPMKernels.pgas2d_dual!(es, g, ps, d, u, v, w; idim = idim, i1 = i1,
                                        i2 = i2, j1 = j1, j2 = j2, eta1 = eta1,
                                        eta2 = eta2, gamma = gamma, pmin = pmin)
                PPMKernels.to_host(which === :e ? es : which === :ge ? g : ps)
            end

            for (which, ref, tag) in ((:e, ref_e, "e"), (:ge, ref_ge, "ge"), (:p, ref_p, "p"))
                runw(name, ::Type{T}) where {T} = run_dual(name, T, which)
                layerA!("pgas2d_dual.$tag", run_dual(:cpu, Float64, which), ref)
                layerB!("pgas2d_dual.$tag", runw)
                layerC!("pgas2d_dual.$tag", runw, ref)
            end
        end
    end
end
