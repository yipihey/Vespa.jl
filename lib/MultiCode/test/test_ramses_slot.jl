# ── Phase 3 gate (ADR-0006 D4): PPMKernels hydro INSIDE RAMSES ────────────────
#
# Certification in the ADR-0002 style, adapted for a cross-SCHEME slot (the
# guest is a different — Enzo-certified — MUSCL-Hancock than the host's, so the
# gates are scheme-tolerance gates, not bit-identity):
#
#  per-step  — from the SAME developed state, one native `godunov_fine!` step
#              vs one guest step with the SAME dt: conserved totals must agree
#              to round-off (both schemes are conservative on a periodic box),
#              raster round-trip must be exact, and the one-step solution
#              difference must be at truncation level (tiny L1, bounded max at
#              the shock).
#  per-run   — the full Sod through the guest slot must conserve to round-off
#              and match the EXACT solution about as well as the native run
#              (L1 within 1.5×).

using Test
using MultiCode
using RamsesLib

const slot_spec = SodSpec()
const SLOT_LEVEL = 6                  # 64³ — the certification resolution

@testset "ADR-0006 D4: PPMKernels-in-RAMSES slot" begin
    if !RamsesLib.available()
        @test_skip false
    else
        @testset "raster round-trip is exact" begin
            r = run_ramses_sod(slot_spec; level = SLOT_LEVEL)   # a developed state
            try
                ras = MultiCode.ramses_raster(r.handle; lev = r.diag.level)
                # interior of the raster reproduces the canonical extraction exactly
                cs = ramses_extract(r.handle; lev = r.diag.level, boxlen = 2.0)
                @test sum(ras.D) ≈ sum(cs.rho) rtol = 1e-13     # ghosts are zero pre-fill
                MultiCode.ramses_deraster!(r.handle, ras; lev = r.diag.level)
                cs2 = ramses_extract(r.handle; lev = r.diag.level, boxlen = 2.0)
                @test cs2.rho == cs.rho && cs2.mom == cs.mom && cs2.etot == cs.etot
            finally
                r.free()
            end
        end

        @testset "per-step: guest vs native from the same state" begin
            # develop structure first (half the comparison time), then branch
            half = SodSpec(t = slot_spec.t / 2)
            r = run_ramses_sod(half; level = SLOT_LEVEL)
            try
                h = r.handle; lev = r.diag.level
                ck, U0 = RamsesLib.get_hydro_all(h, :uold, lev)
                RamsesLib.newdt_fine!(h, lev)
                dt = RamsesLib.get_dt(h, lev).dtnew
                RamsesLib.set_dt!(h, lev, dt)

                # (a) NATIVE branch: one godunov_fine-centered step
                RamsesLib.hydro_step!(h, lev; dt = dt)
                _, Unat = RamsesLib.get_hydro_all(h, :uold, lev)

                # restore the branch point exactly, then (b) the GUEST step
                for v in 1:5
                    RamsesLib.set_hydro!(h, :uold, v, lev, ck, U0[:, :, v])
                end
                RamsesLib.set_dt!(h, lev, dt)
                ramses_ppmk_hydro_step!(h; lev = lev, dt = dt, gamma = slot_spec.gamma,
                                        boxlen = 2.0)
                _, Ugst = RamsesLib.get_hydro_all(h, :uold, lev)

                # conservation: both updates preserve the totals to round-off
                m0 = sum(U0[:, :, 1]); mn = sum(Unat[:, :, 1]); mg = sum(Ugst[:, :, 1])
                e0 = sum(U0[:, :, 5]); en = sum(Unat[:, :, 5]); eg = sum(Ugst[:, :, 5])
                @test abs(mn - m0) / m0 < 1e-12
                @test abs(mg - m0) / m0 < 1e-12
                @test abs(en - e0) / e0 < 1e-11
                @test abs(eg - e0) / e0 < 1e-11

                # scheme tolerance: the two second-order updates differ at
                # truncation level.  The honest scale is the step's OWN update
                # size — the inter-scheme difference must be a small fraction
                # of what the step itself changed (limiter/splitting effects),
                # bounded pointwise at the shock.
                dρ = abs.(Ugst[:, :, 1] .- Unat[:, :, 1])
                upd = abs.(Unat[:, :, 1] .- U0[:, :, 1])
                stepped = count(>(1e-14), upd)
                @test stepped > 0                              # the step did something
                l1 = sum(dρ) / length(dρ)
                l1_upd = sum(upd) / length(upd)
                @test l1 < 0.15 * l1_upd                       # ≤15% of the update itself
                @test l1 < 1e-3                                # absolute belt
                @test maximum(dρ) < 0.05
                @info "per-step certification" dt l1_rho = l1 l1_update = l1_upd ratio = l1 / l1_upd max_drho = maximum(dρ) cells_updated = stepped
            finally
                r.free()
            end
        end

        l1_cpu_guest = Ref(NaN)
        @testset "per-run: guest Sod matches the exact solution like the native" begin
            nat = run_ramses_sod(slot_spec; level = SLOT_LEVEL)
            l1_nat = try
                sod_l1(nat.profile, slot_spec, nat.t)
            finally
                nat.free()
            end
            gst = run_ramses_sod_guest(slot_spec; level = SLOT_LEVEL)
            try
                lg = ledger(gst.cs)
                ref = MultiCode.sod_reference_ledger(slot_spec)
                @test abs(lg.mass - ref.mass) / ref.mass < 1e-10
                @test abs(lg.energy - ref.energy) / ref.energy < 1e-10
                l1_gst = sod_l1(gst.profile, slot_spec, gst.t)
                l1_cpu_guest[] = l1_gst.rho
                @test l1_gst.rho < 1.5 * l1_nat.rho + 1e-4
                @test l1_gst.u < 1.5 * l1_nat.u + 1e-4
                # transverse symmetry survives the raster/scheme swap
                @test gst.profile.scatter < 1e-10
                @info "per-run certification (host=ramses, guest=ppmkernels)" steps = gst.diag.steps l1_native = l1_nat.rho l1_guest = l1_gst.rho
            finally
                gst.free()
            end
        end

        @testset "the guest slot on Metal (f32 GPU inside f64 RAMSES)" begin
            metal_ok = try
                @eval using Metal
                @eval using PPMKernels
                PPMKernels.has_backend(:metal)
            catch
                false
            end
            if !metal_ok
                @test_skip false
            else
                gm = run_ramses_sod_guest(slot_spec; level = SLOT_LEVEL, device = :metal)
                try
                    lg = ledger(gm.cs)
                    ref = MultiCode.sod_reference_ledger(slot_spec)
                    # f32 arithmetic per step, f64 ledger: round-off accumulates
                    # at the f32 floor across ~60 steps on 260k cells
                    @test abs(lg.mass - ref.mass) / ref.mass < 1e-3
                    @test abs(lg.energy - ref.energy) / ref.energy < 1e-3
                    l1_m = sod_l1(gm.profile, slot_spec, gm.t)
                    @test isfinite(l1_m.rho)
                    @test l1_m.rho < 1.1 * l1_cpu_guest[] + 1e-3    # f32 ≈ f64 physics
                    @info "Metal guest slot" steps = gm.diag.steps l1_metal = l1_m.rho l1_cpu = l1_cpu_guest[]
                finally
                    gm.free()
                end
            end
        end
    end
end
