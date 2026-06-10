# ── Phase 5 gate (ADR-0006 flagship 2): RAMSES-RT inside an Enzo simulation ───
#
# Enzo hosts the problem; a persistent RAMSES-RT guest provides the radiation +
# photo-chemistry slot, its ionization state written into Enzo's LIVE species
# fields every cycle.  Gates:
#   (a) uniform field — the front measured from the ENZO-HELD fields tracks the
#       analytic Strömgren solution (the write-back contract carries physics);
#       the host's density is untouched to the bit (the guest owns chemistry
#       ONLY — the slot boundary is a data contract).
#   (b) structured field — the same density through Moray and through the
#       RAMSES-RT guest agree on the I-front within the inter-code band.
#   (c) host validity — after the run, Enzo's OWN machinery (boundary set +
#       its native cooling slot) operates on the guest-written state.

using Test
using MultiCode
using EnzoLib, RamsesLib, CodeBridge

@testset "ADR-0006 flagship 2: RAMSES-RT as Enzo's radiation slot" begin
    ok = EnzoLib.grid_available() && isfile(MultiCode.ENZO_PHOTONTEST_PF) &&
         CodeBridge.available(RamsesLib.BRIDGE, :rt)
    if !ok
        @test_skip false
    else
        @testset "(a) uniform field: physics through the write-back contract" begin
            r = run_enzo_host_ramsesrt(t_end_myr = 5.0, snapshots = [3.0, 5.0])
            try
                ρ0 = r.fields.density
                for (t, ri) in r.history
                    @test isfinite(ri)
                    @test abs(ri - stromgren_radius(t)) / stromgren_radius(t) < 0.12
                end
                @test issorted([ri for (_, ri) in r.history])
                @test maximum(r.fields.xHII) > 0.99           # host fields carry the HII region
                @test minimum(r.fields.xHII) < 0.01
                @test all(ρ0 .== 1.0)                          # host density untouched
                # (c) the host is still a valid Enzo simulation: its own
                # boundary + native cooling machinery run on the guest state
                @test EnzoLib.session_set_boundary(r.enzo, 0) >= 0
                EnzoLib.session_solve_cooling(r.enzo, 0)
                hi = MultiCode._enzo_field_active(r.enzo, MultiCode.FT_HI)
                @test all(isfinite, hi) && all(hi .>= 0)
                @info "flagship 2 (uniform)" history = r.history analytic = [stromgren_radius(t) for (t, _) in r.history]
            finally
                r.free()
            end
        end

        @testset "(b) structured field: guest vs Moray on the same density" begin
            n = 32
            den = [i <= n ÷ 2 ? 0.125 : 1.0 for i in 1:n, j in 1:n, k in 1:n]  # light gas at the source
            # the CONTIGUOUS ionized run from the source — immune to the skin of
            # ionization that periodic wrap-around deposits at the far corner
            # (an M1 artifact a ray code doesn't share; findlast would jump there)
            ray_front(x) = (c = findfirst(<(0.5), x[:, 1, 1]); c === nothing ? 1.0 : (c - 1) / n)

            m = run_moray_stromgren(t_end_myr = 5.0, snapshots = [5.0], density = den)
            x_moray = ray_front(m.fields.xHII)
            m.free()

            # higher reduced c here: in the LIGHT gas the front outruns 0.005c
            # (M1 would be transport-limited, not physics-limited); the dense
            # half absorbs wrap-around photons (mfp ≈ 0.05 kpc), so the seam
            # cannot contaminate the measured ray.
            r = run_enzo_host_ramsesrt(density = den, t_end_myr = 5.0, snapshots = [5.0],
                                       c_fraction = 0.05)
            try
                @test maximum(abs.(r.fields.density .- den)) < 1e-12   # host ran THE field
                x_guest = ray_front(r.fields.xHII)
                @test x_moray > 0 && x_guest > 0
                @test abs(x_guest - x_moray) / ((x_guest + x_moray) / 2) < 0.15
                @info "flagship 2 (structured)" x_moray x_guest ratio = x_guest / x_moray
            finally
                r.free()
            end
        end
    end
end
