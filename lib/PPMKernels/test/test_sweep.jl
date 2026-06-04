# Phase 3 — the composed 1-D directional PPM sweep vs the live Fortran.
#
# The whole calcdiss→inteuler→twoshock→flux_twoshock→euler chain is run end to
# end and certified against EnzoLib.ppm_sweep_1d_full! (which drives the identical
# legacy chain). This is the integration test: every component kernel feeds the
# next, on-device, and the post-dt zone-centred state + face fluxes must match the
# Fortran bit-tight (A), CPU≡Metal in f32 (B), and Metal-f32 vs Fortran (C).

using EnzoLib

@testset "Phase 3 — composed ppm_sweep_1d" begin
    if !EnzoLib.available()
        @test_skip "EnzoLib pilot library not built — skipping Fortran-reference sweep layers"
    else
        idim       = 28
        i1, i2     = 5, 23                                  # ≥4 ghosts each side
        n          = idim
        gamma, dx, dt = 1.4, 1.0, 0.04

        # zone-centred shock tube; pressure is the exact pgas2d EOS of the state
        d0 = zeros(n); u0 = zeros(n); v0 = zeros(n); w0 = zeros(n)
        e0 = zeros(n); p0 = zeros(n); ge0 = zeros(n); gr0 = zeros(n)
        for i in 1:idim
            t = tanh((i - 14.0) * 0.6)
            d0[i] = 0.5625 - 0.4375 * t
            pr    = 0.5500 - 0.4500 * t                     # internal pressure
            u0[i] = 0.20  - 0.10 * t
            v0[i] = 0.08
            w0[i] = -0.05
            ge0[i] = pr / ((gamma - 1) * d0[i])             # specific internal energy
            e0[i]  = ge0[i] + 0.5 * (u0[i]^2 + v0[i]^2 + w0[i]^2)
            p0[i]  = (gamma - 1) * d0[i] * (e0[i] - 0.5 * (u0[i]^2 + v0[i]^2 + w0[i]^2))
            gr0[i] = 0.25 * sinpi((i - 1) / 9)
        end
        dxi = fill(dx, idim)

        base = (:dslice, :eslice, :uslice, :vslice, :wslice)
        cases = (
            ("plain",     (gravity=0, idual=0, isteep=0, iflatten=0, idiff=0, ipresfree=0, dfloor=0.0, eta2=0.0), base),
            ("diff",      (gravity=0, idual=0, isteep=0, iflatten=0, idiff=1, ipresfree=0, dfloor=0.0, eta2=0.0), base),
            ("flatten",   (gravity=0, idual=0, isteep=0, iflatten=3, idiff=0, ipresfree=0, dfloor=0.0, eta2=0.0), base),
            ("steepen",   (gravity=0, idual=0, isteep=1, iflatten=0, idiff=0, ipresfree=0, dfloor=0.0, eta2=0.0), base),
            ("gravity",   (gravity=1, idual=0, isteep=0, iflatten=0, idiff=0, ipresfree=0, dfloor=0.0, eta2=0.0), base),
            ("dual",      (gravity=0, idual=1, isteep=0, iflatten=0, idiff=0, ipresfree=0, dfloor=0.0, eta2=0.1), (base..., :geslice)),
            ("dfloor",    (gravity=0, idual=0, isteep=0, iflatten=0, idiff=0, ipresfree=0, dfloor=0.5, eta2=0.0), base),
            ("full",      (gravity=1, idual=1, isteep=1, iflatten=3, idiff=1, ipresfree=0, dfloor=0.0, eta2=0.1), (base..., :geslice)),
        )

        for (tag, fl, sfields) in cases
            @testset "ppm_sweep_1d [$tag]" begin
                # Fortran reference (mutates copies in place; returns df,ef,uf)
                ds = copy(d0); es = copy(e0); us = copy(u0); vs = copy(v0); ws = copy(w0); ge = copy(ge0)
                rdf, ref, ruf = EnzoLib.ppm_sweep_1d_full!(ds, es, us, vs, ws, copy(p0);
                    i1 = i1, i2 = i2, dx = dx, dt = dt, gamma = gamma, geslice = ge, grslice = gr0,
                    gravity = fl.gravity, idual = fl.idual, eta2 = fl.eta2, isteep = fl.isteep,
                    iflatten = fl.iflatten, idiff = fl.idiff, ipresfree = fl.ipresfree,
                    dfloor = fl.dfloor, fluxes = true)
                refstate = (; dslice = ds, eslice = es, uslice = us, vslice = vs, wslice = ws, geslice = ge,
                            df = rdf, ef = ref, uf = ruf)

                run_sw(name, ::Type{T}, field::Symbol) where {T} = begin
                    be = PPMKernels.backend(name)
                    dev(a) = PPMKernels.to_device(be, a, T)
                    sd, se, sge = dev(d0), dev(e0), dev(ge0)
                    su, sv, sw = dev(u0), dev(v0), dev(w0)
                    df, ef, uf = PPMKernels.ppm_sweep_1d!(sd, se, sge, su, sv, sw, dev(p0),
                        dev(gr0), dev(dxi); idim = idim, i1 = i1, i2 = i2, dt = dt, gamma = gamma,
                        gravity = fl.gravity, idual = fl.idual, eta2 = fl.eta2, isteep = fl.isteep,
                        iflatten = fl.iflatten, idiff = fl.idiff, ipresfree = fl.ipresfree,
                        dfloor = fl.dfloor)
                    slot = (; dslice = sd, eslice = se, uslice = su, vslice = sv, wslice = sw,
                            geslice = sge, df = df, ef = ef, uf = uf)
                    PPMKernels.to_host(slot[field])
                end

                for field in (sfields..., :df, :ef, :uf)
                    reff = refstate[field]
                    layerA!("sweep.$tag.$field", run_sw(:cpu, Float64, field), reff)
                    layerB!("sweep.$tag.$field", (nm, T) -> run_sw(nm, T, field))
                    layerC!("sweep.$tag.$field", (nm, T) -> run_sw(nm, T, field), reff)
                end
            end
        end
    end
end
