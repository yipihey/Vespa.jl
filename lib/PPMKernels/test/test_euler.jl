# Phase 2.6 — euler (conservative flux-divergence update) vs the live Fortran.
#
# The fluxes come from the full certified reference chain (inteuler → twoshock →
# flux_twoshock), so the update kernel is exercised on the data it sees in the
# composed sweep. The six updated zone-centred slices are certified on the A/B/C
# ladder across the gravity × dual-energy × density-floor matrix.

using EnzoLib

@testset "Phase 2.6 — euler conservative update" begin
    if !EnzoLib.available()
        @test_skip "EnzoLib pilot library not built — skipping Fortran-reference euler layers"
    else
        idim, jdim = 24, 2
        i1, i2     = 5, 19
        j1, j2     = 1, jdim
        n          = idim * jdim
        gamma, dt  = 1.4, 0.05
        lin(i, j)  = (j - 1) * idim + i

        d0 = zeros(n); p0 = zeros(n); u0 = zeros(n); v0 = zeros(n); w0 = zeros(n)
        e0 = zeros(n); ge0 = zeros(n); gr0 = zeros(n)
        for j in 1:jdim, i in 1:idim
            k = lin(i, j)
            t = tanh((i - 12.0) * 0.7 + 0.10 * (j - 1))
            d0[k] = 0.5625 - 0.4375 * t
            p0[k] = 0.5500 - 0.4500 * t
            u0[k] = 0.20  - 0.10 * t
            v0[k] = 0.10  + 0.05 * j
            w0[k] = -0.05
            e0[k]  = p0[k] / ((gamma - 1) * d0[k]) + 0.5 * (u0[k]^2 + v0[k]^2 + w0[k]^2)
            ge0[k] = p0[k] / ((gamma - 1) * d0[k])
            gr0[k] = 0.30 * sinpi((i - 1) / 8)
        end
        dxi = ones(idim)

        # fluxes from the certified reference chain (idual=1, gravity inputs present)
        rec = EnzoLib.inteuler(d0, p0, u0, v0, w0; idim = idim, jdim = jdim, i1 = i1, i2 = i2,
                               j1 = j1, j2 = j2, dt = dt, gamma = gamma, geslice = ge0,
                               grslice = gr0, gravity = 1, idual = 1, eta2 = 0.1, dxi = dxi)
        pbar, ubar = EnzoLib.twoshock(rec.dls, rec.drs, rec.pls, rec.prs, rec.uls, rec.urs;
                                      idim = idim, jdim = jdim, i1 = i1, i2 = i2, j1 = j1, j2 = j2,
                                      dt = dt, gamma = gamma)
        fx = EnzoLib.flux_twoshock(d0, e0, ge0, u0, v0, w0,
                                   rec.dls, rec.drs, rec.pls, rec.prs, rec.gels, rec.gers,
                                   rec.uls, rec.urs, rec.vls, rec.vrs, rec.wls, rec.wrs, pbar, ubar;
                                   idim = idim, jdim = jdim, i1 = i1, i2 = i2, j1 = j1, j2 = j2,
                                   dt = dt, gamma = gamma, dx = dxi, idual = 1)

        upd = (:dslice, :eslice, :geslice, :uslice, :vslice, :wslice)
        cases = (("plain",      (gravity=0, idual=0, dfloor=0.0)),
                 ("dual",       (gravity=0, idual=1, dfloor=0.0)),
                 ("gravity",    (gravity=1, idual=0, dfloor=0.0)),
                 ("grav+dual",  (gravity=1, idual=1, dfloor=0.0)),
                 ("dfloor",     (gravity=0, idual=0, dfloor=0.5)))

        for (tag, fl) in cases
            @testset "euler [$tag]" begin
                ref = EnzoLib.euler(d0, e0, ge0, u0, v0, w0, gr0,
                                    fx.df, fx.ef, fx.uf, fx.vf, fx.wf, fx.gef, fx.ges;
                                    idim = idim, jdim = jdim, i1 = i1, i2 = i2, j1 = j1, j2 = j2,
                                    dt = dt, gamma = gamma, dx = dxi, gravity = fl.gravity,
                                    idual = fl.idual, dfloor = fl.dfloor)

                run_eu(name, ::Type{T}, field::Symbol) where {T} = begin
                    be = PPMKernels.backend(name)
                    dev(a) = PPMKernels.to_device(be, a, T)
                    ds, es, ge = dev(d0), dev(e0), dev(ge0)
                    us, vs, ws = dev(u0), dev(v0), dev(w0)
                    PPMKernels.euler!(ds, es, ge, us, vs, ws,
                                      dev(fx.df), dev(fx.ef), dev(fx.uf), dev(fx.vf), dev(fx.wf),
                                      dev(fx.gef), dev(fx.ges), dev(gr0), dev(dxi);
                                      idim = idim, i1 = i1, i2 = i2, j1 = j1, j2 = j2, dt = dt,
                                      gravity = fl.gravity, idual = fl.idual, dfloor = fl.dfloor)
                    slot = (; dslice = ds, eslice = es, geslice = ge, uslice = us, vslice = vs, wslice = ws)
                    PPMKernels.to_host(slot[field])
                end

                fields = fl.idual == 1 ? upd : (:dslice, :eslice, :uslice, :vslice, :wslice)
                for field in fields
                    reff = ref[field]
                    layerA!("euler.$tag.$field", run_eu(:cpu, Float64, field), reff)
                    layerB!("euler.$tag.$field", (nm, T) -> run_eu(nm, T, field))
                    layerC!("euler.$tag.$field", (nm, T) -> run_eu(nm, T, field), reff)
                end
            end
        end
    end
end
