# Phase 2.5 — flux_twoshock (Eulerian fluxes) vs the live Fortran.
#
# Inputs are physically consistent: the L/R interface states come from the
# certified EnzoLib.inteuler reference and the resolved (pbar,ubar) from
# EnzoLib.twoshock, run on a zone-centred shock tube — exactly what the kernel
# sees in the composed sweep. Fluxes are certified on the A/B/C ladder across
# the diffusion × dual-energy matrix. The gas-energy source `ges` reads ub[i+1],
# so it is certified over i1..i2 (the i2+1 edge is undefined in the Fortran too).

using EnzoLib

@testset "Phase 2.5 — flux_twoshock Eulerian fluxes" begin
    if !EnzoLib.available()
        @test_skip "EnzoLib pilot library not built — skipping Fortran-reference flux_twoshock layers"
    else
        idim, jdim = 24, 2
        i1, i2     = 5, 19
        j1, j2     = 1, jdim
        n          = idim * jdim
        gamma, dt  = 1.4, 0.05
        lin(i, j)  = (j - 1) * idim + i

        d0 = zeros(n); p0 = zeros(n); u0 = zeros(n); v0 = zeros(n); w0 = zeros(n)
        e0 = zeros(n); ge0 = zeros(n); dc = zeros(n)
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
            dc[k]  = 0.08 * abs(sinpi((i - 1) / 6))           # synthetic diffusion coeff
        end
        dxi = ones(idim)

        # certified reference chain → consistent L/R states and resolved (pbar,ubar)
        rec = EnzoLib.inteuler(d0, p0, u0, v0, w0; idim = idim, jdim = jdim, i1 = i1, i2 = i2,
                               j1 = j1, j2 = j2, dt = dt, gamma = gamma, geslice = ge0,
                               idual = 1, eta2 = 0.1, dxi = dxi)
        pbar, ubar = EnzoLib.twoshock(rec.dls, rec.drs, rec.pls, rec.prs, rec.uls, rec.urs;
                                      idim = idim, jdim = jdim, i1 = i1, i2 = i2, j1 = j1, j2 = j2,
                                      dt = dt, gamma = gamma)

        full_idx = [lin(i, j) for j in 1:jdim for i in i1:i2 + 1]   # df..wf,gef valid here
        ges_idx  = [lin(i, j) for j in 1:jdim for i in i1:i2]       # ges valid here only

        for (tag, idiff, idual) in (("plain", 0, 0), ("diff", 1, 0),
                                    ("dual", 0, 1), ("diff+dual", 1, 1))
            @testset "flux_twoshock [$tag]" begin
                ref = EnzoLib.flux_twoshock(d0, e0, ge0, u0, v0, w0,
                                            rec.dls, rec.drs, rec.pls, rec.prs, rec.gels, rec.gers,
                                            rec.uls, rec.urs, rec.vls, rec.vrs, rec.wls, rec.wrs,
                                            pbar, ubar; idim = idim, jdim = jdim, i1 = i1, i2 = i2,
                                            j1 = j1, j2 = j2, dt = dt, gamma = gamma, dx = dxi,
                                            diffcoef = dc, idiff = idiff, idual = idual)

                run_fx(name, ::Type{T}, field::Symbol) where {T} = begin
                    be = PPMKernels.backend(name)
                    dev(a) = PPMKernels.to_device(be, a, T)
                    out = (; (f => PPMKernels.device_zeros(be, T, (n,)) for f in
                              (:df, :ef, :uf, :vf, :wf, :gef, :ges))...)
                    PPMKernels.flux_twoshock!(out, dev(rec.dls), dev(rec.drs), dev(rec.pls),
                        dev(rec.prs), dev(rec.gels), dev(rec.gers), dev(rec.uls), dev(rec.urs),
                        dev(rec.vls), dev(rec.vrs), dev(rec.wls), dev(rec.wrs), dev(pbar), dev(ubar),
                        dev(d0), dev(u0), dev(v0), dev(w0), dev(e0), dev(ge0), dev(dxi), dev(dc);
                        idim = idim, i1 = i1, i2 = i2, j1 = j1, j2 = j2, dt = dt, gamma = gamma,
                        idiff = idiff, idual = idual)
                    PPMKernels.to_host(out[field])
                end

                fields = idual == 1 ? (:df, :ef, :uf, :vf, :wf, :gef, :ges) : (:df, :ef, :uf, :vf, :wf)
                for field in fields
                    sel = field === :ges ? ges_idx : full_idx
                    refsel = ref[field][sel]
                    layerA!("flux.$tag.$field", run_fx(:cpu, Float64, field)[sel], refsel)
                    layerB!("flux.$tag.$field", (nm, T) -> run_fx(nm, T, field)[sel])
                    layerC!("flux.$tag.$field", (nm, T) -> run_fx(nm, T, field)[sel], refsel)
                end
            end
        end
    end
end
