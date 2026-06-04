# Phase 2.4 — twoshock (two-shock Riemann solver) vs the live Fortran.
#
# A spread of left/right interface states (varying contrast → both shock and
# rarefaction branches of the Newton iteration) is resolved and (pbar, ubar)
# certified on the A/B/C ladder. EnzoLib.twoshock is the live-Fortran reference.

using EnzoLib

@testset "Phase 2.4 — twoshock Riemann solver" begin
    if !EnzoLib.available()
        @test_skip "EnzoLib pilot library not built — skipping Fortran-reference twoshock layers"
    else
        idim, jdim = 20, 2
        i1, i2     = 3, 18
        j1, j2     = 1, jdim
        n          = idim * jdim
        gamma, pmin = 1.4, 1e-20
        lin(i, j)  = (j - 1) * idim + i

        # a range of mini-Riemann problems across the active region (Sod-like,
        # parameterised so successive interfaces sweep shock↔rarefaction).
        dls = zeros(n); drs = zeros(n); pls = zeros(n)
        prs = zeros(n); uls = zeros(n); urs = zeros(n)
        for j in 1:jdim, i in 1:idim
            k = lin(i, j)
            s = (i - i1) / (i2 - i1) + 0.05 * (j - 1)
            dls[k] = 1.00 - 0.50 * s;  pls[k] = 1.00 - 0.60 * s;  uls[k] = 0.20 * s
            drs[k] = 0.125 + 0.30 * s; prs[k] = 0.10 + 0.40 * s;  urs[k] = -0.10 + 0.10 * s
        end

        for (tag, ipf) in (("riemann", 0), ("presfree", 1))
            @testset "twoshock [$tag]" begin
                rp, ru = EnzoLib.twoshock(dls, drs, pls, prs, uls, urs;
                                          idim = idim, jdim = jdim, i1 = i1, i2 = i2, j1 = j1, j2 = j2,
                                          dt = 0.0, gamma = gamma, pmin = pmin, ipresfree = ipf)

                run_ts(name, ::Type{T}, which::Symbol) where {T} = begin
                    be = PPMKernels.backend(name)
                    dev(a) = PPMKernels.to_device(be, a, T)
                    pb = PPMKernels.device_zeros(be, T, (n,))
                    ub = PPMKernels.device_zeros(be, T, (n,))
                    PPMKernels.twoshock!(pb, ub, dev(dls), dev(drs), dev(pls), dev(prs),
                                         dev(uls), dev(urs); idim = idim, i1 = i1, i2 = i2,
                                         j1 = j1, j2 = j2, gamma = gamma, pmin = pmin, ipresfree = ipf)
                    PPMKernels.to_host(which === :p ? pb : ub)
                end

                for (which, ref, name) in ((:p, rp, "pbar"), (:u, ru, "ubar"))
                    runw(nm, ::Type{T}) where {T} = run_ts(nm, T, which)
                    layerA!("twoshock.$tag.$name", run_ts(:cpu, Float64, which), ref)
                    layerB!("twoshock.$tag.$name", runw)
                    layerC!("twoshock.$tag.$name", runw, ref)
                end
            end
        end
    end
end
