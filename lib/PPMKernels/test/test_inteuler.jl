# Phase 2.3 — inteuler (PPM reconstruction) vs the live Fortran.
#
# A shock-tube profile over TWO rows (exercising the column-major (i,j) indexing
# in every kernel) is reconstructed and the characteristic-corrected interface
# states are certified on the A/B/C ladder, across the supported flag matrix:
# baseline, gravity, dual energy, flattening, steepening, pressure-free.

using EnzoLib

@testset "Phase 2.3 — inteuler PPM reconstruction" begin
    if !EnzoLib.available()
        @test_skip "EnzoLib pilot library not built — skipping Fortran-reference inteuler layers"
    else
        idim, jdim = 24, 2
        i1, i2     = 5, 19                                  # ≥4 ghosts each side
        j1, j2     = 1, jdim
        n          = idim * jdim
        gamma, dt  = 1.4, 0.05
        lin(i, j)  = (j - 1) * idim + i

        d0 = zeros(n); p0 = zeros(n); u0 = zeros(n); v0 = zeros(n); w0 = zeros(n)
        ge0 = zeros(n); gr0 = zeros(n); flat0 = zeros(n)
        for j in 1:jdim, i in 1:idim
            k = lin(i, j)
            t = tanh((i - 12.0) * 0.7 + 0.10 * (j - 1))     # shock, mild row variation
            d0[k] = 0.5625 - 0.4375 * t
            p0[k] = 0.5500 - 0.4500 * t
            u0[k] = 0.20  - 0.10 * t
            v0[k] = 0.10  + 0.05 * j
            w0[k] = -0.05
            ge0[k] = p0[k] / ((gamma - 1) * d0[k])           # gas energy = internal
            gr0[k] = 0.30 * sinpi((i - 1) / 8)               # gravity accel
            flat0[k] = clamp(0.6 - abs(i - 12) * 0.15, 0.0, 1.0)  # synthetic flattener bump
        end
        dxi = ones(idim)

        base_fields = (:dls, :drs, :pls, :prs, :uls, :urs, :vls, :vrs, :wls, :wrs)

        cases = (
            ("baseline",   (gravity=0, idual=0, isteep=0, iflatten=0, ipresfree=0, eta2=0.0), base_fields),
            ("gravity",    (gravity=1, idual=0, isteep=0, iflatten=0, ipresfree=0, eta2=0.0), base_fields),
            ("dual",       (gravity=0, idual=1, isteep=0, iflatten=0, ipresfree=0, eta2=0.1), (base_fields..., :gels, :gers)),
            ("flatten",    (gravity=0, idual=0, isteep=0, iflatten=1, ipresfree=0, eta2=0.0), base_fields),
            ("steepen",    (gravity=0, idual=0, isteep=1, iflatten=0, ipresfree=0, eta2=0.0), base_fields),
            ("presfree",   (gravity=0, idual=0, isteep=0, iflatten=0, ipresfree=1, eta2=0.0), base_fields),
        )

        for (tag, fl, fields) in cases
            @testset "inteuler [$tag]" begin
                ref = EnzoLib.inteuler(d0, p0, u0, v0, w0;
                                       idim = idim, jdim = jdim, i1 = i1, i2 = i2, j1 = j1, j2 = j2,
                                       dt = dt, gamma = gamma, geslice = ge0, grslice = gr0,
                                       dxi = dxi, flatten = flat0,
                                       gravity = fl.gravity, idual = fl.idual, eta2 = fl.eta2,
                                       isteep = fl.isteep, iflatten = fl.iflatten,
                                       ipresfree = fl.ipresfree)

                run_ie(name, ::Type{T}, field::Symbol) where {T} = begin
                    be = PPMKernels.backend(name)
                    dev(a) = PPMKernels.to_device(be, a, T)
                    out = (; (f => PPMKernels.device_zeros(be, T, (n,)) for f in
                              (:dls, :drs, :pls, :prs, :gels, :gers, :uls, :urs, :vls, :vrs, :wls, :wrs))...)
                    PPMKernels.inteuler!(out, dev(d0), dev(p0), dev(u0), dev(v0), dev(w0),
                                         dev(ge0), dev(gr0), dev(dxi), dev(flat0);
                                         idim = idim, i1 = i1, i2 = i2, j1 = j1, j2 = j2,
                                         dt = dt, gamma = gamma, eta2 = fl.eta2,
                                         gravity = fl.gravity, idual = fl.idual,
                                         isteep = fl.isteep, iflatten = fl.iflatten,
                                         ipresfree = fl.ipresfree)
                    PPMKernels.to_host(out[field])
                end

                for field in fields
                    reff = ref[field]
                    layerA!("inteuler.$tag.$field", run_ie(:cpu, Float64, field), reff)
                    layerB!("inteuler.$tag.$field", (nm, T) -> run_ie(nm, T, field))
                    layerC!("inteuler.$tag.$field", (nm, T) -> run_ie(nm, T, field), reff)
                end
            end
        end
    end
end
