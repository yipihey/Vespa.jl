# Phase 2.2 — calcdiss (diffusion + flattening) vs the live Fortran, 1-D regime.
#
# A smooth Sod-like shock (compressive, decreasing velocity) drives wflag=1 and
# makes the flatteners fire non-trivially. We certify each output the 1-D port
# produces against the Fortran called with dimy=dimz=1 (transverse terms vanish):
#   (idiff=1, iflatten=0) → diffcoef     (CW84 A.3, u-divergence)
#   (idiff=0, iflatten=1) → flatten      (CW84 A1–A2)
#   (idiff=0, iflatten=3) → flatten      (CW84 A7–A10, multidim flattener)

using EnzoLib

@testset "Phase 2.2 — calcdiss diffusion + flattening (1-D)" begin
    if !EnzoLib.available()
        @test_skip "EnzoLib pilot library not built — skipping Fortran-reference calcdiss layers"
    else
        idim       = 16
        i1, i2     = 4, 13                                  # ≥3 ghosts each side
        j1, j2     = 1, 1                                   # single row (1-D slab)
        n          = idim
        gamma      = 1.4

        # SHARP shock transition centred at i≈8.5 (tanh), decreasing velocity —
        # steep enough that (p(i+1)−p(i-1))/(p(i+2)−p(i-2)) clears the 0.75
        # flattening threshold at the jump (a smooth ramp leaves flatten ≡ 0).
        d0 = zeros(n); p0 = zeros(n); u0 = zeros(n); e0 = zeros(n)
        for i in 1:idim
            t = tanh((i - 8.5) * 2.5)                       # −1 (left) .. +1 (right)
            d0[i] = 0.5625 - 0.4375 * t                     # 1.0 → 0.125
            p0[i] = 0.5500 - 0.4500 * t                     # 1.0 → 0.1
            u0[i] = 0.20  - 0.30  * t                       # 0.5 → −0.1 (compressive)
            e0[i] = p0[i] / ((gamma - 1) * d0[i]) + 0.5 * u0[i]^2
        end
        vw = zeros(n)                                        # transverse fields (unused, dimy=dimz=1)

        cases = ((1, 0, :d, "diffcoef [idiff=1]"),
                 (0, 1, :f, "flatten  [iflatten=1]"),
                 (0, 3, :f, "flatten  [iflatten=3]"))

        for (idiff, iflatten, which, tag) in cases
            @testset "$tag" begin
                rdc, rfl = EnzoLib.calcdiss(d0, e0, u0, p0, vw, vw;
                                            idim = idim, i1 = i1, i2 = i2, j1 = j1, j2 = j2,
                                            dt = 0.1, gamma = gamma,
                                            idiff = idiff, iflatten = iflatten)
                ref = which === :d ? rdc : rfl
                @test any(!iszero, ref)                     # non-vacuous: the output actually moved

                run_cd(name, ::Type{T}) where {T} = begin
                    be = PPMKernels.backend(name)
                    dc = PPMKernels.device_zeros(be, T, (n,))
                    fl = PPMKernels.device_zeros(be, T, (n,))
                    d  = PPMKernels.to_device(be, d0, T); e = PPMKernels.to_device(be, e0, T)
                    u  = PPMKernels.to_device(be, u0, T); p = PPMKernels.to_device(be, p0, T)
                    PPMKernels.calcdiss!(dc, fl, d, e, u, p; idim = idim, i1 = i1, i2 = i2,
                                         j1 = j1, j2 = j2, gamma = gamma,
                                         idiff = idiff, iflatten = iflatten)
                    PPMKernels.to_host(which === :d ? dc : fl)
                end

                layerA!("calcdiss.$tag", run_cd(:cpu, Float64), ref)
                layerB!("calcdiss.$tag", run_cd)
                layerC!("calcdiss.$tag", run_cd, ref)
            end
        end
    end
end
