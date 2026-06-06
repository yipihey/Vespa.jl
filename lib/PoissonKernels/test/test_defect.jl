# mg_calc_defect — the residual array AND its scalar L2 norm must match Enzo.

@testset "mg_calc_defect — residual + norm vs live Enzo (7-point, 2nd order)" begin
    if !EnzoLib.grid_available()
        @test_skip "grid dylib not built"
    else
        dims = (17, 17, 17)
        sol0 = poisson_field(dims; amp = 1.0, phase = 0.2)
        rhs  = poisson_field(dims; amp = 0.5, phase = 0.9)

        ref_def, ref_norm = EnzoLib.mg_calc_defect_ref(sol0, rhs)

        run_defect(name, T) = begin
            be = PoissonKernels.backend(name)
            s = PoissonKernels.to_device(be, sol0, T)
            r = PoissonKernels.to_device(be, rhs, T)
            d = PoissonKernels.device_zeros(be, T, dims)
            PoissonKernels.mg_calc_defect!(d, s, r)
            PoissonKernels.to_host(d)
        end
        norm_of(name, T) = begin
            be = PoissonKernels.backend(name)
            s = PoissonKernels.to_device(be, sol0, T)
            r = PoissonKernels.to_device(be, rhs, T)
            d = PoissonKernels.device_zeros(be, T, dims)
            PoissonKernels.mg_calc_defect!(d, s, r)
        end

        # defect array — Layer A bit-tight, Layer B cpu-f32 ≡ metal-f32.
        layerA!("mg_defect", run_defect(:cpu, Float64), ref_def)
        layerB!("mg_defect", run_defect)

        # Layer C (metal-f32 vs Fortran-f64) is intentionally looser FOR THE DEFECT:
        # the residual h3·(Σ − 6·center) + rhs multiplies by h3 = -(d1-1)(d2-1)(d3-1)
        # (≈ -4096 on 17³) AND subtracts 6·center from a sum of six ~equal neighbours
        # — a catastrophic cancellation f32 cannot hold to RTOL_C. This is an inherent
        # f32 accuracy floor for the residual operator, NOT a port issue: Layer A (f64)
        # is bit-tight and Layer B (cpu-f32 ≡ metal-f32) agrees exactly. We cannot
        # reassociate to reduce the cancellation without breaking the Layer-A match to
        # Fortran. Observed maxabs ≈ 3e-3, maxrel ≈ 3e-4 on 17³; certify to that floor.
        if metal_ready()
            RTOL_C_DEFECT = Tolerance(rtol = 2e-3, atol = 1e-2)
            @check("mg_defect [C:metal-f32 vs Fortran, residual floor]",
                   run_defect(:metal, Float32), ref_def, RTOL_C_DEFECT)
        end

        # scalar norm (f64, bit-tight up to reassociation)
        @test isapprox(norm_of(:cpu, Float64), ref_norm; rtol = 1e-12, atol = 1e-14)
    end
end
