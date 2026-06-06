# mg_prolong — coarse→fine trilinear prolongation vs live Enzo.

@testset "mg_prolong — coarse→fine vs live Enzo (trilinear, 3D)" begin
    if !EnzoLib.grid_available()
        @test_skip "grid dylib not built"
    else
        sdims = (9, 9, 9)           # coarse
        ddims = (17, 17, 17)        # fine
        src = poisson_field(sdims; amp = 1.0, phase = 0.6)

        ref = EnzoLib.mg_prolong_ref(src, ddims)

        run_prolong(name, T) = begin
            be = PoissonKernels.backend(name)
            s = PoissonKernels.to_device(be, src, T)
            d = PoissonKernels.device_zeros(be, T, ddims)
            PoissonKernels.mg_prolong!(d, s)
            PoissonKernels.to_host(d)
        end

        layerA!("mg_prolong", run_prolong(:cpu, Float64), ref)
        layerB!("mg_prolong", run_prolong)
        layerC!("mg_prolong", run_prolong, ref)
    end
end
