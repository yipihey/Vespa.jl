# mg_restrict — fine→coarse 27-point quadratic restriction vs live Enzo.

@testset "mg_restrict — fine→coarse vs live Enzo (quadratic, 3D)" begin
    if !EnzoLib.grid_available()
        @test_skip "grid dylib not built"
    else
        sdims = (17, 17, 17)        # fine
        ddims = (9, 9, 9)           # coarse = (n+1)÷2
        src = poisson_field(sdims; amp = 1.0, phase = 0.4)

        ref = EnzoLib.mg_restrict_ref(src, ddims)

        run_restrict(name, T) = begin
            be = PoissonKernels.backend(name)
            s = PoissonKernels.to_device(be, src, T)
            d = PoissonKernels.device_zeros(be, T, ddims)
            PoissonKernels.mg_restrict!(d, s)
            PoissonKernels.to_host(d)
        end

        layerA!("mg_restrict", run_restrict(:cpu, Float64), ref)
        layerB!("mg_restrict", run_restrict)
        layerC!("mg_restrict", run_restrict, ref)
    end
end
