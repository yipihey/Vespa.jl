# mg_relax — one red/black Gauss-Seidel sweep must match Enzo's serial mg_relax
# bit-for-bit (THE proof the parallel colour split reproduces the serial sweep).

@testset "mg_relax — Gauss-Seidel vs live Enzo (7-point, 2nd order)" begin
    if !EnzoLib.grid_available()
        @test_skip "grid dylib not built (bash EnzoModules/deps/build_grid_darwin.sh)"
    else
        dims = (17, 17, 17)
        sol0 = poisson_field(dims; amp = 1.0, phase = 0.3)
        rhs  = poisson_field(dims; amp = 0.7, phase = 1.1)

        ref = EnzoLib.mg_relax_ref(sol0, rhs)

        run_relax(name, T) = begin
            be = PoissonKernels.backend(name)
            s = PoissonKernels.to_device(be, sol0, T)
            r = PoissonKernels.to_device(be, rhs, T)
            PoissonKernels.mg_relax!(s, r)
            PoissonKernels.to_host(s)
        end

        layerA!("mg_relax", run_relax(:cpu, Float64), ref)
        layerB!("mg_relax", run_relax)
        layerC!("mg_relax", run_relax, ref)
    end
end
