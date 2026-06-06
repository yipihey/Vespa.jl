# comp_accel — g = -∇φ finite-difference gradient vs live Enzo (2nd order).
# The source (potential) carries `start` ghost cells on each side of the dest.

@testset "comp_accel — acceleration g=-∇φ vs live Enzo (2nd order, 3D)" begin
    if !EnzoLib.grid_available()
        @test_skip "grid dylib not built"
    else
        ddims = (16, 16, 16)
        start = (3, 3, 3)
        sdims = (ddims[1] + 2start[1], ddims[2] + 2start[2], ddims[3] + 2start[3])
        iflag = 1                    # symmetric (face-centred) difference
        del = (0.1, 0.1, 0.1)
        src = poisson_field(sdims; amp = 1.0, phase = 0.7)

        ra1, ra2, ra3 = EnzoLib.comp_accel_ref(src, ddims; iflag = iflag, start = start, del = del)

        run_acc(name, T, comp) = begin
            be = PoissonKernels.backend(name)
            s = PoissonKernels.to_device(be, src, T)
            a1 = PoissonKernels.device_zeros(be, T, ddims)
            a2 = PoissonKernels.device_zeros(be, T, ddims)
            a3 = PoissonKernels.device_zeros(be, T, ddims)
            PoissonKernels.comp_accel!(a1, a2, a3, s; iflag = iflag, start = start, del = del)
            PoissonKernels.to_host(comp == 1 ? a1 : comp == 2 ? a2 : a3)
        end

        for (comp, ref) in ((1, ra1), (2, ra2), (3, ra3))
            layerA!("comp_accel.$comp", run_acc(:cpu, Float64, comp), ref)
            layerB!("comp_accel.$comp", (nm, T) -> run_acc(nm, T, comp))
            layerC!("comp_accel.$comp", (nm, T) -> run_acc(nm, T, comp), ref)
        end
    end
end
