@testset "HydroMethod 10 local PPM dispatch" begin
    mktemp() do path, io
        write(io, """
            HydroMethod = 10
            NumberOfGhostZones = 1
            Gamma = 1.4
            LeftFaceBoundaryCondition = 3 3 3
            RightFaceBoundaryCondition = 3 3 3
            """)
        close(io)

        @test EnzoLib.LOCAL_PPM_HYDROMETHOD == 10
        @test EnzoLib._integer_parameter(path, "HydroMethod", 0) == 10
        @test EnzoLib._integer_parameter(path, "NumberOfGhostZones", 3) == 1
        @test EnzoLib._real_parameter(path, "Gamma", 5 / 3) == 1.4
        @test EnzoLib._periodic_root(path)

        cfg = EnzoLib.local_ppm_engine(path)
        @test cfg.hydro === :julia
        @test cfg.reflux
        @test haskey(cfg.hooks, :hydro)
        @test cfg.gravity === :off
        @test cfg.mhd_ct === :off
    end
end
