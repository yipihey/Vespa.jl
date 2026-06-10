# Standard Enzo AMR hydro problems for HydroMethod=10 in every supported rank:
# 1-D and 2-D Sod tubes against the exact Riemann solution, and a reduced 3-D
# Noh converging shock against its spherical analytic solution.

const LOCAL_PPM_NOH = normpath(joinpath(
    @__DIR__, "..", "..", "..", "..", "run", "Hydro", "Hydro-3D",
    "NohProblem3DAMR", "NohProblem3DAMR.enzo",
))
const LOCAL_PPM_SOD_1D = normpath(joinpath(
    @__DIR__, "..", "..", "..", "..", "run", "Hydro", "Hydro-1D",
    "SodShockTube", "SodShockTubeAMR.enzo",
))
const LOCAL_PPM_SOD_2D = normpath(joinpath(
    @__DIR__, "..", "..", "..", "..", "run", "Hydro", "Hydro-2D",
    "SodShockTube2DAMR", "SodShockTube2DAMR.enzo",
))

function _local_ppm_amr_parameters(path)
    return replace(
        read(path, String),
        r"HydroMethod\s*=\s*0" =>
            "HydroMethod            = 10\nNumberOfGhostZones     = 3",
    )
end

function _run_local_ppm_sod_amr(path)
    mktempdir() do work
        pf = joinpath(work, "SodShockTubeAMR-localppm.enzo")
        write(pf, _local_ppm_amr_parameters(path))
        cd(work) do
            h = EnzoLib.session_init(pf)
            h == C_NULL && error("session_init failed for $pf")
            try
                EnzoLib.session_rebuild(h, 0)
                engine = EnzoLib.local_ppm_engine(pf)
                cycles = 0
                max_level = 0
                while EnzoLib.session_time(h) < EnzoLib.session_stop_time(h) && cycles < 1000
                    EnzoLib.evolve_level!(h, 0, 0.0; engine = engine, regrid = true)
                    EnzoLib.session_rebuild(h, 0)
                    cycles += 1
                    max_level = max(max_level, maximum(
                        l for l in 0:4 if EnzoLib.session_num_grids_on_level(h, l) > 0
                    ))
                end

                grid = EnzoLib.problem_grid_index_on_level(h, 0, 0)
                rank = EnzoLib.problem_grid_rank(h, grid)
                dims = Tuple(EnzoLib.problem_grid_dims(h, grid))
                ng = 3
                active = ntuple(d -> d <= rank ? dims[d] - 2ng : 1, 3)
                density = EnzoLib.problem_get_field(
                    h, EnzoLib.field_index(h, 0; grid = grid), grid
                )
                profile = zeros(active[1])
                transverse_spread = 0.0
                for i in 1:active[1]
                    row = Float64[]
                    for k in 1:active[3], j in 1:active[2]
                        cell = (ng + i) +
                               dims[1] * ((rank >= 2 ? ng + j : 1) - 1) +
                               dims[1] * dims[2] * ((rank >= 3 ? ng + k : 1) - 1)
                        push!(row, density[cell])
                    end
                    profile[i] = sum(row) / length(row)
                    transverse_spread = max(
                        transverse_spread, maximum(row) - minimum(row)
                    )
                end
                time = EnzoLib.session_time(h)
                exact = [
                    exact_riemann_sample(
                        (1.0, 0.0, 1.0), (0.125, 0.0, 0.1), 1.4,
                        ((i - 0.5) / active[1] - 0.5) / time,
                    )[1] for i in 1:active[1]
                ]
                return (
                    rank = rank,
                    cycles = cycles,
                    time = time,
                    max_level = max_level,
                    density = profile,
                    analytic_l1 = sum(abs.(profile .- exact)) / length(profile),
                    transverse_spread = transverse_spread,
                )
            finally
                EnzoLib.free_problem(h)
            end
        end
    end
end

function _local_ppm_noh_parameters()
    src = read(LOCAL_PPM_NOH, String)
    return replace(
        src,
        r"TopGridDimensions\s*=\s*100 100 100" =>
            "TopGridDimensions      = 32 32 32",
        r"HydroMethod\s*=\s*0" =>
            "HydroMethod            = 10\nNumberOfGhostZones     = 3",
        r"StopTime\s*=\s*2.0" =>
            "StopTime               = 0.3\nStopCycle              = 40",
        r"dtDataDump\s*=\s*0.4" =>
            "dtDataDump             = 10.0",
    )
end

function _run_local_ppm_noh_amr()
    mktempdir() do work
        pf = joinpath(work, "NohProblem3DAMR-localppm.enzo")
        write(pf, _local_ppm_noh_parameters())
        cd(work) do
            h = EnzoLib.session_init(pf)
            h == C_NULL && error("session_init failed for $pf")
            try
                EnzoLib.session_rebuild(h, 0)
                engine = EnzoLib.local_ppm_engine(pf)
                cycles = 0
                max_level = 0
                while EnzoLib.session_time(h) < EnzoLib.session_stop_time(h) && cycles < 40
                    EnzoLib.evolve_level!(h, 0, 0.0; engine = engine, regrid = true)
                    EnzoLib.session_rebuild(h, 0)
                    cycles += 1
                    max_level = max(max_level, maximum(
                        l for l in 0:3 if EnzoLib.session_num_grids_on_level(h, l) > 0
                    ))
                end

                grid = EnzoLib.problem_grid_index_on_level(h, 0, 0)
                dims = Tuple(EnzoLib.problem_grid_dims(h, grid))
                ng = 3
                active = ntuple(d -> dims[d] - 2ng, 3)
                density = EnzoLib.problem_get_field(
                    h, EnzoLib.field_index(h, 0; grid = grid), grid
                )
                time = EnzoLib.session_time(h)
                numerical = Float64[]
                exact = Float64[]
                postshock = Float64[]
                preshock_relative_error = Float64[]
                for k in 1:active[3], j in 1:active[2], i in 1:active[1]
                    cell = (ng + i) +
                           dims[1] * (ng + j - 1) +
                           dims[1] * dims[2] * (ng + k - 1)
                    radius = sqrt(
                        ((i - 0.5) / active[1])^2 +
                        ((j - 0.5) / active[2])^2 +
                        ((k - 0.5) / active[3])^2
                    )
                    analytic = radius < time / 3 ? 64.0 : (1 + time / radius)^2
                    push!(numerical, density[cell])
                    push!(exact, analytic)
                    if radius < time / 3
                        push!(postshock, density[cell])
                    else
                        push!(preshock_relative_error, abs(density[cell] - analytic) / analytic)
                    end
                end
                return (
                    cycles = cycles,
                    time = time,
                    max_level = max_level,
                    density = numerical,
                    postshock_mean = sum(postshock) / length(postshock),
                    preshock_mean_relative_error =
                        sum(preshock_relative_error) / length(preshock_relative_error),
                    analytic_relative_l1 =
                        sum(abs.(numerical .- exact)) / sum(exact),
                )
            finally
                EnzoLib.free_problem(h)
            end
        end
    end
end

if !EnzoLib.grid_available()
    @info "Session bridge not built - skipping Local PPM Noh AMR test"
else
    @testset "HydroMethod 10: standard 1-D Sod AMR" begin
        result = _run_local_ppm_sod_amr(LOCAL_PPM_SOD_1D)
        @info "Local PPM 1-D Sod AMR" result.cycles result.time result.max_level result.analytic_l1
        @test result.rank == 1
        @test result.max_level >= 4
        @test all(isfinite, result.density)
        @test minimum(result.density) > 0
        @test result.analytic_l1 < 0.003
    end

    @testset "HydroMethod 10: standard 2-D Sod AMR" begin
        result = _run_local_ppm_sod_amr(LOCAL_PPM_SOD_2D)
        @info "Local PPM 2-D Sod AMR" result.cycles result.time result.max_level result.analytic_l1 result.transverse_spread
        @test result.rank == 2
        @test result.max_level >= 2
        @test all(isfinite, result.density)
        @test minimum(result.density) > 0
        @test result.analytic_l1 < 0.006
        @test result.transverse_spread < 0.04
    end

    @testset "HydroMethod 10: standard 3-D Noh AMR" begin
        result = _run_local_ppm_noh_amr()
        @info "Local PPM Noh AMR" result.cycles result.time result.max_level result.postshock_mean result.preshock_mean_relative_error result.analytic_relative_l1
        @test result.max_level >= 2
        @test all(isfinite, result.density)
        @test minimum(result.density) > 0
        @test abs(result.postshock_mean - 64) / 64 < 0.08
        @test result.preshock_mean_relative_error < 0.03
        @test result.analytic_relative_l1 < 0.05
    end
end
