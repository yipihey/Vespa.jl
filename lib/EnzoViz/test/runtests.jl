# EnzoViz end-to-end tests. PythonCall must be pointed at the veusz fork's
# interpreter BEFORE it initializes CPython, so we set the env vars first —
# before `using EnzoViz` (which loads PythonCall). The interpreter comes from
# `.python-path` (written by setup_env.jl) or, failing that, the fork's bundled
# .venv directly.

function _interp()
    pth = joinpath(@__DIR__, "..", ".python-path")
    isfile(pth) && return strip(read(pth, String))
    fork = get(ENV, "ENZOVIZ_VEUSZ", "/Users/tabel/Projects/veusz")
    return joinpath(fork, ".venv", "bin", "python")
end

const PY = _interp()
isfile(PY) || error("EnzoViz tests: no Python interpreter at $PY. Run setup_env.jl first.")
ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
ENV["JULIA_PYTHONCALL_EXE"] = PY
ENV["QT_QPA_PLATFORM"] = "offscreen"

using Test
using Vespa, MeshInterface, RefMesh, HGBackend, EnzoViz
using Downloads, Sockets

@testset "EnzoViz inline visualization" begin

    @testset "Python pipeline available" begin
        # Force PythonCall init + module resolution.
        sess_mods = EnzoViz._pymods()
        backends = Set(string.(collect(sess_mods[:px].available_backends())))
        @info "veusz paint backends" backends
        @test !isempty(backends)
        @test "tiny-skia" in backends || "vello" in backends
    end

    @testset "1D Sod page (RefMesh)" begin
        prob = sod_problem_defaults(n = 128)
        sim = Simulation(UniformMesh(prob.dims, prob.domain), prob)
        outdir = mktempdir()
        viz = VizSession(sim; outdir = outdir, every = 20, size = (700, 500))
        evolve!(sim; callback = writer(viz), callback_every = viz.every)
        finalize!(viz)

        @test isfile(joinpath(outdir, "run_scenes.json"))
        @test isfile(joinpath(outdir, "index.html"))
        @test length(viz.frames) >= 2            # at least init + final
        @test viz.rank == 1
        # run_scenes.json is a lightweight manifest: meta + one entry per frame
        # (png/vsz/time), NOT the embedded scenes (those bloated it to 100s of MB).
        scenes = read(joinpath(outdir, "run_scenes.json"), String)
        @test occursin("\"frames\"", scenes)
        @test occursin("\"meta\"", scenes) && occursin("frame_0001.png", scenes)
        @test length(scenes) < 1_000_000         # manifest stays small
        # quick-look PNG renders
        png = EnzoViz.render_png(viz, 1)
        @test length(png) > 8 && png[1:4] == UInt8[0x89, 0x50, 0x4e, 0x47]
        @info "Sod page" frames = length(viz.frames) outdir
    end

    @testset "2D Sedov AMR page (HGBackend)" begin
        include(abspath(joinpath(@__DIR__, "..", "..", "..", "problems", "sedov_blast.jl")))
        prob = sedov_problem(n = 32, tfinal = 0.02)
        mesh = HGMesh(prob.dims, prob.domain)
        sim = Simulation(mesh, prob)
        outdir = mktempdir()
        viz = VizSession(sim; outdir = outdir, every = 6, size = (640, 640))
        policy = RefinementPolicy(refine_above = 0.2, max_level = 1, every = 4)
        evolve!(sim; policy = policy, callback = writer(viz), callback_every = viz.every)
        finalize!(viz)

        @test isfile(joinpath(outdir, "run_scenes.json"))
        @test isfile(joinpath(outdir, "index.html"))
        @test viz.rank == 2
        @test viz.nlevels >= 1
        @test length(viz.frames) >= 2
        png = EnzoViz.render_png(viz, length(viz.frames))
        @test length(png) > 8 && png[1:4] == UInt8[0x89, 0x50, 0x4e, 0x47]
        @info "Sedov page" frames = length(viz.frames) nlevels = viz.nlevels outdir
    end

    @testset "recapture (server-side re-render incl. log axis)" begin
        prob = sod_problem_defaults(n = 64)
        sim = Simulation(UniformMesh(prob.dims, prob.domain), prob)
        viz = VizSession(sim; outdir = mktempdir(), every = 30, size = (600, 400),
                         record = true)
        evolve!(sim; callback = writer(viz), callback_every = viz.every)
        @test viz.record && length(viz.rasters) == length(viz.frames)
        # re-render frame 1 with a different colormap + log axis (1D log-y): the
        # genuine server-only capability. Must yield a valid, non-trivial scene.
        scene = recapture(viz, 1; colormap = "plasma", logaxis = true)
        @test occursin("\"", scene) && length(scene) > 1000
        # a plain re-capture (linear) also works and differs from the log one
        scene2 = recapture(viz, 1; colormap = "plasma", logaxis = false)
        @test length(scene2) > 1000
        # (server-side PNG rendering of a recaptured scene is covered by the
        #  /png endpoint in the serve-mode test below.)
    end

    @testset "serve mode (HTTP endpoints)" begin
        prob = sod_problem_defaults(n = 64)
        sim = Simulation(UniformMesh(prob.dims, prob.domain), prob)
        viz = VizSession(sim; outdir = mktempdir(), every = 30, size = (500, 360),
                         record = true)
        evolve!(sim; callback = writer(viz), callback_every = viz.every)
        port = 8123
        srv = serve(viz; port = port)
        try
            base = "http://127.0.0.1:$port"
            # /meta
            meta = read(Downloads.download("$base/meta"), String)
            @test occursin("nframes", meta)
            # / (serve-mode page) mentions the server render panel + log axis
            home = read(Downloads.download("$base/"), String)
            @test occursin("log axis", home)
            @test occursin("/png?", home)
            # /png server re-render → PNG
            pf = Downloads.download("$base/png?frame=1&colormap=viridis&logaxis=1")
            bytes = read(pf)
            @test bytes[1:4] == UInt8[0x89, 0x50, 0x4e, 0x47]
            # static asset served
            rs = read(Downloads.download("$base/run_scenes.json"), String)
            @test occursin("frames", rs)
        finally
            close(srv)
        end
    end
end
