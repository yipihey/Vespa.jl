# Full 2D Sedov demo with the live serve mode.
#   julia --project=lib/EnzoViz/test lib/EnzoViz/examples/sedov_demo.jl [n] [port] [--serve]
# Without --serve: generates the static page under out/Sedov2D_demo and exits.
# With    --serve: also starts the live server and blocks until Ctrl-C.
ENV["JULIA_CONDAPKG_BACKEND"]="Null"
const ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))   # Vespa.jl repo root
ENV["JULIA_PYTHONCALL_EXE"]=strip(read(joinpath(ROOT,"lib/EnzoViz/.python-path"),String))
ENV["QT_QPA_PLATFORM"]="offscreen"

using Vespa, MeshInterface, HGBackend, EnzoViz
include(joinpath(ROOT, "problems", "sedov_blast.jl"))

doserve = "--serve" in ARGS
nums = filter(a -> tryparse(Int, a) !== nothing, ARGS)
n    = isempty(nums) ? 64 : parse(Int, nums[1])
port = length(nums) >= 2 ? parse(Int, nums[2]) : 8088

prob = sedov_problem(n = n, tfinal = 0.05)
sim  = Simulation(HGMesh(prob.dims, prob.domain), prob)
viz  = VizSession(sim; outdir = joinpath(ROOT, "out/Sedov2D_demo"),
                  every = 4, size = (760, 760), record = true)
policy = RefinementPolicy(refine_above = 0.15, max_level = 3, every = 3)

@info "running 2D Sedov" n tfinal = prob.tfinal
evolve!(sim; policy = policy, callback = writer(viz), callback_every = viz.every)
@info "done" frames = length(viz.frames) leaves = n_cells(sim.backend) max_level = max_level(sim.backend)

finalize!(viz)
write(joinpath(viz.outdir, "last_frame.png"), EnzoViz.render_png(viz, length(viz.frames)))
# a log-color server re-render of the last frame, to exercise the recapture path
write(joinpath(viz.outdir, "last_frame_logcolor.png"),
      let s = recapture(viz, length(viz.frames); colormap = "inferno", logcolor = true)
          EnzoViz.pyconvert(Vector{UInt8}, EnzoViz._pymods()[:px].render_scene_to_png(
              EnzoViz.pybytes(codeunits(s)), 760, 760,
              EnzoViz.pytuple((1.0, 1.0, 1.0, 1.0)), viz.backend))
      end)
println("DEMO_STATIC_DONE outdir=", viz.outdir)

if doserve
    srv = serve(viz; port = port, open = false)
    println("SERVING http://127.0.0.1:$port/  (Ctrl-C to stop)")
    try
        while true; sleep(1); end
    catch
        close(srv); println("stopped")
    end
end
