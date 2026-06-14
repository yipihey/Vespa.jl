using PowerFoam
using Printf
using Statistics

const OUTBASE = joinpath(@__DIR__, "out", "native_rebuild_gate_3d")

parse_arg(i, default, T) = length(ARGS) >= i ? parse(T, ARGS[i]) : default

const N = parse_arg(1, 3, Int)
const DT = parse_arg(2, 0.02, Float64)
const MODE = Symbol(length(ARGS) >= 3 ? ARGS[3] : "periodic")
MODE in (:periodic, :bounded) || error("mode must be periodic or bounded")
const RUN_TAG = replace(@sprintf("N%d_dt%.3g_%s", N, DT, MODE), "." => "p")
const OUTDIR = joinpath(OUTBASE, RUN_TAG)

function jittered_lattice(n; amp = 0.018)
    pts = Matrix{Float64}(undef, n^3, 3)
    q = 1
    for k in 1:n, j in 1:n, i in 1:n
        x = (i - 0.5) / n
        y = (j - 0.5) / n
        z = (k - 0.5) / n
        pts[q, 1] = clamp(x + amp * sin(17i + 3j + 5k), 0.05 / n, 1 - 0.05 / n)
        pts[q, 2] = clamp(y + amp * sin(7i + 19j + 2k), 0.05 / n, 1 - 0.05 / n)
        pts[q, 3] = clamp(z + amp * sin(11i + 13j + 23k), 0.05 / n, 1 - 0.05 / n)
        q += 1
    end
    return pts
end

function solid_body_velocity(points)
    v = Matrix{Float64}(undef, size(points, 1), 3)
    @inbounds for i in axes(points, 1)
        x = points[i, 1] - 0.5
        y = points[i, 2] - 0.5
        z = points[i, 3] - 0.5
        v[i, 1] = -0.10 * y
        v[i, 2] =  0.10 * x
        v[i, 3] =  0.04 * z
    end
    return v
end

function summarize(label, points, mesh)
    counts = diff(Int.(mesh.geom.cell_face_offsets))
    return (; label,
            cells = length(mesh.geom.volume),
            faces = length(mesh.geom.c1),
            volume_sum = sum(mesh.geom.volume),
            volume_min = minimum(mesh.geom.volume),
            volume_max = maximum(mesh.geom.volume),
            face_count_min = minimum(counts),
            face_count_max = maximum(counts),
            center_rms = sqrt(mean(sum((mesh.center .- points) .^ 2; dims = 2)[:])))
end

function write_report(path, before, after, drift, disp_rms, disp_max)
    open(path, "w") do io
        println(io, "# Native 3-D Voronoi rebuild gate")
        println(io)
        println(io, "This is a small host-side correctness gate for the native")
        println(io, "PowerFoam 3-D Voronoi rebuild. It rebuilds an AREPO-shaped")
        println(io, "face table and performs one moving-mesh ALE step.")
        println(io)
        @printf(io, "- N: %d^3\n", N)
        @printf(io, "- dt: %.8g\n", DT)
        @printf(io, "- mode: `%s`\n", MODE)
        @printf(io, "- generator displacement rms/max: %.8g / %.8g\n", disp_rms, disp_max)
        println(io)
        println(io, "| state | cells | faces | volume_sum | volume_min | volume_max | face_count_min | face_count_max | center_rms |")
        println(io, "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for r in (before, after)
            @printf(io, "| %s | %d | %d | %.12g | %.8g | %.8g | %d | %d | %.8g |\n",
                    r.label, r.cells, r.faces, r.volume_sum, r.volume_min,
                    r.volume_max, r.face_count_min, r.face_count_max, r.center_rms)
        end
        println(io)
        println(io, "## Conservation Drift")
        println(io)
        println(io, "| dmass | dmx | dmy | dmz | denergy |")
        println(io, "| ---: | ---: | ---: | ---: | ---: |")
        @printf(io, "| %.9g | %.9g | %.9g | %.9g | %.9g |\n",
                drift.mass, drift.mx, drift.my, drift.mz, drift.energy)
        println(io)
        println(io, "This rebuild is deliberately all-pairs rather than AREPO-scale.")
        println(io, "The periodic mode removes the bounded-domain approximation and")
        println(io, "keeps periodic duplicate faces, establishing the native Julia")
        println(io, "face-table contract before replacing the producer with an")
        println(io, "optimized Delaunay-backed implementation.")
    end
end

function main()
    mkpath(OUTDIR)
    points = jittered_lattice(N)
    velocity = solid_body_velocity(points)
    builder = MODE == :periodic ? periodic_voronoi_mesh_arrays_3d :
              bounded_voronoi_mesh_arrays_3d
    mesh0 = builder(points; T = Float64, cell_velocity = velocity)
    state = euler_state_3d(mesh0.geom; rho = 1.0,
                           vx = velocity[:, 1], vy = velocity[:, 2],
                           vz = velocity[:, 3], pressure = 1.0, gamma = 5 / 3)
    total0 = total_conserved_3d(state, mesh0.geom)
    moved = moving_mesh_step_3d!(state, points; dt = DT, gamma = 5 / 3,
                                 boundary = MODE == :periodic ? :periodic : :clamp,
                                 mesh_velocity = velocity, riemann = :hll)
    total1 = total_conserved_3d(state, moved.geom)
    disp = sqrt.(sum((moved.points .- points) .^ 2; dims = 2)[:])
    drift = (; mass = total1.mass - total0.mass,
             mx = total1.mx - total0.mx,
             my = total1.my - total0.my,
             mz = total1.mz - total0.mz,
             energy = total1.energy - total0.energy)
    before = summarize("initial native mesh", points, mesh0)
    after = summarize("rebuilt native mesh", moved.points,
                      (; geom = moved.geom, center = moved.center))
    report = joinpath(OUTDIR, "README.md")
    write_report(report, before, after, drift, sqrt(mean(disp .* disp)), maximum(disp))
    @printf("wrote %s\n", report)
    @printf("native rebuild: cells=%d faces %d -> %d volume %.12g -> %.12g\n",
            before.cells, before.faces, after.faces, before.volume_sum, after.volume_sum)
    @printf("conservation drift: mass %.4g energy %.4g\n", drift.mass, drift.energy)
end

main()
