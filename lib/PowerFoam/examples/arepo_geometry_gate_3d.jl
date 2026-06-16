const _REQUEST_METAL = lowercase(get(ENV, "POWERFOAM_BACKEND", "metal")) == "metal"
const _METAL_IMPORT_ERROR = Ref{Any}(nothing)
if _REQUEST_METAL
    try
        @eval using Metal
    catch err
        _METAL_IMPORT_ERROR[] = err
    end
end

using KernelAbstractions
using LinearAlgebra
using PowerFoam
using Printf
using Statistics
using ArepoLib

const GAMMA = 5 / 3
const AREPO_DIR = "/Users/tabel/Projects/arepo"
const EXAMPLE = joinpath(AREPO_DIR, "examples", "bauer_springel_turbulence_3d")
const OUTBASE = joinpath(@__DIR__, "out", "arepo_geometry_gate_3d")

parse_arg(i, default, T) = length(ARGS) >= i ? parse(T, ARGS[i]) : default

const N = parse_arg(1, 12, Int)
const DT = parse_arg(2, 0.001, Float64)
const RIEMANN = Symbol(length(ARGS) >= 3 ? ARGS[3] : "hll")
const NSTEPS = parse_arg(4, 1, Int)
NSTEPS >= 0 || error("NSTEPS must be nonnegative")
const RUN_TAG = NSTEPS == 1 ? @sprintf("N%d_dt%.3g_%s", N, DT, RIEMANN) :
                @sprintf("N%d_dt%.3g_n%d_%s", N, DT, NSTEPS, RIEMANN)
const OUTDIR = joinpath(OUTBASE, replace(RUN_TAG, "." => "p"))

function maybe_metal_backend()
    _REQUEST_METAL || return nothing
    if _METAL_IMPORT_ERROR[] !== nothing
        @warn "Metal unavailable; running CPU-only parity check" err = _METAL_IMPORT_ERROR[]
        return nothing
    end
    return Metal.MetalBackend()
end

function python_cmd()
    for exe in (get(ENV, "AREPO_PYTHON", ""), joinpath(AREPO_DIR, ".venv", "bin", "python"),
                "python3", "python")
        isempty(exe) && continue
        try
            run(pipeline(Cmd([exe, "-c", "import h5py, numpy"]);
                         stdout = devnull, stderr = devnull))
            return exe
        catch
        end
    end
    error("no Python with h5py/numpy found; set AREPO_PYTHON")
end

function _arepo_solver_name(riemann::Symbol)
    riemann == :hll && return "HLL"
    riemann == :llf && return "LLF"
    riemann == :hllc && return "HLLC"
    riemann == :exact && return "Exact"
    error("unsupported AREPO Riemann solver for parity gate: $riemann")
end

function _set_param_line(text::String, key::AbstractString, value::AbstractString)
    line = @sprintf("%-50s %s", key, value)
    occursin(Regex("(?m)^$(key)\\s"), text) ?
    replace(text, Regex("(?m)^$(key)\\s+.*\$") => line) :
    string(text, endswith(text, "\n") ? "" : "\n", line, "\n")
end

function stage_arepo_case(n; riemann::Symbol = RIEMANN)
    isdir(EXAMPLE) || error("AREPO turbulence example not found at $EXAMPLE")
    dir = mktempdir()
    param = joinpath(dir, "param.txt")
    cp(joinpath(EXAMPLE, "param_decay.txt"), param)
    text = read(param, String)
    text = replace(text,
                   r"(?m)^TimeOfFirstSnapshot\s+.*$" => "TimeOfFirstSnapshot                               2",
                   r"(?m)^TimeBetSnapshot\s+.*$" => "TimeBetSnapshot                                   2")
    text = normalize_param_for_linked_arepo(text)
    text = _set_param_line(text, "HydroRiemannSolver", _arepo_solver_name(riemann))
    write(param, text)
    mkpath(joinpath(dir, "output"))
    py = python_cmd()
    run(pipeline(`$py $(joinpath(EXAMPLE, "create.py")) $dir unused $n 271`;
                 stdout = devnull))
    isfile(joinpath(dir, "IC.hdf5")) || error("AREPO create.py produced no IC.hdf5")
    return dir
end

function normalize_param_for_linked_arepo(text)
    lines = split(text, '\n'; keepempty = true)
    keep = String[]
    for line in lines
        if occursin(r"^SofteningComovingType[2-5]\s", line) ||
           occursin(r"^SofteningMaxPhysType[2-5]\s", line)
            continue
        end
        push!(keep, line)
    end
    text = join(keep, "\n")
    text = replace(text,
                   r"(?m)^SofteningTypeOfPartType1\s+.*$" => "SofteningTypeOfPartType1              1",
                   r"(?m)^SofteningTypeOfPartType2\s+.*$" => "SofteningTypeOfPartType2              1",
                   r"(?m)^SofteningTypeOfPartType3\s+.*$" => "SofteningTypeOfPartType3              1",
                   r"(?m)^SofteningTypeOfPartType4\s+.*$" => "SofteningTypeOfPartType4              1",
                   r"(?m)^SofteningTypeOfPartType5\s+.*$" => "SofteningTypeOfPartType5              1")
    if !occursin(r"(?m)^MinimumComovingHydroSoftening\s", text)
        text *= "\nMinimumComovingHydroSoftening         0.001\n"
    end
    if !occursin(r"(?m)^AdaptiveHydroSofteningSpacing\s", text)
        text *= "AdaptiveHydroSofteningSpacing         1.2\n"
    end
    return text
end

function arepo_initial_export(dir)
    p = ArepoLib.precision_bytes()
    p.ndim == 3 || error("AREPO_LIB must point to a 3-D build; got ndim=$(p.ndim)")
    h = cd(() -> ArepoLib.init("param.txt"), dir)
    try
        ng = ArepoLib.info(h).numgas
        geo = ArepoLib.get_voronoi_3d(h)
        vol = ArepoLib.get_cell_field(h, :volume)
        rho = ArepoLib.get_cell_field(h, :rho)
        pressure = ArepoLib.get_cell_field(h, :pressure)
        center = ArepoLib.get_cell_field(h, :center)
        csnd = ArepoLib.get_cell_field(h, :csnd)
        velvertex = ArepoLib.get_cell_field(h, :velvertex)
        vel = ArepoLib.get_particle_field(h, :vel)[1:ng, :]
        cgrad = ArepoLib.get_hydro_gradients(h)
        conn = ArepoLib.get_gradient_connections_3d(h)
        return (; h, ng, geo, vol, rho, pressure, center, csnd, velvertex,
                vel, cgrad, conn,
                box = ArepoLib.box_size(h), dir)
    catch
        ArepoLib.finalize(h)
        rethrow()
    end
end

function face_centers_from_geometry(geometry)
    nf = length(geometry.nv)
    centers = Matrix{Float64}(undef, nf, 3)
    offset = 0
    for f in 1:nf
        nv = Int(geometry.nv[f])
        verts = @view geometry.verts[(offset + 1):(offset + nv), :]
        centers[f, 1] = mean(@view verts[:, 1])
        centers[f, 2] = mean(@view verts[:, 2])
        centers[f, 3] = mean(@view verts[:, 3])
        offset += nv
    end
    return centers
end

function hydro_gradients_from_arepo(cgrad, be; T = Float32)
    return HydroGradients3D(
        PowerFoam._backend_copy(be, cgrad.drho[:, 1], T),
        PowerFoam._backend_copy(be, cgrad.drho[:, 2], T),
        PowerFoam._backend_copy(be, cgrad.drho[:, 3], T),
        PowerFoam._backend_copy(be, cgrad.dvel[:, 1, 1], T),
        PowerFoam._backend_copy(be, cgrad.dvel[:, 1, 2], T),
        PowerFoam._backend_copy(be, cgrad.dvel[:, 1, 3], T),
        PowerFoam._backend_copy(be, cgrad.dvel[:, 2, 1], T),
        PowerFoam._backend_copy(be, cgrad.dvel[:, 2, 2], T),
        PowerFoam._backend_copy(be, cgrad.dvel[:, 2, 3], T),
        PowerFoam._backend_copy(be, cgrad.dvel[:, 3, 1], T),
        PowerFoam._backend_copy(be, cgrad.dvel[:, 3, 2], T),
        PowerFoam._backend_copy(be, cgrad.dvel[:, 3, 3], T),
        PowerFoam._backend_copy(be, cgrad.dpress[:, 1], T),
        PowerFoam._backend_copy(be, cgrad.dpress[:, 2], T),
        PowerFoam._backend_copy(be, cgrad.dpress[:, 3], T),
    )
end

function timestep_stats(exported)
    dt = arepo_hydro_dt_3d(exported.vol, exported.pressure, exported.rho;
                           gamma = GAMMA, courant = 0.3,
                           max_dt = 0.05, min_dt = 1e-6)
    return (; min = minimum(dt), median = median(dt), max = maximum(dt),
            chosen_global = minimum(dt))
end

function run_reconstructed_once(exported, be; dt, riemann)
    geom, state, _, _ = make_state_and_geom(exported, be)
    gradients = hydro_gradients_from_arepo(exported.cgrad, be; T = Float32)
    face_center = face_centers_from_geometry(exported.geo)
    before = diagnostics("PowerFoam reconstructed", state, geom, 0, 0.0)
    finite_volume_reconstructed_step_3d!(state, geom, gradients, exported.center,
                                         face_center; dt, gamma = GAMMA,
                                         riemann, box_size = exported.box)
    after = diagnostics("PowerFoam reconstructed", state, geom, 1, dt)
    return [before, after], state
end

function hosted_rebuild_stats(exported)
    old_center = copy(exported.center)
    old_faces = length(exported.geo.nv)
    old_vertices = size(exported.geo.verts, 1)
    old_time = ArepoLib.sim_time(exported.h)
    status = ArepoLib.run_step!(exported.h)
    new_geo = ArepoLib.get_voronoi_3d(exported.h)
    new_vol = ArepoLib.get_cell_field(exported.h, :volume)
    new_center = ArepoLib.get_cell_field(exported.h, :center)
    delta = new_center .- old_center
    delta .= ifelse.(delta .> 0.5 * exported.box, delta .- exported.box,
                     ifelse.(delta .< -0.5 * exported.box, delta .+ exported.box, delta))
    disp = sqrt.(sum(delta .^ 2; dims = 2)[:])
    return (; status, time_before = old_time, time_after = ArepoLib.sim_time(exported.h),
            old_faces, new_faces = length(new_geo.nv),
            old_vertices, new_vertices = size(new_geo.verts, 1),
            old_volume_sum = sum(exported.vol), new_volume_sum = sum(new_vol),
            center_disp_rms = sqrt(mean(disp .* disp)),
            center_disp_max = maximum(disp))
end

function make_state_and_geom(exported, be)
    geom_host = arepo_voronoi_mesh_arrays_3d(exported.geo, exported.vol;
                                             T = Float64,
                                             cell_velocity = exported.velvertex)
    state_host = euler_state_3d(geom_host; rho = exported.rho,
                                vx = exported.vel[:, 1],
                                vy = exported.vel[:, 2],
                                vz = exported.vel[:, 3],
                                pressure = exported.pressure,
                                gamma = GAMMA)
    return to_backend(be, geom_host; T = Float32),
           to_backend(be, state_host; T = Float32),
           geom_host,
           state_host
end

function diagnostics(label, state, geom, step, time)
    prim = conserved_to_primitive_3d(state; gamma = GAMMA)
    totals = total_conserved_3d(state, geom)
    v2 = prim.vx .* prim.vx .+ prim.vy .* prim.vy .+ prim.vz .* prim.vz
    cs2 = GAMMA .* prim.pressure ./ prim.rho
    vrms = sqrt(mean(v2))
    mach_rms = sqrt(mean(v2 ./ cs2))
    density_rms = std(prim.rho) / mean(prim.rho)
    return (; label, step, time, mass = totals.mass, mx = totals.mx, my = totals.my,
            mz = totals.mz, energy = totals.energy, vrms, mach_rms, density_rms,
            rho_min = minimum(prim.rho), rho_max = maximum(prim.rho),
            pmin = minimum(prim.pressure))
end

max_abs_diff(a, b) = maximum(abs.(Array(a) .- Array(b)))
compare_states(a, b) = (; D = max_abs_diff(a.D, b.D),
                         Mx = max_abs_diff(a.Mx, b.Mx),
                         My = max_abs_diff(a.My, b.My),
                         Mz = max_abs_diff(a.Mz, b.Mz),
                         E = max_abs_diff(a.E, b.E))

function moving_face_stats(geom)
    vx = Array(geom.face_vx)
    vy = Array(geom.face_vy)
    vz = Array(geom.face_vz)
    nx = Array(geom.normal_x)
    ny = Array(geom.normal_y)
    nz = Array(geom.normal_z)
    speed = sqrt.(vx .* vx .+ vy .* vy .+ vz .* vz)
    normal_speed = vx .* nx .+ vy .* ny .+ vz .* nz
    return (; speed_rms = sqrt(mean(speed .* speed)),
            speed_max = maximum(speed),
            normal_rms = sqrt(mean(normal_speed .* normal_speed)),
            normal_maxabs = maximum(abs.(normal_speed)))
end

function geometry_parity_stats(exported, geom)
    nc = exported.ng
    c1_expected = Int.(exported.geo.c1)
    c2_expected = Int.(exported.geo.c2)
    normals_expected = Matrix{Float64}(exported.geo.normals)
    for f in eachindex(c1_expected)
        if !(1 <= c1_expected[f] <= nc) && (1 <= c2_expected[f] <= nc)
            c1_expected[f], c2_expected[f] = c2_expected[f], c1_expected[f]
            normals_expected[f, 1] = -normals_expected[f, 1]
            normals_expected[f, 2] = -normals_expected[f, 2]
            normals_expected[f, 3] = -normals_expected[f, 3]
        end
    end
    nf = length(c1_expected)
    area_expected = Vector{Float64}(undef, nf)
    nx_expected = Vector{Float64}(undef, nf)
    ny_expected = Vector{Float64}(undef, nf)
    nz_expected = Vector{Float64}(undef, nf)
    offsets_ring = vcat(0, cumsum(Int.(exported.geo.nv)))
    for f in 1:nf
        lo = offsets_ring[f] + 1
        hi = offsets_ring[f + 1]
        sx, sy, sz = PowerFoam._ring_area_normal(@view exported.geo.verts[lo:hi, :])
        area_expected[f] = 0.5 * sqrt(sx * sx + sy * sy + sz * sz)
        gx, gy, gz = normals_expected[f, 1], normals_expected[f, 2], normals_expected[f, 3]
        gn = sqrt(gx * gx + gy * gy + gz * gz)
        nx_expected[f] = gx / gn
        ny_expected[f] = gy / gn
        nz_expected[f] = gz / gn
    end
    counts_expected = zeros(Int, nc)
    for f in 1:nf
        counts_expected[c1_expected[f]] += 1
        c2_expected[f] > 0 && (counts_expected[c2_expected[f]] += 1)
    end
    offsets = Int.(Array(geom.cell_face_offsets))
    counts_actual = offsets[2:end] .- offsets[1:end-1]
    return (;
        cells = nc,
        faces_expected = nf,
        faces_actual = length(geom.c1),
        vertices = size(exported.geo.verts, 1),
        c1_mismatches = count(Int.(Array(geom.c1)) .!= c1_expected),
        c2_mismatches = count(Int.(Array(geom.c2)) .!= c2_expected),
        volume_maxdiff = maximum(abs.(Array(geom.volume) .- exported.vol)),
        area_maxdiff = maximum(abs.(Array(geom.face_area) .- area_expected)),
        normal_maxdiff = maximum(vcat(abs.(Array(geom.normal_x) .- nx_expected),
                                      abs.(Array(geom.normal_y) .- ny_expected),
                                      abs.(Array(geom.normal_z) .- nz_expected))),
        csr_count_maxdiff = maximum(abs.(counts_actual .- counts_expected)),
        csr_entries = length(geom.cell_faces),
        csr_entries_expected = sum(counts_expected),
    )
end

function conservation_drift(initial, final)
    return (; mass = final.mass - initial.mass,
            mx = final.mx - initial.mx,
            my = final.my - initial.my,
            mz = final.mz - initial.mz,
            energy = final.energy - initial.energy)
end

function write_report(path, exported, geom_host, geom_stats, before, cpu_rows, gpu_rows,
                      diffs, gpu_enabled, recon_cpu_rows, recon_gpu_rows,
                      recon_diffs, dt_stats, rebuild)
    cpu_after = cpu_rows[end]
    gpu_after = isempty(gpu_rows) ? nothing : gpu_rows[end]
    face_stats = moving_face_stats(geom_host)
    cpu_drift = conservation_drift(before, cpu_after)
    gpu_drift = gpu_after === nothing ? nothing : conservation_drift(before, gpu_after)
    open(path, "w") do io
        println(io, "# AREPO 3-D geometry gate for PowerFoam")
        println(io)
        println(io, "This initializes the stock AREPO 3-D turbulence case, exports AREPO's")
        println(io, "live Voronoi face rings, converts them to `ArepoMeshArrays3D`, and")
        println(io, "runs moving-face HLL/LLF steps through the Julia face-table kernel")
        println(io, "on CPU and Metal. The first-order path is the multi-step stability")
        println(io, "control; the reconstructed path uses AREPO's exported gradients")
        println(io, "and the Julia face predictor for a one-step production-geometry gate.")
        println(io)
        @printf(io, "- AREPO library: `%s`\n", ArepoLib.libpath())
        @printf(io, "- N: %d^3\n", N)
        @printf(io, "- cells: %d\n", exported.ng)
        @printf(io, "- faces: %d\n", length(exported.geo.nv))
        @printf(io, "- vertices: %d\n", size(exported.geo.verts, 1))
        @printf(io, "- volume sum from AREPO: %.12g\n", sum(exported.vol))
        @printf(io, "- volume sum in PowerFoam table: %.12g\n", sum(geom_host.volume))
        @printf(io, "- dt: %.6g\n", DT)
        @printf(io, "- steps: %d\n", NSTEPS)
        @printf(io, "- Riemann solver: %s\n", RIEMANN)
        @printf(io, "- face speed rms/max: %.8g / %.8g\n",
                face_stats.speed_rms, face_stats.speed_max)
        @printf(io, "- normal face speed rms/maxabs: %.8g / %.8g\n",
                face_stats.normal_rms, face_stats.normal_maxabs)
        @printf(io, "- AREPO-style hydro dt min/median/max: %.8g / %.8g / %.8g\n",
                dt_stats.min, dt_stats.median, dt_stats.max)
        println(io)
        println(io, "## Geometry conversion parity")
        println(io)
        println(io, "| check | value |")
        println(io, "| --- | ---: |")
        @printf(io, "| cells | %d |\n", geom_stats.cells)
        @printf(io, "| faces expected / actual | %d / %d |\n",
                geom_stats.faces_expected, geom_stats.faces_actual)
        @printf(io, "| vertices | %d |\n", geom_stats.vertices)
        @printf(io, "| c1 mismatches | %d |\n", geom_stats.c1_mismatches)
        @printf(io, "| c2 mismatches | %d |\n", geom_stats.c2_mismatches)
        @printf(io, "| volume max abs diff | %.12g |\n", geom_stats.volume_maxdiff)
        @printf(io, "| face area max abs diff | %.12g |\n", geom_stats.area_maxdiff)
        @printf(io, "| normal max abs diff | %.12g |\n", geom_stats.normal_maxdiff)
        @printf(io, "| CSR cell-face count max diff | %d |\n", geom_stats.csr_count_maxdiff)
        @printf(io, "| CSR entries expected / actual | %d / %d |\n",
                geom_stats.csr_entries_expected, geom_stats.csr_entries)
        println(io)
        println(io, "## First-order ALE diagnostics")
        println(io)
        println(io, "| state | step | time | vrms | mach_rms | density_rms | rho_min | rho_max | pmin | mass | energy |")
        println(io, "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for r in (before, cpu_after)
            @printf(io, "| %s | %d | %.8g | %.8g | %.8g | %.8g | %.8g | %.8g | %.8g | %.8g | %.8g |\n",
                    r.label, r.step, r.time, r.vrms, r.mach_rms, r.density_rms,
                    r.rho_min, r.rho_max, r.pmin, r.mass, r.energy)
        end
        if gpu_after !== nothing
            r = gpu_after
            @printf(io, "| %s | %d | %.8g | %.8g | %.8g | %.8g | %.8g | %.8g | %.8g | %.8g | %.8g |\n",
                    r.label, r.step, r.time, r.vrms, r.mach_rms, r.density_rms,
                    r.rho_min, r.rho_max, r.pmin, r.mass, r.energy)
        end
        println(io)
        println(io, "## Conservation drift")
        println(io)
        println(io, "| backend | dmass | dmx | dmy | dmz | denergy |")
        println(io, "| --- | ---: | ---: | ---: | ---: | ---: |")
        @printf(io, "| CPU Float32 | %.9g | %.9g | %.9g | %.9g | %.9g |\n",
                cpu_drift.mass, cpu_drift.mx, cpu_drift.my, cpu_drift.mz,
                cpu_drift.energy)
        if gpu_drift !== nothing
            @printf(io, "| Metal Float32 | %.9g | %.9g | %.9g | %.9g | %.9g |\n",
                    gpu_drift.mass, gpu_drift.mx, gpu_drift.my, gpu_drift.mz,
                    gpu_drift.energy)
        end
        println(io)
        println(io, "## CPU/GPU field differences after final PowerFoam step")
        println(io)
        if gpu_enabled
            println(io, "| field | max_abs_diff |")
            println(io, "| --- | ---: |")
            @printf(io, "| D | %.9g |\n", diffs.D)
            @printf(io, "| Mx | %.9g |\n", diffs.Mx)
            @printf(io, "| My | %.9g |\n", diffs.My)
            @printf(io, "| Mz | %.9g |\n", diffs.Mz)
            @printf(io, "| E | %.9g |\n", diffs.E)
        else
            println(io, "Metal was not available in this Julia environment.")
        end
        println(io)
        println(io, "## Reconstructed predictor one-step gate")
        println(io)
        println(io, "| backend | vrms | mach_rms | density_rms | rho_min | rho_max | pmin | mass | energy |")
        println(io, "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        rcpu = recon_cpu_rows[end]
        @printf(io, "| CPU Float32 | %.8g | %.8g | %.8g | %.8g | %.8g | %.8g | %.8g | %.8g |\n",
                rcpu.vrms, rcpu.mach_rms, rcpu.density_rms, rcpu.rho_min,
                rcpu.rho_max, rcpu.pmin, rcpu.mass, rcpu.energy)
        if !isempty(recon_gpu_rows)
            rgpu = recon_gpu_rows[end]
            @printf(io, "| Metal Float32 | %.8g | %.8g | %.8g | %.8g | %.8g | %.8g | %.8g | %.8g |\n",
                    rgpu.vrms, rgpu.mach_rms, rgpu.density_rms, rgpu.rho_min,
                    rgpu.rho_max, rgpu.pmin, rgpu.mass, rgpu.energy)
        end
        if gpu_enabled
            println(io)
            println(io, "| reconstructed field | CPU/Metal max_abs_diff |")
            println(io, "| --- | ---: |")
            @printf(io, "| D | %.9g |\n", recon_diffs.D)
            @printf(io, "| Mx | %.9g |\n", recon_diffs.Mx)
            @printf(io, "| My | %.9g |\n", recon_diffs.My)
            @printf(io, "| Mz | %.9g |\n", recon_diffs.Mz)
            @printf(io, "| E | %.9g |\n", recon_diffs.E)
        end
        println(io)
        println(io, "## Hosted AREPO mesh rebuild gate")
        println(io)
        println(io, "| status | time_before | time_after | faces | vertices | volume_sum | center_disp_rms | center_disp_max |")
        println(io, "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        @printf(io, "| %s | %.8g | %.8g | %d -> %d | %d -> %d | %.12g -> %.12g | %.8g | %.8g |\n",
                rebuild.status, rebuild.time_before, rebuild.time_after,
                rebuild.old_faces, rebuild.new_faces,
                rebuild.old_vertices, rebuild.new_vertices,
                rebuild.old_volume_sum, rebuild.new_volume_sum,
                rebuild.center_disp_rms, rebuild.center_disp_max)
        println(io)
        println(io, "## Interpretation")
        println(io)
        println(io, "The Julia hydro path now has separate gates for AREPO-gradient")
        println(io, "reconstruction, predictor evolution, ALE fluxes, and the hydro Courant")
        println(io, "timestep rule. The mesh-rebuild row is still hosted by AREPO's native")
        println(io, "3-D tessellator; a native Julia 3-D Voronoi rebuild remains the next")
        println(io, "implementation step.")
    end
end

function write_csv(path, rows)
    open(path, "w") do io
        println(io, "label,step,time,mass,mx,my,mz,energy,vrms,mach_rms,density_rms,rho_min,rho_max,pmin")
        for r in rows
            @printf(io, "%s,%d,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g\n",
                    r.label, r.step, r.time, r.mass, r.mx, r.my, r.mz, r.energy,
                    r.vrms, r.mach_rms, r.density_rms, r.rho_min, r.rho_max,
                    r.pmin)
        end
    end
end

function main()
    mkpath(OUTDIR)
    dir = stage_arepo_case(N)
    exported = arepo_initial_export(dir)
    try
        cpu_be = KernelAbstractions.CPU()
        cpu_geom, cpu_state, geom_host, _ = make_state_and_geom(exported, cpu_be)
        geom_stats = geometry_parity_stats(exported, geom_host)
        before = diagnostics("AREPO init in PowerFoam table", cpu_state, cpu_geom, 0, 0.0)
        cpu_rows = [diagnostics("PowerFoam CPU", cpu_state, cpu_geom, 0, 0.0)]
        for step in 1:NSTEPS
            finite_volume_step_3d!(cpu_state, cpu_geom; dt = DT, gamma = GAMMA,
                                   riemann = RIEMANN)
            push!(cpu_rows, diagnostics("PowerFoam CPU", cpu_state, cpu_geom,
                                        step, step * DT))
        end

        gpu_be = maybe_metal_backend()
        gpu_rows = NamedTuple[]
        diffs = (; D = NaN, Mx = NaN, My = NaN, Mz = NaN, E = NaN)
        if gpu_be !== nothing
            gpu_geom, gpu_state, _, _ = make_state_and_geom(exported, gpu_be)
            push!(gpu_rows, diagnostics("PowerFoam Metal", gpu_state, gpu_geom, 0, 0.0))
            for step in 1:NSTEPS
                finite_volume_step_3d!(gpu_state, gpu_geom; dt = DT, gamma = GAMMA,
                                       riemann = RIEMANN)
                push!(gpu_rows, diagnostics("PowerFoam Metal", gpu_state, gpu_geom,
                                            step, step * DT))
            end
            diffs = compare_states(cpu_state, gpu_state)
        end
        recon_cpu_rows, recon_cpu_state = run_reconstructed_once(exported, cpu_be;
                                                                 dt = DT,
                                                                 riemann = RIEMANN)
        recon_gpu_rows = NamedTuple[]
        recon_diffs = (; D = NaN, Mx = NaN, My = NaN, Mz = NaN, E = NaN)
        if gpu_be !== nothing
            recon_gpu_rows, recon_gpu_state = run_reconstructed_once(exported, gpu_be;
                                                                     dt = DT,
                                                                     riemann = RIEMANN)
            recon_diffs = compare_states(recon_cpu_state, recon_gpu_state)
        end
        dt_stats = timestep_stats(exported)
        rebuild = hosted_rebuild_stats(exported)
        write_csv(joinpath(OUTDIR, "metrics.csv"),
                  isempty(gpu_rows) ? vcat([before], cpu_rows) :
                  vcat([before], cpu_rows, gpu_rows))
        write_report(joinpath(OUTDIR, "README.md"), exported, geom_host, geom_stats, before,
                     cpu_rows, gpu_rows, diffs, gpu_be !== nothing,
                     recon_cpu_rows, recon_gpu_rows, recon_diffs, dt_stats, rebuild)
        @printf("wrote %s\n", joinpath(OUTDIR, "metrics.csv"))
        @printf("wrote %s\n", joinpath(OUTDIR, "README.md"))
        @printf("AREPO geometry: cells=%d faces=%d vertices=%d volume_sum=%.9g\n",
                exported.ng, length(exported.geo.nv), size(exported.geo.verts, 1),
                sum(exported.vol))
        @printf("geometry conversion: c1_mismatch=%d c2_mismatch=%d volume=%.3g area=%.3g normal=%.3g csr_count=%d\n",
                geom_stats.c1_mismatches, geom_stats.c2_mismatches,
                geom_stats.volume_maxdiff, geom_stats.area_maxdiff,
                geom_stats.normal_maxdiff, geom_stats.csr_count_maxdiff)
        cpu_after = cpu_rows[end]
        @printf("CPU after %d steps: vrms=%.6g density_rms=%.6g\n",
                NSTEPS, cpu_after.vrms, cpu_after.density_rms)
        if !isempty(gpu_rows)
            gpu_after = gpu_rows[end]
            @printf("GPU after %d steps: vrms=%.6g density_rms=%.6g\n",
                    NSTEPS, gpu_after.vrms, gpu_after.density_rms)
            @printf("field max abs diffs: D=%.4g Mx=%.4g My=%.4g Mz=%.4g E=%.4g\n",
                    diffs.D, diffs.Mx, diffs.My, diffs.Mz, diffs.E)
            @printf("reconstructed one-step diffs: D=%.4g Mx=%.4g My=%.4g Mz=%.4g E=%.4g\n",
                    recon_diffs.D, recon_diffs.Mx, recon_diffs.My,
                    recon_diffs.Mz, recon_diffs.E)
        end
        @printf("AREPO hosted rebuild: status=%s time %.6g -> %.6g faces %d -> %d\n",
                rebuild.status, rebuild.time_before, rebuild.time_after,
                rebuild.old_faces, rebuild.new_faces)
    finally
        ArepoLib.finalize(exported.h)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
