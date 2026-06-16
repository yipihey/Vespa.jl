include(joinpath(@__DIR__, "arepo_geometry_gate_3d.jl"))

const MESHVEL_OUTBASE = joinpath(@__DIR__, "out", "arepo_mesh_velocity_gate_3d")
const MESHVEL_OUTDIR = joinpath(MESHVEL_OUTBASE, @sprintf("N%d", N))

function mesh_velocity_metrics(exported)
    pos = ArepoLib.get_particle_field(exported.h, :pos)[1:exported.ng, :]
    velvertex = ArepoLib.get_cell_field(exported.h, :velvertex)
    hydro_dt = arepo_hydro_dt_3d(exported.vol, exported.pressure, exported.rho;
                                 gamma = GAMMA, courant = 0.3,
                                 max_dt = 0.05, min_dt = 1e-6)
    dt = minimum(arepo_system_step_3d(hydro_dt))
    vmesh = arepo_mesh_velocity_3d(pos, exported.center, exported.rho,
                                   exported.pressure, exported.vel,
                                   exported.cgrad, exported.vol, exported.geo;
                                   dt, gamma = GAMMA, box_size = exported.box,
                                   cell_shaping_speed = 0.5,
                                   cell_max_angle_factor = 2.25,
                                   use_face_angle = true,
                                   use_sound_speed = true)
    diff = vmesh .- velvertex
    speed_arepo = sqrt.(sum(velvertex .* velvertex; dims = 2)[:])
    speed_pf = sqrt.(sum(vmesh .* vmesh; dims = 2)[:])
    return (; dt, vmesh, velvertex,
            max_abs = maximum(abs.(diff)),
            rms = sqrt(mean(diff .* diff)),
            max_speed_arepo = maximum(speed_arepo),
            max_speed_pf = maximum(speed_pf),
            rms_speed_arepo = sqrt(mean(speed_arepo .* speed_arepo)),
            rms_speed_pf = sqrt(mean(speed_pf .* speed_pf)))
end

function write_meshvel_report(path, exported, m)
    open(path, "w") do io
        println(io, "# AREPO 3-D Mesh-Velocity Gate")
        println(io)
        println(io, "This gate compares PowerFoam's reconstructed non-cosmological")
        println(io, "AREPO-style mesh-generating-point velocity against AREPO's live")
        println(io, "`SphP[].VelVertex` export.  The implemented PowerFoam terms are")
        println(io, "fluid velocity, pressure-gradient half-step, and face-angle CM-drift")
        println(io, "regularization with a sound-speed velocity scale.")
        println(io)
        @printf(io, "- AREPO library: `%s`\n", ArepoLib.libpath())
        @printf(io, "- N: %d^3\n", N)
        @printf(io, "- cells: %d\n", exported.ng)
        @printf(io, "- sync dt used: %.12g\n", m.dt)
        println(io)
        println(io, "| metric | value |")
        println(io, "| --- | ---: |")
        @printf(io, "| max abs component diff | %.12g |\n", m.max_abs)
        @printf(io, "| rms component diff | %.12g |\n", m.rms)
        @printf(io, "| AREPO velvertex rms speed | %.12g |\n", m.rms_speed_arepo)
        @printf(io, "| PowerFoam mesh velocity rms speed | %.12g |\n", m.rms_speed_pf)
        @printf(io, "| AREPO velvertex max speed | %.12g |\n", m.max_speed_arepo)
        @printf(io, "| PowerFoam mesh velocity max speed | %.12g |\n", m.max_speed_pf)
        println(io)
        println(io, "Remaining differences are expected until the bridge exports the")
        println(io, "gravity acceleration/vorticity terms used by AREPO's optional")
        println(io, "regularization velocity cap.")
    end
end

function main_meshvel()
    mkpath(MESHVEL_OUTDIR)
    dir = stage_arepo_case(N)
    exported = arepo_initial_export(dir)
    try
        m = mesh_velocity_metrics(exported)
        report = joinpath(MESHVEL_OUTDIR, "README.md")
        write_meshvel_report(report, exported, m)
        @printf("wrote %s\n", report)
        @printf("mesh velocity max_abs=%.6g rms=%.6g arepo_rms=%.6g pf_rms=%.6g\n",
                m.max_abs, m.rms, m.rms_speed_arepo, m.rms_speed_pf)
    finally
        ArepoLib.finalize(exported.h)
    end
end

main_meshvel()

