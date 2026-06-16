include(joinpath(@__DIR__, "arepo_geometry_gate_3d.jl"))

const GAP_OUTBASE = joinpath(@__DIR__, "out", "arepo_one_step_gap_3d")
const GAP_OUTDIR = joinpath(GAP_OUTBASE, replace(RUN_TAG, "." => "p"))

function id_permutation(ids_before, ids_after, ng)
    pos = Dict{Int64,Int}()
    for i in 1:ng
        pos[Int64(ids_before[i])] = i
    end
    perm = Vector{Int}(undef, ng)
    for i in 1:ng
        perm[i] = pos[Int64(ids_after[i])]
    end
    return perm
end

function primitive_after_arepo_step(h, ng)
    rho = ArepoLib.get_cell_field(h, :rho)
    pressure = ArepoLib.get_cell_field(h, :pressure)
    vel = ArepoLib.get_particle_field(h, :vel)[1:ng, :]
    volume = ArepoLib.get_cell_field(h, :volume)
    center = ArepoLib.get_cell_field(h, :center)
    ids = ArepoLib.get_particle_ids(h)[1:ng]
    return (; rho, pressure, vel, volume, center, ids)
end

function primitive_from_powerfoam(state)
    p = conserved_to_primitive_3d(state; gamma = GAMMA)
    return (; rho = p.rho, pressure = p.pressure,
            vel = hcat(p.vx, p.vy, p.vz))
end

function primitive_gap(pf, arepo, perm)
    return (;
        rho = maximum(abs.(pf.rho[perm] .- arepo.rho)),
        vx = maximum(abs.(pf.vel[perm, 1] .- arepo.vel[:, 1])),
        vy = maximum(abs.(pf.vel[perm, 2] .- arepo.vel[:, 2])),
        vz = maximum(abs.(pf.vel[perm, 3] .- arepo.vel[:, 3])),
        pressure = maximum(abs.(pf.pressure[perm] .- arepo.pressure)),
    )
end

function displacement_stats(center0, center1, perm, box)
    d = center1 .- center0[perm, :]
    d .= ifelse.(d .> 0.5 * box, d .- box, ifelse.(d .< -0.5 * box, d .+ box, d))
    r = sqrt.(sum(d .* d; dims = 2)[:])
    return (; rms = sqrt(mean(r .* r)), max = maximum(r))
end

function write_gap_report(path, exported, dt_arepo, predicted_dt, pf_dt,
                          gap_recon, gap_first, disp, arepo_diag,
                          pf_recon_diag, pf_first_diag)
    open(path, "w") do io
        println(io, "# AREPO vs PowerFoam One-Step Gap")
        println(io)
        println(io, "This diagnostic starts both codes from the same live AREPO state.")
        println(io, "AREPO advances one synchronization step with its native hierarchy,")
        println(io, "mesh regularization, rebuild, predictor, and flux path. PowerFoam")
        println(io, "advances once on the initial exported AREPO geometry for the same")
        println(io, "elapsed time. The gap therefore measures the combined remaining")
        println(io, "predictor/flux/mesh-motion/timestep-hierarchy difference after the")
        println(io, "initial-state, geometry, and gradient gates have passed.")
        println(io)
        @printf(io, "- AREPO library: `%s`\n", ArepoLib.libpath())
        @printf(io, "- N: %d^3\n", N)
        @printf(io, "- cells: %d\n", exported.ng)
        @printf(io, "- initial faces: %d\n", length(exported.geo.nv))
        @printf(io, "- PowerFoam predicted sync dt: %.12g\n", predicted_dt)
        @printf(io, "- AREPO one-step dt: %.12g\n", dt_arepo)
        @printf(io, "- PowerFoam dt used: %.12g\n", pf_dt)
        println(io)
        println(io, "## Primitive Max Absolute Difference")
        println(io)
        println(io, "| path | rho | vx | vy | vz | pressure |")
        println(io, "| --- | ---: | ---: | ---: | ---: | ---: |")
        @printf(io, "| reconstructed PowerFoam vs AREPO | %.12g | %.12g | %.12g | %.12g | %.12g |\n",
                gap_recon.rho, gap_recon.vx, gap_recon.vy, gap_recon.vz, gap_recon.pressure)
        @printf(io, "| first-order PowerFoam vs AREPO | %.12g | %.12g | %.12g | %.12g | %.12g |\n",
                gap_first.rho, gap_first.vx, gap_first.vy, gap_first.vz, gap_first.pressure)
        println(io)
        println(io, "## Mesh Motion")
        println(io)
        @printf(io, "- AREPO center displacement rms/max: %.12g / %.12g\n", disp.rms, disp.max)
        println(io)
        println(io, "## Diagnostics")
        println(io)
        println(io, "| path | mass | mx | my | mz | energy | vrms | mach rms | density rms | pmin |")
        println(io, "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for (label, d) in (("AREPO post-step", arepo_diag),
                           ("PowerFoam reconstructed", pf_recon_diag),
                           ("PowerFoam first-order", pf_first_diag))
            @printf(io, "| %s | %.12g | %.12g | %.12g | %.12g | %.12g | %.12g | %.12g | %.12g | %.12g |\n",
                    label, d.mass, d.mx, d.my, d.mz, d.energy, d.vrms,
                    d.mach_rms, d.density_rms, d.pmin)
        end
    end
end

function arepo_diag_from_primitive(a)
    D = a.rho
    Mx = a.rho .* a.vel[:, 1]
    My = a.rho .* a.vel[:, 2]
    Mz = a.rho .* a.vel[:, 3]
    E = a.pressure ./ (GAMMA - 1) .+
        0.5 .* a.rho .* sum(a.vel .* a.vel; dims = 2)[:]
    geom = ArepoMeshArrays3D(Int32[], Int32[], Int32[1], Int32[], Int32[],
                             a.volume, Float64[], Float64[], Float64[],
                             Float64[], Float64[], Float64[], Float64[])
    state = EulerState3D(D, Mx, My, Mz, E)
    return diagnostics("AREPO primitive", state, geom, 1, 0.0)
end

function main_gap()
    mkpath(GAP_OUTDIR)
    dir = stage_arepo_case(N; riemann = RIEMANN)
    exported = arepo_initial_export(dir)
    try
        ids0 = ArepoLib.get_particle_ids(exported.h)[1:exported.ng]
        center0 = copy(exported.center)
        t0 = ArepoLib.sim_time(exported.h)
        status = ArepoLib.run_step!(exported.h)
        t1 = ArepoLib.sim_time(exported.h)
        dt_arepo = t1 - t0
        arepo_after = primitive_after_arepo_step(exported.h, exported.ng)
        hydro_dt = arepo_hydro_dt_3d(exported.vol, exported.pressure, exported.rho;
                                     gamma = GAMMA, courant = 0.3,
                                     max_dt = 0.05, min_dt = 1e-6)
        predicted_dt = minimum(arepo_system_step_3d(hydro_dt))

        cpu_be = KernelAbstractions.CPU()
        geom, state, geom_host, _ = make_state_and_geom(exported, cpu_be)
        _, recon_state = run_reconstructed_once(exported, cpu_be;
                                                dt = dt_arepo, riemann = RIEMANN)
        finite_volume_step_3d!(state, geom; dt = dt_arepo, gamma = GAMMA,
                               riemann = RIEMANN)

        perm = id_permutation(ids0, arepo_after.ids, exported.ng)
        gap_recon = primitive_gap(primitive_from_powerfoam(recon_state), arepo_after, perm)
        gap_first = primitive_gap(primitive_from_powerfoam(state), arepo_after, perm)
        disp = displacement_stats(center0, arepo_after.center, perm, exported.box)
        arepo_diag = arepo_diag_from_primitive(arepo_after)
        pf_recon_diag = diagnostics("PowerFoam reconstructed", recon_state, geom, 1, dt_arepo)
        pf_first_diag = diagnostics("PowerFoam first-order", state, geom, 1, dt_arepo)
        report = joinpath(GAP_OUTDIR, "README.md")
        write_gap_report(report, exported, dt_arepo, predicted_dt, dt_arepo,
                         gap_recon, gap_first, disp, arepo_diag,
                         pf_recon_diag, pf_first_diag)
        @printf("wrote %s\n", report)
        @printf("AREPO status=%s dt=%.9g predicted=%.9g\n", status, dt_arepo, predicted_dt)
        @printf("reconstructed gap: rho=%.4g vx=%.4g vy=%.4g vz=%.4g p=%.4g\n",
                gap_recon.rho, gap_recon.vx, gap_recon.vy, gap_recon.vz,
                gap_recon.pressure)
        @printf("first-order gap: rho=%.4g vx=%.4g vy=%.4g vz=%.4g p=%.4g\n",
                gap_first.rho, gap_first.vx, gap_first.vy, gap_first.vz,
                gap_first.pressure)
    finally
        ArepoLib.finalize(exported.h)
    end
end

main_gap()
