# Santa Barbara cluster: hydro (PPMKernels) + gravity (PoissonKernels) as :julia slots
# under Enzo's OWN AMR cosmology hierarchy. Enzo orchestrates AMR (refine/subcycle/
# project), particles (push + deposit) and the comoving expansion on CPU; we compute
# the per-grid PHYSICS with our ported kernels — the path to running SB on Metal.
#
#   :gravity slot  → build δ (gas + CIC-deposited DM) → fft_poisson_root! (root FFT) →
#                    comp_accel! → write Enzo's AccelerationField (so gas AND the DM
#                    particles feel our gravity).
#   :hydro slot    → read primitives + accel → ppm_step_3d! (Enzo PPM, gravity-coupled,
#                    dual-energy) → write back.
#
# First cut: ROOT grid only (no refinement at z=63 in a few steps), NON-conservative
# (reflux=false, projection-only AMR), a=1 (z=63 init). Backend/precision is set by
# BE/T below — CPU-f32 is the FAITHFUL comparison for the later Metal-f32 run.
#
# Run (env BACKEND=cpu|metal):
#   <julia> --project=lib/PPMKernels/test lib/EnzoLib/examples/sb_metal_amr.jl [cycles]

using EnzoLib, PPMKernels, PoissonKernels, Dates, Printf
try; @eval using Metal; catch; end

const SB = "/Users/tabel/Projects/enzo-dev/run/CosmologySimulation/SantaBarbaraCluster"
const NG = 4                       # ppm_step_3d! ghost zones (set NumberOfGhostZones=4)
const GAMMA = 5/3
const OMEGA_B = 0.1; const OMEGA_CDM = 0.9
const iD, iV1, iV2, iV3, iTE, iGE = 0, 1, 2, 3, 4, 5
const BE = Symbol(get(ENV, "BACKEND", "cpu"))
const T  = BE === :metal ? Float32 : Float32          # CPU-f32 = faithful comparison
const _step = Ref(0)

envbool(name, default=false) = lowercase(get(ENV, name, string(default))) in ("1", "true", "yes", "on")
envint(name, default) = parse(Int, get(ENV, name, string(default)))
envpath(name, default) = abspath(get(ENV, name, default))

function _campaign_dir()
    stamp = Dates.format(now(), "yyyymmdd-HHMMSS")
    return envpath("SB_CAMPAIGN_OUT", joinpath(@__DIR__, "sb_campaign_out", "run-$stamp-$BE"))
end

function _write_summary(path; rows, backend, precision, maxcyc, stop_on_refinement, status)
    open(path, "w") do io
        println(io, "# Santa Barbara EnzoNG slot campaign")
        println(io)
        println(io, "- backend: `$backend`")
        println(io, "- precision: `$precision`")
        println(io, "- requested cycles: `$maxcyc`")
        println(io, "- stop on refinement: `$stop_on_refinement`")
        println(io, "- status: `$status`")
        println(io)
        println(io, "| cycle | time | grids/level | rho_max | mass_drift | refined | seconds |")
        println(io, "|---:|---:|:---|---:|---:|:---:|---:|")
        for r in rows
            @printf(io, "| %d | %.8g | `%s` | %.8g | %.3e | %s | %.3f |\n",
                    r.cycle, r.time, r.grids, r.rhomax, r.mass_drift,
                    r.refined ? "yes" : "no", r.seconds)
        end
    end
    return path
end

dev(be, a) = PoissonKernels.to_device(be, a, T)
active_of(flat, gd, N) = Array(reshape(Float64.(flat), gd[1], gd[2], gd[3])[NG+1:NG+N, NG+1:NG+N, NG+1:NG+N])

# cubic N grid → (N+2NG)³ with NG-cell PERIODIC ghosts
function pad_periodic(φ::Array{Float64,3})
    N = size(φ, 1); M = N + 2NG; full = Array{Float64,3}(undef, M, M, M)
    @inbounds for k in 1:M, j in 1:M, i in 1:M
        full[i, j, k] = φ[mod(i-NG-1, N)+1, mod(j-NG-1, N)+1, mod(k-NG-1, N)+1]
    end
    full
end
function place_active(act::Array{Float64,3}, gd)
    full = zeros(Float64, gd[1], gd[2], gd[3]); N = size(act, 1)
    full[NG+1:NG+N, NG+1:NG+N, NG+1:NG+N] .= act
    vec(full)
end

cic!(rho, pos, N) = begin
    @inbounds for p in 1:size(pos,1)
        gx=mod(pos[p,1],1.0)*N; gy=mod(pos[p,2],1.0)*N; gz=mod(pos[p,3],1.0)*N
        i=floor(Int,gx);fx=gx-i;j=floor(Int,gy);fy=gy-j;k=floor(Int,gz);fz=gz-k
        i0=mod(i,N)+1;i1=mod(i+1,N)+1;j0=mod(j,N)+1;j1=mod(j+1,N)+1;k0=mod(k,N)+1;k1=mod(k+1,N)+1
        rho[i0,j0,k0]+=(1-fx)*(1-fy)*(1-fz);rho[i1,j0,k0]+=fx*(1-fy)*(1-fz)
        rho[i0,j1,k0]+=(1-fx)*fy*(1-fz);rho[i1,j1,k0]+=fx*fy*(1-fz)
        rho[i0,j0,k1]+=(1-fx)*(1-fy)*fz;rho[i1,j0,k1]+=fx*(1-fy)*fz
        rho[i0,j1,k1]+=(1-fx)*fy*fz;rho[i1,j1,k1]+=fx*fy*fz
    end; rho
end

# ── :gravity slot — root FFT gravity → Enzo AccelerationField; subgrids run the
# CERTIFIED parent-Dirichlet W-cycle (poisson_gravity_hook: prepare_density →
# vcycle dirichlet → set_potential → Enzo's own differencing; bit-identical
# orbits vs gravity=:enzo on GravityTest) ──
const SUBGRID_GRAV! = EnzoLib.poisson_gravity_hook()
function gravity!(h, level, dt)
    level == 0 || return SUBGRID_GRAV!(h, level, dt)
    bep = PoissonKernels.backend(BE)
    n = EnzoLib.session_num_grids_on_level(h, 0)
    for i in 0:n-1
        g = EnzoLib.problem_grid_index_on_level(h, 0, i)
        gd = Tuple(Int.(EnzoLib.problem_grid_dims(h, g))); N = gd[1] - 2NG
        gas = active_of(EnzoLib.read_density(h; grid=g), gd, N)
        pos = EnzoLib.read_particles(h)
        dm = cic!(zeros(N,N,N), pos, N); dm .*= OMEGA_CDM/(sum(dm)/length(dm))
        gas .*= OMEGA_B/(sum(gas)/length(gas))
        δ = gas .+ dm; δ ./= (sum(δ)/length(δ)); δ .-= 1.0
        φ = PoissonKernels.device_zeros(bep, T, (N,N,N))
        PoissonKernels.fft_poisson_root!(φ, dev(bep, δ); G=1.0, a=1.0, boxsize=1.0)
        φf = dev(bep, pad_periodic(Float64.(PoissonKernels.to_host(φ))))
        a1 = PoissonKernels.device_zeros(bep, T, (N,N,N)); a2 = similar(a1); a3 = similar(a1)
        PoissonKernels.comp_accel!(a1, a2, a3, φf; iflag=1, start=(NG,NG,NG), del=(1.0/N,1.0/N,1.0/N))
        EnzoLib.problem_set_acceleration(h, 0, place_active(Float64.(PoissonKernels.to_host(a1)), gd); grid=g)
        EnzoLib.problem_set_acceleration(h, 1, place_active(Float64.(PoissonKernels.to_host(a2)), gd); grid=g)
        EnzoLib.problem_set_acceleration(h, 2, place_active(Float64.(PoissonKernels.to_host(a3)), gd); grid=g)
        # root φ → PotentialField too, so a CHILD's BC interpolation
        # (PrepareDensityField at level 1) reads OUR root solution
        gmfd = EnzoLib.problem_gmf_dims(h, g)
        if gmfd == (N, N, N)
            EnzoLib.problem_set_potential(h, Float64.(PoissonKernels.to_host(φ)), g)
        else
            @warn "root GMF dims ≠ active dims — child BC keeps Enzo's root φ" gmfd N maxlog = 1
        end
    end
    return nothing
end

# chemistry species advected as passive colour (mass-density) fields, riding the
# PPM mass flux: HII(9), H2I(14), HDI(18) — the reduced-network advected species.
# Absent (e.g. plain hydro / SB) ⇒ empty ⇒ zero-overhead no-op.
const SPECIES_FTYPES = (9, 14, 18)
function _species_indices(h, g)
    idx = Int[]
    for ft in SPECIES_FTYPES
        fi = try EnzoLib.field_index(h, ft; grid=g) catch; -1 end
        fi >= 0 && push!(idx, fi)
    end
    return idx
end

# ── :hydro slot — Enzo PPM with our gravity, per grid ──
function hydro!(h, level, dt)
    bep = PPMKernels.backend(BE)
    n = EnzoLib.session_num_grids_on_level(h, level)
    order = isodd(_step[]) ? (3,2,1) : (1,2,3); _step[] += 1
    acosmo = EnzoLib.session_cosmology(h)[1]                 # Enzo's internal a (=1 if non-cosmo)
    for i in 0:n-1
        g = EnzoLib.problem_grid_index_on_level(h, level, i)
        gd = Tuple(Int.(EnzoLib.problem_grid_dims(h, g)))
        # Enzo's PPM solves with cell width = a·CellWidth ("multiply dx by a(n+½)",
        # Grid_SolveHydroEquations.C). The default dx=1 makes the flux divergence
        # (continuity) ~N/a too weak → baryons never catch up.  Use the real dx.
        gl, gr = EnzoLib.problem_grid_edge(h, g)
        dxc = acosmo * (gr[1] - gl[1]) / (gd[1] - 2*NG)      # comoving cell width × a
        f(fi) = dev(bep, EnzoLib.problem_get_field(h, fi, g))
        d, e, ge = f(iD), f(iTE), f(iGE); vx, vy, vz = f(iV1), f(iV2), f(iV3)
        ax = EnzoLib.problem_get_acceleration(h,0,g)
        ay = EnzoLib.problem_get_acceleration(h,1,g)
        az = EnzoLib.problem_get_acceleration(h,2,g)
        if get(ENV, "CIC_DEBUG", "0") == "1" && i == 0
            @printf("  [hydro dbg] a=%.4f dx=%.3e max|g|=%.3e max|vx|=%.3e dt=%.3e (v·dt/dx=%.3f)\n",
                    acosmo, dxc, maximum(abs, ax),
                    maximum(abs, EnzoLib.problem_get_field(h, iV1, g)), dt,
                    maximum(abs, EnzoLib.problem_get_field(h, iV1, g))*dt/dxc); flush(stdout)
        end
        gx = dev(bep, ax); gy = dev(bep, ay); gz = dev(bep, az)
        sp   = _species_indices(h, g)                       # field indices present
        cols = Tuple(f(fi) for fi in sp)                    # device colour arrays
        PPMKernels.ppm_step_3d!(d, e, ge, vx, vy, vz, gx, gy, gz, gd, NG;
                                dt=dt, gamma=GAMMA, order=order, gravity=1, idual=1,
                                eta1=0.001, eta2=0.1,           # Enzo's DualEnergyFormalismEta1/Eta2:
                                dx=dxc, colours=cols)           # without them the dual-energy pressure
                                                                # override never fires → cold high-z gas
                                                                # fragments at the grid scale (~60× excess
                                                                # small-scale baryon P(k) vs native PPM).
        wr(fi, a) = EnzoLib.problem_set_field(h, fi, Float64.(PPMKernels.to_host(a)); grid=g)
        wr(iD, d); wr(iTE, e); wr(iGE, ge); wr(iV1, vx); wr(iV2, vy); wr(iV3, vz)
        for (fi, c) in zip(sp, cols)                         # write advected species back
            wr(fi, c)
        end
    end
    return nothing
end

function main()
    maxcyc = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : envint("SB_MAXCYC", 4)
    stop_on_refinement = envbool("SB_STOP_ON_REFINEMENT", false)
    outdir = _campaign_dir()
    mkpath(outdir)
    csv = joinpath(outdir, "diagnostics.csv")
    summary = joinpath(outdir, "summary.md")
    EnzoLib.grid_available() || error("grid dylib not built")
    pf = joinpath(SB, "SB_metal.enzo")
    write(pf, replace(read(joinpath(SB, "SantaBarbaraCluster.enzo"), String),
                      r"GreensFunctionMaxNumber.*" => "GreensFunctionMaxNumber   = 30\nNumberOfGhostZones        = 4"))
    @printf("SB hydro(PPM)+gravity(FFT) on Enzo AMR — backend=%s precision=%s, %d cycles\n", BE, T, maxcyc)
    cd(SB) do
        h = EnzoLib.session_init(pf); h == C_NULL && error("session_init failed")
        try
            eng = EnzoLib.EngineConfig(; hydro=:julia, gravity=:julia, comoving_expansion=:enzo,
                                       reflux=false,
                                       hooks=Dict{Symbol,Function}(:hydro=>hydro!, :gravity=>gravity!))
            EnzoLib.session_rebuild(h, 0)
            m0 = EnzoLib.session_global_field_integral(h, 0)
            rows = NamedTuple[]
            open(csv, "w") do io
                println(io, "cycle,time,grids_level0,grids_level1,grids_level2,rho_max,mass_drift,refined,seconds")
            end
            @printf("%-4s %-9s %-7s %-16s %-9s %-10s %-8s\n",
                    "cyc", "z", "t", "grids/level", "ρmax", "Δmass/M", "seconds")
            cyc = 0
            status = "completed"
            while cyc < maxcyc
                tstart = time()
                EnzoLib.evolve_level!(h, 0, 0.0; engine=eng, regrid=true)
                seconds = time() - tstart
                EnzoLib.session_rebuild(h, 0)
                ρ = EnzoLib.problem_get_field(h, iD, 0)
                m = EnzoLib.session_global_field_integral(h, 0)
                ngl = Int[EnzoLib.session_num_grids_on_level(h, l) for l in 0:2]
                refined = any(>(0), ngl[2:end])
                t = EnzoLib.session_time(h)
                rhomax = maximum(ρ)
                mass_drift = abs(m - m0) / m0
                row = (cycle = cyc, time = t, grids = string(ngl), rhomax = rhomax,
                       mass_drift = mass_drift, refined = refined, seconds = seconds)
                push!(rows, row)
                open(csv, "a") do io
                    @printf(io, "%d,%.17g,%d,%d,%d,%.17g,%.17g,%s,%.17g\n",
                            cyc, t, ngl[1], ngl[2], ngl[3], rhomax, mass_drift,
                            refined, seconds)
                end
                @printf("%-4d %-9s %-7.4f %-16s %-9.3f %-10.1e %-8.2f\n",
                        cyc, "-", t, string(ngl), rhomax, mass_drift, seconds)
                if any(isnan, ρ)
                    println("  NaN — abort")
                    status = "aborted_nan"
                    break
                end
                cyc += 1
                if stop_on_refinement && refined
                    status = "stopped_at_refinement_onset"
                    break
                end
            end
            _write_summary(summary; rows = rows, backend = BE, precision = T,
                           maxcyc = maxcyc, stop_on_refinement = stop_on_refinement,
                           status = status)
            println(cyc >= maxcyc ? "SB hydro+gravity on Enzo AMR: RAN $maxcyc cycles" :
                    "SB hydro+gravity on Enzo AMR: $status")
            println("diagnostics: $csv")
            println("summary:     $summary")
        finally
            EnzoLib.free_problem(h)
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
