# ── Phase 4.1/4.2 gates (ADR-0006): Moray + the exchange operator ─────────────
#
#  4.1  Moray on its native PhotonTest: the I-front must track the analytic
#       Strömgren solution (Iliev Test-1 physics) — monotone growth, ≤12% of
#       analytic from t ≥ 5 Myr (the known discrete-front lag at 32³).
#  4.2  The exchange: conservative deposit (NGP + CIC, exact ledger), trilinear
#       sampling, and the flagship-3 coupling — an Arepo gas field drives
#       Moray, and Moray's photo-rates come back onto the Voronoi cells as
#       heating, with the injection verified and Arepo still able to step.

using Test
using MultiCode
using EnzoLib, ArepoLib, CodeBridge

@testset "Phase 4.1: Moray ≈ Strömgren (Iliev-1)" begin
    if !(EnzoLib.grid_available() && isfile(MultiCode.ENZO_PHOTONTEST_PF))
        @test_skip false
    else
        r = run_moray_stromgren(t_end_myr = 10.0, snapshots = [3.0, 5.0, 10.0])
        try
            rs = [ri for (t, ri) in r.history]
            @test all(isfinite, rs)
            @test issorted(rs)                                # the front only grows
            for (t, ri) in r.history
                t < 5.0 && continue                           # early-time lag is physical
                @test abs(ri - stromgren_radius(t)) / stromgren_radius(t) < 0.12
            end
            @test maximum(r.fields.xHII) > 0.99               # ionized interior
            @test minimum(r.fields.xHII) < 0.01               # neutral exterior
            @test maximum(r.fields.kphHI) > 0                 # rates populated
            @info "Moray Strömgren gate" history = r.history analytic = [stromgren_radius(t) for (t, _) in r.history]
        finally
            r.free()
        end
    end
end

@testset "Phase 4.2: exchange operator (conservative deposit + sampling)" begin
    # synthetic CellSet: jittered lattice tiling the unit box, smooth ρ field
    m = 24
    dxl = 1.0 / m
    n3 = m^3
    pos = Matrix{Float64}(undef, n3, 3)
    k = 0
    for c in CartesianIndices((m, m, m))
        k += 1
        pos[k, :] .= ((Tuple(c) .- 0.5) .* dxl) .+ 0.2 * dxl .* (rand(3) .- 0.5)
    end
    rho = [1.0 + 0.5 * sin(2π * pos[i, 1]) * cos(2π * pos[i, 2]) for i in 1:n3]
    vol = fill(dxl^3, n3)
    mom = 0.1 .* rho .* ones(n3, 3)
    etot = 2.0 .* rho
    cs = CellSet(:synthetic, pos, vol, rho, mom, etot,
                 (length = 1.0, time = 1.0, density = 1.0), (;))
    lg = ledger(cs)

    for method in (:ngp, :cic)
        g = deposit_to_grid(cs, 16; method = method)     # internal assertions = the hard gate
        @test abs(sum(g.rho) / 16^3 - lg.mass) < 1e-13       # density mean = total mass
        @test abs(sum(g.etot) / 16^3 - lg.energy) < 1e-12
        @test all(>(0), g.vol) && abs(sum(g.vol) / 16^3 - 1.0) < 1e-12   # full coverage
    end
    # trilinear sampling recovers a smooth field at the cell positions
    n = 32
    grid = [sin(2π * (i - 0.5) / n) for i in 1:n, j in 1:n, k in 1:n]
    s = sample_at_points(grid, pos)
    @test maximum(abs.(s .- sin.(2π .* pos[:, 1]))) < 0.02   # O(dx²) interpolation
end

@testset "Phase 4.2: flagship 3 — Arepo gas drives Moray, rates come back" begin
    arepo_dir = ArepoLib.available() ? normpath(dirname(ArepoLib.libpath())) : ""
    py_ok = ArepoLib.available() && MultiCode._arepo_python(arepo_dir) !== nothing
    if !(py_ok && EnzoLib.grid_available() && isfile(MultiCode.ENZO_PHOTONTEST_PF))
        @test_skip false
    else
        # the DONOR: a live Arepo Sod state (its density structure is 1-D —
        # physically uniform in y/z — so its profile broadcasts onto the RT grid).
        # Arepo runs in its OWN worker process (ADR-0006 D2): its C-global state
        # cannot re-init in this process if an earlier test already ran it.
        a = run_arepo_sod(SodSpec(); worker = true)
        try
            cs = a.cs
            # 1-D deposit of the Voronoi state: conservative onto 32 x-bins
            n = 32
            prof = profile_x(cs)
            xs = prof.x
            rho_x = MultiCode.sample_profile_to_bins(prof, n)
            @test length(rho_x) == n
            # mass conservation of the binned profile (volume-weighted mean preserved)
            @test abs(sum(rho_x) / n - ledger(cs).mass) / ledger(cs).mass < 2e-2

            # broadcast ρ(x) onto the 32³ RT grid, LIGHT side toward the corner
            # source (flip the tube), and hand it to Moray.  Reference: the same
            # source in the unmodified uniform PhotonTest gas.
            den = [rho_x[n + 1 - i] for i in 1:n, j in 1:n, k in 1:n]
            # front position along the source ray (the first cell row): the
            # last cell with xHII > 0.5 — the same observable for both runs
            ray_front(f) = (c = findlast(>(0.5), f.xHII[:, 1, 1]); c === nothing ? 0.0 : (c - 0.5) / n)
            uni = run_moray_stromgren(t_end_myr = 8.0, snapshots = [8.0])
            x_uni = ray_front(uni.fields)
            uni.free()
            r = run_moray_stromgren(t_end_myr = 8.0, snapshots = [4.0, 8.0], density = den)
            try
                @test all(isfinite, r.fields.xHII)
                @test maximum(r.fields.xHII) > 0.9            # the source carved an HII region
                @test minimum(r.fields.xHII) < 0.01
                # the front must run FARTHER through the Arepo tube's light gas
                # (ρ=0.125 at the source side) than through the uniform gas:
                # R ∝ n_H^{-2/3} in equilibrium, n_H^{-1/3} pre-equilibrium.
                x_front = ray_front(r.fields)
                @test x_uni > 0                                # the reference front exists
                @test x_front > 1.3 * x_uni
                @info "front comparison" x_uniform = x_uni x_arepo_fed = x_front
                # density round-trip: Moray ran on EXACTLY the Arepo-supplied field
                @test maximum(abs.(r.fields.density .- den)) < 1e-10

                # ── rates BACK onto the Voronoi cells, injected as heating ──
                # sample along the source axis (the corner ray), where the rates live
                ng = ArepoLib.info(a.handle).numgas
                cellpos = hcat(1.0 .- cs.pos[:, 1],                  # un-flip the tube
                               fill(0.05, ng), fill(0.05, ng))
                gamma_heat = sample_at_points(Float64.(r.fields.photogamma), cellpos)
                @test length(gamma_heat) == ng && all(isfinite, gamma_heat)
                @test maximum(gamma_heat) > 0                  # somebody got heated
                u0 = ArepoLib.get_cell_field(a.handle, :utherm)
                kappa = 0.05 * maximum(u0) / max(maximum(gamma_heat), eps())  # declared coupling
                u1 = u0 .+ kappa .* gamma_heat
                ArepoLib.set_cell_field!(a.handle, :utherm, u1)
                @test ArepoLib.get_cell_field(a.handle, :utherm) == u1       # exact injection
                # the heated Arepo state is still a valid simulation state
                @test ArepoLib.run_step!(a.handle) in (:continue, :done, :interrupted)
                u2 = ArepoLib.get_cell_field(a.handle, :utherm)
                @test all(isfinite, u2) && all(u2 .> 0)
                @info "flagship 3 (Moray inside Arepo)" cells = ng heated = count(>(0), gamma_heat) max_xHII = maximum(r.fields.xHII)

            finally
                r.free()
            end
        finally
            a.free()
        end
    end
end

@testset "Phase 5: the EXACT exchange — Arepo Voronoi geometry through R3D" begin
    arepo_dir = ArepoLib.available() ? normpath(dirname(ArepoLib.libpath())) : ""
    lib3d = joinpath(arepo_dir, "libarepo3d.dylib")
    py = ArepoLib.available() ? MultiCode._arepo_python(arepo_dir) : nothing
    if !(py !== nothing && isfile(lib3d))
        @warn "exact-exchange gate skipped (needs libarepo3d: make shared CONFIG=Config_3d.sh BUILD_DIR=build3d LIBRARY=arepo3d)"
        @test_skip false
    else
        # A genuinely 3-D Voronoi donor: the noh_3d example, INIT ONLY (the
        # tessellation is live right after init; a completed run! frees it),
        # in a worker whose env selects the 3-D library flavor — the same
        # wrapper, a different dylib, zero code changes (LazyLib env override).
        example = joinpath(arepo_dir, "examples", "noh_3d")
        dir = mktempdir()
        cp(joinpath(example, "param.txt"), joinpath(dir, "param.txt"))
        mkpath(joinpath(dir, "output"))
        run(pipeline(`$py $(joinpath(example, "create.py")) $dir`; stdout = devnull))
        shm = tempname()
        wfile = joinpath(dir, "arepo3d_worker.jl")
        write(wfile, "using ArepoLib; ArepoLib.serve(; shm = ARGS[1])\n")
        jl = Base.julia_cmd()
        cmd = addenv(setenv(`$jl --startup-file=no --project=$(pkgdir(ArepoLib)) $wfile $shm`;
                            dir = dir), "AREPO_LIB" => lib3d)
        CodeBridge.connect_worker!(ArepoLib.BRIDGE, cmd; shm = shm)
        try
            @test ArepoLib.precision_bytes().ndim == 3        # the 3-D flavor is live
            h = cd(() -> ArepoLib.init("param.txt"), dir)
            ng = ArepoLib.info(h).numgas
            cs = arepo_extract(h; boxlen = ArepoLib.box_size(h))
            geo = ArepoLib.get_voronoi_3d(h)
            @test length(geo.nv) > 3 * ng                 # cells have many faces in 3-D
            @test size(geo.verts, 1) == sum(geo.nv)
            @test all(>=(3), geo.nv)                      # rings are polygons
            # deposit_exact's INTERNAL assertions are the hard gates: each
            # cell's clipped volume equals its SphP volume, and the total
            # mass is conserved — exact geometry, not interpolation.
            ge = deposit_exact(cs, geo, 32)
            volscale = cs.units.volume / cs.units.length^3
            @test abs(volscale - 1.0) < 1e-12             # cubic domain
            @test abs(sum(ge.vol) / 32^3 - 1.0) < 1e-9    # the box is tiled exactly
            gc = deposit_to_grid(cs, 32; method = :cic)
            @test abs(sum(ge.rho) - sum(gc.rho)) / sum(gc.rho) < 1e-9   # same totals
            @info "exact R3D exchange" cells = ng faces = length(geo.nv) verts = size(geo.verts, 1)
            ArepoLib.finalize(h)
        finally
            CodeBridge.disconnect_worker!(ArepoLib.BRIDGE)
            rm(shm; force = true)
        end
    end
end
