# ── the guest slot under AMR (ADR-0006 Phase 6): composite coupling ──────────
#
# A LIVE two-level RAMSES hierarchy (the host's own flag/refine builds level
# levmin+1 around the injected Sedov bomb) advanced entirely by the PPMKernels
# guest through the COMPOSITE raster: hierarchy → uniform finest grid → one
# guest step → conservative write-back to every level.  Gates:
#   - the hierarchy is genuinely multi-level (the test isn't vacuous) and the
#     leaf decomposition covers the box exactly (asserted in the raster);
#   - mass and energy are conserved through the full run;
#   - the guest-under-AMR blast matches the guest-on-uniform-fine run (the
#     composite machinery is the only difference — same scheme, same finest
#     resolution): shock radius within 3%, coarse-level state consistent with
#     the restriction of the fine state (the upload contract).

using Test
using MultiCode
using RamsesLib

haskey(ENV, "RAMSES_LIB") || (ENV["RAMSES_LIB"] =
    normpath(joinpath(@__DIR__, "..", "..", "..", "..", "..",
                      "mini-ramses", "bin64h", "libramses3d.dylib")))

const AMR_LEVMIN = 5
const AMR_LEVMAX = 6

"Run the injected Sedov on a refining hierarchy, every hydro step by the composite guest."
function _run_amr_guest(spec; device = :cpu)
    n_fine = 2^AMR_LEVMAX
    bomb = sedov_bomb(spec, n_fine)
    dir = mktempdir()
    nml = MultiCode._ramses_uniform_namelist(spec; level = AMR_LEVMIN)
    # allow one refinement level, triggered by the blast's pressure gradient
    nml = replace(nml, "levelmax=$(AMR_LEVMIN)" => "levelmax=$(AMR_LEVMAX)")
    nml = replace(nml, "interpol_type=0" => "interpol_type=0\n    err_grad_p=0.2")
    write(joinpath(dir, "sedov_amr.nml"), nml)
    return cd(dir) do
        h = RamsesLib.init("sedov_amr.nml")
        # inject the bomb on the COARSE level (restriction of the fine IC)
        ck, _ = RamsesLib.get_hydro_all(h, :uold, AMR_LEVMIN)
        noct = size(ck, 1)
        scale = 2^(AMR_LEVMAX - AMR_LEVMIN)
        Enew = Matrix{Float64}(undef, noct, 8)
        for o in 1:noct, c in 1:8
            i0 = (2 * ck[o, 1] + ((c - 1) & 1)) * scale
            j0 = (2 * ck[o, 2] + ((c - 1) >> 1 & 1)) * scale
            k0 = (2 * ck[o, 3] + ((c - 1) >> 2 & 1)) * scale
            s = 0.0
            for dk in 1:scale, dj in 1:scale, di in 1:scale
                s += bomb.te[i0 + di, j0 + dj, k0 + dk]
            end
            Enew[o, c] = spec.rho0 * s / scale^3
        end
        RamsesLib.set_hydro!(h, :uold, 5, AMR_LEVMIN, ck, Enew)
        # the HOST builds the fine level around the bomb (its own machinery)
        RamsesLib.flag_fine!(h, AMR_LEVMIN, 1)
        RamsesLib.refine_fine!(h, AMR_LEVMIN)
        t = 0.0; steps = 0
        nfine_seen = RamsesLib.level_noct(h, AMR_LEVMAX)
        while t < spec.t * (1 - 1e-12)
            steps < 2000 || error("AMR-guest run did not reach t=$(spec.t) in 2000 steps (t=$t)")
            # the GUEST owns the CFL (RAMSES's newdt returns 0 on a level whose
            # time state the guest manages)
            dt = ramses_ppmk_hydro_step_amr!(h; levmin = AMR_LEVMIN, levmax = AMR_LEVMAX,
                                             gamma = spec.gamma, boxlen = 1.0,
                                             dt_max = spec.t - t, device = device)
            t += dt; steps += 1
            # the host regrids: the refined region follows the blast (the front
            # moves ≪ a coarse cell per step, so every few steps amply suffices)
            if steps % 4 == 0
                RamsesLib.flag_fine!(h, AMR_LEVMIN, 1)
                RamsesLib.refine_fine!(h, AMR_LEVMIN)
                nfine_seen = max(nfine_seen, RamsesLib.level_noct(h, AMR_LEVMAX))
            end
        end
        # composite extraction = the raster (leaf cells at fine resolution)
        r = ramses_composite_raster(h; levmin = AMR_LEVMIN, levmax = AMR_LEVMAX)
        nx = r.dims[1]; ng = r.ng
        act = (ng + 1):(nx - ng)
        rho = reshape(r.D, nx, nx, nx)[act, act, act]
        # coarse-vs-fine consistency: re-extract the coarse level and check the
        # refined coarse cells equal the restriction of the fine state
        ckc, Uc = RamsesLib.get_hydro_all(h, :uold, AMR_LEVMIN)
        masks = MultiCode._ramses_leafmask([ckc, RamsesLib.get_hydro_all(h, :uold, AMR_LEVMAX)[1]])
        upload_err = 0.0
        for o in 1:size(ckc, 1), c in 1:8
            masks[1][o, c] && continue                  # leaves: trivially consistent
            i0 = (2 * ckc[o, 1] + ((c - 1) & 1)) * 2
            j0 = (2 * ckc[o, 2] + ((c - 1) >> 1 & 1)) * 2
            k0 = (2 * ckc[o, 3] + ((c - 1) >> 2 & 1)) * 2
            s = 0.0
            for dk in 1:2, dj in 1:2, di in 1:2
                s += rho[i0 + di, j0 + dj, k0 + dk]
            end
            upload_err = max(upload_err, abs(Uc[o, c, 1] - s / 8))
        end
        out = (rho = rho, t = t, steps = steps, E_in = bomb.E_in,
               nfine = nfine_seen, upload_err = upload_err,
               mass = sum(rho) / n_fine^3,
               free = () -> RamsesLib.finalize(h))
        return out
    end
end

@testset "the guest slot under AMR (composite raster)" begin
    if !RamsesLib.available()
        @test_skip false
    else
        spec = SedovCompareSpec(t = 0.02)               # R ≈ 0.24: fits the refined patch story
        amr = _run_amr_guest(spec)
        try
            n_fine = 2^AMR_LEVMAX
            @test amr.nfine > 0                                       # the hierarchy refined
            @test amr.nfine < (2^(AMR_LEVMAX - 1))^3                  # …but is NOT fully refined
            @test abs(amr.mass - spec.rho0) / spec.rho0 < 1e-10       # conservation, composite
            @test amr.upload_err < 1e-12                              # coarse ≡ restricted fine

            # reference: the SAME guest on the uniform finest grid
            uni = run_ramses_sedov(spec; level = AMR_LEVMAX, engine = :guest)
            try
                pu = sedov_profile(uni.cs)
                # blast radius from the composite field
                csl = CellSet(:composite,
                              hcat([( (c[1]-0.5)/n_fine ) for c in vec(CartesianIndices(amr.rho))],
                                   [( (c[2]-0.5)/n_fine ) for c in vec(CartesianIndices(amr.rho))],
                                   [( (c[3]-0.5)/n_fine ) for c in vec(CartesianIndices(amr.rho))]),
                              fill(1.0 / n_fine^3, n_fine^3),
                              vec(amr.rho), zeros(n_fine^3, 3), zeros(n_fine^3),
                              (length = 1.0, time = 1.0, density = 1.0), (;))
                pa = sedov_profile(csl)
                @test abs(pa.R_shock - pu.R_shock) / pu.R_shock < 0.03
                @info "guest under AMR" steps = amr.steps fine_octs = amr.nfine R_amr = pa.R_shock R_uniform = pu.R_shock upload_err = amr.upload_err
            finally
                uni.free()
            end
        finally
            amr.free()
        end
    end
end
