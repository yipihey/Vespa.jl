# ── the per-level AMR fast path (ADR-0006 "Next"): ghosts + flux registers ───
#
# The same live two-level Sedov hierarchy as the composite gate, advanced by
# `ramses_ppmk_hydro_step_amr_fast!` — each level on its own bounding-box
# raster with coarse-injected ghosts, conservation restored by flux registers.
# Gates:
#   - ONE isolated step on a fresh refined hierarchy conserves the composite
#     mass and energy to round-off (the register telescoping, no regrid churn);
#   - the full run (host regridding every 4 steps) conserves mass and keeps
#     coarse ≡ restricted fine (the upload contract);
#   - the blast radius matches the composite path on the SAME problem (the
#     physics agreement gate — the paths differ in coarse-region resolution
#     and boundary coupling, so this is % not bits);
#   - the wall-clock and cell-update economics are reported (the point of the
#     fast path: the fine raster covers the refined region, not the domain).

using Test
using Printf
using MultiCode
using RamsesLib

haskey(ENV, "RAMSES_LIB") || (ENV["RAMSES_LIB"] =
    normpath(joinpath(@__DIR__, "..", "..", "..", "..", "..",
                      "mini-ramses", "bin64h", "libramses3d.dylib")))

const FP_LEVMIN = 5
const FP_LEVMAX = 6

"Build the refined Sedov hierarchy (bomb on the coarse level, host flag/refine)."
function _fp_setup(spec)
    n_fine = 2^FP_LEVMAX
    bomb = sedov_bomb(spec, n_fine)
    dir = mktempdir()
    nml = MultiCode._ramses_uniform_namelist(spec; level = FP_LEVMIN)
    nml = replace(nml, "levelmax=$(FP_LEVMIN)" => "levelmax=$(FP_LEVMAX)")
    nml = replace(nml, "interpol_type=0" => "interpol_type=0\n    err_grad_p=0.2")
    write(joinpath(dir, "sedov_amr.nml"), nml)
    h = cd(dir) do
        h = RamsesLib.init("sedov_amr.nml")
        ck, _ = RamsesLib.get_hydro_all(h, :uold, FP_LEVMIN)
        noct = size(ck, 1)
        scale = 2^(FP_LEVMAX - FP_LEVMIN)
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
        RamsesLib.set_hydro!(h, :uold, 5, FP_LEVMIN, ck, Enew)
        RamsesLib.flag_fine!(h, FP_LEVMIN, 1)
        RamsesLib.refine_fine!(h, FP_LEVMIN)
        h
    end
    return (h = h, bomb = bomb, dir = dir)
end

"Composite leaf integrals (mass, total energy) — the conservation ledger."
function _fp_ledger(h)
    r = ramses_composite_raster(h; levmin = FP_LEVMIN, levmax = FP_LEVMAX)
    nx = r.dims[1]; ng = r.ng; n1d = r.n1d
    act = (ng + 1):(nx - ng)
    rho = reshape(r.D, nx, nx, nx)[act, act, act]
    te = reshape(r.Tau, nx, nx, nx)[act, act, act]
    return (mass = sum(rho) / n1d^3, energy = sum(te) / n1d^3, rho = rho, n1d = n1d)
end

"Advance the hierarchy to spec.t with the chosen stepper, host regridding every 4 steps."
function _fp_run(h, spec; fast::Bool)
    t = 0.0; steps = 0; nfine = RamsesLib.level_noct(h, FP_LEVMAX)
    seconds = @elapsed while t < spec.t * (1 - 1e-12)
        steps < 2000 || error("fast-path run did not reach t=$(spec.t) (t=$t)")
        dt = fast ?
            ramses_ppmk_hydro_step_amr_fast!(h; levmin = FP_LEVMIN, levmax = FP_LEVMAX,
                                             gamma = spec.gamma, boxlen = 1.0,
                                             dt_max = spec.t - t) :
            ramses_ppmk_hydro_step_amr!(h; levmin = FP_LEVMIN, levmax = FP_LEVMAX,
                                        gamma = spec.gamma, boxlen = 1.0,
                                        dt_max = spec.t - t)
        t += dt; steps += 1
        if steps % 4 == 0
            RamsesLib.flag_fine!(h, FP_LEVMIN, 1)
            RamsesLib.refine_fine!(h, FP_LEVMIN)
            nfine = max(nfine, RamsesLib.level_noct(h, FP_LEVMAX))
        end
    end
    return (t = t, steps = steps, nfine = nfine, seconds = seconds)
end

"Shock radius from a composite density cube."
function _fp_radius(rho, n1d)
    cs = CellSet(:composite,
                 hcat([(c[1] - 0.5) / n1d for c in vec(CartesianIndices(rho))],
                      [(c[2] - 0.5) / n1d for c in vec(CartesianIndices(rho))],
                      [(c[3] - 0.5) / n1d for c in vec(CartesianIndices(rho))]),
                 fill(1.0 / n1d^3, n1d^3),
                 vec(rho), zeros(n1d^3, 3), zeros(n1d^3),
                 (length = 1.0, time = 1.0, density = 1.0), (;))
    return sedov_profile(cs).R_shock
end

@testset "the per-level AMR fast path (flux registers)" begin
    if !RamsesLib.available()
        @test_skip false
    else
        spec = SedovCompareSpec(t = 0.02)

        # ── gate 1: one isolated step conserves to round-off ─────────────────
        s1 = _fp_setup(spec)
        try
            @test RamsesLib.level_noct(s1.h, FP_LEVMAX) > 0
            before = _fp_ledger(s1.h)
            ramses_ppmk_hydro_step_amr_fast!(s1.h; levmin = FP_LEVMIN, levmax = FP_LEVMAX,
                                             gamma = spec.gamma, boxlen = 1.0)
            after = _fp_ledger(s1.h)
            dm = abs(after.mass - before.mass) / before.mass
            de = abs(after.energy - before.energy) / before.energy
            @test dm < 1e-12                       # the registers telescope exactly
            @test de < 1e-12
            @info "fast path: one-step conservation" dmass = dm denergy = de
        finally
            RamsesLib.finalize(s1.h)
        end

        # ── gates 2–4: the full run vs the composite path (SEQUENTIAL — the
        # capi session is a per-process global, never two live handles) ───────
        sf = _fp_setup(spec)
        rf, lf, upload_err = try
            rf = _fp_run(sf.h, spec; fast = true)
            lf = _fp_ledger(sf.h)
            # the upload contract: refined coarse ≡ restriction of fine
            ckc, Uc = RamsesLib.get_hydro_all(sf.h, :uold, FP_LEVMIN)
            masks = MultiCode._ramses_leafmask(
                [ckc, RamsesLib.get_hydro_all(sf.h, :uold, FP_LEVMAX)[1]])
            ue = 0.0
            for o in 1:size(ckc, 1), c in 1:8
                masks[1][o, c] && continue
                i0 = (2 * ckc[o, 1] + ((c - 1) & 1)) * 2
                j0 = (2 * ckc[o, 2] + ((c - 1) >> 1 & 1)) * 2
                k0 = (2 * ckc[o, 3] + ((c - 1) >> 2 & 1)) * 2
                s = 0.0
                for dk in 1:2, dj in 1:2, di in 1:2
                    s += lf.rho[i0 + di, j0 + dj, k0 + dk]
                end
                ue = max(ue, abs(Uc[o, c, 1] - s / 8))
            end
            (rf, lf, ue)
        finally
            RamsesLib.finalize(sf.h)
        end
        sc = _fp_setup(spec)
        rc, lc = try
            rc = _fp_run(sc.h, spec; fast = false)
            (rc, _fp_ledger(sc.h))
        finally
            RamsesLib.finalize(sc.h)
        end
        @test rf.nfine > 0 && rf.nfine < (2^(FP_LEVMAX - 1))^3    # genuinely multi-level
        @test abs(lf.mass - spec.rho0) / spec.rho0 < 1e-10        # conservation, full run
        @test upload_err < 1e-12
        # physics agreement with the composite path
        Rf = _fp_radius(lf.rho, lf.n1d)
        Rc = _fp_radius(lc.rho, lc.n1d)
        @test abs(Rf - Rc) / Rc < 0.03
        speedup = rc.seconds / rf.seconds
        @info "fast path vs composite" R_fast = Rf R_composite = Rc steps_fast = rf.steps steps_composite = rc.steps s_fast = round(rf.seconds; digits = 2) s_composite = round(rc.seconds; digits = 2) speedup = round(speedup; digits = 2) fine_octs = rf.nfine upload_err = upload_err
    end
end
