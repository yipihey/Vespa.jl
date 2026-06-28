# ADR-0003 part B: conservative `:julia` hydro under Enzo AMR (the SubgridFluxes
# bridge). A :julia hydro slot (Vespa's driver on the live grid) is made
# conservative across coarse–fine boundaries by writing Vespa's recorded face
# fluxes into Enzo's flux registers, so Enzo's own UpdateFromFinerGrids /
# CorrectForRefinedFluxes restore conservation — the same machinery, Vespa's
# numbers. Two flux sets are filled (exactly what SolveHydroEquations fills):
#   • each grid's BoundaryFluxes  = the RefinedFluxes a finer grid carried, and
#   • the parent's SubgridFluxesEstimate[level][i][sub] = the coarse InitialFluxes.
#
# The DECISIVE gate (per the ADR): a refined Sod whose waves stay interior (no
# boundary outflow) conserves total mass/energy to ~round-off WITH the flux
# correction, and drifts to ~1e-3 (the documented reflux signature) WITHOUT it.
# `test_reflux.jl` (Vespa's native composite reflux) is the template.
#
# The reflux harness (helpers, steppers, `_run_reflux`, `read_root_totals`,
# `conservative_julia_hydro_hook`, `_ENZO_DEV`) lives in `reflux_common.jl`, shared with
# the 2D gate (test_julia_reflux_2d.jl) and the GPU/CPU PPMKernels gate (test_gpu_reflux.jl).
#
# Guarded on grid_available() (needs the Session bridge library).

include("reflux_common.jl")

const REFLUX_PF = abspath(joinpath(_ENZO_DEV,
                                   "run", "Hydro", "Hydro-1D", "SodShockTube", "SodShockTubeAMR.enzo"))

if get(ENV, "REFLUX_NOTEST", "") != ""
    @info "REFLUX_NOTEST set — defining helpers only, skipping the testset"
elseif !EnzoLib.grid_available()
    @info "Session bridge not built — skipping :julia reflux (ADR-0003 part B) test"
else
    @testset "ADR-0003 part B: conservative :julia hydro under AMR (SubgridFluxes bridge)" begin
        # (A) THE FLUX BRIDGE IS EXACTLY CONSERVATIVE. On a static multi-level
        # hierarchy, while the waves are still interior to the refined region (so the
        # coarse–fine flux balance is the only conservation term), the recorded
        # Vespa fluxes written into Enzo's registers conserve the composite mass/
        # energy to ROUND-OFF — and disabling the correction (zeros) drifts to ~1e-4,
        # the documented reflux signature. This is the decisive part-B gate
        # (test_reflux.jl is the template): a wrong index/sign/unit shows here.
        on  = _run_reflux(REFLUX_PF; conservative = true,  regrid = false, nsteps = 25)
        off = _run_reflux(REFLUX_PF; conservative = false, regrid = false, nsteps = 25)
        d_on = _drift(on); d_off = _drift(off)
        @info "part B (A) static, waves interior" max_level = on.max_level d_on d_off
        @test on.max_level >= 1                      # AMR actually engaged (≥2 levels)
        @test d_on.mass   < 1e-11                    # flux correction ⇒ conserved to round-off
        @test d_on.energy < 1e-11
        @test d_off.mass  > 1e4 * max(d_on.mass, 1e-16)   # disabling it ⇒ ~1e-4 drift

        # (B) END-TO-END FEATURE-TRACKING AMR. The full run (dynamic regridding to
        # StopTime) with the correction conserves far better than without — the
        # reflux removes the bulk of the coarse–fine non-conservation.
        on2  = _run_reflux(REFLUX_PF; conservative = true)
        off2 = _run_reflux(REFLUX_PF; conservative = false)
        e_on = _drift(on2); e_off = _drift(off2)
        @info "part B (B) full regrid run" cycles = on2.cycles d_on = e_on d_off = e_off
        @test e_on.mass < 1e-3                        # conserves well end-to-end
        @test e_off.mass > 50 * e_on.mass             # reflux is decisive

        # (C) PARENT-GHOST COUPLING (ADR-0003 follow-up #1). Vespa now consumes
        # Enzo's parent-interpolated ghost zones at a subgrid's coarse–fine faces
        # (a ParentGhost BC reading the live grid's ghosts) instead of an Outflow
        # (zero-gradient) copy. The Outflow ghost is the residual end-to-end drift:
        # when a wave sits ON a coarse–fine boundary its rel-error × flux-magnitude
        # is the dominant non-conservation. Reading the parent value instead roughly
        # HALVES the end-to-end drift (measured ~1.79e-5 Outflow → ~8.05e-6 parent-
        # ghost). The sign/index is gated honestly: the negative control (reading the
        # ghost on the WRONG side) makes it WORSE than Outflow, so a wrong index/sign
        # FAILS this assertion rather than passing silently. (The flux bridge stays
        # exactly conservative — subtest A is unchanged at round-off — this is the
        # boundary-ACCURACY half.)
        pg_on  = on2                                  # the default run already uses parent-ghost
        pg_off = _run_reflux(REFLUX_PF; conservative = true, parent_ghost = false)
        e_pg  = _drift(pg_on); e_npg = _drift(pg_off)
        @info "part B (C) parent-ghost vs Outflow" parent_ghost = e_pg outflow = e_npg ratio = e_npg.mass / e_pg.mass
        @test e_npg.mass > 1.5 * e_pg.mass            # parent-ghost cuts the drift (≥1.5×; measured ≈2.2×)
        @test e_pg.mass < 1.2e-5                      # absolute end-to-end bound (measured ≈8.05e-6)
        @test e_pg.energy < 1.5e-5                    # energy likewise (measured ≈9.13e-6)
    end
end
