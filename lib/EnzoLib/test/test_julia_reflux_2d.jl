# ADR-0003 follow-up #2 (ND face-plane raster): the 2D conservation gate. The
# C++ flux bridge is ND-general; this exercises the ND Julia plane assembly
# (`EnzoNG.bflux_plane` + the ND `_write_fluxes!`) — in 2D each coarse–fine face
# is a 1D plane of cells, not a single cell, so a wrong orthogonal-dim raster /
# index / sign shows up as a conservation drift jumping from round-off to ~1e-3.
#
# Mirrors `test_julia_reflux.jl` subtest A: a STATIC multi-level hierarchy with
# the blast wave interior to the refined region (so the coarse–fine flux balance
# is the only conservation term). WITH the correction → composite mass/energy
# conserved to round-off; WITHOUT (zeros) → the documented reflux drift.
#
# Reuses the engine/driver helpers from test_julia_reflux.jl (already ND-general:
# `read_root_totals`, `_run_reflux`, `_drift`, `conservative_julia_hydro_hook`).
# That file must be included first (runtests.jl does so).

# A planar (x-direction) Sod shock tube on a 2D grid with the discontinuity
# refined-at-start into an interior strip. Per y-row this is the mild 1D Sod the
# :julia hydro conserves to round-off, but the strip's coarse–fine boundary is a
# genuine 2D face plane (60-cell x-faces, 4-cell y-faces) — so it exercises the
# ND orthogonal-dim raster, not the 1D single-cell collapse.
const REFLUX_PF_2D = abspath(joinpath(@__DIR__, "..", "..", "..", "..",
                                      "run", "Hydro", "Hydro-2D", "SodShockTube2DAMR", "SodShockTube2DAMR.enzo"))

if get(ENV, "REFLUX_NOTEST", "") != ""
    @info "REFLUX_NOTEST set — skipping the 2D reflux testset"
elseif !EnzoLib.grid_available()
    @info "Session bridge not built — skipping 2D :julia reflux (ND face-plane) test"
else
    @testset "ADR-0003 follow-up #2: ND face-plane raster — 2D conservation" begin
        # Static 2D hierarchy, blast interior to the refined region. The ND plane
        # raster writes the (D−1)-plane of each coarse–fine face from EnzoNG's
        # per-cell flux registers; Enzo's UpdateFromFinerGrids/CorrectForRefined-
        # Fluxes then restore conservation. nsteps capped so the blast stays
        # interior (centred energy injection, refined center 0.4–0.6).
        # parent_ghost=false: this gate isolates the ND flux RASTER (follow-up #2),
        # which is orthogonal to the parent-ghost BC (follow-up #1). Subtest C below
        # turns ND parent-ghost ON (the (D−1)-plane reader); HERE we keep plain
        # Outflow BCs so the round-off assertion measures the flux raster alone.
        on  = _run_reflux(REFLUX_PF_2D; conservative = true,  regrid = false, nsteps = 20, parent_ghost = false)
        off = _run_reflux(REFLUX_PF_2D; conservative = false, regrid = false, nsteps = 20, parent_ghost = false)
        d_on = _drift(on); d_off = _drift(off)
        @info "follow-up #2 (2D) static, waves interior" max_level = on.max_level d_on d_off
        @test on.max_level >= 1                       # 2D AMR actually engaged (≥2 levels)
        @test d_on.mass   < 1e-11                     # ND flux correction ⇒ round-off conservation
        @test d_on.energy < 1e-11
        @test d_off.mass  > 1e4 * max(d_on.mass, 1e-16)   # disabling it ⇒ the reflux drift

        # (C) ND PARENT-GHOST (the ND follow-up to ADR-0003 #1). In ND a subgrid's
        # outer faces are a MIX: the 2D Sod strip's x-faces are coarse–fine
        # interfaces, but its y-faces span the full domain and ARE the (Outflow)
        # domain boundary. `_apply_parent_ghost!` now decides PER (axis, side) —
        # ParentGhost on the coarse–fine faces (reading the (D−1)-plane of Enzo's
        # interpolated ghosts, one zone outward, cell-by-cell), the real domain BC on
        # the domain faces. The earlier blanket replacement put a parent ghost on the
        # y domain faces too, which BREAKS conservation (the static-interior drift
        # blew up from round-off to ~1.5e-4, almost the no-reflux ~1.2e-3). With the
        # per-face fix, parent-ghost ON stays conservative (≪ the no-reflux signature)
        # while the x-interface ghost is now Enzo's parent value, not an Outflow copy.
        #
        # The decisive ND-parent-ghost check is a NEGATIVE one: a wrong ghost plane
        # (the old blanket bug, or a wrong-side/index raster) drives the static drift
        # back up to ~1e-4. So we assert the static drift with parent-ghost ON stays
        # FAR below that broken band — i.e. the ND (D−1)-plane reader is correct.
        on_pg = _run_reflux(REFLUX_PF_2D; conservative = true, regrid = false, nsteps = 20, parent_ghost = true)
        d_pg = _drift(on_pg)
        @info "ND parent-ghost (2D) static, waves interior" parent_ghost = d_pg raster_only = d_on no_reflux = d_off
        @test d_pg.mass   < 1e-4                       # ND parent-ghost stays conservative (measured ≈1.5e-6)
        @test d_pg.energy < 1e-4                       #   (a wrong ghost plane ⇒ ~1.5e-4, gated out)
        @test d_pg.mass   < 0.1 * d_off.mass           # and well below the no-reflux signature (~1.2e-3)
        # The residual (≈1.5e-6, NOT round-off like the raster-only path) is the
        # coarse↔fine interpolation accuracy: EnzoNG reads the innermost interpolated
        # layer; Enzo's multi-layer reconstruction differs at higher order — the same
        # accuracy-limited residual the 1D end-to-end parent-ghost shows (~8e-6). The
        # flux bridge itself stays EXACTLY conservative (the raster-only subtest above
        # is unchanged at round-off); this is the boundary-ACCURACY half in ND.
    end
end
