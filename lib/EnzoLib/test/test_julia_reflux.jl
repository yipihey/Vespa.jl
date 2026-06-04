# ADR-0003 part B: conservative `:julia` hydro under Enzo AMR (the SubgridFluxes
# bridge). A :julia hydro slot (EnzoNG's driver on the live grid) is made
# conservative across coarse–fine boundaries by writing EnzoNG's recorded face
# fluxes into Enzo's flux registers, so Enzo's own UpdateFromFinerGrids /
# CorrectForRefinedFluxes restore conservation — the same machinery, EnzoNG's
# numbers. Two flux sets are filled (exactly what SolveHydroEquations fills):
#   • each grid's BoundaryFluxes  = the RefinedFluxes a finer grid carried, and
#   • the parent's SubgridFluxesEstimate[level][i][sub] = the coarse InitialFluxes.
#
# The DECISIVE gate (per the ADR): a refined Sod whose waves stay interior (no
# boundary outflow) conserves total mass/energy to ~round-off WITH the flux
# correction, and drifts to ~1e-3 (the documented reflux signature) WITHOUT it.
# `test_reflux.jl` (EnzoNG's native composite reflux) is the template.
#
# Guarded on grid_available() (needs the Session bridge library).

using EnzoBackend
import MeshInterface

# EnzoNG conserved component for Enzo BaryonField `fld` (the inverse of the mesh's
# conserved-role → Enzo-field map), or -1 if `fld` is not a hydro conserved field
# (e.g. a colour/species the EnzoNG model does not carry — its flux stays 0).
@inline function _engng_comp_of(mesh::EnzoGridMesh, fld::Int)
    fld == mesh.di && return mesh.cdi
    fld == mesh.ei && return mesh.cei
    for d in 1:3
        fld == mesh.vi[d] && return mesh.cmom[d]
    end
    return -1
end

# Build a fresh EnzoGridMesh + Simulation for grid `gi` (each AMR grid has its own
# level-dependent cell width, so the physical edges come from the live grid).
function _build_grid_sim(h, gi, model, nghost)
    l, r = EnzoLib.problem_grid_edge(h, gi)
    rank = EnzoLib.problem_grid_rank(h, gi)
    domain = ntuple(d -> (l[d], r[d]), rank)
    mesh = EnzoGridMesh(h; grid = gi, nghost = nghost, domain = domain,
                        cons_density = density_index(model),
                        cons_momentum = momentum_indices(model),
                        cons_energy = energy_index(model))
    dims = mesh.active
    prob = Problem(; name = "cons-amr", dims = dims, domain = domain, γ = model.γ,
                   bcs = Outflow(), tfinal = 1.0, cfl = 0.4,
                   init = (x, y, z) -> (1.0, 0.0, 0.0, 0.0, 1.0))  # overwritten by sync
    return mesh, Simulation(mesh, prob; model = model)
end

# Write one (dim, side) flux plane of subgrid entry `sub` for ALL Enzo baryon
# fields (the consumers — CorrectForRefinedFluxes, AddToBoundaryFluxes — loop
# every field and deref each, so unmapped fields must still be allocated/zeroed).
# `F` is EnzoNG's flux NTuple for that face (or nothing ⇒ all zeros). 1D: 1-cell.
function _write_subgrid_plane!(h, level, i, sub, mesh, nf, dim, side, F, Vcell)
    for fld in 0:nf-1
        k = _engng_comp_of(mesh, fld)
        val = (F !== nothing && k > 0) ? F[k] / Vcell : 0.0
        EnzoLib.problem_set_subgrid_flux(h, level, i, sub, fld, dim, side, [val])
    end
    return nothing
end

# Write EnzoNG's recorded fluxes for grid (level, i, flat gi) into Enzo's flux
# registers (the SubgridFluxesEstimate the AMR machinery consumes):
#   • proper subgrids sub=0..nsub-2  → the coarse InitialFluxes at the subgrid's
#     coarse–fine boundary faces (EnzoNG's INTERIOR flux there), and
#   • the last entry sub=nsub-1      → the grid's own outer-boundary flux; Enzo's
#     FinalizeFluxes accumulates THIS into the grid's BoundaryFluxes (the
#     RefinedFluxes its parent projects), giving the correct temporal accumulation
#     across subcycles for free.
# `conservative=false` writes ZEROS (arrays still allocated, but zero correction) —
# the non-conservative baseline that isolates the reflux effect. 1D for now (the
# face planes collapse to single cells); ND plane assembly is a follow-up.
function _write_fluxes!(h, level, i, gi, mesh::EnzoGridMesh{1}, sim, breg, model; conservative::Bool)
    Vcell = MeshInterface.cell_volume(mesh, CartesianIndex(1))
    g0 = EnzoLib.problem_grid_global_start(h, gi)
    nf = EnzoLib.problem_num_fields(h, gi)
    nsub = EnzoLib.problem_num_subgrids(h, level, i)
    dim = 0; axis = 1                                   # 1D

    # ── proper subgrids: coarse InitialFluxes at the coarse–fine interface faces.
    for sub in 0:nsub-2
        ls, _ = EnzoLib.problem_subgrid_flux_extent(h, level, i, sub, dim, 0)  # Left face coarse global
        rs, _ = EnzoLib.problem_subgrid_flux_extent(h, level, i, sub, dim, 1)  # Right face coarse global
        lo_left  = ls[dim+1] - g0[dim+1]                # EnzoNG interior key lo cell (1-based active)
        lo_right = rs[dim+1] - g0[dim+1] + 1
        FL = conservative ? get(breg.interior, (axis, CartesianIndex(lo_left)), nothing) : nothing
        FR = conservative ? get(breg.interior, (axis, CartesianIndex(lo_right)), nothing) : nothing
        _write_subgrid_plane!(h, level, i, sub, mesh, nf, dim, 0, FL, Vcell)
        _write_subgrid_plane!(h, level, i, sub, mesh, nf, dim, 1, FR, Vcell)
    end

    # ── own-boundary entry (last): the grid's outer-face flux (breg.flux :lo/:hi).
    flo = nothing; fhi = nothing
    for ((ax, side, cell), F) in breg.flux
        side === :lo ? (flo = F) : (fhi = F)
    end
    own = nsub - 1
    _write_subgrid_plane!(h, level, i, own, mesh, nf, dim, 0, conservative ? flo : nothing, Vcell)
    _write_subgrid_plane!(h, level, i, own, mesh, nf, dim, 1, conservative ? fhi : nothing, Vcell)
    return nothing
end

# The conservative :julia hydro hook: per grid on `level`, run EnzoNG's driver on
# the live state and write its fluxes into Enzo's registers.
function conservative_julia_hydro_hook(; γ = 1.4, nghost = 3, conservative::Bool = true)
    model = IdealHydro(γ)
    return function (h, level, dt)
        n = EnzoLib.session_num_grids_on_level(h, level)
        for i in 0:n-1
            gi = EnzoLib.problem_grid_index_on_level(h, level, i)
            mesh, sim = _build_grid_sim(h, gi, model, nghost)
            breg = EnzoNG._bflux_register(sim; record_interior = true)
            sync_from_enzo!(sim.sv, mesh)
            step!(sim, dt; bflux = breg)
            sync_to_enzo!(mesh, sim.sv)
            _write_fluxes!(h, level, i, gi, mesh, sim, breg, model; conservative = conservative)
        end
        return nothing
    end
end

# Total mass + energy over the ACTIVE root grid (= the composite total, since
# update_from_finer projects the fine solution onto the coarse cells each step).
function read_root_totals(h; nghost = 3)
    gi = EnzoLib.problem_grid_index_on_level(h, 0, 0)
    dims = EnzoLib.problem_grid_dims(h, gi)
    l, r = EnzoLib.problem_grid_edge(h, gi)
    rank = EnzoLib.problem_grid_rank(h, gi)
    active = ntuple(d -> dims[d] - 2nghost, rank)
    cw = ntuple(d -> (r[d] - l[d]) / active[d], rank)
    Vcell = prod(cw)
    strides = ntuple(d -> d == 1 ? 1 : prod(ntuple(k -> dims[k], d - 1)), rank)
    di = EnzoLib.field_index(h, 0; grid = gi)   # Density
    ei = EnzoLib.field_index(h, 1; grid = gi)   # TotalEnergy (specific)
    dens = EnzoLib.problem_get_field(h, di, gi)
    espec = EnzoLib.problem_get_field(h, ei, gi)
    mass = 0.0; energy = 0.0
    for I in CartesianIndices(active)
        f = 1 + sum((nghost + I[d] - 1) * strides[d] for d in 1:rank)
        ρ = dens[f]
        mass += ρ * Vcell
        energy += ρ * espec[f] * Vcell
    end
    return (mass = mass, energy = energy)
end

# Drive a 2+-level AMR run from Julia with the conservative :julia hydro slot, and
# return the root-grid composite totals before/after. `regrid=false` holds the
# (multi-level) hierarchy static; `nsteps` caps the number of root steps (so the
# decisive run can stop while the waves are still interior to the refined region,
# where the coarse–fine flux balance is the ONLY conservation term).
function _run_reflux(pf; conservative::Bool, regrid::Bool = true, nsteps::Int = 100000)
    eng = EnzoLib.EngineConfig(; hydro = :julia, reflux = true,
                               hooks = Dict{Symbol,Function}(:hydro =>
                                   conservative_julia_hydro_hook(; conservative = conservative)))
    cd(EnzoLib._workdir(pf)) do
        h = EnzoLib.session_init(pf)
        h == C_NULL && error("session_init failed for $pf")
        try
            EnzoLib.session_rebuild(h, 0)
            t0 = read_root_totals(h)
            n = 0
            while EnzoLib.session_time(h) < EnzoLib.session_stop_time(h) && n < nsteps
                EnzoLib.evolve_level!(h, 0, 0.0; engine = eng, regrid = regrid)
                regrid && EnzoLib.session_rebuild(h, 0)
                n += 1
            end
            mx = maximum(L -> length(EnzoLib.grids_on_level(h, L)) > 0 ? L : 0, 0:8)
            return (t0 = t0, t1 = read_root_totals(h), cycles = n, max_level = mx)
        finally
            EnzoLib.free_problem(h)
        end
    end
end

_drift(r) = (mass = abs(r.t1.mass - r.t0.mass) / r.t0.mass,
             energy = abs(r.t1.energy - r.t0.energy) / r.t0.energy)

const REFLUX_PF = abspath(joinpath(@__DIR__, "..", "..", "..", "..",
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
        # EnzoNG fluxes written into Enzo's registers conserve the composite mass/
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
        # reflux removes the bulk of the coarse–fine non-conservation. (The residual
        # ~1e-5 is EnzoNG's Outflow boundary approximation at coarse–fine faces, an
        # accuracy follow-up — EnzoNG should consume Enzo's parent-interpolated
        # ghost zones; the flux bridge itself is exact, per subtest A.)
        on2  = _run_reflux(REFLUX_PF; conservative = true)
        off2 = _run_reflux(REFLUX_PF; conservative = false)
        e_on = _drift(on2); e_off = _drift(off2)
        @info "part B (B) full regrid run" cycles = on2.cycles d_on = e_on d_off = e_off
        @test e_on.mass < 1e-3                        # conserves well end-to-end
        @test e_off.mass > 50 * e_on.mass             # reflux is decisive
    end
end
