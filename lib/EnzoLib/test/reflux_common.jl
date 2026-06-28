# Reusable conservative-`:julia`-under-AMR reflux harness (ADR-0003 part B), shared by the
# 1D/2D native-driver gates (test_julia_reflux*.jl) and the GPU/CPU PPMKernels gate
# (test_gpu_reflux.jl). It lives in the EnzoLib *test* (integration) layer because it needs
# EnzoLib + EnzoBackend + Vespa together, and EnzoBackend depends on EnzoLib (so this cannot
# live in EnzoLib core without a dependency cycle).
#
# The data flow (unchanged from the proven CPU path):
#   reflux_hydro_hook(stepper) → per grid: build EnzoGridMesh+Simulation, sync state in,
#     `stepper` runs a hydro step that FILLS a BoundaryFluxRegister `breg` and writes the
#     updated state back to Enzo, then `_write_fluxes!` rasterizes `breg` into Enzo's flux
#     registers (Vespa.bflux_plane) → Enzo's UpdateFromFinerGrids/CorrectForRefinedFluxes
#     restore conservation.
# The ONLY thing a backend changes is the `stepper` (how `breg` gets filled):
#   • native_stepper()    — Vespa's ghost-free HLLC+PLM+RK2 driver (`step!`), CPU, proven.
#   • ppmkernels_stepper() — PPMKernels `muscl_hancock_step_3d!` (KA; CPU or GPU) + the
#     `frec → register` capture (PPMKernels.boundary_flux_register). Identical downstream.

using EnzoBackend
using Vespa
import MeshInterface
import PPMKernels

# Vespa conserved component for Enzo BaryonField `fld` (the inverse of the mesh's
# conserved-role → Enzo-field map), or -1 if `fld` is not a hydro conserved field.
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

# Per-(axis,side) test: is grid `gi`'s face a real domain boundary (its physical edge
# coincides with the domain edge) rather than a coarse–fine interface?
function _is_domain_face(h, gi, dl, dr, axis::Int, side::Symbol)
    l, r = EnzoLib.problem_grid_edge(h, gi)
    span = dr[axis] - dl[axis]
    tol = 1e-9 * (span == 0 ? 1.0 : abs(span))
    return side === :lo ? abs(l[axis] - dl[axis]) <= tol : abs(r[axis] - dr[axis]) <= tol
end

# Consume Enzo's parent-interpolated ghost zones at a subgrid's coarse–fine faces (ADR-0003
# follow-up #1) instead of an Outflow copy; the original domain BC is kept on real domain
# faces (decided per axis/side). Returns the ParentGhost BC (or nothing on level 0).
function _apply_parent_ghost!(sim, mesh, model, level)
    level <= 0 && return nothing
    h = mesh.h; gi = mesh.grid
    R = MeshInterface.rank(mesh)
    g0 = EnzoLib.problem_grid_index_on_level(h, 0, 0)
    dl, dr = EnzoLib.problem_grid_edge(h, g0)
    cons = enzo_parent_ghost(mesh)
    pg = MeshInterface.ParentGhost((axis, side, cell) ->
             Vespa.cons2prim(model, cons(axis, side, cell)))
    orig = sim.bcs
    pairs = ntuple(R) do axis
        lo = _is_domain_face(h, gi, dl, dr, axis, :lo) ?
                 MeshInterface.bc_on(orig, axis, :lo) : pg
        hi = _is_domain_face(h, gi, dl, dr, axis, :hi) ?
                 MeshInterface.bc_on(orig, axis, :hi) : pg
        (lo, hi)
    end
    sim.bcs = MeshInterface.BoundaryConditions(pairs)
    return pg
end

# Write one (dim, side) flux plane of an Enzo flux register for ALL Enzo baryon fields
# (consumers loop every field and deref each, so unmapped fields must still be zeroed).
function _write_plane!(::Val{R}, setter, mesh, nf, dim::Int, start, stop, g0,
                       flux_off::Int, Vcell, lookup) where {R}
    s = ntuple(d -> start[d], Val(R)); e = ntuple(d -> stop[d], Val(R)); g = ntuple(d -> g0[d], Val(R))
    for fld in 0:nf-1
        comp = _engng_comp_of(mesh, fld)
        plane = Vespa.bflux_plane(Val(R), dim, s, e, g, flux_off, comp, Vcell, lookup)
        setter(fld, plane)
    end
    return nothing
end

# Write `breg` (a BoundaryFluxRegister-like object with `.flux`/`.interior`) for grid
# (level, i, flat gi) into Enzo's flux registers: proper subgrids → coarse InitialFluxes
# (interior register); the last entry → the grid's own outer-boundary flux (flux register).
# `conservative=false` writes ZEROS (arrays still allocated) — the non-conservative baseline.
function _write_fluxes!(h, level, i, gi, mesh::EnzoGridMesh{R}, sim, breg, model; conservative::Bool) where {R}
    Vcell = MeshInterface.cell_volume(mesh, first(CartesianIndices(mesh.active)))
    g0 = EnzoLib.problem_grid_global_start(h, gi)        # length-3 global start
    nf = EnzoLib.problem_num_fields(h, gi)
    nsub = EnzoLib.problem_num_subgrids(h, level, i)

    axis_of(dim) = dim + 1
    interior_lookup(dim, side, start) = begin
        ax = axis_of(dim)
        off = start[dim+1] - g0[dim+1] + (side == 0 ? 0 : 1)
        I -> conservative ? get(breg.interior, (ax, I), nothing) : nothing, off
    end

    for sub in 0:nsub-2
        for dim in 0:R-1, side in 0:1
            st, en = EnzoLib.problem_subgrid_flux_extent(h, level, i, sub, dim, side)
            lk, off = interior_lookup(dim, side, st)
            _write_plane!(Val(R), (fld, pl) -> EnzoLib.problem_set_subgrid_flux(h, level, i, sub, fld, dim, side, pl),
                          mesh, nf, dim, st, en, g0, off, Vcell, lk)
        end
    end

    own = nsub - 1
    for dim in 0:R-1, side in 0:1
        st, en = EnzoLib.problem_subgrid_flux_extent(h, level, i, own, dim, side)
        sym = side == 0 ? :lo : :hi
        boundary_cell = side == 0 ? 1 : mesh.active[dim+1]
        ax = axis_of(dim)
        lk = I -> conservative ? get(breg.flux, (ax, sym, I), nothing) : nothing
        _write_plane!(Val(R), (fld, pl) -> EnzoLib.problem_set_subgrid_flux(h, level, i, own, fld, dim, side, pl),
                      mesh, nf, dim, st, en, g0, boundary_cell, Vcell, lk)
    end
    return nothing
end

# ── steppers: the only piece that differs between backends ───────────────────────────────

# Native Vespa driver: ghost-free HLLC + PLM + SSP-RK2, recording boundary fluxes into `breg`.
native_stepper() = (h, level, gi, mesh, sim, breg, model, dt) -> begin
    step!(sim, dt; bflux = breg)
    sync_to_enzo!(mesh, sim.sv)
    return nothing
end

# PPMKernels MUSCL-Hancock on a KA backend (CPU or GPU), with `frec` flux recording →
# PPMKernels.boundary_flux_register → merged into `breg`. Reads/writes the live grid's
# ghosted fields directly (Enzo has already filled the ghosts — incl. parent-interpolated
# for subgrids — before the hydro slot). 3D only (muscl_hancock_step_3d! is a 3-D driver).
# `scheme` selects the KA driver: :split = muscl_hancock_step_3d! (dimensionally split, fast,
# conserves ~few×1e-4 under reflux) or :unsplit = muscl_step_3d! (operator-unsplit RK2 — a single
# per-face flux ½(F₁+F₂) that telescopes EXACTLY, giving ROUND-OFF AMR conservation, ~2× cost).
function ppmkernels_stepper(; backend::Symbol = :cpu, scheme::Symbol = :split,
                            recon::Symbol = :plm, riemann::Symbol = :hll, predictor::Symbol = :hancock,
                            precision::Type = Float64)
    bep = PPMKernels.backend(backend)
    return (h, level, gi, mesh, sim, breg, model, dt) -> begin
        R = MeshInterface.rank(mesh)
        R == 3 || error("ppmkernels_stepper: the 3-D KA drivers are 3-D only (grid rank $R)")
        ng = mesh.nghost
        gd = ntuple(d -> mesh.active[d] + 2 * ng, 3)
        dx = MeshInterface.cell_width(mesh, first(CartesianIndices(mesh.active)))[1]
        T = precision
        # full ghosted Enzo fields (column-major flat, length prod(gd)) → conserved device arrays
        d  = EnzoLib.problem_get_field(h, mesh.di, gi)
        es = EnzoLib.problem_get_field(h, mesh.ei, gi)
        vget(k) = mesh.vi[k] >= 0 ? EnzoLib.problem_get_field(h, mesh.vi[k], gi) : zeros(length(d))
        vx = vget(1); vy = vget(2); vz = vget(3)
        D   = PPMKernels.to_device(bep, d,        T)
        S1  = PPMKernels.to_device(bep, d .* vx,  T)
        S2  = PPMKernels.to_device(bep, d .* vy,  T)
        S3  = PPMKernels.to_device(bep, d .* vz,  T)
        Tau = PPMKernels.to_device(bep, d .* es,  T)
        frec = ntuple(_ -> ntuple(_ -> PPMKernels.device_zeros(bep, T, (prod(gd),)), 6), 3)
        if scheme === :unsplit
            PPMKernels.muscl_step_3d!(D, S1, S2, S3, Tau, gd, ng;
                dt = dt, gamma = model.γ, dx = dx, fluxrec = frec)
        else
            PPMKernels.muscl_hancock_step_3d!(D, S1, S2, S3, Tau, gd, ng;
                dt = dt, gamma = model.γ, dx = dx,
                recon = recon, riemann = riemann, predictor = predictor, fluxrec = frec)
        end
        # frec → boundary flux register (same keys/units as Vespa's native breg) → merge in
        bset = PPMKernels.boundary_flux_register(frec, gd, ng, dt, dx; nv = 5)
        merge!(breg.flux, bset.flux); merge!(breg.interior, bset.interior)
        # conserved device arrays → Enzo (Density / Velocity / specific TotalEnergy)
        Dh = PPMKernels.to_host(D)
        EnzoLib.problem_set_field(h, mesh.di, Float64.(Dh); grid = gi)
        mesh.vi[1] >= 0 && EnzoLib.problem_set_field(h, mesh.vi[1], Float64.(PPMKernels.to_host(S1)) ./ Float64.(Dh); grid = gi)
        mesh.vi[2] >= 0 && EnzoLib.problem_set_field(h, mesh.vi[2], Float64.(PPMKernels.to_host(S2)) ./ Float64.(Dh); grid = gi)
        mesh.vi[3] >= 0 && EnzoLib.problem_set_field(h, mesh.vi[3], Float64.(PPMKernels.to_host(S3)) ./ Float64.(Dh); grid = gi)
        EnzoLib.problem_set_field(h, mesh.ei, Float64.(PPMKernels.to_host(Tau)) ./ Float64.(Dh); grid = gi)
        return nothing
    end
end

# The conservative :julia hydro hook, parameterized by `stepper` (default = native Vespa).
function reflux_hydro_hook(stepper; γ = 1.4, nghost = 3, conservative::Bool = true, parent_ghost::Bool = true)
    model = IdealHydro(γ)
    return function (h, level, dt)
        n = EnzoLib.session_num_grids_on_level(h, level)
        me = EnzoLib.session_my_rank(h)
        for i in 0:n-1
            gi = EnzoLib.problem_grid_index_on_level(h, level, i)
            EnzoLib.problem_grid_processor(h, gi) == me || continue   # local grids only
            mesh, sim = _build_grid_sim(h, gi, model, nghost)
            breg = Vespa._bflux_register(sim; record_interior = true)
            sync_from_enzo!(sim.sv, mesh)
            parent_ghost && _apply_parent_ghost!(sim, mesh, model, level)
            stepper(h, level, gi, mesh, sim, breg, model, dt)
            _write_fluxes!(h, level, i, gi, mesh, sim, breg, model; conservative = conservative)
        end
        return nothing
    end
end

# Backward-compatible alias: the native-driver hook used by the 1D/2D gates.
conservative_julia_hydro_hook(; γ = 1.4, nghost = 3, conservative::Bool = true, parent_ghost::Bool = true) =
    reflux_hydro_hook(native_stepper(); γ = γ, nghost = nghost,
                      conservative = conservative, parent_ghost = parent_ghost)

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

# Drive a 2+-level AMR run from Julia with a conservative :julia hydro `stepper`, returning
# the root-grid composite totals before/after. `regrid=false` holds the hierarchy static;
# `nsteps` caps the root steps (so the decisive run stops while waves are still interior).
function _run_reflux(pf; conservative::Bool, regrid::Bool = true, nsteps::Int = 100000,
                     parent_ghost::Bool = true, stepper = native_stepper())
    eng = EnzoLib.EngineConfig(; hydro = :julia, reflux = true,
                               hooks = Dict{Symbol,Function}(:hydro =>
                                   reflux_hydro_hook(stepper; conservative = conservative,
                                                     parent_ghost = parent_ghost)))
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

# The Enzo problem-file tree (`run/`) stayed in the sibling enzo-dev checkout when Vespa.jl
# was extracted. Resolve it via ENV["ENZO_DEV_REPO"], else the sibling ~/Projects layout.
const _ENZO_DEV = get(ENV, "ENZO_DEV_REPO",
                      normpath(joinpath(@__DIR__, "..", "..", "..", "..", "enzo-dev")))
