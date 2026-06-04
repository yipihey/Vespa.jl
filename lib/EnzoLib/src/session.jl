# Native ccall binding to the EnzoModules **live-Session** C-ABI (the grid bridge
# library, libenzomodules_grid, which links the full Enzo .so). This exposes
# Enzo's actual time-integration steps on a live hierarchy, so a Julia-driven
# EvolveLevel can call the certified legacy reference for each step — the
# full-replication mode. Mirrors EnzoModules/enzomodules/problems.py (the Session
# the README proves reproduces EvolveHierarchy bit-for-bit).
#
# Build the library with EnzoModules/deps/build_grid_darwin.sh (macOS) /
# build_grid.sh (Linux); located via ENV["ENZOMODULES_GRID_LIB"] or deps/.

"True when the opt-in MPI flavor of the Enzo substrate is selected (`ENV[\"ENZONG_ENZO_MPI\"]==\"1\"`)."
enzo_mpi_enabled() = get(ENV, "ENZONG_ENZO_MPI", "") == "1"

"Path to the EnzoModules grid/session bridge library (links the full Enzo .so)."
function grid_libpath()
    env = get(ENV, "ENZOMODULES_GRID_LIB", "")
    isempty(env) || return abspath(env)
    base = normpath(joinpath(@__DIR__, "..", "..", "..", "..", "EnzoModules", "deps"))
    # Default is the serial bridge; the MPI flavor (built by build_grid_darwin.sh mpi)
    # is opt-in and carries an _mpi suffix so both artifacts coexist.
    names = enzo_mpi_enabled() ?
        ("libenzomodules_grid_mpi.dylib", "libenzomodules_grid_mpi.so") :
        ("libenzomodules_grid.dylib", "libenzomodules_grid.so")
    for name in names
        p = joinpath(base, name); isfile(p) && return p
    end
    return joinpath(base, first(names))
end
grid_available() = isfile(grid_libpath())

# The MPI flavor's libenzo/bridge are linked with `-undefined dynamic_lookup` and
# carry NO libmpitrampoline dependency (linking one would load a SECOND trampoline
# instance and MPItrampoline aborts on double-load).  Their MPI_* symbols are
# resolved at bridge-load time from the host process's already-loaded MPItrampoline
# (MPI.jl's) — but only if it is in GLOBAL scope.  Promote it before dlopening the
# bridge: prefer ENV["ENZONG_MPITRAMPOLINE"] (explicit path), else promote the
# already-loaded image by leaf name (RTLD_NOLOAD).
function _promote_mpitrampoline()
    flags = Libdl.RTLD_GLOBAL | Libdl.RTLD_LAZY
    path = get(ENV, "ENZONG_MPITRAMPOLINE", "")
    try
        if !isempty(path)
            Libdl.dlopen(path, flags)
        else
            Libdl.dlopen("libmpitrampoline.dylib", flags | Libdl.RTLD_NOLOAD)
        end
    catch e
        @warn "EnzoLib: could not promote MPItrampoline to global scope; the MPI \
               bridge's MPI_* symbols may be unresolved. Set ENV[\"ENZONG_MPITRAMPOLINE\"] \
               to libmpitrampoline's path." exception = e
    end
    return nothing
end

const _GHANDLE = Ref{Ptr{Cvoid}}(C_NULL)
function _ghandle()
    if _GHANDLE[] == C_NULL
        grid_available() || error("grid/session library not found at $(grid_libpath()). " *
                                  "Build it: bash EnzoModules/deps/build_grid_darwin.sh")
        if enzo_mpi_enabled()
            _promote_mpitrampoline()
            # RTLD_GLOBAL so the bridge participates in flat MPI_* resolution.
            _GHANDLE[] = Libdl.dlopen(grid_libpath(), Libdl.RTLD_GLOBAL | Libdl.RTLD_LAZY)
        else
            _GHANDLE[] = Libdl.dlopen(grid_libpath())
        end
    end
    return _GHANDLE[]
end
@inline _gsym(name::Symbol) = Libdl.dlsym(_ghandle(), name)

# ── transport seam (ADR-0005) ─────────────────────────────────────────────────
# Every bridge call goes through `@xcall`, which expands to either the in-process
# `ccall` (local, the default and the serial-verified path) or a remote RPC to a
# worker process (remote).  The C symbol + return type + arg types are written
# ONCE at the call site — there is no second hand-maintained interface to drift,
# and the same declaration is what a manifest generator (#2) reads to produce the
# worker dispatch + remote stubs.  Switch with `set_backend!`.
const _BACKEND = Ref{Symbol}(:local)
"Select the bridge transport: `:local` (in-process ccall) or `:remote` (worker RPC)."
set_backend!(b::Symbol) = (b in (:local, :remote) || error("backend must be :local or :remote"); _BACKEND[] = b)
backend() = _BACKEND[]

# Remote dispatch — wired in ADR-0005 #2 (manifest-generated worker + shm buffers).
_rpc(sym::Symbol, ret, argtypes, args) =
    error("remote bridge backend not yet wired for $sym (ADR-0005 #2)")

"""
    @xcall(:c_symbol, RetType, (ArgTypes...), args...)

Backend-dispatching bridge call.  Local → `ccall(_gsym(:c_symbol), RetType,
(ArgTypes...), args...)` (literal types preserved); remote → `_rpc(...)`.
"""
macro xcall(sym, ret, argtypes, args...)
    a = map(esc, args)
    quote
        if _BACKEND[] === :local
            ccall(_gsym($(esc(sym))), $(esc(ret)), $(esc(argtypes)), $(a...))
        else
            _rpc($(esc(sym)), $(esc(ret)), $(esc(argtypes)), ($(a...),))
        end
    end
end

# ── low-level Session / problem entry points (signatures per problems.py) ─────
const Handle = Ptr{Cvoid}

"Initialize a problem and hold the LIVE hierarchy (InitializeNew). Returns a handle."
session_init(paramfile::AbstractString) =
    @xcall(:enzomodules_session_init, Handle, (Cstring,), paramfile)
"Run Enzo's full time integrator (EvolveHierarchy) to stop_time/stop_cycle (0 ⇒ param file)."
evolve_problem(paramfile::AbstractString, stop_time::Real = 0.0, stop_cycle::Integer = 0) =
    @xcall(:enzomodules_evolve_problem, Handle, (Cstring, Cdouble, Cint),
          paramfile, stop_time, stop_cycle)
free_problem(h::Handle) = @xcall(:enzomodules_free_problem, Cvoid, (Handle,), h)

session_time(h::Handle)      = @xcall(:enzomodules_session_time, Cdouble, (Handle,), h)
session_stop_time(h::Handle) = @xcall(:enzomodules_session_stop_time, Cdouble, (Handle,), h)
session_cycle(h::Handle)     = @xcall(:enzomodules_session_cycle, Cint, (Handle,), h)
session_compute_dt(h::Handle, level::Integer = 0) =
    @xcall(:enzomodules_session_compute_dt, Cdouble, (Handle, Cint), h, level)
session_set_dt(h::Handle, dt::Real, level::Integer = 0) =
    @xcall(:enzomodules_session_set_dt, Cvoid, (Handle, Cint, Cdouble), h, level, dt)
session_set_boundary(h::Handle, level::Integer = 0) =
    @xcall(:enzomodules_session_set_boundary, Cint, (Handle, Cint), h, level)
session_solve_hydro(h::Handle, level::Integer = 0) =
    @xcall(:enzomodules_session_solve_hydro, Cint, (Handle, Cint), h, level)
session_advance_time(h::Handle, level::Integer = 0) =
    @xcall(:enzomodules_session_advance_time, Cvoid, (Handle, Cint), h, level)

# AMR EvolveLevel steps (the conservation machinery): flux storage, projection +
# flux-correction from finer levels, and regridding.
session_clear_boundary_fluxes(h::Handle, level::Integer = 0) =
    @xcall(:enzomodules_session_clear_boundary_fluxes, Cvoid, (Handle, Cint), h, level)
session_create_fluxes(h::Handle, level::Integer = 0) =
    @xcall(:enzomodules_session_create_fluxes, Cint, (Handle, Cint), h, level)
session_finalize_fluxes(h::Handle, level::Integer = 0) =
    @xcall(:enzomodules_session_finalize_fluxes, Cint, (Handle, Cint), h, level)
session_update_from_finer(h::Handle, level::Integer = 0) =
    @xcall(:enzomodules_session_update_from_finer, Cint, (Handle, Cint), h, level)
session_copy_baryon_to_old(h::Handle, level::Integer = 0) =
    @xcall(:enzomodules_session_copy_baryon_to_old, Cvoid, (Handle, Cint), h, level)
session_update_particles(h::Handle, level::Integer = 0) =
    @xcall(:enzomodules_session_update_particles, Cvoid, (Handle, Cint), h, level)
session_rebuild(h::Handle, level::Integer = 0) =
    @xcall(:enzomodules_session_rebuild, Cint, (Handle, Cint), h, level)
session_num_grids_on_level(h::Handle, level::Integer) =
    @xcall(:enzomodules_session_num_grids_on_level, Cint, (Handle, Cint), h, level)

# Optional physics steps (no-ops when their physics is off) — for non-hydro problems.
session_gravity(h::Handle, level::Integer = 0) =
    @xcall(:enzomodules_session_gravity, Cint, (Handle, Cint), h, level)
session_solve_cooling(h::Handle, level::Integer = 0) =
    @xcall(:enzomodules_session_solve_cooling, Cint, (Handle, Cint), h, level)
session_star_particles(h::Handle, level::Integer = 0) =
    @xcall(:enzomodules_session_star_particles, Cint, (Handle, Cint), h, level)
session_update_radiation_field(h::Handle, level::Integer = 0) =
    @xcall(:enzomodules_session_update_radiation_field, Cint, (Handle, Cint), h, level)
"Emit + transport photons; `stars=true` converts star particles into sources."
session_evolve_photons(h::Handle, level::Integer = 0; stars::Bool = false) =
    @xcall(:enzomodules_session_evolve_photons_ex, Cint, (Handle, Cint, Cint),
          h, level, stars ? 1 : 0)
"Comoving (Hubble-drag) expansion source terms — call after advance_time for cosmology runs."
session_comoving_expansion(h::Handle, level::Integer = 0) =
    @xcall(:enzomodules_session_comoving_expansion, Cint, (Handle, Cint), h, level)
"CT-MHD: zero+allocate the per-grid AvgElectricField accumulator (level>0 entry; EvolveLevel.C:377)."
session_clear_avg_electric_field(h::Handle, level::Integer = 0) =
    @xcall(:enzomodules_session_clear_avg_electric_field, Cint, (Handle, Cint), h, level)
"CT-MHD: recompute cell-centered B from face-corrected B using finer-grid EMF (post-UFG; EvolveLevel.C:899)."
session_mhd_update_magnetic_field(h::Handle, level::Integer = 0) =
    @xcall(:enzomodules_session_mhd_update_magnetic_field, Cint, (Handle, Cint), h, level)

problem_num_grids(h::Handle) = @xcall(:enzomodules_problem_num_grids, Cint, (Handle,), h)
problem_grid_size(h::Handle, grid::Integer = 0) =
    @xcall(:enzomodules_problem_grid_size, Cint, (Handle, Cint), h, grid)
problem_num_fields(h::Handle, grid::Integer = 0) =
    @xcall(:enzomodules_problem_num_fields, Cint, (Handle, Cint), h, grid)
function problem_field_types(h::Handle, grid::Integer = 0)
    n = problem_num_fields(h, grid)
    t = zeros(Cint, n)
    @xcall(:enzomodules_problem_field_types, Cint, (Handle, Cint, Ptr{Cint}), h, grid, t)
    return Int.(t)
end
"Read one field (flat, incl. ghost zones) by 0-based field index."
function problem_get_field(h::Handle, fi::Integer, grid::Integer = 0)
    out = zeros(Float64, problem_grid_size(h, grid))
    @xcall(:enzomodules_problem_get_field, Cint, (Handle, Cint, Cint, Ptr{Cdouble}),
          h, grid, fi, out)
    return out
end

"""
    problem_set_field(h, fi, data; grid=0)

Write a field (flat, incl. ghost zones) back into the LIVE grid's BaryonField.
This is the enabling primitive for the `:julia` slot swap: a Julia physics method
reads the state (`problem_get_field`), computes, and writes it back here —
mutating the same Enzo memory Enzo's own kernels operate on.
"""
function problem_set_field(h::Handle, fi::Integer, data::Vector{Float64}; grid::Integer = 0)
    length(data) == problem_grid_size(h, grid) ||
        throw(DimensionMismatch("field length $(length(data)) ≠ grid size $(problem_grid_size(h, grid))"))
    @xcall(:enzomodules_problem_set_field, Cvoid, (Handle, Cint, Cint, Ptr{Cdouble}),
          h, grid, fi, data)
    return nothing
end

"""
    problem_set_acceleration(h, dim, data; grid=0)
    problem_get_acceleration(h, dim, data; grid=0)

Write / read the cell-centered `AccelerationField[dim]` (0-based dim) — the gravity
source `SolveHydroEquations` reads. The enabling primitive for a `:julia` gravity
slot: EnzoNG solves Poisson on the live density, computes `g = −∇φ`, and writes it
here, so Enzo's hydro applies the Julia-computed gravity. `set` allocates if the
field is absent (a Julia slot replaces `ComputeAccelerations`, which normally does).
"""
function problem_set_acceleration(h::Handle, dim::Integer, data::Vector{Float64}; grid::Integer = 0)
    length(data) == problem_grid_size(h, grid) ||
        throw(DimensionMismatch("acceleration length $(length(data)) ≠ grid size $(problem_grid_size(h, grid))"))
    @xcall(:enzomodules_problem_set_acceleration, Cvoid, (Handle, Cint, Cint, Ptr{Cdouble}),
          h, grid, dim, data)
    return nothing
end
function problem_get_acceleration(h::Handle, dim::Integer, grid::Integer = 0)
    out = zeros(Float64, problem_grid_size(h, grid))
    @xcall(:enzomodules_problem_get_acceleration, Cvoid, (Handle, Cint, Cint, Ptr{Cdouble}),
          h, grid, dim, out)
    return out
end

# ── ADR-0003 part B: BoundaryFluxes bridge (conservative :julia hydro under AMR) ──
# Write EnzoNG's recorded face fluxes (in Enzo's F·dt/dx units, BaryonField order)
# into Enzo's flux registers so UpdateFromFinerGrids/CorrectForRefinedFluxes restore
# conservation across coarse–fine boundaries. `side`: 0 = Left face, 1 = Right face.

"Flat grid-list index of the `i`-th grid on `level` (GenerateGridArray order) — matches subgrid-flux indexing."
problem_grid_index_on_level(h::Handle, level::Integer, i::Integer) =
    Int(@xcall(:enzomodules_problem_grid_index_on_level, Cint, (Handle, Cint, Cint), h, level, i))

"Global zone index of grid `gi`'s first ACTIVE cell per dim (length 3) — to map local faces to global flux indices."
function problem_grid_global_start(h::Handle, gi::Integer = 0)
    g = zeros(Int64, 3)
    @xcall(:enzomodules_problem_grid_global_start, Cvoid, (Handle, Cint, Ptr{Clonglong}), h, gi, g)
    return Int.(g)
end

"Physical edges (left, right), each length 3, of grid `gi` — for a per-grid mesh's cell width."
function problem_grid_edge(h::Handle, gi::Integer = 0)
    l = zeros(Float64, 3); r = zeros(Float64, 3)
    @xcall(:enzomodules_problem_grid_edge, Cvoid, (Handle, Cint, Ptr{Cdouble}, Ptr{Cdouble}), h, gi, l, r)
    return (l, r)
end

"Number of plane cells in grid `gi`'s `dim` boundary-flux face (1 in 1D)."
problem_boundary_flux_size(h::Handle, gi::Integer, dim::Integer) =
    Int(@xcall(:enzomodules_problem_boundary_flux_size, Cint, (Handle, Cint, Cint), h, gi, dim))

"Global-index extents (start, end), each length 3, of grid `gi`'s `dim`/`side` boundary-flux plane."
function problem_boundary_flux_extent(h::Handle, gi::Integer, dim::Integer, side::Integer)
    s = zeros(Int64, 3); e = zeros(Int64, 3)
    @xcall(:enzomodules_problem_boundary_flux_extent, Cvoid,
          (Handle, Cint, Cint, Cint, Ptr{Clonglong}, Ptr{Clonglong}), h, gi, dim, side, s, e)
    return (Int.(s), Int.(e))
end

"ADD a boundary-flux plane into grid `gi`'s BoundaryFluxes[field][dim][side] (accumulates over subcycles)."
function problem_set_boundary_flux(h::Handle, gi::Integer, field::Integer, dim::Integer,
                                   side::Integer, plane::Vector{Float64})
    @xcall(:enzomodules_problem_set_boundary_flux, Cvoid,
          (Handle, Cint, Cint, Cint, Cint, Ptr{Cdouble}), h, gi, field, dim, side, plane)
    return nothing
end
function problem_get_boundary_flux(h::Handle, gi::Integer, field::Integer, dim::Integer, side::Integer)
    out = zeros(Float64, problem_boundary_flux_size(h, gi, dim))
    @xcall(:enzomodules_problem_get_boundary_flux, Cvoid,
          (Handle, Cint, Cint, Cint, Cint, Ptr{Cdouble}), h, gi, field, dim, side, out)
    return out
end

"Number of subgrid flux entries for the `i`-th grid on `level` (proper subgrids + 1 own-boundary); needs create_fluxes(level)."
problem_num_subgrids(h::Handle, level::Integer, i::Integer) =
    Int(@xcall(:enzomodules_problem_num_subgrids, Cint, (Handle, Cint, Cint), h, level, i))

"Coarse-index global extents (start, end) of subgrid flux (level,i,sub) for `dim`/`side`."
function problem_subgrid_flux_extent(h::Handle, level::Integer, i::Integer, sub::Integer,
                                     dim::Integer, side::Integer)
    s = zeros(Int64, 3); e = zeros(Int64, 3)
    @xcall(:enzomodules_problem_subgrid_flux_extent, Cvoid,
          (Handle, Cint, Cint, Cint, Cint, Cint, Ptr{Clonglong}, Ptr{Clonglong}),
          h, level, i, sub, dim, side, s, e)
    return (Int.(s), Int.(e))
end
problem_subgrid_flux_size(h::Handle, level::Integer, i::Integer, sub::Integer, dim::Integer) =
    Int(@xcall(:enzomodules_problem_subgrid_flux_size, Cint, (Handle, Cint, Cint, Cint, Cint),
              h, level, i, sub, dim))

"SET a coarse InitialFlux plane into SubgridFluxesEstimate[level][i][sub][field][dim][side] (allocates if needed)."
function problem_set_subgrid_flux(h::Handle, level::Integer, i::Integer, sub::Integer,
                                  field::Integer, dim::Integer, side::Integer, plane::Vector{Float64})
    @xcall(:enzomodules_problem_set_subgrid_flux, Cvoid,
          (Handle, Cint, Cint, Cint, Cint, Cint, Cint, Ptr{Cdouble}),
          h, level, i, sub, field, dim, side, plane)
    return nothing
end
function problem_get_subgrid_flux(h::Handle, level::Integer, i::Integer, sub::Integer,
                                  field::Integer, dim::Integer, side::Integer)
    out = zeros(Float64, problem_subgrid_flux_size(h, level, i, sub, dim))
    @xcall(:enzomodules_problem_get_subgrid_flux, Cvoid,
          (Handle, Cint, Cint, Cint, Cint, Cint, Cint, Ptr{Cdouble}),
          h, level, i, sub, field, dim, side, out)
    return out
end

# ── particles (positions) ────────────────────────────────────────────────────
"Spatial rank (1/2/3) of a grid."
problem_grid_rank(h::Handle, grid::Integer = 0) =
    Int(@xcall(:enzomodules_problem_grid_rank, Cint, (Handle, Cint), h, grid))
"Refinement level of grid `grid` (0 = root, -1 if absent) — to iterate a level's grids for a :julia AMR slot."
problem_grid_level(h::Handle, grid::Integer = 0) =
    Int(@xcall(:enzomodules_problem_grid_level, Cint, (Handle, Cint), h, grid))
"Indices of all grids on `level` in the flat grid list (for a per-level :julia slot)."
grids_on_level(h::Handle, level::Integer) =
    [g for g in 0:problem_num_grids(h)-1 if problem_grid_level(h, g) == level]

"This rank's MPI id (0 in the serial flavor)."
session_my_rank(h::Handle) =
    Int(@xcall(:enzomodules_session_my_rank, Cint, (Handle,), h))
"Total MPI rank count (1 in the serial flavor)."
session_num_ranks(h::Handle) =
    Int(@xcall(:enzomodules_session_num_ranks, Cint, (Handle,), h))
"Home processor (rank) of grid `grid` (always 0 in the serial flavor; -1 if absent)."
problem_grid_processor(h::Handle, grid::Integer) =
    Int(@xcall(:enzomodules_problem_grid_processor, Cint, (Handle, Cint), h, grid))

# Grids on `level` RESIDENT on this rank — the only ones whose BaryonField/flux
# registers are allocated here, so the only ones a :julia hydro slot may touch
# (mirrors Enzo's SolveHydroEquations, which skips grids with ProcessorNumber !=
# MyProcessorNumber).  In the serial flavor every grid is local, so this equals
# `grids_on_level`.  Enzo's own SetBoundaryConditions / UpdateFromFinerGrids move
# data across ranks; the :julia kernel only ever runs on local grids.
local_grids_on_level(h::Handle, level::Integer) =
    let me = session_my_rank(h)
        [g for g in grids_on_level(h, level) if problem_grid_processor(h, g) == me]
    end
"Full per-dim Enzo `GridDimension` (incl. ghosts; column-major; always length 3, 1 past the rank)."
function problem_grid_dims(h::Handle, grid::Integer = 0)
    d = zeros(Cint, 3)
    @xcall(:enzomodules_problem_grid_dims, Cvoid, (Handle, Cint, Ptr{Cint}), h, grid, d)
    return Int.(d)
end
"Number of particles living on `grid`."
problem_num_particles(h::Handle, grid::Integer = 0) =
    Int(@xcall(:enzomodules_problem_num_particles, Cint, (Handle, Cint), h, grid))
"Particle positions along axis `dim` (0-based) on `grid`, in code units."
function problem_get_particle_pos(h::Handle, dim::Integer, grid::Integer = 0)
    np = problem_num_particles(h, grid)
    out = zeros(Float64, np)
    np == 0 && return out
    @xcall(:enzomodules_problem_get_particle_pos, Cvoid, (Handle, Cint, Cint, Ptr{Cdouble}),
          h, grid, dim, out)
    return out
end

"""
    read_particles(h) -> Matrix{Float64} (Nparticles × rank)

Gather every particle's position across ALL grids of the hierarchy into one
`Np × rank` matrix (code units). Both the Enzo `EvolveHierarchy` reference and the
Julia-driven `EvolveLevel` move the same particles with the same Enzo routines, so
the resulting point sets agree (bit-for-bit single-grid; AMR-ordering otherwise).
"""
function read_particles(h::Handle)
    ng = problem_num_grids(h)
    rank = ng > 0 ? problem_grid_rank(h, 0) : 3
    blocks = Matrix{Float64}[]
    for g in 0:ng-1
        np = problem_num_particles(h, g)
        np == 0 && continue
        M = Matrix{Float64}(undef, np, rank)
        for d in 0:rank-1
            M[:, d+1] = problem_get_particle_pos(h, d, g)
        end
        push!(blocks, M)
    end
    isempty(blocks) && return Matrix{Float64}(undef, 0, rank)
    return reduce(vcat, blocks)
end

"All baryon fields AND particle positions of the hierarchy: (fields=Dict, particles=Matrix)."
read_state(h::Handle; grid::Integer = 0) =
    (fields = read_all_fields(h; grid = grid), particles = read_particles(h))

"0-based field index of a given Enzo `FieldType` (Density=0, TotalEnergy=1, Velocity1=4)."
function field_index(h::Handle, ftype::Integer; grid::Integer = 0)
    i = findfirst(==(ftype), problem_field_types(h, grid))
    i === nothing && error("no field of FieldType $ftype in grid $grid")
    return i - 1
end

# ── high-level: read Density, the Julia-driven loop, the EvolveHierarchy ref ──
"Read the Density field (FieldType 0) of `grid` from a live/evolved hierarchy."
function read_density(h::Handle; grid::Integer = 0)
    fi = findfirst(==(0), problem_field_types(h, grid))   # FieldType Density == 0
    fi === nothing && error("no Density field in grid $grid")
    return problem_get_field(h, fi - 1, grid)             # 1-based Julia → 0-based C
end

"""
    session_replicate_density(paramfile) -> Vector{Float64}

Drive Enzo's time loop FROM JULIA — a reimplementation of the single-grid
`EvolveLevel` calling the certified legacy steps each cycle (set_boundary →
compute_dt → set_dt → solve_hydro → advance_time) — and return the final Density.
This is full-replication mode: the new Julia driver, the old Enzo routines.
"""
function session_replicate_density(paramfile::AbstractString; level::Integer = 0, maxcycle = 100000)
    return session_evolve_density(paramfile, (h, dt) -> session_solve_hydro(h, level);
                                  level = level, maxcycle = maxcycle)
end

"""
    session_evolve_density(paramfile, hydro!; level=0, maxcycle=…) -> Vector{Float64}

Generic Julia-driven EvolveLevel: each cycle does `set_boundary → compute_dt →
set_dt → hydro!(h, dt) → advance_time` on the live hierarchy, then returns the
final Density. `hydro!` is the swappable hydro SLOT — pass the legacy
`session_solve_hydro` (full replication) or a Julia method that reads/computes/
writes the live grid (the `:julia` mix-and-match swap).
"""
function session_evolve_density(paramfile::AbstractString, hydro!::Function;
                                level::Integer = 0, maxcycle = 100000)
    pf = abspath(paramfile)
    cd(_workdir(pf)) do
        h = session_init(pf)
        h == C_NULL && error("session_init failed for $pf")
        try
            n = 0
            while session_time(h) < session_stop_time(h) && n < maxcycle
                session_set_boundary(h, level)
                dt = session_compute_dt(h, level)
                session_set_dt(h, dt, level)
                hydro!(h, dt)
                session_advance_time(h, level)
                n += 1
            end
            return read_density(h)
        finally
            free_problem(h)
        end
    end
end

"""
    reference_density(paramfile) -> Vector{Float64}

Run Enzo's own monolithic `EvolveHierarchy` (the OLD code) to the parameter
file's StopTime and return the final Density — the bit-for-bit reference.
"""
function reference_density(paramfile::AbstractString)
    pf = abspath(paramfile)
    cd(_workdir(pf)) do
        h = evolve_problem(pf, 0.0, 0)
        h == C_NULL && error("evolve_problem (EvolveHierarchy) failed for $pf")
        try
            return read_density(h)
        finally
            free_problem(h)
        end
    end
end

# A fresh working directory for a run, with the param file's small sibling files
# staged in (refine-region sequences, parameter includes, etc.). Enzo reads such
# auxiliary files RELATIVE TO CWD, so a bare mktempdir would miss them (e.g.
# MHDCTOrszagTangAMR's `RefineRegionFile = RefinementSequence`). The .enzo itself
# is passed by absolute path, so it is not copied; large files (HDF5 ICs) are
# skipped — problems that need those abort on their own anyway.
function _workdir(paramfile::AbstractString)
    d = mktempdir()
    src = dirname(abspath(paramfile))
    isdir(src) && for f in readdir(src)
        p = joinpath(src, f)
        (isfile(p) && filesize(p) < 1_000_000 && !endswith(f, ".enzo")) || continue
        try; cp(p, joinpath(d, f); force = true); catch; end
    end
    return d
end

# Enzo's StopCycle from a .enzo param file (top-grid cycle limit; default 100000,
# i.e. effectively unlimited — StopTime governs). EvolveHierarchy honors this, so
# the Julia-driven loop must too, or it runs to a different epoch than the reference.
function _stop_cycle(paramfile::AbstractString)
    for ln in eachline(paramfile)
        m = match(r"^\s*StopCycle\s*=\s*([0-9]+)", ln)
        m === nothing || return parse(Int, m.captures[1])
    end
    return 100000
end

# ── method-slot registry (ADR-0002) ──────────────────────────────────────────
# Each physics step in the EvolveLevel skeleton is a SLOT resolving to
# :off (skip), :enzo (the certified legacy bridge step), or :julia (an injected
# EnzoNG kernel running on the live grid). EnzoLib has no EnzoNG dependency, so a
# :julia slot is supplied as a hook closure `(h, level, dt)` by the integration
# layer (which has EnzoNG/EnzoBackend). The AMR/conservation plumbing (boundaries,
# flux registers, projection, regrid) is NOT a slot — it always runs.

# ── performance probe (ADR-0002 reporting) ───────────────────────────────────
# A zero-overhead-WHEN-ABSENT accumulator wired into run_slot. It times each
# physics-slot CALL — a whole hydro/gravity step, so the ~20 ns `time_ns()` pair
# is <1e-4 of the work and never an observer effect. When the engine's `probe` is
# `nothing`, run_slot compiles to the bare call (production pays nothing). The
# per-call sample is pushed AFTER the timed window, so bookkeeping never pollutes
# the measurement. Allocation bytes per slot come from `Base.gc_bytes()` deltas.
mutable struct SlotProbe
    ns::Dict{Symbol,Vector{Int}}        # per-slot per-call wall-time samples (ns)
    bytes::Dict{Symbol,Int}             # per-slot total allocated bytes
end
SlotProbe() = SlotProbe(Dict{Symbol,Vector{Int}}(), Dict{Symbol,Int}())
reset!(p::SlotProbe) = (empty!(p.ns); empty!(p.bytes); p)

"Per-slot timing summary: slot ⇒ (calls, min_ns, median_ns, total_ns, bytes)."
function probe_summary(p::SlotProbe)
    out = Dict{Symbol,NamedTuple}()
    for (slot, samp) in p.ns
        s = sort(samp); n = length(s)
        out[slot] = (calls = n, min_ns = n == 0 ? 0 : s[1],
                     median_ns = n == 0 ? 0 : s[(n + 1) ÷ 2],
                     total_ns = sum(s; init = 0), bytes = get(p.bytes, slot, 0))
    end
    return out
end

"""
    EngineConfig(; hydro=:enzo, gravity=:off, cooling=:off, comoving_expansion=:off,
                 mhd_ct=:off, radiation=:off, star_formation=:off, star_sources=false,
                 hooks=Dict{Symbol,Function}())

Per-slot implementation map for the Julia-driven EvolveLevel. Each physics slot is
`:off | :enzo | :julia`; a `:julia` slot must have a matching entry in `hooks`
(a `(h, level, dt)` closure). `all_enzo()`-equivalent (every active slot `:enzo`)
is full replication and reproduces `EvolveHierarchy`.
"""
struct EngineConfig
    hydro::Symbol
    gravity::Symbol
    cooling::Symbol
    comoving_expansion::Symbol
    mhd_ct::Symbol
    radiation::Symbol
    star_formation::Symbol
    star_sources::Bool                  # radiation sub-flag (not a slot)
    hooks::Dict{Symbol,Function}
    probe::Union{Nothing,SlotProbe}     # nothing ⇒ no timing (zero overhead)
    reflux::Bool                        # :julia hydro fills Enzo's flux registers (ADR-0003 part B)
end
function EngineConfig(; hydro::Symbol = :enzo, gravity::Symbol = :off, cooling::Symbol = :off,
                      comoving_expansion::Symbol = :off, mhd_ct::Symbol = :off,
                      radiation::Symbol = :off, star_formation::Symbol = :off,
                      star_sources::Bool = false, hooks::Dict{Symbol,Function} = Dict{Symbol,Function}(),
                      probe::Union{Nothing,SlotProbe} = nothing, reflux::Bool = false)
    EngineConfig(hydro, gravity, cooling, comoving_expansion, mhd_ct, radiation,
                 star_formation, star_sources, hooks, probe, reflux)
end

# Build a config from the legacy boolean flags (every active slot → :enzo) so the
# existing flag-based callers keep their exact behaviour (full replication).
function engine_from_flags(; hydro::Symbol = :enzo, gravity::Bool = false, cooling::Bool = false,
                           radiation::Bool = false, star_sources::Bool = false,
                           star_formation::Bool = false, cosmology::Bool = false,
                           mhdct::Bool = false, hooks::Dict{Symbol,Function} = Dict{Symbol,Function}())
    EngineConfig(; hydro = hydro,
                 gravity = gravity ? :enzo : :off,
                 cooling = cooling ? :enzo : :off,
                 comoving_expansion = cosmology ? :enzo : :off,
                 mhd_ct = mhdct ? :enzo : :off,
                 radiation = radiation ? :enzo : :off,
                 star_formation = star_formation ? :enzo : :off,
                 star_sources = star_sources, hooks = hooks)
end

# The :enzo (legacy bridge) implementation of each single-call physics slot.
enzo_slot(::Val{:hydro}, h, level, dt, cfg) = session_solve_hydro(h, level)
enzo_slot(::Val{:gravity}, h, level, dt, cfg) = session_gravity(h, level)
enzo_slot(::Val{:cooling}, h, level, dt, cfg) = session_solve_cooling(h, level)
enzo_slot(::Val{:comoving_expansion}, h, level, dt, cfg) = session_comoving_expansion(h, level)
enzo_slot(::Val{:radiation}, h, level, dt, cfg) = session_evolve_photons(h, level; stars = cfg.star_sources)
enzo_slot(::Val{:star_formation}, h, level, dt, cfg) = session_star_particles(h, level)

# Run the resolved implementation of `slot` (impl is :enzo or :julia here).
@inline _slot_impl(slot, impl, cfg, h, level, dt) =
    impl === :julia ? cfg.hooks[slot](h, level, dt) : enzo_slot(Val(slot), h, level, dt, cfg)

# Dispatch a slot on (h, level, dt): :off no-op; :julia injected hook; :enzo bridge.
# When a probe is attached, time the call (sample pushed AFTER the timed window).
function run_slot(slot::Symbol, cfg::EngineConfig, h::Handle, level::Integer, dt::Float64)
    impl = getfield(cfg, slot)
    impl === :off && return nothing
    impl === :julia && !haskey(cfg.hooks, slot) &&
        error("EngineConfig: slot :$slot is :julia but no hook was provided")
    cfg.probe === nothing && return _slot_impl(slot, impl, cfg, h, level, dt)
    p = cfg.probe
    a0 = Base.gc_bytes(); t0 = time_ns()
    r = _slot_impl(slot, impl, cfg, h, level, dt)
    el = Int(time_ns() - t0); da = Int(Base.gc_bytes() - a0)
    push!(get!(() -> Int[], p.ns, slot), el)           # bookkeeping outside the timed window
    p.bytes[slot] = get(p.bytes, slot, 0) + da
    return r
end

# ── AMR: the recursive EvolveLevel, written in Julia on the certified steps ───
"""
    evolve_level!(h, level, dt_above; engine=EngineConfig(), regrid=true) -> ncycles

Julia reimplementation of Enzo's recursive `EvolveLevel`, built entirely from the
certified Session steps (mirrors EnzoModules' Python `evolve_level`): clear the
boundary fluxes, then sub-cycle this level to its parent's `dt_above` — each
sub-cycle solving the grids, recursing into level+1, then conservatively
flux-correcting + projecting (`update_from_finer`) on the way back up, regridding
finer levels between sub-cycles. `dt_above == 0` ⇒ a single (top-grid) step.
`hydro!(h, level, dt)` is the swappable hydro slot.
"""
function evolve_level!(h::Handle, level::Integer, dt_above::Float64;
                       engine::Union{EngineConfig,Nothing} = nothing, hydro! = nothing,
                       regrid::Bool = true, gravity::Bool = false, cooling::Bool = false,
                       radiation::Bool = false, star_sources::Bool = false,
                       star_formation::Bool = false, cosmology::Bool = false,
                       mhdct::Bool = false, maxsub::Int = 100000)
    eng = engine !== nothing ? engine :
          engine_from_flags(; hydro = hydro! === nothing ? :enzo : :julia,
                            gravity = gravity, cooling = cooling, radiation = radiation,
                            star_sources = star_sources, star_formation = star_formation,
                            cosmology = cosmology, mhdct = mhdct,
                            hooks = hydro! === nothing ? Dict{Symbol,Function}() :
                                    Dict{Symbol,Function}(:hydro => hydro!))
    rec(l, dta) = evolve_level!(h, l, dta; engine = eng, regrid = regrid, maxsub = maxsub)
    ct = eng.mhd_ct !== :off
    # Enzo's flux-register machinery (clear/create/finalize/project) is the hydro's
    # conservation bookkeeping — SolveHydroEquations fills SubgridFluxes, finalize/
    # project consume them. A plain :julia hydro slot does NOT fill them (so the
    # registers stay gated off, single-grid). But a CONSERVATIVE :julia slot
    # (engine.reflux=true, ADR-0003 part B) writes EnzoNG's recorded fluxes into the
    # registers from its hook, so the machinery runs and Enzo's UpdateFromFinerGrids/
    # CorrectForRefinedFluxes restore conservation across coarse–fine boundaries.
    ef = eng.hydro === :enzo || (eng.hydro === :julia && eng.reflux)
    ef && session_clear_boundary_fluxes(h, level)
    ct && level > 0 && session_clear_avg_electric_field(h, level)   # CT EMF accumulator (EvolveLevel.C:377)
    done = 0.0; n = 0
    while n < maxsub
        session_set_boundary(h, level)                       # interpolate from parent
        dt = session_compute_dt(h, level)
        dt_above > 0.0 && (dt = min(dt, dt_above - done))
        session_set_dt(h, dt, level)
        run_slot(:radiation, eng, h, level, dt)
        ef && session_create_fluxes(h, level)
        run_slot(:gravity, eng, h, level, dt)
        session_copy_baryon_to_old(h, level)
        run_slot(:hydro, eng, h, level, dt)                  # :enzo fills boundary fluxes; :julia owns its own
        run_slot(:cooling, eng, h, level, dt)
        run_slot(:star_formation, eng, h, level, dt)
        session_update_particles(h, level)
        session_advance_time(h, level)
        run_slot(:comoving_expansion, eng, h, level, dt)     # Hubble drag (EvolveLevel.C)
        last = dt_above <= 0.0 || done + dt >= dt_above * (1 - 1e-6)
        if session_num_grids_on_level(h, level + 1) > 0
            session_set_boundary(h, level)                   # refresh before projection
            rec(level + 1, dt)
            session_update_from_finer(h, level)              # project + flux-correct (+ MHD_ProjectFace for CT)
        end
        ct && session_mhd_update_magnetic_field(h, level)    # CT B from EMF (EvolveLevel.C:899, every level)
        ct && session_set_boundary(h, level)                 # UseMHDCT: refresh face-B ghosts (EvolveLevel.C:912)
        ef && session_finalize_fluxes(h, level)
        n += 1; done += dt
        (last || dt <= 0.0) && break
        regrid && session_rebuild(h, level)                  # regrid finer levels
    end
    return n
end

# All baryon fields of a grid as a Dict FieldType ⇒ flat field (incl. ghost zones).
function read_all_fields(h::Handle; grid::Integer = 0)
    ts = problem_field_types(h, grid)
    return Dict{Int,Vector{Float64}}(ts[i] => problem_get_field(h, i - 1, grid) for i in eachindex(ts))
end

"""
    run_amr(paramfile; reader=read_density, hydro!, regrid=true, gravity=…, cooling=…,
            radiation=…, star_sources=…, star_formation=…) -> reader(h)

Drive a full AMR run FROM JULIA — initial regrid, then root steps + regrid to
StopTime — and return `reader(handle)` (e.g. `read_density` or `read_all_fields`).
Mirrors `EvolveHierarchy` / Python `run_amr`; the physics flags enable the
corresponding certified Session steps each cycle.
"""
function run_amr(paramfile::AbstractString; reader = read_density,
                 engine::Union{EngineConfig,Nothing} = nothing, hydro! = nothing,
                 regrid::Bool = true, gravity::Bool = false, cooling::Bool = false,
                 radiation::Bool = false, star_sources::Bool = false,
                 star_formation::Bool = false, cosmology::Bool = false,
                 mhdct::Bool = false, maxcycle::Int = 100000)
    eng = engine !== nothing ? engine :
          engine_from_flags(; hydro = hydro! === nothing ? :enzo : :julia,
                            gravity = gravity, cooling = cooling, radiation = radiation,
                            star_sources = star_sources, star_formation = star_formation,
                            cosmology = cosmology, mhdct = mhdct,
                            hooks = hydro! === nothing ? Dict{Symbol,Function}() :
                                    Dict{Symbol,Function}(:hydro => hydro!))
    pf = abspath(paramfile)
    maxcycle = min(maxcycle, _stop_cycle(pf))    # honor the param file's StopCycle (as EvolveHierarchy does)
    cd(_workdir(pf)) do
        h = session_init(pf)
        h == C_NULL && error("session_init failed for $pf")
        try
            regrid && session_rebuild(h, 0)
            n = 0
            while session_time(h) < session_stop_time(h) && n < maxcycle
                evolve_level!(h, 0, 0.0; engine = eng, regrid = regrid)
                regrid && session_rebuild(h, 0)
                n += 1
            end
            return reader(h)
        finally
            free_problem(h)
        end
    end
end

"AMR run returning the root-grid Density (back-compat wrapper)."
run_amr_density(paramfile::AbstractString; kwargs...) = run_amr(paramfile; reader = read_density, kwargs...)
"AMR run returning ALL root-grid fields (Dict FieldType ⇒ vector)."
run_amr_fields(paramfile::AbstractString; kwargs...) = run_amr(paramfile; reader = read_all_fields, kwargs...)
"AMR run returning both root-grid fields and all particle positions: (fields, particles)."
run_amr_state(paramfile::AbstractString; kwargs...) = run_amr(paramfile; reader = read_state, kwargs...)

"Enzo's own `EvolveHierarchy` to StopTime, returning ALL root-grid fields — the reference."
function evolve_problem_fields(paramfile::AbstractString; grid::Integer = 0)
    pf = abspath(paramfile)
    cd(_workdir(pf)) do
        h = evolve_problem(pf, 0.0, 0)
        h == C_NULL && error("evolve_problem (EvolveHierarchy) failed for $pf")
        try
            return read_all_fields(h; grid = grid)
        finally
            free_problem(h)
        end
    end
end

"Enzo's own `EvolveHierarchy` to StopTime, returning (fields, particles) — the reference."
function evolve_problem_state(paramfile::AbstractString; grid::Integer = 0)
    pf = abspath(paramfile)
    cd(_workdir(pf)) do
        h = evolve_problem(pf, 0.0, 0)
        h == C_NULL && error("evolve_problem (EvolveHierarchy) failed for $pf")
        try
            return read_state(h; grid = grid)
        finally
            free_problem(h)
        end
    end
end
