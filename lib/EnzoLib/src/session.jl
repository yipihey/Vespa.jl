# Native ccall binding to the EnzoModules **live-Session** C-ABI (the grid bridge
# library, libenzomodules_grid, which links the full Enzo .so). This exposes
# Enzo's actual time-integration steps on a live hierarchy, so a Julia-driven
# EvolveLevel can call the certified legacy reference for each step — the
# full-replication mode. Mirrors EnzoModules/enzomodules/problems.py (the Session
# the README proves reproduces EvolveHierarchy bit-for-bit).
#
# Build the library with EnzoModules/deps/build_grid_darwin.sh (macOS) /
# build_grid.sh (Linux); located via ENV["ENZOMODULES_GRID_LIB"] or deps/.

"Path to the EnzoModules grid/session bridge library (links the full Enzo .so)."
function grid_libpath()
    env = get(ENV, "ENZOMODULES_GRID_LIB", "")
    isempty(env) || return abspath(env)
    base = normpath(joinpath(@__DIR__, "..", "..", "..", "..", "EnzoModules", "deps"))
    for name in ("libenzomodules_grid.dylib", "libenzomodules_grid.so")
        p = joinpath(base, name); isfile(p) && return p
    end
    return joinpath(base, "libenzomodules_grid.dylib")
end
grid_available() = isfile(grid_libpath())

const _GHANDLE = Ref{Ptr{Cvoid}}(C_NULL)
function _ghandle()
    if _GHANDLE[] == C_NULL
        grid_available() || error("grid/session library not found at $(grid_libpath()). " *
                                  "Build it: bash EnzoModules/deps/build_grid_darwin.sh")
        _GHANDLE[] = Libdl.dlopen(grid_libpath())
    end
    return _GHANDLE[]
end
@inline _gsym(name::Symbol) = Libdl.dlsym(_ghandle(), name)

# ── low-level Session / problem entry points (signatures per problems.py) ─────
const Handle = Ptr{Cvoid}

"Initialize a problem and hold the LIVE hierarchy (InitializeNew). Returns a handle."
session_init(paramfile::AbstractString) =
    ccall(_gsym(:enzomodules_session_init), Handle, (Cstring,), paramfile)
"Run Enzo's full time integrator (EvolveHierarchy) to stop_time/stop_cycle (0 ⇒ param file)."
evolve_problem(paramfile::AbstractString, stop_time::Real = 0.0, stop_cycle::Integer = 0) =
    ccall(_gsym(:enzomodules_evolve_problem), Handle, (Cstring, Cdouble, Cint),
          paramfile, stop_time, stop_cycle)
free_problem(h::Handle) = ccall(_gsym(:enzomodules_free_problem), Cvoid, (Handle,), h)

session_time(h::Handle)      = ccall(_gsym(:enzomodules_session_time), Cdouble, (Handle,), h)
session_stop_time(h::Handle) = ccall(_gsym(:enzomodules_session_stop_time), Cdouble, (Handle,), h)
session_cycle(h::Handle)     = ccall(_gsym(:enzomodules_session_cycle), Cint, (Handle,), h)
session_compute_dt(h::Handle, level::Integer = 0) =
    ccall(_gsym(:enzomodules_session_compute_dt), Cdouble, (Handle, Cint), h, level)
session_set_dt(h::Handle, dt::Real, level::Integer = 0) =
    ccall(_gsym(:enzomodules_session_set_dt), Cvoid, (Handle, Cint, Cdouble), h, level, dt)
session_set_boundary(h::Handle, level::Integer = 0) =
    ccall(_gsym(:enzomodules_session_set_boundary), Cint, (Handle, Cint), h, level)
session_solve_hydro(h::Handle, level::Integer = 0) =
    ccall(_gsym(:enzomodules_session_solve_hydro), Cint, (Handle, Cint), h, level)
session_advance_time(h::Handle, level::Integer = 0) =
    ccall(_gsym(:enzomodules_session_advance_time), Cvoid, (Handle, Cint), h, level)

# AMR EvolveLevel steps (the conservation machinery): flux storage, projection +
# flux-correction from finer levels, and regridding.
session_clear_boundary_fluxes(h::Handle, level::Integer = 0) =
    ccall(_gsym(:enzomodules_session_clear_boundary_fluxes), Cvoid, (Handle, Cint), h, level)
session_create_fluxes(h::Handle, level::Integer = 0) =
    ccall(_gsym(:enzomodules_session_create_fluxes), Cint, (Handle, Cint), h, level)
session_finalize_fluxes(h::Handle, level::Integer = 0) =
    ccall(_gsym(:enzomodules_session_finalize_fluxes), Cint, (Handle, Cint), h, level)
session_update_from_finer(h::Handle, level::Integer = 0) =
    ccall(_gsym(:enzomodules_session_update_from_finer), Cint, (Handle, Cint), h, level)
session_copy_baryon_to_old(h::Handle, level::Integer = 0) =
    ccall(_gsym(:enzomodules_session_copy_baryon_to_old), Cvoid, (Handle, Cint), h, level)
session_update_particles(h::Handle, level::Integer = 0) =
    ccall(_gsym(:enzomodules_session_update_particles), Cvoid, (Handle, Cint), h, level)
session_rebuild(h::Handle, level::Integer = 0) =
    ccall(_gsym(:enzomodules_session_rebuild), Cint, (Handle, Cint), h, level)
session_num_grids_on_level(h::Handle, level::Integer) =
    ccall(_gsym(:enzomodules_session_num_grids_on_level), Cint, (Handle, Cint), h, level)

problem_num_grids(h::Handle) = ccall(_gsym(:enzomodules_problem_num_grids), Cint, (Handle,), h)
problem_grid_size(h::Handle, grid::Integer = 0) =
    ccall(_gsym(:enzomodules_problem_grid_size), Cint, (Handle, Cint), h, grid)
problem_num_fields(h::Handle, grid::Integer = 0) =
    ccall(_gsym(:enzomodules_problem_num_fields), Cint, (Handle, Cint), h, grid)
function problem_field_types(h::Handle, grid::Integer = 0)
    n = problem_num_fields(h, grid)
    t = zeros(Cint, n)
    ccall(_gsym(:enzomodules_problem_field_types), Cint, (Handle, Cint, Ptr{Cint}), h, grid, t)
    return Int.(t)
end
"Read one field (flat, incl. ghost zones) by 0-based field index."
function problem_get_field(h::Handle, fi::Integer, grid::Integer = 0)
    out = zeros(Float64, problem_grid_size(h, grid))
    ccall(_gsym(:enzomodules_problem_get_field), Cint, (Handle, Cint, Cint, Ptr{Cdouble}),
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
    ccall(_gsym(:enzomodules_problem_set_field), Cvoid, (Handle, Cint, Cint, Ptr{Cdouble}),
          h, grid, fi, data)
    return nothing
end

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
    cd(mktempdir()) do
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
    cd(mktempdir()) do
        h = evolve_problem(pf, 0.0, 0)
        h == C_NULL && error("evolve_problem (EvolveHierarchy) failed for $pf")
        try
            return read_density(h)
        finally
            free_problem(h)
        end
    end
end

# ── AMR: the recursive EvolveLevel, written in Julia on the certified steps ───
"""
    evolve_level!(h, level, dt_above; hydro!, regrid=true) -> ncycles

Julia reimplementation of Enzo's recursive `EvolveLevel`, built entirely from the
certified Session steps (mirrors EnzoModules' Python `evolve_level`): clear the
boundary fluxes, then sub-cycle this level to its parent's `dt_above` — each
sub-cycle solving the grids, recursing into level+1, then conservatively
flux-correcting + projecting (`update_from_finer`) on the way back up, regridding
finer levels between sub-cycles. `dt_above == 0` ⇒ a single (top-grid) step.
`hydro!(h, level, dt)` is the swappable hydro slot.
"""
function evolve_level!(h::Handle, level::Integer, dt_above::Float64;
                       hydro! = (hh, l, dt) -> session_solve_hydro(hh, l),
                       regrid::Bool = true, maxsub::Int = 100000)
    session_clear_boundary_fluxes(h, level)
    done = 0.0; n = 0
    while n < maxsub
        session_set_boundary(h, level)                       # interpolate from parent
        dt = session_compute_dt(h, level)
        dt_above > 0.0 && (dt = min(dt, dt_above - done))
        session_set_dt(h, dt, level)
        session_create_fluxes(h, level)
        session_copy_baryon_to_old(h, level)
        hydro!(h, level, dt)                                 # fills boundary fluxes
        session_update_particles(h, level)
        session_advance_time(h, level)
        last = dt_above <= 0.0 || done + dt >= dt_above * (1 - 1e-6)
        if session_num_grids_on_level(h, level + 1) > 0
            session_set_boundary(h, level)                   # refresh before projection
            evolve_level!(h, level + 1, dt; hydro! = hydro!, regrid = regrid)
            session_update_from_finer(h, level)              # project + flux-correct
        end
        session_finalize_fluxes(h, level)
        n += 1; done += dt
        (last || dt <= 0.0) && break
        regrid && session_rebuild(h, level)                  # regrid finer levels
    end
    return n
end

"""
    run_amr_density(paramfile; hydro!, regrid=true) -> Vector{Float64}

Drive a full AMR run FROM JULIA — an initial regrid, then repeatedly evolve the
whole hierarchy one root step (`evolve_level!(h,0,0)`) and regrid, to StopTime —
and return the root-grid Density. Mirrors `EvolveHierarchy`/Python `run_amr`.
`hydro!` defaults to the legacy `session_solve_hydro` (drives Enzo's hydro under
the Julia-owned AMR control flow).
"""
function run_amr_density(paramfile::AbstractString;
                         hydro! = (hh, l, dt) -> session_solve_hydro(hh, l),
                         regrid::Bool = true, maxcycle::Int = 100000)
    pf = abspath(paramfile)
    cd(mktempdir()) do
        h = session_init(pf)
        h == C_NULL && error("session_init failed for $pf")
        try
            regrid && session_rebuild(h, 0)
            n = 0
            while session_time(h) < session_stop_time(h) && n < maxcycle
                evolve_level!(h, 0, 0.0; hydro! = hydro!, regrid = regrid)
                regrid && session_rebuild(h, 0)
                n += 1
            end
            return read_density(h)                           # root grid
        finally
            free_problem(h)
        end
    end
end
