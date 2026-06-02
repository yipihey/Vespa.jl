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
    pf = abspath(paramfile)
    cd(mktempdir()) do
        h = session_init(pf)
        h == C_NULL && error("session_init failed for $pf")
        try
            n = 0
            while session_time(h) < session_stop_time(h) && n < maxcycle
                session_set_boundary(h, level)
                session_set_dt(h, session_compute_dt(h, level), level)
                session_solve_hydro(h, level)
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
