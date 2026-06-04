"""
    EnzoLib

Native (`ccall`) Julia binding to the **EnzoModules C-ABI** — the stable
`extern "C"` surface (`EnzoModules/src/enzomodules_bridge.h`) over selected
legacy Enzo compute kernels, compiled into a shared library. This is the FFI
floor of the EnzoNG↔Enzo integration: it reuses the *reference legacy kernel*
directly so a Julia rewrite can run in full-replication mode and be certified
against it (see EnzoFixtures + the parity tests).

Pilot scope (no HDF5 / no full-Enzo build): the standalone Fortran kernels
`enzomodules_twoshock` and `enzomodules_ppm_sweep_1d` plus the precision
contract. The C++ `grid::`/Session surface (live hierarchy, `solve_hydro`, …)
links the full Enzo `.so` and is added once that build is available.

Build the library with `EnzoModules/deps/build_pilot.sh` (needs gfortran + g++).
`EnzoLib` locates `libenzomodules_pilot.so` next to that script, or via
`ENV["ENZOMODULES_LIB"]`.
"""
module EnzoLib

using Libdl

# ── library location + lazy handle ───────────────────────────────────────────
# The library path is runtime-determined (env / build location), so we cannot
# name it in a `ccall((:sym, "lib"), …)` literal inside a precompiled module.
# Instead dlopen once (lazily) and `ccall` through the dlsym function pointer.
"Absolute path to the EnzoModules shared library (env override, else the in-repo build)."
function libpath()
    env = get(ENV, "ENZOMODULES_LIB", "")
    isempty(env) || return abspath(env)
    return normpath(joinpath(@__DIR__, "..", "..", "..", "..",
                             "EnzoModules", "deps", "libenzomodules_pilot.so"))
end

"True when the shared library exists on disk (callers can skip live calls without it)."
available() = isfile(libpath())

const _HANDLE = Ref{Ptr{Cvoid}}(C_NULL)
function _handle()
    if _HANDLE[] == C_NULL
        available() || error("EnzoModules library not found at $(libpath()). " *
                             "Build it with EnzoModules/deps/build_pilot.sh, or set ENV[\"ENZOMODULES_LIB\"].")
        _HANDLE[] = Libdl.dlopen(libpath())
    end
    return _HANDLE[]
end
@inline _sym(name::Symbol) = Libdl.dlsym(_handle(), name)

# ── precision contract ───────────────────────────────────────────────────────
"""
    check_precision() -> (baryon_bytes, int_bytes)

Verify the library was built with the precision these bindings assume (8-byte
baryons, 4-byte Fortran ints). Raises on mismatch — precisely the class of silent
bug this framework exists to surface. Mirrors `bridge.check_precision()`.
"""
function check_precision()
    rb = ccall(_sym(:enzomodules_baryon_precision_bytes), Cint, ())
    ib = ccall(_sym(:enzomodules_int_precision_bytes), Cint, ())
    rb == sizeof(Cdouble) ||
        error("EnzoModules built with $rb-byte baryons; bindings assume $(sizeof(Cdouble)).")
    ib == sizeof(Cint) ||
        error("EnzoModules built with $ib-byte Fortran ints; bindings assume $(sizeof(Cint)).")
    return (Int(rb), Int(ib))
end

# ── twoshock: two-shock approximate Riemann solver (twoshock.F) ──────────────
"""
    twoshock(dls, drs, pls, prs, uls, urs; idim, jdim=1, i1, i2, j1=1, j2=1,
             dt, gamma, pmin=1e-20, ipresfree=0, gravity=0, grslice=zeros(idim*jdim),
             idual=0, eta1=0.0) -> (pbar, ubar)

Resolve interface pressure/normal-velocity for left/right reconstructed states
over a column-major `idim×jdim` slab (1-based inclusive bounds). `pls`/`prs` may
be overwritten by the kernel (legacy in/out semantics). Returns the resolved
`(pbar, ubar)` as new vectors.
"""
function twoshock(dls, drs, pls, prs, uls, urs;
                  idim::Integer, jdim::Integer = 1,
                  i1::Integer, i2::Integer, j1::Integer = 1, j2::Integer = 1,
                  dt::Real, gamma::Real, pmin::Real = 1e-20, ipresfree::Integer = 0,
                  gravity::Integer = 0, grslice = zeros(Float64, idim * jdim),
                  idual::Integer = 0, eta1::Real = 0.0)
    n = idim * jdim
    d(v) = Vector{Float64}(v)
    dls, drs, pls, prs, uls, urs, grslice = d(dls), d(drs), d(pls), d(prs), d(uls), d(urs), d(grslice)
    pbar = zeros(Float64, n); ubar = zeros(Float64, n)
    ccall(_sym(:enzomodules_twoshock), Cvoid,
          (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},
           Cint, Cint, Cint, Cint, Cint, Cint,
           Cdouble, Cdouble, Cdouble, Cint,
           Ptr{Cdouble}, Ptr{Cdouble},
           Cint, Ptr{Cdouble}, Cint, Cdouble),
          dls, drs, pls, prs, uls, urs,
          idim, jdim, i1, i2, j1, j2,
          dt, gamma, pmin, ipresfree,
          pbar, ubar,
          gravity, grslice, idual, eta1)
    return pbar, ubar
end

# ── ppm_sweep_1d: one directional PPM hydro update of a 1D slice ─────────────
"""
    ppm_sweep_1d!(dslice, eslice, uslice, vslice, wslice, pslice;
                  i1, i2, dx, dt, gamma, fluxes=false) -> nothing | (df, ef, uf)

The legacy Enzo Eulerian-PPM directional sweep (`inteuler → twoshock →
flux_twoshock → euler`, the numerical core of `Grid_xEulerSweep.C`) — no gravity,
dual energy or colour. Updates the six slices **in place** with the post-`dt`
state. Slices are 1-based Fortran slabs of length `idim = length(dslice)`; `i1..i2`
are the active (non-ghost) cells (PPM needs ≥3 ghosts each side). `pslice` is the
caller-precomputed pressure. With `fluxes=true` returns the density/energy/
normal-momentum face fluxes `(df, ef, uf)`.
"""
function ppm_sweep_1d!(dslice::Vector{Float64}, eslice::Vector{Float64},
                       uslice::Vector{Float64}, vslice::Vector{Float64},
                       wslice::Vector{Float64}, pslice::Vector{Float64};
                       i1::Integer, i2::Integer, dx::Real, dt::Real, gamma::Real,
                       fluxes::Bool = false)
    idim = length(dslice)
    all(==(idim) ∘ length, (eslice, uslice, vslice, wslice, pslice)) ||
        throw(ArgumentError("all slices must have length idim=$idim"))
    df = fluxes ? zeros(Float64, idim) : Vector{Float64}()
    ef = fluxes ? zeros(Float64, idim) : Vector{Float64}()
    uf = fluxes ? zeros(Float64, idim) : Vector{Float64}()
    pdf = fluxes ? pointer(df) : Ptr{Cdouble}(C_NULL)
    pef = fluxes ? pointer(ef) : Ptr{Cdouble}(C_NULL)
    puf = fluxes ? pointer(uf) : Ptr{Cdouble}(C_NULL)
    GC.@preserve df ef uf begin
        ret = ccall(_sym(:enzomodules_ppm_sweep_1d), Cint,
                    (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},
                     Cint, Cint, Cint, Cdouble, Cdouble, Cdouble,
                     Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}),
                    dslice, eslice, uslice, vslice, wslice, pslice,
                    idim, i1, i2, dx, dt, gamma, pdf, pef, puf)
        ret == 0 || error("enzomodules_ppm_sweep_1d returned $ret")
    end
    return fluxes ? (df, ef, uf) : nothing
end

include("session.jl")   # live-Session C-ABI (full-replication via the Enzo grid lib)
include("rpc.jl")       # ADR-0005 #2: :remote transport (subprocess worker + shm)

end # module
