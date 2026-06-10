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
using PPMKernels
using CodeBridge
using CodeBridge: @xcall          # the call sites in session.jl resolve to CodeBridge's macro

# ── library location + lazy handle (CodeBridge.LazyLib) ──────────────────────
# The library path is runtime-determined (env / build location), so we cannot
# name it in a `ccall((:sym, "lib"), …)` literal inside a precompiled module.
# CodeBridge dlopens once (lazily) and we `ccall` through the dlsym pointer.
const PILOT_LIB = CodeBridge.LazyLib(
    env = "ENZOMODULES_LIB",
    default = normpath(joinpath(@__DIR__, "..", "..", "..", "..",
                                "EnzoModules", "deps", "libenzomodules_pilot.so")),
    hint = "Build it with EnzoModules/deps/build_pilot.sh, or set ENV[\"ENZOMODULES_LIB\"].")

"Absolute path to the EnzoModules shared library (env override, else the in-repo build)."
libpath() = CodeBridge.libpath(PILOT_LIB)

"True when the shared library exists on disk (callers can skip live calls without it)."
available() = CodeBridge.available(PILOT_LIB)

@inline _sym(name::Symbol) = CodeBridge.sym(PILOT_LIB, name)

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

# ── pgas2d: gas pressure from total energy, EOS (pgas2d.F) ───────────────────
"""
    pgas2d(dslice, eslice, uslice, vslice, wslice; idim, jdim=1, i1, i2, j1=1, j2=1,
           gamma, pmin=1e-20) -> pslice

Gas pressure on a column-major `idim×jdim` slab via the ideal-gas EOS
`p = (γ-1)·d·(E − ½(u²+v²+w²))`, floored at `pmin`, over the active region
`i1..i2 × j1..j2` (1-based inclusive). Purely local (no cross-cell coupling).
Returns a fresh `pslice`; inputs are not modified. The golden reference for the
PPMKernels `pgas2d!` port.
"""
function pgas2d(dslice, eslice, uslice, vslice, wslice;
                idim::Integer, jdim::Integer = 1,
                i1::Integer, i2::Integer, j1::Integer = 1, j2::Integer = 1,
                gamma::Real, pmin::Real = 1e-20)
    n = idim * jdim
    d(v) = Vector{Float64}(v)
    dslice, eslice, uslice, vslice, wslice = d(dslice), d(eslice), d(uslice), d(vslice), d(wslice)
    pslice = zeros(Float64, n)
    ccall(_sym(:enzomodules_pgas2d), Cvoid,
          (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},
           Cint, Cint, Cint, Cint, Cint, Cint, Cdouble, Cdouble),
          dslice, eslice, pslice, uslice, vslice, wslice,
          idim, jdim, i1, i2, j1, j2, gamma, pmin)
    return pslice
end

# ── pgas2d_dual: gas pressure under the dual-energy formalism (pgas2d_dual.F) ─
"""
    pgas2d_dual(dslice, eslice, geslice, uslice, vslice, wslice; eta1, eta2,
                idim, jdim=1, i1, i2, j1=1, j2=1, gamma, pmin=1e-20)
        -> (eslice_out, geslice_out, pslice_out)

Dual-energy pressure: reconciles the gas energy `geslice` against the total
energy `eslice` per cell (selection parameters `eta1`, `eta2`), updating BOTH
in place, then forms the pressure. Carries a left-to-right sweep dependency — a
cell's `demax = max(d·E)` over `{i-1,i,i+1}` reads the *already-updated* `eslice`
of its left neighbor. Inputs are copied internally; returns the three updated
slices. The golden reference for the PPMKernels `pgas2d_dual!` port.
"""
function pgas2d_dual(dslice, eslice, geslice, uslice, vslice, wslice;
                     eta1::Real, eta2::Real, idim::Integer, jdim::Integer = 1,
                     i1::Integer, i2::Integer, j1::Integer = 1, j2::Integer = 1,
                     gamma::Real, pmin::Real = 1e-20)
    n = idim * jdim
    d(v) = Vector{Float64}(v)
    dslice, eslice, geslice, uslice, vslice, wslice =
        d(dslice), d(eslice), d(geslice), d(uslice), d(vslice), d(wslice)
    pslice = zeros(Float64, n)
    ccall(_sym(:enzomodules_pgas2d_dual), Cvoid,
          (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},
           Cdouble, Cdouble, Cint, Cint, Cint, Cint, Cint, Cint, Cdouble, Cdouble),
          dslice, eslice, geslice, pslice, uslice, vslice, wslice,
          eta1, eta2, idim, jdim, i1, i2, j1, j2, gamma, pmin)
    return eslice, geslice, pslice
end

# ── calcdiss: PPM diffusion coefficient + slope flattening (calcdiss.F) ──────
"""
    calcdiss(dslice, eslice, uslice, pslice, v, w; idim, jdim=1, kdim=1,
             i1, i2, j1=1, j2=1, k=1, nzz=1, idir=1, dimx=idim, dimy=1, dimz=1,
             dx=ones(idim), dy=ones(jdim), dz=ones(kdim), dt, gamma,
             idiff, iflatten) -> (diffcoef, flatten)

Colella–Woodward diffusion coefficients (`idiff`) and slope-flattening (`iflatten`)
for one slice. `v`/`w` are the (possibly 3-D, `dimx·dimy·dimz`) transverse
velocity fields. Returns fresh `diffcoef`/`flatten` slabs (`idim·jdim`); inputs
are not modified. The golden reference for the PPMKernels `calcdiss!` port —
which targets the transverse-free 1-D regime (`dimy=dimz=1`).
"""
function calcdiss(dslice, eslice, uslice, pslice, v, w;
                  idim::Integer, jdim::Integer = 1, kdim::Integer = 1,
                  i1::Integer, i2::Integer, j1::Integer = 1, j2::Integer = 1,
                  k::Integer = 1, nzz::Integer = 1, idir::Integer = 1,
                  dimx::Integer = idim, dimy::Integer = 1, dimz::Integer = 1,
                  dx = ones(Float64, idim), dy = ones(Float64, jdim),
                  dz = ones(Float64, kdim), dt::Real, gamma::Real,
                  idiff::Integer, iflatten::Integer)
    n = idim * jdim
    d(x) = Vector{Float64}(x)
    dslice, eslice, uslice, pslice = d(dslice), d(eslice), d(uslice), d(pslice)
    v, w, dx, dy, dz = d(v), d(w), d(dx), d(dy), d(dz)
    diffcoef = zeros(Float64, n); flatten = zeros(Float64, n)
    ccall(_sym(:enzomodules_calcdiss), Cvoid,
          (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},
           Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},
           Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint,
           Cint, Cint, Cint, Cdouble, Cdouble, Cint, Cint,
           Ptr{Cdouble}, Ptr{Cdouble}),
          dslice, eslice, uslice, v, w, pslice, dx, dy, dz,
          idim, jdim, kdim, i1, i2, j1, j2, k, nzz, idir,
          dimx, dimy, dimz, dt, gamma, idiff, iflatten,
          diffcoef, flatten)
    return diffcoef, flatten
end

# ── inteuler: PPM Eulerian left/right interface states (inteuler.F) ───────────
"""
    inteuler(dslice, pslice, uslice, vslice, wslice; idim, jdim=1, i1, i2, j1=1, j2=1,
             dt, gamma, geslice=zeros(idim*jdim), grslice=zeros(idim*jdim),
             dxi=ones(idim), flatten=zeros(idim*jdim), gravity=0, idual=0, eta1=0.0,
             eta2=0.0, isteep=0, iflatten=0, iconsrec=0, iposrec=0, ipresfree=0)
        -> NamedTuple (dls,drs,pls,prs,gels,gers,uls,urs,vls,vrs,wls,wrs)

PPM parabolic reconstruction → characteristic-corrected interface states. Returns
fresh output slabs; inputs are not modified. `ncolor=0` (no colour advection).
The golden reference for the PPMKernels `inteuler!` port.
"""
function inteuler(dslice, pslice, uslice, vslice, wslice;
                  idim::Integer, jdim::Integer = 1, i1::Integer, i2::Integer,
                  j1::Integer = 1, j2::Integer = 1, dt::Real, gamma::Real,
                  geslice = zeros(Float64, idim * jdim), grslice = zeros(Float64, idim * jdim),
                  dxi = ones(Float64, idim), flatten = zeros(Float64, idim * jdim),
                  gravity::Integer = 0, idual::Integer = 0, eta1::Real = 0.0, eta2::Real = 0.0,
                  isteep::Integer = 0, iflatten::Integer = 0, iconsrec::Integer = 0,
                  iposrec::Integer = 0, ipresfree::Integer = 0)
    n = idim * jdim
    d(x) = Vector{Float64}(x)
    dslice, pslice, uslice, vslice, wslice = d(dslice), d(pslice), d(uslice), d(vslice), d(wslice)
    geslice, grslice, dxi, flatten = d(geslice), d(grslice), d(dxi), d(flatten)
    o = (; (k => zeros(Float64, n) for k in
            (:dls, :drs, :pls, :prs, :gels, :gers, :uls, :urs, :vls, :vrs, :wls, :wrs))...)
    ccall(_sym(:enzomodules_inteuler), Cvoid,
          (Ptr{Cdouble}, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},
           Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},
           Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cdouble, Cdouble,
           Cint, Cint, Cint, Cint, Cdouble, Cdouble, Cint,
           Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},
           Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},
           Cint, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}),
          dslice, pslice, gravity, grslice, geslice, uslice, vslice, wslice, dxi, flatten,
          idim, jdim, i1, i2, j1, j2, idual, eta1, eta2,
          isteep, iflatten, iconsrec, iposrec, dt, gamma, ipresfree,
          o.dls, o.drs, o.pls, o.prs, o.gels, o.gers, o.uls, o.urs, o.vls, o.vrs, o.wls, o.wrs,
          0, C_NULL, C_NULL, C_NULL)
    return o
end

# ── flux_twoshock: Eulerian fluxes from the resolved interface states ────────
"""
    flux_twoshock(dslice, eslice, geslice, uslice, vslice, wslice,
                  dls, drs, pls, prs, gels, gers, uls, urs, vls, vrs, wls, wrs,
                  pbar, ubar; idim, jdim=1, i1, i2, j1=1, j2=1, dt, gamma,
                  dx=ones(idim), diffcoef=zeros(idim*jdim), idiff=0, idual=0,
                  eta1=0.0, ifallback=0) -> NamedTuple (df,ef,uf,vf,wf,gef,ges)

Time-averaged Eulerian fluxes for the two-shock solver. Returns fresh flux slabs
(`gef`/`ges` meaningful only when `idual=1`); inputs are not modified. `ncolor=0`.
The golden reference for the PPMKernels `flux_twoshock!` port.
"""
function flux_twoshock(dslice, eslice, geslice, uslice, vslice, wslice,
                       dls, drs, pls, prs, gels, gers, uls, urs, vls, vrs, wls, wrs,
                       pbar, ubar; idim::Integer, jdim::Integer = 1, i1::Integer, i2::Integer,
                       j1::Integer = 1, j2::Integer = 1, dt::Real, gamma::Real,
                       dx = ones(Float64, idim), diffcoef = zeros(Float64, idim * jdim),
                       idiff::Integer = 0, idual::Integer = 0, eta1::Real = 0.0,
                       ifallback::Integer = 0)
    n = idim * jdim
    d(x) = Vector{Float64}(x)
    ins = d.((dslice, eslice, geslice, uslice, vslice, wslice, dls, drs, pls, prs,
              gels, gers, uls, urs, vls, vrs, wls, wrs, pbar, ubar, dx, diffcoef))
    o = (; (k => zeros(Float64, n) for k in (:df, :ef, :uf, :vf, :wf, :gef, :ges))...)
    ccall(_sym(:enzomodules_flux_twoshock), Cvoid,
          (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},
           Ptr{Cdouble}, Ptr{Cdouble}, Cint, Cint, Cint, Cint, Cint, Cint,
           Cdouble, Cdouble, Cint, Cint, Cdouble, Cint,
           Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},
           Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},
           Ptr{Cdouble}, Ptr{Cdouble},
           Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},
           Ptr{Cdouble}, Ptr{Cdouble},
           Cint, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}),
          ins[1], ins[2], ins[3], ins[4], ins[5], ins[6], ins[21], ins[22],
          idim, jdim, i1, i2, j1, j2, dt, gamma, idiff, idual, eta1, ifallback,
          ins[7], ins[8], ins[9], ins[10], ins[11], ins[12], ins[13], ins[14],
          ins[15], ins[16], ins[17], ins[18], ins[19], ins[20],
          o.df, o.ef, o.uf, o.vf, o.wf, o.gef, o.ges,
          0, C_NULL, C_NULL, C_NULL, C_NULL)
    return o
end

# ── euler: conservative flux-divergence update of the zone-centred state ──────
"""
    euler(dslice, eslice, geslice, uslice, vslice, wslice, grslice,
          df, ef, uf, vf, wf, gef, ges; idim, jdim=1, i1, i2, j1=1, j2=1, dt, gamma,
          dx=ones(idim), diffcoef=zeros(idim*jdim), idiff=0, gravity=0, idual=0,
          eta1=0.0, eta2=0.0, dfloor=0.0)
        -> NamedTuple (dslice,eslice,geslice,uslice,vslice,wslice)

Update the zone-centred state by the flux divergence (eq. 3.1) plus the
dual-energy and gravity source terms. Returns the updated slices (fresh copies).
The golden reference for the PPMKernels `euler!` port.
"""
function euler(dslice, eslice, geslice, uslice, vslice, wslice, grslice,
               df, ef, uf, vf, wf, gef, ges; idim::Integer, jdim::Integer = 1,
               i1::Integer, i2::Integer, j1::Integer = 1, j2::Integer = 1,
               dt::Real, gamma::Real, dx = ones(Float64, idim),
               diffcoef = zeros(Float64, idim * jdim), idiff::Integer = 0,
               gravity::Integer = 0, idual::Integer = 0, eta1::Real = 0.0,
               eta2::Real = 0.0, dfloor::Real = 0.0)
    d(x) = Vector{Float64}(x)
    ds, es, ge, us, vs, ws, gr = d(dslice), d(eslice), d(geslice), d(uslice), d(vslice), d(wslice), d(grslice)
    dfA, efA, ufA, vfA, wfA, gefA, gesA = d(df), d(ef), d(uf), d(vf), d(wf), d(gef), d(ges)
    dxA, dc = d(dx), d(diffcoef)
    ccall(_sym(:enzomodules_euler), Cvoid,
          (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},
           Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},
           Cint, Cint, Cint, Cint, Cint, Cint, Cdouble, Cdouble, Cint, Cint,
           Cint, Cdouble, Cdouble,
           Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},
           Ptr{Cdouble}, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cdouble),
          ds, es, gr, ge, us, vs, ws, dxA, dc,
          idim, jdim, i1, i2, j1, j2, dt, gamma, idiff, gravity,
          idual, eta1, eta2, dfA, efA, ufA, vfA, wfA, gefA, gesA,
          0, C_NULL, C_NULL, dfloor)
    return (; dslice = ds, eslice = es, geslice = ge, uslice = us, vslice = vs, wslice = ws)
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

# ── ppm_sweep_1d_full!: production directional sweep (dual energy + gravity + colour) ──
"""
    ppm_sweep_1d_full!(dslice, eslice, uslice, vslice, wslice, pslice;
                       i1, i2, dx, dt, gamma,
                       geslice=nothing, grslice=nothing, gravity=0,
                       idual=0, eta1=0.0, eta2=0.0,
                       isteep=0, iflatten=0, iconsrec=0, iposrec=0,
                       idiff=0, ipresfree=0, ifallback=0,
                       pmin=1e-20, dfloor=1e-20,
                       ncolor=0, colslice=nothing, fluxes=false)
        -> nothing | (df, ef, uf)

The full production Enzo PPM directional sweep (`calcdiss → inteuler → twoshock →
flux_twoshock → euler`) with the dual-energy formalism, gravity, colour advection
and slope-flattening/diffusion exposed as parameters — the golden reference the
composed Metal/KA sweep is certified against. Generalises [`ppm_sweep_1d!`](@ref):
with every feature off it is bitwise-identical to it.

`dslice/eslice/uslice/vslice/wslice` and (when enabled) `geslice`/`colslice` are
updated **in place**. `geslice` (gas energy) is required when `idual≠0`; `grslice`
(gravity accel) when `gravity≠0`; `colslice` (length `idim*ncolor`) when `ncolor>0`.
"""
function ppm_sweep_1d_full!(dslice::Vector{Float64}, eslice::Vector{Float64},
                            uslice::Vector{Float64}, vslice::Vector{Float64},
                            wslice::Vector{Float64}, pslice::Vector{Float64};
                            i1::Integer, i2::Integer, dx::Real, dt::Real, gamma::Real,
                            geslice::Union{Nothing,Vector{Float64}} = nothing,
                            grslice::Union{Nothing,Vector{Float64}} = nothing,
                            gravity::Integer = 0, idual::Integer = 0,
                            eta1::Real = 0.0, eta2::Real = 0.0,
                            isteep::Integer = 0, iflatten::Integer = 0,
                            iconsrec::Integer = 0, iposrec::Integer = 0,
                            idiff::Integer = 0, ipresfree::Integer = 0,
                            ifallback::Integer = 0, pmin::Real = 1e-20,
                            dfloor::Real = 1e-20, ncolor::Integer = 0,
                            colslice::Union{Nothing,Vector{Float64}} = nothing,
                            fluxes::Bool = false)
    idim = length(dslice)
    all(==(idim) ∘ length, (eslice, uslice, vslice, wslice, pslice)) ||
        throw(ArgumentError("all slices must have length idim=$idim"))
    df = fluxes ? zeros(Float64, idim) : Vector{Float64}()
    ef = fluxes ? zeros(Float64, idim) : Vector{Float64}()
    uf = fluxes ? zeros(Float64, idim) : Vector{Float64}()
    pge = geslice === nothing ? Ptr{Cdouble}(C_NULL) : pointer(geslice)
    pgr = grslice === nothing ? Ptr{Cdouble}(C_NULL) : pointer(grslice)
    pcol = colslice === nothing ? Ptr{Cdouble}(C_NULL) : pointer(colslice)
    pdf = fluxes ? pointer(df) : Ptr{Cdouble}(C_NULL)
    pef = fluxes ? pointer(ef) : Ptr{Cdouble}(C_NULL)
    puf = fluxes ? pointer(uf) : Ptr{Cdouble}(C_NULL)
    GC.@preserve df ef uf geslice grslice colslice begin
        ret = ccall(_sym(:enzomodules_ppm_sweep_1d_full), Cint,
                    (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},
                     Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},
                     Cint, Cint, Cint, Cdouble, Cdouble, Cdouble,
                     Cint, Ptr{Cdouble},
                     Cint, Cdouble, Cdouble,
                     Cint, Cint, Cint, Cint,
                     Cint, Cint, Cint,
                     Cdouble, Cdouble,
                     Cint, Ptr{Cdouble},
                     Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}),
                    dslice, eslice, pge, uslice, vslice, wslice, pslice,
                    idim, i1, i2, dx, dt, gamma,
                    gravity, pgr,
                    idual, eta1, eta2,
                    isteep, iflatten, iconsrec, iposrec,
                    idiff, ipresfree, ifallback,
                    pmin, dfloor,
                    ncolor, pcol,
                    pdf, pef, puf)
        ret == 0 || error("enzomodules_ppm_sweep_1d_full returned $ret")
    end
    return fluxes ? (df, ef, uf) : nothing
end

include("session.jl")   # live-Session C-ABI (full-replication via the Enzo grid lib)
include("local_ppm.jl") # HydroMethod=10: conservative one-ghost local PPM
include("rpc.jl")       # ADR-0005 #2: :remote transport (subprocess worker + shm)

end # module
