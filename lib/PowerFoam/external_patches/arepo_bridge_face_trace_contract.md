# AREPO Bridge Face-Trace Contract

This is the external bridge contract needed for the next PowerFoam physics
parity gate.  It is kept in PowerFoam because the current sandbox can only
write this repository; the implementation targets the sibling checkouts:

- `/Users/tabel/Projects/arepo/src/bridge/arepo_bridge.h`
- `/Users/tabel/Projects/arepo/src/bridge/arepo_bridge.c`
- `/Users/tabel/Projects/Arepo.jl/lib/ArepoLib/src/fields.jl`

## Goal

Expose the per-face hydro traces that AREPO actually uses during
`compute_interface_fluxes()` so PowerFoam can compare the one-step update below
the cell-aggregate level.

The export must run after AREPO initialization, before `arepo_run_step()` frees
or rebuilds the live mesh, and should mirror the same active-face selection used
by `compute_interface_fluxes()`.

## C ABI

Add these prototypes to `arepo_bridge.h`.

```c
long long arepo_get_hydro_timebins(int *bins, int *active, int *sync,
                             int *active_list,
                             long long *ti_current, double *timebase_interval,
                             long n, long long max_active);

long long arepo_get_hydro_face_traces_3d(
    int *c1, int *c2, int *active, double *face_dt,
    double *area, double *normal, double *face_center, double *vel_face,
    double *state_center_l, double *state_center_r,
    double *state_face_l, double *state_face_r,
    double *flux_lab, long nfaces_max);
```

Return conventions:

- `arepo_get_hydro_timebins` returns the active-particle count on success,
  `-required_active_count` if `max_active` is too small, `-1` if not
  initialized, and `-2` if `n != NumGas`.
- `bins[i]` is `P[i].TimeBinHydro` for gas cell `i`.
- `active[i]` is true when cell `i` is active at `All.Ti_Current`.
- `sync[b]` is `TimeBinSynchronized[b]`.
- `active_list[k]` is `TimeBinsHydro.ActiveParticleList[k]` as a 0-based local
  gas index.
- `ti_current` and `timebase_interval` mirror AREPO globals.
- `arepo_get_hydro_face_traces_3d` returns the number of faces written; if the
  provided face capacity is too small it returns `-required_face_count`.
- `c1/c2` are 0-based local gas indices, with `-1` for non-local/ghost sides.
- vector or state buffers are column-major with `nfaces` rows.
- `normal` and `face_center` have three columns.
- `state_center_l/r`, `state_face_l/r`, and `flux_lab` have five columns:
  `rho`, `vx`, `vy`, `vz`, `pressure_or_energy_flux`.  For `flux_lab`, the
  columns are `mass`, `momentum_x`, `momentum_y`, `momentum_z`, `energy`.
- `state_face_l/r` must be after spatial extrapolation, time extrapolation,
  boundary checks, and velocity rotation into AREPO's face frame.
- `flux_lab` must be after HLL/LLF/HLLC/exact solver handling, advection terms,
  lab-frame conversion, momentum-flux turnback, scalar-state hooks, and flux
  limiting, but before multiplication by `face_dt * area * 0.5`.

## AREPO Implementation Sketch

The face-trace function should factor the body of `compute_interface_fluxes()`
into a helper that can either apply the flux or record it.  Avoid implementing a
second predictor/flux path.  A minimal staging structure is enough:

```c
struct arepo_bridge_face_trace
{
  int c1, c2, active;
  double face_dt, area;
  double normal[3], face_center[3], vel_face[3];
  double center_l[5], center_r[5];
  double face_l[5], face_r[5];
  double flux_lab[5];
};
```

The exporter should fill this structure at the exact point where
`compute_interface_fluxes()` has finished `face_limit_fluxes()` and cosmological
factor application, but before the conserved cell update loop.  That location
captures the flux PowerFoam must match.

## Julia Binding

Add these exports in `ArepoLib/src/fields.jl`.

```julia
export get_hydro_timebins, get_hydro_face_traces_3d

function get_hydro_timebins(h::Handle)
    _check(h)
    n = num_gas(h)
    bins = Vector{Cint}(undef, n)
    active = Vector{Cint}(undef, n)
    sync = Vector{Cint}(undef, 64)
    ti = Ref{Clonglong}(0)
    tbi = Ref{Cdouble}(0)
    maxactive = max(1, n)
    while true
        active_list0 = Vector{Cint}(undef, maxactive)
        got = GC.@preserve bins active sync active_list0 @xcall(
            :arepo_get_hydro_timebins, Clonglong,
            (Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ref{Clonglong}, Ref{Cdouble}, Clong, Clonglong),
            bins, active, sync, active_list0, ti, tbi, Clong(n), Clonglong(maxactive))
        got == -1 && error("get_hydro_timebins: not initialized or bad bridge pointers")
        got == -2 && error("get_hydro_timebins: NumGas changed while reading scheduler state")
        if got < 0
            required = Int(-got)
            required <= maxactive && error("get_hydro_timebins failed ($got)")
            maxactive = required
            continue
        end
        nactive = Int(got)
        return (; bins = Int.(bins),
                active = active .!= 0,
                synchronized = sync .!= 0,
                active_list = Int.(active_list0[1:nactive]) .+ 1,
                ti_current = Int(ti[]),
                timebase_interval = Float64(tbi[]))
    end
end

function get_hydro_face_traces_3d(h::Handle)
    _check(h)
    maxf = max(1024, 24 * num_gas(h))
    while true
        c1 = Vector{Cint}(undef, maxf)
        c2 = Vector{Cint}(undef, maxf)
        active = Vector{Cint}(undef, maxf)
        face_dt = Vector{Float64}(undef, maxf)
        area = Vector{Float64}(undef, maxf)
        normal = Matrix{Float64}(undef, maxf, 3)
        face_center = Matrix{Float64}(undef, maxf, 3)
        vel_face = Matrix{Float64}(undef, maxf, 3)
        state_center_l = Matrix{Float64}(undef, maxf, 5)
        state_center_r = Matrix{Float64}(undef, maxf, 5)
        state_face_l = Matrix{Float64}(undef, maxf, 5)
        state_face_r = Matrix{Float64}(undef, maxf, 5)
        flux_lab = Matrix{Float64}(undef, maxf, 5)
        got = GC.@preserve c1 c2 active face_dt area normal face_center vel_face state_center_l state_center_r state_face_l state_face_r flux_lab @xcall(
            :arepo_get_hydro_face_traces_3d, Clonglong,
            (Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Ptr{Cdouble},
             Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},
             Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Clong),
            c1, c2, active, face_dt, area, normal, face_center, vel_face,
            state_center_l, state_center_r, state_face_l, state_face_r,
            flux_lab, Clong(maxf))
        got == -2 && error("get_hydro_face_traces_3d: not a 3-D build or no live mesh")
        if got < 0
            required = Int(-got)
            required <= maxf && error("get_hydro_face_traces_3d failed ($got)")
            maxf = required
            continue
        end
        nf = Int(got)
        return (; c1 = Int.(c1[1:nf]) .+ 1,
                c2 = Int.(c2[1:nf]) .+ 1,
                active = active[1:nf] .!= 0,
                face_dt = face_dt[1:nf],
                area = area[1:nf],
                normal = normal[1:nf, :],
                face_center = face_center[1:nf, :],
                vel_face = vel_face[1:nf, :],
                state_center_l = state_center_l[1:nf, :],
                state_center_r = state_center_r[1:nf, :],
                state_face_l = state_face_l[1:nf, :],
                state_face_r = state_face_r[1:nf, :],
                flux_lab = flux_lab[1:nf, :])
    end
end
```

## PowerFoam Gate

`examples/arepo_face_trace_gate_3d.jl` is now the consumer.  It skips cleanly
while the bridge lacks `get_hydro_face_traces_3d`, then compares:

- face topology and active mask,
- predicted left/right primitive states in AREPO's face frame,
- lab-frame flux times face area, before timestep multiplication,
- hydro timebins and active-cell lists once `get_hydro_timebins` is present.
