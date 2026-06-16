# enzo_resident — GPU-resident particle state for the Enzo cosmology evolve loop.
#
# Moves the per-cycle `session_update_particles` work (CIC interpolation of the
# acceleration field onto ~N³ particles + the leapfrog half-kick/drift/half-kick)
# off Enzo's CPU and onto the GPU via the device-agnostic PoissonKernels particle
# kernels, while keeping the particle position/velocity/mass arrays RESIDENT on the
# device across cycles. Gravity deposits straight from the resident arrays (no
# per-cycle particle read), and the push writes the updated state back to Enzo once
# per cycle so Enzo's own compute_dt / rebuild / diagnostics see fresh particles.
#
# The comoving coefficients come from Enzo's own CosmologyComputeExpansionFactor
# (`EnzoLib.session_expansion_factor`), evaluated at the exact sub-step times Enzo
# uses, so the update matches `session_update_particles` to round-off:
#   · interp half-drift  a(t+¼dt)      (Grid::ComputeAccelerations' +½dt forward drift)
#   · main drift         a(t+½dt)      (UpdateParticlePosition)
#   · semi-implicit kick a,ȧ(t+½dt)    (UpdateParticleVelocity, METHOD3)
#
# The acceleration grid the particles interpolate from is Enzo's AccelerationField
# (the NORMAL-difference, cell-centred field PPM cosmology uses) read back via
# `problem_get_acceleration`. Interp geometry on the unit periodic box: cell width
# dx = 1/N, left edge = −NGg·dx where NGg = (grid_dim − N)/2 (the grid ghost depth).

"""
    ResidentParticles

Device-resident particle state (positions, velocities, mass) plus the cached
acceleration grids and the interp geometry, for one Enzo grid. Build with
[`resident_particles_init`]; drive with [`particle_push_gpu!`].
"""
mutable struct ResidentParticles{BE,V,A}
    be::BE
    grid::Int
    np::Int
    N::Int                  # active cells / particles per dim
    Mg::Int                 # acceleration-grid dimension (N + 2·NGg)
    NGg::Int                # grid ghost depth per side
    cellsize::Float64       # dx = 1/N (unit box)
    leftedge::NTuple{3,Float64}
    px::V; py::V; pz::V      # resident positions  (device vectors)
    vx::V; vy::V; vz::V      # resident velocities
    mass::V                 # resident masses
    axp::V; ayp::V; azp::V   # per-particle accel scratch (device vectors)
    gx::A; gy::A; gz::A      # acceleration grids (device, Mg³), refreshed per cycle
    wrap::Float64           # period for the drift wrap (0 ⇒ no wrap)
end

"""
    resident_particles_init(h, be, ::Type{T}; grid=0, wrap=1.0) -> ResidentParticles

Upload the particles of Enzo `grid` onto backend `be` at precision `T` (once),
allocate the per-particle accel scratch and the acceleration-grid buffers, and
capture the interp geometry from `problem_grid_dims` (unit periodic box). `wrap`
is the drift period (1.0 keeps box-normalized positions in [0,1) so f32 stays
accurate over a long run; 0 disables wrapping).
"""
function resident_particles_init(h::EnzoLib.Handle, be, ::Type{T}; grid::Integer = 0,
                                 wrap::Real = 1.0) where {T}
    np = EnzoLib.problem_num_particles(h, grid)
    np > 0 || error("resident_particles_init: grid $grid has no particles")
    N = round(Int, cbrt(np))
    N^3 == np || error("resident_particles_init: $np particles is not a perfect cube")
    gd = Tuple(Int.(EnzoLib.problem_grid_dims(h, grid)))
    Mg = gd[1]
    (gd[2] == Mg && gd[3] == Mg) || error("resident_particles_init: non-cubic grid dims $gd")
    NGg = (Mg - N) ÷ 2
    cellsize = 1.0 / N
    le = (-NGg * cellsize, -NGg * cellsize, -NGg * cellsize)
    d(x) = PoissonKernels.to_device(be, x, T)
    px = d(EnzoLib.problem_get_particle_pos(h, 0, grid))
    py = d(EnzoLib.problem_get_particle_pos(h, 1, grid))
    pz = d(EnzoLib.problem_get_particle_pos(h, 2, grid))
    vx = d(EnzoLib.problem_get_particle_vel(h, 0, grid))
    vy = d(EnzoLib.problem_get_particle_vel(h, 1, grid))
    vz = d(EnzoLib.problem_get_particle_vel(h, 2, grid))
    mass = d(EnzoLib.problem_get_particle_mass(h, grid))
    z = () -> PoissonKernels.device_zeros(be, T, (np,))
    g = () -> PoissonKernels.device_zeros(be, T, (Mg, Mg, Mg))
    ResidentParticles{typeof(be),typeof(px),typeof(g())}(be, Int(grid), np, N, Mg, NGg,
        cellsize, le, px, py, pz, vx, vy, vz, mass, z(), z(), z(), g(), g(), g(), Float64(wrap))
end

"Copy Enzo's three AccelerationField components into the resident accel grids."
function refresh_accel!(st::ResidentParticles, h::EnzoLib.Handle)
    for (gi, gr) in enumerate((st.gx, st.gy, st.gz))
        a = reshape(Float64.(EnzoLib.problem_get_acceleration(h, gi - 1, st.grid)), st.Mg, st.Mg, st.Mg)
        copyto!(gr, eltype(gr).(a))
    end
    return st
end

"""
    particle_push_gpu!(st, h, level, dt; sync=true)

The GPU replacement for `session_update_particles` on the resident particles of
`st`: refresh the acceleration grids from Enzo, build the comoving coefficients
from `session_expansion_factor`, then run interp → ½-kick → drift → ½-kick on the
device. With `sync=true` (default) the updated positions/velocities are written
back to Enzo so the rest of the cycle (compute_dt, rebuild, diagnostics) sees
them.
"""
function particle_push_gpu!(st::ResidentParticles, h::EnzoLib.Handle, level::Integer,
                            dt::Real; sync::Bool = true, refresh::Bool = true)
    refresh && refresh_accel!(st, h)
    t = EnzoLib.session_time(h)
    a_q, _      = EnzoLib.session_expansion_factor(h, t + 0.25 * dt)   # interp half-drift
    a_h, dadt_h = EnzoLib.session_expansion_factor(h, t + 0.5 * dt)    # drift + kick
    dcoef = 0.5 * dt / a_q
    half  = 0.5 * dt
    kcoef = 0.5 * dadt_h / a_h * half
    dcoef_drift = dt / a_h
    PoissonKernels.interp_accel_to_particles!(st.axp, st.ayp, st.azp,
        st.px, st.py, st.pz, st.vx, st.vy, st.vz, st.gx, st.gy, st.gz;
        dcoef = dcoef, cellsize = st.cellsize, leftedge = st.leftedge)
    PoissonKernels.particle_kick!(st.vx, st.vy, st.vz, st.axp, st.ayp, st.azp; ts = half, coef = kcoef)
    PoissonKernels.particle_drift!(st.px, st.py, st.pz, st.vx, st.vy, st.vz;
                                   coef = dcoef_drift, wrap = st.wrap)
    PoissonKernels.particle_kick!(st.vx, st.vy, st.vz, st.axp, st.ayp, st.azp; ts = half, coef = kcoef)
    sync && sync_to_enzo!(st, h)
    return nothing
end

"Write the resident positions/velocities back into Enzo (so its CPU side is current)."
function sync_to_enzo!(st::ResidentParticles, h::EnzoLib.Handle)
    th(x) = Float64.(PoissonKernels.to_host(x))
    EnzoLib.problem_set_particle_pos(h, 0, th(st.px); grid = st.grid)
    EnzoLib.problem_set_particle_pos(h, 1, th(st.py); grid = st.grid)
    EnzoLib.problem_set_particle_pos(h, 2, th(st.pz); grid = st.grid)
    EnzoLib.problem_set_particle_vel(h, 0, th(st.vx); grid = st.grid)
    EnzoLib.problem_set_particle_vel(h, 1, th(st.vy); grid = st.grid)
    EnzoLib.problem_set_particle_vel(h, 2, th(st.vz); grid = st.grid)
    return nothing
end

"""
    gpu_particle_push_slot(st; sync=true) -> (h, level, dt) -> nothing

A hook closure for `EngineConfig(particle_push=:julia, hooks=Dict(:particle_push=>…))`
that drives [`particle_push_gpu!`] on the resident state `st`.
"""
gpu_particle_push_slot(st::ResidentParticles; sync::Bool = true) =
    (h, level, dt) -> particle_push_gpu!(st, h, level, dt; sync = sync)
