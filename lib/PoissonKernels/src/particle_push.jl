# particle_push — KA port of Enzo's particle gravity interpolation + leapfrog
# drift/kick (the per-cycle `session_update_particles` work), device-agnostic.
#
# Enzo splits the per-particle gravity update across two routines that this file
# reproduces bit-for-bit (the `b8` f64 build is the oracle):
#
#   Grid::ComputeAccelerations   (src/enzo/Grid_ComputeAccelerations.C)
#     · difference φ → AccelerationField   (comp_accel!, iflag=0, cell-centred)
#     · drift particles +½dt forward  (UpdateParticlePosition(+0.5 dtFixed))
#     · CIC-interpolate the field onto each particle  (cic_interp.F, 8-point)
#     · drift particles −½dt back      (exactly reversible — same |coef|)
#   UpdateParticlePositions      (src/enzo/UpdateParticlePositions.C)
#     · half-kick(½dt) → drift(dt) → half-kick(½dt)            [leapfrog]
#
# We never mutate the stored position for the interpolation half-drift: the
# interp kernel evaluates the field at  x + dcoef·v  inline (dcoef = ½dt/a at the
# interp time), which is identical to drift-forward / interp / drift-back.
#
# Cosmology coefficients (a, ȧ) come from Enzo's own CosmologyComputeExpansionFactor
# via `EnzoLib.session_expansion_factor` so the comoving factors match to round-off:
#   · interp half-drift   dcoef = ½·dt / a(t+¼dt)
#   · main drift          coef  =   dt / a(t+½dt)
#   · semi-implicit kick  ts = ½dt,  coef = ½·(ȧ/a)·ts  with a,ȧ at t+½dt
#     v ← ((1−coef)·v + g·ts) / (1+coef)        (g = ParticleAcceleration, already /a)
#
# Particle layout is the SoA used by `cic_deposit!` (px,py,pz,vx,vy,vz device
# vectors, one thread per particle). The acceleration grids gx,gy,gz are 3-D
# device arrays on the Enzo *grid* mesh (GridDimension, NG ghost zones); the
# interp geometry is passed explicitly: cellsize = dx, leftedge = CellLeftEdge[0]
# (the leftmost ghost edge, e.g. −NG·dx for a unit periodic box).

# ── CIC interpolation of the three accel grids onto particles ─────────────────
# No `where {T}` on the @kernel (would box type params → Metal InvalidIR); the
# element type flows in via the array args, scalars converted by the launcher.
@kernel function _interp_accel_kernel!(axp, ayp, azp,
                                       @Const(px), @Const(py), @Const(pz),
                                       @Const(vx), @Const(vy), @Const(vz),
                                       @Const(gx), @Const(gy), @Const(gz),
                                       dcoef, invcell, lex, ley, lez,
                                       half, c05, c1, e1, e2, e3)
    p = @index(Global)
    @inbounds begin
        # interpolate at the +½dt forward-drifted position (cancels on the way back)
        xq = px[p] + dcoef * vx[p]
        yq = py[p] + dcoef * vy[p]
        zq = pz[p] + dcoef * vz[p]
        # cell-coordinate, clamped exactly as cic_interp.F (clamp uses half = 0.5001)
        xpos = min(max((xq - lex) * invcell, half), e1)
        ypos = min(max((yq - ley) * invcell, half), e2)
        zpos = min(max((zq - lez) * invcell, half), e3)
        # index + weight use plain 0.5 (RKIND), NOT 0.5001
        i1 = unsafe_trunc(Int, xpos + c05)
        j1 = unsafe_trunc(Int, ypos + c05)
        k1 = unsafe_trunc(Int, zpos + c05)
        dxr = oftype(xpos, i1) + c05 - xpos
        dyr = oftype(ypos, j1) + c05 - ypos
        dzr = oftype(zpos, k1) + c05 - zpos
        ex = c1 - dxr; ey = c1 - dyr; ez = c1 - dzr
        # 8-point trilinear (cic_interp.F:158-174)
        axp[p] = gx[i1  , j1  , k1  ] * dxr * dyr * dzr +
                 gx[i1+1, j1  , k1  ] * ex  * dyr * dzr +
                 gx[i1  , j1+1, k1  ] * dxr * ey  * dzr +
                 gx[i1+1, j1+1, k1  ] * ex  * ey  * dzr +
                 gx[i1  , j1  , k1+1] * dxr * dyr * ez  +
                 gx[i1+1, j1  , k1+1] * ex  * dyr * ez  +
                 gx[i1  , j1+1, k1+1] * dxr * ey  * ez  +
                 gx[i1+1, j1+1, k1+1] * ex  * ey  * ez
        ayp[p] = gy[i1  , j1  , k1  ] * dxr * dyr * dzr +
                 gy[i1+1, j1  , k1  ] * ex  * dyr * dzr +
                 gy[i1  , j1+1, k1  ] * dxr * ey  * dzr +
                 gy[i1+1, j1+1, k1  ] * ex  * ey  * dzr +
                 gy[i1  , j1  , k1+1] * dxr * dyr * ez  +
                 gy[i1+1, j1  , k1+1] * ex  * dyr * ez  +
                 gy[i1  , j1+1, k1+1] * dxr * ey  * ez  +
                 gy[i1+1, j1+1, k1+1] * ex  * ey  * ez
        azp[p] = gz[i1  , j1  , k1  ] * dxr * dyr * dzr +
                 gz[i1+1, j1  , k1  ] * ex  * dyr * dzr +
                 gz[i1  , j1+1, k1  ] * dxr * ey  * dzr +
                 gz[i1+1, j1+1, k1  ] * ex  * ey  * dzr +
                 gz[i1  , j1  , k1+1] * dxr * dyr * ez  +
                 gz[i1+1, j1  , k1+1] * ex  * dyr * ez  +
                 gz[i1  , j1+1, k1+1] * dxr * ey  * ez  +
                 gz[i1+1, j1+1, k1+1] * ex  * ey  * ez
    end
end

"""
    interp_accel_to_particles!(axp,ayp,azp, px,py,pz, vx,vy,vz, gx,gy,gz;
                               dcoef, cellsize, leftedge) -> (axp,ayp,azp)

CIC-interpolate the three acceleration grids `gx,gy,gz` (3-D device arrays on the
Enzo grid mesh) onto each particle, writing the per-particle accelerations
`axp,ayp,azp` (device vectors). The field is evaluated at the half-step
forward-drifted position `x + dcoef·v` (`dcoef = ½·dt/a` at the interpolation
time) — Enzo's `Grid::ComputeAccelerations` drift-forward/interp/drift-back, done
without mutating the stored positions. `cellsize` is the grid cell width and
`leftedge = (lex,ley,lez)` is `CellLeftEdge[·][0]` (the leftmost ghost edge).
Mirrors `cic_interp.F` (8-point, `half = 0.5001`).
"""
function interp_accel_to_particles!(axp::AbstractVector{T}, ayp, azp,
                                    px, py, pz, vx, vy, vz,
                                    gx::AbstractArray{<:Any,3}, gy, gz;
                                    dcoef::Real, cellsize::Real,
                                    leftedge) where {T}
    be = KA.get_backend(axp)
    d1, d2, d3 = size(gx)
    half = T(0.5001)
    e1 = T(d1) - half; e2 = T(d2) - half; e3 = T(d3) - half
    _interp_accel_kernel!(be)(axp, ayp, azp, px, py, pz, vx, vy, vz,
                              gx, gy, gz,
                              T(dcoef), T(1) / T(cellsize),
                              T(leftedge[1]), T(leftedge[2]), T(leftedge[3]),
                              half, T(0.5), T(1), e1, e2, e3; ndrange = length(axp))
    return axp, ayp, azp
end

# ── CIC force interp straight from the POTENTIAL (no stored accel field) ──────
# Same difference-then-interpolate as _interp_accel_kernel! (so momentum-conserving,
# self-force-free), but g = −∇φ is central-differenced inline at each of the 8 CIC cells
# from the padded potential `φ` — needs ≥2 ghost cells so cell±1 is in bounds for all 8.
@kernel function _interp_force_phi_kernel!(axp, ayp, azp,
                                           @Const(px), @Const(py), @Const(pz),
                                           @Const(vx), @Const(vy), @Const(vz), @Const(φ),
                                           dcoef, invcell, lex, ley, lez,
                                           half, c05, c1, e1, e2, e3, hc)
    p = @index(Global)
    @inbounds begin
        xq = px[p] + dcoef*vx[p]; yq = py[p] + dcoef*vy[p]; zq = pz[p] + dcoef*vz[p]
        xpos = min(max((xq-lex)*invcell, half), e1)
        ypos = min(max((yq-ley)*invcell, half), e2)
        zpos = min(max((zq-lez)*invcell, half), e3)
        i1 = unsafe_trunc(Int, xpos+c05); j1 = unsafe_trunc(Int, ypos+c05); k1 = unsafe_trunc(Int, zpos+c05)
        dxr = oftype(xpos,i1)+c05-xpos; dyr = oftype(ypos,j1)+c05-ypos; dzr = oftype(zpos,k1)+c05-zpos
        ex = c1-dxr; ey = c1-dyr; ez = c1-dzr
        gxc(i,j,k) = -hc*(φ[i+1,j,k]-φ[i-1,j,k])
        gyc(i,j,k) = -hc*(φ[i,j+1,k]-φ[i,j-1,k])
        gzc(i,j,k) = -hc*(φ[i,j,k+1]-φ[i,j,k-1])
        w(a,b,c) = a*b*c
        axp[p] = gxc(i1,j1,k1)*w(dxr,dyr,dzr)+gxc(i1+1,j1,k1)*w(ex,dyr,dzr)+gxc(i1,j1+1,k1)*w(dxr,ey,dzr)+gxc(i1+1,j1+1,k1)*w(ex,ey,dzr)+
                 gxc(i1,j1,k1+1)*w(dxr,dyr,ez)+gxc(i1+1,j1,k1+1)*w(ex,dyr,ez)+gxc(i1,j1+1,k1+1)*w(dxr,ey,ez)+gxc(i1+1,j1+1,k1+1)*w(ex,ey,ez)
        ayp[p] = gyc(i1,j1,k1)*w(dxr,dyr,dzr)+gyc(i1+1,j1,k1)*w(ex,dyr,dzr)+gyc(i1,j1+1,k1)*w(dxr,ey,dzr)+gyc(i1+1,j1+1,k1)*w(ex,ey,dzr)+
                 gyc(i1,j1,k1+1)*w(dxr,dyr,ez)+gyc(i1+1,j1,k1+1)*w(ex,dyr,ez)+gyc(i1,j1+1,k1+1)*w(dxr,ey,ez)+gyc(i1+1,j1+1,k1+1)*w(ex,ey,ez)
        azp[p] = gzc(i1,j1,k1)*w(dxr,dyr,dzr)+gzc(i1+1,j1,k1)*w(ex,dyr,dzr)+gzc(i1,j1+1,k1)*w(dxr,ey,dzr)+gzc(i1+1,j1+1,k1)*w(ex,ey,dzr)+
                 gzc(i1,j1,k1+1)*w(dxr,dyr,ez)+gzc(i1+1,j1,k1+1)*w(ex,dyr,ez)+gzc(i1,j1+1,k1+1)*w(dxr,ey,ez)+gzc(i1+1,j1+1,k1+1)*w(ex,ey,ez)
    end
end

"""
    interp_force_from_potential!(axp,ayp,azp, px,py,pz, vx,vy,vz, φ;
                                 dcoef, cellsize, leftedge) -> (axp,ayp,azp)

Like [`interp_accel_to_particles!`], but reads the padded **potential** `φ` and forms
`g = −∇φ` (central difference) inline at the 8 CIC cells — no precomputed accel grids.
`cellsize = dx` and the central-difference uses the same `1/(2dx)`; `φ` must carry ≥2
ghost cells.  Mathematically identical (difference-then-interpolate) to the stored-accel path.
"""
function interp_force_from_potential!(axp::AbstractVector{T}, ayp, azp,
                                      px, py, pz, vx, vy, vz, φ::AbstractArray{<:Any,3};
                                      dcoef::Real, cellsize::Real, leftedge) where {T}
    be = KA.get_backend(axp); d1,d2,d3 = size(φ)
    half = T(0.5001); e1 = T(d1)-half; e2 = T(d2)-half; e3 = T(d3)-half
    _interp_force_phi_kernel!(be)(axp, ayp, azp, px, py, pz, vx, vy, vz, φ,
                                  T(dcoef), T(1)/T(cellsize), T(leftedge[1]), T(leftedge[2]), T(leftedge[3]),
                                  half, T(0.5), T(1), e1, e2, e3, T(0.5)/T(cellsize); ndrange = length(axp))
    return axp, ayp, azp
end

# ── Ghost-free force interp from the GLOBAL nc³ potential ─────────────────────
# The global-potential path is equivalent to the padded-potential path for periodic
# particles in [0,1): the pad offset cancels out of the CIC weights and the ghost
# fill is exactly the same periodic index wrap. It avoids allocating/filling a
# transient `(nc+2ng)³` potential block for the particle push.
@inline _wrap1(i::Int, n::Int) = mod(i - 1, n) + 1
@kernel function _interp_force_global_phi_kernel!(axp, ayp, azp,
                                                  @Const(px), @Const(py), @Const(pz),
                                                  @Const(vx), @Const(vy), @Const(vz), @Const(φg),
                                                  dcoef, invcell, c05, c1,
                                                  n1::Int, n2::Int, n3::Int, hc)
    p = @index(Global)
    @inbounds begin
        xq = px[p] + dcoef * vx[p]
        yq = py[p] + dcoef * vy[p]
        zq = pz[p] + dcoef * vz[p]
        xpos = xq * invcell
        ypos = yq * invcell
        zpos = zq * invcell
        i1 = unsafe_trunc(Int, xpos + c05)
        j1 = unsafe_trunc(Int, ypos + c05)
        k1 = unsafe_trunc(Int, zpos + c05)
        dxr = oftype(xpos, i1) + c05 - xpos
        dyr = oftype(ypos, j1) + c05 - ypos
        dzr = oftype(zpos, k1) + c05 - zpos
        ex = c1 - dxr; ey = c1 - dyr; ez = c1 - dzr
        φr(i,j,k) = φg[_wrap1(i, n1), _wrap1(j, n2), _wrap1(k, n3)]
        gxc(i,j,k) = -hc * (φr(i+1,j,k) - φr(i-1,j,k))
        gyc(i,j,k) = -hc * (φr(i,j+1,k) - φr(i,j-1,k))
        gzc(i,j,k) = -hc * (φr(i,j,k+1) - φr(i,j,k-1))
        w(a,b,c) = a * b * c
        axp[p] = gxc(i1,j1,k1)*w(dxr,dyr,dzr)+gxc(i1+1,j1,k1)*w(ex,dyr,dzr)+gxc(i1,j1+1,k1)*w(dxr,ey,dzr)+gxc(i1+1,j1+1,k1)*w(ex,ey,dzr)+
                 gxc(i1,j1,k1+1)*w(dxr,dyr,ez)+gxc(i1+1,j1,k1+1)*w(ex,dyr,ez)+gxc(i1,j1+1,k1+1)*w(dxr,ey,ez)+gxc(i1+1,j1+1,k1+1)*w(ex,ey,ez)
        ayp[p] = gyc(i1,j1,k1)*w(dxr,dyr,dzr)+gyc(i1+1,j1,k1)*w(ex,dyr,dzr)+gyc(i1,j1+1,k1)*w(dxr,ey,dzr)+gyc(i1+1,j1+1,k1)*w(ex,ey,dzr)+
                 gyc(i1,j1,k1+1)*w(dxr,dyr,ez)+gyc(i1+1,j1,k1+1)*w(ex,dyr,ez)+gyc(i1,j1+1,k1+1)*w(dxr,ey,ez)+gyc(i1+1,j1+1,k1+1)*w(ex,ey,ez)
        azp[p] = gzc(i1,j1,k1)*w(dxr,dyr,dzr)+gzc(i1+1,j1,k1)*w(ex,dyr,dzr)+gzc(i1,j1+1,k1)*w(dxr,ey,dzr)+gzc(i1+1,j1+1,k1)*w(ex,ey,dzr)+
                 gzc(i1,j1,k1+1)*w(dxr,dyr,ez)+gzc(i1+1,j1,k1+1)*w(ex,dyr,ez)+gzc(i1,j1+1,k1+1)*w(dxr,ey,ez)+gzc(i1+1,j1+1,k1+1)*w(ex,ey,ez)
    end
end

"""
    interp_force_from_global_potential!(axp,ayp,azp, px,py,pz, vx,vy,vz, φg;
                                        dcoef, nc) -> (axp,ayp,azp)

Like [`interp_force_from_potential!`] but reads the global `nc³` potential directly
with periodic index wrap. This avoids the padded potential allocation used by the
legacy particle path.
"""
function interp_force_from_global_potential!(axp::AbstractVector{T}, ayp, azp,
                                             px, py, pz, vx, vy, vz, φg::AbstractArray{<:Any,3};
                                             dcoef::Real, nc) where {T}
    be = KA.get_backend(axp)
    n1, n2, n3 = Int(nc[1]), Int(nc[2]), Int(nc[3])
    _interp_force_global_phi_kernel!(be)(axp, ayp, azp, px, py, pz, vx, vy, vz, φg,
                                         T(dcoef), T(n1), T(0.5), T(1),
                                         n1, n2, n3, T(0.5) * T(n1);
                                         ndrange = length(axp))
    return axp, ayp, azp
end

@kernel function _kick_kernel!(vx, vy, vz, @Const(axp), @Const(ayp), @Const(azp),
                               ts, coef1, coef2)
    p = @index(Global)
    @inbounds begin
        Tv = eltype(vx)
        vx[p] = Tv((coef1 * vx[p] + axp[p] * ts) * coef2)
        vy[p] = Tv((coef1 * vy[p] + ayp[p] * ts) * coef2)
        vz[p] = Tv((coef1 * vz[p] + azp[p] * ts) * coef2)
    end
end

"""
    particle_kick!(vx,vy,vz, axp,ayp,azp; ts, coef) -> (vx,vy,vz)

Semi-implicit comoving velocity half-kick (Enzo VELOCITY_METHOD3):
`v ← ((1−coef)·v + g·ts) / (1+coef)`, with `ts = ½dt` the accel timestep and
`coef = ½·(ȧ/a)·ts` (a,ȧ at `t+½dt`). The per-particle acceleration `g` is the
output of [`interp_accel_to_particles!`] (already divided by `a`). Called twice
per cycle, once on each side of the drift. With `coef = 0` this is the plain
`v += g·ts` non-comoving kick.
"""
function particle_kick!(vx::AbstractVector, vy, vz, axp, ayp, azp;
                        ts::Real, coef::Real)
    be = KA.get_backend(vx)
    C = promote_type(eltype(vx), eltype(axp), Float32)
    c = C(coef)
    _kick_kernel!(be)(vx, vy, vz, axp, ayp, azp,
                      C(ts), one(C) - c, one(C) / (one(C) + c);
                      ndrange = length(vx))
    return vx, vy, vz
end

# ── comoving drift (Grid_UpdateParticlePosition.C) ────────────────────────────
@kernel function _drift_kernel!(px, py, pz, @Const(vx), @Const(vy), @Const(vz),
                                coef, wrap, dowrap::Int)
    p = @index(Global)
    @inbounds begin
        x = px[p] + coef * vx[p]
        y = py[p] + coef * vy[p]
        z = pz[p] + coef * vz[p]
        if dowrap == 1
            x = mod(x, wrap); y = mod(y, wrap); z = mod(z, wrap)
        end
        px[p] = x; py[p] = y; pz[p] = z
    end
end

"""
    particle_drift!(px,py,pz, vx,vy,vz; coef, wrap=0) -> (px,py,pz)

Drift positions `x ← x + coef·v` (`coef = dt/a` at `t+½dt`, Enzo
`UpdateParticlePosition`). `wrap > 0` additionally wraps each coordinate into
`[0,wrap)` — physically identical under periodicity and needed in the resident
loop to keep f32 positions from drifting many box-lengths over a long run; leave
`wrap = 0` (no wrap) to match a single Enzo update bit-for-bit.
"""
function particle_drift!(px::AbstractVector{T}, py, pz, vx, vy, vz;
                         coef::Real, wrap::Real = 0) where {T}
    be = KA.get_backend(px)
    dowrap = wrap > 0 ? 1 : 0
    _drift_kernel!(be)(px, py, pz, vx, vy, vz, T(coef), T(wrap), dowrap;
                       ndrange = length(px))
    return px, py, pz
end
