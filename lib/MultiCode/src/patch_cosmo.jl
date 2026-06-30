# patch_cosmo.jl — standalone RAMSES super-comoving cosmology for the patch cicass run.
#
# RAMSES super-comoving (amr/units.f90, init_time.f90 friedman): code time = the
# super-conformal time τ (dτ = dt/a²), with da/dτ = √(a³(Ωm+ΩΛa³+Ωk·a+Ωr/a)) in
# units where H0=1.  In these variables the gas/particle equations are PLAIN
# (leapfrog, no Hubble-drag source — it is absorbed into the unit scalings); the
# ONLY cosmological coupling is (a) the dτ↔a stepping, (b) the Poisson source
# 1.5·Ωm·a·δ, and (c) the per-a physical unit scalings used by the chemistry.
# Matching these reproduces RAMSES (the cicass reference).

const _KB    = 1.3806490e-16     # erg/K
const _MH    = 1.6605390e-24     # g (amu)
const _RHOC  = 1.8800000e-29     # g/cc  critical density at h=1 (RAMSES constants.f90)
const _MPC   = 3.0856776e24      # cm
const _KMS   = 1.0e5             # cm/s
const _ARAD  = 7.5657e-15        # erg/cc/K⁴
const _SIGT  = 6.6524e-25        # cm²
const _CL    = 2.99792458e10     # cm/s
const _TCMB0 = 2.726             # K

"Cosmology + box for the super-comoving patch run.  `box` is the comoving box in Mpc/h."
struct Cosmo
    Om::Float64; OL::Float64; Ok::Float64; Or::Float64
    h0::Float64        # H0 = 100·h  [km/s/Mpc]
    box::Float64       # comoving box [Mpc/h]
    fb::Float64        # baryon fraction Ωb/Ωm
    XH::Float64
end
function Cosmo(; Om::Real, OL::Real, h0::Real, box::Real, Ob::Real, XH::Real=0.76, Or::Real=0.0)
    Cosmo(Om, OL, 1 - Om - OL - Or, Or, h0, box, Ob/Om, XH)
end

H0_inv_s(c::Cosmo) = c.h0 * _KMS / _MPC                      # H0 in 1/s
"da/dτ in super-conformal time (τ in units of 1/H0)."
dadtau(c::Cosmo, a) = sqrt(a^3 * (c.Om + c.OL*a^3 + c.Ok*a + c.Or/a))
"Code timestep dτ for a target Δln a (τ in 1/H0 units)."
dtau_for_dlna(c::Cosmo, a, dlna) = a * dlna / dadtau(c, a)
"Physical Hubble rate H(a) [1/s]."
Hofa(c::Cosmo, a) = H0_inv_s(c) * sqrt(c.Om/a^3 + c.Or/a^4 + c.OL + c.Ok/a^2)

"RAMSES physical unit factors at scale factor `a` (units.f90:21-26)."
function cosmo_units(c::Cosmo, a)
    sd = c.Om * _RHOC * (c.h0/100)^2 / a^3
    st = a^2 / H0_inv_s(c)
    sl = a * c.box * _MPC / (c.h0/100)
    sv = sl / st
    (d=sd, l=sl, t=st, v=sv, T2=_MH/_KB*sv^2, nH=c.XH/_MH*sd)
end

"Linear growth factor D(a) (Heath 1977; matter+Λ(+curv,rad))."
function growth_D(c::Cosmo, a)
    E(x) = sqrt(c.Om/x^3 + c.Or/x^4 + c.OL + c.Ok/x^2)
    f(x) = 1.0 / (x*E(x))^3
    n = 4000; h = a/n; s = 0.0
    @inbounds for i in 1:n
        x0 = (i-1)*h + 1e-12; x1 = i*h
        s += 0.5*(f(x0)+f(x1))*h
    end
    return E(a) * s
end

"Compton drag rate Γ/H at redshift `z` given ionized fraction `xe` (cgs)."
function compton_drag_over_H(c::Cosmo, z, xe)
    Γ = (4/3) * _ARAD * (_TCMB0*(1+z))^4 * xe * c.XH * _SIGT / (_CL*_MH)
    return Γ / Hofa(c, z_to_a(z))
end
z_to_a(z) = 1.0 / (1.0 + z)
a_to_z(a) = 1.0/a - 1.0

"""
    compton_drag_patches!(pg, f)

Apply Compton drag to the baryon gas of every patch: damp the peculiar velocity
toward the GLOBAL mass-weighted bulk by factor `f = exp(−Γ/H·Δln a)`, keeping the
internal energy fixed (frame-agnostic — never damps the coherent streaming bulk).
Mirrors `ramses_compton_drag!`.
"""
function compton_drag_patches!(pg::PatchGrid, f::Real)
    # mass-weighted bulk velocity from ON-DEVICE interior reductions (no full-grid
    # gather/host-transfer — that copy was the dominant high-z per-cycle cost).
    M = 0.0; px = 0.0; py = 0.0; pz = 0.0
    for p in pg.patches
        M  += _interior_sum(pg, p.D)
        px += _interior_sum(pg, p.S1); py += _interior_sum(pg, p.S2); pz += _interior_sum(pg, p.S3)
    end
    T = pg.T; ff = T(f); vx = T(px/M); vy = T(py/M); vz = T(pz/M)
    for p in pg.patches
        ke0 = (p.S1.^2 .+ p.S2.^2 .+ p.S3.^2) ./ (2 .* p.D)
        p.S1 .= p.D .* vx .+ (p.S1 .- p.D .* vx) .* ff
        p.S2 .= p.D .* vy .+ (p.S2 .- p.D .* vy) .* ff
        p.S3 .= p.D .* vz .+ (p.S3 .- p.D .* vz) .* ff
        ke1 = (p.S1.^2 .+ p.S2.^2 .+ p.S3.^2) ./ (2 .* p.D)
        p.Tau .+= ke1 .- ke0                                # internal energy unchanged
    end
    return nothing
end

"""
    push_particles!(parts, gx, gy, gz, leftedge, cellsize, dtau; scratch=nothing)

Super-comoving KDK particle update: interp the global ghosted accel field at the
half-drifted position, then kick–drift–kick (plain leapfrog, `coef=0` since the
expansion is in the super-comoving units), wrapping positions to `[0,1)`.
`parts` is `(px,py,pz,vx,vy,vz,mass)` device vectors with global box-normalized
positions; `gx,gy,gz` is the field from `particle_accel_field`.
"""
function push_particles!(parts, φpad, leftedge::Real, cellsize::Real, dtau::Real;
                         scratch=nothing)
    if scratch === nothing
        axp = similar(parts.px); ayp = similar(parts.px); azp = similar(parts.px)
    else
        axp, ayp, azp = scratch
    end
    half = 0.5*dtau
    le = (leftedge, leftedge, leftedge)
    # force = −∇φ central-differenced at the CIC cells from the padded potential (no stored accel)
    PoissonKernels.interp_force_from_potential!(axp, ayp, azp,
        parts.px, parts.py, parts.pz, parts.vx, parts.vy, parts.vz, φpad;
        dcoef=half, cellsize=cellsize, leftedge=le)
    PoissonKernels.particle_kick!(parts.vx, parts.vy, parts.vz, axp, ayp, azp; ts=half, coef=0.0)
    PoissonKernels.particle_drift!(parts.px, parts.py, parts.pz, parts.vx, parts.vy, parts.vz;
        coef=dtau, wrap=1.0)
    PoissonKernels.particle_kick!(parts.vx, parts.vy, parts.vz, axp, ayp, azp; ts=half, coef=0.0)
    return (axp, ayp, azp)
end

# ── Morton (Z-order) particle sort: restore deposit/gather locality, bit-identical physics ─────────
# The DM CIC deposit is ~17× slower for scrambled vs grid-ordered particles (uncoalesced atomics), and
# the Lagrangian order decays as the DM clusters.  Re-sorting the SoA by each particle's Eulerian-cell
# Morton code restores 3D locality (deposit AND the force gather in push_particles!).  The INTEGER
# deposit and the per-particle push are order-INDEPENDENT ⇒ the sort changes ONLY storage order, not
# the physics (bit-identical).  A tracked Lagrangian `id`, permuted with the sort, preserves the
# particle→Lagrangian-grid map needed to rebuild the phase-space sheet from snapshots.
@inline function _part1by2(n::UInt32)
    n &= 0x000003ff
    n = (n | (n << 16)) & 0xff0000ff
    n = (n | (n <<  8)) & 0x0300f00f
    n = (n | (n <<  4)) & 0x030c30c3
    n = (n | (n <<  2)) & 0x09249249
    return n
end
@inline _morton3(x::UInt32, y::UInt32, z::UInt32) = _part1by2(x) | (_part1by2(y) << 1) | (_part1by2(z) << 2)

@kernel function _morton_packed_k!(packed, @Const(px), @Const(py), @Const(pz), N::Int32, Nm1::UInt32)
    p = @index(Global)
    @inbounds begin
        ix = min(unsafe_trunc(UInt32, floor(px[p]*N)), Nm1)
        iy = min(unsafe_trunc(UInt32, floor(py[p]*N)), Nm1)
        iz = min(unsafe_trunc(UInt32, floor(pz[p]*N)), Nm1)
        packed[p] = (UInt64(_morton3(ix, iy, iz)) << 32) | UInt64(p - 1)   # (Morton key | 0-based idx)
    end
end

"""
    morton_sort_particles!(parts; N) -> parts

Reorder the particle SoA in place by the Morton (Z-order) code of each particle's Eulerian cell on the
`N³` grid, restoring deposit/gather locality.  Permutes every AbstractVector field (px..vz and a tracked
`id`); the scalar `mass` is untouched.  Bit-identical physics (deposit + push are order-independent).
Uses a packed-UInt64 key-value `sort!` (GPU radix).  Requires `N ≤ 1024` (Morton fits 30 bits).
"""
function morton_sort_particles!(parts; N::Integer)
    px = parts.px; Np = length(px); be = PoissonKernels.KA.get_backend(px)
    packed = similar(px, UInt64)
    _morton_packed_k!(be)(packed, px, parts.py, parts.pz, Int32(N), UInt32(N - 1); ndrange = Np)
    PoissonKernels.KA.synchronize(be)
    sort!(packed)                                                  # radix by (Morton<<32 | idx)
    perm = Int32.((packed .& 0x00000000ffffffff) .+ UInt64(1))     # 1-based gather permutation
    for f in keys(parts)
        a = getfield(parts, f)
        a isa AbstractVector && (a .= a[perm])
    end
    return parts
end
