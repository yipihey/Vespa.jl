"""
    arepo_direct_gravity_accel!(ax, ay, az, x, y, z, m; G=1.0, softening=0.0, periodic=false)

Compute direct Newtonian accelerations for a small particle set stored in
structure-of-arrays form. The input arrays `x`, `y`, `z`, and `m` must all have
the same length `N`, and the output arrays `ax`, `ay`, and `az` must be
preallocated to that same length.

The current scaffold is intentionally narrow:

- `periodic=false` only
- scalar softening only
- `O(N^2)` pair loop suitable for tiny oracle gates, not production PM/tree use
- pure Julia with no package/export wiring
"""
function arepo_direct_gravity_accel!(ax::AbstractVector{Ta},
                                     ay::AbstractVector{Ta},
                                     az::AbstractVector{Ta},
                                     x::AbstractVector{Tx},
                                     y::AbstractVector{Ty},
                                     z::AbstractVector{Tz},
                                     m::AbstractVector{Tm};
                                     G::Real = 1.0,
                                     softening::Real = 0.0,
                                     periodic::Bool = false) where {Ta,Tx,Ty,Tz,Tm}
    n = _arepo_direct_gravity_check_lengths(ax, ay, az, x, y, z, m)
    periodic &&
        error("arepo_direct_gravity_accel!: periodic=true is not implemented in this scaffold")
    softening >= 0 || error("arepo_direct_gravity_accel!: softening must be nonnegative")

    Gf = float(G)
    eps2 = float(softening)^2
    fill!(ax, zero(Ta))
    fill!(ay, zero(Ta))
    fill!(az, zero(Ta))

    @inbounds for i in 1:n-1
        xi = x[i]
        yi = y[i]
        zi = z[i]
        mi = m[i]
        for j in i+1:n
            dx = x[j] - xi
            dy = y[j] - yi
            dz = z[j] - zi
            r2 = dx * dx + dy * dy + dz * dz + eps2
            invr = inv(sqrt(r2))
            invr3 = invr * invr * invr
            pair = Gf * invr3

            ai = pair * m[j]
            aj = pair * mi

            ax[i] += ai * dx
            ay[i] += ai * dy
            az[i] += ai * dz

            ax[j] -= aj * dx
            ay[j] -= aj * dy
            az[j] -= aj * dz
        end
    end

    return ax, ay, az
end

"""
    arepo_direct_gravity_accel(x, y, z, m; T=promote_type(...), kwargs...)

Allocate acceleration arrays and call `arepo_direct_gravity_accel!`.
"""
function arepo_direct_gravity_accel(x::AbstractVector,
                                    y::AbstractVector,
                                    z::AbstractVector,
                                    m::AbstractVector;
                                    T = promote_type(eltype(x), eltype(y), eltype(z),
                                                     eltype(m), Float64),
                                    kwargs...)
    n = _arepo_direct_gravity_check_lengths(x, y, z, m)
    ax = zeros(T, n)
    ay = zeros(T, n)
    az = zeros(T, n)
    arepo_direct_gravity_accel!(ax, ay, az, x, y, z, m; kwargs...)
    return ax, ay, az
end

"""
    arepo_direct_gravity_potential_energy(x, y, z, m; G=1.0, softening=0.0, periodic=false)

Return the total pairwise gravitational potential energy for the same direct
tiny-`N` scaffold used by `arepo_direct_gravity_accel!`.
"""
function arepo_direct_gravity_potential_energy(x::AbstractVector,
                                               y::AbstractVector,
                                               z::AbstractVector,
                                               m::AbstractVector;
                                               G::Real = 1.0,
                                               softening::Real = 0.0,
                                               periodic::Bool = false)
    n = _arepo_direct_gravity_check_lengths(x, y, z, m)
    periodic &&
        error("arepo_direct_gravity_potential_energy: periodic=true is not implemented in this scaffold")
    softening >= 0 ||
        error("arepo_direct_gravity_potential_energy: softening must be nonnegative")

    Gf = float(G)
    eps2 = float(softening)^2
    energy = 0.0

    @inbounds for i in 1:n-1
        xi = x[i]
        yi = y[i]
        zi = z[i]
        mi = m[i]
        for j in i+1:n
            dx = x[j] - xi
            dy = y[j] - yi
            dz = z[j] - zi
            r2 = dx * dx + dy * dy + dz * dz + eps2
            invr = inv(sqrt(r2))
            energy -= Gf * mi * m[j] * invr
        end
    end

    return energy
end

"""
    arepo_direct_gravity_oracle(x, y, z, m; kwargs...)

Convenience wrapper returning a named tuple with accelerations and total
potential energy.
"""
function arepo_direct_gravity_oracle(x::AbstractVector,
                                     y::AbstractVector,
                                     z::AbstractVector,
                                     m::AbstractVector;
                                     kwargs...)
    ax, ay, az = arepo_direct_gravity_accel(x, y, z, m; kwargs...)
    pe = arepo_direct_gravity_potential_energy(x, y, z, m; kwargs...)
    return (ax = ax, ay = ay, az = az, potential_energy = pe)
end

"""
    arepo_direct_gravity_kick_drift_step(x, y, z, m, vx, vy, vz; dt, kwargs...)

Advance a frozen tiny-`N` particle set with a single kick-drift step.
The helper is intentionally scalar and allocation-friendly enough for smoke
tests, not for production integration loops.
"""
function arepo_direct_gravity_kick_drift_step(x::AbstractVector,
                                              y::AbstractVector,
                                              z::AbstractVector,
                                              m::AbstractVector,
                                              vx::AbstractVector,
                                              vy::AbstractVector,
                                              vz::AbstractVector;
                                              dt::Real,
                                              kwargs...)
    n = _arepo_direct_gravity_check_lengths(x, y, z, m, vx, vy, vz)
    dtf = float(dt)
    oracle = arepo_direct_gravity_oracle(x, y, z, m; kwargs...)

    x1 = similar(x, promote_type(eltype(x), Float64), n)
    y1 = similar(y, promote_type(eltype(y), Float64), n)
    z1 = similar(z, promote_type(eltype(z), Float64), n)
    vx1 = similar(vx, promote_type(eltype(vx), Float64), n)
    vy1 = similar(vy, promote_type(eltype(vy), Float64), n)
    vz1 = similar(vz, promote_type(eltype(vz), Float64), n)
    m1 = copy(m)

    @inbounds for i in 1:n
        vx1[i] = float(vx[i]) + dtf * oracle.ax[i]
        vy1[i] = float(vy[i]) + dtf * oracle.ay[i]
        vz1[i] = float(vz[i]) + dtf * oracle.az[i]
        x1[i] = float(x[i]) + dtf * vx1[i]
        y1[i] = float(y[i]) + dtf * vy1[i]
        z1[i] = float(z[i]) + dtf * vz1[i]
    end

    net_force = (
        x = sum(m .* oracle.ax),
        y = sum(m .* oracle.ay),
        z = sum(m .* oracle.az),
    )
    max_abs_accel = maximum(sqrt.(oracle.ax .* oracle.ax .+
                                  oracle.ay .* oracle.ay .+
                                  oracle.az .* oracle.az))
    return (
        x = x1,
        y = y1,
        z = z1,
        m = m1,
        vx = vx1,
        vy = vy1,
        vz = vz1,
        ax = oracle.ax,
        ay = oracle.ay,
        az = oracle.az,
        potential_energy = oracle.potential_energy,
        momentum_residual = net_force,
        max_abs_accel = max_abs_accel,
        dt = dtf,
    )
end

function _arepo_direct_gravity_check_lengths(arrays::AbstractVector...)
    isempty(arrays) && error("_arepo_direct_gravity_check_lengths: expected at least one array")
    n = length(arrays[1])
    for array in arrays
        length(array) == n ||
            error("direct gravity scaffold: all arrays must share the same length")
    end
    return n
end
