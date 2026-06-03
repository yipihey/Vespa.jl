# Ideal-gas equation of state and conserved↔primitive conversion.
#
# Conserved vector U  = (ρ, ρvx, ρvy, ρvz, E)
# Primitive vector W  = (ρ,  vx,  vy,  vz, p)
# with  E = p/(γ-1) + ½ρ|v|².

"Convert a conserved state `U` to primitive `W` for adiabatic index `γ`."
@inline function cons2prim(U::NTuple{5,T}, γ) where {T}
    g = T(γ)                          # γ at the field precision ⇒ homogeneous-T output
    ρ = U[1]
    inv = one(T) / ρ
    vx = U[2] * inv
    vy = U[3] * inv
    vz = U[4] * inv
    kinetic = T(0.5) * ρ * (vx * vx + vy * vy + vz * vz)
    p = (g - one(T)) * (U[5] - kinetic)
    # Pressure floor. Without a dual-energy formalism, in very cold supersonic flow
    # (e.g. the Zel'dovich pancake, thermal energy orders of magnitude below kinetic)
    # the difference U[5]−kinetic loses all precision and p can go ≤ 0. A *strictly
    # positive* floor is required: p=0 gives c=0, which makes the HLLC contact-speed
    # denominator ρL(SL−unL)−ρR(SR−unR) degenerate to 0/0 when velocities straddle
    # zero. Floor to a negligible (1e-12) fraction of the kinetic energy density —
    # scale-free, keeps c>0, and is a no-op wherever the gas is resolved (p ≫ floor).
    return (ρ, vx, vy, vz, max(p, T(1e-12) * kinetic))
end

"Convert a primitive state `W` to conserved `U` for adiabatic index `γ`."
@inline function prim2cons(W::NTuple{5,T}, γ) where {T}
    ρ, vx, vy, vz, p = W
    E = p / (T(γ) - one(T)) + T(0.5) * ρ * (vx * vx + vy * vy + vz * vz)
    return (ρ, ρ * vx, ρ * vy, ρ * vz, E)
end

"Adiabatic sound speed for primitive state `W`."
@inline sound_speed(W::NTuple{5,T}, γ) where {T} = sqrt(T(γ) * W[5] / W[1])
