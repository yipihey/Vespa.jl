# Ideal-gas equation of state and conserved↔primitive conversion.
#
# Conserved vector U  = (ρ, ρvx, ρvy, ρvz, E)
# Primitive vector W  = (ρ,  vx,  vy,  vz, p)
# with  E = p/(γ-1) + ½ρ|v|².

"Convert a conserved state `U` to primitive `W` for adiabatic index `γ`."
@inline function cons2prim(U::NTuple{5,T}, γ) where {T}
    ρ = U[1]
    inv = one(T) / ρ
    vx = U[2] * inv
    vy = U[3] * inv
    vz = U[4] * inv
    kinetic = T(0.5) * ρ * (vx * vx + vy * vy + vz * vz)
    p = (γ - 1) * (U[5] - kinetic)
    return (ρ, vx, vy, vz, p)
end

"Convert a primitive state `W` to conserved `U` for adiabatic index `γ`."
@inline function prim2cons(W::NTuple{5,T}, γ) where {T}
    ρ, vx, vy, vz, p = W
    E = p / (γ - 1) + T(0.5) * ρ * (vx * vx + vy * vy + vz * vz)
    return (ρ, ρ * vx, ρ * vy, ρ * vz, E)
end

"Adiabatic sound speed for primitive state `W`."
@inline sound_speed(W::NTuple{5,T}, γ) where {T} = sqrt(γ * W[5] / W[1])
