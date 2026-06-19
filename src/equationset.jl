# The EQUATION SET — the swappable physics/variable model (ADR: physics is data).
#
# Vespa's conservative finite-volume driver (flux accumulation, SSP-RK2,
# refluxing) is generic over the *equation set*: it asks the model for the
# variable count, the conserved-field names, the role indices (which components
# are density / momentum / energy — for gravity & cosmology source terms), and
# the physics kernels (cons2prim/prim2cons/sound_speed/Riemann flux). The choice
# of hydro variables is therefore a value, not a hardcoded global — so MHD
# (+B,+φ), dual-energy (+gas energy), multispecies (+chemistry), or a different
# primitive/energy formulation are new `EquationSet`s, not core surgery.
#
# `IdealHydro` is the default and reproduces the prior 5-variable ideal-gas core
# bit-for-bit. The variable count is a static (`nvars`), so conserved tuples are
# built `Val`-sized and the driver stays type-stable.

"""
    EquationSet

Supertype of every physics/variable model. A model declares its conserved
variable count and names, the role indices used by source terms, and the
hydro kernels. Implement the generic functions below for a new equation set.
"""
abstract type EquationSet end

"""
    IdealHydro(γ)

Adiabatic ideal-gas hydrodynamics. Conserved `(ρ, ρvx, ρvy, ρvz, E)`, primitive
`(ρ, vx, vy, vz, p)`, HLLC flux. The default `EquationSet`.
"""
struct IdealHydro <: EquationSet
    γ::Float64
end

# ── the EquationSet interface (default = IdealHydro) ─────────────────────────
"Number of conserved variables (static for a concrete model ⇒ Val-sized tuples)."
@inline nvars(::IdealHydro) = 5
"Names of the conserved fields, in order (drives `FieldSpec`/field allocation)."
@inline conserved_names(::IdealHydro) =
    (:density, :momentum_x, :momentum_y, :momentum_z, :total_energy)
"Index of the mass-density component."
@inline density_index(::IdealHydro) = 1
"Indices of the momentum components (x,y,z) — used by gravity/cosmology sources."
@inline momentum_indices(::IdealHydro) = (2, 3, 4)
"Index of the total-energy component."
@inline energy_index(::IdealHydro) = 5
"Adiabatic index (EOS parameter), if the model has one."
@inline adiabatic_index(m::IdealHydro) = m.γ

"Primitive `W` from conserved `U` for this model."
@inline cons2prim(m::IdealHydro, U) = cons2prim(U, m.γ)        # eos.jl kernel
"Conserved `U` from primitive `W` for this model."
@inline prim2cons(m::IdealHydro, W) = prim2cons(W, m.γ)        # eos.jl kernel
"Sound speed for primitive `W`."
@inline sound_speed(m::IdealHydro, W) = sound_speed(W, m.γ)    # eos.jl kernel
"Riemann flux across a face normal to `axis`, from reconstructed states `WL`,`WR`."
@inline riemann_flux(m::IdealHydro, WL, WR, axis::Int) = hllc_flux(WL, WR, m.γ, axis)

# Convenience: a Val of the (static) variable count, for type-stable `ntuple`.
@inline nvars_val(m::EquationSet) = Val(nvars(m))
