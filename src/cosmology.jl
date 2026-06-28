# Cosmology: comoving coordinates with an expanding background, in **Enzo-compatible
# code units**. This file is Phase C1 — the unit system and the background
# expansion a(t) only (no hydro/gravity coupling yet; that is C2/C3).
#
# All formulas are transcribed from classic Enzo (src/enzo/) so a run set up with
# the same (Ω, H₀, box, z) gives numbers directly comparable to Enzo:
#   * CosmologyGetUnits.C              — the code-unit factors (a-normalized to 1 at z_i)
#   * CosmologyComputeExpansionFactor.C — the Friedmann ODE in code time units
#   * CosmologyComputeExpansionTimestep.C — the expansion timestep limiter
#   * Grid_ComovingExpansionTerms.C    — the semi-implicit Hubble drag (used in C2)
#
# Convention: the expansion factor `a` is normalized to **1 at the initial
# redshift** z_i (Enzo's `uaye = 1/(1+z_i)` absorbs the rest). Code TIME has its
# zero at the Big Bang (a = 0), matching Enzo's `InitialTimeInCodeUnits`; the
# simulation clock `sim.t` starts at 0 ≙ z_i, so absolute code time is
# `t_initial + sim.t`.

# ── physical constants (CGS), matching src/enzo/phys_constants.h ──────────────
const GRAV_CONST_CGS = 6.67428e-8        # cm³ g⁻¹ s⁻²
const MPC_CM         = 3.0857e24         # cm
const KM_CM          = 1.0e5             # cm

"""
    Cosmology

Background cosmology + Enzo-compatible code units for a comoving run. Holds the
density parameters, H₀ (in units of 100 km/s/Mpc), the comoving box size (Mpc/h),
the initial/final redshifts, and the expansion-timestep safety factor, plus the
mutable expansion state `(a, dadt)` cached at the current time and the absolute
code times `t_initial` (≙ z_i) and `t_final` (≙ z_f). Attach to a `Simulation`
with [`enable_cosmology!`](@ref); the driver consults `sim.cosmo` (default
`nothing` ⇒ ordinary non-cosmological hydro).
"""
mutable struct Cosmology
    OmegaMatter::Float64
    OmegaLambda::Float64
    OmegaRadiation::Float64
    HubbleConstantNow::Float64     # h, in 100 km/s/Mpc
    ComovingBoxSize::Float64       # Mpc/h (comoving)
    InitialRedshift::Float64
    FinalRedshift::Float64
    MaxExpansionRate::Float64      # dt ≤ MaxExpansionRate·a/ȧ (default 0.01)
    # cached expansion state at the current absolute code time:
    a::Float64
    dadt::Float64
    # absolute code times (zero at the Big Bang, a=0):
    t_initial::Float64             # ≙ z_i (Enzo InitialTimeInCodeUnits)
    t_final::Float64               # ≙ z_f
end

# Curvature closes the energy budget: Ω_k = 1 − Ω_m − Ω_Λ − Ω_r.
@inline omega_curvature(c::Cosmology) =
    1.0 - c.OmegaMatter - c.OmegaLambda - c.OmegaRadiation

# Scale factor ↔ redshift (a = 1 at z_i, by construction).
@inline scale_factor_at_redshift(c::Cosmology, z) = (1 + c.InitialRedshift) / (1 + z)
@inline redshift_at_scale_factor(c::Cosmology, a) = (1 + c.InitialRedshift) / a - 1
"Current redshift from the cached expansion factor."
@inline redshift(c::Cosmology) = redshift_at_scale_factor(c, c.a)

# ── Friedmann ODE in code time units (CosmologyComputeExpansionFactor.C:56-60) ─
# da/dt = √[ (2/(3 Ω_m a)) · (Ω_m + Ω_k·ã + Ω_Λ·ã³ + Ω_r/ã) ],  ã ≡ a/(1+z_i).
# Autonomous in t (no explicit time), so code-time's zero point is a free offset
# which we fix at the Big Bang (a=0) to match Enzo's absolute times.
@inline function friedmann_dadt(c::Cosmology, a::Float64)
    ã = a / (1 + c.InitialRedshift)
    Ωk = omega_curvature(c)
    return sqrt((2.0 / (3.0 * c.OmegaMatter * a)) *
                (c.OmegaMatter + Ωk * ã + c.OmegaLambda * ã^3 + c.OmegaRadiation / ã))
end

# Absolute code time at expansion factor `a`: t(a) = ∫₀ᵃ da′/(da/dt). The
# integrand 1/(da/dt) ∝ √a′ has a square-root cusp at a′=0 that Simpson resolves
# poorly; the substitution a′=s² (da′=2s ds) removes it — the transformed
# integrand g(s)=2s/(da/dt)(s²) ∝ s² is smooth (and 0) at s=0 — so composite
# Simpson over [0,√a] is accurate over the full a∈[0,100] range a run can span.
# Returns code time with t(a=0)=0.
function time_from_scale_factor(c::Cosmology, a::Float64; nint::Int = 2048)
    a <= 0 && return 0.0
    n = iseven(nint) ? nint : nint + 1
    smax = sqrt(a)
    h = smax / n
    g(s) = s <= 0 ? 0.0 : 2.0 * s / friedmann_dadt(c, s * s)
    acc = g(0.0) + g(smax)
    @inbounds for i in 1:(n - 1)
        acc += (isodd(i) ? 4.0 : 2.0) * g(i * h)
    end
    return acc * h / 3.0
end

"""
    expansion_at(c, t_abs) -> (a, dadt)

Expansion factor and its time derivative at absolute code time `t_abs` (zero at
the Big Bang). Inverts the monotonic `time_from_scale_factor` by Newton iteration
(derivative is exactly `da/dt`), so it is pure and re-entrant — no dependence on
call order. Used for the n+½ and n+1 expansion factors within a step.
"""
function expansion_at(c::Cosmology, t_abs::Float64)
    a = c.a                                   # warm start from the cached value
    for _ in 1:60
        da_dt = friedmann_dadt(c, a)
        resid = time_from_scale_factor(c, a) - t_abs
        Δ = resid * da_dt                     # Newton: a -= resid / (dt/da) = resid·(da/dt)
        a -= Δ
        a = a < 1e-12 ? 1e-12 : a
        abs(Δ) <= 1e-13 * a && break
    end
    return a, friedmann_dadt(c, a)
end

"Update the cached `(a, dadt)` to absolute code time `t_abs`."
function set_expansion!(c::Cosmology, t_abs::Float64)
    c.a, c.dadt = expansion_at(c, t_abs)
    return c.a
end

# ── Enzo-compatible code units (CosmologyGetUnits.C) ──────────────────────────
# Two factors (density, length) are redshift-dependent — evaluate at the `a` of
# interest. Velocity/temperature/time use the *initial* redshift (fixed).
"""
    cosmology_units(c, a) -> NamedTuple

CGS code-unit factors at expansion factor `a`: `density` (g/cm³), `length` (cm),
`velocity` (cm/s), `time` (s), `temperature` (K, for μ=1). Transcribed from
`CosmologyGetUnits.C`; `density` and `length` carry the (1+z) scaling.
"""
function cosmology_units(c::Cosmology, a::Float64)
    z   = redshift_at_scale_factor(c, a)
    zi1 = 1 + c.InitialRedshift
    h   = c.HubbleConstantNow
    box = c.ComovingBoxSize
    density     = 1.8788e-29 * c.OmegaMatter * h^2 * (1 + z)^3
    length      = MPC_CM * box / h / (1 + z)
    time        = 2.519445e17 / sqrt(c.OmegaMatter) / h / zi1^1.5
    velocity    = 1.22475e7 * box * sqrt(c.OmegaMatter) * sqrt(zi1)
    temperature = 1.81723e6 * box^2 * c.OmegaMatter * zi1
    return (density = density, length = length, velocity = velocity,
            time = time, temperature = temperature)
end

"""
    gravitational_constant_code(c) -> Float64

The code-units `4πG` (`GravitationalConstant = 4πG·ρunit·tunit²`). By Enzo's unit
construction this is ≈ 1 — the "4πG = 1" cosmology normalization the comoving
Poisson solve relies on. Evaluated at the initial redshift.
"""
function gravitational_constant_code(c::Cosmology)
    u = cosmology_units(c, 1.0)              # a = 1 at z_i
    return 4.0 * π * GRAV_CONST_CGS * u.density * u.time^2
end

# ── expansion timestep limiter (CosmologyComputeExpansionTimestep.C:47) ───────
"Maximum step from the expansion limiter at the cached state: MaxExpansionRate·a/ȧ."
@inline expansion_dt(c::Cosmology) = c.MaxExpansionRate * c.a / c.dadt

"""
    Cosmology(; OmegaMatter, OmegaLambda, OmegaRadiation=0, HubbleConstantNow,
              ComovingBoxSize, InitialRedshift, FinalRedshift, MaxExpansionRate=0.01)

Build a background cosmology, computing the absolute code times at z_i and z_f
(Big Bang ≙ t=0) and caching the initial expansion state (a=1 at z_i). The
simulation clock `sim.t` runs 0 → `t_final − t_initial`; set the problem's
`tfinal` to that span for a run from z_i to z_f.
"""
function Cosmology(; OmegaMatter::Real, OmegaLambda::Real, OmegaRadiation::Real = 0.0,
                   HubbleConstantNow::Real, ComovingBoxSize::Real,
                   InitialRedshift::Real, FinalRedshift::Real,
                   MaxExpansionRate::Real = 0.01)
    c = Cosmology(Float64(OmegaMatter), Float64(OmegaLambda), Float64(OmegaRadiation),
                  Float64(HubbleConstantNow), Float64(ComovingBoxSize),
                  Float64(InitialRedshift), Float64(FinalRedshift),
                  Float64(MaxExpansionRate), 1.0, 0.0, 0.0, 0.0)
    c.t_initial = time_from_scale_factor(c, 1.0)                    # a = 1 at z_i
    c.t_final   = time_from_scale_factor(c, scale_factor_at_redshift(c, c.FinalRedshift))
    set_expansion!(c, c.t_initial)                                 # cache (a=1, ȧ) at z_i
    return c
end

"Code-time span from z_i to z_f. `evolve!` integrates to this when cosmology is on."
@inline cosmology_tfinal(c::Cosmology) = c.t_final - c.t_initial

# ── Hubble expansion source terms (Grid_ComovingExpansionTerms.C) ─────────────
"""
    apply_expansion_terms!(sim, dt, a, dadt)

Apply the comoving expansion (Hubble drag) to the conserved state as a
semi-implicit operator-split step, with `a`, `dadt` time-centered at n+½. With
`C = dt·ȧ/a`, Enzo's scheme (VELOCITY_METHOD3 / ENERGY_METHOD3) multiplies the
*peculiar velocity* by `(1−½C)/(1+½C)` and the *specific total energy* by
`(1−C)/(1+C)`. In conserved-density variables (density is unchanged by the drag)
this is exactly: momentum density ×`(1−½C)/(1+½C)`, energy density ×`(1−C)/(1+C)`.
The energy factor is exact for γ=5/3 (Enzo's note: "extra term missing if γ≠5/3").
"""
function apply_expansion_terms!(sim::Simulation, dt::Real, a::Real, dadt::Real)
    C  = dt * dadt / a                         # expansion bookkeeping (geometry/time precision)
    fv = (1.0 - 0.5 * C) / (1.0 + 0.5 * C)     # peculiar-velocity redshift (rate ȧ/a)
    fe = (1.0 - C) / (1.0 + C)                 # total energy (rate 2ȧ/a)
    mom = momentum_indices(sim.model); ei = energy_index(sim.model)
    sv = sim.sv
    Tf = _Tf(sim); fvT = Tf(fv); feT = Tf(fe)  # → field precision at the boundary
    for_each_cell(sim.backend) do c
        U = get_U(sv, c)
        # momenta redshift, energy decays, all other variables unchanged
        set_U!(sv, c, ntuple(k -> k in mom ? U[k] * fvT : (k == ei ? U[k] * feT : U[k]),
                             nvars_val(sim.model)))
    end
    return nothing
end

"""
    enable_cosmology!(sim; OmegaMatter, OmegaLambda, OmegaRadiation=0,
                      HubbleConstantNow, ComovingBoxSize, InitialRedshift,
                      FinalRedshift, MaxExpansionRate=0.01, gravity=true) -> Cosmology

Attach a comoving background to `sim` (so `evolve!`/`step!` run in comoving
coordinates with the Hubble expansion terms and the 1/a flux scaling) and return
it. The expansion factor starts at a=1 (z_i) and `evolve!` integrates to z_f.
When `gravity` is true (default) and self-gravity is not already on, comoving
self-gravity is enabled with the 4πG=1 normalization (`G = 1/4π`, periodic,
overdensity source).
"""
function enable_cosmology!(sim::Simulation; OmegaMatter::Real, OmegaLambda::Real,
                           OmegaRadiation::Real = 0.0, HubbleConstantNow::Real,
                           ComovingBoxSize::Real, InitialRedshift::Real,
                           FinalRedshift::Real, MaxExpansionRate::Real = 0.01,
                           gravity::Bool = true, ka = nothing)
    c = Cosmology(; OmegaMatter, OmegaLambda, OmegaRadiation, HubbleConstantNow,
                  ComovingBoxSize, InitialRedshift, FinalRedshift, MaxExpansionRate)
    sim.cosmo = c
    set_expansion!(c, c.t_initial)                         # a = 1 at sim.t = 0
    if gravity && sim.grav === nothing
        # comoving Poisson: 4πG = 1; `ka` selects the GPU/CPU device CG.
        enable_gravity!(sim; G = 1.0 / (4π), bcs = Periodic(), ka = ka)
    end
    return c
end

# ── Compton drag (gas ↔ CMB thermal coupling; Grid_ComptonCoolingTerms / cool1d) ─
# Inverse-Compton scattering off CMB photons relaxes the gas temperature toward
# T_cmb = T_cmb0·(1+z) at the rate Γ_C = (8 σ_T a_rad T_cmb⁴)/(3 m_e c)·x_e (per
# free electron), which scales as (1+z)⁴ and so tightly couples gas to the CMB at
# high z. Applied as an operator-split semi-implicit relaxation on the gas thermal
# energy: T ← (T + Γ_C·dt·T_cmb)/(1 + Γ_C·dt) (unconditionally stable, exact for
# constant T_cmb). The free-electron fraction `electron_fraction = n_e/n_H` and mean
# molecular weight `mu` are inputs here (they come from ChemistryKernels in a full
# run); density is unchanged, kinetic energy untouched — only thermal energy decays.
const SIGMA_T_CGS = 6.6524587e-25      # Thomson cross section, cm²
const A_RAD_CGS   = 7.5657e-15         # radiation constant, erg cm⁻³ K⁻⁴
const M_E_CGS     = 9.10938e-28        # electron mass, g
const C_CGS       = 2.99792458e10      # cm/s
const K_B_CGS     = 1.380649e-16       # erg/K
const M_H_CGS     = 1.6726e-24         # hydrogen mass, g
const T_CMB0_K    = 2.725              # CMB temperature today, K

"Physical gas temperature (K) from a cell's conserved state, given μ and the units."
@inline function _cell_temperature(sim::Simulation, U, γ, mu::Float64, vel_unit::Float64)
    di = density_index(sim.model); ei = energy_index(sim.model); mom = momentum_indices(sim.model)
    ρ = U[di]
    ke = 0.0
    @inbounds for d in 1:rank(sim.backend)
        ke += U[mom[d]]^2
    end
    ke *= 0.5 / ρ
    e_spec = (U[ei] - ke) / ρ                              # code specific thermal energy
    return (γ - 1) * mu * M_H_CGS / K_B_CGS * e_spec * vel_unit^2, ke   # (T[K], kinetic density)
end

"""
    apply_compton_drag!(sim, dt; electron_fraction=1.0, mu=0.6)

Relax the gas temperature toward `T_cmb = 2.725(1+z)` by inverse-Compton scattering
over a code-time step `dt` (semi-implicit; density and kinetic energy unchanged).
`electron_fraction` (= n_e/n_H) and `mu` parametrize the ionization state. No-op
without cosmology.
"""
function apply_compton_drag!(sim::Simulation, dt::Real; electron_fraction::Real = 1.0,
                             mu::Real = 0.6)
    cosmo = sim.cosmo
    cosmo === nothing && return nothing
    a = cosmo.a
    z = redshift_at_scale_factor(cosmo, a)
    u = cosmology_units(cosmo, a)
    Tcmb = T_CMB0_K * (1 + z)
    ΓC = 8 * SIGMA_T_CGS * A_RAD_CGS * Tcmb^4 / (3 * M_E_CGS * C_CGS) * Float64(electron_fraction)
    dt_s = Float64(dt) * u.time                            # code dt → seconds
    fac = ΓC * dt_s
    γ = sim.γ; muf = Float64(mu)
    ei = energy_index(sim.model)
    e_to_T = (γ - 1) * muf * M_H_CGS / K_B_CGS * u.velocity^2   # specific-energy → T[K]
    Tf = _Tf(sim)
    for_each_cell(sim.backend) do c
        U = get_U(sim.sv, c)
        Tgas, ke = _cell_temperature(sim, U, γ, muf, u.velocity)
        Tnew = (Tgas + fac * Tcmb) / (1 + fac)             # semi-implicit relaxation
        ρ = U[density_index(sim.model)]
        eth_new = Tnew / e_to_T * ρ                        # back to thermal energy density
        set_U!(sim.sv, c, ntuple(k -> k == ei ? Tf(ke + eth_new) : U[k], nvars_val(sim.model)))
        return nothing
    end
    return nothing
end

# ── native cosmology driver (Phase 4: the full stack, no Enzo) ────────────────
"""
    evolve_cosmology!(sim; a_final, policy=nothing, regrid_every=0, compton=false,
                      electron_fraction=1.0, mu=0.6, maxstep=100_000, verbose=false) -> nsteps

Drive a self-gravitating dark-matter run on the native KA/Julia stack from the
current expansion factor to `a_final`, with adaptive refinement. Each step takes an
expansion-limited substep, pushes the particles (comoving KDK; gravity solved via
`sim.grav`, GPU when `grav.ka` is set), optionally applies Compton drag to the gas,
and advances the background. With a `policy` (and `regrid_every > 0`) it regrids on
the refinement indicator every `regrid_every` steps — e.g. DM overdensity for the
cosmological zoom. No Enzo: hierarchy, reflux, gravity, particles, and expansion are
all Vespa's own. (Gas hydro + chemistry coupling under one clock is the remaining
integration; this driver is the gravitational/expansion backbone.)
"""
function evolve_cosmology!(sim::Simulation; a_final::Real, policy = nothing,
                           regrid_every::Integer = 0, compton::Bool = false,
                           electron_fraction::Real = 1.0, mu::Real = 0.6,
                           maxstep::Integer = 100_000, verbose::Bool = false, ka = nothing)
    cosmo = sim.cosmo
    cosmo === nothing && error("evolve_cosmology! requires enable_cosmology!")
    sim.grav === nothing && error("evolve_cosmology! requires self-gravity")
    ka === nothing || (sim.grav.ka = ka)                # device Poisson + push when ka given
    step = 0
    solve_poisson!(sim, sim.grav)                       # φ for the initial state
    while cosmo.a < a_final && step < maxstep
        if policy !== nothing && regrid_every > 0 && step % regrid_every == 0
            n = regrid!(sim, policy)
            verbose && n > 0 && @printf("  step %d: regrid +%d (max_level %d)\n",
                                        step, n, max_level(sim.backend))
        end
        dt = expansion_dt(cosmo)
        a_next, _ = expansion_at(cosmo, cosmo.t_initial + sim.t + dt)
        a_next > a_final && (dt *= (a_final - cosmo.a) / (a_next - cosmo.a))  # land on a_final
        sim.particles === nothing || push_particles!(sim, dt; ka = ka)
        compton && apply_compton_drag!(sim, dt; electron_fraction = electron_fraction, mu = mu)
        sim.t += dt
        set_expansion!(cosmo, cosmo.t_initial + sim.t)
        step += 1
    end
    verbose && @printf("evolve_cosmology!: %d steps to a=%.4f (z=%.3f)\n",
                       step, cosmo.a, redshift(cosmo))
    return step
end

# ── Zel'dovich pancake (the C3 gate; ZeldovichPancakeInitialize.C) ────────────
# Newton-solve ξ from  x = ξ − (A/kx)·sin(kx·ξ)  (the Zel'dovich Lagrange↔Euler
# map; unique for A < 1). The grid coordinate `x` is Eulerian; ξ is the matching
# Lagrangian coordinate whose displaced position is x.
@inline function _zeldovich_xi(x::Float64, A::Float64, kx::Float64)
    ξ = x
    for _ in 1:100
        f  = ξ - A * sin(kx * ξ) / kx - x
        df = 1.0 - A * cos(kx * ξ)
        Δ  = f / df
        ξ -= Δ
        abs(Δ) < 1e-13 && break
    end
    return ξ
end

"""
    zeldovich_state(x, a; zi, zc, kx, OmegaBaryon=1.0, center=0.5) -> (ρ, v)

Analytic Zel'dovich (ρ, peculiar v) in code units at Eulerian position `x` and
expansion factor `a` (EdS growing mode, a=1 at z_i). The dimensionless amplitude
`A(a) = a·(1+z_c)/(1+z_i)` reaches 1 (caustic) at z = z_c; the code-unit velocity
amplitude grows as `−√(2/3)·√a·(1+z_c)/((1+z_i)·kx)`. The perturbation is centered
on `center` (default the box midpoint 0.5) so mass converges to a density peak
there (the pancake) with a void at the periodic edges. This is both the initial
condition (at a=1) and the oracle the evolved gas is checked against.
"""
function zeldovich_state(x::Float64, a::Float64; zi::Real, zc::Real, kx::Real,
                         OmegaBaryon::Real = 1.0, center::Real = 0.5)
    A    = a * (1 + zc) / (1 + zi)
    vamp = -sqrt(2 / 3) * sqrt(a) * (1 + zc) / ((1 + zi) * kx)
    ξ    = _zeldovich_xi(x - center, Float64(A), Float64(kx))
    ρ    = OmegaBaryon / (1 - A * cos(kx * ξ))
    v    = vamp * sin(kx * ξ)
    return ρ, v
end

"""
    zeldovich_pancake(; n, FinalRedshift, InitialRedshift=20, CollapseRedshift=1,
                      HubbleConstantNow=0.5, ComovingBoxSize=64, OmegaBaryon=1,
                      InitialTemperature=100, γ=5/3) -> (problem, cosmology_kwargs)

Build the 1D Zel'dovich-pancake `Problem` (EdS, unit-length box, one wave) and the
keyword tuple for [`enable_cosmology!`](@ref). The initial gas is the Zel'dovich
state at a=1 with a cold adiabatic pressure `p = T_code·ρ·(ρ/Ω_b)^{γ-1}`
(`T_code = InitialTemperature/TemperatureUnits`). Caller: build the `Simulation`,
then `enable_cosmology!(sim; cosmology_kwargs...)`.
"""
function zeldovich_pancake(; n::Integer, FinalRedshift::Real,
                           InitialRedshift::Real = 20.0, CollapseRedshift::Real = 1.0,
                           HubbleConstantNow::Real = 0.5, ComovingBoxSize::Real = 64.0,
                           OmegaBaryon::Real = 1.0, InitialTemperature::Real = 100.0,
                           γ::Real = 5 / 3)
    kx = 2π                                           # one wave across the [0,1] box
    cosmo_kwargs = (OmegaMatter = 1.0, OmegaLambda = 0.0,
                    HubbleConstantNow = HubbleConstantNow, ComovingBoxSize = ComovingBoxSize,
                    InitialRedshift = InitialRedshift, FinalRedshift = FinalRedshift)
    Tunit = cosmology_units(Cosmology(; cosmo_kwargs...), 1.0).temperature
    Tcode = InitialTemperature / Tunit
    function init(x, y, z)
        ρ, v = zeldovich_state(Float64(x), 1.0; zi = InitialRedshift, zc = CollapseRedshift,
                               kx = kx, OmegaBaryon = OmegaBaryon)
        p = Tcode * ρ * (ρ / OmegaBaryon)^(γ - 1)     # cold adiabatic pressure
        return (ρ, v, 0.0, 0.0, p)
    end
    prob = Problem(; name = "ZeldovichPancake", dims = (Int(n),), domain = ((0.0, 1.0),),
                   γ = γ, bcs = Periodic(), init = init, tfinal = 1.0, cfl = 0.3)
    return prob, cosmo_kwargs
end
