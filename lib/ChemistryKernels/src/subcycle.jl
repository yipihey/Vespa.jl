# subcycle.jl — the per-cell sub-cycling driver: grackle's solve_rate_cool inner
# loop specialized to the v2026 reduced model.  Repeats, until the requested macro
# step `dt` is consumed, a self-limited sub-step `dtit` that:
#   1. evaluates T, all rates (with the Peebles k2 override), edot, and the
#      net e⁻/HI rates (rate_timestep_g);
#   2. sizes dtit to a ≤10% change in n_e, n_HI and the thermal energy
#      (solve_rate_cool_g.F:638-674, 782-824);
#   3. advances the energy — IMPLICITLY for the stiff CMB-Compton term when it is
#      stiff (K·Δt>1), else explicitly (solve_rate_cool_g.F:861-883);
#   4. advances the species by one backward-Euler sweep (`network_step`).
#
# Everything in physical CGS (number densities [cm⁻³], e [erg/g], t [s]).  State
# is carried in the grackle mass-equivalent y-convention (yH2I=2·n(H2) etc., as in
# network_step.jl), with the network "density" d = ρ/m_H so yHeI=(1−fh)·d.  Pure &
# allocation-free (isbits NamedTuples) ⇒ runs in a KA kernel and is AD-ready.

export build_rates, evolve_cell

const _SUB_ITMAX = 5_000         # subcycle cap (bounds GPU kernel time; well-behaved
                                 # cells converge in ≪100 steps — this is a watchdog)
const _SUB_TINY  = 1.0e-20

# A 10%-change sub-step from a rate, with no constraint (Inf) when the rate is
# ~0.  NOTE: unlike grackle (whose code-unit edot/dedot are O(1), so its absolute
# `tiny8` floor is harmless), our rates are PHYSICAL CGS (volumetric ė ~ 1e-30),
# so an absolute floor would EXCEED real rates and corrupt them — we guard the
# division instead, never the value.
@inline _step10(X, rate) = abs(rate) > zero(rate) ? abs(oftype(rate, 0.1) * X / rate) :
                           typemax(rate)

"""
    build_rates(T, Trad, nHI, Hz; deuterium=false)

NamedTuple of every reaction-rate coefficient the network needs, at gas
temperature `T` and CMB temperature `Trad`. k2 is the Peebles override (needs the
neutral-H density `nHI` and Hubble rate `Hz`); k27/k28 are the CMB photo-rates;
all others are the Wave-1 analytic fits. Pure.
"""
@inline function build_rates(T, Trad, nHI, Hz; deuterium::Bool = false)
    R = typeof(T)
    k2_val = peebles_k2(T, nHI, Hz)
    # C-weighted β₁s: matches k2=α_B×C so equilibrium gives true Saha (C cancels).
    # At high z (xe≈1, nHI≈0): C→1, k_beta1s→β₁s (drives Saha).
    # At z≈1200 (C≈0.006): k_beta1s negligible vs recombination → freeze-out preserved.
    k_b1s = beta1s_freq(T) * k2_val / (recfast_alpha(T) * R(1.0e6))
    # Helium Saha factors at the CMB temperature (cosmological photoionisation
    # equilibrium; → fully neutral He at low z, costing nothing for late-time gas).
    she1, she2 = helium_saha_pair(Trad)
    base = (; k1=k1(T), k2=k2_val, k3=k3(T), k4=k4(T), k5=k5(T),
            k6=k6(T), k7=k7(T), k8=k8(T), k9=k9(T), k10=k10(T), k11=k11(T),
            k12=k12(T), k13=k13(T), k14=k14(T), k15=k15(T), k16=k16(T), k17=k17(T),
            k18=k18(T), k19=k19(T), k22=k22(T), k57=k57(T), k58=k58(T),
            k27=k27_cmb(Trad), k28=k28_cmb(Trad), k_beta1s=k_b1s,
            she1=she1, she2=she2)
    deuterium || return base
    return merge(base, (; k50=k50(T), k51=k51(T), k52=k52(T), k53=k53(T),
                        k54=k54(T), k55=k55(T), k56=k56(T)))
end

# Net rate-of-change of n_e and n_HI (rate_timestep_g, molecular branch, reduced:
# no radiation/dust/shielding).  Used only to size the chemistry sub-step.
@inline function _de_hi_dot(yHI, yHII, yde, yH2I, yHM, yH2II, yHeI, yHeII, yHeIII, K)
    R = typeof(yHI)
    HIdot = -K.k1*yHI*yde - K.k7*yHI*yde - K.k8*yHM*yHI - K.k9*yHII*yHI -
            K.k10*yH2II*yHI/R(2) - R(2)*K.k22*yHI^3 + K.k2*yHII*yde +
            R(2)*K.k13*yHI*yH2I/R(2) + K.k11*yHII*yH2I/R(2) +
            R(2)*K.k12*yde*yH2I/R(2) + K.k14*yHM*yde + K.k15*yHM*yHI +
            R(2)*K.k16*yHM*yHII + R(2)*K.k18*yH2II*yde/R(2) + K.k19*yH2II*yHM/R(2) -
            K.k57*yHI*yHI - K.k58*yHI*yHeI/R(4) - K.k_beta1s*yHI
    dedot = K.k1*yHI*yde + K.k3*yHeI*yde/R(4) + K.k5*yHeII*yde/R(4) +
            K.k8*yHM*yHI + K.k15*yHM*yHI + K.k17*yHM*yHII + K.k14*yHM*yde -
            K.k2*yHII*yde - K.k4*yHeII*yde/R(4) - K.k6*yHeIII*yde/R(4) -
            K.k7*yHI*yde - K.k18*yH2II*yde/R(2) + K.k57*yHI*yHI + K.k58*yHI*yHeI/R(4) +
            K.k_beta1s*yHI
    return dedot, HIdot
end

"""
    evolve_cell(rho, e, HII_m, H2I_m, HDI_m, dt, z; hubble, Om, OL, fh, deuterium)

Sub-cycle one cell over macro-step `dt` [s].  `rho` = gas mass density [g/cm³],
`e` = specific internal energy [erg/g]; `HII_m`/`H2I_m`/`HDI_m` = species MASS
densities [g/cm³] (ρ·x).  Returns the updated `(e, HII_m, H2I_m, HDI_m)`. Pure.
"""
@inline function evolve_cell(rho, e, HII_m, H2I_m, HDI_m, dt, z;
                             hubble = 71.0, Om = 0.27, OL = 0.73,
                             fh = FH_DEFAULT, deuterium::Bool = false,
                             hubble_expansion::Bool = false,
                             adot_over_a = NaN)
    R    = typeof(e)
    mh   = R(MH); tiny = R(_SUB_TINY)
    d    = rho / mh                       # network density (∝ n)
    z0   = R(z)                           # redshift at step BEGIN
    Hz0  = hubble_z_of(z0; hubble = hubble, Om = Om, OL = OL)
    # ȧ/a [1/s] for the ADIABATIC term: analytic by default, OR a caller-supplied value
    # (Enzo's own CosmologyComputeExpansionFactor at the step endpoints, ln(a1/a0)/Δt)
    # so the adiabatic integral matches the host's expansion EXACTLY.  (a1≈a0 on sub-
    # resolution steps → 0, i.e. no expansion: fine.)
    Hz_ad = isnan(adot_over_a) ? Hz0 : R(adot_over_a)
    # When the host supplies its expansion rate (cosmological one-zone use), evolve the
    # redshift ACROSS the macro-step inside the sub-cycle: z(t)=(1+z0)exp(-ȧ/a·t)−1.
    # The CMB Compton target T_cmb(z), the Compton coefficient, and the recombination
    # H(z) then track z continuously instead of being frozen at z0 — essential for the
    # host's large (CIC_MAXEXP) steps, accurate in both the Compton-locked (high-z) and
    # decoupled (low-z) limits.  When no rate is supplied (default), z is held at z0.
    evolve_z = !isnan(adot_over_a)
    yHeI = (one(R) - R(fh)) * d

    # advected species → y-convention number densities; nₑ = n_HII initially
    yHII  = HII_m / mh
    yH2I  = H2I_m / mh                     # = 2·n(H2)
    yHDI  = deuterium ? HDI_m / mh : zero(R)
    yHI   = max((R(fh)*rho - HII_m - H2I_m) / mh, tiny)
    yde   = yHII
    yHM   = tiny; yH2II = tiny
    yDI   = deuterium ? R(DTOH_SEED)*yHI  : zero(R)
    yDII  = deuterium ? R(DTOH_SEED)*yHII : zero(R)

    ttot = zero(R)
    iter = 0
    while ttot < dt && iter < _SUB_ITMAX
        iter += 1
        rem = dt - ttot

        # redshift at the current point in the sub-cycle (frozen at z0 unless the host
        # handed us its expansion rate, in which case z evolves across the macro-step).
        zt   = evolve_z ? (one(R) + z0) * exp(-Hz_ad * ttot) - one(R) : z0
        Tc   = comp2_cmb(zt)              # CMB temperature at zt
        c1   = comp1_cmb(zt)             # Compton coefficient at zt
        Hz   = evolve_z ? hubble_z_of(zt; hubble = hubble, Om = Om, OL = OL) : Hz0

        # temperature from the current state (number densities; nH2=yH2I/2 etc.)
        T = gas_temperature(rho, e, yHI, yHII, yHeI/R(4), tiny, tiny, yde,
                            yHM, yH2I/R(2), yH2II/R(2); gamma = GAMMA_DEFAULT)
        Trad = Tc
        K  = build_rates(T, Trad, yHI, Hz; deuterium = deuterium)

        # cooling rate (volumetric, signed) + temstart shutoff (no cooling at
        # the temperature floor — set to exactly 0, not a spurious-sign tiny).
        nHD  = yHDI / R(3)
        edot = cooling_edot(yHI, yHII, yHeI/R(4), yde, yH2I/R(2), nHD, T, zt)
        if T <= R(1.01)*R(MIN_TEMPERATURE) && edot < zero(R)
            edot = zero(R)
        end
        # adiabatic Hubble cooling: de/dt = -2H·e (γ=5/3); volumetric = ×ρ.
        # Only active when the host hands adiabatic cooling to the kernel.
        if hubble_expansion
            edot -= R(2) * Hz_ad * e * rho
        end

        # chemistry 10% sub-step (no constraint when a rate is ~0)
        dedot, HIdot = _de_hi_dot(yHI, yHII, yde, yH2I, yHM, yH2II,
                                  yHeI, tiny, tiny, K)
        dtit = min(_step10(yde, dedot), _step10(yHI, HIdot), rem, R(0.5)*dt)

        # energy 10% sub-step + CMB-Compton stiffness split
        edot_c    = -c1 * (T - Tc) * yde            # Compton part (volumetric)
        edot_rest = edot - edot_c
        Kc        = c1 * yde * (T / e) / rho        # specific Compton frequency
        stiff     = Kc * rem > one(R)
        de_spec   = (stiff ? edot_rest : edot) / rho
        dtit = min(dtit, _step10(e, de_spec))

        # energy update
        if stiff
            B = (c1*yde*Tc + edot_rest) / rho       # specific source
            e = (e + B*dtit) / (one(R) + Kc*dtit)
        else
            e = e + (edot/rho)*dtit
        end
        e = max(e, tiny)

        # species update (one backward-Euler sweep)
        s = network_step(d, fh, yHI, yHII, yde, yH2I, yHM, yH2II,
                         yDI, yDII, yHDI, K, dtit; deuterium = deuterium)
        yHI=s.yHI; yHII=s.yHII; yde=s.yde; yH2I=s.yH2I; yHM=s.yHM
        yH2II=s.yH2II; yDI=s.yDI; yDII=s.yDII; yHDI=s.yHDI

        ttot += dtit
    end

    return e, yHII*mh, yH2I*mh, (deuterium ? yHDI*mh : HDI_m)
end
