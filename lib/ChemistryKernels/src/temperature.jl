# temperature.jl — gas temperature with the H2 variable-γ correction.
#
# Implements the temperature path of the Abel/Anninos et al. 1997 network:
#   pressure      : P = (Γ-1)·ρ·e, then corrected for the H2 vibrational
#                   degrees of freedom (Γ → Γ1) when n_H2/n_other > 1e-3.
#   temperature   : T = P_corrected · utem / n_total, floored at 1 K.
#
# Written in PHYSICAL CGS (ρ [g/cm³], e [erg/g], number densities [cm⁻³]); the
# code-unit↔CGS boundary (and the comoving a³ factor) live in boundary.jl. T = P /
# (kB·n_total) is the CGS-explicit form of the code-unit expression utem·P_code/nd_code
# (see the unit reduction: utem = mh·v²/kB, nd_code = n·mh/ρu, e_code = e/v², ρ_code = ρ/ρu
# ⇒ all units cancel to (Γ-1)·ρ·e/(kB·n)).  Pure & allocation-free (AD-friendly).

export gas_temperature, temperature_from_reduced, temperature_grid

# Default Gamma (5/3) and minimum returned temperature (1 K).
const GAMMA_DEFAULT = 5.0 / 3.0          # adiabatic index Gamma
const MIN_TEMPERATURE = 1.0              # temperature floor [K]

"""
    gas_temperature(rho, eint, nHI, nHII, nHeI, nHeII, nHeIII, nde, nHM, nH2, nH2II; gamma)

Gas temperature [K] from physical CGS mass density `rho` [g/cm³], specific internal
energy `eint` [erg/g], and the per-species PHYSICAL number densities [cm⁻³].
`nH2`/`nH2II` are H2 and H2⁺ *molecule* number densities. Applies the pressure
(H2 γ-correction) and temperature relations of the Abel/Anninos et al. 1997
network. Pure.
"""
@inline function gas_temperature(rho, eint, nHI, nHII, nHeI, nHeII, nHeIII,
                                 nde, nHM, nH2, nH2II; gamma = GAMMA_DEFAULT)
    R   = typeof(eint)
    g   = R(gamma)
    gm1 = g - one(R)
    gammaInv = one(R) / gm1                       # GammaInverse = 1/(Γ-1)
    kB  = R(KBOLTZ)

    # number_density (no H2) and nH2: the network's species split.
    n_no_h2 = nHeI + nHeII + nHeIII + nHI + nHII + nHM + nde
    nH2tot  = nH2 + nH2II
    n_tot   = n_no_h2 + nH2tot

    P    = gm1 * rho * eint                        # raw pressure [erg/cm³]
    Traw = P / (kB * n_tot)                        # utem·P_code/nd_code, CGS form
    temp = max(Traw, one(R))                       # max(...,1) temperature estimate

    # GammaH2Inverse: 0.5*5 unless there's a reasonable amount of H2.
    GammaH2Inv = R(0.5) * R(5)
    if nH2tot / n_no_h2 > R(1.0e-3)
        x = R(6100.0) / temp
        if x < R(10.0)
            ex = exp(x)
            GammaH2Inv = R(0.5) * (R(5) + R(2) * x * x * ex / (ex - one(R))^2)
        end
    end
    Gamma1 = one(R) + (nH2tot + n_no_h2) /
                      (nH2tot * GammaH2Inv + n_no_h2 * gammaInv)

    T = Traw * (Gamma1 - one(R)) / gm1             # P *= (Γ1-1)/(Γ-1), then /n_tot
    return max(T, R(MIN_TEMPERATURE))
end

"""
    temperature_from_reduced(rho, eint, HIImass, H2Imass; fh, gamma)

Temperature for the v2026 reduced network straight from the advected fields
(physical CGS): reconstruct HI = fh·ρ − HII − H2I, helium all neutral, nₑ = n_HII,
H⁻/H2⁺ = the floor value — the reduced-network reconstruction — then
`gas_temperature`. Deuterium is ignored (the temperature relation ignores D). Pure.
"""
@inline function temperature_from_reduced(rho, eint, HIImass, H2Imass;
                                          fh = FH_DEFAULT, gamma = GAMMA_DEFAULT)
    R    = typeof(eint)
    mh   = R(MH)
    tiny = R(TINY)
    nHII = HIImass / mh
    nH2  = H2Imass / (R(2) * mh)                   # H2I is the H2 MASS density
    nHI  = max((R(fh) * rho - HIImass - H2Imass) / mh, tiny)
    nHeI = (one(R) - R(fh)) * rho / (R(4) * mh)
    return gas_temperature(rho, eint, nHI, nHII, nHeI, tiny, tiny, nHII,
                           tiny, nH2, tiny; gamma = gamma)
end

# ── device launcher (mirrors @scalarkernel; reconstructs + computes per cell) ──
@kernel function _temperature_k!(Tout, @Const(rho), @Const(eint),
                                 @Const(HII), @Const(H2I), fh, gamma)
    i = @index(Global)
    @inbounds Tout[i] = temperature_from_reduced(rho[i], eint[i], HII[i], H2I[i];
                                                 fh = fh, gamma = gamma)
end

"""
    temperature_grid(name, ::Type{T}, rho, eint, HII, H2I; fh, gamma) -> Vector{T}

Run `temperature_from_reduced` over arrays of the reduced-network fields (all in
PHYSICAL CGS) on backend `name` at precision `T`. With density_units =
length_units = time_units = 1, code units ≡ CGS.
"""
function temperature_grid(name::Symbol, ::Type{Tprec}, rho, eint, HII, H2I;
                          fh = FH_DEFAULT, gamma = GAMMA_DEFAULT) where {Tprec}
    be = backend(name)
    n  = length(rho)
    dr = to_device(be, collect(rho),  Tprec)
    de = to_device(be, collect(eint), Tprec)
    dh = to_device(be, collect(HII),  Tprec)
    d2 = to_device(be, collect(H2I),  Tprec)
    o  = device_zeros(be, Tprec, (n,))
    _temperature_k!(be)(o, dr, de, dh, d2, Tprec(fh), Tprec(gamma); ndrange = n)
    return to_host(o)
end
