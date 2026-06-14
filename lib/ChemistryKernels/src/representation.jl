# ── internal representation (the f32-safe, host-unit-free convention) ─────────
#
# State is carried in PHYSICAL CGS number densities (cm^-3) per species, with the
# DYNAMICS expressed through dimensionless groups built from abundances relative
# to the hydrogen nucleus density n_H:
#
#     y_i  = n_i / n_H            (dimensionless, O(1) for the dominant species)
#     ν_2  = k · n_H              (2-body reaction frequency, s^-1)
#     ν_3  = k · n_H^2            (3-body reaction frequency, s^-1)
#
# so the backward-Euler groups (ν·dt) and the abundances both stay in f32 range
# across z=1000→20 and n_H ∈ [1e-3, 1e18] cm^-3 — without coupling to any host
# code-unit normalization (unlike grackle's kUnit-scaled tables).  Rates and
# cooling are evaluated from the analytic fits in CGS (cm^3/s, erg cm^3/s).
#
# The host boundary (`boundary.jl`, Wave 4) converts the host's code-unit mass
# densities to these physical number densities and back; this file provides the
# unit-conversion primitives and the species reconstruction shared by both.

"""
    cgs_number_densities(rho, HII, H2I, HDI, density_units, a_value; fh=FH_DEFAULT, deuterium)

Convert host code-unit mass densities to PHYSICAL CGS number densities (cm^-3) of
the reduced-network species, reconstructing the non-advected ones exactly as
`grackle_reduced.c` does:
  nₑ = n_HII ;  n_HI = fh·ρ − n_HII − n_H2(mass) ;  n_HeI = (1−fh)·ρ (mass→/4) ;
  H⁻, H₂⁺ (and D⁺) start at `tiny` (equilibrium fills them).
Density is taken to PHYSICAL via density_units/a³ (the reduced wrapper passes a
comoving ρ with comoving_coordinates=0, so grackle applies /a³; we do it here).
Returns a NamedTuple of number densities + n_H.
"""
function cgs_number_densities(rho, HII, H2I, HDI, density_units, a_value;
                              fh = FH_DEFAULT, deuterium::Bool = false)
    a3 = a_value^3
    du = density_units / a3                       # comoving code → physical cgs
    rho_cgs = rho * du
    nH   = fh * rho_cgs / MH                       # H nuclei per cm^3
    nHII = HII * du / MH
    nH2  = H2I * du / (2 * MH)                     # H2I is the H2 MASS density (2·n(H2))
    nHI  = max(nH - nHII - 2 * nH2, TINY)          # H conservation (2 H per H2)
    nHeI = (1 - fh) * rho_cgs / (4 * MH)
    nHDI = deuterium ? HDI * du / (3 * MH) : 0.0   # HD mass = 3 amu
    return (; nH, nHI, nHII, ne = nHII, nHeI,
            nHM = TINY, nH2 = nH2, nH2II = TINY,
            nDI = deuterium ? DTOH_SEED * nHI : 0.0,
            nDII = deuterium ? DTOH_SEED * nHII : 0.0,
            nHDI, du)
end

"""
    temperature_units(length_units, time_units)

The energy→temperature factor `T = (γ-1)·μ·mh·v_units²/kB · e_int` uses
`v_units = length_units/time_units`; returns `mh·v_units²/kB` (grackle's
`get_temperature_units`, comoving_coordinates=0 → no a-factor on velocity).
"""
temperature_units(length_units, time_units) =
    MH * (length_units / time_units)^2 / KBOLTZ
