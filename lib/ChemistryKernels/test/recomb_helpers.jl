# recomb_helpers.jl — shared one-zone / cosmology helpers for the recombination test
# files (test_recombination_mixing.jl and test_recombination_field.jl).  Included
# AFTER `using ChemistryKernels` so the module exports (hubble_z_of, MH, KBOLTZ,
# FAlphaTable, FA_ZERO, solve_chem_mixing!, temperature_from_reduced) are in scope.

_mean(v) = sum(v) / length(v)

# H(z) [s⁻¹] for standard Planck18-ish cosmology.
_Hz(z; h=71.0, Om=0.27, OL=0.73) =
    hubble_z_of(Float64(z); hubble=h, Om=Om, OL=OL)

# Saha ionisation fraction for hydrogen at gas temperature T [K] and n_H [cm⁻³].
# S(T) = 2 × (2πm_e kB/h²)^{3/2} T^{3/2} exp(−χ_H/T)  [cm⁻³]
# Prefactor computed from CGS constants; good to <0.1% in the relevant range.
function saha_xe(T::Float64, n_H::Float64)
    chi_K = 157807.0          # H ionisation energy / kB [K]  (13.6 eV)
    S     = 4.824e15 * T^1.5 * exp(-chi_K / T)   # [cm⁻³]
    r     = S / n_H           # dimensionless
    return 0.5 * (-r + sqrt(r*r + 4.0*r))        # positive root
end

# Mean baryon number density of H at redshift z (flat Planck18-ish cosmology).
function n_H_at_z(z::Float64; fh=0.76)
    rho_b0 = 0.044 * 9.47e-30          # Ω_b × ρ_crit,0 [g/cm³]
    return fh * rho_b0 * (1.0 + z)^3 / ChemistryKernels.MH   # [cm⁻³]
end

# Specific internal energy [erg/g] for baryon temperature T, ionisation fraction x_e,
# mass density rho [g/cm³], hydrogen mass fraction fh, adiabatic index gamma.
function e_from_T(T::Float64, x_e::Float64, rho::Float64; fh=0.76, gamma=5/3)
    # total particle number density: n = n_HI + n_HII + n_e + n_He (fully neutral He)
    n_tot = rho / ChemistryKernels.MH * (fh*(1.0 + x_e) + (1.0 - fh)/4.0)
    return n_tot * ChemistryKernels.KBOLTZ * T / (rho * (gamma - 1.0))
end

# Simple one-zone cosmological integrator in physical CGS (code units = CGS, so
# density_units = length_units = time_units = 1).
# Returns (z_vals, xe_vals, Tb_vals) arrays; xe = xe_H = n_HII/n_H.
# Hubble expansion cooling is handled INSIDE the chemistry subcycler (hubble_expansion=true)
# rather than by an external e_scale pre-step; Compton coupling drives T_m → T_CMB.
function integrate_onezone(; z_start=5000.0, z_end=200.0,
                            n_steps=500, hubble=71.0, Om=0.27, OL=0.73,
                            fh=0.76, n_H_override=nothing,
                            x_e_init=nothing, T_init=nothing,
                            fa_table::FAlphaTable = FA_ZERO,
                            smoothed_density_fn = nothing,
                            recfast_fudge::Float64 = 1.0,
                            recfast_hswitch::Bool  = false)
    logzp = range(log(1+z_start), log(1+z_end); length=n_steps+1)

    z0   = z_start
    nH0  = isnothing(n_H_override) ? n_H_at_z(z0; fh=fh) : n_H_override
    xe0  = isnothing(x_e_init) ? min(saha_xe(2.725*(1+z0), nH0), 1.0 - 1e-6) : Float64(x_e_init)
    T0   = isnothing(T_init)   ? 2.725*(1+z0) : Float64(T_init)
    rho0 = nH0 * ChemistryKernels.MH / fh

    rho_v = [rho0]
    e_v   = [e_from_T(T0, xe0, rho0; fh=fh)]
    HII_v = [xe0 * nH0 * ChemistryKernels.MH]
    H2I_v = [1.0e-40]
    nsm_v = [rho0]

    z_out  = Float64[z0]
    xe_out = Float64[xe0]
    Tb_out = Float64[T0]

    for k in 1:n_steps
        z_hi  = exp(logzp[k])   - 1.0
        z_lo  = exp(logzp[k+1]) - 1.0
        z_mid = 0.5*(z_hi + z_lo)
        Hz    = _Hz(z_mid; h=hubble, Om=Om, OL=OL)
        dz    = z_hi - z_lo                            # > 0
        dt    = dz / ((1.0 + z_mid) * Hz)             # [s]

        nH    = isnothing(n_H_override) ? n_H_at_z(z_lo; fh=fh) : n_H_override
        rho_c = nH * ChemistryKernels.MH / fh

        rho_scale = rho_c / rho_v[1]                  # < 1 (expansion)
        rho_v[1]  = rho_c
        HII_v[1] *= rho_scale   # preserves x_e = HII/(ρ·fh/mH)
        H2I_v[1] *= rho_scale

        nsm_v[1] = isnothing(smoothed_density_fn) ? rho_c : smoothed_density_fn(z_lo)

        Xe_m = HII_v[1] / (rho_c * fh)   # current xe_H for n1s_smoothed approximation

        solve_chem_mixing!(rho_v, e_v, HII_v, H2I_v, nsm_v;
                           a_value    = 1.0/(1.0+z_lo),
                           dt         = dt,
                           density_units = 1.0, length_units = 1.0, time_units = 1.0,
                           fa_table   = fa_table,
                           Xe_mean    = clamp(Xe_m, 0.0, 1.0),
                           recfast_fudge  = recfast_fudge,
                           recfast_hswitch = recfast_hswitch,
                           hubble_expansion = true,
                           hubble=hubble, Om=Om, OL=OL, fh=fh)

        xe_c = HII_v[1] / (rho_c * fh)
        Tb_c = temperature_from_reduced(rho_c, e_v[1], HII_v[1], H2I_v[1]; fh=fh)

        push!(z_out,  z_lo)
        push!(xe_out, xe_c)
        push!(Tb_out, Tb_c)
    end

    return z_out, xe_out, Tb_out
end
