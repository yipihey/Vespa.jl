# recombination.jl — the v2026 RECFAST/Peebles H recombination override.
#
# When cmb_recombination is on (always, in the reduced model), grackle replaces
# the case-B rate k2 by a per-cell value suppressed by the Peebles C-factor — the
# probability that a fresh recombination to an excited state cascades to ground
# rather than being re-ionized by a redshifting Lyα photon.  This is what makes
# the network reproduce the recfast x_e(z) freeze-out for z ≲ 1000.
#
# Direct transcription of solve_rate_cool_g.F:1385-1409 (the inline block) in
# physical CGS — k2 [cm³/s] = α_B·10⁶·C, dropping grackle's `·dom·tbase1` code-
# unit conversion.  The C-factor needs H(z) (Sobolev escape K = λ³/(8πH)),
# computed exactly as solve_chemistry.c:170-180.  Pure & allocation-free.

export hubble_z_of, peebles_k2

# RECFAST constants (solve_rate_cool_g.F:1387-1391); rec_fu = 1 (pure Peebles).
const _REC_CR  = 1.799920e14
const _REC_CDB = 3.945150e4
const _REC_LAM = 1.215668e-7        # Lyα wavelength [m] (formula is in SI)
const _REC_A8  = 8.2245809          # 2γ decay rate of H 2s [s⁻¹]
# Hubble unit conversion (solve_chemistry.c:177): km/s/Mpc → s⁻¹.
const _MPC_CM  = 3.0856775807e24
const _OR_FAC  = 4.15e-5            # Ω_r·h² (CMB photons + 3 ν, T_cmb=2.725 K)

"""
    hubble_z_of(z; hubble, Om, OL)

Hubble rate H(z) [s⁻¹] from the cosmology, exactly as solve_chemistry.c:177-180
(radiation Ω_r = 4.15e-5/h² from T_cmb=2.725 K photons + 3 ν; curvature closes
the budget). `hubble` = H₀ [km/s/Mpc]. Pure.
"""
@inline function hubble_z_of(z; hubble = 71.0, Om = 0.27, OL = 0.73)
    R   = typeof(z)
    zp1 = one(R) + z
    H0  = R(hubble) * R(1.0e5) / R(_MPC_CM)
    hh  = R(hubble) / R(100.0)
    Or  = R(_OR_FAC) / (hh * hh)
    Ok  = one(R) - R(Om) - R(OL) - Or
    return H0 * sqrt(Or*zp1^4 + R(Om)*zp1^3 + Ok*zp1^2 + R(OL))
end

"""
    peebles_k2(T, nHI, Hz)

CaseB H recombination rate k2 [cm³/s] with the Peebles C-factor suppression, at
temperature `T` [K], neutral-H number density `nHI` [cm⁻³], and Hubble rate `Hz`
[s⁻¹]. (solve_rate_cool_g.F:1393-1407.) Pure.
"""
@inline function peebles_k2(T, nHI, Hz)
    R   = typeof(T)
    tt  = T / R(1.0e4)
    aB  = R(1.0e-19) * R(4.309) * tt^R(-0.6166) /
          (one(R) + R(0.6703) * tt^R(0.5300))            # α_B [m³/s]
    n1s = nHI * R(1.0e6)                                  # cm⁻³ → m⁻³
    bet = aB * (R(_REC_CR) * T)^R(1.5) * exp(-R(_REC_CDB) / T)
    K   = R(_REC_LAM)^3 / (R(8.0) * R(π) * Hz)
    KL  = K * R(_REC_A8) * n1s
    KB  = K * bet * n1s
    C   = (one(R) + KL) / (one(R) + KL + KB)             # rec_fu = 1
    return aB * R(1.0e6) * C                              # m³/s → cm³/s
end
