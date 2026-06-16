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

export hubble_z_of, recfast_alpha, peebles_k2, beta1s_freq, helium_saha_pair
export total_electron_fraction, helium_HeI_rate_AB

# RECFAST constants (solve_rate_cool_g.F:1387-1391); rec_fu = 1 (pure Peebles).
const _REC_CR  = 1.799920e14
const _REC_CDB = 3.945150e4
const _REC_LAM = 1.215668e-7        # Lyα wavelength [m] (formula is in SI)
const _REC_A8  = 8.2245809          # 2γ decay rate of H 2s [s⁻¹]
const _CHI_H_K = 157807.0           # H(1s) ionisation energy / k_B [K] (13.6 eV)
# Helium ionisation energies / k_B [K] (CODATA: He I 24.5874 eV, He II 54.4178 eV)
const _CHI_HEI  = 285335.0          # He I  → He II   (24.5874 eV)
const _CHI_HEII = 631515.0          # He II → He III  (54.4178 eV)
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
    recfast_alpha(T) -> α_B [m³/s]

Case-B H recombination coefficient from the RecFast fit (Hui & Gnedin 1997).
Identical to the `aB` formula inside `peebles_k2`; factored out so the HyRec
table backend can replace it without touching the kernel (swap one call site).
Returns m³/s — same unit convention as `peebles_k2`.
"""
@inline function recfast_alpha(T::Real)
    R  = typeof(T)
    tt = T / R(1.0e4)
    return R(1.0e-19) * R(4.309) * tt^R(-0.6166) /
           (one(R) + R(0.6703) * tt^R(0.5300))
end

"""
    peebles_k2(T, nHI, Hz)

CaseB H recombination rate k2 [cm³/s] with the Peebles C-factor suppression, at
temperature `T` [K], neutral-H number density `nHI` [cm⁻³], and Hubble rate `Hz`
[s⁻¹]. (solve_rate_cool_g.F:1393-1407.) Pure.
"""
@inline function peebles_k2(T, nHI, Hz)
    R   = typeof(T)
    aB  = recfast_alpha(T)                               # α_B [m³/s]
    n1s = nHI * R(1.0e6)                                  # cm⁻³ → m⁻³
    bet = aB * (R(_REC_CR) * T)^R(1.5) * exp(-R(_REC_CDB) / T)
    K   = R(_REC_LAM)^3 / (R(8.0) * R(π) * Hz)
    KL  = K * R(_REC_A8) * n1s
    KB  = K * bet * n1s
    C   = (one(R) + KL) / (one(R) + KL + KB)             # rec_fu = 1
    return aB * R(1.0e6) * C                              # m³/s → cm³/s
end

"""
    beta1s_freq(T) -> β₁s [s⁻¹]

CMB photoionisation rate of H(1s) per neutral H atom. At z < 1200 (T < 3272 K)
this is < 10⁻¹⁶ s⁻¹ and entirely negligible; it becomes the dominant process
maintaining Saha equilibrium at z > 2000. Pure; allocation-free.

  β₁s = β₂p × exp(−(χ₁s − χ₂p) / T)  where β₂p = α_B × (C_R·T)^{3/2} × exp(−χ₂p/T)
"""
@inline function beta1s_freq(T::Real)
    R   = typeof(T)
    tt  = T / R(1.0e4)
    aB  = R(1.0e-19) * R(4.309) * tt^R(-0.6166) /
          (one(R) + R(0.6703) * tt^R(0.5300))             # α_B [m³/s]
    bet = aB * (R(_REC_CR) * T)^R(1.5) * exp(-R(_REC_CDB) / T)   # β₂p
    return bet * exp(-R(_CHI_H_K - _REC_CDB) / T)                 # β₁s [s⁻¹]
end

"""
    helium_saha_pair(T) -> (S1, S2)   [both in cm⁻³]

Saha ionisation factors for helium at temperature `T` [K]:

  S1 = n_HeII·n_e / n_HeI   = 4·n_Q·exp(−χ_HeI /T)     (He I  ⇌ He II)
  S2 = n_HeIII·n_e / n_HeII = 1·n_Q·exp(−χ_HeII/T)     (He II ⇌ He III)

with the quantum concentration n_Q = (2π m_e k_B T/h²)^{3/2} (here `(_REC_CR·T)^1.5`,
converted m⁻³ → cm⁻³) and statistical-weight ratios 4 (=2·2/1) and 1 (=1·2/2).
`T` should be the CMB radiation temperature: cosmological He ionisation is
photoionisation equilibrium with the CMB, and using T_rad makes He fully neutral
automatically at low z (cold radiation), so it adds nothing to late-time gas.
Pure; allocation-free.
"""
@inline function helium_saha_pair(T::Real)
    R = typeof(T)
    nQ = (R(_REC_CR) * T)^R(1.5) * R(1.0e-6)                 # n_Q [cm⁻³]
    s1 = R(4.0) * nQ * exp(-R(_CHI_HEI)  / T)                # [cm⁻³]
    s2 =          nQ * exp(-R(_CHI_HEII) / T)                # [cm⁻³]
    return s1, s2
end

"""
    total_electron_fraction(xHII, nH, Trad; fh=FH_DEFAULT, niter=8) -> x_e

Total free-electron fraction x_e = n_e/n_H, given the hydrogen ionisation fraction
`xHII = n_HII/n_H`, the hydrogen number density `nH` [cm⁻³], and the CMB radiation
temperature `Trad` [K].  Adds the helium contribution n_e(He)/n_H = f_He·(x_HeII +
2·x_HeIII), where the He ionisation is the Saha equilibrium (`helium_saha_pair`)
solved self-consistently with n_e (a few fixed-point iterations; He is a small
perturbation so convergence is fast).  f_He = (1−f_h)/(4 f_h).

This reconstructs the SAME total electron density the network carries internally
(`network_step`'s charge-conservation n_e), but converged — use it to report x_e
when only the advected n_HII is available.  At low z (cold CMB) He is neutral and
x_e = xHII exactly.  Pure.
"""
@inline function total_electron_fraction(xHII::Real, nH::Real, Trad::Real;
                                         fh::Real = FH_DEFAULT, niter::Int = 8)
    R   = typeof(xHII)
    fHe = (one(R) - R(fh)) / (R(4) * R(fh))         # n_He/n_H
    s1, s2 = helium_saha_pair(R(Trad))
    xe  = xHII                                       # initial guess (He neutral)
    for _ in 1:niter
        ne  = max(xe * R(nH), R(1.0e-30))
        r1  = s1 / ne                                # n_HeII /n_HeI
        r2  = s2 / ne                                # n_HeIII/n_HeII
        den = one(R) + r1 + r1 * r2
        xe  = xHII + fHe * (r1 + R(2) * r1 * r2) / den
    end
    return xe
end

"""
    total_electron_fraction(xHII, xHeII, nH, Trad; fh=FH_DEFAULT) -> x_e

Total free-electron fraction when He⁺ is carried explicitly (evolved/advected):
`xHeII = n_HeII/n_H` is taken as given (so the He I freeze-out it encodes is
preserved), and only He⁺⁺ is added from Saha (`helium_saha_pair`, exact since
He²⁺⇌He⁺ is always fast):

    x_e = xHII + xHeII + 2·xHeIII,   xHeIII = xHeII · S2 / n_e   (self-consistent n_e).

Use this — NOT the 3-argument Saha form — to report x_e from a run that advects
He⁺, otherwise the freeze-out (z≈2000-2500) is lost to Saha. Pure.
"""
@inline function total_electron_fraction(xHII::Real, xHeII::Real, nH::Real, Trad::Real;
                                         fh::Real = FH_DEFAULT, niter::Int = 6)
    R = typeof(xHII)
    _, s2 = helium_saha_pair(R(Trad))
    xe = R(xHII) + R(xHeII)                          # initial guess (He⁺⁺ ≈ 0)
    for _ in 1:niter
        ne     = max(xe * R(nH), R(1.0e-30))
        xHeIII = R(xHeII) * s2 / ne                  # n_HeIII/n_H = xHeII·S2/n_e
        xe     = R(xHII) + R(xHeII) + R(2) * xHeIII
    end
    return xe
end

"""
    helium_HeI_rate_AB(Trad, nH, Hz, xH1, xHeII, fHe) -> (A, B)   [s⁻¹, s⁻¹]

He I (He⁺ ⇌ He⁰) recombination as a linear rate dxHeII/dt = A − B·xHeII, where
`xHeII = n_HeII/n_H` and `xH1 = n_HI/n_H` (neutral H). Direct transcription of
HyRec-2 `helium.c::rec_helium_dxHeIIdlna` (fsR = meR = 1): a Saha-anchored net
recombination with the full He I radiative-transfer escape probability — the
2¹S→1¹S two-photon channel (Λ=50.94 s⁻¹), the 2¹P→1¹S resonance with Sobolev
escape, the incoherent 2¹P width, the H-continuum-opacity enhancement (∝1/xH1,
which lets He I Lyα photons escape as neutral H grows and so COMPLETES He
recombination by z≈1700), and the 2³P intercombination line.  `Trad` is the CMB
temperature [K]; `nH` total H [cm⁻³]; `Hz` [s⁻¹]; `fHe = n_He/n_H`.

Because A − B·xHeII = ydown·(xHeI·s − xHeII·xe) and equilibrium gives the exact He
Saha (`helium_saha_pair`), this reduces to Saha when recombination is fast (high z,
matching it to <0.1%) and captures the z≈2000-2500 freeze-out when it is slow.
Use with a backward-Euler step: xHeII⁺ = (xHeII + A·dt)/(1 + B·dt). Pure.
"""
@inline function helium_HeI_rate_AB(Trad::Real, nH::Real, Hz::Real,
                                    xH1::Real, xHeII::Real, fHe::Real)
    R    = typeof(Trad)
    Tr   = R(Trad)
    xe   = R(xHeII) + (one(R) - R(xH1))          # total electron fraction
    xHeI = max(R(fHe) - R(xHeII), R(1.0e-30))
    s0   = R(2.414194e15) * Tr * sqrt(Tr) / R(nH) * R(4.0)
    s    = s0 * exp(-R(285325.0) / Tr)            # He I Saha factor (× n_H)
    y2s  = exp(R(46090.0) / Tr) / s0
    y2p  = exp(R(39101.0) / Tr) / s0 * R(3.0)
    etacinv = R(9.15776e22) * R(Hz) / R(nH) / max(R(xH1), R(1.0e-30))
    g2pinc = R(1.976e6)/(one(R)-exp(-R(6989.0)/Tr)) + R(6.03e6)/(exp(R(19754.0)/Tr)-one(R)) +
             R(1.06e8)/(exp(R(21539.0)/Tr)-one(R)) + R(2.18e6)/(exp(R(28496.0)/Tr)-one(R)) +
             R(3.37e7)/(exp(R(29224.0)/Tr)-one(R)) + R(1.04e6)/(exp(R(32414.0)/Tr)-one(R)) +
             R(1.51e7)/(exp(R(32781.0)/Tr)-one(R))
    tau2p   = R(4.277e-8) * R(nH) / R(Hz) * xHeI
    dnuline = g2pinc * tau2p / (R(4.0) * R(π) * R(π))
    tauc    = dnuline / etacinv
    enh     = sqrt(one(R) + R(π)*R(π)*tauc) + R(7.74)*tauc/(one(R)+R(70.0)*tauc)
    pesc    = enh/tau2p +
              (one(R)-exp(-R(1.023e-7)*tau2p)) *
              (R(0.964525)*exp(R(2947.0)/Tr) - enh*exp(-R(6.14e13)/etacinv)) / tau2p
    ydown   = R(50.94)*y2s + R(1.7989e9)*y2p*pesc
    # dxHeII/dt = ydown·(xHeI·s − xHeII·xe) = ydown·fHe·s − ydown·(s+xe)·xHeII
    return ydown * R(fHe) * s, ydown * (s + xe)
end
