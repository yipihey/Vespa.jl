# emission.jl — the public per-channel / per-line emissivity surface, plus the summed
# cooling rate that ChemistryKernels consumes.
#
# Every radiative cooling process is exposed as its own VOLUMETRIC emissivity
# [erg s⁻¹ cm⁻³] given the local number densities, gas T and redshift z:
#   • discrete LINES where natural (H Lyα; the metal fine-structure lines);
#   • channel-RATES for continuum/lumped processes (bremsstrahlung, recombination,
#     collisional ionisation, H₂/HD bands).
# The same per-channel pieces sum to the network's total cooling (`cooling_rate_total`),
# so cooling and synthetic emission share one source of truth.
#
# Sign: emissivity ≥ 0 = energy radiated away (cooling). `emiss_compton` is signed
# (negative ⇒ CMB heating). Pure, @inline, precision-generic, allocation-free.

export emiss_HI_lyalpha, emiss_H_collion, emiss_H_recomb, emiss_brem,
       emiss_HeI_exc, emiss_HeII_exc, emiss_HeI_collion, emiss_HeII_collion,
       emiss_HeI_collion_S, emiss_HeII_recomb_rad, emiss_HeII_recomb_diel,
       emiss_HeIII_recomb, emiss_H2, emiss_HD, emiss_compton,
       lya_emissivity, metal_line_emissivities, radiative_channels, cooling_rate_total

# ── H / He per-channel volumetric emissivities ────────────────────────────────
# (coefficient × density product, matching the weighting in cooling_rate_total)
@inline emiss_HI_lyalpha(nHI, nde, T) = ceHI(T) * nHI * nde          # H Lyα (1216 Å) coll. exc.
@inline emiss_H_collion(nHI, nde, T)  = ciHI(T) * nHI * nde          # H collisional ionisation
@inline emiss_H_recomb(nHII, nde, T)  = reHII(T) * nHII * nde        # H recombination (Case B cascade)
@inline emiss_brem(nHII, nde, T)      = brem(T) * nHII * nde         # free-free (H⁺ term)

@inline emiss_HeI_exc(nHeI, nde, T)        = ceHeI(T)   * nHeI   * nde
@inline emiss_HeII_exc(nHeII, nde, T)      = ceHeII(T)  * nHeII  * nde
@inline emiss_HeI_collion(nHeI, nde, T)    = ciHeI(T)   * nHeI   * nde
@inline emiss_HeII_collion(nHeII, nde, T)  = ciHeII(T)  * nHeII  * nde
@inline emiss_HeI_collion_S(nHeI, nde, T)  = ciHeIS(T)  * nHeI   * nde
@inline emiss_HeII_recomb_rad(nHeII, nde, T)  = reHeII1(T) * nHeII  * nde
@inline emiss_HeII_recomb_diel(nHeII, nde, T) = reHeII2(T) * nHeII  * nde
@inline emiss_HeIII_recomb(nHeIII, nde, T)    = reHeIII(T) * nHeIII * nde

# ── H₂ / HD band emissivities (Galli-Palla two-level w/ CMB floor; HD CMB-gated) ──
# Identical algebra to the `h2`/`hd` blocks of the legacy cooling assembler.
@inline function emiss_H2(nHI, nHII, nHeI, nde, nH2, T, z; ih2optical::Bool=false, nH=nothing)
    R = typeof(T); one_ = one(R); Tc = comp2_cmb(R(z))
    galdl = GAHI(T)*nHI + GAH2(T)*nH2 + GAHe(T)*nHeI + GAHp(T)*nHII + GAel(T)*nde
    cool_gas = H2LTE(T) / (one_ + H2LTE(T) / galdl)
    galdl_c = GAHI(Tc)*nHI + GAH2(Tc)*nH2 + GAHe(Tc)*nHeI + GAHp(Tc)*nHII + GAel(Tc)*nde
    cool_cmb = H2LTE(Tc) / (one_ + H2LTE(Tc) / galdl_c)
    fudge = one_
    if ih2optical && nH !== nothing
        fudge = min((R(nH) / R(8.0e9))^R(-0.45), one_)
    end
    return fudge * nH2 * (cool_gas - cool_cmb)
end

@inline function emiss_HD(nHI, nHD, T, z)
    R = typeof(T); Tc = comp2_cmb(R(z))
    T > Tc || return zero(R)
    hdlte = HDlte(T)
    return nHD * hdlte / (one(R) + (hdlte / nHI) / max(HDlow(T), R(TINY)))
end

# ── Compton (CMB inverse-Compton scattering; signed: <0 ⇒ heating) ────────────
@inline emiss_compton(nde, T, z) = (R = typeof(T); comp1_cmb(R(z)) * (T - comp2_cmb(R(z))) * nde)

# ── per-line accessors ────────────────────────────────────────────────────────
"H Lyα (1216 Å) volumetric line emissivity [erg s⁻¹ cm⁻³]."
@inline lya_emissivity(nHI, nde, T) = emiss_HI_lyalpha(nHI, nde, T)

"""
    metal_line_emissivities(T, z, nHI, nHII, nde, nH2, nH, ab) -> NamedTuple

All 15 metal fine-structure line emissivities [erg s⁻¹ cm⁻³], weighted by
n_H·a_X·f_stage (live-nₑ ionisation) and the high-T taper — the per-line breakdown
of `metal_cooling_rate`. Keys are (ion, λµm) tagged. Allocation-free (isbits NamedTuple).
"""
@inline function metal_line_emissivities(T, z, nHI, nHII, nde, nH2, nH, ab::MetalAbundances{R}) where {R}
    Trad = comp2_cmb(R(z)); nH2o = R(0.75)*nH2; nH2p = R(0.25)*nH2
    taper = T >= R(2.0e4) ? zero(R) : (T > R(1.0e4) ? _hot_taper(T) : one(R))
    fC = _fion_C(T,nde,nHI,nHII); fSi = _fion_Si(T,nde,nHI,nHII); fFe = _fion_Fe(T,nde,nHI,nHII)
    wCI  = nH*ab.C*(one(R)-fC)*taper;  wCII = nH*ab.C*fC*taper
    wOI  = nH*ab.O*taper
    wSiI = nH*ab.Si*(one(R)-fSi)*taper; wSiII= nH*ab.Si*fSi*taper
    wFe  = nH*ab.Fe*fFe*taper
    cI  = _cool_CI_lines(T,Trad,nHI,nHII,nH2o,nH2p,nde)
    oI  = _cool_OI_lines(T,Trad,nHI,nHII,nH2o,nH2p,nde)
    siI = _cool_SiI_lines(T,Trad,nHI,nHII,nH2o,nH2p,nde)
    fe  = _cool_FeII_lines(T,Trad,nHI,nde)
    return (CI_609=wCI*cI[1], CI_230=wCI*cI[2], CI_369=wCI*cI[3],
            CII_158=wCII*_cool_CII(T,Trad,nHI,nH2o,nH2p,nde),
            OI_63=wOI*oI[1], OI_44=wOI*oI[2], OI_146=wOI*oI[3],
            SiI_130=wSiI*siI[1], SiI_45=wSiI*siI[2], SiI_68=wSiI*siI[3],
            SiII_35=wSiII*_cool_SiII(T,Trad,nHI,nde),
            FeII_26=wFe*fe[1], FeII_35=wFe*fe[2], FeII_51=wFe*fe[3], FeII_87=wFe*fe[4])
end

# ── diagnostics: per-channel breakdown (NamedTuple; inspection, not the bit-exact path) ──
"""
    radiative_channels(nHI,nHII,nHeI,nde,nH2,nHD,T,z; nH, metals) -> NamedTuple

Per-channel volumetric cooling [erg s⁻¹ cm⁻³] (HI exc/ci/rec, brem, H₂, HD, Compton,
metals, total) for inspection. `total` matches `cooling_rate_total` to summation order.
"""
@inline function radiative_channels(nHI, nHII, nHeI, nde, nH2, nHD, T, z;
                                    ih2optical::Bool=false, nH=nothing, metals=nothing)
    R = typeof(T)
    hi_exc = emiss_HI_lyalpha(nHI, nde, T); hi_ci = emiss_H_collion(nHI, nde, T)
    hi_rec = emiss_H_recomb(nHII, nde, T);  bre   = emiss_brem(nHII, nde, T)
    h2 = emiss_H2(nHI, nHII, nHeI, nde, nH2, T, z; ih2optical=ih2optical, nH=nH)
    hd = emiss_HD(nHI, nHD, T, z); comp = emiss_compton(nde, T, z)
    met = metals === nothing ? zero(R) :
          metal_cooling_rate(T, R(z), nHI, nHII, nde, nH2, R(nH), metals)
    tot = (hi_exc + hi_ci) + hi_rec + bre + h2 + hd + comp + met
    return (atomic_HI_exc=hi_exc, atomic_HI_ci=hi_ci, atomic_HI_rec=hi_rec, brem=bre,
            h2=h2, hd=hd, compton=comp, metals=met, total=tot)
end

# ── the summed cooling rate ChemistryKernels consumes (bit-identical to legacy edot) ──
"""
    cooling_rate_total(nHI,nHII,nHeI,nde,nH2,nHD,T,z; ih2optical, nH, metals) -> Λ

Total radiative cooling [erg s⁻¹ cm⁻³] (≥0 ⇒ net cooling), reproducing the legacy
`-(cooling_edot)`: atomic(HI exc+ci, HII rec, brem) + H₂ + HD + Compton + metals, in the
EXACT original term order (He cooling omitted, as in the reduced network). ChemistryKernels'
`cooling_edot = -cooling_rate_total`.
"""
@inline function cooling_rate_total(nHI, nHII, nHeI, nde, nH2, nHD, T, z;
                                    ih2optical::Bool=false, nH=nothing, metals=nothing)
    R    = typeof(T)
    one_ = one(R)
    Tc   = comp2_cmb(R(z))

    atomic = (ceHI(T) + ciHI(T)) * nHI * nde +
             reHII(T) * nHII * nde +
             brem(T)  * nHII * nde

    galdl = GAHI(T) * nHI + GAH2(T) * nH2 + GAHe(T) * nHeI +
            GAHp(T) * nHII + GAel(T) * nde
    h2lte = H2LTE(T)
    cool_gas = h2lte / (one_ + h2lte / galdl)
    galdl_c = GAHI(Tc) * nHI + GAH2(Tc) * nH2 + GAHe(Tc) * nHeI +
              GAHp(Tc) * nHII + GAel(Tc) * nde
    h2lte_c  = H2LTE(Tc)
    cool_cmb = h2lte_c / (one_ + h2lte_c / galdl_c)
    fudge = one_
    if ih2optical && nH !== nothing
        fudge = min((R(nH) / R(8.0e9))^R(-0.45), one_)
    end
    h2 = fudge * nH2 * (cool_gas - cool_cmb)

    hd = zero(R)
    if T > Tc
        hdlte  = HDlte(T)
        hdlte1 = hdlte / nHI
        hdlow1 = max(HDlow(T), R(TINY))
        hd = nHD * hdlte / (one_ + hdlte1 / hdlow1)
    end

    compton = comp1_cmb(R(z)) * (T - Tc) * nde

    metal = metals === nothing ? zero(R) :
            metal_cooling_rate(T, R(z), nHI, nHII, nde, nH2, R(nH), metals)

    return atomic + h2 + hd + compton + metal
end
