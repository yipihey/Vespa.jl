# cooling_compton.jl — CMB Compton cooling coefficients as pure functions of z.
#
# The Compton cooling of the Abel/Anninos et al. 1997 network builds two scalars:
#
#   comp1 = compa * (1+z)^4        [code units; cooling coefficient]
#   comp2 = 2.73 * (1+z)           [K; CMB temperature]
#
# where `compa` is the Compton coupling constant.  The raw CGS value is
# 5.65e-36 [erg cm³ K^-4 s^-1] (Peebles 1971).
#
# The two functions below return CGS values (no time_units factor).  The host
# code applies unit conversion at the boundary, matching the same convention as
# rates_cmb.jl (k27/k28 also returned without time_units).

# compa: Compton cooling constant [erg cm³ K^-4 s^-1]  (Peebles 1971)
const COMPA = 5.65e-36

# comp2: CMB radiation temperature [K] as a function of redshift z.
# The Abel/Anninos et al. 1997 network uses the rounded value 2.73; we use the
# physically correct T_CMB,0 = 2.725 K (Fixsen 2009, consistent with the
# radiation density _OR_FAC in recombination.jl and the CAMB reference fixture).
@inline comp2_cmb(z::Real) = (R = typeof(z); R(2.725) * (R(1.0) + z))

# comp1: Compton coupling coefficient (CGS) as a function of redshift z.
#   comp1 = compa * (1+z)^4
# The (1+z)^4 scaling reflects the energy density of CMB photons ∝ T_cmb^4 ∝ (1+z)^4.
@inline comp1_cmb(z::Real) = (R = typeof(z); R(COMPA) * (R(1.0) + z)^4)
