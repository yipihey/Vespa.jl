# cooling_compton.jl — CMB Compton cooling coefficients as pure functions of z.
#
# The Fortran cooling kernel (cool1d_multi_g.F:196-199) builds two scalars:
#
#   comp1 = compa * (1+z)^4        [code units; cooling coefficient]
#   comp2 = 2.73 * (1+z)           [K; CMB temperature]
#
# where `compa` is the Compton coupling constant stored in code units inside
# grackle.  The raw CGS value is computed in rate_functions.c:1312:
#
#   double comp_rate(double units, chemistry_data *my_chemistry) {
#       return 5.65e-36 / units;
#   }
#
# so the CGS `compa` is 5.65e-36 [erg cm³ K^-4 s^-1] (Peebles 1971).
#
# The two functions below return CGS values (no time_units factor).  The host
# code applies unit conversion at the boundary, matching the same convention as
# rates_cmb.jl (k27/k28 also returned without time_units).
#
# Reference:
#   cool1d_multi_g.F:198-199  (comp1/comp2 assignment)
#   rate_functions.c:1309-1313  (compa = 5.65e-36 / units → raw CGS = 5.65e-36)

# compa: Compton cooling constant [erg cm³ K^-4 s^-1]  (Peebles 1971)
# rate_functions.c:1312: return 5.65e-36 / units;  → CGS value = 5.65e-36
const COMPA = 5.65e-36

# comp2: CMB radiation temperature [K] as a function of redshift z.
# Grackle (cool1d_multi_g.F:199) uses the rounded value 2.73; we use the
# physically correct T_CMB,0 = 2.725 K (Fixsen 2009, consistent with the
# radiation density _OR_FAC in recombination.jl and the CAMB reference fixture).
@inline comp2_cmb(z::Real) = (R = typeof(z); R(2.725) * (R(1.0) + z))

# comp1: Compton coupling coefficient (CGS) as a function of redshift z.
# cool1d_multi_g.F:198:  comp1 = compa * (1._DKIND + zr)**4
# The (1+z)^4 scaling reflects the energy density of CMB photons ∝ T_cmb^4 ∝ (1+z)^4.
@inline comp1_cmb(z::Real) = (R = typeof(z); R(COMPA) * (R(1.0) + z)^4)
