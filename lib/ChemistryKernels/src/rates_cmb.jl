# rates_cmb.jl — CMB photo-destruction rates for H- (k27) and H2+ (k28).
#
# These rates are NOT tabulated in grackle's rate_functions.c.  They are
# computed per-call in solve_chemistry.c:152-161 as literal analytic formulas
# of the CMB radiation temperature Trad = 2.73*(1+z) [K]:
#
#   k27: H- + γ_CMB → H + e       (Galli & Palla 1998, H4; de Jong 1972)
#   k28: H2+ + γ_CMB → H + H+     (Galli & Palla 1998, H9, LTE; Argyros 1974 /
#                                   Stancil 1994 — LTE because the CMB keeps H2+
#                                   vibrational levels thermally excited)
#
# Units: CGS s^-1.  The host multiplies by time_units before handing the rate
# to the Fortran solver, exactly as the UV-background photo-rates are handled
# (update_UVbackground_rates.c: k27 *= time_units).  These functions return the
# raw CGS value WITHOUT the time_units factor.
#
# Reference: solve_chemistry.c:158-160 (yipihey/grackle, branch cmb-photo-rates)
# Constants cross-checked against solve_chemistry.c — no grackle oracle for these.

# ── k27 : H- + γ_CMB → H + e  (GP98 H4) ─────────────────────────────────────
# solve_chemistry.c:158:
#   my_uvb_rates.k27 += 1.1e-1 * pow(Trad, 2.13) * exp(-8823.0 / Trad) * tu;
@inline function k27_cmb(Trad::Real)
    R = typeof(Trad)
    return R(1.1e-1) * Trad^R(2.13) * exp(-R(8823.0) / Trad)
end
@scalarkernel k27_cmb

# ── k28 : H2+ + γ_CMB → H + H+  (GP98 H9, LTE) ─────────────────────────────
# solve_chemistry.c:160:
#   my_uvb_rates.k28 += 1.63e7 * exp(-32400.0 / Trad) * tu;
@inline function k28_cmb(Trad::Real)
    R = typeof(Trad)
    return R(1.63e7) * exp(-R(32400.0) / Trad)
end
@scalarkernel k28_cmb
