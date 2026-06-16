# rates_cmb.jl — CMB photo-destruction rates for H- (k27) and H2+ (k28).
#
# These rates are not tabulated functions of gas temperature; they are
# evaluated per-call as literal analytic formulas of the CMB radiation
# temperature Trad = 2.73*(1+z) [K], following the original network of
# Abel/Anninos et al. 1997:
#
#   k27: H- + γ_CMB → H + e       (Galli & Palla 1998, H4; de Jong 1972)
#   k28: H2+ + γ_CMB → H + H+     (Galli & Palla 1998, H9, LTE; Argyros 1974 /
#                                   Stancil 1994 — LTE because the CMB keeps H2+
#                                   vibrational levels thermally excited)
#
# Units: CGS s^-1.  The host multiplies by time_units before handing the rate
# to the solver, exactly as the UV-background photo-rates are handled (k27 *=
# time_units).  These functions return the raw CGS value WITHOUT the time_units
# factor.

# ── k27 : H- + γ_CMB → H + e  (GP98 H4) ─────────────────────────────────────
# Rate (per CGS, before the time_units factor tu):
#   k27 = 1.1e-1 * Trad^2.13 * exp(-8823.0 / Trad)
@inline function k27_cmb(Trad::Real)
    R = typeof(Trad)
    return R(1.1e-1) * Trad^R(2.13) * exp(-R(8823.0) / Trad)
end
@scalarkernel k27_cmb

# ── k28 : H2+ + γ_CMB → H + H+  (GP98 H9, LTE) ─────────────────────────────
# Rate (per CGS, before the time_units factor tu):
#   k28 = 1.63e7 * exp(-32400.0 / Trad)
@inline function k28_cmb(Trad::Real)
    R = typeof(Trad)
    return R(1.63e7) * exp(-R(32400.0) / Trad)
end
@scalarkernel k28_cmb
