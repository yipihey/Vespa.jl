# recombination_clumping.jl — density-dependent Lyα-mixing recombination.
#
# Extends the Peebles C-factor in `peebles_k2` to account for small-scale baryon
# clumping: the Lyα escape rate R_α (Sobolev escape integral) depends on the
# neutral density *averaged over the Lyα mean-free-path volume* rather than the
# local cell density.  The host MHD code supplies that mean density as a per-cell
# field; we interpolate with mixing fraction f_α(z) from a user table.
#
# Core change (Eq. ★ from the brief):
#   n1s_eff = f_α · n_smoothed + (1-f_α) · n_local
# Replace n_local → n1s_eff in the Λ_2γ term (KL) of the Peebles C-factor; keep
# n_local in the β_e photoionisation term (KB) — "only the escape is non-local".
#
# When f_α = 0 (FA_ZERO default): n1s_eff = n_local → peebles_k2_mixing is
# *bit-identical* to peebles_k2; solve_chem_mixing! is bit-identical to solve_chem!
#
# Rate backend: RecFast analytic α_B (first cut). The `recfast_alpha` function is
# the seam; swap in a LogTable of HyRec values without touching the kernel.
#
# Reference: Jedamzik, Abel & Ali-Haïmoud (2025); see also Peebles (1968),
# Ma & Bertschinger (1995), and recombination.jl for the base RECFAST constants.

export recfast_gauss_factor, recfast_v2_kl_factor, peebles_k2_mixing, n1s_effective
export build_rates_mixing, evolve_cell_mixing, solve_chem_mixing!

# recfast_alpha is defined in recombination.jl (loaded before this file).

# ── RECFAST fudge + v2 Gaussian correction ───────────────────────────────────
#
# The RECFAST recombination "fudge" is NOT a correction to the Λ₂γ escape rate.
# Verified against the canonical codes (HyRec-2 hydrogen.c::rec_TLA_dxHIIdlna and
# CAMB recfast.f90::ION):
#
#   * The fudge `fu` multiplies the case-B recombination coefficient α_B.
#     In the Peebles C-factor it appears as
#         C_eff = fu·(1 + KL) / (1 + KL + fu·KB)
#     i.e. it scales the whole rate and the photoionization term KB in the
#     denominator — NOT the Λ₂γ term KL.  (HyRec puts `Fudge·α_B` on both the
#     recombination prefactor and β; CAMB folds the same `fu` into the C-factor.
#     The two forms are algebraically identical: k2 = fu·α_B·(1+KL)/(1+KL+fu·KB).)
#   * RECFAST v1 uses fu = 1.14 (flat).  RECFAST v2 (CAMB Hswitch=True) uses
#     fu = 1.125 PLUS a multiplicative double-Gaussian correction `gauss(z)` on
#     the Lyα escape factor K = (λ³/8πH)·gauss — which scales BOTH KL and KB.
#
# HyRec's own PEEBLES mode (fu=1) reproduces our previous "v1" error profile
# (+8% at z=700, falling to <1% at z=1100): that growing low-z tail is the
# intrinsic error of the three-level atom, not a bug.  Applying fu=1.14 to α_B
# collapses it to <1.5% everywhere — that is the physically correct fix.

const _RECFAST_V1_FUDGE = 1.14    # RECFAST v1 (flat fudge on α_B)
const _RECFAST_V2_FUDGE = 1.125   # RECFAST v2 (fudge on α_B; Gaussian on K)

"""
    recfast_gauss_factor(z) -> Float64

RECFAST v2 multiplicative correction to the Lyα escape factor K (CAMB
`Hswitch=True`; recfast.f90 line `K = CK/Hz*(1 + AGauss1·… + AGauss2·…)`):

  gauss = 1 + G₁(z) + G₂(z)
  G₁ = -0.14  × exp(-((ln(1+z) - 7.28) / 0.18)²)   [peak z ≈ 1449]
  G₂ =  0.079 × exp(-((ln(1+z) - 6.73) / 0.33)²)   [peak z ≈  836]

This scales K — and therefore BOTH KL and KB in the Peebles C-factor — bringing
x_e(z) within ~0.1-0.3% of HyRec. Returns 1.0 for RECFAST v1 (no Hswitch).
"""
@inline function recfast_gauss_factor(z::Real)
    lnzp1 = log(1.0 + Float64(z))
    g1 = -0.14  * exp(-((lnzp1 - 7.28) / 0.18)^2)
    g2 =  0.079 * exp(-((lnzp1 - 6.73) / 0.33)^2)
    return 1.0 + g1 + g2
end

# Backward-compatible alias (deprecated): the old name implied the correction
# applied to the Λ₂γ "KL" term, which was incorrect.  Kept so external callers
# don't break; returns the K-factor Gaussian correction.
@inline recfast_v2_kl_factor(z::Real) = recfast_gauss_factor(z)

# ── Eq. ★ : effective neutral density ────────────────────────────────────────

"""
    n1s_effective(nHI_local, n_smoothed, Xe_mean, f_alpha, ::Val{SN}) -> same units

Effective neutral-H number density for the Sobolev escape integral (Eq. ★):
    n1s_eff = f_α · n1s_smoothed + (1-f_α) · n1s_local.

`n_smoothed` is the smoothed H field from the host, interpreted as:
  - SN=true  : already the smoothed *neutral* density  (n1s_smoothed = n_smoothed)
  - SN=false : total smoothed H density; approximate  n1s_smoothed ≈ n_smoothed·(1−Xe_mean)
    using the global mean ionisation fraction Xe_mean (accurate because x_e varies
    slowly across the mixing length near the recombination epoch).

Implemented as a single fused multiply-add: branch-free in the hot path. Pure.
"""
@inline function n1s_effective(nHI_local::T, n_smoothed::T, Xe_mean::T, f_alpha::T,
                               ::Val{SN}) where {T,SN}
    n1s_sm = SN ? n_smoothed : n_smoothed * (one(T) - Xe_mean)
    return muladd(f_alpha, n1s_sm - nHI_local, nHI_local)
end

# ── Generalised Peebles k2 ────────────────────────────────────────────────────

"""
    peebles_k2_mixing(T, nHI_local, nHI_eff, Hz; fudge=1, gauss=1) -> k2 [cm³/s]

CaseB H recombination rate with the RECFAST Peebles C-factor, using an effective
neutral density `nHI_eff` [cm⁻³] for the Λ_2γ escape term (KL) but keeping the
local density `nHI_local` for the β_e photoionisation term (KB):

    K  = gauss · λ³/(8π·Hz)                  (Lyα Sobolev escape; gauss = v2 correction)
    KL = K · Λ_2γ · n1s_eff                  (mixing density; non-local escape)
    KB = K · β_e  · n1s_local                (local density for β_e)
    C  = fudge · (1 + KL) / (1 + KL + fudge·KB)
    k2 = α_B · C                             [cm³/s]

This is the exact RECFAST recombination coefficient (verified against HyRec-2
`rec_TLA_dxHIIdlna` and CAMB `recfast.f90`):
  * `fudge` (fu) multiplies α_B — it scales the whole rate and the KB term in
    the C-factor denominator, NOT the Λ₂γ term KL.  fudge=1 → pure Peebles
    (HyRec PEEBLES); 1.14 → RECFAST v1; 1.125 → RECFAST v2 base.
  * `gauss` is the multiplicative Gaussian correction on the Lyα K-factor
    (CAMB Hswitch; `recfast_gauss_factor(z)`), scaling BOTH KL and KB. =1 for v1.

When nHI_eff = nHI_local and fudge = gauss = 1 this is bit-identical to
`peebles_k2`. Pure.
"""
@inline function peebles_k2_mixing(T::Real, nHI_local::Real, nHI_eff::Real, Hz::Real;
                                   fudge::Real = one(typeof(T)),
                                   gauss::Real = one(typeof(T)))
    R = typeof(T)
    aB           = recfast_alpha(T)
    fu           = R(fudge)
    n1s_local_m3 = R(nHI_local) * R(1.0e6)        # cm⁻³ → m⁻³
    n1s_eff_m3   = R(nHI_eff)   * R(1.0e6)
    bet = aB * (R(_REC_CR) * T)^R(1.5) * exp(-R(_REC_CDB) / T)
    K   = R(gauss) * R(_REC_LAM)^3 / (R(8.0) * R(π) * R(Hz))  # v2 Gaussian scales K
    KL  = K * R(_REC_A8) * n1s_eff_m3
    KB  = K * bet        * n1s_local_m3
    C   = fu * (one(R) + KL) / (one(R) + KL + fu * KB)        # fudge on α_B (RECFAST)
    return aB * R(1.0e6) * C                                  # m³/s → cm³/s
end

# ── Rate assembler (identical to build_rates but k2 uses mixing) ─────────────

"""
    build_rates_mixing(T, Trad, nHI, nHI_eff, Hz; fudge=1, gauss=1, deuterium=false) -> NamedTuple

Like `build_rates` but substitutes `peebles_k2_mixing(T, nHI, nHI_eff, Hz; fudge, gauss)`
for k2. All other rates are identical. Pure.
"""
@inline function build_rates_mixing(T, Trad, nHI, nHI_eff, Hz;
                                    fudge::Real = one(typeof(T)),
                                    gauss::Real = one(typeof(T)),
                                    deuterium::Bool = false)
    R      = typeof(T)
    k2_val = peebles_k2_mixing(T, nHI, nHI_eff, Hz; fudge=fudge, gauss=gauss)
    # β₁s = CMB photoionisation of H(1s): evaluate at Trad (see build_rates) so it does
    # NOT spuriously Saha-ionise UV-heated low-z gas where T≫Trad.
    k_b1s  = beta1s_freq(Trad) * k2_val / (recfast_alpha(T) * R(1.0e6))
    she1, she2 = helium_saha_pair(Trad)
    base = (; k1=k1(T), k2=k2_val,
            k3=k3(T), k4=k4(T), k5=k5(T),
            k6=k6(T), k7=k7(T), k8=k8(T), k9=k9(T), k10=k10(T), k11=k11(T),
            k12=k12(T), k13=k13(T), k14=k14(T), k15=k15(T), k16=k16(T), k17=k17(T),
            k18=k18(T), k19=k19(T), k22=k22(T), k57=k57(T), k58=k58(T),
            k27=k27_cmb(Trad), k28=k28_cmb(Trad), k_beta1s=k_b1s,
            she1=she1, she2=she2)
    deuterium || return base
    return merge(base, (; k50=k50(T), k51=k51(T), k52=k52(T), k53=k53(T),
                        k54=k54(T), k55=k55(T), k56=k56(T)))
end

# ── Per-cell mixing subcycler ─────────────────────────────────────────────────

"""
    evolve_cell_mixing(rho, e, HII_m, H2I_m, HDI_m, n_sm_cgs, dt, z;
                       f_alpha, Xe_mean, smoothed_is_neutral, hubble, Om, OL,
                       fh, deuterium) -> (e, HII_m, H2I_m, HDI_m)

Sub-cycle one cell over macro-step `dt` [s] with Lyα-mixing recombination.
Identical to `evolve_cell` except `build_rates_mixing` is called each substep with
`nHI_eff` computed from the per-cell smoothed density `n_sm_cgs` [cm⁻³].

`n_sm_cgs` is the smoothed H number density from the host (physical CGS). Interpreted
as total n_H when `smoothed_is_neutral=Val(false)` (default; approximates n1s via
global Xe_mean), or as n1s directly when `Val(true)`. Pure; allocation-free.
"""
@inline function evolve_cell_mixing(rho, e, HII_m, H2I_m, HDI_m,
                                    n_sm_cgs, dt, z;
                                    f_alpha    = zero(typeof(e)),
                                    Xe_mean    = zero(typeof(e)),
                                    smoothed_is_neutral::Val{SN} = Val(false),
                                    fudge      = one(typeof(e)),
                                    gauss      = one(typeof(e)),
                                    hubble = 71.0, Om = 0.27, OL = 0.73,
                                    fh = FH_DEFAULT,
                                    deuterium::Bool = false,
                                    helium::Bool = false,
                                    HeII_m = zero(typeof(e)),
                                    hubble_expansion::Bool = false,
                                    uvb::Bool = false,
                                    GamHI = zero(typeof(e)), GamHeI = zero(typeof(e)),
                                    GamHeII = zero(typeof(e)),
                                    piHI = zero(typeof(e)), piHeI = zero(typeof(e)),
                                    piHeII = zero(typeof(e)),
                                    metals = nothing) where {SN}
    R    = typeof(e)
    mh   = R(MH); tiny = R(_SUB_TINY)
    d    = rho / mh
    Hz   = hubble_z_of(R(z); hubble = hubble, Om = Om, OL = OL)
    Tc   = comp2_cmb(R(z))
    c1   = comp1_cmb(R(z))
    nHe4 = (one(R) - R(fh)) * d                # total He in ×4 (mass-equiv) convention
    nHe  = nHe4 / R(4)                          # total He number density [cm⁻³]
    nH_h = R(fh) * d                            # hydrogen number density [cm⁻³]
    fHe  = nHe / nH_h                           # n_He/n_H
    # He⁺ number density carried as state (HeII_m is the He⁺ MASS density; He mass =
    # 4·m_H, so n(He⁺) = HeII_m/(4·m_H) and yHeII(×4) = HeII_m/m_H).
    nHeII = helium ? (HeII_m / mh) / R(4) : zero(R)
    yHeI  = nHe4                                # neutral-He reservoir for cooling/T

    yHII  = HII_m / mh
    yH2I  = H2I_m / mh
    yHDI  = deuterium ? HDI_m / mh : zero(R)
    yHI   = max((R(fh)*rho - HII_m - H2I_m) / mh, tiny)
    yde   = yHII
    yHM   = tiny; yH2II = tiny
    yDI   = deuterium ? R(DTOH_SEED)*yHI  : zero(R)
    yDII  = deuterium ? R(DTOH_SEED)*yHII : zero(R)

    fa  = R(f_alpha)
    Xem = R(Xe_mean)
    nsm = R(n_sm_cgs)
    fud = R(fudge)
    gss = R(gauss)
    # UV-background photoionisation [s⁻¹] and photoheating [erg s⁻¹ per ion] for this
    # step (all 0 unless a UVB was supplied to solve_chem_mixing!).
    gHI = R(GamHI); gHeI = R(GamHeI); gHeII = R(GamHeII)
    pHI = R(piHI); pHeI = R(piHeI); pHeII = R(piHeII)

    ttot = zero(R)
    iter = 0
    while ttot < dt && iter < _SUB_ITMAX
        iter += 1
        rem = dt - ttot

        T = gas_temperature(rho, e, yHI, yHII, yHeI/R(4), tiny, tiny, yde,
                            yHM, yH2I/R(2), yH2II/R(2); gamma = GAMMA_DEFAULT)
        Trad = Tc

        # effective neutral density for the Sobolev escape rate
        nHI_eff = n1s_effective(yHI, nsm, Xem, fa, smoothed_is_neutral)
        K = build_rates_mixing(T, Trad, yHI, nHI_eff, Hz;
                               fudge = fud, gauss = gss, deuterium = deuterium)

        # UV-background He photoionisation equilibrium (default He path).  Solve the
        # collisional-radiative + photo He equilibrium ONCE, up front, so this substep's
        # cooling, photoheating and electron balance all see the SAME He state (and we
        # hand it to network_step instead of letting it re-solve).  helium=true carries
        # its own advected He⁺ (Task B will add Γ there); with no UVB this branch is
        # skipped → the original default path is bit-identical.
        uvb_eq    = uvb && !helium
        nHeII_now = zero(R)
        yHeII_eq  = zero(R); yHeIII_eq = zero(R)
        if uvb_eq
            ne0 = max(yde, tiny)
            _, nHeII_e, nHeIII_e =
                helium_equilibrium(K.she1, K.she2, K.k3, K.k4, K.k5, K.k6, ne0, nHe;
                                   GamHeI = gHeI, GamHeII = gHeII)
            yHeII_eq  = R(4) * nHeII_e                 # ×4 mass-equiv convention
            yHeIII_eq = R(4) * nHeIII_e
            yHeI      = max(nHe4 - yHeII_eq - yHeIII_eq, tiny)
            nHeII_now = nHeII_e
        elseif helium
            nHeII_now = nHeII                          # carried (start-of-substep) He⁺
        end

        nHD  = yHDI / R(3)
        edot = cooling_edot(yHI, yHII, yHeI/R(4), yde, yH2I/R(2), nHD, T, R(z);
                            nH = R(fh)*d, metals = metals)
        if T <= R(1.01)*R(MIN_TEMPERATURE) && edot < zero(R)
            edot = zero(R)
        end
        # UV-background photoheating [erg cm⁻³ s⁻¹], a +edot source (energy per
        # photoionisation × number density of that species; n_HeI = yHeI/4).  Applied
        # after the floor shutoff so heating can lift gas off the temperature floor.
        edot += pHI*yHI + pHeI*(yHeI/R(4)) + pHeII*nHeII_now
        if hubble_expansion
            edot -= R(2) * Hz * e * rho
        end

        dedot, HIdot = _de_hi_dot(yHI, yHII, yde, yH2I, yHM, yH2II,
                                  yHeI, tiny, tiny, K; GamHI = gHI)
        dtit = min(_step10(yde, dedot), _step10(yHI, HIdot), rem, R(0.5)*dt)

        edot_c    = -c1 * (T - Tc) * yde
        edot_rest = edot - edot_c
        Kc        = c1 * yde * (T / e) / rho
        stiff     = Kc * rem > one(R)
        de_spec   = (stiff ? edot_rest : edot) / rho
        dtit = min(dtit, _step10(e, de_spec))

        if stiff
            B = (c1*yde*Tc + edot_rest) / rho
            e = (e + B*dtit) / (one(R) + Kc*dtit)
        else
            e = e + (edot/rho)*dtit
        end
        e = max(e, tiny)

        # ── Helium ionisation (evolved He⁺ with the full He I freeze-out) ─────
        # He³⁺/He²⁺ and the z≳3000 He⁺ plateau are fast ⇒ 3-level Saha (exact).
        # At z≲3000 (He²⁺≈0) He⁺→He⁰ freezes out ⇒ integrate He⁺ with the
        # HyRec He I rate (helium_HeI_rate_AB), backward-Euler.  Carried in nHeII.
        if helium
            ne = max(yde, tiny)
            if R(z) > R(3000.0)
                r1 = K.she1 / ne;  r2 = K.she2 / ne
                den = one(R) + r1 + r1*r2
                nHeII   = nHe * r1 / den
                nHeIII  = nHe * r1 * r2 / den
            else
                xHeII = nHeII / nH_h
                A, B  = helium_HeI_rate_AB(Trad, nH_h, Hz, yHI/nH_h, xHeII, fHe)
                nHeII = ((xHeII + A*dtit) / (one(R) + B*dtit)) * nH_h
                nHeIII = nHeII * K.she2 / ne                  # He²⁺ Saha (≈0 here)
            end
            yHeII_x  = R(4) * nHeII                            # ×4 mass-equiv convention
            yHeIII_x = R(4) * nHeIII
            yHeI     = max(nHe4 - yHeII_x - yHeIII_x, tiny)   # for next substep's cooling/T
            s = network_step(d, fh, yHI, yHII, yde, yH2I, yHM, yH2II,
                             yDI, yDII, yHDI, K, dtit; deuterium = deuterium,
                             yHeII_in = yHeII_x, yHeIII_in = yHeIII_x, GamHI = gHI)
        elseif uvb_eq
            # consume the up-front He equilibrium (already includes Γ_HeI/Γ_HeII)
            s = network_step(d, fh, yHI, yHII, yde, yH2I, yHM, yH2II,
                             yDI, yDII, yHDI, K, dtit; deuterium = deuterium,
                             yHeII_in = yHeII_eq, yHeIII_in = yHeIII_eq, GamHI = gHI)
        else
            s = network_step(d, fh, yHI, yHII, yde, yH2I, yHM, yH2II,
                             yDI, yDII, yHDI, K, dtit; deuterium = deuterium,
                             GamHI = gHI, GamHeI = gHeI, GamHeII = gHeII)
        end
        yHI=s.yHI; yHII=s.yHII; yde=s.yde; yH2I=s.yH2I; yHM=s.yHM
        yH2II=s.yH2II; yDI=s.yDI; yDII=s.yDII; yHDI=s.yHDI

        # Enforce H-nuclei conservation (only with a UVB, so the validated no-UVB path
        # is bit-identical).  The operator-split, Gauss-Seidel backward-Euler updates HI
        # and HII as separate fixed points; under strong photoionisation their sum drifts
        # a few % from the true H budget.  Renormalise the H species (each y counts H
        # nuclei: yH2I=2·nH₂, yH2II=2·nH₂⁺) to the exact total — the standard
        # `make_consistent` step.  nₑ is re-derived from charge balance next substep.
        if uvb
            SH = yHI + yHII + yH2I + yHM + yH2II
            fH = (R(fh) * d) / max(SH, tiny)
            yHI *= fH; yHII *= fH; yH2I *= fH; yHM *= fH; yH2II *= fH
        end

        ttot += dtit
    end

    return e, yHII*mh, yH2I*mh, (deuterium ? yHDI*mh : HDI_m),
           (helium ? R(4)*nHeII*mh : HeII_m)
end

# ── KA kernel ────────────────────────────────────────────────────────────────

@kernel function _evolve_mixing_k!(e, HII, H2I, HDI, HeII, @Const(rho), @Const(n_sm),
                                   du, vu2, tu, dt, z,
                                   f_alpha, Xe_mean, fudge, gauss,
                                   hubble, Om, OL, fh, deut, hel, hub_exp,
                                   uvb_on, GamHI, GamHeI, GamHeII, piHI, piHeI, piHeII,
                                   @Const(aC), @Const(aO), @Const(aSi), @Const(aFe), hasmetals,
                                   ::Val{SN}) where {SN}
    i = @index(Global)
    @inbounds begin
        T    = eltype(e)
        hd_in    = deut ? HDI[i]*du : zero(T)
        he_in    = hel  ? HeII[i]*du : zero(T)
        # n_sm[i] is smoothed baryon mass density [code units] (same units as rho);
        # multiply by fh to extract the H fraction before converting to number density.
        n_sm_cgs = n_sm[i] * du * T(fh) / T(MH)
        mab = hasmetals ? MetalAbundances{T}(aC[i], aO[i], aSi[i], aFe[i]) :
                          MetalAbundances{T}()
        en, hii, h2, hd, he = evolve_cell_mixing(
            rho[i]*du, e[i]*vu2, HII[i]*du, H2I[i]*du, hd_in,
            n_sm_cgs, dt*tu, z;
            f_alpha  = T(f_alpha),
            Xe_mean  = T(Xe_mean),
            smoothed_is_neutral = Val(SN),
            fudge = T(fudge), gauss = T(gauss),
            hubble   = T(hubble), Om = T(Om), OL = T(OL),
            fh       = T(fh), deuterium = deut,
            helium = hel, HeII_m = he_in, hubble_expansion = hub_exp,
            uvb = uvb_on,
            GamHI = T(GamHI), GamHeI = T(GamHeI), GamHeII = T(GamHeII),
            piHI = T(piHI), piHeI = T(piHeI), piHeII = T(piHeII), metals = mab)
        e[i]   = en  / vu2
        HII[i] = hii / du
        H2I[i] = h2  / du
        deut && (HDI[i] = hd / du)
        hel  && (HeII[i] = he / du)
    end
end

# ── Public API ────────────────────────────────────────────────────────────────

"""
    solve_chem_mixing!(rho, e_int, HII, H2I, n_smoothed; [HDI,]
                       a_value, dt, density_units, length_units, time_units,
                       fa_table, Xe_mean, smoothed_is_neutral,
                       hubble, Om, OL, fh, deuterium, backend, precision)

Evolve the v2026 reduced chemistry/cooling with Lyα-mixing recombination over `dt`
(code time units) for every cell.  Mirrors `solve_chem!` but takes one extra
positional argument:

  `n_smoothed` — smoothed baryon mass density [same code units as `rho`], pre-computed
  by the host MHD code as the mean baryon density over the Lyα mean-free-path volume.
  The kernel converts to H number density via `n_smoothed * density_units * fh / MH`.

Keyword arguments:
  `uvb`                 — optional metagalactic UV/X-ray background (`UVBackground`,
                           e.g. `fg20_uvb()`).  When given, its rates at this step's `z`
                           are threaded into the network: Γ_HI photoionises H (HI→HII+e
                           in `network_step`), Γ_HeI/Γ_HeII drive the He ionisation
                           equilibrium, and piHI/piHeI/piHeII photoheat the gas (a +edot
                           source).  `nothing` (default) ⇒ primordial-only, bit-identical
                           to the no-UVB path.
  `fa_table`            — `FAlphaTable` with f_α(z). Default: `FA_ZERO` (f_α ≡ 0,
                           no mixing, bit-identical to `solve_chem!`).
  `Xe_mean`             — global mean free-electron fraction n_e/n_H for this step.
                           Used to convert smoothed n_H → n1s when smoothed_is_neutral=false.
  `smoothed_is_neutral`  — if `true`, `n_smoothed` is already the smoothed neutral
                           density n1s; if `false` (default), approximate n1s_smoothed ≈
                           n_smoothed × (1 − Xe_mean).
  `recfast_fudge`       — RECFAST fudge `fu` on α_B (enters the C-factor as
                           fu·(1+KL)/(1+KL+fu·KB)). Default 1.0 (the pure-Peebles
                           default; = HyRec PEEBLES mode). Set to 1.14 for
                           RECFAST v1.  Overridden to 1.125 when `recfast_hswitch`.
  `recfast_hswitch`     — if `true`, use RECFAST v2: fudge fu = 1.125 on α_B PLUS
                           a multiplicative Gaussian correction gauss(z) =
                           1 + G₁(z) + G₂(z) on the Lyα escape factor K (two
                           Gaussians in ln(1+z), CAMB 1.6.6 defaults; scales both
                           KL and KB).  Brings x_e(z) within ~0.1-0.3% of HyRec.
                           Default: `false`.
  `HeII`, `helium`      — helium ionisation handling:
                           • DEFAULT (`helium=false`, no `HeII`): He I/II/III in
                             Saha equilibrium with the CMB each step.  Total x_e is
                             correct to <0.1% at z≳3000 and z≲1700, but ~3% LOW in
                             the He⁺→He⁰ freeze-out window z≈2000-2500 (Saha has no
                             radiative-transfer delay).  This is the shipped default:
                             He⁺⁺/He⁺ and the deep-recombination x_e are exact, and
                             the only error is the quantified ~3% transient at z≈2000.
                           • OPT-IN (`helium=true` + an advected `HeII` vector, the
                             He⁺ MASS density = 4·n(He⁺)·m_H): evolve He⁺ with the
                             full He I recombination (HyRec radiative transfer),
                             capturing the freeze-out → total x_e <0.1% vs HyRec
                             across z=1900-8000.  He⁺⁺ stays Saha (always fast);
                             report total x_e via `total_electron_fraction(xHII,
                             xHeII, nH, Trad)`.  Needed only when x_e in z≈2000-2500
                             must be better than ~3%.
"""
function solve_chem_mixing!(rho::AbstractVector, e_int::AbstractVector,
                            HII::AbstractVector, H2I::AbstractVector,
                            n_smoothed::AbstractVector;
                            HDI::Union{Nothing,AbstractVector} = nothing,
                            HeII::Union{Nothing,AbstractVector} = nothing,
                            a_value::Real, dt::Real,
                            density_units::Real, length_units::Real, time_units::Real,
                            fa_table::FAlphaTable = FA_ZERO,
                            Xe_mean::Real = 0.0,
                            smoothed_is_neutral::Bool = false,
                            recfast_fudge::Real = 1.0,
                            recfast_hswitch::Bool = false,
                            hubble_expansion::Bool = false,
                            uvb::Union{Nothing,UVBackground} = nothing,
                            metals = nothing,
                            hubble::Real = 71.0, Om::Real = 0.27, OL::Real = 0.73,
                            fh::Real = 0.76, deuterium::Bool = false,
                            helium::Bool = false,
                            backend::Symbol = :cpu, precision::Type = Float64)
    n = length(rho)
    @assert length(e_int) == n && length(HII) == n && length(H2I) == n
    @assert length(n_smoothed) == n
    deut = deuterium && HDI !== nothing
    deut && @assert length(HDI) == n
    hel = helium && HeII !== nothing
    hel && @assert length(HeII) == n
    hasmetals = metals !== nothing
    if hasmetals
        @assert length(metals.C)==n && length(metals.O)==n &&
                length(metals.Si)==n && length(metals.Fe)==n
    end

    P   = precision
    be  = ChemistryKernels.backend(backend)
    du  = P(density_units)
    vu2 = P((length_units / time_units)^2)
    tu  = P(time_units)
    z   = P(1.0 / a_value - 1.0)

    # f_α for this step (scalar, from the redshift table)
    f_alpha = P(fa_at(fa_table, Float64(z)))

    # RECFAST fudge on α_B (C_eff = fu·(1+KL)/(1+KL+fu·KB)) and the v2 Gaussian
    # correction on the Lyα escape K (scales both KL and KB):
    #   recfast_hswitch=false: fudge = recfast_fudge (1.0 = pure Peebles), no Gaussian
    #   recfast_hswitch=true:  fudge = 1.125 (RECFAST v2) + Gaussian gauss(z)
    fudge = P(recfast_hswitch ? _RECFAST_V2_FUDGE : recfast_fudge)
    gauss = P(recfast_hswitch ? recfast_gauss_factor(Float64(z)) : 1.0)

    # UV-background rates for this step (scalars, evaluated once at z).  Mapping:
    # uvb_rates → (k24=Γ_HI, k25=Γ_HeII, k26=Γ_HeI, piHI, piHeI, piHeII).
    uvb_on = uvb !== nothing
    gHI = gHeI = gHeII = zero(P); pHI = pHeI = pHeII = zero(P)
    if uvb_on
        (k24, k25, k26, qHI, qHeI, qHeII) = uvb_rates(uvb, Float64(z))
        gHI = P(k24); gHeI = P(k26); gHeII = P(k25)
        pHI = P(qHI); pHeI = P(qHeI); pHeII = P(qHeII)
    end

    d_rho = to_device(be, collect(rho),        P)
    d_e   = to_device(be, collect(e_int),       P)
    d_HII = to_device(be, collect(HII),         P)
    d_H2I = to_device(be, collect(H2I),         P)
    d_HDI = deut ? to_device(be, collect(HDI),  P) : device_zeros(be, P, (n,))
    d_HeII = hel ? to_device(be, collect(HeII), P) : device_zeros(be, P, (n,))
    d_nsm = to_device(be, collect(n_smoothed),  P)

    d_aC = hasmetals ? to_device(be, collect(metals.C),  P) : device_zeros(be, P, (n,))
    d_aO = hasmetals ? to_device(be, collect(metals.O),  P) : device_zeros(be, P, (n,))
    d_aSi= hasmetals ? to_device(be, collect(metals.Si), P) : device_zeros(be, P, (n,))
    d_aFe= hasmetals ? to_device(be, collect(metals.Fe), P) : device_zeros(be, P, (n,))

    SN = smoothed_is_neutral
    _evolve_mixing_k!(be)(d_e, d_HII, d_H2I, d_HDI, d_HeII, d_rho, d_nsm,
                          du, vu2, tu, P(dt), z,
                          f_alpha, P(Xe_mean), fudge, gauss,
                          P(hubble), P(Om), P(OL), P(fh), deut, hel, hubble_expansion,
                          uvb_on, gHI, gHeI, gHeII, pHI, pHeI, pHeII,
                          d_aC, d_aO, d_aSi, d_aFe, hasmetals, Val(SN);
                          ndrange = n)

    e_int      .= to_host(d_e)
    HII        .= to_host(d_HII)
    H2I        .= to_host(d_H2I)
    deut && (HDI .= to_host(d_HDI))
    hel  && (HeII .= to_host(d_HeII))
    return nothing
end
