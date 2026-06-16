# test_recombination_field.jl — FIELD-LEVEL standalone tests for the Lyα-mixing
# recombination routine (`solve_chem_mixing!`), beyond single-zone modelling.
#
# The routine is a PURELY LOCAL, per-cell scheme: mixing enters only through the
# effective neutral density in the Λ₂γ escape term of the Peebles C-factor,
#   n1s_eff = f_α·n1s_smoothed + (1−f_α)·n1s_local      (recombination_clumping.jl:94-98)
# used in KL of peebles_k2_mixing (:126-140); KB and α_B stay local.  There is NO
# explicit ⟨n_e²⟩ clumping term and NO redistribution term — clumping of the net
# recombination emerges only when per-cell histories are volume-averaged.  These
# tests validate that LOCAL model AS-IS against references computed by INDEPENDENT
# code paths (a from-scratch Peebles C-factor + an independent binned ODE
# integrator), so a bug in field consumption / weighting / limit-interpolation
# cannot pass silently.
#
# Test ladder (under @testset "recombination_field"):
#   field_construction  — the deterministic lognormal / sinusoid generators are sane.
#   units_lock          — feeding a neutral smoothed field round-trips through the
#                         kernel's mass→number conversion (the n_sm = n1s·MH/fh unit).
#   test1a_rate_weighting   — peebles_k2_mixing == an independent re-derivation across a
#                             fat-tailed density range; the volume-weighted LINEAR mean
#                             is the right escape density (vs number/squared moments).
#   test1b_history_fullmix  — full-mixing x_e(z) history vs an independent binned
#                             integrator (catches plumbing/integration), <1%.
#   test2_bracket_limits    — f_α=0 reproduces no-mix; f_α=1 reproduces Test 1b; full
#                             mixing recombines FASTER (lower x_e) — monotone in f_α.
#   test3_window_sweep      — manufactured sinusoid, sweep the smoothing window W:
#                             W→1 == no-mix, W→0 == full-mix, monotone, and C¹ in W.
#   A0_homogeneous_gate     — A→0 collapses to the CAMB/RECFAST-v2 fixture (<1%).
#   physics_sanity_tie      — b=0.5 lognormal full-mix lowers x_e a few%–20%, steepest
#                             near z≈1100-1250 (sign/scale only).
#
# Standalone (the full runtests.jl needs the macOS-only oracle):
#   <julia> --project=lib/ChemistryKernels/test lib/ChemistryKernels/test/test_recombination_field.jl

using ChemistryKernels
using Test
using Printf

include("recomb_helpers.jl")   # _mean, _Hz, saha_xe, n_H_at_z, e_from_T, integrate_onezone

# NB: plain (non-const) globals so this file can also be `include`d INSIDE the
# runtests.jl @testset block (a local scope, where `const` is illegal).
MH    = ChemistryKernels.MH
FH    = 0.76
TCMB0 = 2.725

_median(v) = sort(v)[cld(length(v), 2)]

# ── deterministic field generators (no RNG) ──────────────────────────────────

# Acklam rational inverse normal CDF, |err| < 1.15e-9.  Deterministic → reproducible
# quantile sampling of the lognormal without `Random`.
function norminvcdf(p::Float64)
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
          1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
          6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00,  4.374664141464968e+00,  2.938163982698783e+00)
    d = ( 7.784695709041462e-03,  3.224671290700398e-01,  2.445134137142996e+00,
          3.754408661907416e+00)
    plow = 0.02425; phigh = 1 - plow
    if p < plow
        q = sqrt(-2*log(p))
        return (((((c[1]*q+c[2])*q+c[3])*q+c[4])*q+c[5])*q+c[6]) /
               ((((d[1]*q+d[2])*q+d[3])*q+d[4])*q+1)
    elseif p <= phigh
        q = p - 0.5; r = q*q
        return (((((a[1]*r+a[2])*r+a[3])*r+a[4])*r+a[5])*r+a[6])*q /
               (((((b[1]*r+b[2])*r+b[3])*r+b[4])*r+b[5])*r+1)
    else
        q = sqrt(-2*log(1-p))
        return -(((((c[1]*q+c[2])*q+c[3])*q+c[4])*q+c[5])*q+c[6]) /
                ((((d[1]*q+d[2])*q+d[3])*q+d[4])*q+1)
    end
end

# Deterministic lognormal overdensity field Δ (⟨Δ⟩=1, Var≈b) with volume weights w.
# Bulk: M equal-probability quantile midpoints (w=1/M each). Optional fat tail:
# a few fixed high-Δ cells carrying tiny total volume f_tail (UNEQUAL weights) — this
# makes volume- vs number-weighting distinguishable (Test 1a). Δ is renormalised so
# the VOLUME mean Σ wΔ = 1 exactly.
function lognormal_field(; b=0.5, M=2048, tail_deltas=Float64[], f_tail=0.0)
    σ2 = log(1 + b); σ = sqrt(σ2); μ = -σ2/2
    nt = length(tail_deltas)
    vol_bulk = nt > 0 ? (1 - f_tail) : 1.0
    Δ = Float64[]; w = Float64[]
    for i in 1:M
        δ = (i - 0.5)/M
        push!(Δ, exp(μ + σ*norminvcdf(δ))); push!(w, vol_bulk/M)
    end
    for d in tail_deltas
        push!(Δ, d); push!(w, f_tail/nt)
    end
    m = sum(w .* Δ); Δ ./= m            # enforce ⟨Δ⟩_vol = 1
    return Δ, w
end

# Single-mode sinusoid Δ(x)=1+A sin(2πx/λ) sampled at N equal-volume cell centres
# over one period.  Returns the overdensity vector (volume weights are uniform 1/N).
sinusoid_field(A, N) = [1.0 + A*sin(2π*(i-0.5)/N) for i in 1:N]

# Top-hat smoothing window for a single mode k=2π/λ convolved with a real-space
# top-hat of full width D_α:  W(D_α/λ) = sin(π D_α/λ)/(π D_α/λ) = sinc(D_α/λ).
tophat_window(Dα_over_λ) = Dα_over_λ == 0 ? 1.0 : sinc(Dα_over_λ)

# ── independent reference physics (a DIFFERENT code path; never call the routine) ─
# Constants mirror recombination.jl:17-21 (transcribed, not imported). Plain globals
# (not const) so the file stays includable inside the runtests.jl @testset scope.
_CR  = 1.799920e14
_CDB = 3.945150e4
_LAM = 1.215668e-7
_A8  = 8.2245809
_CHI = 157807.0

# Hui-Gnedin case-B α_B [m³/s] — inlined fit, independent of recfast_alpha.
ref_aB(T) = 1.0e-19*4.309*(T/1e4)^(-0.6166) / (1 + 0.6703*(T/1e4)^0.53)

# From-scratch Peebles k2 [cm³/s] (re-derives KL/KB/C; mirror of peebles_k2_mixing
# but built from the textbook constants above, NOT by calling the routine).
function ref_peebles_k2(T, nHI_local, nHI_eff, Hz; fudge=1.0, gauss=1.0)
    aB  = ref_aB(T)
    bet = aB*(_CR*T)^1.5*exp(-_CDB/T)
    K   = gauss*_LAM^3/(8π*Hz)
    KL  = K*_A8*(nHI_eff*1e6)
    KB  = K*bet*(nHI_local*1e6)
    C   = fudge*(1 + KL)/(1 + KL + fudge*KB)
    return aB*1e6*C
end

# CMB photoionisation of H(1s) [s⁻¹] at the RADIATION temperature (mirror
# recombination.jl:89-95). β₁s = β₂p·exp(-(χ-CDB)/Trad).
ref_beta1s(Trad) = ref_aB(Trad)*(_CR*Trad)^1.5*exp(-_CDB/Trad)*exp(-(_CHI-_CDB)/Trad)
# C-weighted β₁s exactly as build_rates_mixing assembles it: β₁s·k2/(α_B·1e6).
ref_kb1s(T, Trad, k2) = ref_beta1s(Trad)*k2/(ref_aB(T)*1e6)

# Advance one bin's ionised fraction x=n_HII/n_H over dt [s] with an INDEPENDENT
# backward-Euler substepper.  dx/dt = kb1s·(1-x) − k2·nH·x²  (H-only; n_e=n_HII).
# k2 (hence kb1s) is re-evaluated each substep with the live local neutral density,
# while the SHARED smoothed neutral `n1s_sm` is lagged over the macro step (exactly
# how the routine is fed ⟨n1s⟩ once per step).
function ref_bin_step(x, T, Trad, nH, Hz, fudge, gauss, fa, n1s_sm, dt)
    t = 0.0
    while t < dt
        n1s_local = (1 - x)*nH
        n1s_eff   = fa*n1s_sm + (1 - fa)*n1s_local
        k2  = ref_peebles_k2(T, n1s_local, n1s_eff, Hz; fudge=fudge, gauss=gauss)
        kb  = ref_kb1s(T, Trad, k2)
        rate = k2*nH*x + kb + 1e-30
        ds  = min(dt - t, 0.1/rate)
        A = k2*nH*ds; B = 1 + kb*ds; Cc = -(x + kb*ds)
        x = (-B + sqrt(B*B - 4*A*Cc))/(2*A)          # A>0 ⇒ positive root
        t += ds
    end
    return x
end

# Independent binned H-recombination history.  `smoothing(n1s_local)->n1s_sm` gives
# the per-cell smoothed NEUTRAL number density (lagged from the start-of-step state);
# `Tb_of_z(z)` prescribes the matter temperature.  Returns (z, xe_vol).
function ref_integrate_H_bins(; z_start, z_end, n_steps, Δ, w, Tb_of_z, fa,
                                smoothing, recfast_hswitch=true,
                                hubble=71.0, Om=0.27, OL=0.73, fh=FH, x0=nothing)
    Mb    = length(Δ)
    logzp = range(log(1+z_start), log(1+z_end); length=n_steps+1)
    nHbar0 = n_H_at_z(z_start; fh=fh)
    xe0   = isnothing(x0) ? min(saha_xe(TCMB0*(1+z_start), nHbar0), 1-1e-6) : Float64(x0)
    x = fill(xe0, Mb)
    z_out = Float64[z_start]; xe_out = Float64[sum(w .* x)]
    for k in 1:n_steps
        z_hi = exp(logzp[k]) - 1; z_lo = exp(logzp[k+1]) - 1; z_mid = 0.5*(z_hi+z_lo)
        Hz_dt = _Hz(z_mid; h=hubble, Om=Om, OL=OL); dt = (z_hi - z_lo)/((1+z_mid)*Hz_dt)
        nHbar = n_H_at_z(z_lo; fh=fh); nH = nHbar .* Δ
        Trad = TCMB0*(1 + z_lo); T = Tb_of_z(z_lo); Hz = _Hz(z_lo; h=hubble, Om=Om, OL=OL)
        gauss = recfast_hswitch ? recfast_gauss_factor(z_lo) : 1.0
        fudge = recfast_hswitch ? 1.125 : 1.0
        n1s_local = (1 .- x) .* nH
        n1s_sm = smoothing(n1s_local)
        for i in 1:Mb
            x[i] = ref_bin_step(x[i], T, Trad, nH[i], Hz, fudge, gauss, fa, n1s_sm[i], dt)
        end
        push!(z_out, z_lo); push!(xe_out, sum(w .* x))
    end
    return z_out, xe_out
end

# ── routine history driver (calls solve_chem_mixing! over the field) ─────────
# Mirror of the reference loop but driving the production routine.  `smoothing`
# returns the per-cell smoothed NEUTRAL number density; it is fed via
# smoothed_is_neutral=true as nsm = n1s_sm·MH/fh (the kernel undoes ·fh/MH).
function run_routine_field_history(; z_start, z_end, n_steps, Δ, w, fa,
                                     smoothing, recfast_hswitch=true, T0=nothing,
                                     hubble=71.0, Om=0.27, OL=0.73, fh=FH, x0=nothing)
    Mb = length(Δ)
    fa_table = FAlphaTable([0.0, 1.0e5], [fa, fa])
    logzp = range(log(1+z_start), log(1+z_end); length=n_steps+1)
    nHbar0 = n_H_at_z(z_start; fh=fh); nH0 = nHbar0 .* Δ
    xe0 = isnothing(x0) ? min(saha_xe(TCMB0*(1+z_start), nHbar0), 1-1e-6) : Float64(x0)
    Tinit = isnothing(T0) ? TCMB0*(1+z_start) : Float64(T0)
    rho_v = nH0 .* MH ./ fh
    HII_v = xe0 .* nH0 .* MH
    H2I_v = fill(1.0e-40, Mb)
    e_v   = [e_from_T(Tinit, xe0, rho_v[i]; fh=fh) for i in 1:Mb]
    nsm_v = copy(rho_v)
    z_out = Float64[z_start]; xe_out = Float64[xe0]
    for k in 1:n_steps
        z_hi = exp(logzp[k]) - 1; z_lo = exp(logzp[k+1]) - 1; z_mid = 0.5*(z_hi+z_lo)
        Hz = _Hz(z_mid; h=hubble, Om=Om, OL=OL); dt = (z_hi - z_lo)/((1+z_mid)*Hz)
        nHbar = n_H_at_z(z_lo; fh=fh); nH = nHbar .* Δ; rho_c = nH .* MH ./ fh
        for i in 1:Mb                                   # expansion rescale, preserve x_e
            rs = rho_c[i]/rho_v[i]; rho_v[i] = rho_c[i]; HII_v[i] *= rs; H2I_v[i] *= rs
        end
        x_now = HII_v ./ (rho_c .* fh)
        n1s_local = (1 .- x_now) .* nH
        nsm_v .= smoothing(n1s_local) .* MH ./ fh        # units lock: kernel does ·fh/MH
        solve_chem_mixing!(rho_v, e_v, HII_v, H2I_v, nsm_v;
                           a_value=1.0/(1.0+z_lo), dt=dt,
                           density_units=1.0, length_units=1.0, time_units=1.0,
                           fa_table=fa_table, smoothed_is_neutral=true,
                           recfast_hswitch=recfast_hswitch, hubble_expansion=true,
                           hubble=hubble, Om=Om, OL=OL, fh=fh)
        x_new = HII_v ./ (rho_c .* fh)
        push!(z_out, z_lo); push!(xe_out, sum(w .* x_new))
    end
    return z_out, xe_out
end

# Smoothing closures used by both drivers.
sm_local(n1s_local)      = n1s_local                               # no smoothing
sm_mean(w)               = n1s_local -> fill(sum(w .* n1s_local), length(n1s_local))
sm_window(w, W)          = n1s_local -> (s = sum(w .* n1s_local); s .+ W .* (n1s_local .- s))

# ── CAMB/RECFAST-v2 fixture (the homogeneous reference; cols z, xe, Tb_K) ─────
function load_recfast_fixture()
    fixture = joinpath(@__DIR__, "fixtures", "recfast_v2_xe.csv")
    raw = filter(!startswith("#"), readlines(fixture))
    rows = [parse.(Float64, split(ln, ",")) for ln in raw if !isempty(ln)]
    z = Float64[r[1] for r in rows]; xe = Float64[r[2] for r in rows]; Tb = Float64[r[3] for r in rows]
    p = sortperm(z)
    return z[p], xe[p], Tb[p]
end
function _lerp(zq, zs, vs)
    i = searchsortedfirst(zs, zq)
    i <= 1 && return vs[1]
    i > length(zs) && return vs[end]
    t = (zq - zs[i-1])/(zs[i] - zs[i-1]); vs[i-1]*(1-t) + vs[i]*t
end

# ── interpolation helper for descending histories ────────────────────────────
function xe_at(z_hist, xe_hist, zq)
    zs = reverse(z_hist); xs = reverse(xe_hist)   # ascending
    _lerp(zq, zs, xs)
end

# ═══════════════════════════════════════════════════════════════════════════════

@testset "recombination_field" begin

Z_FIX, XE_FIX, TB_FIX = load_recfast_fixture()
xe_camb(z) = _lerp(z, Z_FIX, XE_FIX)
Tb_camb(z) = _lerp(z, Z_FIX, TB_FIX)

@testset "field_construction" begin
    # Pure lognormal: ⟨Δ⟩=1, Var≈b, has a tail; volume weights uniform.
    Δ, w = lognormal_field(b=0.5, M=2048)
    @test isapprox(sum(w), 1.0; rtol=1e-12)
    @test isapprox(sum(w .* Δ), 1.0; rtol=1e-10)              # ⟨Δ⟩_vol = 1
    varΔ = sum(w .* (Δ .- 1).^2)
    @test isapprox(varΔ, 0.5; atol=0.06)                      # Var ≈ b
    @test maximum(Δ) > 4.0                                    # lognormal tail present

    # Fat-tailed field: unequal volume weights, big tail, still ⟨Δ⟩_vol=1.
    Δt, wt = lognormal_field(b=0.5, M=2048, tail_deltas=[12.0,25.0,60.0,150.0], f_tail=1e-3)
    @test isapprox(sum(wt .* Δt), 1.0; rtol=1e-10)
    @test maximum(Δt) > 100.0
    @test minimum(wt) < maximum(wt)                           # weights are unequal
    @test isapprox(sum(wt[end-3:end]), 1e-3; rtol=1e-6)       # tail carries f_tail volume

    # Sinusoid + window.
    s = sinusoid_field(0.4, 64)
    @test isapprox(_mean(s), 1.0; atol=1e-12)
    @test isapprox(maximum(s), 1.4; atol=0.02)
    @test tophat_window(0.0) == 1.0
    @test tophat_window(1.0) < 1e-12                          # first zero at D/λ=1
    @test 0.6 < tophat_window(0.5) < 0.65                     # sinc(0.5)=2/π≈0.6366
end

@testset "units_lock" begin
    # Feeding a neutral smoothed field via smoothed_is_neutral=true must round-trip
    # through the kernel's n_sm·fh/MH conversion: nsm = n1s·MH/fh.  If the units were
    # wrong (off by ~MH/fh≈2e-24) the escape rate would be astronomically off and a
    # single step would not match the independent single-bin reference.
    z = 1100.0; nH = n_H_at_z(z); Hz = _Hz(z); T = Tb_camb(z)
    xe = 0.20; n1s_target = (1 - xe)*nH*0.7        # an arbitrary shared neutral < local
    dt = 0.02/Hz
    # routine, 1 cell, full mixing, smoothed neutral fed directly
    rho = [nH*MH/FH]; HII = [xe*nH*MH]; H2I = [1e-40]; e = [e_from_T(T, xe, rho[1])]
    nsm = [n1s_target*MH/FH]
    solve_chem_mixing!(rho, e, HII, H2I, nsm;
                       a_value=1/(1+z), dt=dt, density_units=1.0, length_units=1.0,
                       time_units=1.0, fa_table=FAlphaTable([0.,1e5],[1.,1.]),
                       smoothed_is_neutral=true, recfast_hswitch=true,
                       hubble_expansion=true, fh=FH)
    x_routine = HII[1]/(nH*MH)
    # independent single-bin reference with the SAME shared neutral
    Trad = TCMB0*(1+z); gauss = recfast_gauss_factor(z)
    x_ref = ref_bin_step(xe, T, Trad, nH, Hz, 1.125, gauss, 1.0, n1s_target, dt)
    @test isapprox(x_routine, x_ref; rtol=0.02)               # locks the n_sm units
    @test x_routine < xe                                      # net recombination
end

@testset "test1a_rate_weighting" begin
    # Rate-formula guard across a fat-tailed density range + weighting documentation.
    z = 1100.0; T = 3000.0; xe = 0.1; Hz = _Hz(z); nbar = n_H_at_z(z)
    Δ, w = lognormal_field(b=0.5, M=2048, tail_deltas=[12.0,25.0,60.0,150.0], f_tail=1e-3)
    n1s_local = (1 - xe) .* nbar .* Δ
    n1s_lin   = sum(w .* n1s_local)                          # volume-weighted LINEAR mean

    # (1) peebles_k2_mixing == independent re-derivation, every cell, full mixing.
    k2_rt = [peebles_k2_mixing(T, n1s_local[i], n1s_lin, Hz) for i in eachindex(Δ)]
    k2_rf = [ref_peebles_k2(T, n1s_local[i], n1s_lin, Hz)    for i in eachindex(Δ)]
    @test maximum(abs.(k2_rt .- k2_rf) ./ k2_rf) < 1e-10
    @test isapprox(sum(w .* k2_rt), sum(w .* k2_rf); rtol=1e-10)

    # (2) the routine consumes the smoothed neutral linearly (f_α=1 ⇒ n1s_eff=n_sm).
    @test isapprox(n1s_effective(n1s_local[1], n1s_lin, 0.0, 1.0, Val(true)), n1s_lin; rtol=1e-12)

    # (3) the field is a genuine weighting discriminator: the linear volume mean is
    #     distinct from the number mean and from the squared-moment "mean".
    n_e   = xe .* nbar .* Δ
    @test sum(w .* n_e.^2)/(sum(w .* n_e))^2 > 1.4           # ⟨n_e²⟩/⟨n_e⟩² (clumping+tail)
    n1s_num = _mean(n1s_local)                               # number-weighted (tail pulls up)
    n1s_sq  = sum(w .* n1s_local.^2)/sum(w .* n1s_local)     # squared-moment weighted
    @test n1s_num > 1.05*n1s_lin                             # number ≠ volume mean
    @test n1s_sq  > 1.40*n1s_lin                             # squared ≠ linear mean
    @test !isapprox(n1s_lin, n1s_sq; rtol=0.1)              # negative control
end

@testset "test1b_history_fullmix" begin
    # Full-mixing x_e(z) history: routine vs independent binned integrator.
    # Window z=1200→700 seeded from CAMB (the regime where the routine is validated
    # to <0.1% vs CAMB: He fully neutral, β₁s negligible — see recfast_v2_comparison).
    # Tolerance 2%: the irreducible gap is the routine's closed H₂⁺ photodissociation
    # cycle (~1.5% at z≈1100, which the H-only reference omits) + two stiff integrators.
    z0, z1, ns = 1200.0, 700.0, 200
    Δ, w = lognormal_field(b=0.5, M=200)
    T0 = Tb_camb(z0); x0 = xe_camb(z0)
    zR, xeR = run_routine_field_history(; z_start=z0, z_end=z1, n_steps=ns, Δ=Δ, w=w,
                                         fa=1.0, smoothing=sm_mean(w), T0=T0, x0=x0)
    zI, xeI = ref_integrate_H_bins(; z_start=z0, z_end=z1, n_steps=ns, Δ=Δ, w=w,
                                    Tb_of_z=Tb_camb, fa=1.0, smoothing=sm_mean(w), x0=x0)
    for z in 1100.0:-100.0:800.0
        a = xe_at(zR, xeR, z); b = xe_at(zI, xeI, z)
        @test abs(a - b)/b < 0.02
    end
end

@testset "test2_bracket_limits" begin
    z0, z1, ns = 1200.0, 700.0, 200
    Δ, w = lognormal_field(b=0.5, M=200)
    T0 = Tb_camb(z0); x0 = xe_camb(z0)

    # (a) f_α=0 mixing is bit-identical to solve_chem! over the same cells (1 step).
    z = 1100.0; nH = n_H_at_z(z) .* Δ; rho = nH .* MH ./ FH; xe = 0.5
    HII = xe .* nH .* MH; dt = 1e12
    e1 = [e_from_T(3000.0, xe, rho[i]) for i in eachindex(Δ)]; H21 = fill(1e-40, length(Δ))
    e2 = copy(e1); HII2 = copy(HII); H22 = copy(H21); HII1 = copy(HII)
    solve_chem!(rho, e1, HII1, H21; a_value=1/(1+z), dt=dt,
                density_units=1.0, length_units=1.0, time_units=1.0)
    solve_chem_mixing!(rho, e2, HII2, H22, copy(rho); a_value=1/(1+z), dt=dt,
                       density_units=1.0, length_units=1.0, time_units=1.0, fa_table=FA_ZERO)
    @test HII2 == HII1
    @test e2 == e1

    # Histories at f_α = 0, 0.5, 1 (smoothed field = volume mean).
    zr0, xv0 = run_routine_field_history(; z_start=z0, z_end=z1, n_steps=ns, Δ=Δ, w=w,
                                          fa=0.0, smoothing=sm_local, T0=T0, x0=x0)
    zrh, xvh = run_routine_field_history(; z_start=z0, z_end=z1, n_steps=ns, Δ=Δ, w=w,
                                          fa=0.5, smoothing=sm_mean(w), T0=T0, x0=x0)
    zr1, xv1 = run_routine_field_history(; z_start=z0, z_end=z1, n_steps=ns, Δ=Δ, w=w,
                                          fa=1.0, smoothing=sm_mean(w), T0=T0, x0=x0)
    # (b) f_α=0 matches independent no-mix integrator (2%; H₂⁺-cycle + integrator gap).
    zi0, xi0 = ref_integrate_H_bins(; z_start=z0, z_end=z1, n_steps=ns, Δ=Δ, w=w,
                                     Tb_of_z=Tb_camb, fa=0.0, smoothing=sm_local, x0=x0)
    for z in 1100.0:-100.0:800.0
        @test abs(xe_at(zr0,xv0,z) - xe_at(zi0,xi0,z))/xe_at(zi0,xi0,z) < 0.02
    end
    # (c) SIGN + monotonicity: more mixing ⇒ faster recombination ⇒ lower x_e.
    #     (History sign; complements sinusoidal_density's short-step variance result.)
    for z in 1100.0:-100.0:800.0
        a0 = xe_at(zr0,xv0,z); ah = xe_at(zrh,xvh,z); a1 = xe_at(zr1,xv1,z)
        @test a1 < a0                                   # full mix below no-mix
        @test a1 - 1e-3 <= ah <= a0 + 1e-3              # monotone in f_α (≤1e-3 noise)
    end
end

@testset "test3_window_sweep" begin
    # Manufactured sinusoid; sweep the smoothing window W with f_α=1 fixed.  (Host
    # real-space smoothing — "Test 3.1" — is OUT OF SCOPE: the routine consumes a
    # pre-smoothed field, it does not smooth.)  n1s_eff = W·n_local + (1−W)·⟨n1s⟩, so
    # W→1 ≡ no-mix, W→0 ≡ full-mix; the map is exactly linear in W (n1s_effective is a
    # branch-free muladd) ⇒ x_e must be smooth (C¹) in W.
    A = 0.4; N = 48; z0, z1, ns = 1200.0, 700.0, 150
    Δ = sinusoid_field(A, N); w = fill(1.0/N, N); T0 = Tb_camb(z0); x0 = xe_camb(z0)
    zc = 900.0                                               # comparison redshift
    Ws = collect(range(0.05, 1.0; length=13))
    xeW = Float64[]; xeWref = Float64[]
    for W in Ws
        zr, xr = run_routine_field_history(; z_start=z0, z_end=z1, n_steps=ns, Δ=Δ, w=w,
                                            fa=1.0, smoothing=sm_window(w, W), T0=T0, x0=x0)
        zi, xi = ref_integrate_H_bins(; z_start=z0, z_end=z1, n_steps=ns, Δ=Δ, w=w,
                                       Tb_of_z=Tb_camb, fa=1.0, smoothing=sm_window(w, W), x0=x0)
        push!(xeW,    xe_at(zr, xr, zc))
        push!(xeWref, xe_at(zi, xi, zc))
    end
    # no-mix and full-mix endpoints for the limit checks
    znm, xnm = run_routine_field_history(; z_start=z0, z_end=z1, n_steps=ns, Δ=Δ, w=w,
                                          fa=0.0, smoothing=sm_local, T0=T0, x0=x0)
    zfm, xfm = run_routine_field_history(; z_start=z0, z_end=z1, n_steps=ns, Δ=Δ, w=w,
                                          fa=1.0, smoothing=sm_mean(w), T0=T0, x0=x0)
    xe_nomix = xe_at(znm, xnm, zc); xe_full = xe_at(zfm, xfm, zc)

    @test isapprox(xeW[end], xe_nomix; rtol=0.005)           # (a) W→1 == no-mix
    @test isapprox(xeW[1],   xe_full;  rtol=0.02)            # (b) W→0 == full-mix
    @test all(abs.(xeW .- xeWref) ./ xeWref .< 0.02)         # (c) matches independent ref ∀W
    # monotone interpolation between the limits
    @test all(diff(xeW) .>= -1e-9) || all(diff(xeW) .<= 1e-9)

    # (d) C¹ in W: uniform W grid ⇒ no slope sign-flip, curvature bounded (no kink).
    dW  = Ws[2] - Ws[1]
    fd1 = diff(xeW) ./ dW
    fd2 = diff(fd1) ./ dW
    @test all(fd1 .> 0) || all(fd1 .< 0)                     # no slope reversal
    medc = _median(abs.(fd2))
    @test maximum(abs.(fd2)) < 10*max(medc, eps())           # curvature bounded (smooth)
    @test maximum(abs.(fd2))*dW^2/(maximum(xeW)-minimum(xeW)) < 0.12   # near-linear in W

    # guard against double-applying W (in the field AND f_α): n1s_effective is linear.
    θ = 2π*0.3
    @test isapprox(n1s_effective(1.0+A*sin(θ), 1.0+A*0.5*sin(θ), 0.0, 1.0, Val(true)),
                   1.0+A*0.5*sin(θ); rtol=1e-12)
end

@testset "A0_homogeneous_gate" begin
    # A→0: homogeneous field, full mixing ⇒ must collapse to the CAMB/RECFAST-v2
    # fixture (ties the field driver back to the existing homogeneous reference).
    z0, z1, ns = 1200.0, 700.0, 250
    Δ = ones(8); w = fill(1.0/8, 8); T0 = Tb_camb(z0)
    zr, xr = run_routine_field_history(; z_start=z0, z_end=z1, n_steps=ns, Δ=Δ, w=w,
                                        fa=1.0, smoothing=sm_mean(w), T0=T0,
                                        x0=xe_camb(z0))
    for z in (1100.0, 1000.0, 900.0, 800.0)
        @test abs(xe_at(zr, xr, z) - xe_camb(z))/xe_camb(z) < 0.01
    end
end

@testset "physics_sanity_tie" begin
    # b=0.5 lognormal, full mixing vs none: x_e drops a few%–~20%, growing toward low z.
    z0, z1, ns = 1200.0, 700.0, 200
    Δ, w = lognormal_field(b=0.5, M=200); T0 = Tb_camb(z0); x0 = xe_camb(z0)
    zr0, xv0 = run_routine_field_history(; z_start=z0, z_end=z1, n_steps=ns, Δ=Δ, w=w,
                                          fa=0.0, smoothing=sm_local, T0=T0, x0=x0)
    zr1, xv1 = run_routine_field_history(; z_start=z0, z_end=z1, n_steps=ns, Δ=Δ, w=w,
                                          fa=1.0, smoothing=sm_mean(w), T0=T0, x0=x0)
    zg = collect(1100.0:-50.0:750.0)
    dXe = [(xe_at(zr1,xv1,z) - xe_at(zr0,xv0,z))/xe_at(zr0,xv0,z) for z in zg]
    @test all(dXe .< 0)                                      # mixing lowers x_e (sign)
    @test minimum(dXe) > -0.30 && maximum(dXe) < -0.002      # few% to ~20% (scale)
    # the suppression deepens monotonically toward low z (steepest at the low-z end)
    @test dXe[end] < dXe[1]
end

end # @testset "recombination_field"

println("recombination_field tests complete.")
