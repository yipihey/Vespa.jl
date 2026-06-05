# ── PPML: Piecewise-Parabolic Method on a Local stencil (Ustyugov+ 2009) ──────
# Per-cell reconstruction / characteristic-trace primitives, ported faithfully from
# the Rust reference (morton_code/src/hydro/ppml.rs). These are pure, allocation-free,
# precision-generic `@inline` functions on scalar primitives so they inline on both
# the CPU and the Metal GPU (no NTuple-of-NTuple, no per-cell arrays).
#
# Primitive state order matches the Rust: (ρ, u, v, w, p) where `u` is the velocity
# ALONG the sweep axis (rotated to local-x per axis), `p` is PRESSURE (not eint —
# the characteristic eigenvectors are written in terms of pressure). v,w are the two
# transverse velocities (passively advected by the contact).
#
# The defining PPML feature is the *characteristic-traced half-step predictor*
# (`_ppml_face_left/right`): each cell carries a parabola anchored on a stored face
# pair (qL, ⟨q⟩, qR); the face state at t+dt/2 is the parabola integrated over each
# incoming wave's departure region, projected onto the Euler eigenvectors. The face
# pair itself is limited (RGK characteristic limiter + CW84 monotonize + CW84 shock
# flattening) before the trace and re-derived from the Riemann star states after the
# update — the stateful "Local" bookkeeping lives in ppml_grid.jl.

# ── parabola q(ξ)=a+bξ+cξ², ξ∈[0,1], with q(0)=qL, q(1)=qR, ∫₀¹=qavg ───────────
@inline _ppml_pcoef(qL::T, qa::T, qR::T) where {T} =
    (qL, -T(4) * qL + T(6) * qa - T(2) * qR, T(3) * (qL - T(2) * qa + qR))   # (a,b,c)

# (1/(hi−lo))·∫_lo^hi q(ξ)dξ = a + b·(lo+hi)/2 + c·(lo²+lo·hi+hi²)/3
@inline _ppml_pavg(a::T, b::T, c::T, lo::T, hi::T) where {T} =
    a + b * T(0.5) * (lo + hi) + c * (lo * lo + lo * hi + hi * hi) / T(3)

# ── CW84 (1984 §1.6 eq 1.10) monotonicity limiter on one (qL,⟨q⟩,qR) triple ────
@inline function _ppml_monotonize(qL::T, qa::T, qR::T) where {T}
    dq = qR - qL; qmid = T(0.5) * (qL + qR); dq2_6 = dq * dq / T(6)
    diff = (qa - qmid) * dq
    if (qR - qa) * (qa - qL) <= zero(T)
        return (qa, qa)                       # local extremum ⇒ flatten to constant
    elseif diff > dq2_6
        return (T(3) * qa - T(2) * qR, qR)    # overshoot on the right
    elseif diff < -dq2_6
        return (qL, T(3) * qa - T(2) * qL)    # overshoot on the left
    end
    return (qL, qR)
end

# ── CW84 (1984 §A) shock-flatten coefficient ω∈[0,1] from the 5-point stencil ──
# (p,u at i−2..i+2; `u` axial). ω=1 collapses to piecewise-constant. Smooth ramp on
# the relative pressure jump (η_lo=0.1, η_hi=1/3) per the Rust's high-Mach tuning.
@inline function _ppml_flatten_omega(pmm::T, umm::T, pm::T, um::T, p0::T, u0::T,
                                     pp::T, up::T, ppp::T, upp::T) where {T}
    dp3 = pp - pm; dp5 = ppp - pmm
    um <= up && return zero(T)                              # convergent-flow gate
    pmin = min(pm, pp)
    pmin <= zero(T) && return zero(T)
    s_i = abs(dp5) < T(1e-3) * pmin ? zero(T) : dp3 / dp5   # concentration gate
    s_i <= T(0.5) && return zero(T)
    η = abs(dp3) / pmin
    return clamp((η - T(0.1)) / (T(1) / T(3) - T(0.1)), zero(T), one(T))
end

# 3-point flatten ω (Rust ppml_flatten_omega_3pt): the NARROW-stencil flattener that
# drops the 5-point concentration test, reading only i−1,i,i+1 — so the whole
# reconstruction is 1-ghost-local (the only thing that made it wider was the 5-pt
# variant). Binary (more dissipative on smooth-but-steep gradients than the 5-pt ramp);
# the Rust uses it at refinement edges where the wide stencil isn't available.
@inline function _ppml_flatten_omega_3pt(pm::T, um::T, p0::T, u0::T, pp::T, up::T) where {T}
    um <= up && return zero(T)                              # convergent-flow gate
    pmin = min(pm, pp); dp3 = pp - pm
    (pmin <= zero(T) || abs(dp3) / pmin <= one(T) / T(3)) && return zero(T)
    return one(T)
end

# blend a face pair toward the cell average by ω (0=full PPM, 1=constant).
@inline _ppml_flat1(q::T, qa::T, f::T) where {T} = (one(T) - f) * q + f * qa

# ── WENO5 interface reconstruction (Jiang-Shu) + smooth-extremum fallback ──────
# The median/RGK limiter (+ CW84) clamps SMOOTH extrema to the cell average (1st-order
# there). Ustyugov+ 2009 §6 recovers 5th-order accuracy at smooth extrema by replacing
# the clamped face pair with a 5-point WENO5 reconstruction — the piece that makes this
# the FULL PPML, not a subset. `_ppml_weno5` returns the (left, right) interface values
# q_{i−1/2}, q_{i+1/2} from the 5-cell stencil (qmm,qm,q0,qp,qpp = q_{i−2 … i+2}).
@inline function _ppml_weno5(qmm::T, qm::T, q0::T, qp::T, qpp::T) where {T}
    ε = T(1e-6); c13 = T(13)/T(12); c14 = T(0.25); s6 = one(T)/T(6)
    d0 = T(0.1); d1 = T(0.6); d2 = T(0.3)
    # right face q_{i+1/2} (left-biased): 3 candidate stencils + smoothness indicators
    vr0 = (T(2)*qmm - T(7)*qm + T(11)*q0)*s6
    vr1 = (-qm + T(5)*q0 + T(2)*qp)*s6
    vr2 = (T(2)*q0 + T(5)*qp - qpp)*s6
    β0 = c13*(qmm - T(2)*qm + q0)^2 + c14*(qmm - T(4)*qm + T(3)*q0)^2
    β1 = c13*(qm - T(2)*q0 + qp)^2 + c14*(qm - qp)^2
    β2 = c13*(q0 - T(2)*qp + qpp)^2 + c14*(T(3)*q0 - T(4)*qp + qpp)^2
    a0 = d0/(ε+β0)^2; a1 = d1/(ε+β1)^2; a2 = d2/(ε+β2)^2; sa = a0+a1+a2
    qR = (a0*vr0 + a1*vr1 + a2*vr2)/sa
    # left face q_{i−1/2} (right-biased): mirror stencils (β1 is shared/central)
    vl0 = (T(2)*qpp - T(7)*qp + T(11)*q0)*s6
    vl1 = (-qp + T(5)*q0 + T(2)*qm)*s6
    vl2 = (T(2)*q0 + T(5)*qm - qmm)*s6
    γ0 = c13*(qpp - T(2)*qp + q0)^2 + c14*(qpp - T(4)*qp + T(3)*q0)^2
    γ2 = c13*(q0 - T(2)*qm + qmm)^2 + c14*(T(3)*q0 - T(4)*qm + qmm)^2
    b0 = d0/(ε+γ0)^2; b1 = d1/(ε+β1)^2; b2 = d2/(ε+γ2)^2; sb = b0+b1+b2
    qL = (b0*vl0 + b1*vl1 + b2*vl2)/sb
    return (qL, qR)
end

# Smooth-extremum fallback for one primitive: if cell i is a LOCAL EXTREMUM of the cell
# averages (so the limiter clamped the face pair to the constant `q0`) AND it is SMOOTH
# (consistent second-difference sign across i−2 … i+2, the Colella-Sekora test), replace
# the clamped (qL,qR) with the WENO5 reconstruction. Else keep the limited values.
@inline function _ppml_extremum_fix(qL::T, qR::T, qmm::T, qm::T, q0::T, qp::T, qpp::T) where {T}
    if (qp - q0)*(q0 - qm) <= zero(T)                    # local extremum of the averages
        d2m = qmm - T(2)*qm + q0; d20 = qm - T(2)*q0 + qp; d2p = q0 - T(2)*qp + qpp
        if d2m*d20 > zero(T) && d20*d2p > zero(T)        # smooth ⇒ consistent curvature
            return _ppml_weno5(qmm, qm, q0, qp, qpp)
        end
    end
    return (qL, qR)
end

# positivity-guarded variant for ρ and p: a WENO5 face that goes non-positive (a smooth
# minimum near zero) reverts to the limited value.
@inline function _ppml_exfix_pos(qL::T, qR::T, qmm::T, qm::T, q0::T, qp::T, qpp::T) where {T}
    (a, b) = _ppml_extremum_fix(qL, qR, qmm, qm, q0, qp, qpp)
    return (a > zero(T) && b > zero(T)) ? (a, b) : (qL, qR)
end

# ── characteristic projections at a basis state (ρ,p,cs) along local-x ─────────
# δw=(δρ,δu,δv,δw,δp) → wave amplitudes δα such that δw = Σ δα_k r_k.
@inline function _ppml_prim2char(δρ::T, δu::T, δv::T, δw::T, δp::T,
                                 ρ::T, cs::T) where {T}
    h = T(0.5) * ρ / cs; hc = T(0.5) / (cs * cs)
    return (-h * δu + hc * δp,            # l₁ (left acoustic)
            δρ - δp / (cs * cs),          # l₂ (entropy)
            δv,                           # l₃ (vy shear)
            δw,                           # l₄ (vz shear)
            h * δu + hc * δp)             # l₅ (right acoustic)
end

# δα → δw  (inverse of _ppml_prim2char).
@inline function _ppml_char2prim(a1::T, a2::T, a3::T, a4::T, a5::T,
                                 ρ::T, cs::T) where {T}
    c2 = cs * cs
    return (a1 + a2 + a5,                 # δρ
            -cs / ρ * a1 + cs / ρ * a5,   # δu
            a3,                           # δv
            a4,                           # δw
            c2 * a1 + c2 * a5)            # δp
end

# minmod(a,b): 0 if opposite signs (or either 0), else signed-smaller-magnitude.
@inline _ppml_minmod(a::T, b::T) where {T} =
    a * b <= zero(T) ? zero(T) : (abs(a) < abs(b) ? a : b)

# ── Rider-Greenough-Kamm characteristic-variable limiter on one face pair ──────
# (w_minus, w_self, w_plus) cell averages; (wL,wR) the stored face pair. Returns the
# limited (wL', wR'). Stage 1: clamp each face delta to the neighbour gradient. Stage
# 2: enforce in-cell parabola monotonicity (δα**_L = minmod(δα*_L, −2 δα*_R)).
@inline function _ppml_rgk(ρm::T, um::T, vm::T, wm::T, pm::T,
                           ρs::T, us::T, vs::T, ws::T, ps::T,
                           ρp::T, up::T, vp::T, wp::T, pp::T,
                           ρL::T, uL::T, vL::T, wL::T, pL::T,
                           ρR::T, uR::T, vR::T, wR::T, pR::T, g::T) where {T}
    cs = sqrt(g * ps / ρs)
    nm = _ppml_prim2char(ρm - ρs, um - us, vm - vs, wm - ws, pm - ps, ρs, cs)
    np = _ppml_prim2char(ρp - ρs, up - us, vp - vs, wp - ws, pp - ps, ρs, cs)
    dl = _ppml_prim2char(ρL - ρs, uL - us, vL - vs, wL - ws, pL - ps, ρs, cs)
    dr = _ppml_prim2char(ρR - ρs, uR - us, vR - vs, wR - ws, pR - ps, ρs, cs)
    # stage 1 — slope-limit each face delta against its neighbour gradient
    l1 = _ppml_minmod(dl[1], nm[1]); l2 = _ppml_minmod(dl[2], nm[2]); l3 = _ppml_minmod(dl[3], nm[3])
    l4 = _ppml_minmod(dl[4], nm[4]); l5 = _ppml_minmod(dl[5], nm[5])
    r1 = _ppml_minmod(dr[1], np[1]); r2 = _ppml_minmod(dr[2], np[2]); r3 = _ppml_minmod(dr[3], np[3])
    r4 = _ppml_minmod(dr[4], np[4]); r5 = _ppml_minmod(dr[5], np[5])
    # stage 2 — in-cell parabola monotonicity
    ll1 = _ppml_minmod(l1, -T(2) * r1); ll2 = _ppml_minmod(l2, -T(2) * r2); ll3 = _ppml_minmod(l3, -T(2) * r3)
    ll4 = _ppml_minmod(l4, -T(2) * r4); ll5 = _ppml_minmod(l5, -T(2) * r5)
    rr1 = _ppml_minmod(r1, -T(2) * l1); rr2 = _ppml_minmod(r2, -T(2) * l2); rr3 = _ppml_minmod(r3, -T(2) * l3)
    rr4 = _ppml_minmod(r4, -T(2) * l4); rr5 = _ppml_minmod(r5, -T(2) * l5)
    dLp = _ppml_char2prim(ll1, ll2, ll3, ll4, ll5, ρs, cs)
    dRp = _ppml_char2prim(rr1, rr2, rr3, rr4, rr5, ρs, cs)
    return (ρs + dLp[1], us + dLp[2], vs + dLp[3], ws + dLp[4], ps + dLp[5],
            ρs + dRp[1], us + dRp[2], vs + dRp[3], ws + dRp[4], ps + dRp[5])
end

# ── CW84 monotonize + ρ/p positivity guard across all 5 primitives ────────────
@inline function _ppml_monotonize_all(ρL::T, uL::T, vL::T, wL::T, pL::T,
                                      ρa::T, ua::T, va::T, wa::T, pa::T,
                                      ρR::T, uR::T, vR::T, wR::T, pR::T) where {T}
    (ρl, ρr) = _ppml_monotonize(ρL, ρa, ρR)
    (ul, ur) = _ppml_monotonize(uL, ua, uR)
    (vl, vr) = _ppml_monotonize(vL, va, vR)
    (wl, wr) = _ppml_monotonize(wL, wa, wR)
    (pl, pr) = _ppml_monotonize(pL, pa, pR)
    (ρl <= zero(T) || ρr <= zero(T)) && (ρl = ρa; ρr = ρa)   # density positivity guard
    (pl <= zero(T) || pr <= zero(T)) && (pl = pa; pr = pa)   # pressure positivity guard
    return (ρl, ul, vl, wl, pl, ρr, ur, vr, wr, pr)
end

# ── characteristic-traced half-step face states ───────────────────────────────
# `right=true` ⇒ +axis face (incoming waves λ_k>0, departure [1−σ,1], geom=wR);
# `right=false` ⇒ −axis face (λ_k<0, departure [0,σ], geom=wL). Returns the traced
# (ρ,u,v,w,p); falls back to the cell average if ρ or p go non-positive.
@inline function _ppml_trace(ρL::T, uL::T, vL::T, wL::T, pL::T,
                             ρa::T, ua::T, va::T, wa::T, pa::T,
                             ρR::T, uR::T, vR::T, wR::T, pR::T,
                             dt_dx::T, g::T, right::Bool) where {T}
    cs = sqrt(g * pa / ρa)
    (aρ, bρ, cρ) = _ppml_pcoef(ρL, ρa, ρR); (au, bu, cu) = _ppml_pcoef(uL, ua, uR)
    (av, bv, cv) = _ppml_pcoef(vL, va, vR); (aw, bw, cw) = _ppml_pcoef(wL, wa, wR)
    (ap, bp, cp) = _ppml_pcoef(pL, pa, pR)
    ρg = right ? ρR : ρL; ug = right ? uR : uL; vg = right ? vR : vL
    wg = right ? wR : wL; pg = right ? pR : pL
    # per-distinct-eigenvalue departure region: λ ∈ {u−cs, u, u+cs}
    region(λ) = begin
        act = right ? (λ > zero(T)) : (λ < zero(T))
        σ = min(abs(λ) * dt_dx, one(T))
        (act, right ? (one(T) - σ) : zero(T), right ? one(T) : σ)
    end
    dρ = zero(T); du = zero(T); dv = zero(T); dw = zero(T); dp = zero(T)
    # left/right acoustic waves (k=1: u−cs, k=5: u+cs)
    (acm, lom, him) = region(ua - cs)
    if acm
        δu = _ppml_pavg(au, bu, cu, lom, him) - ug
        δp = _ppml_pavg(ap, bp, cp, lom, him) - pg
        α = -ρa / (T(2) * cs) * δu + δp / (T(2) * cs * cs)
        dρ += α; du += -cs / ρa * α; dp += cs * cs * α
    end
    (acp, lop, hip) = region(ua + cs)
    if acp
        δu = _ppml_pavg(au, bu, cu, lop, hip) - ug
        δp = _ppml_pavg(ap, bp, cp, lop, hip) - pg
        α = ρa / (T(2) * cs) * δu + δp / (T(2) * cs * cs)
        dρ += α; du += cs / ρa * α; dp += cs * cs * α
    end
    # entropy + transverse shears (k=2,3,4: λ=u, shared departure region)
    (ac0, lo0, hi0) = region(ua)
    if ac0
        δρ = _ppml_pavg(aρ, bρ, cρ, lo0, hi0) - ρg
        δp = _ppml_pavg(ap, bp, cp, lo0, hi0) - pg
        δv = _ppml_pavg(av, bv, cv, lo0, hi0) - vg
        δw = _ppml_pavg(aw, bw, cw, lo0, hi0) - wg
        dρ += δρ - δp / (cs * cs)          # entropy r₂ = (1,0,0,0,0)
        dv += δv                           # shear  r₃
        dw += δw                           # shear  r₄
    end
    ρf = ρg + dρ; pf = pg + dp
    (ρf <= zero(T) || pf <= zero(T)) && return (ρa, ua, va, wa, pa)   # fallback
    return (ρf, ug + du, vg + dv, wg + dw, pf)
end

@inline _ppml_face_right(ρL, uL, vL, wL, pL, ρa, ua, va, wa, pa, ρR, uR, vR, wR, pR, dt_dx, g) =
    _ppml_trace(ρL, uL, vL, wL, pL, ρa, ua, va, wa, pa, ρR, uR, vR, wR, pR, dt_dx, g, true)
@inline _ppml_face_left(ρL, uL, vL, wL, pL, ρa, ua, va, wa, pa, ρR, uR, vR, wR, pR, dt_dx, g) =
    _ppml_trace(ρL, uL, vL, wL, pL, ρa, ua, va, wa, pa, ρR, uR, vR, wR, pR, dt_dx, g, false)

# ── HLL flux + star state from pressure-based L/R primitive face states ────────
# Inputs are (ρ,u,v,w,p) — u the normal velocity. Returns the 6 conservative fluxes
# (Fρ,FS1,FS2,FS3,FE,FG; FG = gas-energy advective flux ρe·u, e=p/((γ−1)ρ)) PLUS the
# single HLL intermediate (star) state converted to primitives (ρ*,u*,v*,w*,p*) —
# the seed the PPML corrector RGK-reshapes into the next step's face pair. Same wave
# speeds + algebra as the certified `_hll6`, just pressure-in/star-out.
@inline function _ppml_hll(ρl::T, ul::T, vl::T, wl::T, pl::T,
                           ρr::T, ur::T, vr::T, wr::T, pr::T, g::T, gm1::T) where {T}
    h = T(0.5)
    v2l = ul*ul + vl*vl + wl*wl; csl = sqrt(g*pl/ρl); el = pl/(gm1*ρl); etl = el + h*v2l
    UlD = ρl; UlS1 = ρl*ul; UlS2 = ρl*vl; UlS3 = ρl*wl; UlE = ρl*etl; UlG = ρl*el
    FlD = ρl*ul; FlS1 = UlS1*ul + pl; FlS2 = UlS2*ul; FlS3 = UlS3*ul; FlE = (UlE + pl)*ul; FlG = UlG*ul
    lpl = ul + csl; lml = ul - csl
    v2r = ur*ur + vr*vr + wr*wr; csr = sqrt(g*pr/ρr); er = pr/(gm1*ρr); etr = er + h*v2r
    UrD = ρr; UrS1 = ρr*ur; UrS2 = ρr*vr; UrS3 = ρr*wr; UrE = ρr*etr; UrG = ρr*er
    FrD = ρr*ur; FrS1 = UrS1*ur + pr; FrS2 = UrS2*ur; FrS3 = UrS3*ur; FrE = (UrE + pr)*ur; FrG = UrG*ur
    lpr = ur + csr; lmr = ur - csr
    ap = max(zero(T), max(lpl, lpr)); am = max(zero(T), max(-lml, -lmr))
    s = ap + am; si = s > zero(T) ? one(T) / s : zero(T)
    F1 = (ap*FlD  + am*FrD  - ap*am*(UrD  - UlD))  * si
    F2 = (ap*FlS1 + am*FrS1 - ap*am*(UrS1 - UlS1)) * si
    F3 = (ap*FlS2 + am*FrS2 - ap*am*(UrS2 - UlS2)) * si
    F4 = (ap*FlS3 + am*FrS3 - ap*am*(UrS3 - UlS3)) * si
    F5 = (ap*FlE  + am*FrE  - ap*am*(UrE  - UlE))  * si
    F6 = (ap*FlG  + am*FrG  - ap*am*(UrG  - UlG))  * si
    # HLL intermediate (star) conserved state U* = (ap·Ur + am·Ul − (Fr−Fl))/(ap+am)
    UsD  = (ap*UrD  + am*UlD  - (FrD  - FlD))  * si
    UsS1 = (ap*UrS1 + am*UlS1 - (FrS1 - FlS1)) * si
    UsS2 = (ap*UrS2 + am*UlS2 - (FrS2 - FlS2)) * si
    UsS3 = (ap*UrS3 + am*UlS3 - (FrS3 - FlS3)) * si
    UsE  = (ap*UrE  + am*UlE  - (FrE  - FlE))  * si
    ρs = UsD; us = UsS1/UsD; vs = UsS2/UsD; ws = UsS3/UsD
    ps = gm1 * (UsE - h*ρs*(us*us + vs*vs + ws*ws))
    return (F1, F2, F3, F4, F5, F6, ρs, us, vs, ws, ps)
end

# ── HLLC flux + contact face state (Toro, Davis wave speeds) ───────────────────
# Contact-resolving Riemann solver: the same (ρ,u,v,w,p)-in / (6 fluxes + contact
# face state)-out interface as `_ppml_hll`, but it resolves the entropy/contact wave
# (sharper contacts ⇒ much less dissipation than HLL). The "star" returned is the
# resolved state at x/t=0 (the single agreed-on interface state — the PPML corrector
# seed). Falls back to HLL if a star state would be non-positive (Rust convention).
@inline function _ppml_hllc(ρl::T, ul::T, vl::T, wl::T, pl::T,
                            ρr::T, ur::T, vr::T, wr::T, pr::T, g::T, gm1::T) where {T}
    h = T(0.5)
    csl = sqrt(g*pl/ρl); csr = sqrt(g*pr/ρr)
    el = pl/(gm1*ρl); er = pr/(gm1*ρr)
    El = ρl*(el + h*(ul*ul + vl*vl + wl*wl)); Er = ρr*(er + h*(ur*ur + vr*vr + wr*wr))
    # L/R conserved (D,S1,S2,S3,E,G=ρe) + physical fluxes
    UlD=ρl; UlS1=ρl*ul; UlS2=ρl*vl; UlS3=ρl*wl; UlE=El; UlG=ρl*el
    FlD=ρl*ul; FlS1=UlS1*ul+pl; FlS2=UlS2*ul; FlS3=UlS3*ul; FlE=(El+pl)*ul; FlG=UlG*ul
    UrD=ρr; UrS1=ρr*ur; UrS2=ρr*vr; UrS3=ρr*wr; UrE=Er; UrG=ρr*er
    FrD=ρr*ur; FrS1=UrS1*ur+pr; FrS2=UrS2*ur; FrS3=UrS3*ur; FrE=(Er+pr)*ur; FrG=UrG*ur
    # Davis wave-speed estimates + the contact (star) speed
    SL = min(ul - csl, ur - csr); SR = max(ul + csl, ur + csr)
    den = ρl*(SL - ul) - ρr*(SR - ur)
    Sstar = (pr - pl + ρl*ul*(SL - ul) - ρr*ur*(SR - ur)) / den
    if SL >= zero(T)                                     # fully right-going ⇒ all-L
        return (FlD, FlS1, FlS2, FlS3, FlE, FlG, ρl, ul, vl, wl, pl)
    elseif SR <= zero(T)                                 # fully left-going ⇒ all-R
        return (FrD, FrS1, FrS2, FrS3, FrE, FrG, ρr, ur, vr, wr, pr)
    end
    pstar = pl + ρl*(SL - ul)*(Sstar - ul)               # = pr + ρr(SR−ur)(S*−ur)
    if pstar <= zero(T)                                  # star unphysical ⇒ HLL fallback
        return _ppml_hll(ρl, ul, vl, wl, pl, ρr, ur, vr, wr, pr, g, gm1)
    end
    if Sstar >= zero(T)                                  # contact on the right ⇒ left star
        ρs = ρl*(SL - ul)/(SL - Sstar)
        UsD=ρs; UsS1=ρs*Sstar; UsS2=ρs*vl; UsS3=ρs*wl
        UsE = ρs*(El/ρl + (Sstar - ul)*(Sstar + pl/(ρl*(SL - ul)))); UsG = ρs*el
        return (FlD + SL*(UsD-UlD), FlS1 + SL*(UsS1-UlS1), FlS2 + SL*(UsS2-UlS2),
                FlS3 + SL*(UsS3-UlS3), FlE + SL*(UsE-UlE), FlG + SL*(UsG-UlG),
                ρs, Sstar, vl, wl, pstar)
    else                                                 # contact on the left ⇒ right star
        ρs = ρr*(SR - ur)/(SR - Sstar)
        UsD=ρs; UsS1=ρs*Sstar; UsS2=ρs*vr; UsS3=ρs*wr
        UsE = ρs*(Er/ρr + (Sstar - ur)*(Sstar + pr/(ρr*(SR - ur)))); UsG = ρs*er
        return (FrD + SR*(UsD-UrD), FrS1 + SR*(UsS1-UrS1), FrS2 + SR*(UsS2-UrS2),
                FrS3 + SR*(UsS3-UrS3), FrE + SR*(UsE-UrE), FrG + SR*(UsG-UrG),
                ρs, Sstar, vr, wr, pstar)
    end
end
