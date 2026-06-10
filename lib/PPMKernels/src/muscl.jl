# ── MUSCL: PLM reconstruction + HLL Riemann flux (Enzo HydroMethod=3, HD_RK) ──
# KA port of Enzo's hydro_rk default pure-hydro solver: PLM (minmod-θ limiter)
# reconstruction to interface L/R states + the HLL Riemann flux, per interface.
# Faithful to Rec_PLM.C (`plm`) + Riemann_HLL.C (`hll`); certified bit-tight
# against the live solver via `EnzoLib.hydro_rk_line` (enzomodules_hydro_rk_line).
#
# Primitive rows: [ρ, e_int, vx, vy, vz] (vx = velocity along the swept axis).
# One per-interface kernel fuses reconstruct + Riemann (HLL_PLM does the same).

export muscl_flux_line!

# Enzo sign(): sign(0) = −1 (macro `(A)>0 ? 1 : -1`).
@inline _msign(x::T) where {T} = x > zero(T) ? one(T) : -one(T)

# 3-arg minmod (ReconstructionRoutines.h): nonzero only when a,b,c share a sign.
@inline function _minmod3(a::T, b::T, c::T) where {T}
    sa = _msign(a); sb = _msign(b); sc = _msign(c)
    return T(0.25) * (sa + sb) * abs(sa + sc) * min(abs(a), min(abs(b), abs(c)))
end

# limited PLM slope minmod((v−vm1)θ, (vp1−v)θ, ½(vp1−vm1)); the right-face value is
# v + ½·slope, the left-face value v − ½·slope (Hancock needs both; plm_point only
# the right one). `_plm_pt` is the right-face value (plm_point in Rec_PLM.C).
@inline _slope(vm1::T, v::T, vp1::T, θ::T) where {T} =
    _minmod3((v - vm1) * θ, (vp1 - v) * θ, T(0.5) * (vp1 - vm1))
@inline _plm_pt(vm1::T, v::T, vp1::T, θ::T) where {T} = v + T(0.5) * _slope(vm1, v, vp1, θ)

# HLL flux of one conserved field from L/R fluxes/states (Riemann_HLL.C).
@inline _hll(ap::T, am::T, fl::T, fr::T, ul::T, ur::T) where {T} =
    (ap * fl + am * fr - ap * am * (ur - ul)) / (ap + am)

# HLL flux 5-vector for hydro from explicit L/R *primitive* states (ρ,eint,u,v,w),
# u = normal velocity. Returns (Fρ, Fpu, Fpv, Fpw, FE). Shared by the bare flux line
# and the Hancock kernel (identical wave-speed + flux algebra as Riemann_HLL.C).
@inline function _hll5(ρl::T, el::T, ul::T, vl::T, wl::T,
                       ρr::T, er::T, ur::T, vr::T, wr::T, g::T, gm1::T) where {T}
    h = T(0.5)
    v2l = ul*ul + vl*vl + wl*wl; pl = gm1*ρl*el; csl = sqrt(g*pl/ρl); etl = el + h*v2l
    UlD = ρl; UlS1 = ρl*ul; UlS2 = ρl*vl; UlS3 = ρl*wl; UlE = ρl*etl
    FlD = ρl*ul; FlS1 = UlS1*ul + pl; FlS2 = UlS2*ul; FlS3 = UlS3*ul; FlE = (UlE + pl)*ul
    lpl = ul + csl; lml = ul - csl
    v2r = ur*ur + vr*vr + wr*wr; pr = gm1*ρr*er; csr = sqrt(g*pr/ρr); etr = er + h*v2r
    UrD = ρr; UrS1 = ρr*ur; UrS2 = ρr*vr; UrS3 = ρr*wr; UrE = ρr*etr
    FrD = ρr*ur; FrS1 = UrS1*ur + pr; FrS2 = UrS2*ur; FrS3 = UrS3*ur; FrE = (UrE + pr)*ur
    lpr = ur + csr; lmr = ur - csr
    ap = max(zero(T), max(lpl, lpr)); am = max(zero(T), max(-lml, -lmr))
    return (_hll(ap, am, FlD,  FrD,  UlD,  UrD),
            _hll(ap, am, FlS1, FrS1, UlS1, UrS1),
            _hll(ap, am, FlS2, FrS2, UlS2, UrS2),
            _hll(ap, am, FlS3, FrS3, UlS3, UrS3),
            _hll(ap, am, FlE,  FrE,  UlE,  UrE))
end

# HLL flux 6-vector: the 5 hydro fluxes PLUS the advective gas-energy flux for the
# dual-energy formalism (Enzo hydro_rk: Eint=ρe is a passively advected field,
# U=ρe, F=ρe·u, same HLL wave speeds). Used only on the dual path.
@inline function _hll6(ρl::T, el::T, ul::T, vl::T, wl::T,
                       ρr::T, er::T, ur::T, vr::T, wr::T, g::T, gm1::T) where {T}
    h = T(0.5)
    v2l = ul*ul + vl*vl + wl*wl; pl = gm1*ρl*el; csl = sqrt(g*pl/ρl); etl = el + h*v2l
    UlD = ρl; UlS1 = ρl*ul; UlS2 = ρl*vl; UlS3 = ρl*wl; UlE = ρl*etl; UlG = ρl*el
    FlD = ρl*ul; FlS1 = UlS1*ul + pl; FlS2 = UlS2*ul; FlS3 = UlS3*ul; FlE = (UlE + pl)*ul; FlG = UlG*ul
    lpl = ul + csl; lml = ul - csl
    v2r = ur*ur + vr*vr + wr*wr; pr = gm1*ρr*er; csr = sqrt(g*pr/ρr); etr = er + h*v2r
    UrD = ρr; UrS1 = ρr*ur; UrS2 = ρr*vr; UrS3 = ρr*wr; UrE = ρr*etr; UrG = ρr*er
    FrD = ρr*ur; FrS1 = UrS1*ur + pr; FrS2 = UrS2*ur; FrS3 = UrS3*ur; FrE = (UrE + pr)*ur; FrG = UrG*ur
    lpr = ur + csr; lmr = ur - csr
    ap = max(zero(T), max(lpl, lpr)); am = max(zero(T), max(-lml, -lmr))
    return (_hll(ap, am, FlD,  FrD,  UlD,  UrD),
            _hll(ap, am, FlS1, FrS1, UlS1, UrS1),
            _hll(ap, am, FlS2, FrS2, UlS2, UrS2),
            _hll(ap, am, FlS3, FrS3, UlS3, UrS3),
            _hll(ap, am, FlE,  FrE,  UlE,  UrE),
            _hll(ap, am, FlG,  FrG,  UlG,  UrG))
end

# HLLC flux 6-vector (Toro, Davis wave speeds) — contact-RESOLVING twin of `_hll6`,
# same (ρ,eint,u,v,w) L/R interface signature. Resolves the entropy/contact wave that
# rides on a background advection (which plain HLL smears badly), so an advected smooth
# wave keeps its amplitude + shape far better. Falls back to HLL on a non-positive star.
@inline function _hllc6(ρl::T, el::T, ul::T, vl::T, wl::T,
                        ρr::T, er::T, ur::T, vr::T, wr::T, g::T, gm1::T) where {T}
    h = T(0.5); pl = gm1*ρl*el; pr = gm1*ρr*er
    csl = sqrt(g*pl/ρl); csr = sqrt(g*pr/ρr)
    El = ρl*(el + h*(ul*ul + vl*vl + wl*wl)); Er = ρr*(er + h*(ur*ur + vr*vr + wr*wr))
    UlD=ρl; UlS1=ρl*ul; UlS2=ρl*vl; UlS3=ρl*wl; UlG=ρl*el
    FlD=ρl*ul; FlS1=UlS1*ul+pl; FlS2=UlS2*ul; FlS3=UlS3*ul; FlE=(El+pl)*ul; FlG=UlG*ul
    UrD=ρr; UrS1=ρr*ur; UrS2=ρr*vr; UrS3=ρr*wr; UrG=ρr*er
    FrD=ρr*ur; FrS1=UrS1*ur+pr; FrS2=UrS2*ur; FrS3=UrS3*ur; FrE=(Er+pr)*ur; FrG=UrG*ur
    SL = min(ul - csl, ur - csr); SR = max(ul + csl, ur + csr)
    Sstar = (pr - pl + ρl*ul*(SL - ul) - ρr*ur*(SR - ur)) / (ρl*(SL - ul) - ρr*(SR - ur))
    SL >= zero(T) && return (FlD, FlS1, FlS2, FlS3, FlE, FlG)
    SR <= zero(T) && return (FrD, FrS1, FrS2, FrS3, FrE, FrG)
    pstar = pl + ρl*(SL - ul)*(Sstar - ul)
    pstar <= zero(T) && return _hll6(ρl, el, ul, vl, wl, ρr, er, ur, vr, wr, g, gm1)
    if Sstar >= zero(T)
        ρs = ρl*(SL - ul)/(SL - Sstar)
        UsD=ρs; UsS1=ρs*Sstar; UsS2=ρs*vl; UsS3=ρs*wl
        UsE = ρs*(El/ρl + (Sstar - ul)*(Sstar + pl/(ρl*(SL - ul)))); UsG = ρs*el
        return (FlD + SL*(UsD-UlD), FlS1 + SL*(UsS1-UlS1), FlS2 + SL*(UsS2-UlS2),
                FlS3 + SL*(UsS3-UlS3), FlE + SL*(UsE-El), FlG + SL*(UsG-UlG))
    else
        ρs = ρr*(SR - ur)/(SR - Sstar)
        UsD=ρs; UsS1=ρs*Sstar; UsS2=ρs*vr; UsS3=ρs*wr
        UsE = ρs*(Er/ρr + (Sstar - ur)*(Sstar + pr/(ρr*(SR - ur)))); UsG = ρs*er
        return (FrD + SR*(UsD-UrD), FrS1 + SR*(UsS1-UrS1), FrS2 + SR*(UsS2-UrS2),
                FrS3 + SR*(UsS3-UrS3), FrE + SR*(UsE-Er), FrG + SR*(UsG-UrG))
    end
end

# Two-shock (van Leer 1979) Riemann flux, inline per-interface — the SAME solver Enzo's
# PPM-DirectEuler uses (`twoshock.F` + `flux_twoshock.F`), extracted to a scalar function
# with the `(ρ,eint,u,v,w)` interface of `_hll6`. It Newton-iterates for the contact
# (pbar,ubar) and resolves the time-averaged x/t=0 state, so it captures the ACOUSTIC wave
# structure (not just the contact) — the least diffusive of the three for smooth flow.
@inline function _twoshock6(ρl::T, el::T, ul::T, vl::T, wl::T,
                            ρr::T, er::T, ur::T, vr::T, wr::T, g::T, gm1::T) where {T}
    tn = T(1e-20)
    pl = max(gm1*ρl*el, tn); pr = max(gm1*ρr*er, tn)
    qa = (g + one(T))/(T(2)*g); gp1 = g + one(T)
    cl = sqrt(g*pl*ρl); cr = sqrt(g*pr*ρr)
    ps = max((cr*pl + cl*pr + cr*cl*(ul - ur))/(cr + cl), tn)
    old_ps = ps; ubl = zero(T); ubr = zero(T); dpdul = zero(T); dpdur = zero(T)
    tol = T === Float64 ? T(1e-14) : T(1e-7); conv = false
    for _n in 2:8
        if !conv
            zl = cl*sqrt(one(T) + qa*(ps/pl - one(T)))
            zr = cr*sqrt(one(T) + qa*(ps/pr - one(T)))
            ubl = ul - (ps - pl)/zl; ubr = ur + (ps - pr)/zr
            dpdul = -T(4)*zl^3/ρl/(T(4)*zl^2/ρl - gp1*(ps - pl))
            dpdur =  T(4)*zr^3/ρr/(T(4)*zr^2/ρr - gp1*(ps - pr))
            ps = max(ps + (ubr - ubl)*dpdur*dpdul/(dpdur - dpdul), tn)
            delta = ps - old_ps; old_ps = ps
            abs(delta/ps) < tol && (conv = true)
        end
    end
    ps < tn && (ps = min(pl, pr))
    pbar = ps; ubar = ubl + (ubr - ubl)*dpdur/(dpdur - dpdul)
    # resolve the time-averaged (bar) state at x/t=0 (Colella RAREFACTION2 branch)
    sn = (-ubar >= zero(T)) ? one(T) : -one(T)
    u0, p0, d0 = sn < zero(T) ? (ul, pl, ρl) : (ur, pr, ρr)
    c0 = sqrt(max(g*p0/d0, tn))
    z0 = c0*d0*sqrt(max(one(T) + qa*(pbar/p0 - one(T)), tn))
    dbar = one(T)/(one(T)/d0 - (pbar - p0)/max(z0*z0, tn))
    cbar = sqrt(max(g*pbar/dbar, tn))
    l0, lbar = pbar < p0 ? (u0*sn + c0, sn*ubar + cbar) : (u0*sn + z0/d0, u0*sn + z0/d0)
    frac = l0 - lbar; frac = frac < tn ? tn : frac
    frac = min(max((zero(T) - lbar)/frac, zero(T)), one(T))
    pbv = p0*frac + pbar*(one(T) - frac); dbv = d0*frac + dbar*(one(T) - frac); ubv = u0*frac + ubar*(one(T) - frac)
    lbar >= zero(T) && (pbv = pbar; dbv = dbar; ubv = ubar)
    l0 < zero(T) && (pbv = p0; dbv = d0; ubv = u0)
    vbv, wbv, gebv = ubv > zero(T) ? (vl, wl, el) : (vr, wr, er)   # transverse + gas energy upwind on ub
    ebv = pbv/(gm1*dbv) + T(0.5)*(ubv*ubv + vbv*vbv + wbv*wbv)
    dub = ubv*dbv
    return (dub, dub*ubv + pbv, dub*vbv, dub*wbv, dub*ebv + pbv*ubv, dub*gebv)
end

# `idual==1` ⇒ also write the gas-energy flux `fge` (the dual-energy 6th flux); the
# branch is uniform across threads. Non-dual callers pass idual=0 + a dummy fge.
@kernel function _muscl_hll_kernel!(fd, fs1, fs2, fs3, fe, fge,
                                    @Const(rho), @Const(eint), @Const(vx),
                                    @Const(vy), @Const(vz),
                                    ncells::Int, nfi::Int, nghost::Int, j1::Int,
                                    gamma, theta, small_rho, idual::Int)
    gi, gj = @index(Global, NTuple)              # gi: interface 1..active+1; gj: pencil
    j = j1 + gj - 1
    cl = (j - 1) * ncells + nghost + gi - 1      # left cell of interface gi
    fo = (j - 1) * nfi + gi                       # flux output index
    T = eltype(fd)
    g = gamma; gm1 = g - one(T); θ = theta
    @inbounds begin
        # ── PLM reconstruction: L state = right face of cell cl, R = left face of cl+1
        ρl = max(_plm_pt(rho[cl-1],  rho[cl],   rho[cl+1], θ), small_rho)
        el =     _plm_pt(eint[cl-1], eint[cl],  eint[cl+1], θ)
        ul =     _plm_pt(vx[cl-1],   vx[cl],    vx[cl+1],  θ)
        vl =     _plm_pt(vy[cl-1],   vy[cl],    vy[cl+1],  θ)
        wl =     _plm_pt(vz[cl-1],   vz[cl],    vz[cl+1],  θ)
        ρr = max(_plm_pt(rho[cl+2],  rho[cl+1], rho[cl],   θ), small_rho)
        er =     _plm_pt(eint[cl+2], eint[cl+1], eint[cl], θ)
        ur =     _plm_pt(vx[cl+2],   vx[cl+1],  vx[cl],    θ)
        vr =     _plm_pt(vy[cl+2],   vy[cl+1],  vy[cl],    θ)
        wr =     _plm_pt(vz[cl+2],   vz[cl+1],  vz[cl],    θ)

        F = _hll6(ρl, el, ul, vl, wl, ρr, er, ur, vr, wr, g, gm1)
        fd[fo] = F[1]; fs1[fo] = F[2]; fs2[fo] = F[3]; fs3[fo] = F[4]; fe[fo] = F[5]
        idual == 1 && (fge[fo] = F[6])
    end
end

"""
    muscl_flux_line!(fd, fs1, fs2, fs3, fe, rho, eint, vx, vy, vz;
                     ncells, nghost, jdim=1, gamma, theta=1.5, small_rho=1e-10,
                     fge=nothing)

Fill the five interface-flux arrays (density, 3 momenta, total energy) over
`active+1` interfaces per pencil via fused PLM+HLL. `rho/eint/vx/vy/vz` are the
primitive lines (length `ncells·jdim`, `nghost` ghosts each side); flux arrays
have length `(active+1)·jdim` where `active = ncells − 2·nghost`. Element type of
`fd` sets the working precision. Passing `fge` (a 6th flux array) turns on the
dual-energy gas-energy flux (Eint=ρe advection).
"""
function muscl_flux_line!(fd, fs1, fs2, fs3, fe, rho, eint, vx, vy, vz;
                          ncells::Integer, nghost::Integer, jdim::Integer = 1,
                          gamma::Real, theta::Real = 1.5, small_rho::Real = 1e-10,
                          fge = nothing)
    be = KA.get_backend(fd)
    T = eltype(fd)
    ncells, nghost = Int(ncells), Int(nghost)
    active = ncells - 2 * nghost
    nfi = active + 1
    _muscl_hll_kernel!(be)(fd, fs1, fs2, fs3, fe, fge === nothing ? fd : fge,
                           rho, eint, vx, vy, vz,
                           ncells, nfi, nghost, 1, T(gamma), T(theta), T(small_rho),
                           fge === nothing ? 0 : 1;
                           ndrange = (nfi, Int(jdim)))
    KA.synchronize(be)
    return fd, fs1, fs2, fs3, fe
end

# ── MUSCL-Hancock: PLM + half-step predictor + HLL, 2nd-order in space AND time ─
# The Hancock predictor evolves a cell's two PLM boundary-extrapolated states by
# ½dt using the *conservative* physical flux (Toro §14.4): with W⁻,W⁺ the cell's
# minus/plus face states, ΔF = F(W⁻) − F(W⁺), both faces get  U += (dt/2dx)·ΔF.
# It is a LOCAL per-cell operation (stencil cl−1..cl+2, the same reach the bare
# reconstruction already uses), so it folds into ONE fused per-interface kernel —
# no extra sweep, no extra transpose. That lets the grid driver be dimensionally
# SPLIT (one Riemann solve per direction = 3 sweeps/step, like PPM) yet stay 2nd
# order, instead of the unsplit RK2's 2 stages × 3 axes = 6 sweeps/step.

# Hancock ½-step predictor shared by PLM and PPM reconstruction: given a cell's two
# boundary-extrapolated primitive face states (a = minus/left, b = plus/right; ρ
# already floored), evolve BOTH by cpred·(F(W⁻) − F(W⁺)) in conserved form and
# return the evolved primitives. `cpred = dt/(2·dx)`; cpred=0 ⇒ untouched faces.
@inline function _hancock_predict(ρa::T, ea::T, ua::T, va::T, wa::T,
                                  ρb::T, eb::T, ub::T, vb::T, wb::T,
                                  gm1::T, cpred::T, small_rho::T) where {T}
    h = T(0.5)
    pa = gm1*ρa*ea; v2a = ua*ua + va*va + wa*wa; Ea = ρa*(ea + h*v2a)
    pb = gm1*ρb*eb; v2b = ub*ub + vb*vb + wb*wb; Eb = ρb*(eb + h*v2b)
    Ua1 = ρa; Ua2 = ρa*ua; Ua3 = ρa*va; Ua4 = ρa*wa; Ua5 = Ea
    Ub1 = ρb; Ub2 = ρb*ub; Ub3 = ρb*vb; Ub4 = ρb*wb; Ub5 = Eb
    Fa1 = ρa*ua; Fa2 = Ua2*ua + pa; Fa3 = Ua3*ua; Fa4 = Ua4*ua; Fa5 = (Ea + pa)*ua
    Fb1 = ρb*ub; Fb2 = Ub2*ub + pb; Fb3 = Ub3*ub; Fb4 = Ub4*ub; Fb5 = (Eb + pb)*ub
    d1 = cpred*(Fa1-Fb1); d2 = cpred*(Fa2-Fb2); d3 = cpred*(Fa3-Fb3); d4 = cpred*(Fa4-Fb4); d5 = cpred*(Fa5-Fb5)
    ρas = max(Ua1+d1, small_rho); uas = (Ua2+d2)/ρas; vas = (Ua3+d3)/ρas; was = (Ua4+d4)/ρas
    eas = max((Ua5+d5)/ρas - h*(uas*uas + vas*vas + was*was), small_rho)   # eint floor ⇒ p>0
    ρbs = max(Ub1+d1, small_rho); ubs = (Ub2+d2)/ρbs; vbs = (Ub3+d3)/ρbs; wbs = (Ub4+d4)/ρbs
    ebs = max((Ub5+d5)/ρbs - h*(ubs*ubs + vbs*vbs + wbs*wbs), small_rho)
    return (ρas, eas, uas, vas, was, ρbs, ebs, ubs, vbs, wbs)
end

# `cpred = dt/(2·dx)`; cpred=0 makes this reduce to the bare reconstruction+HLL
# (used to certify the reconstruction/HLL path against `muscl_flux_line!`).
@inline function _hancock_faces(rm::T, r0::T, rp::T, em::T, e0::T, ep::T,
                                um::T, u0::T, up::T, vm::T, v0::T, vp::T,
                                wm::T, w0::T, wp::T, θ::T, gm1::T, cpred::T, small_rho::T) where {T}
    h = T(0.5)
    sρ = _slope(rm, r0, rp, θ); se = _slope(em, e0, ep, θ); su = _slope(um, u0, up, θ)
    sv = _slope(vm, v0, vp, θ); sw = _slope(wm, w0, wp, θ)
    # minus (a) / plus (b) PLM boundary-extrapolated primitive states
    ρa = max(r0 - h*sρ, small_rho); ea = e0 - h*se; ua = u0 - h*su; va = v0 - h*sv; wa = w0 - h*sw
    ρb = max(r0 + h*sρ, small_rho); eb = e0 + h*se; ub = u0 + h*su; vb = v0 + h*sv; wb = w0 + h*sw
    return _hancock_predict(ρa, ea, ua, va, wa, ρb, eb, ub, vb, wb, gm1, cpred, small_rho)
end

# PPM (piecewise-parabolic) faces: identical Hancock predictor, but the cell's two
# face states are the monotonized-parabola edges (a_L=ql, a_R=qr) from the certified
# `_iv_recon_cell` (intvar.jl), reconstructed per primitive. `cl..c6` are the
# uniform-grid PPM coefficients; `idx`/`ci` are the cell's flat index / swept-axis
# coordinate. Plain monotonized parabola (isteep=0, iflatten=0 ⇒ steepen/flatten
# arrays unused, so the field array itself is passed as the dummy).
@inline function _ppm_hancock_faces(rho, eint, vx, vy, vz,
                                    c1, c2, c3, c4, c5, c6, idx::Int, ci::Int,
                                    gm1::T, cpred::T, small_rho::T) where {T}
    (ρa, ρb, _, _) = _iv_recon_cell(rho,  c1, c2, c3, c4, c5, c6, rho,  rho,  idx, ci, 0, 0)
    (ea, eb, _, _) = _iv_recon_cell(eint, c1, c2, c3, c4, c5, c6, eint, eint, idx, ci, 0, 0)
    (ua, ub, _, _) = _iv_recon_cell(vx,   c1, c2, c3, c4, c5, c6, vx,   vx,   idx, ci, 0, 0)
    (va, vb, _, _) = _iv_recon_cell(vy,   c1, c2, c3, c4, c5, c6, vy,   vy,   idx, ci, 0, 0)
    (wa, wb, _, _) = _iv_recon_cell(vz,   c1, c2, c3, c4, c5, c6, vz,   vz,   idx, ci, 0, 0)
    return _hancock_predict(max(ρa, small_rho), ea, ua, va, wa,
                            max(ρb, small_rho), eb, ub, vb, wb, gm1, cpred, small_rho)
end

# PPM faces with a CHARACTERISTIC-TRACE time-update (instead of the Hancock half-step):
# reconstruct the (ρ,p,u,v,w) monotonized parabola per primitive (same certified
# `_iv_recon_cell`), then time-centre each face by `_ppml_face_left/right` — which
# integrates the WHOLE parabola over each wave's departure region (keeping the curvature
# term the Hancock step drops), exactly the PPM time-averaging DirectEuler does. Needs
# the PRESSURE line `pr` (the trace works in p); converts the traced face p→eint for the
# Riemann flux. Returns the same (minus-face | plus-face) 10-tuple as `_ppm_hancock_faces`.
@inline function _ppm_trace_faces(rho, pr, vx, vy, vz, c1, c2, c3, c4, c5, c6,
                                  idx::Int, ci::Int, g::T, gm1::T, dt_dx::T, small_rho::T) where {T}
    pmin = gm1 * small_rho
    (ρL, ρR, _, _) = _iv_recon_cell(rho, c1, c2, c3, c4, c5, c6, rho, rho, idx, ci, 0, 0)
    (pL, pR, _, _) = _iv_recon_cell(pr,  c1, c2, c3, c4, c5, c6, pr,  pr,  idx, ci, 0, 0)
    (uL, uR, _, _) = _iv_recon_cell(vx,  c1, c2, c3, c4, c5, c6, vx,  vx,  idx, ci, 0, 0)
    (vL, vR, _, _) = _iv_recon_cell(vy,  c1, c2, c3, c4, c5, c6, vy,  vy,  idx, ci, 0, 0)
    (wL, wR, _, _) = _iv_recon_cell(vz,  c1, c2, c3, c4, c5, c6, vz,  vz,  idx, ci, 0, 0)
    ρLf = max(ρL, small_rho); ρRf = max(ρR, small_rho); pLf = max(pL, pmin); pRf = max(pR, pmin)
    ρa = max(rho[idx], small_rho); pa = max(pr[idx], pmin); ua = vx[idx]; va = vy[idx]; wa = vz[idx]
    (Lρ, Lu, Lv, Lw, Lp) = _ppml_face_left(ρLf, uL, vL, wL, pLf, ρa, ua, va, wa, pa, ρRf, uR, vR, wR, pRf, dt_dx, g)
    (Rρ, Ru, Rv, Rw, Rp) = _ppml_face_right(ρLf, uL, vL, wL, pLf, ρa, ua, va, wa, pa, ρRf, uR, vR, wR, pRf, dt_dx, g)
    return (Lρ, Lp/(gm1*Lρ), Lu, Lv, Lw, Rρ, Rp/(gm1*Rρ), Ru, Rv, Rw)   # p→eint for the flux
end

# Three-cell, one-ghost-local PPM reconstruction.  The unlimited edges are the
# unique quadratic consistent with the cell averages (q_{i-1},q_i,q_{i+1}):
# qL/R = q_i ∓ (q_{i+1}-q_{i-1})/4 + Δ²q_i/12.  Smooth cells use CW84
# monotonization; compression/pressure-jump sensors continuously blend all
# primitives toward a monotonized-central PLM profile, reaching the robust TVD
# endpoint at shocks. Pressure-smooth entropy contacts get a density-only
# steepening pass, so contact control does not flatten pressure/velocity.
# Unlike PPML this is stateless: no carried face pair or star-state corrector.
@inline function _ppm_local_edges_unlimited(qm::T, q0::T, qp::T) where {T}
    slope = (qp - qm) * T(0.25)
    curve = (qm - T(2)*q0 + qp) / T(12)
    return (q0 - slope + curve, q0 + slope + curve)
end

@inline function _ppm_local_edges(qm::T, q0::T, qp::T) where {T}
    qL, qR = _ppm_local_edges_unlimited(qm, q0, qp)
    return _ppml_monotonize(qL, q0, qR)
end

@inline function _thinc_edges(qm::T, q0::T, qp::T, β::T) where {T}
    θ = qp > qm ? one(T) : -one(T)
    qlo = min(qm, qp); qhi = max(qm, qp); dq = qhi - qlo
    C = clamp((q0 - qlo) / dq, T(1e-6), one(T) - T(1e-6))
    eb = exp(β * T(0.5))
    r = exp(β * θ * (T(2) * C - one(T)))
    z = max((eb - r / eb) / (r * eb - one(T) / eb), T(1e-12))
    x0 = log(z) / (T(2) * β)
    qL = qlo + T(0.5) * dq * (one(T) + θ * tanh(β * (-T(0.5) - x0)))
    qR = qlo + T(0.5) * dq * (one(T) + θ * tanh(β * ( T(0.5) - x0)))
    return (qL, qR)
end

@inline function _label_brackets_contact(label, label2, idx::Int, level1::T, level2::T,
                                         use_moment2::Int) where {T}
    am = label[idx-1]; ap = label[idx+1]
    a0 = label[idx]
    span = max(abs(ap - am), T(1e-6))
    reach = use_moment2 == 1 ? T(3.0) : T(1.5)
    w = zero(T)
    if level1 >= zero(T)
        d = abs(a0 - level1); d = min(d, one(T) - d)
        w = max(w, clamp(reach - d / span, zero(T), one(T)))
    end
    if level2 >= zero(T)
        d = abs(a0 - level2); d = min(d, one(T) - d)
        w = max(w, clamp(reach - d / span, zero(T), one(T)))
    end
    if use_moment2 == 1
        var = max(label2[idx] - a0*a0, zero(T))
        # A clean material sheet has low carried-label variance. The first
        # scalar prototype is only upwinded, so keep this penalty deliberately
        # soft rather than letting numerical label diffusion disable THINC.
        clean = one(T) - T(0.5) * clamp(var / (T(4)*span*span + T(1e-12)), zero(T), one(T))
        w *= clean
    end
    return w
end

@inline function _ppm_local_trace_faces(rho, pr, vx, vy, vz, label, label2, idx::Int,
                                        g::T, gm1::T, dt_dx::T, small_rho::T,
                                        use_label::Int, use_moment2::Int,
                                        level1::T, level2::T) where {T}
    pmin = gm1 * small_rho
    ρm = rho[idx-1]; ρa0 = rho[idx]; ρp = rho[idx+1]
    um = vx[idx-1]; ua0 = vx[idx]; up = vx[idx+1]
    vm = vy[idx-1]; va0 = vy[idx]; vp = vy[idx+1]
    wm = vz[idx-1]; wa0 = vz[idx]; wp = vz[idx+1]
    pm = pr[idx-1]; pa0 = pr[idx]; pp = pr[idx+1]
    (ρL0, ρR0) = _ppm_local_edges_unlimited(ρm, ρa0, ρp)
    (pL0, pR0) = _ppm_local_edges_unlimited(pm, pa0, pp)
    (uL0, uR0) = _ppm_local_edges_unlimited(um, ua0, up)
    (vL0, vR0) = _ppm_local_edges_unlimited(vm, va0, vp)
    (wL0, wR0) = _ppm_local_edges_unlimited(wm, wa0, wp)
    pbase = min(pm, pp)
    ρbase = min(ρm, ρp)
    ηp = pbase > zero(T) ? abs(pp - pm) / pbase : one(T)
    ηρ = ρbase > zero(T) ? abs(ρp - ρm) / ρbase : one(T)
    cs = sqrt(g * max(pa0, pmin) / max(ρa0, small_rho))
    comp = um > up ? (um-up)/cs : zero(T)
    # Smoothly blend PPM -> monotonized-central PLM instead of switching the whole compressive
    # region to PLM at one threshold. Mild compression keeps most curvature;
    # strong shocks reach the same robust PLM endpoint as before.
    αp = clamp((ηp - T(0.05)) / T(0.45), zero(T), one(T))
    αu = clamp((comp - T(0.1)) / T(0.9), zero(T), one(T))
    αshock = um > up ? max(αp, αu) : zero(T)
    smooth_p = one(T) - clamp(ηp / T(0.05), zero(T), one(T))
    smooth_u = one(T) - clamp(comp / T(0.1), zero(T), one(T))
    monotone_ρ = (ρp - ρa0) * (ρa0 - ρm) > zero(T) ? one(T) : zero(T)
    contact = clamp(ηρ / T(0.005), zero(T), one(T)) * smooth_p * smooth_u * monotone_ρ
    label_contact = use_label == 1 ?
        _label_brackets_contact(label, label2, idx, level1, level2, use_moment2) : one(T)
    σcontact = T(0.02) * contact
    αρ = max(αshock, contact)
    α = αshock
    (ρLq, ρRq) = _ppml_monotonize(ρL0, ρa0, ρR0)
    (uLq, uRq) = _ppml_monotonize(uL0, ua0, uR0)
    (vLq, vRq) = _ppml_monotonize(vL0, va0, vR0)
    (wLq, wRq) = _ppml_monotonize(wL0, wa0, wR0)
    (pLq, pRq) = _ppml_monotonize(pL0, pa0, pR0)
    h = T(0.5); θ = T(2)
    sρ = _slope(ρm, ρa0, ρp, θ); su = _slope(um, ua0, up, θ)
    sv = _slope(vm, va0, vp, θ); sw = _slope(wm, wa0, wp, θ)
    sp = _slope(pm, pa0, pp, θ)
    ρLp = ρa0 - h*sρ; ρRp = ρa0 + h*sρ
    uLp = ua0 - h*su; uRp = ua0 + h*su
    vLp = va0 - h*sv; vRp = va0 + h*sv
    wLp = wa0 - h*sw; wRp = wa0 + h*sw
    pLp = pa0 - h*sp; pRp = pa0 + h*sp
    β = one(T) - α
    βρ = one(T) - αρ
    ρL=βρ*ρLq + αρ*ρLp; ρR=βρ*ρRq + αρ*ρRp; uL=β*uLq + α*uLp; uR=β*uRq + α*uRp
    vL=β*vLq + α*vLp; vR=β*vRq + α*vRp; wL=β*wLq + α*wLp; wR=β*wRq + α*wRp
    pL=β*pLq + α*pLp; pR=β*pRq + α*pRp
    γc = one(T) - σcontact
    ρL = γc * ρL + σcontact * ρm
    ρR = γc * ρR + σcontact * ρp
    if contact > zero(T) && label_contact > zero(T)
        ρLt, ρRt = _thinc_edges(ρm, ρa0, ρp, T(1.6))
        bv0 = abs(ρL - ρm) + abs(ρR - ρp)
        bvt = abs(ρLt - ρm) + abs(ρRt - ρp)
        thinc_gain = use_moment2 == 1 ? T(0.05) : use_label == 1 ? T(0.15) : T(0.04)
        use_t = bvt < bv0 ? thinc_gain * contact * label_contact : zero(T)
        keep_t = one(T) - use_t
        ρL = keep_t * ρL + use_t * ρLt
        ρR = keep_t * ρR + use_t * ρRt
    end
    # The MC endpoint is already TVD; do not add PPML's piecewise-constant shock
    # flattening on top. Positivity and the final CW84 guard remain below.
    (ρL, uL, vL, wL, pL, ρR, uR, vR, wR, pR) =
        _ppml_monotonize_all(ρL, uL, vL, wL, pL, ρa0, ua0, va0, wa0, pa0,
                            ρR, uR, vR, wR, pR)
    ρLf = max(ρL, small_rho); ρRf = max(ρR, small_rho)
    pLf = max(pL, pmin); pRf = max(pR, pmin)
    ρa = max(ρa0, small_rho); pa = max(pa0, pmin)
    ua = ua0; va = va0; wa = wa0
    (Lρ, Lu, Lv, Lw, Lp) = _ppml_face_left(
        ρLf, uL, vL, wL, pLf, ρa, ua, va, wa, pa, ρRf, uR, vR, wR, pRf, dt_dx, g)
    (Rρ, Ru, Rv, Rw, Rp) = _ppml_face_right(
        ρLf, uL, vL, wL, pLf, ρa, ua, va, wa, pa, ρRf, uR, vR, wR, pRf, dt_dx, g)
    return (Lρ, Lp/(gm1*Lρ), Lu, Lv, Lw, Rρ, Rp/(gm1*Rρ), Ru, Rv, Rw)
end

@inline function _ppm_local_scalar_faces(rho, scalar, idx::Int, small_rho::T) where {T}
    qm = scalar[idx-1] / max(rho[idx-1], small_rho)
    q0 = scalar[idx]   / max(rho[idx],   small_rho)
    qp = scalar[idx+1] / max(rho[idx+1], small_rho)
    return _ppm_local_edges(qm, q0, qp)
end

@inline function _ppm_local_scalar_left(rho, scalar, idx::Int, small_rho::T) where {T}
    qm = scalar[idx-1] / max(rho[idx-1], small_rho)
    q0 = scalar[idx]   / max(rho[idx],   small_rho)
    qp = scalar[idx+1] / max(rho[idx+1], small_rho)
    qL, _ = _ppm_local_edges(qm, q0, qp)
    return qL
end

@inline function _ppm_local_scalar_right(rho, scalar, idx::Int, small_rho::T) where {T}
    qm = scalar[idx-1] / max(rho[idx-1], small_rho)
    q0 = scalar[idx]   / max(rho[idx],   small_rho)
    qp = scalar[idx+1] / max(rho[idx+1], small_rho)
    _, qR = _ppm_local_edges(qm, q0, qp)
    return qR
end

@inline function _constant_faces(rho, pr, vx, vy, vz, idx::Int, gm1::T, small_rho::T) where {T}
    ρ = max(rho[idx], small_rho); p = max(pr[idx], gm1*small_rho)
    e = p / (gm1*ρ); u = vx[idx]; v = vy[idx]; w = vz[idx]
    return (ρ, e, u, v, w, ρ, e, u, v, w)
end

@kernel function _muscl_hancock_kernel!(fd, fs1, fs2, fs3, fe, fge,
                                        @Const(rho), @Const(eint), @Const(vx),
                                        @Const(vy), @Const(vz),
                                        ncells::Int, nfi::Int, nghost::Int, j1::Int,
                                        gamma, theta, cpred, small_rho, idual::Int, rie::Int)
    gi, gj = @index(Global, NTuple)
    j = j1 + gj - 1
    cl = (j - 1) * ncells + nghost + gi - 1      # left cell of interface gi
    fo = (j - 1) * nfi + gi
    T = eltype(fd); g = gamma; gm1 = g - one(T); θ = theta; cp = cpred; sr = small_rho
    @inbounds begin
        # left cell cl → its evolved PLUS (right) face; right cell cl+1 → its evolved MINUS (left) face
        Lf = _hancock_faces(rho[cl-1], rho[cl], rho[cl+1], eint[cl-1], eint[cl], eint[cl+1],
                            vx[cl-1], vx[cl], vx[cl+1], vy[cl-1], vy[cl], vy[cl+1],
                            vz[cl-1], vz[cl], vz[cl+1], θ, gm1, cp, sr)
        Rf = _hancock_faces(rho[cl], rho[cl+1], rho[cl+2], eint[cl], eint[cl+1], eint[cl+2],
                            vx[cl], vx[cl+1], vx[cl+2], vy[cl], vy[cl+1], vy[cl+2],
                            vz[cl], vz[cl+1], vz[cl+2], θ, gm1, cp, sr)
        F = rie == 2 ? _twoshock6(Lf[6], Lf[7], Lf[8], Lf[9], Lf[10], Rf[1], Rf[2], Rf[3], Rf[4], Rf[5], g, gm1) :
            rie == 1 ? _hllc6(Lf[6], Lf[7], Lf[8], Lf[9], Lf[10], Rf[1], Rf[2], Rf[3], Rf[4], Rf[5], g, gm1) :
                       _hll6(Lf[6], Lf[7], Lf[8], Lf[9], Lf[10], Rf[1], Rf[2], Rf[3], Rf[4], Rf[5], g, gm1)
        fd[fo] = F[1]; fs1[fo] = F[2]; fs2[fo] = F[3]; fs3[fo] = F[4]; fe[fo] = F[5]
        idual == 1 && (fge[fo] = F[6])
    end
end

# PPM-Hancock per-interface kernel: same structure as `_muscl_hancock_kernel!` but
# parabolic reconstruction (via `_ppm_hancock_faces`). `c1..c6` are the uniform-grid
# PPM coefficients (length ≥ ncells); the swept-axis coordinate of cell cl is the
# index into them. Needs nghost ≥ 3 (parabola reads cl−2 … cl+3 across an interface).
@kernel function _ppm_hancock_kernel!(fd, fs1, fs2, fs3, fe, fge,
                                      @Const(rho), @Const(eint), @Const(vx), @Const(vy), @Const(vz),
                                      @Const(c1), @Const(c2), @Const(c3), @Const(c4), @Const(c5), @Const(c6),
                                      ncells::Int, nfi::Int, nghost::Int, j1::Int,
                                      gamma, cpred, small_rho, idual::Int, rie::Int)
    gi, gj = @index(Global, NTuple)
    j = j1 + gj - 1
    cl = (j - 1) * ncells + nghost + gi - 1      # left cell of interface gi (flat index)
    cil = nghost + gi - 1                          # …and its swept-axis coordinate
    fo = (j - 1) * nfi + gi
    T = eltype(fd); g = gamma; gm1 = g - one(T); cp = cpred; sr = small_rho
    @inbounds begin
        Lf = _ppm_hancock_faces(rho, eint, vx, vy, vz, c1, c2, c3, c4, c5, c6, cl,     cil,     gm1, cp, sr)
        Rf = _ppm_hancock_faces(rho, eint, vx, vy, vz, c1, c2, c3, c4, c5, c6, cl + 1, cil + 1, gm1, cp, sr)
        F = rie == 2 ? _twoshock6(Lf[6], Lf[7], Lf[8], Lf[9], Lf[10], Rf[1], Rf[2], Rf[3], Rf[4], Rf[5], g, gm1) :
            rie == 1 ? _hllc6(Lf[6], Lf[7], Lf[8], Lf[9], Lf[10], Rf[1], Rf[2], Rf[3], Rf[4], Rf[5], g, gm1) :
                       _hll6(Lf[6], Lf[7], Lf[8], Lf[9], Lf[10], Rf[1], Rf[2], Rf[3], Rf[4], Rf[5], g, gm1)
        fd[fo] = F[1]; fs1[fo] = F[2]; fs2[fo] = F[3]; fs3[fo] = F[4]; fe[fo] = F[5]
        idual == 1 && (fge[fo] = F[6])
    end
end

# PPM-TRACE per-interface kernel: like `_ppm_hancock_kernel!` but the characteristic-trace
# time-update (`_ppm_trace_faces`, needs the PRESSURE line `pr` + `dt_dx` instead of cpred).
@kernel function _ppm_trace_kernel!(fd, fs1, fs2, fs3, fe, fge,
                                    @Const(rho), @Const(pr), @Const(vx), @Const(vy), @Const(vz),
                                    @Const(c1), @Const(c2), @Const(c3), @Const(c4), @Const(c5), @Const(c6),
                                    ncells::Int, nfi::Int, nghost::Int, j1::Int,
                                    gamma, dt_dx, small_rho, idual::Int, rie::Int)
    gi, gj = @index(Global, NTuple)
    j = j1 + gj - 1
    cl = (j - 1) * ncells + nghost + gi - 1
    cil = nghost + gi - 1
    fo = (j - 1) * nfi + gi
    T = eltype(fd); g = gamma; gm1 = g - one(T); dd = dt_dx; sr = small_rho
    @inbounds begin
        Lf = _ppm_trace_faces(rho, pr, vx, vy, vz, c1, c2, c3, c4, c5, c6, cl,     cil,     g, gm1, dd, sr)
        Rf = _ppm_trace_faces(rho, pr, vx, vy, vz, c1, c2, c3, c4, c5, c6, cl + 1, cil + 1, g, gm1, dd, sr)
        F = rie == 2 ? _twoshock6(Lf[6], Lf[7], Lf[8], Lf[9], Lf[10], Rf[1], Rf[2], Rf[3], Rf[4], Rf[5], g, gm1) :
            rie == 1 ? _hllc6(Lf[6], Lf[7], Lf[8], Lf[9], Lf[10], Rf[1], Rf[2], Rf[3], Rf[4], Rf[5], g, gm1) :
                       _hll6(Lf[6], Lf[7], Lf[8], Lf[9], Lf[10], Rf[1], Rf[2], Rf[3], Rf[4], Rf[5], g, gm1)
        fd[fo] = F[1]; fs1[fo] = F[2]; fs2[fo] = F[3]; fs3[fo] = F[4]; fe[fo] = F[5]
        idual == 1 && (fge[fo] = F[6])
    end
end

# One-ghost-local PPM-trace kernel.  Active cells use the three-cell quadratic;
# the boundary-adjacent ghost contributes a degenerate constant state because a
# second ghost is unavailable.  A host code should compute each inter-block face
# flux once (or reflux it), as usual for a one-ghost AMR/Godunov update.
@kernel function _ppm_local_trace_kernel!(fd, fs1, fs2, fs3, fe, fge,
                                          @Const(rho), @Const(pr), @Const(vx),
                                          @Const(vy), @Const(vz), @Const(label), @Const(label2),
                                          ncells::Int, nfi::Int, nghost::Int, j1::Int,
                                          gamma, dt_dx, small_rho, idual::Int, rie::Int,
                                          periodic::Int, use_label::Int, use_moment2::Int,
                                          level1, level2)
    gi, gj = @index(Global, NTuple)
    j = j1 + gj - 1
    cl = (j - 1) * ncells + nghost + gi - 1
    fo = (j - 1) * nfi + gi
    T = eltype(fd); g = gamma; gm1 = g - one(T); dd = dt_dx; sr = small_rho
    @inbounds begin
        base = (j - 1) * ncells
        li = periodic == 1 && gi == 1 ? base + ncells - nghost : cl
        ri = periodic == 1 && gi == nfi ? base + nghost + 1 : cl + 1
        Lf = periodic == 0 && gi == 1 ?
             _constant_faces(rho, pr, vx, vy, vz, li, gm1, sr) :
             _ppm_local_trace_faces(rho, pr, vx, vy, vz, label, label2, li, g, gm1, dd, sr,
                                    use_label, use_moment2, T(level1), T(level2))
        Rf = periodic == 0 && gi == nfi ?
             _constant_faces(rho, pr, vx, vy, vz, ri, gm1, sr) :
             _ppm_local_trace_faces(rho, pr, vx, vy, vz, label, label2, ri, g, gm1, dd, sr,
                                    use_label, use_moment2, T(level1), T(level2))
        F = rie == 2 ? _twoshock6(Lf[6], Lf[7], Lf[8], Lf[9], Lf[10], Rf[1], Rf[2], Rf[3], Rf[4], Rf[5], g, gm1) :
            rie == 1 ? _hllc6(Lf[6], Lf[7], Lf[8], Lf[9], Lf[10], Rf[1], Rf[2], Rf[3], Rf[4], Rf[5], g, gm1) :
                       _hll6(Lf[6], Lf[7], Lf[8], Lf[9], Lf[10], Rf[1], Rf[2], Rf[3], Rf[4], Rf[5], g, gm1)
        fd[fo] = F[1]; fs1[fo] = F[2]; fs2[fo] = F[3]; fs3[fo] = F[4]; fe[fo] = F[5]
        idual == 1 && (fge[fo] = F[6])
    end
end

"""
    muscl_hancock_flux_line!(fd, fs1, fs2, fs3, fe, rho, eint, vx, vy, vz;
                             ncells, nghost, jdim=1, gamma, theta=1.5, cpred,
                             small_rho=1e-10, recon=:plm, coeffs=nothing)

Like [`muscl_flux_line!`](@ref) but with the MUSCL-Hancock ½-step predictor folded
in. `cpred = dt/(2·dx)` is the predictor coefficient; `cpred=0` recovers the bare
reconstruction+HLL flux line (up to the conserved↔primitive round-trip). `recon`
selects PLM (`:plm`, minmod-θ) or PPM (`:ppm`, monotonized parabola); the PPM path
needs `coeffs = (c1,…,c6)` (the uniform-grid coefficients) and `nghost ≥ 3`. Passing
`fge` (a 6th flux array) turns on the dual-energy gas-energy flux (Eint=ρe).
"""
function muscl_hancock_flux_line!(fd, fs1, fs2, fs3, fe, rho, eint, vx, vy, vz;
                                  ncells::Integer, nghost::Integer, jdim::Integer = 1,
                                  gamma::Real, theta::Real = 1.5, cpred::Real,
                                  small_rho::Real = 1e-10, recon::Symbol = :plm, coeffs = nothing,
                                  fge = nothing, riemann::Symbol = :hll, predictor::Symbol = :hancock,
                                  pr = nothing, face_periodic::Bool = false, contact_label = nothing,
                                  contact_label2 = nothing,
                                  contact_level1::Real = -1, contact_level2::Real = -1)
    be = KA.get_backend(fd); T = eltype(fd)
    ncells, nghost = Int(ncells), Int(nghost)
    active = ncells - 2 * nghost; nfi = active + 1
    gef = fge === nothing ? fd : fge; idu = fge === nothing ? 0 : 1
    rc = riemann === :twoshock ? 2 : riemann === :hllc ? 1 : 0
    if recon === :ppm_local
        (predictor !== :trace || pr === nothing) &&
            error("muscl_hancock_flux_line!: recon=:ppm_local requires predictor=:trace and pr")
        label = contact_label === nothing ? rho : contact_label
        label2 = contact_label2 === nothing ? label : contact_label2
        _ppm_local_trace_kernel!(be)(fd, fs1, fs2, fs3, fe, gef,
                                     rho, pr, vx, vy, vz, label, label2,
                                     ncells, nfi, nghost, 1, T(gamma), T(2 * cpred),
                                     T(small_rho), idu, rc, face_periodic ? 1 : 0,
                                     contact_label === nothing ? 0 : 1,
                                     contact_label2 === nothing ? 0 : 1,
                                     T(contact_level1), T(contact_level2);
                                     ndrange = (nfi, Int(jdim)))
    elseif recon === :ppm && predictor === :trace
        (coeffs === nothing || pr === nothing) && error("muscl_hancock_flux_line!: predictor=:trace needs coeffs + pr")
        c1, c2, c3, c4, c5, c6 = coeffs
        _ppm_trace_kernel!(be)(fd, fs1, fs2, fs3, fe, gef, rho, pr, vx, vy, vz,
                               c1, c2, c3, c4, c5, c6,
                               ncells, nfi, nghost, 1, T(gamma), T(2 * cpred), T(small_rho), idu, rc;
                               ndrange = (nfi, Int(jdim)))
    elseif recon === :ppm
        coeffs === nothing && error("muscl_hancock_flux_line!: recon=:ppm needs coeffs=(c1,…,c6)")
        c1, c2, c3, c4, c5, c6 = coeffs
        _ppm_hancock_kernel!(be)(fd, fs1, fs2, fs3, fe, gef, rho, eint, vx, vy, vz,
                                 c1, c2, c3, c4, c5, c6,
                                 ncells, nfi, nghost, 1, T(gamma), T(cpred), T(small_rho), idu, rc;
                                 ndrange = (nfi, Int(jdim)))
    else
        _muscl_hancock_kernel!(be)(fd, fs1, fs2, fs3, fe, gef, rho, eint, vx, vy, vz,
                                   ncells, nfi, nghost, 1, T(gamma), T(theta), T(cpred), T(small_rho), idu, rc;
                                   ndrange = (nfi, Int(jdim)))
    end
    KA.synchronize(be)
    return fd, fs1, fs2, fs3, fe
end

export muscl_hancock_flux_line!
