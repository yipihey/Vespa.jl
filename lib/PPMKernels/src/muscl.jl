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

@kernel function _muscl_hll_kernel!(fd, fs1, fs2, fs3, fe,
                                    @Const(rho), @Const(eint), @Const(vx),
                                    @Const(vy), @Const(vz),
                                    ncells::Int, nfi::Int, nghost::Int, j1::Int,
                                    gamma, theta, small_rho)
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

        F = _hll5(ρl, el, ul, vl, wl, ρr, er, ur, vr, wr, g, gm1)
        fd[fo] = F[1]; fs1[fo] = F[2]; fs2[fo] = F[3]; fs3[fo] = F[4]; fe[fo] = F[5]
    end
end

"""
    muscl_flux_line!(fd, fs1, fs2, fs3, fe, rho, eint, vx, vy, vz;
                     ncells, nghost, jdim=1, gamma, theta=1.5, small_rho=1e-10)

Fill the five interface-flux arrays (density, 3 momenta, total energy) over
`active+1` interfaces per pencil via fused PLM+HLL. `rho/eint/vx/vy/vz` are the
primitive lines (length `ncells·jdim`, `nghost` ghosts each side); flux arrays
have length `(active+1)·jdim` where `active = ncells − 2·nghost`. Element type of
`fd` sets the working precision.
"""
function muscl_flux_line!(fd, fs1, fs2, fs3, fe, rho, eint, vx, vy, vz;
                          ncells::Integer, nghost::Integer, jdim::Integer = 1,
                          gamma::Real, theta::Real = 1.5, small_rho::Real = 1e-10)
    be = KA.get_backend(fd)
    T = eltype(fd)
    ncells, nghost = Int(ncells), Int(nghost)
    active = ncells - 2 * nghost
    nfi = active + 1
    _muscl_hll_kernel!(be)(fd, fs1, fs2, fs3, fe, rho, eint, vx, vy, vz,
                           ncells, nfi, nghost, 1, T(gamma), T(theta), T(small_rho);
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
    # conserved + normal-direction physical flux of each face
    pa = gm1*ρa*ea; v2a = ua*ua + va*va + wa*wa; Ea = ρa*(ea + h*v2a)
    pb = gm1*ρb*eb; v2b = ub*ub + vb*vb + wb*wb; Eb = ρb*(eb + h*v2b)
    Ua1 = ρa; Ua2 = ρa*ua; Ua3 = ρa*va; Ua4 = ρa*wa; Ua5 = Ea
    Ub1 = ρb; Ub2 = ρb*ub; Ub3 = ρb*vb; Ub4 = ρb*wb; Ub5 = Eb
    Fa1 = ρa*ua; Fa2 = Ua2*ua + pa; Fa3 = Ua3*ua; Fa4 = Ua4*ua; Fa5 = (Ea + pa)*ua
    Fb1 = ρb*ub; Fb2 = Ub2*ub + pb; Fb3 = Ub3*ub; Fb4 = Ub4*ub; Fb5 = (Eb + pb)*ub
    # Hancock half-step: both faces += cpred·(F⁻ − F⁺)
    d1 = cpred*(Fa1-Fb1); d2 = cpred*(Fa2-Fb2); d3 = cpred*(Fa3-Fb3); d4 = cpred*(Fa4-Fb4); d5 = cpred*(Fa5-Fb5)
    # evolved conserved → primitive (a = left/minus face, b = right/plus face)
    ρas = max(Ua1+d1, small_rho); uas = (Ua2+d2)/ρas; vas = (Ua3+d3)/ρas; was = (Ua4+d4)/ρas
    eas = (Ua5+d5)/ρas - h*(uas*uas + vas*vas + was*was)
    ρbs = max(Ub1+d1, small_rho); ubs = (Ub2+d2)/ρbs; vbs = (Ub3+d3)/ρbs; wbs = (Ub4+d4)/ρbs
    ebs = (Ub5+d5)/ρbs - h*(ubs*ubs + vbs*vbs + wbs*wbs)
    return (ρas, eas, uas, vas, was, ρbs, ebs, ubs, vbs, wbs)
end

@kernel function _muscl_hancock_kernel!(fd, fs1, fs2, fs3, fe,
                                        @Const(rho), @Const(eint), @Const(vx),
                                        @Const(vy), @Const(vz),
                                        ncells::Int, nfi::Int, nghost::Int, j1::Int,
                                        gamma, theta, cpred, small_rho)
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
        F = _hll5(Lf[6], Lf[7], Lf[8], Lf[9], Lf[10],   # plus face of cl
                  Rf[1], Rf[2], Rf[3], Rf[4], Rf[5],    # minus face of cl+1
                  g, gm1)
        fd[fo] = F[1]; fs1[fo] = F[2]; fs2[fo] = F[3]; fs3[fo] = F[4]; fe[fo] = F[5]
    end
end

"""
    muscl_hancock_flux_line!(fd, fs1, fs2, fs3, fe, rho, eint, vx, vy, vz;
                             ncells, nghost, jdim=1, gamma, theta=1.5, cpred,
                             small_rho=1e-10)

Like [`muscl_flux_line!`](@ref) but with the MUSCL-Hancock ½-step predictor folded
in. `cpred = dt/(2·dx)` is the predictor coefficient; `cpred=0` recovers the bare
reconstruction+HLL flux line (up to the conserved↔primitive round-trip).
"""
function muscl_hancock_flux_line!(fd, fs1, fs2, fs3, fe, rho, eint, vx, vy, vz;
                                  ncells::Integer, nghost::Integer, jdim::Integer = 1,
                                  gamma::Real, theta::Real = 1.5, cpred::Real,
                                  small_rho::Real = 1e-10)
    be = KA.get_backend(fd); T = eltype(fd)
    ncells, nghost = Int(ncells), Int(nghost)
    active = ncells - 2 * nghost; nfi = active + 1
    _muscl_hancock_kernel!(be)(fd, fs1, fs2, fs3, fe, rho, eint, vx, vy, vz,
                               ncells, nfi, nghost, 1, T(gamma), T(theta), T(cpred), T(small_rho);
                               ndrange = (nfi, Int(jdim)))
    KA.synchronize(be)
    return fd, fs1, fs2, fs3, fe
end

export muscl_hancock_flux_line!
