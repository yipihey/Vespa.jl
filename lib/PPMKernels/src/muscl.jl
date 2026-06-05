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

# PLM point value v + ½·minmod((v−vm1)θ, (vp1−v)θ, ½(vp1−vm1))  (plm_point).
@inline _plm_pt(vm1::T, v::T, vp1::T, θ::T) where {T} =
    v + T(0.5) * _minmod3((v - vm1) * θ, (vp1 - v) * θ, T(0.5) * (vp1 - vm1))

# HLL flux of one conserved field from L/R fluxes/states (Riemann_HLL.C).
@inline _hll(ap::T, am::T, fl::T, fr::T, ul::T, ur::T) where {T} =
    (ap * fl + am * fr - ap * am * (ur - ul)) / (ap + am)

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

        # ── L flux/state
        v2l = ul*ul + vl*vl + wl*wl
        pl = gm1 * ρl * el
        csl = sqrt(g * pl / ρl)
        etl = el + T(0.5) * v2l
        UlD = ρl; UlS1 = ρl*ul; UlS2 = ρl*vl; UlS3 = ρl*wl; UlE = ρl*etl
        FlD = ρl*ul; FlS1 = UlS1*ul + pl; FlS2 = UlS2*ul; FlS3 = UlS3*ul
        FlE = (UlE + pl) * ul                          # rho*(0.5 v2 + h)*vx
        lpl = ul + csl; lml = ul - csl

        # ── R flux/state
        v2r = ur*ur + vr*vr + wr*wr
        pr = gm1 * ρr * er
        csr = sqrt(g * pr / ρr)
        etr = er + T(0.5) * v2r
        UrD = ρr; UrS1 = ρr*ur; UrS2 = ρr*vr; UrS3 = ρr*wr; UrE = ρr*etr
        FrD = ρr*ur; FrS1 = UrS1*ur + pr; FrS2 = UrS2*ur; FrS3 = UrS3*ur
        FrE = (UrE + pr) * ur
        lpr = ur + csr; lmr = ur - csr

        ap = max(zero(T), max(lpl, lpr))
        am = max(zero(T), max(-lml, -lmr))

        fd[fo]  = _hll(ap, am, FlD,  FrD,  UlD,  UrD)
        fs1[fo] = _hll(ap, am, FlS1, FrS1, UlS1, UrS1)
        fs2[fo] = _hll(ap, am, FlS2, FrS2, UlS2, UrS2)
        fs3[fo] = _hll(ap, am, FlS3, FrS3, UlS3, UrS3)
        fe[fo]  = _hll(ap, am, FlE,  FrE,  UlE,  UrE)
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
